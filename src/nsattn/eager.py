# This is an inefficient, naive, eager math implementation of Native Sparse Attention (NSA).
# It is written to encourage discussion on (what I perceive to be) ambiguities in the paper.
# Please Ctrl-F for all comments with "TODO" or "NOTE" in them.

import os
import math
from functools import cache, partial

import torch
import torch.nn.functional as F
from torch import Tensor as TT, nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask, flex_attention

from jaxtyping import Float
QShape = Float[TT, "b h s d"]
Gating = Float[TT, "b h s 3"]

"""NOTE: assumptions in this implementation:

0. You don't care about performance. There is no efficiency in this code.
   It will take more memory/flops than SDPA because we have no kernels.

1. For maximal simplicity, I only support the following conditions:
   * prefill sequence lengths, meaning q.size(-2) == k.size(-2) == v.size(-2)
   * full multi-head, meaning no grouped/multi-query, so q.size(-3) == k.size(-3)

2. We match the paper's settings on w/l'/l/d:
   > For NSA, we set compression block size 𝑙 = 32, sliding stride 𝑑 = 16,
   > selected block size 𝑙′ = 64, selected block count 𝑛 = 16
   > (including fixed activating the 1 initial block and 2 local blocks),
   > and sliding window size 𝑤 = 512.

"""
DEBUG = os.environ.get("DEBUG_NSA", "0") == "1"
if DEBUG:
    SLIDING_WINDOW = 8
    SELECTED_COUNT = 2
    SELECTED_BSIZE = 8
    COMPRESSION_BS = 4
    SLIDING_STRIDE = 2
    FLEX_BLOCKSIZE = 8
else:
    SLIDING_WINDOW = 512
    COMPRESSION_BS = 32
    SLIDING_STRIDE = 16
    SELECTED_BSIZE = 64
    SELECTED_COUNT = 16
    FLEX_BLOCKSIZE = 128


### SWA IMPL (not compiled) ###
def sliding_window_causal(score, b, h, q_idx, kv_idx): return torch.where((q_idx >= kv_idx) & (q_idx - kv_idx <= SLIDING_WINDOW), score, -float("inf"))
def sliding_window_mask(b, h, q_idx, kv_idx): return (q_idx >= kv_idx) & (q_idx - kv_idx <= SLIDING_WINDOW)
create_block_mask_cached = cache(partial(create_block_mask, BLOCK_SIZE=FLEX_BLOCKSIZE))
swa_flex = partial(flex_attention, score_mod=sliding_window_causal)
def swa_attention(q: TT, k: TT, v: TT):
    block_mask = create_block_mask_cached(sliding_window_mask, 1, 1, q.size(-2), q.size(-2), device=q.device)
    if DEBUG: print('SWA:', block_mask)
    return swa_flex(q,k,v,block_mask=block_mask)


### COMPERSSION IMPL ###
def Linear(a: int, b: int): return nn.Linear(a,b,bias=False)
class Phi(nn.Module):
    # TODO: The paper doesn't describe what 𝜑 does. At all. So...
    # In lieu of any instructions, I do repeated linear downpooling of 2->1 tokens && silu
    def __init__(self, dim: int, block_l: int):
        super().__init__()
        downpools = int(math.log2(block_l))
        assert 1<<downpools == block_l
        self.down = nn.ModuleList([Linear(dim*2, dim) for _ in range(downpools)])
        self.stop = Linear(dim,dim)
    def forward(self, x: TT) -> TT:
        for l in self.down:
            x = x.unflatten(-2, (x.size(-2)//2, 2)).flatten(-2)
            x = F.silu(l(x))
        return self.stop(x)
class Compressor(nn.Module):
    def __init__(self, dim: int, block_l: int = COMPRESSION_BS, stride_d: int = SLIDING_STRIDE):
        super().__init__()
        self.stride_d = stride_d
        # TODO: paper doesn't describe positional embedding used. so I just LPE.
        self.pe = nn.Parameter(torch.randn(block_l, dim))
        self.mlp = Phi(dim, block_l)
    def forward(self, k: TT):
        # TODO: it is possible this unfold is off-by-one :(
        k = k.unfold(-2, self.pe.size(-2), self.stride_d).mT

        pe = self.pe
        for _ in range(k.ndim - self.pe.ndim): pe = pe[None]
        k = k+pe

        # Pad k_cmp with empty 0th token, both for compute reasons (divisibility) and to avoid causal leak
        k = torch.cat([torch.zeros_like(k[:,:,:1]), k], dim=2)
        return self.mlp(k).squeeze(-2)


### NSA IMPL ###
def nsa_attention(
    q: QShape, k: QShape, v: QShape,
    g: Gating, k_cmp: TT, v_cmp: TT,
    *,
    selected_block_size: int=SELECTED_BSIZE,    # size of compression block
    top_B: int=SELECTED_COUNT,                  # number of top-n blocks
) -> TT:
    """
    naive eager Native Sparse Attention (NSA) implementation.

    It splits attention into three branches:
      - Compression: attend over block-level (averaged) representations from blocks that lie entirely in the past.
      - Selection: from eligible blocks (those having at least one past token), select top_B blocks
                   (via vectorized topk) and then attend over the fine-grained tokens in each block that are in the past.
      - Sliding Window: attend to a fixed window (of size `window+1`) of past tokens (including the current token).

    The three branch outputs are then combined using the gating score tensor `g`, assumed to be of shape [B, H, S, 3].

    Returns:
      output: Tensor of shape [B, H, S, D]
    """
    S,D = q.size(-2),q.size(-1)
    scale = 1 / math.sqrt(D) # <-- default Scale for DPA, changeme to 1/D if you want

    ### Naive compressed token calc
    # NOTE: the paper does a plain (q.mT@k).softmax(-1). That works for them, because their math is described for the decode case.
    # But for prefill w/ full sequence q, you need to consider causal masking.
    attn_mask = torch.ones(k_cmp.size(-2), k_cmp.size(-2), dtype=torch.bool, device=q.device)
    assert k_cmp[:,:,0].nonzero().numel() == 0, "I assume k_cmp is padded with a 0th token for the attention mask."
    # We don't want compressed chunks to causal leak (see Fig 2). In effect, we
    # need to block each q position from seeing COMPRESSION_BS future tokens.
    # Our implementation of Compressor pads k_cmp with a 0th token, which is a padding of SLIDING_STRIDE.
    # so we need to further rotate the tril by COMPRESSION_BS//SLIDING_STRIDE-1.
    attn_mask = attn_mask.tril(1-COMPRESSION_BS//SLIDING_STRIDE)
    attn_mask[:COMPRESSION_BS,0] = True # don't block the 0th padding token to avoid inf
    # then, we repeat interleave it to match q shape.
    attn_mask = attn_mask.repeat_interleave(COMPRESSION_BS*SLIDING_STRIDE//COMPRESSION_BS, dim=-2)
    if DEBUG: __import__("nsattn").util.mask_to_img(attn_mask, 'attn_cmp.png', 'CMP attn mask (w/ 0-padded key)')
    attn_bias = torch.zeros_like(attn_mask,dtype=q.dtype).masked_fill_(~attn_mask, float('-inf'))
    assert attn_bias[:COMPRESSION_BS,1:].isinf().all()
    assert attn_bias[COMPRESSION_BS,1] == 0
    # (I also add the sdpa scale, because I think it should be used)
    p_cmp = ((q @ k_cmp.mT) * scale + attn_bias).softmax(-1)
    o_cmp = p_cmp @ v_cmp
    # Because we need p_cmp to compute p_slc, we do naive math attention above. (TODO fuse cmp & slc)

    # ""Importance Score Computation""
    # TODO: this is a hardcoded solution for the globally defined values of l'/l/d.
    # It will become incorrect if you change those. Also, it might be off-by-one.
    assert selected_block_size // COMPRESSION_BS == 2
    assert COMPRESSION_BS // SLIDING_STRIDE == 2
    w = torch.tensor([1.0, 2.0, 2.0, 2.0, 1.0], device='cuda').expand(1,1,5)
    p_slc = F.conv1d(p_cmp.flatten(end_dim=-2).unsqueeze(-2), w, stride=4, padding=1)
    p_slc = p_slc.view(*p_cmp.shape[:-1], p_slc.size(-1))
    assert p_slc.size(-1) == k.size(-2) // selected_block_size
    assert q.size(1) == k.size(1), "GQA not supported yet!" # we'd need to sum over group here

    # TODO: The paper says,
    # > (including fixed activating the 1 initial block and 2 local blocks),
    # But I do not know what this means. I believe the "1 initial block" may refer to an attention-sink-like deal,
    # but I do not know what the "2 local blocks" are.
    # TODO: there is no recourse here if sequence length exceeds the number of blocks to be selected!
    _, indices = p_slc.topk(top_B,dim=-1)
    # NOTE: we cannot compute a minimal \tilde{K}^{slc} because it is independent per q_t,
    # so we simply materalize a full bool mask from p_slc and use it in SDPA.
    slc_blocks = k.size(-2)//selected_block_size
    mask = torch.zeros(*indices.shape[:3], slc_blocks, dtype=torch.bool)
    # NOTE: I double-mask off with a tril(-1) to account for low seq indices, where n*l' >> t.
    tril = torch.ones(*indices.shape[:2], slc_blocks, slc_blocks, dtype=torch.bool).tril(-1)
    mask = mask.scatter_(-1, indices, True) & tril.repeat_interleave(selected_block_size,-2)
    mask = mask.repeat_interleave(selected_block_size,-1)
    if DEBUG: __import__("nsattn").util.mask_to_img(mask.float().mean(0).mean(0), 'attn_slc.png', 'Averaged SLC attn mask')
    o_slc = F.scaled_dot_product_attention(q,k,v, attn_mask=mask, is_causal=False, scale=scale, enable_gqa=False)

    # just defer SWA to a flexattn impl. Obviously should be fused with other 3 branches.
    o_swa = swa_attention(q,k,v)

    ### Gating ###
    # This isn't an optimized approach but it is relatively simple.
    res = torch.stack([o_cmp,o_slc,o_swa],dim=-2)
    res = res * g.unsqueeze(-1)
    return res.sum(-2)

class Gate(nn.Sequential):
    def __init__(self, d: int, h: int, variants: int=3):
        # NOTE: we use *independent* gating weights per-head. I believe this should work better than global gating.
        super().__init__(
            # just an MLP (TODO: what does deepseek actually use?)
            Linear(d, d//2),
            nn.SiLU(),
            Linear(d//2, variants*h),
            # final sigmoid to constrain gating values
            nn.Sigmoid(),
        )

### Final module ###
class NSA(nn.Module):
    def __init__(self, dim: int, heads: int, *, d_h: int=64):
        super().__init__()
        self.h = heads
        self.wq = Linear(dim, heads*d_h)
        # TODO: open qn on whether k/v should be unique per branch or not.
        # for now we just use the same for all.
        self.wk = Linear(dim, heads*d_h)
        self.wv = Linear(dim, heads*d_h)
        self.wo = Linear(heads*d_h, dim)
        self.gate = Gate(dim, heads)
        # 3.3.1 Token Compression module...
        self.f_cmp_k = Compressor(d_h)
        self.f_cmp_v = Compressor(d_h)
    def forward(self, x: TT):
        g = self.gate(x)
        q,k,v = self.wq(x), self.wk(x), self.wv(x)
        q,k,v,g = (t.unflatten(-1,(self.h,-1)).transpose(-2,-3) for t in [q,k,v,g])
        # TODO: add rope or some other PE here, ofc
        k_cmp,v_cmp = self.f_cmp_k(k), self.f_cmp_v(v)
        o = nsa_attention(q,k,v,g,k_cmp,v_cmp)
        return self.wo(o.transpose(-2,-3).flatten(-2))


### test ###
def test():
    __import__("lovely_tensors").monkey_patch()
    with torch.device('cuda'):
        S = 128 if DEBUG else 2048
        D = 1536
        nsa = NSA(D, D//64, d_h = 64)
        y = nsa(x := torch.randn(2, S, D))
        print(x)
        print(y)

if __name__ == '__main__': test()