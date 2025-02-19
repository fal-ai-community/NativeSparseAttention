# nsattn
Various implementations of Native Sparse Attention (2502.11089).

```bash
git clone https://github.com/fal-ai-community/NativeSparseAttention && cd NativeSparseAttention
uv sync
```
or
```bash
uv pip install git+https://github.com/fal-ai-community/NativeSparseAttention
```

## [`eager.py`](src/nsattn/eager.py)
An inefficient, naive, eager math implementation of Native Sparse Attention (NSA).

```bash
# Run (useless) test
uv run python -m nsattn.eager
```

It is written to encourage discussion on (what I perceive to be) ambiguities in the paper.

Please Ctrl-F for all comments with "TODO" or "NOTE" in them.

## future
- [ ] discuss ambiguities and unknowns from the paper
- [ ] support varying l'/l/d, and grouped heads
- [ ] separate prefill/decode impls
- [ ] write kernel(s)
- [X] pray to frontier labs for model that can one-shot impl