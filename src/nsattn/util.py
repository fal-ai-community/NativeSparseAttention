import torch
import matplotlib.pyplot as plt

def mask_to_img(mask: torch.Tensor, path: str, title: str):
    """Convert a boolean attention mask to a matplotlib image and save it.
    
    Args:
        mask: Boolean tensor of shape [..., L, S] representing an attention mask
        path: Path where to save the image file
    """
    img = mask.cpu().numpy().astype(float)
    h, w = img.shape[-2:]
    aspect = w / h
    aspect **= 0.5
    plt.figure(figsize=(12, 12/aspect))
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.colorbar()
    plt.xlabel('Key Position')
    plt.ylabel('Query Position') 
    plt.savefig(path)
    plt.close()
