import torch
from typing import Optional

def gen_grd(im_size: tuple, 
            fovs: Optional[tuple] = None,
            balanced: Optional[bool] = False) -> torch.Tensor:
    """
    Generates a grid of points given image size and FOVs

    Parameters:
    -----------
    im_size : tuple
        image dimensions
    fovs : tuple
        field of views, same size as im_size
    
    Returns:
    --------
    grd : torch.Tensor
        grid of points with shape (*im_size, len(im_size))
    """
    if fovs is None:
        fovs = (1,) * len(im_size)
    if balanced:
        lins = [
            fovs[i] * torch.linspace(-1/2, 1/2, im_size[i]) 
            for i in range(len(im_size))
            ]
    else:
        lins = [
            fovs[i] * torch.arange(-(im_size[i]//2), im_size[i]//2 + (im_size[i] % 2)) / (im_size[i]) 
            for i in range(len(im_size))
            ]
    grds = torch.meshgrid(*lins, indexing='ij')
    grd = torch.cat(
        [g[..., None] for g in grds], dim=-1)
        
    return grd.type(torch.float32)