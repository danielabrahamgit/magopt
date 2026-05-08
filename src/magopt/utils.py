import torch
import numpy as np
from typing import Optional

def gen_pts_sphere_surf(N):
    """
    Deterministic sampling on the surface of a sphere with uniform distribution of points.
    
    Args
    ----
    N : int
        number of points to generate
        
    Returns
    -------
    pts : torch.Tensor
        coordinates on the surface of the sphere with shape (N, 3)
    """
    pts = []
    N_count = 0
    a = 4 * torch.pi / N
    d = a ** 0.5
    M_theta = round(torch.pi / d)
    d_theta = torch.pi / M_theta
    d_phi = a / d_theta
    while N_count < N:
        for m in range(M_theta):
            theta = torch.pi * (m + 0.5) / M_theta
            M_phi = round(2 * torch.pi * np.sin(theta) / d_phi)
            for n in range(M_phi):
                phi = 2 * torch.pi * n / M_phi
                pts.append([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
                N_count += 1
    pts = torch.tensor(pts).type(torch.float32)
    return pts

def gen_pts_ellip_surf_random(x_diam: float, 
                             y_diam: float, 
                             z_diam: float, 
                             num_samples: int = 1000) -> torch.Tensor:
    """
    Generates random uniform samples on the surface of an ellipsoid.
    
    Args
    ----
    x_diam : float
        diameter of the ellipsoid in the x direction
    y_diam : float
        diameter of the ellipsoid in the y direction
    z_diam : float
        diameter of the ellipsoid in the z direction
    num_samples : int
        number of samples to generate
        
    Returns
    -------
    crds : torch.Tensor
        coordinates on the surface of the ellipsoid with shape (num_samples, 3)
    """
    crds = torch.zeros(num_samples, 3)
    diags = torch.tensor([x_diam, y_diam, z_diam], dtype=torch.float32) / 2
    wmax = 1 / diags.min()
    n = 0
    while n < num_samples:
        crd = torch.randn(3)
        crd /= crd.norm()
        w = (crd / diags).norm()
        if torch.rand(1) < w / wmax:
            crds[n] = crd * diags
            n += 1
    return crds

def gen_grd(im_size: tuple, 
            fovs: Optional[tuple] = None,
            balanced: Optional[bool] = False) -> torch.Tensor:
    """
    Generates a grid of points given image size and FOVs

    Args
    ----
    im_size : tuple
        image dimensions
    fovs : tuple
        field of views, same size as im_size
    
    Returns
    -------
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