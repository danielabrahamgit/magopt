import torch
from typing import Tuple

from ...quadrature import _gauss_legendre


def ellipse_quadrature(a: torch.Tensor,
                       b: torch.Tensor,
                       R: torch.Tensor,
                       center: torch.Tensor,
                       n_pts: int,
                       dtype: torch.dtype,
                       device: torch.device
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build quadrature points and weighted tangents for an ellipse.

    Args:
    -----
    a : torch.Tensor
        Semi-major axis.
    b : torch.Tensor
        Semi-minor axis.
    R : torch.Tensor
        Rotation matrix with shape [3, 3].
    center : torch.Tensor
        Ellipse center with shape [3].
    n_pts : int
        Number of quadrature points.
    dtype : torch.dtype
        Tensor dtype for outputs.
    device : torch.device
        Tensor device for outputs.

    Returns:
    --------
    tuple[torch.Tensor, torch.Tensor]
        World positions and weighted tangents.
    """
    t, w = _gauss_legendre(n_pts)
    t = t.to(dtype=dtype, device=device)
    w = w.to(dtype=dtype, device=device)

    two_pi = torch.tensor(2.0 * torch.pi, dtype=dtype, device=device)
    phi = two_pi * t
    wphi = two_pi * w

    c = torch.cos(phi)
    s = torch.sin(phi)
    zero = torch.zeros_like(c)

    r_local = torch.stack([a * c, b * s, zero], dim=-1)
    dl_dphi_local = torch.stack([-a * s, b * c, zero], dim=-1)

    r_t = r_local @ R.T + center[None, :]
    dl = (dl_dphi_local @ R.T) * wphi[:, None]
    return r_t, dl
