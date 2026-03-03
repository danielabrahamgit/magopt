import functools
import numpy as np
import torch


@functools.lru_cache(maxsize=16)
def _gauss_legendre(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return Gauss-Legendre nodes and weights on [0, 1].

    Args:
    -----
    n : int
        Number of quadrature points.

    Returns:
    --------
    tuple[torch.Tensor, torch.Tensor]
        Nodes and weights on [0, 1].
    """
    xi, wi = np.polynomial.legendre.leggauss(n)
    t = torch.tensor((xi + 1) / 2, dtype=torch.float64)
    w = torch.tensor(wi / 2, dtype=torch.float64)
    return t, w


@functools.lru_cache(maxsize=32)
def _tanh_sinh(n: int):
    """
    Return tanh-sinh nodes and weights on [0, 1].

    Args:
    -----
    n : int
        Nominal number of quadrature samples.

    Returns:
    --------
    tuple[torch.Tensor, torch.Tensor]
        Filtered nodes and weights on [0, 1].
    """
    h = 4.0 / n
    k = torch.arange(-n // 2, n // 2 + 1, dtype=torch.float64)

    u = h * k
    sinh_u = torch.sinh(u)
    cosh_u = torch.cosh(u)

    t = 0.5 * (1.0 + torch.tanh((torch.pi / 2) * sinh_u))

    cosh_pi_sinh = torch.cosh((torch.pi / 2) * sinh_u)
    dt = 0.5 * h * (torch.pi / 2) * cosh_u / (cosh_pi_sinh ** 2)

    mask = (t > 1e-15) & (t < 1.0 - 1e-15) & (dt > 1e-30)
    return t[mask].contiguous(), dt[mask].contiguous()
