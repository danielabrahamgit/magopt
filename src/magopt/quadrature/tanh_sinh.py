import torch
import numpy as np

# @functools.lru_cache(maxsize=32)
def _tanh_sinh(n: int):
    """
    Tanh-sinh (double exponential) quadrature nodes/weights on [0, 1].

    The substitution t = 1/2(1 + tanh(pi/2 · sinh(u))) transforms [0,1]
    such that weights cluster exponentially near BOTH endpoints.
    This is specifically superior to Gauss-Legendre for integrands
    with endpoint singularities — exactly the case for adjacent wire
    segments where R-> 0 near the shared vertex.

    Parameters
    ----------
    n : int
        Number of points. Typical values:
            n=20  — well-separated segments
            n=40  — adjacent segments
            n=80  — nearly-touching segments
    """
    h = 4.0 / n
    k = torch.arange(-n // 2, n // 2 + 1, dtype=torch.float64)

    u     = h * k
    sinh_u = torch.sinh(u)
    cosh_u = torch.cosh(u)

    # nodes on [0, 1]
    t  = 0.5 * (1.0 + torch.tanh((torch.pi / 2) * sinh_u))

    # weights  dt/du * h
    cosh_pi_sinh = torch.cosh((torch.pi / 2) * sinh_u)
    dt = 0.5 * h * (torch.pi / 2) * cosh_u / (cosh_pi_sinh ** 2)

    # drop points that have collapsed to exactly 0 or 1
    # (happens at the tails for large n — weight is numerically zero anyway)
    mask = (t > 1e-15) & (t < 1.0 - 1e-15) & (dt > 1e-30)
    return t[mask].contiguous(), dt[mask].contiguous()