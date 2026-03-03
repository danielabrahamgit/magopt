import torch
import numpy as np

# @functools.lru_cache(maxsize=16)
def _gauss_legendre(n: int) -> tuple[torch.Tensor,torch.Tensor]:
    """
    Return nodes/weights on [0,1] for n-point Gauss-Legendre rule.
    
    Args:
    -----
    n : int
        Number of sample points

    Returns:
    --------
    tuple[torch.Tensor,torch.Tensor]:
        Eval point locations and weights, respectively
    """
    xi, wi = np.polynomial.legendre.leggauss(n)
    t = torch.tensor((xi + 1) / 2, dtype=torch.float64)
    w = torch.tensor(wi / 2,       dtype=torch.float64)
    return t, w