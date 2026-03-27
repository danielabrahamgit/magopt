import torch
import numpy as np
from scipy.special import ellipk, ellipe

def interp1d(x_query, x_table, y_table):
    """
    Differentiable 1D linear interpolation in PyTorch.
    All tensors must be on the same device.
    Assumes x_table is sorted.

    Args:
    -----
    x_query : torch.Tensor
        Query locations.
    x_table : torch.Tensor
        Monotonic sample locations.
    y_table : torch.Tensor
        Sample values at x_table.

    Returns:
    --------
    torch.Tensor
        Interpolated values at x_query.
    """
    # Clamp x_query within bounds
    x_query = torch.clamp(x_query, x_table[0], x_table[-1])

    # Search sorted indices
    idxs = torch.searchsorted(x_table, x_query, right=True)
    idxs = torch.clamp(idxs, 1, len(x_table) - 1)

    x0 = x_table[idxs - 1]
    x1 = x_table[idxs]
    y0 = y_table[idxs - 1]
    y1 = y_table[idxs]

    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x_query - x0)

class EllipKLookup(torch.nn.Module):
    def __init__(self, resolution=1000, eps=1e-6):
        """
        Build lookup table for complete elliptic integral K(m).

        Args:
        -----
        resolution : int
            Number of lookup samples.
        eps : float
            Clamp margin from 0 and 1.

        Returns:
        --------
        None
            Initializes module buffers.
        """
        super().__init__()
        m_vals = np.linspace(eps, 1 - eps, resolution)
        K_vals = ellipk(m_vals)

        self.register_buffer('m_vals', torch.tensor(m_vals, dtype=torch.float32))
        self.register_buffer('K_vals', torch.tensor(K_vals, dtype=torch.float32))

    def forward(self, m_query):
        """
        Evaluate interpolated K(m) values.

        Args:
        -----
        m_query : torch.Tensor
            Query modulus values in [eps, 1 - eps].

        Returns:
        --------
        torch.Tensor
            Approximated K(m_query).
        """
        return interp1d(m_query, self.m_vals, self.K_vals)
    
class EllipELookup(torch.nn.Module):
    def __init__(self, resolution=1000, eps=1e-6):
        """
        Build lookup table for complete elliptic integral E(m).

        Args:
        -----
        resolution : int
            Number of lookup samples.
        eps : float
            Clamp margin from 0 and 1.

        Returns:
        --------
        None
            Initializes module buffers.
        """
        super().__init__()
        m_vals = np.linspace(eps, 1 - eps, resolution)
        E_vals = ellipe(m_vals)

        self.register_buffer('m_vals', torch.tensor(m_vals, dtype=torch.float32))
        self.register_buffer('E_vals', torch.tensor(E_vals, dtype=torch.float32))

    def forward(self, m_query):
        """
        Evaluate interpolated E(m) values.

        Args:
        -----
        m_query : torch.Tensor
            Query modulus values in [eps, 1 - eps].

        Returns:
        --------
        torch.Tensor
            Approximated E(m_query).
        """
        return interp1d(m_query, self.m_vals, self.E_vals)
