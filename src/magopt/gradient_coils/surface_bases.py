import torch
from typing import Optional
from math import ceil
from torch.special import chebyshev_polynomial_t, chebyshev_polynomial_u

def _fourier_bases(x: torch.Tensor, 
                   n_modes: int,
                   ns: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the cosine and sine bases for the given modes.
    
    Args
    ----
    x : torch.Tensor
        Input tensor with shape (...) scaled between [0, 1]
    n_modes : int, optional
        Number of modes to use for the cosine and sine bases. If None, uses all modes up to n_modes//2.
    ns : torch.Tensor, optional
        Modes to use for the cosine and sine bases. If None, uses all modes up to n_modes//2.
        
    Returns
    -------
    torch.Tensor
        The cosine and sine bases with shape (..., n_modes)
    """
    assert n_modes % 2 == 1, "n_modes must be odd"
    if ns is None:
        ns = torch.arange(n_modes//2, device=x.device)
    bases = torch.exp(2j * torch.pi * ns * x[..., None])
    bases = torch.cat([bases.real, bases.imag[..., 1:]], dim=-1)[..., :n_modes]
    return bases

def _deriv_fourier_bases(x: torch.Tensor, 
                         n_modes: int,
                         ns: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the derivative of the Fourier bases for the given modes.
    
    Args
    ----
    x : torch.Tensor
        Input tensor with shape (...) scaled between [0, 1]
    n_modes : int
        Number of modes to use for the derivative of the Fourier bases.
    ns : torch.Tensor, optional
        Modes to use for the derivative of the Fourier bases. If None, uses all modes up to n_modes//2.
        
    Returns
    -------
    torch.Tensor
        The derivative of the Fourier bases with shape (..., n_modes)
    """
    if ns is None:
        ns = torch.arange(ceil(n_modes/2), device=x.device)
    bases = torch.exp(2j * torch.pi * ns * x[..., None]) * 2j * torch.pi * ns
    bases = torch.cat([bases.real, bases.imag], dim=-1)[..., :n_modes]
    return bases

def _chebyshev_bases(x: torch.Tensor, 
                     n_modes: int,
                     ns: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the Chebyshev bases for the given modes.
    
    Args
    ----
    x : torch.Tensor
        Input tensor with shape (...) scaled between [0, 1]
    n_modes : int
        Number of modes to use for the Chebyshev bases.
    ns : torch.Tensor, optional
        Modes to use for the Chebyshev bases. If None, uses all modes up to n_modes.
        
    Returns
    -------
    torch.Tensor
        The Chebyshev bases with shape (..., n_modes)
    """
    x_scaled = 2 * x - 1
    if ns is None:
        ns = torch.arange(n_modes, device=x.device)
    bases = chebyshev_polynomial_t(x_scaled[..., None], ns)
    return bases

def _deriv_chebyshev_bases(x: torch.Tensor, 
                           n_modes: int,
                           ns: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the derivative of the Chebyshev bases for the given modes.
    
    Args
    ----
    x : torch.Tensor
        Input tensor with shape (...) scaled between [0, 1]
    n_modes : int
        Number of modes to use for the derivative of the Chebyshev bases.
    ns : torch.Tensor, optional
        Modes to use for the derivative of the Chebyshev bases. If None, uses all modes up to n_modes.
        
    Returns
    -------
    torch.Tensor
        The derivative of the Chebyshev bases with shape (..., n_modes)
    """
    x_scaled = 2 * x - 1
    if ns is None:
        ns = torch.arange(n_modes, device=x.device)
    bases = 2 * ns * chebyshev_polynomial_u(x_scaled[..., None], ns - 1)
    return bases

def _triangular_bases(x: torch.Tensor, 
                      n_modes: int,
                      ns: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the triangular bases for the given modes.
    
    Args
    ----
    x : torch.Tensor
        Input tensor with shape (...) scaled between [0, 1]
    n_modes : int
        Number of modes to use for the triangular bases.
    ns : torch.Tensor, optional
        Modes to use for the triangular bases. If None, uses all modes up to n_modes.
        
    Returns
    -------
    torch.Tensor
        The triangular bases with shape (..., n_modes)
    """
    N = n_modes - 1
    if ns is None:
        ns = torch.arange(n_modes, device=x.device)
    mask = (x[..., None] - ns/N).abs() <= 1/ N
    bases = 1 - (N * x[..., None] - ns).abs()
    return bases * mask.float()
    
def _deriv_triangular_bases(x: torch.Tensor, 
                            n_modes: int,
                            ns: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the derivative of the triangular bases for the given modes.
    
    Args
    ----
    x : torch.Tensor
        Input tensor with shape (...) scaled between [0, 1]
    n_modes : int
        Number of modes to use for the derivative of the triangular bases.
    ns : torch.Tensor, optional
        Modes to use for the derivative of the triangular bases. If None, uses all modes up to n_modes.
        
    Returns
    -------
    torch.Tensor
        The derivative of the triangular bases with shape (..., n_modes)
    """
    N = n_modes - 1
    if ns is None:
        ns = torch.arange(n_modes, device=x.device)
    mask = (x[..., None] - ns/N).abs() <= 1/ N
    bases = -N * torch.sign(N * x[..., None] - ns)
    return bases * mask.float()

def _gauss_bases(x: torch.Tensor, 
                 n_modes: int,
                 ns: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the Gaussian bases for the given modes.
    
    Args
    ----
    x : torch.Tensor
        Input tensor with shape (...) scaled between [0, 1]
    n_modes : int
        Number of modes to use for the Gaussian bases.
    ns : torch.Tensor, optional
        Modes to use for the Gaussian bases. If None, uses all modes up to n_modes.
        
    Returns
    -------
    torch.Tensor
        The Gaussian bases with shape (..., n_modes)
    """
    if ns is None:
        ns = torch.arange(n_modes, device=x.device)
    
    xs = x[..., None]
    nfrac = ns / (n_modes - 1)
    bases = torch.exp(-(n_modes * (xs - nfrac)) ** 2)
    return bases

def _deriv_gauss_bases(x: torch.Tensor, 
                       n_modes: int,
                       ns: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the derivative of the Gaussian bases for the given modes.
    
    Args
    ----
    x : torch.Tensor
        Input tensor with shape (...) scaled between [0, 1]
    n_modes : int
        Number of modes to use for the derivative of the Gaussian bases.
    ns : torch.Tensor, optional
        Modes to use for the derivative of the Gaussian bases. If None, uses all modes up to n_modes.
        
    Returns
    -------
    torch.Tensor
        The derivative of the Gaussian bases with shape (..., n_modes)
    """
    xs = x[..., None]
    nfrac = ns / (n_modes - 1)
    bases = -2 * (n_modes ** 2) * (xs - nfrac) * torch.exp(-(n_modes * (xs - nfrac)) ** 2)
    return bases