import torch
import torch.profiler
import numpy as np
import matplotlib.pyplot as plt

from math import ceil
from tqdm import tqdm
from matplotlib import cm
from skimage import measure
from typing import Optional
from einops import rearrange, einsum
from scipy.interpolate import interp1d
from torch.special import chebyshev_polynomial_t, chebyshev_polynomial_u

from .gradient_coil import gradient_coil
from ..bspline import BSpline1D

def _fourier_bases(x: torch.Tensor, 
                   n_modes: int,
                   ns: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the cosine and sine bases for the given modes.
    
    Args
    ----
    x : torch.Tensor
        Input tensor with shape (...) scaled between [0, 2*pi]
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
    bases = torch.exp(1j * ns * x[..., None])
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
        Input tensor with shape (...) scaled between [0, 2*pi]
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
    bases = torch.exp(1j * ns * x[..., None]) * 1j * ns
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
        Input tensor with shape (...) scaled between [-1, +1]
    n_modes : int
        Number of modes to use for the Chebyshev bases.
    ns : torch.Tensor, optional
        Modes to use for the Chebyshev bases. If None, uses all modes up to n_modes.
        
    Returns
    -------
    torch.Tensor
        The Chebyshev bases with shape (..., n_modes)
    """
    if ns is None:
        ns = torch.arange(n_modes, device=x.device)
    bases = chebyshev_polynomial_t(x[..., None], ns)
    return bases

def _deriv_chebyshev_bases(x: torch.Tensor, 
                           n_modes: int,
                           ns: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the derivative of the Chebyshev bases for the given modes.
    
    Args
    ----
    x : torch.Tensor
        Input tensor with shape (...) scaled between [-1, +1]
    n_modes : int
        Number of modes to use for the derivative of the Chebyshev bases.
    ns : torch.Tensor, optional
        Modes to use for the derivative of the Chebyshev bases. If None, uses all modes up to n_modes.
        
    Returns
    -------
    torch.Tensor
        The derivative of the Chebyshev bases with shape (..., n_modes)
    """
    if ns is None:
        ns = torch.arange(n_modes, device=x.device)
    bases = ns * chebyshev_polynomial_u(x[..., None], ns - 1)
    return bases

class stream_func_coil(gradient_coil):
    
    def __init__(self,
                 zs_spline: torch.Tensor,
                 as_spline: torch.Tensor,
                 bs_spline: Optional[torch.Tensor] = None,
                 lamda_spline: float = 1e-2,
                 num_z_modes: int = 10,
                 num_theta_modes: int = 15,
                 num_theta: int = 200,
                 num_zs: int = 200,
                 theta_bases: str = "fourier",
                 z_bases: str = "chebyshev"):
        """
        Args
        ----
        zs_spline : torch.Tensor
            shape (N,) representing the z positions of the spline interpolation points.
        as_spline : torch.Tensor
            shape (K,) representing the the x-radii positions of the spline interpolation points.
        bs_spline : Optional[torch.Tensor]
            shape (K,) representing the the y-radii positions of the spline interpolation points. If None, assumes circular cross-section.
        lamda_spline : float, optional
            Regularization parameter for the spline fitting.
        num_z_modes : int, optional
            Number of basis modes to use for representing the coil current distribution in the z direction.
        num_theta_modes : int, optional
            Number of basis modes to use for representing the coil current distribution in the theta direction.
        num_theta : int, optional
            Number of theta points to use for surface integration.
        num_zs : int, optional
            Number of z points to use for surface integration.
        """
        
        # Consts
        self.torch_dev = zs_spline.device
        self.zmin = zs_spline.min().item()
        self.zmax = zs_spline.max().item()
        self.num_theta = num_theta
        self.num_zs = num_zs
        
        # Make sure zs_spline is uniformly spaced
        assert torch.allclose(zs_spline[1:] - zs_spline[:-1], zs_spline[1] - zs_spline[0]), "zs_spline must be uniformly spaced"

        # surface parameters
        self.as_spline = BSpline1D(len(as_spline), zs_spline.min().item(), zs_spline.max().item(),
                                   boundary="clamp", dtype=as_spline.dtype, device=as_spline.device)
        self.as_spline.fit_lstsq(zs_spline, as_spline, lamda_spline)
        if bs_spline is None:
            self.bs_spline = self.as_spline
        else:
            self.bs_spline = BSpline1D(len(bs_spline), zs_spline.min().item(), zs_spline.max().item(),
                                       boundary="clamp", dtype=bs_spline.dtype, device=bs_spline.device)
            self.bs_spline.fit_lstsq(zs_spline, bs_spline, lamda_spline)

        # Stream interpolation parameters
        self.M = num_theta_modes
        self.K = num_z_modes
        
        # Indices and bases stream basis functions for z
        if z_bases == 'chebyshev':
            ks = torch.arange(self.K, device=self.torch_dev)
            self.z_bases = lambda x : _chebyshev_bases(x, self.K, ks)
            self.z_bases_deriv = lambda x : _deriv_chebyshev_bases(x, self.K, ks)
        elif z_bases == 'fourier':
            # x is in [-1, +1]
            ks = torch.arange(ceil(self.K/2), device=self.torch_dev)
            self.z_bases = lambda x : _fourier_bases(x * torch.pi, self.K, ks)
            self.z_bases_deriv = lambda x : _deriv_fourier_bases(x * torch.pi, self.K, ks) * torch.pi
        else:
            raise ValueError(f"Invalid z_bases: {z_bases}")
        
        # Indices and bases stream basis functions for theta
        if theta_bases == 'fourier':
            ms = torch.arange(ceil(self.M/2), device=self.torch_dev)
            self.theta_bases = lambda x : _fourier_bases(x, self.M, ms)
            self.theta_bases_deriv = lambda x : _deriv_fourier_bases(x, self.M, ms)
        elif theta_bases == 'chebyshev':
            ms = torch.arange(self.M, device=self.torch_dev)
            self.theta_bases = lambda x : _chebyshev_bases(x, self.M, ms)
            self.theta_bases_deriv = lambda x : _deriv_chebyshev_bases(x, self.M, ms)
        else:
            raise ValueError(f"Invalid theta_bases: {theta_bases}")
        
        # Last matrix of Levi-Civita tensor
        self.lct = torch.tensor([[0, -1, 0], 
                                 [1, 0, 0], 
                                 [0, 0, 0]], device=self.torch_dev, dtype=torch.float32)

    def _interp_surface_radii(self,
                              zs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        interpolated radii and derivatives using B-splines
        
        Args
        ----
        zs : torch.Tensor
            z positions on the surface (...,)
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            a(z), a'(z), b(z), b'(z) (...,)
        """
        a = self.as_spline(zs)
        a_prime = self.as_spline.dy(zs)
        b = self.bs_spline(zs)
        b_prime = self.bs_spline.dy(zs)
        return a, a_prime, b, b_prime

    def _surface_positions(self,
                           thetas: torch.Tensor,
                           zs: torch.Tensor) -> torch.Tensor:
        """
        Computes the 3D positions on the surface for given (theta, z) coordinates

        Args
        ----
        thetas : torch.Tensor
            Angular positions on the surface (...,)
        zs : torch.Tensor
            z positions on the surface (...,)

        Returns
        -------
        torch.Tensor
            The 3D positions on the surface (..., 3)
        """
        a, _, b, _ = self._interp_surface_radii(zs)
        x = a * torch.cos(thetas)
        y = b * torch.sin(thetas)
        return torch.stack([x, y, zs], dim=-1)

    def _surface_tangent_vectors(self,
                                 thetas: torch.Tensor,
                                 zs: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the surface tangent vectors at the given (theta, z) positions

        Args
        ----
        thetas : torch.Tensor
            Angular positions on the surface (...,)
        zs : torch.Tensor
            z positions on the surface (...,)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The surface tangent vectors t_theta, t_z with shapes (..., 3)
        """
        # Get surface radii
        a, ap, b, bp = self._interp_surface_radii(zs)
        
        # Compute theta tangent
        t_theta = torch.zeros((*thetas.shape, 3), dtype=thetas.dtype, device=thetas.device)
        t_theta[..., 0] = -a * torch.sin(thetas)
        t_theta[..., 1] = b * torch.cos(thetas)
        t_theta[..., 2] = 0.0

        # Compute z tangent
        t_z = torch.zeros((*zs.shape, 3), dtype=zs.dtype, device=zs.device)
        t_z[..., 0] = ap * torch.cos(thetas)
        t_z[..., 1] = bp * torch.sin(thetas)
        t_z[..., 2] = 1.0

        return t_theta, t_z
    
    def _normal_vector_and_ginv(self,
                                tangent_theta: torch.Tensor,
                                tangent_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the normal vector and inverse metric tensor (Ginv) at the given tangent vectors
        
        Args
        ----
        tangent_theta : torch.Tensor
            Tangent vector in the theta direction (..., 3)
        tangent_z : torch.Tensor
            Tangent vector in the z direction (..., 3)
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The normal vector (..., 3) and inverse metric tensor Ginv (..., 2, 2)
        """
        mat = torch.stack([tangent_theta, tangent_z], dim=-1) # (..., 3, 2)
        G = mat.mT @ mat # (..., 2, 2)
        det = G[..., 0, 0] * G[..., 1, 1] - G[..., 0, 1] * G[..., 1, 0]
        G_inv = torch.zeros_like(G)
        G_inv[..., 0, 0] = G[..., 1, 1]
        G_inv[..., 1, 1] = G[..., 0, 0]
        G_inv[..., 0, 1] = -G[..., 0, 1]
        G_inv[..., 1, 0] = -G[..., 1, 0]
        G_inv = G_inv / det[..., None, None]
        normal = torch.cross(tangent_z, tangent_theta, dim=-1)
        
        return normal, G_inv
        
    def _stream_function_surface_gradient_bases(self,
                                                thetas: torch.Tensor,
                                                zs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the derivative of the stream function w.r.t. theta and z

        Args
        ----
        thetas : torch.Tensor
            Angular positions on the surface (...,)
        zs : torch.Tensor
            z positions on the surface (...,)

        Returns
        -------
        combined_bases_theta : torch.Tensor
            The derivative of the stream function w.r.t. theta with shape (..., K + M)
        combined_bases_zed : torch.Tensor
            The derivative of the stream function w.r.t. z with shape (..., K + M)
        """

        # Gradient of stream function w.r.t. theta
        etas = 2 * (zs - self.zmin) / (self.zmax - self.zmin) - 1 # in [-1, +1]
        tbs_deriv = self.theta_bases_deriv(thetas) # (..., M)
        zbs = self.z_bases(etas) # (..., K)
        combined_bases = zbs[..., None] * tbs_deriv[..., None, :] # ... K, M
        combined_bases_theta = combined_bases.reshape((*zs.shape, -1))

        # Gradient of stream function w.r.t. z
        zbs_deriv = self.z_bases_deriv(etas) # (..., K)
        zbs_deriv *= 2 / (self.zmax - self.zmin)
        tbs = self.theta_bases(thetas) # (..., M)
        combined_bases = zbs_deriv[..., None] * tbs[..., None, :] # ... K, M
        combined_bases_zed = combined_bases.reshape((*zs.shape, -1))

        return combined_bases_theta, combined_bases_zed
    
    def _stream_function_surface_gradient(self,
                                          thetas: torch.Tensor,
                                          zs: torch.Tensor,
                                          stream_coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the derivative of the stream function w.r.t. theta and z

        Args
        ----
        thetas : torch.Tensor
            Angular positions on the surface (...,)
        zs : torch.Tensor
            z positions on the surface (...,)
        stream_coeffs : torch.Tensor
            Stream basis coefficients with shape (Ncoeff,)

        Returns
        -------
        dphi_dtheta : torch.Tensor
            The derivative of the stream function w.r.t. theta with shape (..., Ncoeff)
        dphi_dz : torch.Tensor
            The derivative of the stream function w.r.t. z with shape (..., Ncoeff)
        """

        # Gradient of stream function w.r.t. theta
        etas = 2 * (zs - self.zmin) / (self.zmax - self.zmin) - 1 # in [-1, +1]
        tbs_deriv = self.theta_bases_deriv(thetas) # (..., M)
        zbs = self.z_bases(etas) # (..., K)
        combined_bases = zbs[..., None] * tbs_deriv[..., None, :] # ... K, M
        combined_bases = combined_bases.reshape((*zs.shape, -1))
        dphi_dtheta = combined_bases @ stream_coeffs

        # Gradient of stream function w.r.t. z
        zbs_deriv = self.z_bases_deriv(etas) # (..., K)
        zbs_deriv *= 2 / (self.zmax - self.zmin)
        tbs = self.theta_bases(thetas) # (..., M)
        combined_bases = zbs_deriv[..., None] * tbs[..., None, :] # ... K, M
        combined_bases = combined_bases.reshape((*zs.shape, -1))
        dphi_dz = combined_bases @ stream_coeffs # (..., Ncoeff)

        return dphi_dtheta, dphi_dz

    def _stream_function(self,
                         thetas: torch.Tensor,
                         zs: torch.Tensor,
                         stream_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Computes the stream function phi(theta, z)
        
        Args
        ----
        thetas : torch.Tensor
            Angular positions on the surface (...,)
        zs : torch.Tensor
            z positions on the surface (...,)
        stream_coeffs : torch.Tensor
            Coefficients for the Chebyshev and Fourier expansions with shape (Ncoeff,)
            
        Returns
        -------
        torch.Tensor
            The stream function with shape (...)
        """
        # Build Z-bases
        etas = 2 * (zs - self.zmin) / (self.zmax - self.zmin) - 1 # in [-1, +1]
        zbs = self.z_bases(etas) # (..., K)
        
        # Build theta-bases
        tbs = self.theta_bases(thetas) # (..., M)
        
        # Combine
        combined_bases = zbs[..., None] * tbs[..., None, :] # (..., K, M)
        combined_bases = combined_bases.reshape((*zs.shape, -1))
        
        # Evaluate
        return combined_bases @ stream_coeffs
 
    def _current_density_dS_bases(self,
                                  thetas: torch.Tensor,
                                  zs: torch.Tensor,
                                  return_grad_phi: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the current density for each basis function and the dS factor for surface integration
        
        Args
        ----
        thetas : torch.Tensor
            Angular positions on the surface (...,)
        zs : torch.Tensor
            z positions on the surface (...,)
        return_grad_phi : bool
            If True, return the gradient of the stream function

        Returns
        -------
        Js : torch.Tensor
            The current density basis functions Js (..., Ncoeff, 3)
        dS_factor : torch.Tensor
            The dS factor for surface integration (...)
        grad_phi : torch.Tensor, optional
            The gradient of the stream function with shape (..., 3)
        """
            
        # Flatten thetas, zs
        theta_shape = thetas.shape
        zs_shape = zs.shape
        thetas = thetas.reshape((-1,))
        zs = zs.reshape((-1,))
        Nsurf = len(thetas)
        assert thetas.shape == zs.shape

        # Build bases for stream gradient
        bases_theta, bases_zed = self._stream_function_surface_gradient_bases(thetas, zs) # (Nsurf, ncoeff)
        dphi_stack = torch.stack([bases_theta, bases_zed], dim=-1) # (Nsurf, ncoeff, 2)
        
        # Get surface tangent vectors
        t_theta, t_z = self._surface_tangent_vectors(thetas, zs)
        tangent_stack = torch.stack([t_theta, t_z], dim=-1) # (..., 3, 2)
        
        # Get normal vector and inverse metric tensor
        n, G_inv = self._normal_vector_and_ginv(t_theta, t_z)
        n_hat = n / n.norm(dim=-1, keepdim=True)
        
        # Current density
        grad_phi = tangent_stack[:, None,] @ (G_inv[:, None,] @ dphi_stack[..., None])
        grad_phi = grad_phi[..., 0] # (..., 3)
        Js = torch.cross(n_hat[:, None,], grad_phi, dim=-1) # (Nsurf, ncoeff, 3)
        
        # dS factor for surface integration
        dS_factor = n.norm(dim=-1)[:, None] # (Nsurf)
        
        # Reshape back to original
        Js = Js.reshape((*theta_shape, Js.shape[-2], 3))
        dS_factor = dS_factor.reshape(theta_shape)
        
        if return_grad_phi:
            return Js, dS_factor, grad_phi
        else:
            return Js, dS_factor
   
    def _current_density_dS(self,
                            thetas: torch.Tensor,
                            zs: torch.Tensor,
                            stream_coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the current density vector at the given (theta, z) positions 
        and the dS factor for surface integration
        
        Args
        ----
        thetas : torch.Tensor
            Angular positions on the surface (...,)
        zs : torch.Tensor
            z positions on the surface (...,)
        stream_coeffs : torch.Tensor
            Coefficients for the Chebyshev and Fourier expansions with shape (Ncoeffs,)
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The current density vector Js (..., 3) and dS factor (...) for surface integration
        """
        # Get surface tangent vectors
        t_theta, t_z = self._surface_tangent_vectors(thetas, zs)
        tangent_stack = torch.stack([t_theta, t_z], dim=-1) # (..., 3, 2)
        
        # Get normal vector and inverse metric tensor
        n, G_inv = self._normal_vector_and_ginv(t_theta, t_z)
        n_hat = n / n.norm(dim=-1, keepdim=True)
        
        # Get stream function gradients
        dphi_dtheta, dphi_dz = self._stream_function_surface_gradient(thetas, zs, stream_coeffs)
        dphi_stack = torch.stack([dphi_dtheta, dphi_dz], dim=-1) # (..., 2)
        
        # Compute current density
        grad_phi = tangent_stack @ (G_inv @ dphi_stack[..., None])
        grad_phi = grad_phi[..., 0] # (..., 3)
        Js = torch.cross(n_hat, grad_phi, dim=-1) # (..., 3)
        
        # Compute dS factor for surface integration
        dS_factor = n.norm(dim=-1) # (...)        
        
        return Js, dS_factor
    
    def evaluate_fields(self,
                        coeffs: torch.Tensor,
                        crds_bfield: torch.Tensor,
                        crds_gfield: torch.Tensor,
                        crds_efield: torch.Tensor,
                        batch_size: Optional[int] = 2**3,
                        verbose: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the magnetic and gradient fields at the specified coordinates given the stream function coefficients.
        
        Args
        ----
        coeffs : torch.Tensor
            Coefficients for the Chebyshev and Fourier expansions with shape (Ncoeffs,)
        crds_bfield : torch.Tensor
            shape (*bshape, 3) representing the coordinates where magnetic field is evaluated.   
        crds_gfield : torch.Tensor
            shape (*gshape, 3) representing the coordinates where the gradient field is evaluated.
        crds_efield : torch.Tensor
            shape (*eshape, 3) representing the coordinates where the electric field is evaluated.
        batch_size : Optional[int]
            Batch size for processing surface points in chunks to save memory
            if None, process all points at once
        verbose : bool
            If True, print progress bars
        
        Returns
        -------
        bfield : torch.Tensor
            Magnetic field at observation points with shape (*bshape, 3)
        gfield : torch.Tensor
            Gradient field at observation points with shape (*gshape, 3)
        efield : torch.Tensor
            Electric field at observation points with shape (*eshape, 3)
        """
        # Consts
        dtype = crds_bfield.dtype
        Ncoeffs = self.M * self.K
        mu0_over_4pi = 1e-7 # in T m / A
        I = torch.eye(3, device=self.torch_dev, dtype=dtype)
        gshape = crds_gfield.shape[:-1]
        bshape = crds_bfield.shape[:-1]
        eshape = crds_efield.shape[:-1]
        
        # flatten input coordinates
        crds_bfield = crds_bfield.view((-1, 3))
        crds_gfield = crds_gfield.view((-1, 3))
        crds_efield = crds_efield.view((-1, 3))
        
        # Flattened surface theta, z, and positions
        # theta_surf = torch.linspace(0, 2 * torch.pi, self.num_theta, device=self.torch_dev, dtype=dtype)
        theta_surf = torch.arange(0, 2 * torch.pi, 2 * torch.pi / self.num_theta, device=self.torch_dev, dtype=dtype)
        dtheta = theta_surf[1] - theta_surf[0]
        z_surf = torch.linspace(self.zmin, self.zmax, self.num_zs, device=self.torch_dev, dtype=dtype)
        dz = z_surf[1] - z_surf[0]
        theta_surf, z_surf = torch.meshgrid(theta_surf, z_surf, indexing='ij') # (T, Z)
        theta_surf = theta_surf.reshape(-1) # Nsurf
        z_surf = z_surf.reshape(-1) # Nsurf
        crds_surf = self._surface_positions(theta_surf, z_surf) # (Nsurf, 3)
        Nsurf = crds_surf.shape[0]
        
        # Placeholder for fields
        bfields = torch.zeros((crds_bfield.shape[0], 3), dtype=dtype, device=self.torch_dev) # Bx By Bz
        gfields = torch.zeros((crds_gfield.shape[0], 3), dtype=dtype, device=self.torch_dev) # dBz/dx, dBz/dy, dBz/dz
        efields = torch.zeros((crds_efield.shape[0], 3), dtype=dtype, device=self.torch_dev) # Ex, Ey, Ez
        
        # Get current density
        Js, dS_factor = self._current_density_dS(theta_surf, z_surf, coeffs) # (Nsurf, 3)
        dS_factor *= dtheta * dz
        
        # Integrate over surface to get fields
        batch_size = Nsurf if batch_size is None else batch_size
        for n1 in tqdm(range(0, Nsurf, batch_size), 'Evaluating Fields', disable=not verbose):
            n2 = min(n1 + batch_size, Nsurf)
            
            # --------- Magnetic field ---------
            # Difference vectors
            diff = crds_bfield[:, None, :] - crds_surf[None, n1:n2, :] # (N, Nsurf, 3)
            
            # Bfield calc
            numer = torch.cross(Js[None, n1:n2], diff, dim=-1) * dS_factor[None, n1:n2, None]
            denom = diff.norm(dim=-1, keepdim=True) ** 3 # (N, Nsurf, 1)
            bfields += mu0_over_4pi * (numer / denom).sum(dim=1)
            
            # --------- Gradient field ---------
            # Difference vectors
            diff = crds_gfield[:, None, :] - crds_surf[None, n1:n2, :] # (N, Nsurf, 3)
            
            # # Gfield calc (old method)
            # nrm_sq = diff.norm(dim=-1) ** 2
            # numer = I * nrm_sq[..., None, None]  - 3 * diff[..., :, None] * diff[..., None, :] # (nbatch, Nsurf, 3, 3)
            # numer = torch.cross(Js[None, n1:n2, :, None], numer, dim=-2)[..., -1, :] * dS_factor[None, n1:n2, None]
            # denom = nrm_sq[..., None] ** (5/2)
            # gfields += mu0_over_4pi * (numer / denom).sum(dim=1)
            
            # Gfield calc
            denom1 = diff.norm(dim=-1, keepdim=True) ** 3 # N Nsurf 1
            numer1 = Js[None, n1:n2, :] @ self.lct.T # 1 Nsurf 3
            numer1 = numer1 * dS_factor[None, n1:n2, None]
            denom2 = diff.norm(dim=-1, keepdim=True) ** 5 # N Nsurf 1
            numer2 = (Js[None, n1:n2, :, None] * self.lct[None, None]).sum(dim=-2) # 1 Nsurf 3
            numer2 = (numer2 * diff).sum(dim=-1) # N Nsurf
            numer2 = numer2[..., None] * diff # N Nsurf 3
            numer2 = numer2 * dS_factor[None, n1:n2, None]
            gfields += mu0_over_4pi * (numer1 / denom1 + 3 * numer2 / denom2).sum(dim=1)
            
            
            # --------- Electric field ---------
            # Difference vectors
            diff = crds_efield[:, None, :] - crds_surf[None, n1:n2, :] # (N, Nsurf, 3)
            
            # Efield calc
            numer = (Js[None, n1:n2, :] * dS_factor[None, n1:n2, None]) # (N, Nsurf, 3)
            denom = diff.norm(dim=-1, keepdim=True) # (N, Nsurf, 1)
            efields += mu0_over_4pi * (numer / denom).sum(dim=1)

        return bfields.reshape(*bshape, 3), gfields.reshape(*gshape, 3), efields.reshape(*eshape, 3)

    def build_winding_tolerance_matrix(self,
                                       num_theta: int = 50,
                                       num_zs: int = 200) -> torch.Tensor:
        """
        Matrix relating the stream function coefficients to the winding tolerance.
        We want the current to flow in the correct direction around the coil.
        
        ||winding_tol @ stream_coeffs||_2 <= upper_bound
        
        Args
        -----
        num_theta : int, optional
            Number of theta points on the two boundaries to enforce zero exit current, defaults to 100
        num_zs : int, optional
            Number of z points on the two boundaries to enforce zero exit current, defaults to 100
            
        Returns
        -------
        winding_tol : torch.Tensor
            Winding tolerance matrix with shape (Nsurf, ncoeff, 3)
        """
        
        # Gen theta zed points
        thetas = torch.arange(0, 2 * torch.pi, 2 * torch.pi / num_theta, device=self.torch_dev)
        zs = torch.linspace(self.zmin, self.zmax, num_zs, device=self.torch_dev)
        thetas, zs = torch.meshgrid(thetas, zs, indexing='ij') # (T, Z)
        thetas = thetas.reshape(-1) # Nsurf
        zs = zs.reshape(-1) # Nsurf
        
        # Coimpute surface gradient of stream function
        _, _, winding_tol = self._current_density_dS_bases(thetas, zs, return_grad_phi=True)
        
        return winding_tol

    def build_current_boundary_matrix(self,
                                      verbose: bool = False) -> torch.Tensor:
        """
        Matrix relating the stream function coefficients to the current boundary conditions.
        We want zero current to flow off the ends of the surface.
        
        Args
        -----
        num_theta : int
            Number of theta points on the two boundaries to enforce zero exit current
        verbose : bool
            If True, print progress bars
        
        Returns
        -------
        torch.Tensor
            Current boundary condition matrix A of shape (2 * num_theta, ncoeffs)
        """
        # Consts
        Ncoeffs = self.M * self.K
        
        # Gen theta points
        # thetas = torch.linspace(0, 2 * torch.pi, num_theta, device=self.torch_dev)
        thetas = torch.arange(0, 2 * torch.pi, 2 * torch.pi / self.num_theta, device=self.torch_dev)
        assert len(thetas) == self.num_theta
        
        # Current matrix placeholder
        Jmat = torch.zeros((Ncoeffs, 2 * self.num_theta), dtype=torch.float32, device=self.torch_dev)
        
        # Get current density at boundaries in normal direction
        one_hot = torch.zeros(Ncoeffs, device=self.torch_dev)
        for c in tqdm(range(Ncoeffs), 'Building Field Matrix', disable=not verbose):
            one_hot *= 0
            one_hot[c] = 1.0        
            
            # Get normal current at zmin
            zmins = torch.full_like(thetas, self.zmin)
            _, normal_zmin = self._surface_tangent_vectors(thetas, zmins)
            Js_min, _ = self._current_density_dS(thetas, zmins, one_hot) # (Nsurf, 3)
            Jn_min = (Js_min * normal_zmin).sum(dim=-1) # (Nsurf,)
            
            # Get normal current at zmax
            zmaxs = torch.full_like(thetas, self.zmax)
            _, normal_zmax = self._surface_tangent_vectors(thetas, zmaxs)
            Js_max, _ = self._current_density_dS(thetas, zmaxs, one_hot) # (Nsurf, 3)
            Jn_max = (Js_max * normal_zmax).sum(dim=-1) # (Nsurf,)
            
            Jmat[c, :self.num_theta] = Jn_min
            Jmat[c, self.num_theta:] = Jn_max
        
        return Jmat.T
    
    def build_field_matrices_fast(self,
                                  crds_bfield: torch.Tensor,
                                  crds_gfield: torch.Tensor,
                                  crds_efield: torch.Tensor,
                                  batch_size: Optional[int] = 2**4,
                                  verbose: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Builds the matrix that maps stream function coefficients to magnetic field at observation points
        
        Args
        ----
        crds_bfield : torch.Tensor
            shape (*bshape, 3) representing the coordinates where magnetic field is evaluated.   
        crds_gfield : torch.Tensor
            shape (*gshape, 3) representing the coordinates where the gradient field is evaluated.
        crds_efield : torch.Tensor
            shape (*eshape, 3) representing the coordinates where the electric field is evaluated.
        batch_size : Optional[int]
            Batch size for processing surface points in chunks to save memory
            if None, process all points at once
        verbose : bool
            If True, print progress bars
        
        Returns
        -------
        bfields : torch.Tensor
            Magnetic field matrix B (*bshape, ncoeff, 3)
        gfields : torch.Tensor
            Gradient field matrix G (*gshape, ncoeff, 3)
        efields : torch.Tensor
            Electric field matrix E (*eshape, ncoeff, 3)
        """
        
        # Consts
        batch_size = batch_size if batch_size is not None else Nsurf
        mu0_over_4pi = 1e-7 # in T m / A
        I = torch.eye(3, device=self.torch_dev, dtype=crds_bfield.dtype)
        gshape = crds_gfield.shape[:-1]
        bshape = crds_bfield.shape[:-1]
        eshape = crds_efield.shape[:-1]
        crds_bfield = crds_bfield.reshape((-1, 3))
        crds_gfield = crds_gfield.reshape((-1, 3))
        crds_efield = crds_efield.reshape((-1, 3))
        
        # Generate surface points
        # thetas = torch.linspace(0, 2 * torch.pi, self.num_theta, device=self.torch_dev)
        thetas = torch.arange(0, 2 * torch.pi, 2 * torch.pi / self.num_theta, device=self.torch_dev)
        assert len(thetas) == self.num_theta
        dtheta = thetas[1] - thetas[0]
        zs = torch.linspace(self.zmin, self.zmax, self.num_zs, device=self.torch_dev)
        dz = zs[1] - zs[0]
        thetas, zs = torch.meshgrid(thetas, zs, indexing='ij') # (T, Z)
        thetas = thetas.reshape(-1) # Nsurf
        zs = zs.reshape(-1) # Nsurf
        crds_surf = self._surface_positions(thetas, zs) # (Nsurf, 3)
        Nsurf = crds_surf.shape[0]

        # Get current density bases and dS factor
        Js, dS_factor = self._current_density_dS_bases(thetas, zs) # (Nsurf, ncoeff, 3), (Nsurf,)
        dS_factor *= dtheta * dz

        # Placeholder for field matrices
        Ncoeffs = self.M * self.K
        bfields = torch.zeros((crds_bfield.shape[0], Ncoeffs, 3), dtype=crds_bfield.dtype, device=self.torch_dev) # Bx By Bz
        gfields = torch.zeros((crds_gfield.shape[0], Ncoeffs, 3), dtype=crds_bfield.dtype, device=self.torch_dev) # dBz/dx, dBz/dy, dBz/dz
        afields = torch.zeros((crds_efield.shape[0], Ncoeffs, 3), dtype=crds_bfield.dtype, device=self.torch_dev) # Ax, Ay, Az
        
        # Evaluate bio savart on current density
        for n1 in tqdm(range(0, Nsurf, batch_size), 'Building Field Matrices', disable=not verbose):
            n2 = min(n1 + batch_size, Nsurf)
            
            # --------- Magnetic field ---------
            diff_mgnt = crds_bfield[:, None, None, :] - crds_surf[None, n1:n2, None, :] # (N, Nsurf, 1, 3)
            numer_mgnt = torch.cross(Js[None, n1:n2], diff_mgnt, dim=-1) * dS_factor[None, n1:n2, None, None] # (N, Nsurf, Ncoeff, 3)
            denom_mgnt = diff_mgnt.norm(dim=-1, keepdim=True) ** 3 # (N, Nsurf, 1, 1)
            bfields += mu0_over_4pi * (numer_mgnt / denom_mgnt).sum(dim=1)
            
            # --------- Gradient field ---------
            diff_grdnt = crds_gfield[:, None, :] - crds_surf[None, n1:n2, :] # (N, Nsurf, 3)
            denom1 = diff_grdnt.norm(dim=-1) ** 3 # (N, Nsurf,)
            numer1 = (self.lct[None, None, :, :] @ Js[n1:n2, :, :, None])[..., 0]
            denom2 = diff_grdnt.norm(dim=-1) ** 5 # (N, Nsurf,)
            numer2 = (self.lct[None, None, :, :] @ diff_grdnt[..., None])[..., 0] # N Nsurf 3
            numer2 = (Js[None, n1:n2, :, :] * numer2[:, :, None, :]).sum(dim=-1) # N Nsurf ncoeff
            numer2 = (numer2[..., None] * diff_grdnt[:, :, None, :]) * 3
            integrand = ((numer1 / denom1[:, :, None, None]) + (numer2 / denom2[:, :, None, None])) * dS_factor[None, n1:n2, None, None]
            gfields += mu0_over_4pi * integrand.sum(dim=1)
            
            # # --------- Gradient field (old method) ---------
            # diff_grdnt = crds_gfield[:, None, None, :] - crds_surf[None, n1:n2, None, :] # (N, Nsurf, 1, 3)
            # nrm_sq = diff_grdnt.norm(dim=-1) ** 2 # (N, Nsurf, 1)
            # numer_grdt = I * nrm_sq[..., None, None]  - 3 * diff_grdnt[..., :, None] * diff_grdnt[..., None, :] # (N, Nsurf, 1, 3, 3)
            # numer_grdt = torch.cross(Js[None, n1:n2, :, :, None], numer_grdt, dim=-2)[..., -1, :] * dS_factor[None, n1:n2, None, None]
            # denom_grdt = nrm_sq[..., None] ** (5/2)
            # gfields += mu0_over_4pi * (numer_grdt / denom_grdt).sum(dim=1)
            
            # --------- Magnetic vector potential ---------
            diff_afld = crds_efield[:, None, None, :] - crds_surf[None, n1:n2, None, :] # (N, Nsurf, 1, 3)
            numer_afld = (Js[None, n1:n2, :] * dS_factor[None, n1:n2, None, None]) # (N, Nsurf, Ncoeff, 3)
            denom_afld = diff_afld.norm(dim=-1, keepdim=True) # (N, Nsurf, 1, 1)
            afields += mu0_over_4pi * (numer_afld / denom_afld).sum(dim=1)

        # Reshape to original input shape
        bfields = bfields.reshape((*bshape, Ncoeffs, 3))
        gfields = gfields.reshape((*gshape, Ncoeffs, 3))
        efields = -afields.reshape((*eshape, Ncoeffs, 3)) # E = -A
        return bfields, gfields, efields

    def build_field_matrices(self,
                             crds_bfield: torch.Tensor,
                             crds_gfield: torch.Tensor,
                             crds_efield: torch.Tensor,
                             verbose: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Builds the matrix that maps stream function coefficients to magnetic field at observation points
        
        Args
        ----
        crds_bfield : torch.Tensor
            shape (*bshape, 3) representing the coordinates where magnetic field is evaluated.   
        crds_gfield : torch.Tensor
            shape (*gshape, 3) representing the coordinates where the gradient field is evaluated.
        crds_efield : torch.Tensor
            shape (*eshape, 3) representing the coordinates where the electric field is evaluated.
        batch_size : Optional[int]
            Batch size for processing surface points in chunks to save memory
            if None, process all points at once
        
        Returns
        -------
        bfields : torch.Tensor
            Magnetic field matrix B (*bshape, ncoeff, 3)
        gfields : torch.Tensor
            Gradient field matrix G (*gshape, ncoeff, 3)
        efields : torch.Tensor
            Electric field matrix E (*eshape, ncoeff, 3)
        """
        return self.build_field_matrices_fast(crds_bfield, crds_gfield, crds_efield, verbose=verbose)
        # Placeholder for field matrices
        Ncoeffs = self.M * self.K
        bfields = torch.zeros((*crds_bfield.shape[:-1], Ncoeffs, 3), dtype=crds_bfield.dtype, device=self.torch_dev) # Bx By Bz
        gfields = torch.zeros((*crds_gfield.shape[:-1], Ncoeffs, 3), dtype=crds_bfield.dtype, device=self.torch_dev) # dBz/dx, dBz/dy, dBz/dz
        efields = torch.zeros((*crds_efield.shape[:-1], Ncoeffs, 3), dtype=crds_bfield.dtype, device=self.torch_dev) # Ex, Ey, Ez
        
        # Evaluate fields for each one-hot coeff
        for c in tqdm(range(Ncoeffs), 'Building Field Matrix', disable=not verbose):
            one_hot = torch.zeros(Ncoeffs, device=self.torch_dev)
            one_hot[c] = 1.0
            bfields[..., c, :], gfields[..., c, :], efields[..., c, :] = self.evaluate_fields(one_hot, crds_bfield, crds_gfield, crds_efield, verbose=False)
            
        return bfields, gfields, efields

    def build_magnetic_energy_matrix(self,
                                     batch_size: Optional[int] = 2**4,
                                     verbose: bool = True,) -> torch.Tensor:
        """
        Builds the magnetic energy matrix M such that the magnetic energy is coeffs^T M coeffs.
        
        Formulat is 
        mu_0 / 8pi * int_r int_r' J(r) . J(r') / |r - r'| dr dr'
        
        Args
        ----
        batch_size : Optional[int]
            Batch size for processing surface points in chunks to save memory
            if None, process all points at once
        verbose : bool
            If True, print progress bars
            
        Returns
        -------
        torch.Tensor
            The magnetic energy matrix with shape (ncoeff, ncoeff).
        """
        # Consts
        Ncoeffs = self.M * self.K
        mu0_over_4pi = 1e-7
        debug_method = True
        
        # Gen surface theta, z, and positions
        # theta_surf = torch.linspace(0, 2 * torch.pi, self.num_theta, device=self.torch_dev)
        theta_surf = torch.arange(0, 2 * torch.pi, 2 * torch.pi / self.num_theta, device=self.torch_dev)
        dtheta = theta_surf[1] - theta_surf[0]
        assert len(theta_surf) == self.num_theta
        z_surf = torch.linspace(self.zmin, self.zmax, self.num_zs, device=self.torch_dev)
        dz = z_surf[1] - z_surf[0]
        theta_surf, z_surf = torch.meshgrid(theta_surf, z_surf, indexing='ij') # (T, Z)
        theta_surf = theta_surf.reshape(-1) # Nsurf
        z_surf = z_surf.reshape(-1) # Nsurf
        crds_surf = self._surface_positions(theta_surf, z_surf) # (Nsurf, 3)
        Nsurf = crds_surf.shape[0]
        
        # Get current density bases and dS factor
        Js, dS_factor = self._current_density_dS_bases(theta_surf, z_surf) # (Nsurf, ncoeff, 3)
        dS_factor *= dtheta * dz
        
        # placeholder for energy matrix
        W = torch.zeros((Ncoeffs, Ncoeffs), dtype=torch.float32, device=self.torch_dev)
        
        # Loop over surface points
        for n1 in tqdm(range(0, Nsurf, batch_size), 'Building Energy Matrix', disable=not verbose):
            n2 = min(n1 + batch_size, Nsurf)
            
            # Magnetic energy integral
            eps = 1e-3 ** 2
            diff = crds_surf[:, None, :] - crds_surf[None, n1:n2, :] # (Nsurf, Nbatch, 3)
            inv_nrm = 1.0 / (diff.square().sum(dim=-1) + eps).sqrt() # (Nsurf, Nbatch)
            if not debug_method:
                dot_prod = einsum(Js * dS_factor[:, None, None], 
                                Js[n1:n2] * dS_factor[n1:n2, None, None], 
                                'N Ci d, B Co d -> Ci Co N B') # very very slow
                integral = einsum(dot_prod, inv_nrm, 'Ci Co N B, N B -> Ci Co')
            else:
                inv_nrm = inv_nrm.T
                integral = einsum(Js[None,] * dS_factor[None, :, None, None] * inv_nrm[:, :, None, None], 
                                Js[n1:n2, None] * dS_factor[n1:n2, None, None, None], 
                                'B N Ci d, B N Co d -> Ci Co') # very very slow
            W += 0.5 * mu0_over_4pi * integral
            
        L = 2 * W / (1 ** 2) # Inductance from energy formula
        return L

    # TODO fix .cpu() calls by making interpolation on GPU
    def stream_to_contour(self,
                          stream_coeffs: torch.Tensor,
                          num_theta: int = 100,
                          num_z: int = 200,
                          dstream: Optional[float] = None,) -> tuple[list[torch.Tensor], float]:
        """
        Computes the contour paths of the stream function at given levels
        
        Args
        ----
        stream_coeffs : torch.Tensor
            Coefficients for the Chebyshev and Fourier expansions with shape (Ncoeffs,)
        num_theta : int
            Number of theta points on the surface
        num_z : int
            Number of z points on the surface
        dstream : float
            Level spacing for countours
            If none, uses 50 levels spaced evenly between the min and max of the stream function
            
        Returns
        -------
        list[torch.Tensor]
            List of contour paths, i^th entry of list has shape (Npoints_i, 3)
        """
        # Evaluate stream function
        zs = torch.linspace(self.zmin, self.zmax, num_z, device=self.torch_dev)
        thetas = torch.linspace(0, 2 * torch.pi, num_theta, device=self.torch_dev)
        # thetas = torch.arange(0, 2 * torch.pi, 2 * torch.pi / num_theta, device=self.torch_dev)
        TT, ZZ = torch.meshgrid(thetas, zs, indexing='ij') # (T, Z)
        stream = self._stream_function(TT, ZZ, stream_coeffs) # (T, Z)

        # Set default dstream
        if dstream is None:
            dstream = (stream.max() - stream.min()).item() / 100
            print(f'Steam min = {stream.min():1.2e}')
            print(f'Steam max = {stream.max():1.2e}')
            print(f'dstream = {dstream:1.2e}')
            
        # Level sets
        levels = torch.arange(stream.min(), stream.max(), dstream) + dstream/2
        thetas = thetas.cpu().numpy()
        zs = zs.cpu().numpy()
        xyz_contours = []
        theta_z_contours = []
        for level in levels:
            
            # Find countours
            stream_cat = torch.cat([stream[-1:, :], stream], dim=0)
            # stream_cat = torch.cat([stream, stream[:1, :]], dim=0)
            countours_rc = measure.find_contours(stream_cat.cpu().numpy(), level=level)
            
            # Plot countours
            for rc in countours_rc:
                
                # Interpolate to get thetas and zs
                thetas_contour = interp1d(np.arange(len(thetas)), thetas, 
                                          kind='linear', fill_value='extrapolate')(rc[:, 0])
                zs_contour = interp1d(np.arange(len(zs)), zs,
                                      kind='linear', fill_value='extrapolate')(rc[:, 1])
                
                # Convert to 3D coordinates
                thetas_contour = torch.from_numpy(thetas_contour).to(self.torch_dev)
                zs_contour = torch.from_numpy(zs_contour).to(self.torch_dev)
                xyz_contour = self._surface_positions(thetas_contour, zs_contour).cpu()
                
                # Save flattened contour path
                thetas_zs_contour = torch.stack([thetas_contour, zs_contour], dim=-1).type(torch.float32)
                theta_z_contours.append(thetas_zs_contour.reshape(-1, 2))
                xyz_contour = xyz_contour.type(torch.float32)
                xyz_contours.append(xyz_contour.reshape(-1, 3))
                
        # For each contour, make sure the winding direction is correct
        flipped = 0
        for i in range(len(xyz_contours)):
            theta_zs_surf = theta_z_contours[i][:1]
            J_surf, _ = self._current_density_dS(theta_zs_surf[:, 0], theta_zs_surf[:, 1], stream_coeffs)
            tangent = xyz_contours[i][1] - xyz_contours[i][0]
            sign = torch.sign(J_surf.unsqueeze(0).cpu() @ tangent)
            if sign < 0:
                flipped += 1
                xyz_contours[i] = xyz_contours[i].flip(dims=[0])
        print(f'Flipped {flipped}/{len(xyz_contours)} contours')
        return xyz_contours, dstream
    
    def stream_to_winding(self,
                          stream_coeffs: torch.Tensor,
                          num_theta: int = 100,
                          num_z: int = 200,
                          dstream: Optional[float] = None,) -> tuple[list[torch.Tensor], float]:
        """
        Computes the contour paths of the stream function at given levels
        
        Args
        ----
        stream_coeffs : torch.Tensor
            Coefficients for the Chebyshev and Fourier expansions with shape (Ncoeffs,)
        num_theta : int
            Number of theta points on the surface
        num_z : int
            Number of z points on the surface
        dstream : float
            Level spacing for countours
            If none, uses 50 levels spaced evenly between the min and max of the stream function
            
        Returns
        -------
        list[torch.Tensor]
            List of contour paths, i^th entry of list has shape (Npoints_i, 3)
        """
        print(f'WARNING: winding calculation not implemented')
        # Evaluate stream function
        zs = torch.linspace(self.zmin, self.zmax, num_z, device=self.torch_dev)
        thetas = torch.linspace(0, 2 * torch.pi, num_theta, device=self.torch_dev)
        # thetas = torch.arange(0, 2 * torch.pi, 2 * torch.pi / num_theta, device=self.torch_dev)
        TT, ZZ = torch.meshgrid(thetas, zs, indexing='ij') # (T, Z)
        stream = self._stream_function(TT, ZZ, stream_coeffs) # (T, Z)

        # Set default dstream
        if dstream is None:
            dstream = (stream.max() - stream.min()).item() / 100
            print(f'Steam min = {stream.min():1.2e}')
            print(f'Steam max = {stream.max():1.2e}')
            print(f'dstream = {dstream:1.2e}')
            
        # List of connected winding segments. There may be a few since the stream may have disconnected level sets
        winding_segments = []
            
        # Level sets
        levels = torch.arange(stream.min(), stream.max(), dstream) + dstream/2
        thetas = thetas.cpu().numpy()
        zs = zs.cpu().numpy()
        xyz_contours = []
        tz_contours = []
        theta_z_contours = []
        stream_levels = []
        for level in levels:
            
            # Find countours
            stream_cat = torch.cat([stream, stream[:1, :]], dim=0)
            countours_rc = measure.find_contours(stream_cat.cpu().numpy(), level=level)
            
            # Plot countours
            for rc in countours_rc:
                
                # Interpolate to get thetas and zs
                thetas_contour = interp1d(np.arange(len(thetas)), thetas, 
                                          kind='linear', fill_value='extrapolate')(rc[:, 0])
                zs_contour = interp1d(np.arange(len(zs)), zs,
                                      kind='linear', fill_value='extrapolate')(rc[:, 1])
                
                
                # Convert to 3D coordinates
                thetas_contour = torch.from_numpy(thetas_contour).to(self.torch_dev)
                zs_contour = torch.from_numpy(zs_contour).to(self.torch_dev)
                tz_contours.append(torch.stack([thetas_contour, zs_contour], dim=-1).type(torch.float32))
                xyz_contour = self._surface_positions(thetas_contour, zs_contour).cpu()
                
                # Save flattened contour path
                thetas_zs_contour = torch.stack([thetas_contour, zs_contour], dim=-1).type(torch.float32)
                theta_z_contours.append(thetas_zs_contour.reshape(-1, 2))
                xyz_contour = xyz_contour.type(torch.float32)
                xyz_contours.append(xyz_contour.reshape(-1, 3))
                stream_levels.append(level)
                
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(stream.cpu().rot90(), cmap='viridis',
                   extent=[thetas.min(), thetas.max(), zs.min(), zs.max()], 
                   aspect='auto')
        for tz_contour in tz_contours:
            plt.plot(tz_contour[:, 0].cpu(), tz_contour[:, 1].cpu(), color='red')
            plt.scatter(tz_contour[0, 0].cpu(), tz_contour[0, 1].cpu(), color='green', marker='x')
        plt.colorbar()
        
        # plot 3D surface with countours
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(TT.cpu(), ZZ.cpu(), stream.cpu(), cmap='viridis')
        for k, tz_contour in enumerate(tz_contours):
            height = tz_contour[:, 1].cpu() * 0 + stream_levels[k]
            ax.plot(tz_contour[:, 0].cpu(), tz_contour[:, 1].cpu(), height, color='red')
            ax.scatter(tz_contour[0, 0].cpu(), tz_contour[0, 1].cpu(), height[0], color='green', marker='x')
        
        plt.figure()
        plt.hist(stream.cpu().flatten(), bins=100)
        for level in levels:
            plt.axvline(level, color='red', linestyle='--')
        plt.xlabel('Stream Function Value')
        plt.ylabel('Count')
        plt.title('Stream Function Distribution')
        plt.legend()
        plt.show()
        breakpoint()
                
        # For each contour, make sure the winding direction is correct
        flipped = 0
        for i in range(len(xyz_contours)):
            theta_zs_surf = theta_z_contours[i][:1]
            J_surf, _ = self._current_density_dS(theta_zs_surf[:, 0], theta_zs_surf[:, 1], stream_coeffs)
            tangent = xyz_contours[i][1] - xyz_contours[i][0]
            sign = torch.sign(J_surf.unsqueeze(0).cpu() @ tangent)
            if sign < 0:
                flipped += 1
                xyz_contours[i] = xyz_contours[i].flip(dims=[0])
        print(f'Flipped {flipped}/{len(xyz_contours)} contours')
        
        return xyz_contours, dstream
    
    def show_countour(self,
                      stream_coeffs: torch.Tensor,
                      num_theta: int = 100,
                      num_z: int = 50,
                      dstream: Optional[float] = None,
                      fig: Optional[plt.Figure] = None,
                      ax: Optional[plt.Axes] = None,) -> tuple[plt.Figure, plt.Axes]:
        """
        Plots countours of the stream function on the surface
        
        Args
        ----
        stream_coeffs : torch.Tensor
            Coefficients for the Chebyshev and Fourier expansions with shape (Ncoeffs,)
        num_theta : int
            Number of theta points on the surface
        num_z : int
            Number of z points on the surface
        dstream : float
            Level spacing for countours
            If none, uses 50 levels spaced evenly between the min and max of the stream function
        fig : Optional[plt.Figure]
            Matplotlib Figure to plot on. If None, a new figure and axes are created
        ax : Optional[plt.Axes]
            Matplotlib Axes to plot on. If None, a new figure and axes are created
            
        Returns
        -------
        fig : plt.Figure
            The figure showing the countours.
        ax : plt.Axes
            The axis showing the countours.
        """
        # If not ax provided, create one
        if fig is None or ax is None:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, projection='3d')
        
        # Set axes labels
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_zlabel('z (cm)')
        ax.set_aspect('equal')
        
        # Plot contours
        xyz_contours, dstream = self.stream_to_contour(stream_coeffs, num_theta, num_z, dstream=dstream)
        for xyz_contour in xyz_contours:
            ax.plot(xyz_contour[..., 0] * 1e2, xyz_contour[..., 1] * 1e2, xyz_contour[..., 2] * 1e2, color='red', alpha=0.3)
        return fig, ax
        
    def show_current_density(self,
                             stream_coeffs: torch.Tensor,
                             num_theta: int = 100,
                             num_z: int = 50,
                             fig: Optional[plt.Figure] = None,
                             ax: Optional[plt.Axes] = None,):
        """
        Plots current density as vector field on the surface
        
        Args
        ----
        stream_coeffs : torch.Tensor
            Coefficients for the Chebyshev and Fourier expansions with shape (Ncoeffs,)
        num_theta : int
            Number of theta points on the surface
        num_z : int
            Number of z points on the surface
        fig : Optional[plt.Figure]
            Matplotlib Figure to plot on. If None, a new figure and axes are created
        ax : Optional[plt.Axes]
            Matplotlib Axes to plot on. If None, a new figure and axes are created
        """
        
        # If not ax provided, create one
        if fig is None or ax is None:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, projection='3d')
        
        # Evaluate current density
        thetas = torch.linspace(0, 2 * torch.pi, num_theta, device=self.torch_dev)
        zs = torch.linspace(self.zmin, self.zmax, num_z, device=self.torch_dev)
        thetas, zs = torch.meshgrid(thetas, zs, indexing='ij') # (T, Z)
        crds = self._surface_positions(thetas, zs)
        Js, _ = self._current_density_dS(thetas, zs, stream_coeffs)
        
        # Quiver plot
        crds = crds.reshape(-1, 3).cpu() * 1e2
        Js = Js.reshape(-1, 3).cpu()
        mag = Js.norm(dim=-1, keepdim=True)
        Js_dir = (Js / mag)
        mag /= mag.max()
        Js = Js_dir * (mag ** 0.5)
        ax.quiver(crds[..., 0], crds[..., 1], crds[..., 2],
                  Js[..., 0], Js[..., 1], Js[..., 2],
                  length=2, alpha=0.4,
                  color='red')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_zlabel('z (cm)')
        ax.set_aspect('equal')
        
        return fig, ax
        
    def show_design(self,
                    coeffs: torch.Tensor,
                    num_theta: int = 100,
                    num_z: int = 50,
                    body_surf: Optional[torch.Tensor] = None,
                    show_1d: bool = True,
                    colorbar: bool = True,
                    fig: Optional[plt.Figure] = None,
                    ax: Optional[plt.Axes] = None) -> tuple[plt.Figure, plt.Axes]:
        # If not ax provided, create one
        if fig is None or ax is None:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, projection='3d')
            
        # Gen coordinates
        zs = torch.linspace(self.zmin, self.zmax, num_z, device=self.torch_dev)
        thetas = torch.linspace(0, 2 * torch.pi, num_theta, device=self.torch_dev)
        thetas, zs = torch.meshgrid(thetas, zs, indexing='ij') # (T, Z)
        
        # Cartesian
        crds = self._surface_positions(thetas, zs)
        xs, ys, zs = crds[..., 0], crds[..., 1], crds[..., 2]
        vals = self._stream_function(thetas, zs, coeffs)
        
        # to CPU
        xs, ys, zs, vals = xs.cpu(), ys.cpu(), zs.cpu(), vals.cpu()
        
        # Plot
        if body_surf is not None:
            alpha = 0.5
        else:
            alpha = 1.0
        norm = plt.Normalize(vmin=-vals.abs().max(), vmax=vals.abs().max())
        colormap = cm.berlin
        colors = colormap(norm(vals))
        sm = cm.ScalarMappable(norm=norm, cmap=colormap)
        ax.plot_surface(xs * 1e2, ys * 1e2, zs * 1e2,
                        facecolors=colors,
                        rcount=zs.shape[0],
                        ccount=zs.shape[1],
                        alpha=alpha,
                        shade=False,
                        linewidth=0,
                        edgecolor='none')
        # ax.set_xlabel('x (cm)')
        # ax.set_ylabel('y (cm)')
        # ax.set_zlabel('z (cm)')
        ax.set_aspect('equal')
        
        # # Turn off grid
        # ax.grid(False)
        # ax.axis('off')
        
        # Show colorbar
        if colorbar:    
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('Stream Function Value')    
        
        # Show body surface
        if body_surf is not None:
            ax.plot_surface(body_surf[..., 0].cpu(), body_surf[..., 1].cpu(), body_surf[..., 2].cpu(),
                            color='navajowhite', alpha=1.0, shade=True, 
                            linewidth=0,
                            edgecolor='none',)
            plt.axis('equal')
            
        
        # Evaluate current density over +x, -x, +y, -y lines
        if show_1d:
            thetas = torch.tensor([0.0, np.pi, np.pi/2, 3*np.pi/2], device=self.torch_dev)
            zs = torch.linspace(self.zmin, self.zmax, num_z, device=self.torch_dev)
            thetas, zs = torch.meshgrid(thetas, zs, indexing='ij') # (4, Z)
            stream = self._stream_function(thetas, zs, coeffs)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            angles = ['+x', '-x', '+y', '-y']
            for i in range(4):
                ax.plot(zs[i].cpu(), stream[i, :].cpu(), label=angles[i])
            ax.set_xlabel('z (m)')
            ax.set_ylabel('Stream Function Value')
            ax.legend()
            ax.set_title('Stream Function Along Principal Axes')
            
        return fig, ax
    
                    
# TODO build via analytic bases
def test_fast_transforms(thetas: torch.Tensor,
                         zs: torch.Tensor,
                         stream_coeffs: torch.Tensor,
                         M_fourier: int = 10,
                         K_chebyshev: int = 10) -> torch.Tensor:
    assert zs.min().item() >= -1.0 and zs.max().item() <= 1.0, "zs must be in [-1, +1]"
    
    # Naive transform
    ms = torch.arange(0, M_fourier, device=thetas.device)
    ks = torch.arange(0, K_chebyshev, device=thetas.device)
    bs = 2 ** 8
    for c1 in tqdm(range(0, len(stream_coeffs), bs)):
        c2 = min(c1 + bs, len(stream_coeffs))
        stream_coeffs_batch = torch.zeros((c2-c1, len(stream_coeffs)), device=thetas.device)
        diag_idx = torch.arange(c1, c2, device=thetas.device)
        stream_coeffs_batch[diag_idx-c1, diag_idx] = 1.0
        fourier_bases = torch.cat([torch.cos(ms * thetas[..., None]),
                                    torch.sin(ms * thetas[..., None])], dim=-1) # ... (M 2)
        chebyshev_bases = chebyshev_polynomial_t(zs[..., None], ks) # ... K
        combined_bases = chebyshev_bases[..., None] * fourier_bases[..., None, :] # ... K, (M 2)
        combined_bases = combined_bases.reshape((*zs.shape, -1))
        tform_naive = einsum(combined_bases, stream_coeffs_batch, 
                             '... C, N C -> N ...')
        
    # Fast transform
    from mr_recon.fourier import sigpy_nufft
    nft = sigpy_nufft((K_chebyshev, M_fourier,), width=4)
    nft.beta = nft.optimal_beta(torch_dev=thetas.device)
    def nufft2d(cs, thetas, alphas):
        trj = torch.stack([
            -K_chebyshev * alphas,
            -M_fourier * thetas,], dim=-1) / (2 * torch.pi) # ..., 2
        tform_fast = nft.forward(cs[None,], trj[None,])[0]
        tform_fast *= torch.exp(1j * thetas * (M_fourier // 2))
        tform_fast *= torch.exp(1j * alphas * (K_chebyshev // 2))
        tform_fast = tform_fast.real * (M_fourier * K_chebyshev) ** 0.5
        return tform_fast
    for c1 in tqdm(range(0, len(stream_coeffs), bs)):
        c2 = min(c1 + bs, len(stream_coeffs))
        stream_coeffs_batch = torch.zeros((c2-c1, len(stream_coeffs)), device=thetas.device)
        diag_idx = torch.arange(c1, c2, device=thetas.device)
        stream_coeffs_batch[diag_idx-c1, diag_idx] = 1.0
        stream_coeffs_rs = rearrange(stream_coeffs_batch, 'N (K two M) -> N K M two', K=K_chebyshev, M=M_fourier)
        cs = stream_coeffs_rs[..., 0] -1j * stream_coeffs_rs[..., 1] # K, M
        tform_fast = 0.5 * (nufft2d(cs, thetas, torch.acos(zs)) + \
                            nufft2d(cs, thetas, -torch.acos(zs)))
    
    quit()
    # Compare
    import matplotlib
    matplotlib.use('webAgg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,5))
    plt.plot(tform_naive.cpu(), label='Naive')
    plt.plot(tform_fast.cpu(), label='Fast')
    plt.legend()
    plt.show()
