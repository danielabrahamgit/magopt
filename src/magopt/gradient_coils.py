import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib import cm
from matplotlib import animation
from typing import Optional
from einops import rearrange, einsum
from torch.special import chebyshev_polynomial_t, chebyshev_polynomial_u

from .bspline import BSpline1D
from .sim.elip import EllipELookup, EllipKLookup
from .sim.analytic import (
    calc_bfield_loop_jacobian, 
    calc_bfield_loop, 
    calc_inductance_matrix, 
    calc_mag_potential_loop
)

class gradient_coil:
    
    def __init__(self):
        return None
    
    def build_magnetic_energy_matrix(self) -> torch.Tensor:
        """
        Magnetic energy is evaluated as coeffs^T M coeffs.
        
        Returns
        -------
        torch.Tensor
            The magnetic energy matrix with shape (Ncoeff, Ncoeff).
        """
        raise NotImplementedError("Subclass must implement build_magnetic_energy_matrix")
    
    def build_field_matrices(self,
                             crds_gfield: torch.Tensor,
                             crds_bfield: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Builds linear mappings from fields to coil coefficients.
        
        Args
        ----
        crds_gfield : torch.Tensor
            shape (Ng, 3) representing the coordinates where the gradient field is evaluated.
        crds_bfield : torch.Tensor
            shape (Nb, 3) representing the coordinates where magnetic field is evaluated.
        
        Returns
        -------
        gfield_mat : torch.Tensor
            shape (3, Ng, Ncoeff) mapping coil coefficients to gradient fields dBz/dx, dBz/dy, dBz/dz
        bfield_mat : torch.Tensor
            shape (3, Nb, Ncoeff) mapping coil coefficients to magnetic fields Bx, By, Bz
        """
        raise NotImplementedError("Subclass must implement build_field_matrices")
    
    def show_design(self) -> None:
        """
        Visualizes the coil design.
        """
        raise NotImplementedError("Subclass must implement show_design")
    
class circular_z_gradient(gradient_coil):
    
    def __init__(self,
                 zs: torch.Tensor,
                 zs_spline: torch.Tensor,
                 rs_spline: torch.Tensor,
                 lamda_spline: float = 1e-2,
                 M_fourier_modes: Optional[int] = 10,
                 K_maxwell: Optional[int] = None,
                 max_fourier_cycles: int = 3,):
        """
        Args
        ----
        zs_spline : torch.Tensor
            shape (K,) representing the z positions of the spline interpolation points.
        rs_spline : torch.Tensor
            shape (K,) representing the radius positions of the spline interpolation points.
        lamda_spline : float, optional
            Regularization parameter for the spline fitting.
        zs : torch.Tensor
            shape (N,) representing the z positions of the coil loops.
        M_fourier_modes : int, optional
            Number of Fourier modes to use for representing the coil current distribution.
        max_fourier_cycles : int, optional
            Maximum number of cycles in the Fourier representation.
        """
        # Consts
        self.torch_dev = zs.device
        self.N = len(zs)
        
        # Assert uniform spacing
        assert torch.allclose(zs[1:] - zs[:-1], zs[1] - zs[0]), "zs must be uniformly spaced"
        assert torch.allclose(zs_spline[1:] - zs_spline[:-1], zs_spline[1] - zs_spline[0]), "zs_spline must be uniformly spaced"
        assert len(zs_spline) == len(rs_spline), "zs_spline and rs_spline must have the same length"
        
        # Offset coil in z
        self.zmin = torch.nn.Parameter(zs.min().clone(),
                                       requires_grad=False)
        self.zmax = torch.nn.Parameter(zs.max().clone(),
                                       requires_grad=False)
        self.zofs = torch.nn.Parameter(torch.tensor(0.0, device=self.torch_dev),
                                       requires_grad=False)
        
        # Create spline object for radius interpolation
        self.spline = BSpline1D(n_coeff=len(zs_spline), xmin=0, xmax=1, 
                                boundary="clamp", dtype=zs_spline.dtype, device=zs_spline.device)
        
        # Fit spline coefficients
        eta_spline = (zs_spline - zs_spline.min().item()) / (zs_spline.max().item() - zs_spline.min().item())
        self.spline.fit_lstsq(eta_spline, rs_spline, lam=lamda_spline)
        
        # No current mapping
        if M_fourier_modes is None and K_maxwell is None:
            self.Imat = torch.eye(self.N, device=self.torch_dev)
        # Fourier current mapping
        elif M_fourier_modes is not None:
            ns = torch.arange(self.N, device=self.torch_dev) # indexes Amp-turns output
            ks = torch.linspace(0, max_fourier_cycles, M_fourier_modes//2, device=self.torch_dev) # indexes coeffs input
            mat = torch.exp(-2j * torch.pi * ns[:, None] * ks[None, :] / self.N) # [N, M/2]
            self.Imat = torch.cat([mat.real, mat.imag], dim=1) / self.N # [N, M]
        elif K_maxwell is not None:
            # self.Imat = torch.eye(self.N, device=self.torch_dev)
            # ns = torch.arange(self.N, device=self.torch_dev)
            # idx_eye = self.N//3
            # idxs = torch.argwhere((ns - idx_eye).abs() >= K_maxwell//2)[:, 0]
            # self.Imat = self.Imat[:, idxs]
            
            sigma = 0.04
            # centers = zs.clone()
            centers = torch.linspace(self.zmin, self.zmax, 10, device=self.torch_dev)
            zs_flip = zs.flip(dims=[0])
            z_eye = -0.025
            eye_width = 0.05
            win = 1.0 * ((zs_flip - z_eye).abs() >= (eye_width / 2))[:, None]
            gaussians = torch.exp(-0.5 * ((zs_flip[:, None] - centers[None, :]) / sigma) ** 2) * win
            self.Imat = gaussians / gaussians.sum(dim=0, keepdim=True)
    
            
            # import matplotlib.pyplot as plt
            # Imat = self.Imat.cpu().numpy()
            # zs = zs.cpu().numpy()
            # for k in range(Imat.shape[1]):
            #     plt.plot(zs, Imat[:, k] + k * 0.1)
            # plt.show()
            # quit()
            
            
        # Need to make these for fast elliptic integral lookup
        self.elip_e = EllipELookup().to(self.torch_dev)
        self.elip_k = EllipKLookup().to(self.torch_dev)
    
    def get_coil_zs(self) -> torch.Tensor:
        return torch.linspace(self.zmin, self.zmax, self.N, device=self.torch_dev) + self.zofs
    
    def interp_rs(self,
                  zs: torch.Tensor) -> torch.Tensor:
        """
        Interpolates the coil radii to the provided z positions
        
        Args
        ----
        zs : torch.Tensor
            shape (N,) representing the z positions to interpolate to.
        
        Returns
        -------
        torch.Tensor
            The interpolated coil radii at the specified z positions with shape (N,).
        """
        etas = (zs - self.zmin - self.zofs) / (self.zmax - self.zmin)
        etas = torch.clamp(etas, 0.0, 1.0)
        return self.spline(etas)
    
    def build_magnetic_energy_matrix(self) -> torch.Tensor:
        zs = self.get_coil_zs()
        rs = self.interp_rs(zs)
        L = calc_inductance_matrix(rs, zs, ellipe=self.elip_e, ellipk=self.elip_k) * 0.5 # factor of 1/2 in energy expression
        return self.Imat.T @ L @ self.Imat
        
    def build_field_matrices(self,
                             crds_bfield: torch.Tensor,
                             crds_gfield: torch.Tensor,
                             crds_efield: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Uses analytic expressions for the field from circular loops to compute the field matrices.
        
        Args
        ----
        crds_bfield : torch.Tensor
            shape (Nb, 3) representing the coordinates where magnetic field is evaluated.   
        crds_gfield : torch.Tensor
            shape (Ng, 3) representing the coordinates where the gradient field is evaluated.
        crds_efield : torch.Tensor
            shape (Ne, 3) representing the coordinates where the electric field is evaluated.
            
        Returns
        -------
        bfield_mat : torch.Tensor
            shape (Nb, Ncoeff, 3) mapping coil coefficients to magnetic fields Bx, By, Bz
        gfield_mat : torch.Tensor
            shape (Ng, Ncoeff, 3) mapping coil coefficients to gradient fields dBz/dx, dBz/dy, dBz/dz
        efield_mat : torch.Tensor
            shape (Ne, Ncoeff, 3) mapping coil coefficients to electric fields Ex, Ey, Ez
        """
        # Get coil geometry
        zs = self.get_coil_zs()
        rs = self.interp_rs(zs)
        
        # Compute gradient fields
        gfields_mat = calc_bfield_loop_jacobian(crds_gfield[:, None], rs[None,], zs[None,], self.elip_e, self.elip_k)[..., -1, :]
        gfields_mat = rearrange(gfields_mat, 'Nb N d -> d Nb N') @ self.Imat
        
        # Compute magnetic fields
        bfields_mat = calc_bfield_loop(crds_bfield[:, None], rs[None,], zs[None,], self.elip_e, self.elip_k)
        bfields_mat = rearrange(bfields_mat, 'Nb N d -> d Nb N') @ self.Imat
        
        # Compute magnetic potential
        afields_mat = calc_mag_potential_loop(crds_efield[:, None], rs[None,], zs[None,], self.elip_e, self.elip_k)
        afields_mat = rearrange(afields_mat, 'Ne N d -> d Ne N') @ self.Imat

        return bfields_mat.moveaxis(0, -1), gfields_mat.moveaxis(0, -1), afields_mat.moveaxis(0, -1)

    def evaluate_fields(self,
                        coeffs: torch.Tensor,
                        crds_bfield: torch.Tensor,
                        crds_gfield: torch.Tensor,
                        crds_efield: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        crds_gfield_flt = crds_gfield.reshape((-1, 3))
        crds_bfield_flt = crds_bfield.reshape((-1, 3))
        crds_efield_flt = crds_efield.reshape((-1, 3))
        bfield, gfield, efield = self.build_field_matrices(crds_bfield_flt, crds_gfield_flt, crds_efield_flt)
        bfield = einsum(bfield, coeffs, 'Nb C d, C -> Nb d')
        gfield = einsum(gfield, coeffs, 'Ng C d, C -> Ng d')
        efield = einsum(efield, coeffs, 'Ne C d, C -> Ne d')
        bfield = bfield.reshape((*crds_bfield.shape[:-1], bfield.shape[-1]))
        gfield = gfield.reshape((*crds_gfield.shape[:-1], gfield.shape[-1]))
        efield = efield.reshape((*crds_efield.shape[:-1], efield.shape[-1]))
        return bfield, gfield, efield

    def show_design(self,
                    coeffs: torch.Tensor,
                    alpha: float = 1.0,
                    num_theta: int = 100,
                    vmin: float = None,
                    vmax: float = None) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
        """
        Show the coil design.

        Args
        ----
        coeffs : torch.Tensor
            shape (Ncoeff,) representing the coil coefficients.
        alpha : float, optional
            Transparency of the coil surface.
        num_theta : int, optional
            Number of theta points to use for surface integration.
        vmin : float, optional
            Minimum value for color scaling. If None, uses the min current value.
        vmax : float, optional
            Maximum value for color scaling. If None, uses the max current value.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure showing the coil design.
        ax : matplotlib.axes._subplots.Axes3DSubplot
            The 3D axis showing the coil design.
        axl : matplotlib.axes._subplots.AxesSubplot
            The axis showing the coil loop coefficients.
        """
        
        # Gen ring coordinates
        thetas = torch.linspace(0, 2 * torch.pi, num_theta, device=self.torch_dev)
        zs = self.get_coil_zs()
        thetas, zs = torch.meshgrid(thetas, zs, indexing='ij')
        rs = self.interp_rs(zs) * 1e2 # in cm
        zs = zs * 1e2 # in cm
        xs = rs * torch.cos(thetas)
        ys = rs * torch.sin(thetas)
        
        # Current values for coloring
        loop_coeffs = self.Imat @ coeffs
        vals = loop_coeffs[None, :].repeat(num_theta, 1)
        
        # Show coil surface
        fig = plt.figure(figsize=(14,7))
        ax = fig.add_subplot(111, projection='3d')
        if vmin is None:
            vmin = -vals.abs().max().item()
        if vmax is None:
            vmax = vals.abs().max().item()
        norm = plt.Normalize(vmin, vmax)
        colormap = cm.bwr
        colors = colormap(norm(vals.cpu().numpy()))
        sm = cm.ScalarMappable(norm=norm, cmap=colormap)
        ax.plot_surface(xs.cpu(), ys.cpu(), zs.cpu(), 
                        facecolors=colors,
                        alpha=alpha,
                        rcount=zs.shape[0],
                        ccount=zs.shape[1],
                        shade=True,
                        edgecolor='none',
                        linewidth=0)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Current (A-turns)')

        # Show loop coefficients
        fig = plt.figure()
        axl = fig.add_subplot(111)
        axl.plot(1e2 * self.get_coil_zs().cpu().flip(dims=[0]), loop_coeffs.cpu().flip(dims=[0]))
        axl.set_title('Loop Coefficients')
        axl.set_xlabel('Z-Position (cm)')
        axl.set_ylabel('Current (A-turns)')
        
        return fig, ax, axl

    def animate_thetas(self,
                       coeffs: list[torch.Tensor],
                       spline_coeffs: list[torch.Tensor],
                       zmins: list[float],
                       zmaxs: list[float],
                       num_theta: int = 100,    
                       body_surf: Optional[torch.Tensor] = None,) -> None:
        
        def set_axes_equal(ax):
            """Set 3D plot axes to equal scale so that spheres look like spheres."""
            x_limits = ax.get_xlim3d()
            y_limits = ax.get_ylim3d()
            z_limits = ax.get_zlim3d()

            x_range = abs(x_limits[1] - x_limits[0])
            x_middle = np.mean(x_limits)
            y_range = abs(y_limits[1] - y_limits[0])
            y_middle = np.mean(y_limits)
            z_range = abs(z_limits[1] - z_limits[0])
            z_middle = np.mean(z_limits)

            # The plot bounding box is a cube with side = max_range
            max_range = max([x_range, y_range, z_range])

            ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
            ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
            ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])
            ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
                
        # Consts
        Imat = self.Imat.cpu()
        if body_surf is not None:
            alpha = 0.5
        else:
            alpha = 1.0
        fig = plt.figure(figsize=(14,7))
        ax = fig.add_subplot(111, projection='3d')
        vals_last = Imat @ coeffs[-1]
        norm = plt.Normalize(-vals_last.abs().max(), vals_last.abs().max())
        colormap = cm.bwr
        sm = cm.ScalarMappable(norm=norm, cmap=colormap)
            
        # Show body surface
        if body_surf is not None:
            xyz_cm = body_surf * 1e2 # in cm
            surf_body = ax.plot_surface(xyz_cm[..., 0].cpu(), xyz_cm[..., 1].cpu(), xyz_cm[..., 2].cpu(),
                            color='navajowhite', alpha=1.0, shade=True, 
                            linewidth=0,
                            edgecolor='none',)
        else:
            surf_body = None
            
        def gen_surf_mats_coil(idx):
            # Gen ring coordinates
            zs = torch.linspace(zmins[idx], zmaxs[idx], self.N)
            thetas = torch.linspace(0, 2 * torch.pi, num_theta)
            thetas, zs = torch.meshgrid(thetas, zs, indexing='ij')
            etas = (zs - zmins[idx]) / (zmaxs[idx] - zmins[idx])
            etas = torch.clamp(etas, 0.0, 1.0)
            self.spline.coeff.data = spline_coeffs[idx]
            rs = self.spline(etas) * 1e2 # in cm
            zs = zs * 1e2 # in cm
            xs = rs * torch.cos(thetas)
            ys = rs * torch.sin(thetas)
        
            # Current values for coloring
            loop_coeffs = Imat @ coeffs[idx]
            vals = loop_coeffs[None, :].repeat(num_theta, 1)
            
            return xs, ys, zs, vals
        
        # First frame
        xs, ys, zs, vals = gen_surf_mats_coil(0)
        colors = colormap(norm(vals))
        surf = ax.plot_surface(xs, ys, zs, 
                        facecolors=colors,
                        alpha=alpha,
                        rcount=zs.shape[0],
                        ccount=zs.shape[1],
                        shade=True,
                        edgecolor='none',
                        linewidth=0)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Current (A-turns)')
        plt.axis('equal')
        ax.set_box_aspect([1, 1, 1])
        set_axes_equal(ax)
        
        # Frame update function
        def update(frame):
            nonlocal surf
            # frame = min(frame * 10, len(coeffs)-1)
            surf.remove()
            xs, ys, zs, vals = gen_surf_mats_coil(frame)
            colors = colormap(norm(vals))
            surf = ax.plot_surface(xs, ys, zs, 
                            facecolors=colors,
                            alpha=alpha,
                            rcount=zs.shape[0],
                            ccount=zs.shape[1],
                            shade=True,
                            edgecolor='none',
                            linewidth=0)
            ax.set_box_aspect([1, 1, 1])
            return surf
        
        ani = animation.FuncAnimation(fig, update, frames=len(coeffs), interval=200, blit=False)
        ani.save('plz.gif', writer='pillow', fps=30, bitrate=1800)
        plt.close(fig)
    
    def coil_winding_pattern(self,
                             coeffs: torch.Tensor,
                             N_per_loop: int = 100) -> torch.Tensor:
        """
        Generates a coil winding pattern based on the current distribution.
        
        Args
        ----
        coeffs : torch.Tensor
            shape (Ncoeff,) representing the coil coefficients.
        N_per_loop : int
            Number of points per loop in the winding pattern.
            
        Returns
        -------
        torch.Tensor
            The coil winding pattern with shape (npts, 3).
        """
        
        # Get z-coordinates and radii
        zs = self.get_coil_zs()
        rs = self.interp_rs(zs)
        
        # Theta as a function of z
        nturns = self.Imat @ coeffs
        thetas = 2 * torch.pi * torch.cumsum(nturns, dim=0)
        thetas = torch.cat([thetas[:1]*0, thetas], dim=0)
        rs = torch.cat([rs[:1], rs], dim=0)
        dz = zs[1] - zs[0]
        zs = torch.cat([zs[:1] - dz, zs], dim=0)
        N = len(thetas)
        
        # Fit splines to theta(z), r(z)
        lamda_spline = 1e-3
        N_spline = max(N, 4)
        rspline = BSpline1D(N_spline, zs.min(), zs.max(), device=coeffs.device)
        rspline.fit_lstsq(zs, rs, lam=lamda_spline)
        thetaspline = BSpline1D(N_spline, zs.min(), zs.max(), device=coeffs.device)
        thetaspline.fit_lstsq(zs, thetas, lam=lamda_spline)
        
        # Fine z grid
        num_pts = round(nturns.abs().max().round().item() * N_per_loop * N)
        zs_fine = torch.linspace(zs.min(), zs.max(), num_pts, device=coeffs.device)
        rs_fine = rspline(zs_fine)
        thetas_fine = thetaspline(zs_fine)
        
        # plt.figure()
        # plt.plot(zs_fine.cpu(), thetas_fine.cpu())
        # plt.scatter(zs.cpu(), thetas.cpu(), color='r')
        # plt.show()
        # quit()
        
        # Convert to parametric form
        xs = rs_fine * torch.cos(thetas_fine)
        ys = rs_fine * torch.sin(thetas_fine)
        zs = zs_fine
        
        return torch.stack([xs, ys, zs], dim=-1)
         
    def coil_winding_pattern_old(self,
                                coeffs: torch.Tensor,
                                N_per_loop: int = 500) -> torch.Tensor:
        """
        Generates a coil winding pattern based on the current distribution.
        
        Args
        ----
        coeffs : torch.Tensor
            shape (Ncoeff,) representing the coil coefficients.
        N_per_loop : int
            Number of points per loop in the winding pattern.
            
        Returns
        -------
        torch.Tensor
            The coil winding pattern with shape (N, 3).
        """
        
        # Get z-coordinates and radii
        zs = self.get_coil_zs()
        rs = self.interp_rs(zs)
        
        # Number of turns per segment
        nturns = self.Imat @ coeffs
        # nturns = (nturns[1:] + nturns[:-1]) / 2
        dthetas = 2 * torch.pi * nturns
        dthetas = torch.cat([dthetas, dthetas[-1:]], dim=0)
        thetas = torch.cumsum(dthetas, dim=0)
        
        # Variable number of points per segment
        num_pts = torch.maximum(N_per_loop * (nturns * 0 + 1), nturns * N_per_loop).ceil().int()
        
        # Initialize empty tensor for wire coordinates
        wire_crds = torch.empty((0, 3), device=coeffs.device)
        
        # Keep track of these
        dzprev = zs[1] - zs[0]
        drprev = rs[1] - rs[0]
        dthetaprev = dthetas[0]
        
        # Second order interpolation for smooth winding
        def second_order_interp(t, z0, z1, dz0):
            return (z1 - z0 - dz0) * t**2 + dz0 * t + z0
        for k in range(len(zs) - 1):
            
            # Grab stuff for this segment
            M = num_pts[k]
            dtheta = dthetas[k]
            dz = zs[k+1] - zs[k]
            dr = rs[k+1] - rs[k]
            t = torch.linspace(0, 1, M, device=coeffs.device)[:-1]
            
            # Calculate theta, z, and radius at each point
            theta_t = second_order_interp(t, thetas[k], thetas[k+1], 2 * dtheta - dthetaprev)
            z_t = second_order_interp(t, zs[k], zs[k+1], 2 * dz - dzprev)
            r_t = second_order_interp(t, rs[k], rs[k+1], 2 * dr - drprev)

            # Create points in cylindrical coordinates
            pts = torch.stack([
                r_t * torch.cos(theta_t),
                r_t * torch.sin(theta_t),
                z_t
            ], dim=-1)
            wire_crds = torch.cat([wire_crds, pts], dim=0)
            
            # Update for next iteration
            dthetaprev = 2 * dtheta - dthetaprev
            dzprev = 2 * dz - dzprev
            drprev = 2 * dr - drprev
            
        return wire_crds
        
class elliptical_frustum_stream(gradient_coil):
    
    def __init__(self,
                 zs_spline: torch.Tensor,
                 as_spline: torch.Tensor,
                 bs_spline: Optional[torch.Tensor] = None,
                 lamda_spline: float = 1e-2,
                 K_cheby_modes: int = 10,
                 M_fourier_modes: int = 15,
                 num_theta: int = 200,
                 num_zs: int = 200,):
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
        K_cheby_modes : int, optional
            Number of Chebyshev modes to use for representing the coil current distribution in the z direction.
        M_fourier_modes : int, optional
            Number of Fourier modes to use for representing the coil current distribution in the theta direction.
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
        self.M_fourier_modes = M_fourier_modes
        self.K_cheby_modes = K_cheby_modes
        
        # Indices for Chebyshev and Fourier modes
        self.ks = torch.arange(self.K_cheby_modes, device=self.torch_dev)
        self.ms = torch.arange(self.M_fourier_modes, device=self.torch_dev)

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
        tuple[torch.Tensor, torch.Tensor]
            Bases for the derivatives of the stream function w.r.t. theta and z
            shapes (..., Ncoeff)
        """

        # Gradient of stream function w.r.t. theta
        etas = 2 * (zs - self.zmin) / (self.zmax - self.zmin) - 1 # in [-1, +1]
        fourier_bases = torch.cat([-self.ms * torch.sin(self.ms * thetas[..., None]),
                                    self.ms * torch.cos(self.ms * thetas[..., None])], dim=-1) # ... (M 2)
        chebyshev_bases = self.ks * chebyshev_polynomial_t(etas[..., None], self.ks)
        combined_bases = chebyshev_bases[..., None] * fourier_bases[..., None, :] # ... K, (M 2)
        combined_bases_theta = combined_bases.reshape((*zs.shape, -1))

        # Gradient of stream function w.r.t. z
        fourier_bases = torch.cat([torch.cos(self.ms * thetas[..., None]),
                                   torch.sin(self.ms * thetas[..., None])], dim=-1) # ... (M 2)
        chebyshev_bases = self.ks * chebyshev_polynomial_u(etas[..., None], self.ks - 1) * (2 / (self.zmax - self.zmin))
        combined_bases = chebyshev_bases[..., None] * fourier_bases[..., None, :] # ... K, (M 2)
        combined_bases_zed = combined_bases.reshape((*zs.shape, -1))

        return combined_bases_theta, combined_bases_zed
    
    def _current_density_dS_bases(self,
                                  thetas: torch.Tensor,
                                  zs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the current density for each basis function and the dS factor for surface integration
        
        Args
        ----
        thetas : torch.Tensor
            Angular positions on the surface (...,)
        zs : torch.Tensor
            z positions on the surface (...,)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The current density basis functions Js (..., Ncoeff, 3) and dS factor (...) for surface integration
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
        return Js, dS_factor
    
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
            Coefficients for the Chebyshev and Fourier expansions with shape (K * M * 2,)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The derivatives of the stream function w.r.t. theta and z
        """

        # Gradient of stream function w.r.t. theta
        etas = 2 * (zs - self.zmin) / (self.zmax - self.zmin) - 1 # in [-1, +1]
        fourier_bases = torch.cat([-self.ms * torch.sin(self.ms * thetas[..., None]),
                                    self.ms * torch.cos(self.ms * thetas[..., None])], dim=-1) # ... (M 2)
        chebyshev_bases = self.ks * chebyshev_polynomial_t(etas[..., None], self.ks)
        combined_bases = chebyshev_bases[..., None] * fourier_bases[..., None, :] # ... K, (M 2)
        combined_bases = combined_bases.reshape((*zs.shape, -1))
        dphi_dtheta = combined_bases @ stream_coeffs

        # Gradient of stream function w.r.t. z
        fourier_bases = torch.cat([torch.cos(self.ms * thetas[..., None]),
                                   torch.sin(self.ms * thetas[..., None])], dim=-1) # ... (M 2)
        chebyshev_bases = self.ks * chebyshev_polynomial_u(etas[..., None], self.ks - 1) * (2 / (self.zmax - self.zmin))
        combined_bases = chebyshev_bases[..., None] * fourier_bases[..., None, :] # ... K, (M 2)
        combined_bases = combined_bases.reshape((*zs.shape, -1))
        dphi_dz = combined_bases @ stream_coeffs

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
            Coefficients for the Chebyshev and Fourier expansions with shape (K * M * 2,)
            
        Returns
        -------
        torch.Tensor
            The stream function (...) evaluated at the given theta and z positions 
        """
        # Build Bases
        etas = 2 * (zs - self.zmin) / (self.zmax - self.zmin) - 1 # in [-1, +1]
        fourier_bases = torch.cat([torch.cos(self.ms * thetas[..., None]),
                                   torch.sin(self.ms * thetas[..., None])], dim=-1) # ... (M 2)
        chebyshev_bases = chebyshev_polynomial_t(etas[..., None], self.ks) # ... K
        combined_bases = chebyshev_bases[..., None] * fourier_bases[..., None, :] # ... K, (M 2)
        combined_bases = combined_bases.reshape((*zs.shape, -1))
        
        # Evaluate
        return combined_bases @ stream_coeffs

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
            Coefficients for the Chebyshev and Fourier expansions with shape (K * M * 2,)
            
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
    
    def show_design(self,
                    stream_coeffs: torch.Tensor,
                    num_theta: int = 100,
                    num_z: int = 50,
                    body_surf: Optional[torch.Tensor] = None,) -> None:
        
        # Gen coordinates
        zs = torch.linspace(self.zmin, self.zmax, num_z, device=self.torch_dev)
        thetas = torch.linspace(0, 2 * torch.pi, num_theta, device=self.torch_dev)
        thetas, zs = torch.meshgrid(thetas, zs, indexing='ij') # (T, Z)
        
        # Cartesian
        crds = self._surface_positions(thetas, zs)
        xs, ys, zs = crds[..., 0], crds[..., 1], crds[..., 2]
        vals = self._stream_function(thetas, zs, stream_coeffs)
        
        # to CPU
        xs, ys, zs, vals = xs.cpu(), ys.cpu(), zs.cpu(), vals.cpu()
        
        # Plot
        if body_surf is not None:
            alpha = 0.5
        else:
            alpha = 1.0
        fig = plt.figure(figsize=(14,7))
        norm = plt.Normalize(vmin=-vals.abs().max(), vmax=vals.abs().max())
        colormap = cm.berlin
        colors = colormap(norm(vals))
        sm = cm.ScalarMappable(norm=norm, cmap=colormap)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xs, ys, zs,
                        facecolors=colors,
                        rcount=zs.shape[0],
                        ccount=zs.shape[1],
                        alpha=alpha,
                        shade=False,
                        linewidth=0,
                        edgecolor='none')
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
        thetas = torch.tensor([0.0, np.pi, np.pi/2, 3*np.pi/2], device=self.torch_dev)
        zs = torch.linspace(self.zmin, self.zmax, num_z, device=self.torch_dev)
        thetas, zs = torch.meshgrid(thetas, zs, indexing='ij') # (4, Z)
        stream = self._stream_function(thetas, zs, stream_coeffs)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        angles = ['+x', '-x', '+y', '-y']
        for i in range(4):
            ax.plot(zs[i].cpu(), stream[i, :].cpu(), label=angles[i])
        ax.set_xlabel('z (m)')
        ax.set_ylabel('Stream Function Value')
        ax.legend()
        ax.set_title('Stream Function Along Principal Axes')
        
    def build_current_boundary_matrix(self,
                                      num_theta: int = 100,
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
            Current boundary condition matrix A of shape (ncoeff, 2 * num_theta)
        """
        # Consts
        MK2 = self.M_fourier_modes * self.K_cheby_modes * 2
        
        # Gen theta points
        thetas = torch.linspace(0, 2 * torch.pi, num_theta, device=self.torch_dev)
        
        # Current matrix placeholder
        Jmat = torch.zeros((MK2, 2 * num_theta), dtype=torch.float32, device=self.torch_dev)
        
        # Get current density at boundaries in normal direction
        one_hot = torch.zeros(MK2, device=self.torch_dev)
        for c in tqdm(range(MK2), 'Building Field Matrix', disable=not verbose):
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
            
            Jmat[c, :num_theta] = Jn_min
            Jmat[c, num_theta:] = Jn_max
        
        return Jmat
    
    def build_field_matrices_fast(self,
                                  crds_bfield: torch.Tensor,
                                  crds_gfield: torch.Tensor,
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
        """
        
        # Consts
        batch_size = batch_size if batch_size is not None else Nsurf
        mu0_over_4pi = 1e-7 # in T m / A
        I = torch.eye(3, device=self.torch_dev, dtype=crds_bfield.dtype)
        gshape = crds_gfield.shape[:-1]
        bshape = crds_bfield.shape[:-1]
        crds_bfield = crds_bfield.reshape((-1, 3))
        crds_gfield = crds_gfield.reshape((-1, 3))
        
        # Generate surface points
        thetas = torch.linspace(0, 2 * torch.pi, self.num_theta, device=self.torch_dev)
        zs = torch.linspace(self.zmin, self.zmax, self.num_zs, device=self.torch_dev)
        thetas, zs = torch.meshgrid(thetas, zs, indexing='ij') # (T, Z)
        thetas = thetas.reshape(-1) # Nsurf
        zs = zs.reshape(-1) # Nsurf
        crds_surf = self._surface_positions(thetas, zs) # (Nsurf, 3)
        Nsurf = crds_surf.shape[0]

        # Get current density bases and dS factor
        Js, dS_factor = self._current_density_dS_bases(thetas, zs)

        # Placeholder for field matrices
        MK2 = self.M_fourier_modes * self.K_cheby_modes * 2
        bfields = torch.zeros((crds_bfield.shape[0], MK2, 3), dtype=crds_bfield.dtype, device=self.torch_dev) # Bx By Bz
        gfields = torch.zeros((crds_gfield.shape[0], MK2, 3), dtype=crds_bfield.dtype, device=self.torch_dev) # dBz/dx, dBz/dy, dBz/dz
        
        # Evaluate bio savart on current density
        for n1 in tqdm(range(0, Nsurf, batch_size), 'Building Field Matrices', disable=not verbose):
            n2 = min(n1 + batch_size, Nsurf)
            
            # --------- Magnetic field ---------
            diff_mgnt = crds_bfield[:, None, None, :] - crds_surf[None, n1:n2, None, :] # (N, Nsurf, 1, 3)
            numer_mgnt = torch.cross(Js[None, n1:n2], diff_mgnt, dim=-1) * dS_factor[None, n1:n2, None, None] # (N, Nsurf, Ncoeff, 3)
            denom_mgnt = diff_mgnt.norm(dim=-1, keepdim=True) ** 3 # (N, Nsurf, 1, 1)
            bfields += mu0_over_4pi * (numer_mgnt / denom_mgnt).sum(dim=1)
            
            # --------- Gradient field ---------
            diff_grdnt = crds_gfield[:, None, None, :] - crds_surf[None, n1:n2, None, :] # (N, Nsurf, 1, 3)
            nrm_sq = diff_grdnt.norm(dim=-1) ** 2 # (N, Nsurf, 1)
            numer_grdt = I * nrm_sq[..., None, None]  - 3 * diff_grdnt[..., :, None] * diff_grdnt[..., None, :] # (N, Nsurf, 1, 3, 3)
            numer_grdt = torch.cross(Js[None, n1:n2, :, :, None], numer_grdt, dim=-2)[..., -1, :] * dS_factor[None, n1:n2, None, None]
            denom_grdt = nrm_sq[..., None] ** (5/2)
            gfields += mu0_over_4pi * (numer_grdt / denom_grdt).sum(dim=1)

        # Reshape to original input shape
        bfields = bfields.reshape((*bshape, MK2, 3))
        gfields = gfields.reshape((*gshape, MK2, 3))
        return bfields, gfields

    def build_field_matrices(self,
                             crds_bfield: torch.Tensor,
                             crds_gfield: torch.Tensor,
                             verbose: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Builds the matrix that maps stream function coefficients to magnetic field at observation points
        
        Args
        ----
        crds_bfield : torch.Tensor
            shape (*bshape, 3) representing the coordinates where magnetic field is evaluated.   
        crds_gfield : torch.Tensor
            shape (*gshape, 3) representing the coordinates where the gradient field is evaluated.
        num_theta : int
            Number of theta points on the surface for integration
        num_z : int
            Number of z points on the surface for integration
        batch_size : Optional[int]
            Batch size for processing surface points in chunks to save memory
            if None, process all points at once
        
        Returns
        -------
        bfields : torch.Tensor
            Magnetic field matrix B (*bshape, ncoeff, 3)
        gfields : torch.Tensor
            Gradient field matrix G (*gshape, ncoeff, 3)
        """
        return self.build_field_matrices_fast(crds_bfield, crds_gfield, verbose=verbose)
        # Placeholder for field matrices
        MK2 = self.M_fourier_modes * self.K_cheby_modes * 2
        bfields = torch.zeros((*crds_bfield.shape[:-1], MK2, 3), dtype=crds_bfield.dtype, device=self.torch_dev) # Bx By Bz
        gfields = torch.zeros((*crds_gfield.shape[:-1], MK2, 3), dtype=crds_bfield.dtype, device=self.torch_dev) # dBz/dx, dBz/dy, dBz/dz
        
        # Evaluate fields for each one-hot coeff
        for c in tqdm(range(MK2), 'Building Field Matrix', disable=not verbose):
            one_hot = torch.zeros(MK2, device=self.torch_dev)
            one_hot[c] = 1.0
            bfields[..., c, :], gfields[..., c, :] = self.evaluate_fields(one_hot, crds_bfield, crds_gfield, verbose=False)
            
        return bfields, gfields

    def evaluate_fields(self,
                        stream_coeffs: torch.Tensor,
                        crds_bfield: torch.Tensor,
                        crds_gfield: torch.Tensor,
                        batch_size: Optional[int] = 2**3,
                        verbose: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the magnetic and gradient fields at the specified coordinates given the stream function coefficients.
        
        Args
        ----
        stream_coeffs : torch.Tensor
            Coefficients for the Chebyshev and Fourier expansions with shape (K * M * 2,)
        crds_bfield : torch.Tensor
            shape (*bshape, 3) representing the coordinates where magnetic field is evaluated.   
        crds_gfield : torch.Tensor
            shape (*gshape, 3) representing the coordinates where the gradient field is evaluated.
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
        """
        # Consts
        dtype = crds_bfield.dtype
        MK2 = self.M_fourier_modes * self.K_cheby_modes * 2
        mu0_over_4pi = 1e-7 # in T m / A
        I = torch.eye(3, device=self.torch_dev, dtype=dtype)
        gshape = crds_gfield.shape[:-1]
        bshape = crds_bfield.shape[:-1]
        
        # flatten input coordinates
        crds_bfield = crds_bfield.view((-1, 3))
        crds_gfield = crds_gfield.view((-1, 3))
        
        # Flattened surface theta, z, and positions
        theta_surf = torch.linspace(0, 2 * torch.pi, self.num_theta, device=self.torch_dev, dtype=dtype)
        z_surf = torch.linspace(self.zmin, self.zmax, self.num_zs, device=self.torch_dev, dtype=dtype)
        theta_surf, z_surf = torch.meshgrid(theta_surf, z_surf, indexing='ij') # (T, Z)
        theta_surf = theta_surf.reshape(-1) # Nsurf
        z_surf = z_surf.reshape(-1) # Nsurf
        crds_surf = self._surface_positions(theta_surf, z_surf) # (Nsurf, 3)
        Nsurf = crds_surf.shape[0]
        
        # Placeholder for fields
        bfields = torch.zeros((crds_bfield.shape[0], 3), dtype=dtype, device=self.torch_dev) # Bx By Bz
        gfields = torch.zeros((crds_gfield.shape[0], 3), dtype=dtype, device=self.torch_dev) # dBz/dx, dBz/dy, dBz/dz
        
        # Get current density
        Js, dS_factor = self._current_density_dS(theta_surf, z_surf, stream_coeffs) # (Nsurf, 3)
        
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
            
            # Gfield calc
            nrm_sq = diff.norm(dim=-1) ** 2
            numer = I * nrm_sq[..., None, None]  - 3 * diff[..., :, None] * diff[..., None, :] # (nbatch, Nsurf, 3, 3)
            numer = torch.cross(Js[None, n1:n2, :, None], numer, dim=-2)[..., -1, :] * dS_factor[None, n1:n2, None]
            denom = nrm_sq[..., None] ** (5/2)
            gfields += mu0_over_4pi * (numer / denom).sum(dim=1)

        return bfields.reshape(*bshape, 3), gfields.reshape(*gshape, 3)

    def build_magnetic_energy_matrix(self,
                                     batch_size: Optional[int] = 2**0,
                                     verbose: bool = True) -> torch.Tensor:
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
        MK2 = self.M_fourier_modes * self.K_cheby_modes * 2
        mu0_over_4pi = 1e-7
        
        # Gen surface theta, z, and positions
        theta_surf = torch.linspace(0, 2 * torch.pi, self.num_theta, device=self.torch_dev)
        z_surf = torch.linspace(self.zmin, self.zmax, self.num_zs, device=self.torch_dev)
        theta_surf, z_surf = torch.meshgrid(theta_surf, z_surf, indexing='ij') # (T, Z)
        theta_surf = theta_surf.reshape(-1) # Nsurf
        z_surf = z_surf.reshape(-1) # Nsurf
        crds_surf = self._surface_positions(theta_surf, z_surf) # (Nsurf, 3)
        Nsurf = crds_surf.shape[0]
        
        # Get current density bases and dS factor
        Js, dS_factor = self._current_density_dS_bases(theta_surf, z_surf) # (Nsurf, ncoeff, 3)
        
        # placeholder for energy matrix
        L = torch.zeros((MK2, MK2), dtype=torch.float32, device=self.torch_dev)
        
        # Loop over surface points
        for n1 in tqdm(range(0, Nsurf, batch_size), 'Building Energy Matrix', disable=not verbose):
            n2 = min(n1 + batch_size, Nsurf)
            
            # Magnetic energy integral
            eps = 1e-3 ** 2
            diff = crds_surf[:, None, :] - crds_surf[None, n1:n2, :] # (Nsurf, Nbatch, 3)
            inv_nrm = 1.0 / (diff.square().sum(dim=-1) + eps).sqrt() # (Nsurf, Nbatch)
            dot_prod = einsum(Js * dS_factor[:, None, None], 
                              Js[n1:n2] * dS_factor[n1:n2, None, None], 
                              'N Ci d, B Co d -> Ci Co N B') # very very slow
            integral = einsum(dot_prod, inv_nrm, 'Ci Co N B, N B -> Ci Co')
            L += 0.5 * mu0_over_4pi * integral
            
        return L
                    
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
