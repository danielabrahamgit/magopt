import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib import animation
from typing import Optional
from einops import rearrange, einsum

from .gradient_coil import gradient_coil
from ..bspline import BSpline1D
from ..sim.elip import EllipELookup, EllipKLookup
from ..sim.analytic import (
    calc_bfield_loop_jacobian, 
    calc_bfield_loop,
    calc_inductance_matrix, 
    calc_mag_potential_loop
)

class circular_z_coil(gradient_coil):
    
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
        gfield_mat : torch.Tensor
            shape (3, Ng, Ncoeff) mapping coil coefficients to gradient fields dBz/dx, dBz/dy, dBz/dz
        bfield_mat : torch.Tensor
            shape (3, Nb, Ncoeff) mapping coil coefficients to magnetic fields Bx, By, Bz
        efield_mat : torch.Tensor
            shape (3, Ne, Ncoeff) mapping coil coefficients to electric fields Ex, Ey, Ez
        """
        # Get coil geometry
        zs = self.get_coil_zs()
        rs = self.interp_rs(zs)
        
        # Compute gradient fields
        centers = torch.stack([zs*0, zs*0, zs], dim=-1)
        normals = torch.stack([zs*0, zs*0, zs*0 + 1], dim=-1)
        gfields_mat = calc_bfield_loop_jacobian(crds_gfield[:, None], rs[None,], centers[None,], normals[None,], self.elip_e, self.elip_k)[..., -1, :]
        gfields_mat = rearrange(gfields_mat, 'Nb N d -> d Nb N') @ self.Imat
        
        # Compute magnetic fields
        bfields_mat = calc_bfield_loop(crds_bfield[:, None], rs[None,], centers[None,], normals[None,], self.elip_e, self.elip_k)
        bfields_mat = rearrange(bfields_mat, 'Nb N d -> d Nb N') @ self.Imat
        
        # Compute magnetic potential
        afields_mat = calc_mag_potential_loop(crds_efield[:, None], rs[None,], centers[None,], normals[None,], self.elip_e, self.elip_k)
        afields_mat = rearrange(afields_mat, 'Ne N d -> d Ne N') @ self.Imat        

        return gfields_mat, bfields_mat, afields_mat

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
  