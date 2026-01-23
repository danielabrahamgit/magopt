import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from einops import einsum
from matplotlib import cm
from typing import Optional

from body_model import (
    uv_to_xyz, 
    uv_to_normals, 
    dxyz_duv
)

class body_charge_model:
    
    """
    Models the charge accumulation on tissue surfaces due to time-varying magnetic fields. 
    Method described in:
    Roemer et al., MRM 2021 on Electric field calculation and peripheral nerve stimulation (PNS) prediction for head and body gradient coils
    
    
    Efield Theory
    -------------
    The total E-field is given by \\
    Ecoil(r) = A(r) x \\
    Echarge(r) = B(r) c \\
    Etotal(r) = -Ecoil(r) + Echarge(r)  \\
    Etotal(r) = -A(r) x + B(r) c 
    
    Charge conservation at the surface gives us \\
    n(r) . Etotal(r) = 0 \\ 
    meaning that  \\
    c = M^+ M^T Y x \\
    where \\
    M(r) = n(r) . B(r) for r on the surface \\
    Y(r) = n(r) . A(r) for r on the surface 
    
    So the total efield is  \\
    Etotal(r) = -A(r) x + B(r) M^+ M^T Y x \\
    Etotal(r) = -A(r) x + P(r) Y x \\
    Etotal(r) = (-A(r) + P(r) Y) x \\
    where P = M^+ M^T Y
    
    Charge density
    --------------
    q_{m,m}(u, v) = a_{m,n} * U_m(u) * V_n(v)
    
    Parameterization
    ----------------
    u : azimuthal angle [0, 2pi]
    v : longitudinal direction [0, 1]
    
    Basis functions
    ---------------
    U_m (u) : Fourier modes in azimuthal direction
    V_n (v) : Linear 'hat' functions in longitudinal direction
    """
    
    def __init__(self,
                 ulin: torch.Tensor,
                 vlin: torch.Tensor,
                 M_fourier_modes: int = 50,
                 N_hat_modes: int = 10):
        """
        Args
        ----
        us : torch.Tensor
            1D tensor of u values shape (num_us,)
        vs : torch.Tensor
            1D tensor of v values shape (num_vs,)
        M_fourier_modes : int
            Number of Fourier modes in azimuthal direction
        N_hat_modes : int
            Number of hat functions in longitudinal direction
        torch_dev : torch.device
            Torch device to use (e.g. 'cpu' or 'cuda:0')
        """
        self.M = M_fourier_modes
        self.N = N_hat_modes
        self.torch_dev = ulin.device
        
        # Generate surface points, normals, and areas for future use
        self.num_us = len(ulin)
        self.num_vs = len(vlin)
        surface_quanities = self._gen_surface_pts(ulin, vlin)
        self.uv_crds  = surface_quanities[0]
        self.xyz_crds = surface_quanities[1]
        self.normals  = surface_quanities[2]
        self.areas    = surface_quanities[3]
        self.P = None # Placeholder for P matrix
        
    def calc_efield_charge_matrix(self,
                                  crds_efield: torch.Tensor,
                                  num_us: int = 500,
                                  num_vs: int = 500,
                                  urange: Optional[tuple] = None,
                                  vrange: Optional[tuple] = None,
                                  surface_batch_size: Optional[int] = None,
                                  verbose: bool = True) -> torch.Tensor:
        """
        Builds the E-field matrix that maps the surface charge coefficients to the E-field at the specified coordinates.
        
        Args
        ----
        crds_efield : torch.Tensor
            Coordinates where the E-field is evaluated with shape (Ne, 3)
        num_us : int
            Number of points in the azimuthal (u) direction
        num_vs : int
            Number of points in the longitudinal (v) direction
        urange : tuple
            Range of u values (min, max)
        vrange : tuple
            Range of v values (min, max)
        surface_batch_size : int
            Number of surface points to process in each batch
        verbose : bool
            Whether to show progress bars
            
        Returns
        -------
        torch.Tensor
            E-field matrix with shape (Ne, ncoeff, 3) in V/m
        """
        # Defaults and constants
        if urange is None:
            urange = (0, 2 * torch.pi)
        if vrange is None:
            vrange = (0, 1)
        if surface_batch_size is None:
            surface_batch_size = (num_us * num_vs) // 10  # Process 10% of surface at a time
        eps0 = 8.8541878188e-12  # Vacuum permittivity
        torch_dev = crds_efield.device
        
        # Generate (u, v) grid
        us = torch.linspace(urange[0], urange[1], num_us, device=torch_dev)
        vs = torch.linspace(vrange[0], vrange[1], num_vs, device=torch_dev)
        du = us[1] - us[0]
        dv = vs[1] - vs[0]
        us, vs = torch.meshgrid(us, vs, indexing='ij')
        us = us.reshape(-1)
        vs = vs.reshape(-1)
        S = len(us) # Total number of surface points
        
        # Generate surface coordinates and normals
        crds_surf = uv_to_xyz(us, vs)
        normals = uv_to_normals(us, vs)
        normals_normalized = normals / torch.linalg.norm(normals, axis=-1, keepdim=True)
        tang_surf = dxyz_duv(us, vs)
        areas = torch.cross(tang_surf[:, 0, :], tang_surf[:, 1, :], dim=-1).norm(dim=-1) * du * dv
        
        # Move surface just a bit inwards to avoid singularities
        delta = 1e-4
        crds_surf = crds_surf - delta * normals_normalized

        # E-field matrix placeholder
        ncoeff = 2 * self.M * self.N
        Emat_charge = torch.zeros((*crds_efield.shape[:-1], ncoeff, 3), dtype=torch.float32, device=torch_dev)
        
        # Heavy batching, this is one time so that's okay.
        for n in tqdm(range(self.N), 'Building Surface E-field Matrix',  disable=not verbose):
            for m in range(self.M):
                for s1 in range(0, S, surface_batch_size):
                    s2 = min(s1 + surface_batch_size, S)
                
                    # Build bases
                    Um_sin, Um_cos = self._ubases(us[s1:s2], m) # S
                    Vn = self._vbases(vs[s1:s2], n) # S
                    
                    # Distance kernel
                    diff = crds_efield[:, None] - crds_surf[None, s1:s2] # Ne S 3
                    denom = 1 / torch.linalg.norm(diff, axis=-1, keepdim=True) ** 3 # Ne S 1
                    
                    # Integrate
                    integrand = (diff * denom * (Vn * Um_cos * areas[s1:s2])[None, :, None]).sum(dim=1) # Ne 3
                    Emat_charge[:, 2 * (m * self.N + n), :] +=  integrand * 1 / (4 * torch.pi * eps0) # Cosine term
                    integrand = (diff * denom * (Vn * Um_sin * areas[s1:s2])[None, :, None]).sum(dim=1) # Ne 3
                    Emat_charge[:, 2 * (m * self.N + n) + 1, :] +=  integrand * 1 / (4 * torch.pi * eps0) # Sine term

        return Emat_charge
    
    def _calc_P_matrix(self,
                       Efield_charge_mat_surf: torch.Tensor,
                       Efield_coil_mat_surf: torch.Tensor,
                       lam: float = 1e-12) -> torch.Tensor:
        """
        Calculates the projection matrix P used to enforce charge conservation on the surface.
        
        Args
        ----
        Efield_charge_mat_surf : torch.Tensor
            Charge E-field matrix over the surface with shape (Ne, C, 3)
        Efield_coil_mat_surf : torch.Tensor
            Coil E-field matrix over the surface with shape (Ne, X, 3)
        lam : float
            Regularization parameter for pseudo-inverse
        
        Returns
        -------
        P : torch.Tensor
            Projection matrix with shape (C, X)
        """
        
        # Build M matrix
        M = einsum(Efield_charge_mat_surf, self.normals, 'Ne C d, Ne d -> Ne C') * self.areas[:, None] # Ne C
        
        # Build Y matrix
        Y = einsum(Efield_coil_mat_surf, self.normals, 'Ne X d, Ne d -> Ne X') * self.areas[:, None] # Ne X

        # Linear solve
        MTM = M.T @ M
        MTY = M.T @ Y
        I = torch.eye(MTM.shape[0], device=MTM.device)
        P = torch.linalg.solve(MTM + lam * I, MTY)
        # P = lin_solve(MTM, MTY, lamda=lam, solver='solve')
        
        return P
    
    def build_efield_matrix(self,
                            Efield_charge_mat: torch.Tensor,
                            Efield_coil_mat: torch.Tensor,
                            Efield_charge_mat_surf: Optional[torch.Tensor] = None,
                            Efield_coil_mat_surf: Optional[torch.Tensor] = None,
                            l2_reg: float = 1e-9) -> torch.Tensor:
        """
        Solves for the charge coefficients given the E-field matrix and the time derivative of the vector potential.
        
        Args
        ----
        Efield_charge_mat : torch.Tensor
            Charge E-field matrix with shape (Ne, C, 3)
        Efield_coil_mat : torch.Tensor
            Maps from gradient coefficients to time derivative of the vector potential with shape (Ne, X, 3)
        Efield_charge_mat_surf : torch.Tensor
            Charge E-field matrix over the surface with shape (Ne_surf, C, 3). 
            If None, uses Efield_charge_mat for P matrix calculation
        Efield_coil_mat_surf : torch.Tensor
            Coil E-field matrix over the surface with shape (Ne_surf, X, 3). 
            If None, uses Efield_coil_mat for P matrix calculation
        l2_reg : float
            Regularization parameter for pseudo-inverse in P matrix calculation
        
        Returns
        -------
        Efield_mat : torch.Tensor
            Total E-field matrix with shape (Ne, ncoeff_grad, 3)
        """
        # Consts
        Ne_surf = self.uv_crds.shape[0]

        # Calculate P matrix
        if Efield_charge_mat_surf is None:
            Efield_charge_mat_surf = Efield_charge_mat
            assert Ne_surf == Efield_charge_mat_surf.shape[0]
        if Efield_coil_mat_surf is None:
            Efield_coil_mat_surf = Efield_coil_mat
            assert Ne_surf == Efield_coil_mat_surf.shape[0]
        P = self._calc_P_matrix(Efield_charge_mat_surf, Efield_coil_mat_surf, l2_reg) # C X

        # Build total E-field matrix
        Efield_mat = -Efield_coil_mat + einsum(Efield_charge_mat, P, 'Ne C d, C X -> Ne X d')

        return Efield_mat

    def show_surface(self,
                     fields: Optional[torch.Tensor] = None,
                     alpha: float = 1.0,
                     colorbar_label: str = '',
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None,
                     ax: Optional[plt.Axes] = None,
                     fig: Optional[plt.Figure] = None) -> tuple[plt.Figure, plt.Axes]:
        """
        Plots the surface with the norm of the fields as color.
        
        Args
        ----
        fields : torch.Tensor
            The fields to plot with same shape as self.xyz_crds
        alpha : float
            Transparency of the surface
        colorbar_label : str
            Label for the colorbar
        vmin : float
            Minimum value for color scaling
        vmax : float
            Maximum value for color scaling
        ax : plt.Axes
            Matplotlib Axes to plot on. If None, a new figure and axes are created
        fig : plt.Figure
            Matplotlib Figure to plot on. If None, a new figure and axes are created
            
        Returns
        -------
        fig : plt.Figure
            The figure showing the surface.
        ax : plt.Axes
            The axis showing the surface.
        """
        
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
        
        # If not ax provided, create one
        if ax is None:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, projection='3d')
            
        # Reshape to regular grid
        xyz_crds_grd = self.xyz_crds.reshape(self.num_us, self.num_vs, 3)
        if fields is not None:
            fields = fields.reshape(self.num_us, self.num_vs, 3)
        
        # Get surface crds in cm
        xs = xyz_crds_grd[:, :, 0].cpu() * 1e2 # cm
        ys = xyz_crds_grd[:, :, 1].cpu() * 1e2 # cm
        zs = xyz_crds_grd[:, :, 2].cpu() * 1e2 # cm
        
        # Plot surface
        if fields is None:
            ax.plot_surface(xs, ys, zs, 
                            color='navajowhite',
                            alpha=alpha,
                            rcount=self.num_us,
                            ccount=self.num_vs,
                            shade=True, 
                            linewidth=0,
                            edgecolor='none',)
        else:
            # Use field norm for color
            vals = fields.norm(dim=-1).cpu()
            if vmin is None:
                vmin = 0
            if vmax is None:
                vmax = vals.abs().max()
            norm = plt.Normalize(vmin, vmax)
            colormap = cm.jet
            colors = colormap(norm(vals))
            sm = cm.ScalarMappable(norm=norm, cmap=colormap)
            
            # Plot surface, where color is norm of fields
            ax.plot_surface(xs, ys, zs, 
                            facecolors=colors,
                            alpha=alpha,
                            rcount=self.num_us,
                            ccount=self.num_vs,
                            shade=True, 
                            linewidth=0,
                            edgecolor='none',)
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(colorbar_label)
            
        # set equal aspect
        set_axes_equal(ax)
        
        return fig, ax

    def _vbases(self,
                v: torch.Tensor,
                n: torch.Tensor) -> torch.Tensor:
        """
        Hat functions in longitudinal direction
        """
        N = self.N - 1
        mask = (v - n / N).abs() <= 1 / N
        v_bases = 1 - (N * v - n).abs()
        return v_bases * mask.float()
    
    def _ubases(self,
                u: torch.Tensor,
                m: torch.Tensor,) -> torch.Tensor:
        """
        Fourier modes in azimuthal direction
        """
        u_cos = torch.cos(m * u)
        u_sin = torch.sin(m * u)
        u_bases = torch.stack([u_cos, u_sin], dim=0)
        return u_bases
           
    def _gen_surface_pts(self,
                         ulin: torch.Tensor,
                         vlin: torch.Tensor) -> tuple[torch.Tensor, 
                                                      torch.Tensor, 
                                                      torch.Tensor, 
                                                      torch.Tensor]:
        """
        Generates (u, v) on a surface grid for computing E-field due to the coil and surface charges.
        
        Args
        ----
        ulin : torch.Tensor
            1D tensor of u values shape (num_us,)
        vlin : torch.Tensor
            1D tensor of v values shape (num_vs,)
            
        Returns
        -------
        uv_crds : torch.Tensor
            (u, v) coordinates with shape (num_us * num_vs, 2)
        xyz_crds : torch.Tensor
            Surface coordinates with shape (num_us * num_vs, 3)
        normals : torch.Tensor
            Surface normals with shape (num_us * num_vs, 3)
        areas : torch.Tensor
            Surface area elements with shape (num_us * num_vs,)
        """
        # Generate (u, v) grid
        dus = ulin.diff()
        dus = torch.cat([dus, dus[-1:]], dim=0)
        dvs = vlin.diff()
        dvs = torch.cat([dvs, dvs[-1:]], dim=0)
        us, vs = torch.meshgrid(ulin, vlin, indexing='ij')
        dus, dvs = torch.meshgrid(dus, dvs, indexing='ij')
        du = dus.reshape(-1)
        dv = dvs.reshape(-1)
        us = us.reshape(-1)
        vs = vs.reshape(-1)
        uv_crds = torch.stack([us, vs], dim=-1)
        duv_crds = torch.stack([du, dv], dim=-1)
        
        # Generate surface coordinates and normals
        xyz_crds = uv_to_xyz(us, vs) * 1e-3 # m
        tang_surf = dxyz_duv(us, vs) * 1e-3 # m
        normals = torch.cross(tang_surf[:, 0, :], tang_surf[:, 1, :], dim=-1)
        areas = torch.linalg.norm(normals, axis=-1) * duv_crds.prod(dim=-1)
        normals = normals / areas[:, None]
        
        return uv_crds, xyz_crds, normals, areas
