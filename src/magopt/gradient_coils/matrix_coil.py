import torch
import matplotlib.pyplot as plt
import numpy as np

from typing import Optional
from einops import rearrange, einsum
from .gradient_coil import gradient_coil
from ..sim.elliptic_lookup import EllipELookup, EllipKLookup
from ..sim import parametric_wire
from ..sim.analytic import (
    calc_bfield_loop_jacobian, 
    calc_bfield_loop,
    calc_inductance_loop, 
    calc_mag_potential_loop,
    _transform_coordinates,
)

class matrix_coil(gradient_coil):
    
    def __init__(self,
                 radii: torch.Tensor,
                 centers: torch.Tensor,
                 thetas_phis: torch.Tensor):
        """
        Args
        ----
        radii : torch.Tensor
            shape (N,) representing the radii of the coil loops.
        centers : torch.Tensor
            shape (N, 3) representing the centers of the coil loops.
        thetas_phis : torch.Tensor
            shape (N, 2) representing the phi and theta angles of the coil loops.
        """
        self.radii = torch.nn.Parameter(radii, requires_grad=False)
        self.centers = torch.nn.Parameter(centers, requires_grad=False)
        self.thetas_phis = torch.nn.Parameter(thetas_phis, requires_grad=False)
        self.elip_e = EllipELookup().to(radii.device)
        self.elip_k = EllipKLookup().to(radii.device)
    
    def _get_normals(self) -> torch.Tensor:
        """
        Gets the normals of the coil loops.
        """
        phis = self.thetas_phis[:, 1]
        thetas = self.thetas_phis[:, 0]
        normals = torch.stack([torch.sin(phis) * torch.cos(thetas),
                               torch.sin(phis) * torch.sin(thetas),
                               torch.cos(phis)], dim=1) # N 3
        return normals
   
    def build_coil_constraints(self,
                               Pmax: float = float('inf'),
                               Lmax: float = float('inf'),
                               tmax: float = float('inf'),
                               res_per_m: float = 0.1) -> torch.Tensor:
        """
        Asserts that each loop coil has a power limit Pmax, inductance limit Lmax, and rise time limit tmax.
        
        If our coil design variable is the amp-turn value X, then we generate constraints such that 
        |X_n| <= constraint_n
        
        Args
        ----
        Pmax : float
            Maximum power limit in Watts.
        Lmax : float
            Maximum inductance limit in Henry.
        tmax : float
            Maximum rise time limit in seconds.
        res_per_m : float
            Resistance per meter of the wire
        
        Returns
        -------
        constraints : torch.Tensor
            shape (N,) representing the constraints for each loop coil.
        """
        
        # Compute base resistance 
        lengths = self.radii * 2 * torch.pi # TODO arb geometry
        res_per_turn = res_per_m * lengths
        
        # Compute base inductance
        ind_per_turn = calc_inductance_loop(self.radii) # TODO arb geometry
        
        # Compute upper bound 
        tightest_term = torch.min((Lmax / ind_per_turn) ** 0.5, tmax * res_per_turn / ind_per_turn)
        upper_bound_sq = (Pmax / res_per_turn) * (tightest_term - 1)
        constraints = upper_bound_sq ** 0.5
        
        return constraints
   
    def build_field_matrices(self,
                             crds_bfield: torch.Tensor,
                             crds_gfield: torch.Tensor,
                             crds_efield: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Builds linear mappings from fields to coil coefficients.
        
        Args
        ----
        crds_bfield : torch.Tensor
            shape (Nb, 3) representing the coordinates where magnetic field is evaluated.
        crds_gfield : torch.Tensor
            shape (Ng, 3) representing the coordinates where the gradient field is evaluated.
        crds_efield : torch.Tensor
            shape (Ne, 3) representing the coordinates where electric field is evaluated.
        
        Returns
        -------
        bfield_mat : torch.Tensor
            shape (Nb, Ncoeff, 3) mapping coil coefficients to magnetic fields Bx, By, Bz
        gfield_mat : torch.Tensor
            shape (Ng, Ncoeff, 3) mapping coil coefficients to gradient fields dBz/dx, dBz/dy, dBz/dz
        efield_mat : torch.Tensor
            shape (Ne, Ncoeff, 3) mapping coil coefficients to electric fields Ex, Ey, Ez
        """
        # Convert phis_thetas to normal vectors
        normals = self._get_normals()
        
        analytic = True
        # Parametric wire method
        if not analytic:
            
            # Gen base loop coordinates
            thetas = torch.linspace(0, 2 * torch.pi, 1000, 
                                    device=self.radii.device)
            xs = torch.cos(thetas)
            ys = torch.sin(thetas)
            zs = torch.zeros_like(thetas)
            crds_loop = torch.stack([xs, ys, zs], dim=-1)
            
            # Transform to loop to point in normal direction
            crds_loop_new = _transform_coordinates(self.radii[:, None, None] * crds_loop[None, :, :], 
                                                   self.centers[:, None, :], 
                                                   normals[:, None, :],
                                                   flip_order=True)[0]
            
            # Compute fields per loop 
            bfields = []
            gfields = []
            afields = []
            for i in range(len(self.radii)):
                
                # Create parametric wire
                pw = parametric_wire(wire_pts=crds_loop_new[i], verbose=False)
                
                # Magnetic field
                bfield = pw.calc_bfield(spatial_crds=crds_bfield)
                gfield = pw.calc_bfield_jacobian(spatial_crds=crds_gfield)[..., -1, :]
                afield = pw.calc_mag_potential(spatial_crds=crds_efield)
                
                bfields.append(bfield.T)
                gfields.append(gfield.T)
                afields.append(afield.T)
                
            bfield_mat = torch.stack(bfields, dim=-1).moveaxis(0, -1)
            gfield_mat = torch.stack(gfields, dim=-1).moveaxis(0, -1)
            afield_mat = torch.stack(afields, dim=-1).moveaxis(0, -1)
            
        # Analytical method
        else:
            # Magnetic fields
            bfield_mat = calc_bfield_loop(spatial_crds=crds_bfield[None, :, :], 
                                          R=self.radii[:, None], 
                                          center=self.centers[:, None, :], 
                                          normal=normals[:, None, :],
                                          ellipe=self.elip_e,
                                          ellipk=self.elip_k)
            bfield_mat = rearrange(bfield_mat, 'Ncoeff Nb three -> Nb Ncoeff three ')
            
            # Gradient fields
            gfield_mat = calc_bfield_loop_jacobian(spatial_crds=crds_gfield[None, :, :], 
                                                   R=self.radii[:, None], 
                                                   center=self.centers[:, None, :], 
                                                   normal=normals[:, None, :],
                                                   ellipe=self.elip_e,
                                                   ellipk=self.elip_k)[..., -1, :]
            gfield_mat = rearrange(gfield_mat, 'Ncoeff Nb three -> Nb Ncoeff three')
            
            # Magnetic potential fields
            afield_mat = calc_mag_potential_loop(spatial_crds=crds_efield[None, :, :], 
                                                R=self.radii[:, None], 
                                                center=self.centers[:, None, :], 
                                                normal=normals[:, None, :],
                                                ellipe=self.elip_e,
                                                ellipk=self.elip_k)
            afield_mat = rearrange(afield_mat, 'Ncoeff Ne three -> Ne Ncoeff three')

        
        # Efield is the negative of the magnetic potential
        return bfield_mat, gfield_mat, -afield_mat
        
    def build_magnetic_energy_matrix(self) -> torch.Tensor:
        """
        Builds the magnetic energy matrix.
        """
        inductances = calc_inductance_loop(self.radii)
        return torch.diag(inductances)

    def evaluate_fields(self,
                        coeffs: torch.Tensor,
                        crds_bfield: torch.Tensor,
                        crds_gfield: torch.Tensor,
                        crds_efield: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates the magnetic and gradient fields at the specified coordinates given the matrix coil coefficients.
        
        Args
        ----
        stream_coeffs : torch.Tensor
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
        # Flatten 
        crds_bfield_flt = crds_bfield.reshape((-1, 3))
        crds_gfield_flt = crds_gfield.reshape((-1, 3))
        crds_efield_flt = crds_efield.reshape((-1, 3))
        
        # Eval matrices
        bfield_mat, gfield_mat, efield_mat = self.build_field_matrices(crds_bfield_flt, crds_gfield_flt, crds_efield_flt)
        gfield = einsum(gfield_mat, coeffs, 'N C d, C -> N d')
        bfield = einsum(bfield_mat, coeffs, 'N C d, C -> N d')
        efield = einsum(efield_mat, coeffs, 'N C d, C -> N d')
        
        # Reshape to original shape
        gfield = gfield.reshape((*crds_gfield.shape[:-1], gfield.shape[-1]))
        bfield = bfield.reshape((*crds_bfield.shape[:-1], bfield.shape[-1]))
        efield = efield.reshape((*crds_efield.shape[:-1], efield.shape[-1]))
        
        return bfield, gfield, efield

    @staticmethod
    def gen_pts_cylinder(Nrad, Nz):
        thetas = torch.arange(Nrad) / Nrad * 2 * torch.pi
        zs = torch.linspace(-0.5, 0.5, Nz)
        thetas, zs = torch.meshgrid(thetas, zs, indexing='ij')
        pts = torch.stack((torch.cos(thetas), torch.sin(thetas), zs), dim=-1)
        pts = pts.reshape(-1, 3)
        pts -= pts.mean(dim=0, keepdim=True)
        return pts

    @staticmethod
    def gen_pts_sphere(N):
        pts = []
        N_count = 0
        a = 4 * torch.pi / N
        d = a ** 0.5
        M_theta = round(torch.pi / d)
        d_theta = torch.pi / M_theta
        d_phi = a / d_theta
        while N_count < N:
            for m in range(M_theta):
                theta = torch.pi * (m + 0.5) / M_theta
                M_phi = round(2 * torch.pi * np.sin(theta) / d_phi)
                for n in range(M_phi):
                    phi = 2 * torch.pi * n / M_phi
                    pts.append([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
                    N_count += 1
        pts = torch.from_numpy(np.array(pts)).type(torch.float32)
        return pts
    
    @staticmethod
    def fibonacci_spherical_cap(n_points: int,
                                phi_max: float,
                                radius: float = 1.0) -> torch.Tensor:
        """
        Generate approximately equally spaced points on a spherical cap using
        a Fibonacci / golden-angle construction.

        Parameters
        ----------
        n_points : int
            Number of points to generate.
        phi_max : float
            Maximum polar angle in radians.
            Convention:
                phi = 0   -> +z axis
                phi = pi  -> -z axis
        radius : float
            Sphere radius.

        Returns
        -------
        pts : torch.Tensor, shape [n_points, 3]
            Cartesian coordinates of sampled points.
        """
        # Indices centered in each equal-area band
        i = torch.arange(n_points)

        # Uniform-in-area sampling on the spherical cap:
        # cos(phi) should be uniform on [cos(phi_max), 1]
        cos_phi_max = torch.cos(torch.tensor(phi_max))
        u = 1.0 - (i + 0.5) / n_points * (1.0 - cos_phi_max)
        phi = torch.arccos(u)

        # Golden-angle azimuths
        golden_ratio = (1.0 + 5.0 ** 0.5) / 2.0
        golden_angle = 2.0 * torch.pi / golden_ratio
        theta = torch.remainder(i * golden_angle, 2.0 * torch.pi)

        sin_phi = torch.sin(phi)
        x = radius * sin_phi * torch.cos(theta)
        y = radius * sin_phi * torch.sin(theta)
        z = radius * torch.cos(phi)

        pts = torch.stack([x, y, z], dim=-1)
        return pts

    def show_design(self,
                    coeffs: Optional[torch.Tensor] = None) -> None:
        """
        Visualizes the coil design.
        
        Args
        ----
        coeffs : Optional[torch.Tensor]
            shape (Ncoeff,) representing the coil coefficients.
            If None, the current design is shown with unity currents.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure showing the coil design.
        ax : matplotlib.axes._subplots.Axes3DSubplot
            The 3D axis showing the coil design.
        """
        
        # Make 3D plot object
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Gen base loop coordinates
        thetas = torch.linspace(0, 2 * torch.pi, 100, 
                                device=self.radii.device)
        xs = torch.cos(thetas)
        ys = torch.sin(thetas)
        zs = torch.zeros_like(thetas)
        crds_loop = torch.stack([xs, ys, zs], dim=-1)
        normals = self._get_normals()
        
        # Colors from coeffs
        vals = (coeffs / coeffs.abs().max()) ** 3
        vals = vals.cpu()
    
        # Plot each loop in 3D with varying colors based on current
        for i in range(len(self.radii)):
            # Per loop quantities
            r = self.radii[i]
            c = self.centers[i]
            n = normals[i]
            
            # Transform to ring to point in normal direction
            crds_loop_new = _transform_coordinates(r * crds_loop, 
                                                   c[None, :], 
                                                   n[None, :],
                                                   flip_order=True)[0].cpu() * 1e2
            
            # Use RDBu_R colormap for current values
            # -1 --> blue, 0 ---> white, 1 --> red
            colors = plt.get_cmap('berlin')((vals[i] + 1) / 2) 
            # colors = 'black'
            
            # Plot
            ax.plot(crds_loop_new[..., 0], 
                    crds_loop_new[..., 1], 
                    crds_loop_new[..., 2], 
                    color=colors,
                    # color='black',
                    )
            
        ax.axis('equal')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')
        
        # Show loop coefficients
        fig = plt.figure()
        axl = fig.add_subplot(111)
        axl.plot(coeffs.cpu().flip(dims=[0]))
        axl.set_title('Loop Coefficients')
        axl.set_ylabel('Current (A-turns)')
            
        return fig, ax, axl