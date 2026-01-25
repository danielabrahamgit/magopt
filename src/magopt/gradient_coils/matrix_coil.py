import torch
import matplotlib.pyplot as plt

from typing import Optional
from einops import rearrange
from .gradient_coil import gradient_coil
from ..sim.elip import EllipELookup, EllipKLookup
from ..sim.analytic import (
    calc_bfield_loop_jacobian, 
    calc_bfield_loop,
    calc_inductance_loop, 
    calc_mag_potential_loop,
    _transform_coordinates
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
        self.radii = radii
        self.centers = centers
        self.thetas_phis = thetas_phis
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
   
    def build_field_matrices(self,
                             crds_gfield: torch.Tensor,
                             crds_bfield: torch.Tensor,
                             crds_efield: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Builds linear mappings from fields to coil coefficients.
        
        Args
        ----
        crds_gfield : torch.Tensor
            shape (Ng, 3) representing the coordinates where the gradient field is evaluated.
        crds_bfield : torch.Tensor
            shape (Nb, 3) representing the coordinates where magnetic field is evaluated.
        crds_efield : torch.Tensor
            shape (Ne, 3) representing the coordinates where electric field is evaluated.
        
        Returns
        -------
        gfield_mat : torch.Tensor
            shape (3, Ng, Ncoeff) mapping coil coefficients to gradient fields dBz/dx, dBz/dy, dBz/dz
        bfield_mat : torch.Tensor
            shape (3, Nb, Ncoeff) mapping coil coefficients to magnetic fields Bx, By, Bz
        efield_mat : torch.Tensor
            shape (3, Ne, Ncoeff) mapping coil coefficients to electric fields Ex, Ey, Ez
        """
        # Convert phis_thetas to normal vectors
        normals = self._get_normals()
        
        # Magnetic fields
        bfield_mat = calc_bfield_loop(spatial_crds=crds_bfield[None, :, :], 
                                      R=self.radii[:, None], 
                                      center=self.centers[:, None, :], 
                                      normal=normals[:, None, :],
                                      ellipe=self.elip_e,
                                      ellipk=self.elip_k)
        bfield_mat = rearrange(bfield_mat, 'Ncoeff Nb three -> three Nb Ncoeff')
        
        # Gradient fields
        gfield_mat = calc_bfield_loop_jacobian(spatial_crds=crds_gfield[None, :, :], 
                                               R=self.radii[:, None], 
                                               center=self.centers[:, None, :], 
                                               normal=normals[:, None, :],
                                               ellipe=self.elip_e,
                                               ellipk=self.elip_k)[..., -1, :]
        gfield_mat = rearrange(gfield_mat, 'Ncoeff Nb three -> three Nb Ncoeff')
        
        # Magnetic potential fields
        afield_mat = calc_mag_potential_loop(spatial_crds=crds_efield[None, :, :], 
                                             R=self.radii[:, None], 
                                             center=self.centers[:, None, :], 
                                             normal=normals[:, None, :],
                                             ellipe=self.elip_e,
                                             ellipk=self.elip_k)
        afield_mat = rearrange(afield_mat, 'Ncoeff Ne three -> three Ne Ncoeff')
        
        return gfield_mat, bfield_mat, afield_mat
        
    def build_magnetic_energy_matrix(self) -> torch.Tensor:
        """
        Builds the magnetic energy matrix.
        """
        inductances = calc_inductance_loop(self.radii)
        return torch.diag(inductances)

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
                                                   flip_order=True).cpu()
            
            # Plot
            ax.plot(crds_loop_new[..., 0], crds_loop_new[..., 1], crds_loop_new[..., 2], color='r')
            
        ax.axis('equal')
            
        return fig, ax