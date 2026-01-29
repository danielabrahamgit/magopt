import torch
import matplotlib.pyplot as plt

from typing import Optional
from einops import rearrange
from .gradient_coil import gradient_coil
from ..sim.elip import EllipELookup, EllipKLookup
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
                
            bfield_mat = torch.stack(bfields, dim=-1)
            gfield_mat = torch.stack(gfields, dim=-1)
            afield_mat = torch.stack(afields, dim=-1)
            
        # Analytical method
        else:
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
            colors = plt.get_cmap('RdBu_r')((vals[i] + 1) / 2) 
            # colors = 'black'
            
            # Plot
            ax.plot(crds_loop_new[..., 0], 
                    crds_loop_new[..., 1], 
                    crds_loop_new[..., 2], color=colors)
            
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