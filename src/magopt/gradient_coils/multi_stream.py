import torch
import matplotlib.pyplot as plt

from typing import Optional
from .gradient_coil import gradient_coil

class multi_stream(gradient_coil):
    
    def __init__(self,
                 grad_coils: list[gradient_coil]):
        """
        Args
        ----
        grad_coils : list[gradient_coil]
            List of gradient coils to combine.
        """
        self.grad_coils = grad_coils
        self.Ngrad = len(self.grad_coils)
    
    def get_num_coeffs(self) -> int:
        return sum([grad_coil.get_num_coeffs() for grad_coil in self.grad_coils])
    
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
        bfields = None
        gfields = None
        efields = None
        coef_idx = 0
        for grad_coil in self.grad_coils:
            Ncoeffs_i = grad_coil.get_num_coeffs()
            coef_slc = slice(coef_idx, coef_idx + Ncoeffs_i)
            coeffs_i = coeffs[coef_slc]
            bfields_i, gfields_i, efields_i = grad_coil.evaluate_fields(coeffs_i, crds_bfield, crds_gfield, crds_efield, batch_size, verbose)
            if bfields is None:
                bfields = bfields_i
                gfields = gfields_i
                efields = efields_i
            else:
                bfields += bfields_i
                gfields += gfields_i
                efields += efields_i
            coef_idx += Ncoeffs_i
        return bfields, gfields, efields
    
    def build_magnetic_energy_matrix(self) -> torch.Tensor:
        """
        Magnetic energy is evaluated as coeffs^T M coeffs.
        
        Returns
        -------
        torch.Tensor
            The magnetic energy matrix with shape (Ncoeff, Ncoeff).
        """
        Ncoeff = self.get_num_coeffs()
        Ms = None
        coef_idx = 0
        for grad_coil in self.grad_coils:
            Ncoeff_i = grad_coil.get_num_coeffs()
            coef_slc = slice(coef_idx, coef_idx + Ncoeff_i)
            Mi = grad_coil.build_magnetic_energy_matrix()
            if Ms is None:
                Ms = torch.zeros((Ncoeff, Ncoeff), device=Mi.device)
            Ms[coef_slc, coef_slc] = Mi
            coef_idx += Ncoeff_i
        return Ms
    
    def build_field_matrices(self,
                             crds_bfield: torch.Tensor,
                             crds_gfield: torch.Tensor,
                             crds_efield: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        gfield_mat = []
        bfield_mat = []
        efield_mat = []
        for grad_coil in self.grad_coils:
            bfield_mat_i, gfield_mat_i, efield_mat_i = grad_coil.build_field_matrices(
                crds_bfield=crds_bfield, 
                crds_gfield=crds_gfield, 
                crds_efield=crds_efield
            )
            bfield_mat.append(bfield_mat_i)
            gfield_mat.append(gfield_mat_i)
            efield_mat.append(efield_mat_i)
        return (
            torch.cat(bfield_mat, dim=-2),
            torch.cat(gfield_mat, dim=-2),
            torch.cat(efield_mat, dim=-2),
        )
    
    def build_winding_tolerance_matrix(self) -> torch.Tensor:
        """
        Matrix relating the stream function coefficients to the winding tolerance.
        This lets us control the winding density
        
        ||winding_tol @ stream_coeffs||_2 <= upper_bound

            
        Returns
        -------
        winding_tol : torch.Tensor
            Winding tolerance matrix with shape (Nsurf, ncoeff, 3)
        """
        
        winding_tols = [grad_coil.build_winding_tolerance_matrix() for grad_coil in self.grad_coils]
        torch_dev = winding_tols[0].device
        Ncoeffs = self.get_num_coeffs()
        Nsurf = sum([winding_tol.shape[0] for winding_tol in winding_tols])
        winding_tol = torch.zeros(((Nsurf, Ncoeffs, 3)), device=torch_dev)
        surf_idx = 0
        coef_idx = 0
        for i in range(self.Ngrad):
            Nsurf_i = winding_tols[i].shape[0]
            Ncoeffs_i = winding_tols[i].shape[1]
            surf_slc = slice(surf_idx, surf_idx + Nsurf_i)
            coef_slc = slice(coef_idx, coef_idx + Ncoeffs_i)
            winding_tol[surf_slc, coef_slc, :] = winding_tols[i]
            surf_idx += Nsurf_i
            coef_idx += Ncoeffs_i
        return winding_tol
    
    def build_current_boundary_matrix(self) -> torch.Tensor:
        """
        Matrix relating the stream function coefficients to the current boundary conditions.
        We want zero current to flow off the ends of the surface.
        
        Returns
        -------
        torch.Tensor
            Current boundary condition matrix of shape (Nbound, Ncoeffs)
        """
        current_boundaries = [grad_coil.build_current_boundary_matrix() for grad_coil in self.grad_coils]
        torch_dev = current_boundaries[0].device
        Ncoeffs = self.get_num_coeffs()
        Nboundaries = sum([current_boundary.shape[0] for current_boundary in current_boundaries])
        current_boundary = torch.zeros(((Nboundaries, Ncoeffs)), device=torch_dev)
        coeff_idx = 0
        boundary_idx = 0
        for i in range(self.Ngrad):
            Nboundaries_i = current_boundaries[i].shape[0]
            Ncoeffs_i = current_boundaries[i].shape[1]
            coeff_slc = slice(coeff_idx, coeff_idx + Ncoeffs_i)
            boundary_slc = slice(boundary_idx, boundary_idx + Nboundaries_i)
            current_boundary[boundary_slc, coeff_slc] = current_boundaries[i]
            coeff_idx += Ncoeffs_i
            boundary_idx += Nboundaries_i
        return current_boundary
    
    def stream_to_contour(self,
                          coeffs: torch.Tensor,
                          num_u: int = 100,
                          num_v: int = 100,
                          dstream: Optional[float] = None,) -> tuple[list[torch.Tensor], float]:
        """
        Computes the contour paths of the stream function at given levels
        
        Args
        ----
        stream_coeffs : torch.Tensor
            Coefficients for the Chebyshev and Fourier expansions with shape (Ncoeffs,)
        num_u : int
            Number of u points on the surface
        num_v : int
            Number of v points on the surface
        dstream : float
            Level spacing for countours
            If none, uses 50 levels spaced evenly between the min and max of the stream function
            
        Returns
        -------
        list[torch.Tensor]
            List of contour paths, i^th entry of list has shape (Npoints_i, 3)
        """
        xyz_contours = []
        coef_idx = 0
        for grad_coil in self.grad_coils:
            Ncoeffs_i = grad_coil.get_num_coeffs()
            coef_slc = slice(coef_idx, coef_idx + Ncoeffs_i)
            coeffs_i = coeffs[coef_slc]
            xyz_contours_i, dstream_i = grad_coil.stream_to_contour(coeffs_i, num_u, num_v, dstream=dstream)
            if dstream is None:
                dstream = dstream_i
            else:
                assert dstream == dstream_i, "dstream must be the same for all gradient coils"
            xyz_contours += xyz_contours_i
            coef_idx += Ncoeffs_i
        return xyz_contours, dstream
    
    
    def show_design(self,
                    coeffs: torch.Tensor,
                    num_u: int = 50,
                    num_v: int = 50,
                    body_surf: Optional[torch.Tensor] = None,
                    show_1d: bool = False,
                    colorbar: bool = True,
                    fig: Optional[plt.Figure] = None,
                    ax: Optional[plt.Axes] = None) -> tuple[plt.Figure, plt.Axes]:
        coef_idx = 0
        for grad_coil in self.grad_coils:
            Ncoeffs_i = grad_coil.get_num_coeffs()
            coef_slc = slice(coef_idx, coef_idx + Ncoeffs_i)
            coeffs_i = coeffs[coef_slc]
            fig, ax = grad_coil.show_design(coeffs_i, num_u, num_v, body_surf, show_1d, colorbar, fig, ax)
            coef_idx += Ncoeffs_i
        return fig, ax
    
    def show_countour(self,
                      coeffs: torch.Tensor,
                      num_u: int = 50,
                      num_v: int = 50,
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
        coef_idx = 0
        for grad_coil in self.grad_coils:
            Ncoeffs_i = grad_coil.get_num_coeffs()
            coef_slc = slice(coef_idx, coef_idx + Ncoeffs_i)
            coeffs_i = coeffs[coef_slc]
            fig, ax = grad_coil.show_countour(coeffs_i, num_u, num_v, dstream, fig, ax)
            coef_idx += Ncoeffs_i
        return fig, ax