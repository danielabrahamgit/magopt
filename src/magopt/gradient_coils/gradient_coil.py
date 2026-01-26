import torch

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
                             crds_bfield: torch.Tensor,
                             crds_efield: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        raise NotImplementedError("Subclass must implement build_field_matrices")
    
    def show_design(self) -> None:
        """
        Visualizes the coil design.
        """
        raise NotImplementedError("Subclass must implement show_design")
