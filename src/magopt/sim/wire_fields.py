import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Optional

MU0 = 4e-7 * torch.pi # T*m/A
EPSILON_STABILITY = 1e-9 # Small value to avoid division by zero in calculations

class parametric_wire:
    """Arbitrary wire path defined by a set of points in 3D space."""
    
    def __init__(self,
                 wire_pts: torch.Tensor,
                 wire_seg_batch: Optional[int] = None,
                 verbose: Optional[bool] = True):
        """ 
        Args:
        -----
        wire_pts : torch.Tensor
            Wire spatial points with shape (N, 3) in units [m]
        wire_seg_batch : Optional[int]
            Batching over the N wire segments for calculations
        verbose : Optional[bool]
            If True, prints progress during calculations.
        """
        self.wire_pts = wire_pts
        self.verbose = verbose
        if wire_seg_batch is None:
            self.wire_seg_batch = wire_pts.shape[0]
        else:
            self.wire_seg_batch = wire_seg_batch
    
    def calc_resistance(self,
                        resistivity: float,
                        diameter: float) -> float:
        """ 
        Calculates the resistance of the wire using the formula:
        R = rho * L / A
        where rho is the resistivity
        
        Args:        
        -----
        resistivity : float
            Resistivity of the wire material [Ohm m]
        diameter : float
            Diameter of the wire [m]
        """
        # Return the resistance using the formula R = ρ * L / A
        length = self.wire_pts.diff(dim=0).norm(dim=-1).sum().item()
        area = torch.pi * (diameter / 2) ** 2
        return resistivity * length / area
    
    @staticmethod
    def phi_grover(segi_start, segi_end, segj_start, segj_end):
        
        r11 = (segi_start - segj_start).norm(dim=-1)
        r12 = (segi_start - segj_end).norm(dim=-1)
        r21 = (segi_end - segj_start).norm(dim=-1)
        r22 = (segi_end - segj_end).norm(dim=-1)
        Li = (segi_end - segi_start).norm(dim=-1)
        Lj = (segj_end - segj_start).norm(dim=-1)

        term1 =  r11 * ((r11 + r12 + Li + Lj) / (r11 + r21 + Li - Lj)).log()
        term2 =  r22 * ((r21 + r22 + Li + Lj) / (r12 + r22 - Li + Lj)).log()
        term3 = -r12 * ((r11 + r12 + Li + Lj) / (r12 + r22 - Li + Lj)).log()
        term4 = -r21 * ((r21 + r22 + Li + Lj) / (r11 + r21 + Li - Lj)).log()
        
        return (term1 + term2 + term3 + term4) 
    
    def calc_inductance(self,
                        radius_wire: Optional[float] = 0.5e-3) -> float:
        """
        Calculates the inductance of the wire using the formula:
        L = mu0 / 4pi * int_C1, int_C2 dx1 \cdot dx2 / |x1 - x2|
        
        Args:
        -----
        radius_wire : float
            radius_wire of the wire [m]. If not provided, defaults to 1/2 mm.
        
        Returns:
        --------
        inductance : float
            Inductance of the wire [H]
        """
        # Constants
        wire_pts = self.wire_pts
        N = self.wire_pts.shape[0]
        
        # Place holder for output inductance
        inductance = 0.0
        
        # Batch over wire segments
        for n1 in tqdm(range(0, N, self.wire_seg_batch), 
                       'Calculating inductance', 
                       disable=not self.verbose):
            n2 = min(n1 + self.wire_seg_batch, N-1)
            
            # Get ds segment vectors
            dx1_batch = wire_pts[n1+1:n2+1] - wire_pts[n1:n2] # B 3
            dx2 = wire_pts[1:] - wire_pts[:-1] # N-1 3
            
            # Get the positions of each segment
            x1_batch = (wire_pts[n1+1:n2+1] + wire_pts[n1:n2]) / 2 # B 3
            x2 = (wire_pts[1:] + wire_pts[:-1]) / 2 # N-1 3
            
            # Main terms for inductance calculation
            nrm = (x1_batch[:, None, :] - x2[None, :, :]).norm(dim=-1) # B N-1
            dot_prod = dx1_batch @ dx2.T  # B N-1
            
            # Remove diagonal terms
            idxs = torch.arange(n1, n2, device=wire_pts.device)
            dot_prod[idxs-n1, idxs] = 0.0
            nrm[idxs-n1, idxs] = 1.0 
            
            # Accumulate the off-diagonal terms
            inductance += MU0 * (dot_prod / nrm).sum() / (4 * torch.pi)
            
            # # Grover method
            # segj_start = wire_pts[None, :-1] # 1 N-1 3
            # segj_end = wire_pts[None, 1:] # 1 N-1 3
            # segi_start = wire_pts[n1:n2, None, :] # B 1 3
            # segi_end = wire_pts[n1+1:n2+1, None, :] # B 1 3
            # phis = self.phi_grover(segi_start, segi_end, segj_start, segj_end)
            
            # # Remove diagonal terms
            # idxs = torch.arange(n1, n2, device=wire_pts.device)
            # phis[idxs-n1, idxs] = 0.0
            
            # # Accumulate the off-diagonal terms
            # inductance += (MU0 * phis / (4 * torch.pi)).sum()
            
            # Accumulate diagonal terms
            lx1 = dx1_batch.norm(dim=-1)
            inductance += (MU0 * lx1 * ((2 * (lx1 + EPSILON_STABILITY) / radius_wire).log() - 1) / (2 * torch.pi)).sum()
        return inductance.item()
    
    def calc_mag_potential(self,
                           spatial_crds: torch.Tensor) -> torch.Tensor:
        """
        Calculates the magnetic vector potential produced by a unit current 
        at the given spatial coordinates using the biot-savart law.

        Args:
        -----
        spatial_crds : torch.Tensor
            Spatial coordinates with shape (..., 3)
            
        Returns:
        --------
        mag_potential : torch.Tensor
            Magnetic vector potential at the spatial coordinates with shape (...,)
        """
        # Constants
        wire_pts = self.wire_pts
        N = wire_pts.shape[0]
        crds = spatial_crds.reshape((-1, 3))
        
        # Place holder for output potential
        mag_potential = torch.zeros((crds.shape[0], 3), device=crds.device, dtype=crds.dtype)
        
        # Batch over wire segments
        for n1 in tqdm(range(0, N-1, self.wire_seg_batch), 
                       'Calculating Magnetic Potential', 
                       disable=not self.verbose):
            n2 = min(n1 + self.wire_seg_batch, N-1)
            
            # Get ds segment vectors
            ds_batch = wire_pts[n1+1:n2+1] - wire_pts[n1:n2] # B 3
            
            # Get the positions of each segment
            pts_batch = (wire_pts[n1+1:n2+1] + wire_pts[n1:n2]) / 2 # B 3
            
            # Calculate the norm of r vectors and unit vectors
            r_batch = crds[:, None, :] - pts_batch[None, :, :] # S B 3
            norm_batch = EPSILON_STABILITY + torch.linalg.norm(r_batch, dim=-1)[..., None] # S B 1
            
            # Bio savart copmutation
            bsl = ds_batch / norm_batch # S B 3
            mag_potential += MU0 * bsl.sum(dim=1) / (4 * torch.pi) # S 3
        
        return mag_potential.reshape(spatial_crds.shape[:-1] + (3,))
           
    def calc_bfield(self,
                    spatial_crds: torch.Tensor) -> torch.Tensor:
        """
        Calculates the magnetic field produced by a unit current 
        at the given spatial coordinates using the Biot-Savart law.
        
        Args:
        -----
        spatial_crds : torch.Tensor
            Spatial coordinates with shape (..., 3)
        eps : Optional[float]
            Small value to avoid division by zero in the Biot-Savart law.
        
        Returns:
        --------
        bfield : torch.Tensor
            Magnetic field at the spatial coordinates with shape (..., 3).
        """
        # Constants
        wire_pts = self.wire_pts
        N = wire_pts.shape[0]
        crds = spatial_crds.reshape((-1, 3))
        
        # Place holder for output magnetic field
        bfield = torch.zeros((crds.shape[0], 3), device=crds.device, dtype=crds.dtype)
        
        # Batch over wire segments
        for n1 in tqdm(range(0, N-1, self.wire_seg_batch), 
                       'Calculating B-field', 
                       disable=not self.verbose):
            n2 = min(n1 + self.wire_seg_batch, N-1)
            
            # Get ds segment vectors
            ds_batch = wire_pts[n1+1:n2+1] - wire_pts[n1:n2] # B 3
            
            # Get the positions of each segment
            pts_batch = (wire_pts[n1+1:n2+1] + wire_pts[n1:n2]) / 2 # B 3
            
            # Get r vectors from points to spatial coordinates
            r_batch = crds[:, None, :] - pts_batch[None, :, :] # S B 3
            
            # Calculate the norm of r vectors and unit vectors
            norm_batch = EPSILON_STABILITY + torch.linalg.norm(r_batch, dim=-1)[..., None] # S B 1
            r_hat_batch = r_batch / norm_batch # S B 3
            
            # Bio savart copmutation
            bsl = torch.linalg.cross(ds_batch[None, :, :], r_hat_batch) / (norm_batch ** 2) # S B 3
            bfield += MU0 * bsl.sum(dim=1) / (4 * torch.pi) # S 3
        
        return bfield.reshape(spatial_crds.shape[:-1] + (3,))
    
    def calc_bfield_jacobian(self,
                             spatial_crds: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Jacobian of the magnetic field produced by a unit current 
        at the given spatial coordinates using the Biot-Savart law.
        
        Args:
        -----
        spatial_crds : torch.Tensor
            Spatial coordinates with shape (..., 3)
        
        Returns:
        --------
        bfield_jacobian : torch.Tensor
            Jacobian of the magnetic field at the spatial coordinates with shape (..., 3, 3).
            The last two dimensions correspond to dB_i/dx_j.
        """
        # Constants
        wire_pts = self.wire_pts
        N = wire_pts.shape[0]
        crds = spatial_crds.reshape((-1, 3))
        
        # Place holder for output magnetic field Jacobian
        bfield_jacobian = torch.zeros((crds.shape[0], 3, 3), device=crds.device, dtype=crds.dtype)
        
        def skew_from_vec(v):
            """Return [v]_x for v (..., 3) -> (..., 3, 3)."""
            vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
            O = torch.zeros_like(vx)
            return torch.stack([
                torch.stack([ O, -vz,  vy], dim=-1),
                torch.stack([ vz,  O, -vx], dim=-1),
                torch.stack([-vy,  vx,  O], dim=-1),
            ], dim=-2)
        
        # Batch over wire segments
        for n1 in tqdm(range(0, N-1, self.wire_seg_batch), 
                       'Calculating B-field Jacobian', 
                       disable=not self.verbose):
            n2 = min(n1 + self.wire_seg_batch, N-1)
            
            # Get ds segment vectors
            ds_batch = wire_pts[n1+1:n2+1] - wire_pts[n1:n2] # B 3
            
            # Get the positions of each segment
            pts_batch = (wire_pts[n1+1:n2+1] + wire_pts[n1:n2]) / 2 # B 3
            
            # Get r vectors from points to spatial coordinates and norms
            r_batch = crds[:, None, :] - pts_batch[None, :, :] # S B 3
            norm_batch = EPSILON_STABILITY + torch.linalg.norm(r_batch, dim=-1)[..., None, None] # S B 1 1
            
            # Skew term
            skew_term = skew_from_vec(ds_batch[None, :, :]) / (norm_batch ** 3)
            
            # Cross term
            cross_term = -3 * torch.cross(ds_batch[None, :, :], r_batch, dim=-1)[..., None] * r_batch[..., None, :]
            cross_term /= (norm_batch ** 5)
            
            # Bio savart accumulation
            bfield_jacobian += MU0 * (skew_term + cross_term).sum(dim=1) / (4 * torch.pi) # S 3
        
        return bfield_jacobian.reshape(spatial_crds.shape[:-1] + (3, 3))

    def show_wire(self, fig=None, ax=None):
        """Displays the wire points."""

        wire_pts = self.wire_pts.cpu() * 1e2 # cm
        if fig is None or ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.plot(wire_pts[:, 0], wire_pts[:, 1], wire_pts[:, 2],
                alpha=.8, linewidth=0.5, color='red')
        ax.set_xlabel('X [cm]')
        ax.set_ylabel('Y [cm]')
        ax.set_zlabel('Z [cm]')
        plt.axis('equal')
        
        return fig, ax