
import torch
from typing import Optional, Tuple

from ..filament.bfield_kernels import base_bfield, _gauss_legendre
from ...constants import MU0, EPSILON_STABILITY


class elliptical_bfield(base_bfield):
    """
    Analytic-parametric ellipse Biot-Savart kernels.

    Ellipse is parameterized in its local frame by:
        r_local(phi) = [ a cosphi, b sinphi, 0 ]
    with tangent:
        dr_local/dphi = [ -a sinphi, b cosphi, 0 ]

    A world-frame ellipse is obtained via a rigid transform:
        r_world(phi) = center + r_local(phi) @ R^T
        dl_world   = (dr_local/dphi @ R^T) dphi

    These "full_*_kernel_gauss" methods compute the *full loop integral*
    using Gauss-Legendre quadrature over phi in [0, 2pi], avoiding polygonal
    discretization.
    """

    @staticmethod
    def _ellipse_quadrature(a: torch.Tensor,
                            b: torch.Tensor,
                            R: torch.Tensor,
                            center: torch.Tensor,
                            n_pts: int,
                            dtype: torch.dtype,
                            device: torch.device
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
        -----
        a : torch.Tensor
            Semi-major axis.
        b : torch.Tensor
            Semi-minor axis.
        R : torch.Tensor
            Rotation matrix with shape [3, 3].
        center : torch.Tensor
            Loop center with shape [3].
        n_pts : int
            Number of quadrature points.
        dtype : torch.dtype
            Tensor dtype for outputs.
        device : torch.device
            Tensor device for outputs.

        Returns:
        --------
        tuple[torch.Tensor, torch.Tensor]
            r_t : [n, 3] world points on the ellipse
            dl  : [n, 3] world tangent vectors already multiplied by dphi weights,
                  i.e. dl_k ~ (dr/dphi)(phi_k) * (w_k * 2pi)
        """
        t, w = _gauss_legendre(n_pts)  # on [0,1]
        t = t.to(dtype=dtype, device=device)
        w = w.to(dtype=dtype, device=device)

        two_pi = torch.tensor(2.0 * torch.pi, dtype=dtype, device=device)
        phi = two_pi * t                     # [n]
        wphi = two_pi * w                    # [n]  (dphi weights)

        c = torch.cos(phi)
        s = torch.sin(phi)

        zero = torch.zeros_like(c)

        r_local = torch.stack([a * c, b * s, zero], dim=-1)           # [n,3]
        dl_dphi_local = torch.stack([-a * s, b * c, zero], dim=-1)    # [n,3]

        # Row-vector convention: x_world = x_local @ R^T + center
        r_t = r_local @ R.T + center[None, :]                         # [n,3]
        dl = (dl_dphi_local @ R.T) * wphi[:, None]                    # [n,3]

        return r_t, dl

    def full_potential_kernel_gauss(self,
                                    crds:   torch.Tensor,
                                    a:      float,
                                    b:      float,
                                    center: Optional[torch.Tensor] = None,
                                    R:      Optional[torch.Tensor] = None,
                                    n_pts:  int = 80,
                                    chunk:  int = 65536) -> torch.Tensor:
        """
        Magnetic vector potential of a unit-current elliptical loop:

            A(r) = (mu0 / 4pi) loop_int dl / |r - r'(phi)|

        Args:
            crds   : [..., 3] evaluation points [m]
            a, b   : semi-major, semi-minor axes [m]
            center : [3] translation (default 0)
            R      : [3,3] rotation matrix from local->world (default I)
            n_pts  : Gauss-Legendre points over phi in [0,2pi]
            chunk  : evaluation-point chunking to control memory

        Returns:
            A : [..., 3] [T*m/A]
        """
        if center is None:
            center = torch.zeros(3, dtype=crds.dtype, device=crds.device)
        if R is None:
            R = torch.eye(3, dtype=crds.dtype, device=crds.device)

        a_t = torch.tensor(a, dtype=crds.dtype, device=crds.device)
        b_t = torch.tensor(b, dtype=crds.dtype, device=crds.device)

        r_t, dl = self._ellipse_quadrature(a_t, b_t, R, center, n_pts, crds.dtype, crds.device)  # [n,3],[n,3]

        flat = crds.reshape(-1, 3)
        out = torch.zeros((flat.shape[0], 3), dtype=crds.dtype, device=crds.device)

        # Chunked over evaluation points (usually many more than quadrature nodes)
        for i in range(0, flat.shape[0], chunk):
            pts = flat[i:i+chunk]                                   # [S,3]
            diff = pts[:, None, :] - r_t[None, :, :]                # [S,n,3]
            Rn = diff.norm(dim=-1).clamp(min=EPSILON_STABILITY)     # [S,n]
            integrand = dl[None, :, :] / Rn[:, :, None]             # [S,n,3]
            out[i:i+chunk] = (MU0 / (4 * torch.pi)) * integrand.sum(dim=1)

        return out.reshape(crds.shape[:-1] + (3,))

    def full_bfield_kernel_gauss(self,
                                 crds:   torch.Tensor,
                                 a:      float,
                                 b:      float,
                                 center: Optional[torch.Tensor] = None,
                                 R:      Optional[torch.Tensor] = None,
                                 n_pts:  int = 120,
                                 chunk:  int = 65536) -> torch.Tensor:
        """
        Magnetic field of a unit-current elliptical loop:

            B(r) = (mu0 / 4pi) loop_int dl x (r - r') / |r - r'|^3

        Args:
            crds   : [..., 3] evaluation points [m]
            a, b   : semi-major, semi-minor axes [m]
            center : [3] translation (default 0)
            R      : [3,3] rotation matrix (default I)
            n_pts  : Gauss-Legendre points over phi in [0,2pi]
            chunk  : evaluation-point chunking to control memory

        Returns:
            B : [..., 3] [T/A]
        """
        if center is None:
            center = torch.zeros(3, dtype=crds.dtype, device=crds.device)
        if R is None:
            R = torch.eye(3, dtype=crds.dtype, device=crds.device)

        a_t = torch.tensor(a, dtype=crds.dtype, device=crds.device)
        b_t = torch.tensor(b, dtype=crds.dtype, device=crds.device)

        r_t, dl = self._ellipse_quadrature(a_t, b_t, R, center, n_pts, crds.dtype, crds.device)

        flat = crds.reshape(-1, 3)
        out = torch.zeros((flat.shape[0], 3), dtype=crds.dtype, device=crds.device)

        for i in range(0, flat.shape[0], chunk):
            pts = flat[i:i+chunk]                                   # [S,3]
            diff = pts[:, None, :] - r_t[None, :, :]                # [S,n,3]
            Rn = diff.norm(dim=-1).clamp(min=EPSILON_STABILITY)     # [S,n]
            dl_cross = torch.linalg.cross(dl[None, :, :].expand_as(diff), diff, dim=-1)  # [S,n,3]
            integrand = dl_cross / (Rn[:, :, None] ** 3)            # [S,n,3]
            out[i:i+chunk] = (MU0 / (4 * torch.pi)) * integrand.sum(dim=1)

        return out.reshape(crds.shape[:-1] + (3,))

    def full_jacobian_kernel_gauss(self,
                                   crds:   torch.Tensor,
                                   a:      float,
                                   b:      float,
                                   center: Optional[torch.Tensor] = None,
                                   R:      Optional[torch.Tensor] = None,
                                   n_pts:  int = 160,
                                   chunk:  int = 20000) -> torch.Tensor:
        """
        Jacobian of the magnetic field for a unit-current elliptical loop:

            J_ij(r) = dB_i/dx_j

        Using the analytic Biot-Savart Jacobian kernel integrated along the loop:

            dB/dr = (mu0/4pi) loop_int [ [dl]_x / |rho|^3  - 3 (dlxrho) outer rho / |rho|^5 ]

            rho = r - r'(phi)

        Args:
            crds   : [..., 3] evaluation points [m]
            a, b   : semi-major, semi-minor axes [m]
            center : [3] translation (default 0)
            R      : [3,3] rotation (default I)
            n_pts  : Gauss-Legendre points over phi in [0,2pi]
            chunk  : evaluation-point chunking (Jacobian is heavier)

        Returns:
            J : [..., 3, 3] [T/(A*m)] with J[..., i, j] = dB_i/dx_j
        """
        if center is None:
            center = torch.zeros(3, dtype=crds.dtype, device=crds.device)
        if R is None:
            R = torch.eye(3, dtype=crds.dtype, device=crds.device)

        a_t = torch.tensor(a, dtype=crds.dtype, device=crds.device)
        b_t = torch.tensor(b, dtype=crds.dtype, device=crds.device)

        r_t, dl = self._ellipse_quadrature(a_t, b_t, R, center, n_pts, crds.dtype, crds.device)

        flat = crds.reshape(-1, 3)
        out = torch.zeros((flat.shape[0], 3, 3), dtype=crds.dtype, device=crds.device)

        # Precompute skew(dl_k) for each quadrature node: [n,3,3]
        skew_dl = self._skew_from_vec(dl)  # [n,3,3]

        for i in range(0, flat.shape[0], chunk):
            pts = flat[i:i+chunk]                                        # [S,3]
            diff = pts[:, None, :] - r_t[None, :, :]                     # [S,n,3]
            Rn = diff.norm(dim=-1).clamp(min=EPSILON_STABILITY)          # [S,n]

            # skew term: [dl]_x / |rho|^3
            skew_term = skew_dl[None, :, :, :] / (Rn[:, :, None, None] ** 3)  # [S,n,3,3]

            # cross term: -3 (dlxrho) outer rho / |rho|^5
            dl_cross = torch.linalg.cross(dl[None, :, :].expand_as(diff), diff, dim=-1)  # [S,n,3]
            cross_term = -3.0 * dl_cross[:, :, :, None] * diff[:, :, None, :] / (Rn[:, :, None, None] ** 5)  # [S,n,3,3]

            out[i:i+chunk] = (MU0 / (4 * torch.pi)) * (skew_term + cross_term).sum(dim=1)

        return out.reshape(crds.shape[:-1] + (3, 3))
