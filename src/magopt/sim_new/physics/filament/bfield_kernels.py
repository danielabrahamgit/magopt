import torch
import numpy as np
import functools


from ...constants import MU0, EPSILON_STABILITY
from .inductance_kernels import _gauss_legendre, _tanh_sinh


class base_bfield:
    """
    Mixin providing per-batch kernels for magnetic vector potential,
    B-field, and B-field Jacobian via the Biot-Savart law.

    Each kernel computes the contribution of a BATCH of segments to
    the field at a set of evaluation points. The calling method in
    parametric_wire handles building segments, batching with tqdm,
    accumulating, and reshaping - the same split used in base_inductance
    between neumann_kernel_* and calc_self_inductance.

    Kernel signatures:
        ds   : torch.Tensor [B, 3]  segment vectors (Q - P)
        mid  : torch.Tensor [B, 3]  segment midpoints
        crds : torch.Tensor [S, 3]  evaluation coordinates
    """

    # -- Helper ----------------------------------------------------------------

    @staticmethod
    def _skew_from_vec(v: torch.Tensor) -> torch.Tensor:
        """
        Skew-symmetric matrix [v]_x such that [v]_x @ u = v x u.

        Args:
        -----
        v : torch.Tensor
            Vector field with shape (..., 3)

        Returns:
        --------
        torch.Tensor:
            Skew-symmetric matrices with shape (..., 3, 3)
        """
        vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
        O = torch.zeros_like(vx)
        return torch.stack([
            torch.stack([ O, -vz,  vy], dim=-1),
            torch.stack([ vz,  O, -vx], dim=-1),
            torch.stack([-vy,  vx,  O], dim=-1),
        ], dim=-2) # [..., 3, 3]

    # -- Vector potential kernels ----------------------------------------------

    def potential_kernel_midpoint(self,
                                  ds:   torch.Tensor,
                                  mid:  torch.Tensor,
                                  crds: torch.Tensor) -> torch.Tensor:
        """
        Midpoint-rule contribution of a batch of segments to the magnetic
        vector potential at evaluation points crds.

            A_batch = (mu0/4pi) * sum_b dl_b / |r - mid_b|

        Args:
        -----
        ds : torch.Tensor
            Segment vectors (Q - P) with shape (B, 3) [m]
        mid : torch.Tensor
            Segment midpoints with shape (B, 3) [m]
        crds : torch.Tensor
            Evaluation coordinates with shape (S, 3) [m]

        Returns:
        --------
        torch.Tensor:
            Potential contribution with shape (S, 3) [T*m]
        """
        distances = torch.cdist(crds, mid) # [S, B]
        return (MU0 / (4 * torch.pi)) * (
            ds[None, :, :] / (distances[:, :, None] + EPSILON_STABILITY)
        ).sum(dim=1) # [S, 3]

    def potential_kernel_gauss(self,
                               ds:    torch.Tensor,
                               mid:   torch.Tensor,
                               crds:  torch.Tensor,
                               n_pts: int = 10) -> torch.Tensor:
        """
        Gauss-Legendre contribution of a batch of segments to the magnetic
        vector potential at evaluation points crds.

            A_batch = (mu0/4pi) * sum_b int_0^1 dl_b / |r - r'_b(t)| dt

        Exponentially convergent in n_pts for field points away from the wire.

        Args:
        -----
        ds : torch.Tensor
            Segment vectors (Q - P) with shape (B, 3) [m]
        mid : torch.Tensor
            Segment midpoints with shape (B, 3) [m]
        crds : torch.Tensor
            Evaluation coordinates with shape (S, 3) [m]
        n_pts : int
            Number of Gauss-Legendre quadrature points

        Returns:
        --------
        torch.Tensor:
            Potential contribution with shape (S, 3) [T*m]
        """
        t, w  = _gauss_legendre(n_pts)
        t     = t.to(dtype=ds.dtype, device=ds.device) # [n]
        w     = w.to(dtype=ds.dtype, device=ds.device) # [n]

        start = mid - ds / 2                                         # [B, 3]
        r_t   = start[:, None, :] + t[None, :, None] * ds[:, None, :] # [B, n, 3]

        diff  = crds[:, None, None, :] - r_t[None, :, :, :]         # [S, B, n, 3]
        R     = diff.norm(dim=-1).clamp(min=EPSILON_STABILITY)       # [S, B, n]

        # int_0^1 dl / |r - r'(t)| dt ~ sum_k w_k * dl / R_k
        integrand = (w[None, None, :, None] / R[:, :, :, None]) * ds[None, :, None, :] # [S, B, n, 3]
        return (MU0 / (4 * torch.pi)) * integrand.sum(dim=(1, 2)) # [S, 3]

    # -- B-field kernels -------------------------------------------------------

    def bfield_kernel_midpoint(self,
                               ds:   torch.Tensor,
                               mid:  torch.Tensor,
                               crds: torch.Tensor) -> torch.Tensor:
        """
        Midpoint-rule contribution of a batch of segments to the magnetic
        field at evaluation points crds.

            B_batch = (mu0/4pi) * sum_b (dl_b x r_hat_b) / |r - mid_b|^2

        Args:
        -----
        ds : torch.Tensor
            Segment vectors (Q - P) with shape (B, 3) [m]
        mid : torch.Tensor
            Segment midpoints with shape (B, 3) [m]
        crds : torch.Tensor
            Evaluation coordinates with shape (S, 3) [m]

        Returns:
        --------
        torch.Tensor:
            Field contribution with shape (S, 3) [T/A]
        """
        r         = crds[:, None, :] - mid[None, :, :]                          # [S, B, 3]
        norm      = EPSILON_STABILITY + torch.linalg.norm(r, dim=-1)[..., None] # [S, B, 1]
        r_hat     = r / norm                                                     # [S, B, 3]
        bsl       = torch.linalg.cross(ds[None, :, :], r_hat) / (norm ** 2)     # [S, B, 3]
        return (MU0 / (4 * torch.pi)) * bsl.sum(dim=1) # [S, 3]

    def bfield_kernel_gauss(self,
                            ds:    torch.Tensor,
                            mid:   torch.Tensor,
                            crds:  torch.Tensor,
                            n_pts: int = 10) -> torch.Tensor:
        """
        Gauss-Legendre contribution of a batch of segments to the magnetic
        field at evaluation points crds.

            B_batch = (mu0/4pi) * sum_b int_0^1 (dl_b x (r - r'_b(t))) / |r - r'_b(t)|^3 dt

        Exponentially convergent in n_pts for field points away from the wire.

        Args:
        -----
        ds : torch.Tensor
            Segment vectors (Q - P) with shape (B, 3) [m]
        mid : torch.Tensor
            Segment midpoints with shape (B, 3) [m]
        crds : torch.Tensor
            Evaluation coordinates with shape (S, 3) [m]
        n_pts : int
            Number of Gauss-Legendre quadrature points

        Returns:
        --------
        torch.Tensor:
            Field contribution with shape (S, 3) [T/A]
        """
        t, w  = _gauss_legendre(n_pts)
        t     = t.to(dtype=ds.dtype, device=ds.device) # [n]
        w     = w.to(dtype=ds.dtype, device=ds.device) # [n]

        start = mid - ds / 2                                           # [B, 3]
        r_t   = start[:, None, :] + t[None, :, None] * ds[:, None, :] # [B, n, 3]

        diff  = crds[:, None, None, :] - r_t[None, :, :, :]           # [S, B, n, 3]
        R     = diff.norm(dim=-1).clamp(min=EPSILON_STABILITY)         # [S, B, n]

        # (dl x (r - r'(t))) / |r - r'(t)|^3
        dl_cross = torch.linalg.cross(
            ds[None, :, None, :].expand_as(diff), diff
        )                                                              # [S, B, n, 3]

        integrand = w[None, None, :, None] * dl_cross / R[..., None] ** 3 # [S, B, n, 3]
        return (MU0 / (4 * torch.pi)) * integrand.sum(dim=(1, 2)) # [S, 3]

    # -- B-field Jacobian kernel -----------------------------------------------

    def jacobian_kernel_midpoint(self,
                                 ds:   torch.Tensor,
                                 mid:  torch.Tensor,
                                 crds: torch.Tensor) -> torch.Tensor:
        """
        Midpoint-rule contribution of a batch of segments to the magnetic
        field Jacobian dB_i/dx_j at evaluation points crds.

        Derived analytically from the Biot-Savart integrand:

            dB/dr = (mu0/4pi) * [ [dl]_x / |r|^3
                                  - 3 (dl x r) (x) r / |r|^5 ]

        where [dl]_x is the skew-symmetric matrix of dl and (x) is the
        outer product.

        Args:
        -----
        ds : torch.Tensor
            Segment vectors (Q - P) with shape (B, 3) [m]
        mid : torch.Tensor
            Segment midpoints with shape (B, 3) [m]
        crds : torch.Tensor
            Evaluation coordinates with shape (S, 3) [m]

        Returns:
        --------
        torch.Tensor:
            Jacobian contribution with shape (S, 3, 3) [T/(A*m)]
            Index convention: out[..., i, j] = dB_i/dx_j
        """
        r    = crds[:, None, :] - mid[None, :, :]                                    # [S, B, 3]
        norm = EPSILON_STABILITY + torch.linalg.norm(r, dim=-1)[..., None, None]     # [S, B, 1, 1]

        # [dl]_x / |r|^3 term
        skew_term  = self._skew_from_vec(ds[None, :, :]) / (norm ** 3)               # [S, B, 3, 3]

        # -3 (dl x r) (x) r / |r|^5 term
        dl_cross_r = torch.linalg.cross(ds[None, :, :], r, dim=-1)                   # [S, B, 3]
        cross_term = -3 * dl_cross_r[..., None] * r[..., None, :] / (norm ** 5)      # [S, B, 3, 3]

        return (MU0 / (4 * torch.pi)) * (skew_term + cross_term).sum(dim=1) # [S, 3, 3]

    def jacobian_kernel_gauss(self,
                              ds:    torch.Tensor,
                              mid:   torch.Tensor,
                              crds:  torch.Tensor,
                              n_pts: int = 10) -> torch.Tensor:
        """
        Gauss-Legendre contribution of a batch of segments to the magnetic
        field Jacobian dB_i/dx_j at evaluation points crds.

        Integrates the analytic Biot-Savart Jacobian kernel along each segment:

            dB/dr = (mu0/4pi) * int_0^1 [ [dl]_x / |r - r'(t)|^3
                                          - 3 (dl x (r-r'(t))) (x) (r-r'(t)) / |r - r'(t)|^5 ] dt

        Exponentially convergent in n_pts for field points away from the wire.

        Args:
        -----
        ds : torch.Tensor
            Segment vectors (Q - P) with shape (B, 3) [m]
        mid : torch.Tensor
            Segment midpoints with shape (B, 3) [m]
        crds : torch.Tensor
            Evaluation coordinates with shape (S, 3) [m]
        n_pts : int
            Number of Gauss-Legendre quadrature points

        Returns:
        --------
        torch.Tensor:
            Jacobian contribution with shape (S, 3, 3) [T/(A*m)]
            Index convention: out[..., i, j] = dB_i/dx_j
        """
        t, w  = _gauss_legendre(n_pts)
        t     = t.to(dtype=ds.dtype, device=ds.device) # [n]
        w     = w.to(dtype=ds.dtype, device=ds.device) # [n]

        start = mid - ds / 2                                           # [B, 3]
        r_t   = start[:, None, :] + t[None, :, None] * ds[:, None, :] # [B, n, 3]

        diff  = crds[:, None, None, :] - r_t[None, :, :, :]           # [S, B, n, 3]
        R     = diff.norm(dim=-1).clamp(min=EPSILON_STABILITY)         # [S, B, n]

        # [dl]_x / |r - r'(t)|^3 term
        skew_term  = self._skew_from_vec(
            ds[None, :, None, :].expand(*diff.shape[:-1], 3)
        ) / R[..., None, None] ** 3                                    # [S, B, n, 3, 3]

        # -3 (dl x (r-r'(t))) (x) (r-r'(t)) / |r - r'(t)|^5 term
        dl_cross   = torch.linalg.cross(
            ds[None, :, None, :].expand_as(diff), diff
        )                                                              # [S, B, n, 3]
        cross_term = -3 * dl_cross[..., None] * diff[..., None, :] / R[..., None, None] ** 5 # [S, B, n, 3, 3]

        integrand  = w[None, None, :, None, None] * (skew_term + cross_term) # [S, B, n, 3, 3]
        return (MU0 / (4 * torch.pi)) * integrand.sum(dim=(1, 2)) # [S, 3, 3]