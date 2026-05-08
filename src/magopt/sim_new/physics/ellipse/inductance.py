import torch
from typing import Optional, Tuple
import warnings

from ..filament.inductance_kernels import _gauss_legendre, _tanh_sinh
from ..filament.inductance_kernels import base_inductance



class elliptical_inductance(base_inductance):
    """
    Mutual- and self-inductance for elliptical current loops via
    Gauss-Legendre quadrature of the Neumann integral.

    For two closed loops the mutual inductance is:

        M = (mu0 / 4pi) loop_intloop_int (dl1 * dl2) / |r1(phi1) - r2(phi2)|

    Because the loops are parameterized continuously, the double
    integral over (phi1, phi2) in [0,2pi]^2 is evaluated directly -
    no polygonal discretization is required.

    Self-inductance uses singularity subtraction: the near-diagonal
    singular part of the Neumann integrand is integrated analytically,
    and only the smooth remainder is handled by GL quadrature.  This
    avoids the catastrophically slow convergence of a naive product rule
    when the regularisation scale delta << loop size.
    """

    # ------------------------------------------------------------------
    # Quadrature helper (identical to elliptical_bfield)
    # ------------------------------------------------------------------
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
        Gauss-Legendre quadrature points and weighted tangent vectors
        for an ellipse in world coordinates.

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
            World positions and weighted tangents.
        """
        t, w = _gauss_legendre(n_pts)
        t = t.to(dtype=dtype, device=device)
        w = w.to(dtype=dtype, device=device)

        two_pi = torch.tensor(2.0 * torch.pi, dtype=dtype, device=device)
        phi = two_pi * t
        wphi = two_pi * w

        c = torch.cos(phi)
        s = torch.sin(phi)
        zero = torch.zeros_like(c)

        r_local = torch.stack([a * c, b * s, zero], dim=-1)
        dl_dphi_local = torch.stack([-a * s, b * c, zero], dim=-1)

        r_t = r_local @ R.T + center[None, :]
        dl = (dl_dphi_local @ R.T) * wphi[:, None]

        return r_t, dl

    # ------------------------------------------------------------------
    # Mutual inductance between two elliptical loops
    # ------------------------------------------------------------------
    def mutual_inductance_gauss(self,
                                a1: float, b1: float,
                                a2: float, b2: float,
                                center1: Optional[torch.Tensor] = None,
                                center2: Optional[torch.Tensor] = None,
                                R1: Optional[torch.Tensor] = None,
                                R2: Optional[torch.Tensor] = None,
                                n_pts: int = 80,
                                dtype: torch.dtype = torch.float64,
                                device: torch.device = torch.device('cpu')
                                ) -> torch.Tensor:
        """
        Mutual inductance between two unit-current elliptical loops via
        Gauss-Legendre quadrature of the Neumann integral:

            M = (mu0 / 4pi) loop_int1 loop_int2 (dl1 * dl2) / |r1 - r2|

        Args:
        -----
        a1, b1 : float
            Semi-major / semi-minor axes of loop 1 [m].
        a2, b2 : float
            Semi-major / semi-minor axes of loop 2 [m].
        center1, center2 : [3] tensors, optional
            World-frame centres (default origin).
        R1, R2 : [3,3] tensors, optional
            Rotation matrices local -> world (default identity).
        n_pts : int
            Number of Gauss-Legendre nodes per loop.
        dtype, device : torch types

        Returns:
        --------
        M : scalar tensor [H]  (henries)
        """
        if center1 is None:
            center1 = torch.zeros(3, dtype=dtype, device=device)
        if center2 is None:
            center2 = torch.zeros(3, dtype=dtype, device=device)
        if R1 is None:
            R1 = torch.eye(3, dtype=dtype, device=device)
        if R2 is None:
            R2 = torch.eye(3, dtype=dtype, device=device)

        a1_t = torch.tensor(a1, dtype=dtype, device=device)
        b1_t = torch.tensor(b1, dtype=dtype, device=device)
        a2_t = torch.tensor(a2, dtype=dtype, device=device)
        b2_t = torch.tensor(b2, dtype=dtype, device=device)

        r1, dl1 = self._ellipse_quadrature(a1_t, b1_t, R1, center1,
                                           n_pts, dtype, device)
        r2, dl2 = self._ellipse_quadrature(a2_t, b2_t, R2, center2,
                                           n_pts, dtype, device)

        # Pairwise distances: |r1_i - r2_j|   ->  [n, m]
        diff = r1[:, None, :] - r2[None, :, :]
        dist = diff.norm(dim=-1).clamp(min=EPSILON_STABILITY)

        # Pairwise dot products: dl1_i * dl2_j  ->  [n, m]
        dot = (dl1[:, None, :] * dl2[None, :, :]).sum(dim=-1)

        M = (MU0 / (4 * torch.pi)) * (dot / dist).sum()

        return M

    # ------------------------------------------------------------------
    # Self-inductance of one elliptical loop (singularity subtraction)
    # ------------------------------------------------------------------
    def self_inductance_gauss(self,
                              a: float,
                              b: float,
                              r_wire: float,
                              center: Optional[torch.Tensor] = None,
                              R: Optional[torch.Tensor] = None,
                              n_pts: int = 80,
                              frequency: str = 'dc',
                              dtype: torch.dtype = torch.float64,
                              device: torch.device = torch.device('cpu')
                              ) -> torch.Tensor:
        """
        Self-inductance of a unit-current elliptical loop via the
        singularity-subtracted Neumann integral.

        The full integrand is:

            F(phi1, phi2) = t(phi1)*t(phi2) / sqrt(|r(phi1)-r(phi2)|^2 + delta^2)

        where t(phi) = dr/dphi is the tangent vector and delta is the GMD
        regularisation parameter (encodes wire cross-section).

        Near the diagonal phi1 ~ phi2, F has a sharp peak of width ~delta/|t|
        that a product GL rule cannot resolve efficiently.  We subtract
        the leading singular model:

            F_sing(phi, Delta) = |t(phi)|^2 / sqrt(|t(phi)|^2*Delta^2 + delta^2)

        which captures the integrable 1/|Delta| singularity.  F_sing
        integrates analytically over Delta in [-pi, pi]:

            int F_sing dDelta = 2|t(phi)| * arcsinh(pi|t(phi)| / delta)

        The remainder F - F_sing is smooth and bounded everywhere
        (including the diagonal), so the product GL rule converges
        exponentially for it.

        GMD regularisation
        ------------------
        The regularised Neumann integral yields:

            L_reg = mu0R [ln(8R/delta) - 2]     (for a circle of radius R)

        Choosing delta to reproduce the correct total inductance:

            DC  (uniform current):  delta = r_wire * exp(-1/4)
                -> L = mu0R [ln(8R/r) - 7/4]  (includes internal L)
            HF  (surface current):  delta = r_wire
                -> L = mu0R [ln(8R/r) - 2]    (external only)

        Args:
        -----
        a, b : float
            Semi-major / semi-minor axes [m].
        r_wire : float
            Wire radius [m].
        center, R : optional
            Position / orientation (do not affect result - inductance
            is a geometric invariant).
        n_pts : int
            Gauss-Legendre nodes for each of the two angular variables.
            With singularity subtraction, 80 is typically sufficient.
        frequency : {'dc', 'hf'}
            Selects the GMD exponent.
        dtype, device : torch types

        Returns:
        --------
        L : scalar tensor [H]
        """
        # Self-inductance is rotation/translation invariant - work in local frame
        a_t = torch.tensor(a, dtype=dtype, device=device)
        b_t = torch.tensor(b, dtype=dtype, device=device)

        # GMD regularisation
        if frequency == 'dc':
            # GMD of uniform-current circular cross-section = r*exp(-1/4)
            # Adds the internal inductance: mu0R/4 for a circle
            delta = r_wire * torch.exp(torch.tensor(-0.25, dtype=dtype, device=device))
        else:
            # HF skin-effect: current on surface, internal L -> 0
            # delta = r_wire gives pure external inductance
            delta = torch.tensor(r_wire, dtype=dtype, device=device)

        delta2 = delta ** 2

        # -- GL nodes for phi (outer) and Delta (inner), both on [0, 2pi] --
        t_out, w_out = _gauss_legendre(n_pts)
        t_out = t_out.to(dtype=dtype, device=device)
        w_out = w_out.to(dtype=dtype, device=device)
        phi_bar = 2.0 * torch.pi * t_out        # [N]
        wphi_bar = 2.0 * torch.pi * w_out       # [N]

        t_in, w_in = _gauss_legendre(n_pts)
        t_in = t_in.to(dtype=dtype, device=device)
        w_in = w_in.to(dtype=dtype, device=device)
        Delta = 2.0 * torch.pi * t_in           # [M]  in [0, 2pi]
        wDelta = 2.0 * torch.pi * w_in          # [M]

        # Wrap Delta to [-pi, pi] for the singular model distance
        Delta_wrapped = torch.where(Delta > torch.pi,
                                    Delta - 2.0 * torch.pi,
                                    Delta)       # [M]

        # -- Tangent magnitude |t(phi)| at each outer node -------------
        # t(phi) = [-a sinphi, b cosphi, 0]
        # |t(phi)|^2 = a^2sin^2phi + b^2cos^2phi
        cos_bar = torch.cos(phi_bar)
        sin_bar = torch.sin(phi_bar)
        t_mag2 = (a_t * sin_bar) ** 2 + (b_t * cos_bar) ** 2  # [N]
        t_mag = torch.sqrt(t_mag2)                              # [N]

        # -- SINGULAR PART (analytic in Delta) ----------------------------
        # int-pi^pi |t|^2/sqrt(|t|^2Delta^2 + delta^2) dDelta = 2|t| arcsinh(pi|t|/delta)
        L_sing_1d = 2.0 * t_mag * torch.arcsinh(
            torch.pi * t_mag / delta)             # [N]
        L_singular = (wphi_bar * L_sing_1d).sum()

        # -- FULL INTEGRAND F(phi, phi - Delta) -----------------------------
        phi1 = phi_bar                            # [N]
        phi2 = phi1[:, None] - Delta[None, :]     # [N, M]

        # Ellipse positions in LOCAL frame (rotation irrelevant for L)
        r1 = torch.stack([a_t * torch.cos(phi1),
                          b_t * torch.sin(phi1),
                          torch.zeros_like(phi1)], dim=-1)       # [N, 3]

        r2 = torch.stack([a_t * torch.cos(phi2),
                          b_t * torch.sin(phi2),
                          torch.zeros_like(phi2)], dim=-1)       # [N, M, 3]

        # Tangent vectors
        t1 = torch.stack([-a_t * sin_bar,
                           b_t * cos_bar,
                           torch.zeros_like(phi1)], dim=-1)      # [N, 3]

        t2 = torch.stack([-a_t * torch.sin(phi2),
                           b_t * torch.cos(phi2),
                           torch.zeros_like(phi2)], dim=-1)      # [N, M, 3]

        # t1 * t2
        dot_t = (t1[:, None, :] * t2).sum(dim=-1)               # [N, M]

        # |r1 - r2|^2 + delta^2
        diff = r1[:, None, :] - r2                               # [N, M, 3]
        dist2_reg = (diff ** 2).sum(dim=-1) + delta2             # [N, M]

        F_full = dot_t / torch.sqrt(dist2_reg)                   # [N, M]

        # -- SINGULAR MODEL F_sing ------------------------------------
        F_sing = t_mag2[:, None] / torch.sqrt(
            t_mag2[:, None] * Delta_wrapped[None, :] ** 2 + delta2)  # [N, M]

        # -- SMOOTH REMAINDER -----------------------------------------
        F_reg = F_full - F_sing                                  # [N, M]

        L_regular = (wphi_bar[:, None] * wDelta[None, :] * F_reg).sum()

        L = (MU0 / (4.0 * torch.pi)) * (L_singular + L_regular)

        return L

    # ------------------------------------------------------------------
    # Mutual inductance (adaptive)
    # ------------------------------------------------------------------
    def mutual_inductance_adaptive(self,
                                   a1: float, b1: float,
                                   a2: float, b2: float,
                                   center1: Optional[torch.Tensor] = None,
                                   center2: Optional[torch.Tensor] = None,
                                   R1: Optional[torch.Tensor] = None,
                                   R2: Optional[torch.Tensor] = None,
                                   n_pts: int = 40,
                                   rtol: float = 1e-8,
                                   max_pts: int = 640,
                                   dtype: torch.dtype = torch.float64,
                                   device: torch.device = torch.device('cpu')
                                   ) -> torch.Tensor:
        """
        Adaptive mutual inductance - doubles quadrature order until the
        result converges to *rtol* or *max_pts* is reached.

        Args:
        -----
        a1 : float
            Semi-major axis of loop 1.
        b1 : float
            Semi-minor axis of loop 1.
        a2 : float
            Semi-major axis of loop 2.
        b2 : float
            Semi-minor axis of loop 2.
        center1 : Optional[torch.Tensor]
            Center of loop 1.
        center2 : Optional[torch.Tensor]
            Center of loop 2.
        R1 : Optional[torch.Tensor]
            Rotation matrix of loop 1.
        R2 : Optional[torch.Tensor]
            Rotation matrix of loop 2.
        n_pts : int
            Initial quadrature order.
        rtol : float
            Relative convergence tolerance.
        max_pts : int
            Maximum quadrature order.
        dtype : torch.dtype
            Tensor dtype.
        device : torch.device
            Tensor device.

        Returns:
        --------
        torch.Tensor
            Mutual inductance estimate.
        """
        prev = self.mutual_inductance_gauss(
            a1, b1, a2, b2, center1, center2, R1, R2,
            n_pts=n_pts, dtype=dtype, device=device)

        current_n = n_pts
        while current_n < max_pts:
            next_n = min(current_n * 2, max_pts)
            curr = self.mutual_inductance_gauss(
                a1, b1, a2, b2, center1, center2, R1, R2,
                n_pts=next_n, dtype=dtype, device=device)

            err = (curr - prev).abs()
            if err <= rtol * curr.abs():
                return curr

            prev = curr
            current_n = next_n

        warnings.warn(
            f"mutual_inductance_adaptive: rtol={rtol} not met at "
            f"max_pts={max_pts}. Approx relative error = "
            f"{(err / (curr.abs() + 1e-30)).item():.2e}",
            RuntimeWarning, stacklevel=2)
        return curr

    # ------------------------------------------------------------------
    # Self-inductance (adaptive)
    # ------------------------------------------------------------------
    def self_inductance_adaptive(self,
                                 a: float,
                                 b: float,
                                 r_wire: float,
                                 center: Optional[torch.Tensor] = None,
                                 R: Optional[torch.Tensor] = None,
                                 n_pts: int = 40,
                                 rtol: float = 1e-8,
                                 max_pts: int = 640,
                                 frequency: str = 'dc',
                                 dtype: torch.dtype = torch.float64,
                                 device: torch.device = torch.device('cpu')
                                 ) -> torch.Tensor:
        """
        Adaptive self-inductance - doubles quadrature order until the
        result converges to *rtol* or *max_pts* is reached.

        Args:
        -----
        a : float
            Semi-major axis.
        b : float
            Semi-minor axis.
        r_wire : float
            Wire radius.
        center : Optional[torch.Tensor]
            Loop center.
        R : Optional[torch.Tensor]
            Rotation matrix.
        n_pts : int
            Initial quadrature order.
        rtol : float
            Relative convergence tolerance.
        max_pts : int
            Maximum quadrature order.
        frequency : str
            Current profile mode.
        dtype : torch.dtype
            Tensor dtype.
        device : torch.device
            Tensor device.

        Returns:
        --------
        torch.Tensor
            Self-inductance estimate.
        """
        prev = self.self_inductance_gauss(
            a, b, r_wire, center, R,
            n_pts=n_pts, frequency=frequency, dtype=dtype, device=device)

        current_n = n_pts
        while current_n < max_pts:
            next_n = min(current_n * 2, max_pts)
            curr = self.self_inductance_gauss(
                a, b, r_wire, center, R,
                n_pts=next_n, frequency=frequency, dtype=dtype, device=device)

            err = (curr - prev).abs()
            if err <= rtol * curr.abs():
                return curr

            prev = curr
            current_n = next_n

        warnings.warn(
            f"self_inductance_adaptive: rtol={rtol} not met at "
            f"max_pts={max_pts}. Approx relative error = "
            f"{(err / (curr.abs() + 1e-30)).item():.2e}",
            RuntimeWarning, stacklevel=2)
        return curr

    # ------------------------------------------------------------------
    # Convenience: circular loop (a == b == radius)
    # ------------------------------------------------------------------
    def circular_self_inductance_gauss(self,
                                       radius: float,
                                       r_wire: float,
                                       n_pts: int = 80,
                                       frequency: str = 'dc',
                                       dtype: torch.dtype = torch.float64,
                                       device: torch.device = torch.device('cpu')
                                       ) -> torch.Tensor:
        """
        Self-inductance of a circular loop (special case a = b = radius).

        Validates against Wien's formula:

            L = mu0 R [ ln(8R/r) - k ]

        k = 7/4 (DC, includes internal inductance)
        k = 2   (HF, external only)

        Args:
        -----
        radius : float
            Loop radius.
        r_wire : float
            Wire radius.
        n_pts : int
            Quadrature order.
        frequency : str
            Current profile mode.
        dtype : torch.dtype
            Tensor dtype.
        device : torch.device
            Tensor device.

        Returns:
        --------
        torch.Tensor
            Circular loop self-inductance.
        """
        return self.self_inductance_gauss(
            a=radius, b=radius, r_wire=r_wire,
            n_pts=n_pts, frequency=frequency,
            dtype=dtype, device=device)

    def circular_self_inductance_exact(self,
                                       radius: float,
                                       r_wire: float,
                                       frequency: str = 'dc',
                                       dtype: torch.dtype = torch.float64,
                                       device: torch.device = torch.device('cpu')
                                       ) -> torch.Tensor:
        """
        Wien's exact formula for a thin circular loop:

            L = mu0 R [ ln(8R/r) - k ]

        where:
            k = 7/4  (DC: uniform current, includes internal inductance)
            k = 2    (HF: skin effect, external inductance only)

        Args:
        -----
        radius : float
            Loop radius.
        r_wire : float
            Wire radius.
        frequency : str
            Current profile mode.
        dtype : torch.dtype
            Tensor dtype.
        device : torch.device
            Tensor device.

        Returns:
        --------
        torch.Tensor
            Exact circular loop self-inductance.
        """
        R_t = torch.tensor(radius, dtype=dtype, device=device)
        r_t = torch.tensor(r_wire, dtype=dtype, device=device)
        mu0 = torch.tensor(MU0, dtype=dtype, device=device)
        k = torch.tensor(1.75 if frequency == 'dc' else 2.0,
                         dtype=dtype, device=device)
        return mu0 * R_t * (torch.log(8.0 * R_t / r_t) - k)
