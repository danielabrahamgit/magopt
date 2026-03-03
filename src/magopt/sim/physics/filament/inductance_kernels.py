import numpy as np
import torch
from typing import Optional
import functools            
import warnings

from ...core.constants import MU0, EPSILON_STABILITY

from ..filament.inductance_kernels import _gauss_legendre, _tanh_sinh


class base_inductance:
    def neumann_kernel_analytic(self,
                                A: torch.Tensor,
                                B: torch.Tensor,
                                C: torch.Tensor,
                                D: torch.Tensor,
                                warn_angle_deg: float = 30.0) -> torch.Tensor:
        """
        Exact Neumann mutual-inductance kernel via Grover's f(x) = x*ln(x) formula.

        flux = cos(theta) * [f(AD) + f(BC) - f(AC) - f(BD)]

        Exact for COLLINEAR segments. For non-collinear adjacent segments the
        cos(theta) factor accounts for the direction difference but the distances
        AC/AD/BC/BD are still computed from the true 3-D endpoint positions.
        Error scales as O(sin^2 theta) - reliable for theta < ~30deg (fine mesh on smooth
        curves), degrades on coarse meshes. Use quadrature for well-separated
        non-parallel pairs.

        Args:
        -----
        A : torch.Tensor
            Starting point of line segment AB
        B : torch.Tensor
            Ending point of line segment AB
        C : torch.Tensor
            Starting point of line segment CD
        D : torch.Tensor
            Ending point of line segment CD
        warn_angle_deg : float
            Emit a RuntimeWarning if any adjacent pair's inter-segment angle
            exceeds this value. Set to 180 to silence.

        Returns:
        --------
        torch.Tensor:
            Flux between two wire segments
        """
        def f(x):
            """
            Stable helper for x * log(x) near zero.

            Args:
            -----
            x : torch.Tensor
                Input distance tensor.

            Returns:
            --------
            torch.Tensor
                Stable evaluation of x * log(x).
            """
            return torch.where(x < 1e-14,
                               torch.zeros_like(x),
                               x * torch.log(x.clamp(min=torch.finfo(x.dtype).tiny)))

        AC = (A - C).norm(dim=-1)
        AD = (A - D).norm(dim=-1)
        BC = (B - C).norm(dim=-1)
        BD = (B - D).norm(dim=-1)

        # Direction cosine: +1 for parallel, -1 for anti-parallel, etc.
        e1 = (B - A) / (B - A).norm(dim=-1, keepdim=True).clamp(min=1e-30)
        e2 = (D - C) / (D - C).norm(dim=-1, keepdim=True).clamp(min=1e-30)
        cos_theta = (e1 * e2).sum(dim=-1)

        # Warn if any pair deviates too far from collinear
        if warn_angle_deg < 180.0:
            worst_cos = cos_theta.abs().min().item()
            angle     = float(np.degrees(np.arccos(np.clip(worst_cos, -1.0, 1.0))))
            if angle > warn_angle_deg:
                warnings.warn(
                    f"neumann_kernel_analytic: max inter-segment angle = {angle:.1f} deg "
                    f"exceeds warn_angle_deg={warn_angle_deg:.0f} deg. "
                    f"Accuracy degrades as O(sin^2theta) for non-collinear pairs - "
                    f"consider finer discretisation or quadrature.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return cos_theta * (f(AD) + f(BC) - f(AC) - f(BD))
    
    def neumann_kernel_gauss(self,
                            A: torch.Tensor, 
                            B: torch.Tensor, 
                            C: torch.Tensor, 
                            D: torch.Tensor,
                            n_pts: int = 10,
                            rtol: Optional[float] = None,
                            atol: Optional[float] = None,
                            max_pts: int = 320) -> torch.Tensor:
        """
        Gauss-Legendre quadrature for Neumann mutual-inductance kernel over STRAIGHT segments

        flux = int_dt int_ds ((B-A) dot (D-C)) / abs(A + t*(B-A) - C - s*(D-C)) dt ds
        dt,ds are [0,1] parametrizations of loop over segment 1 and segment 2, respectively.

        Args:
        -----
        A : torch.Tensor
            Starting point of line segment AB 
        B : torch.Tensor
            Ending point of line segment AB
        C : torch.Tensor
            Starting point of line segment CD
        D : torch.Tensor
            Ending point of line segment CD
        n_pts : int
            Number of gauss points
        rtol : Optional[float]
            Relative tolerance for integral convergence
        atol : Optional[float]
            Absolute tolerance for integral convergence
        max_pts : int
            Maximum n_pts if  using variable convergence

        Returns:
        --------
        torch.Tensor:
            Flux between two arbitrary 3-D wires
        """

        def _eval(n: int) -> torch.Tensor:
            """
            Single fixed-order evaluation.

            Args:
            -----
            n : int
                Quadrature order.

            Returns:
            --------
            torch.Tensor
                Kernel integral value.
            """
            t, w = _gauss_legendre(n)
            t = t.to(dtype=A.dtype, device=A.device)
            w = w.to(dtype=A.dtype, device=A.device)

            d1 = B - A # [..., 3]
            d2 = D - C # [..., 3]
            dot12 = (d1 * d2).sum(dim=-1) # [...]

            t_ = t.view(*([1] * (A.dim() - 1)), n, 1)
            r1 = A.unsqueeze(-2) + t_ * d1.unsqueeze(-2) # [..., n, 3]
            r2 = C.unsqueeze(-2) + t_ * d2.unsqueeze(-2) # [..., n, 3]

            diff = r1.unsqueeze(-2) - r2.unsqueeze(-3) # [..., n, n, 3]
            R = diff.norm(dim=-1).clamp(min=1e-14) # [..., n, n]

            ww = w.unsqueeze(-1) * w.unsqueeze(-2) # [n, n]
            integral = (ww / R).sum(dim=(-2, -1)) # [...]

            return dot12 * integral
        
        if rtol is None and atol is None:
            return _eval(n_pts)
        
        _rtol = rtol if rtol is not None else 0.0
        _atol = atol if atol is not None else 0.0

        current_n = n_pts
        current_val = _eval(current_n)

        while current_n < max_pts:
            next_n = current_n * 2
            next_val = _eval(next_n)

            # |I_new - I_old| <= atol + rtol * |I_new|  (element-wise)
            err = (next_val - current_val).abs()
            threshold = _atol + _rtol * next_val.abs()
            converged = (err <= threshold).all()

            current_n = next_n
            current_val = next_val

            if converged:
                break
        else:
            warnings.warn(
                f"neumann_kernel_gauss: tolerance not met at max_pts={max_pts}. "
                f"Max relative error ~ {(err / (next_val.abs() + 1e-30)).max().item():.2e}",
                RuntimeWarning,
                stacklevel=2,
            )

        return current_val
    
    def neumann_kernel_tanh_sinh(self,
                              A: torch.Tensor,
                              B: torch.Tensor,
                              C: torch.Tensor,
                              D: torch.Tensor,
                              n_pts:   int            = 40,
                              rtol:    Optional[float] = None,
                              atol:    Optional[float] = None,
                              max_pts: int            = 640) -> torch.Tensor:
        """
        Tanh-sinh (double exponential) quadrature for the Neumann kernel.

        flux = int_dt int_ds ((B-A) dot (D-C)) / abs(A + t*(B-A) - C - s*(D-C)) dt ds
        dt,ds are [0,1] parametrizations of loop over segment 1 and segment 2, respectively. 

        Preferred over Gauss-Legendre for ADJACENT segments (shared vertex),
        where the integrand is nearly singular near the endpoints. The
        double-exponential clustering of nodes near t=0 and t=1 gives
        dramatically faster convergence in that regime.

        For well-separated segments, Gauss-Legendre is equally good and
        slightly cheaper - use neumann_kernel_gauss there.

        Args:
        ----------
        A : torch.Tensor
            Starting point of line segment AB 
        B : torch.Tensor
            Ending point of line segment AB
        C : torch.Tensor
            Starting point of line segment CD
        D : torch.Tensor
            Ending point of line segment CD
        n_pts : int
            Number of gauss points
        rtol : Optional[float]
            Relative tolerance for integral convergence
        atol : Optional[float]
            Absolute tolerance for integral convergence
        max_pts : int
            Maximum n_pts if  using variable convergence

        Returns:
        --------
        torch.Tensor:
            Flux between two arbitrary 3-D wires
       

        Returns:
        --------
        torch.Tensor [...]
            Kernel value (multiply by mu0/4pi for mutual inductance [H])
        """

        def _eval(n: int) -> torch.Tensor:
            """
            Single fixed-order tanh-sinh evaluation.

            Args:
            -----
            n : int
                Quadrature order.

            Returns:
            --------
            torch.Tensor
                Kernel integral value.
            """
            t, w = _tanh_sinh(n)
            t = t.to(dtype=A.dtype, device=A.device)
            w = w.to(dtype=A.dtype, device=A.device)

            n_actual = t.shape[0] # may differ from n after masking

            d1 = B - A # [..., 3]
            d2 = D - C # [..., 3]
            dot12 = (d1 * d2).sum(dim=-1) # [...]

            t_ = t.view(*([1] * (A.dim() - 1)), n_actual, 1)

            r1 = A.unsqueeze(-2) + t_ * d1.unsqueeze(-2) # [..., n, 3]
            r2 = C.unsqueeze(-2) + t_ * d2.unsqueeze(-2) # [..., n, 3]

            diff = r1.unsqueeze(-2) - r2.unsqueeze(-3) # [..., n, n, 3]
            R = diff.norm(dim=-1).clamp(min=1e-14) # [..., n, n]

            # outer weight product
            ww = w.unsqueeze(-1) * w.unsqueeze(-2) # [n, n]
            integral = (ww / R).sum(dim=(-2, -1)) # [...]

            return dot12 * integral

        if rtol is None and atol is None:
            return _eval(n_pts)

        _rtol = rtol or 0.0
        _atol = atol or 0.0

        current_n = n_pts
        current_val = _eval(current_n)

        while current_n < max_pts:
            next_n = current_n * 2
            next_val = _eval(next_n)

            err = (next_val - current_val).abs()
            threshold = _atol + _rtol * next_val.abs()

            current_n = next_n
            current_val = next_val

            if (err <= threshold).all():
                break
        else:
            import warnings
            warnings.warn(
                f"{type(self).__name__}.neumann_kernel_tanh_sinh: "
                f"tolerance not met at max_pts={max_pts}. "
                f"Max relative error ~ "
                f"{(err / (next_val.abs() + 1e-30)).max().item():.2e}",
                RuntimeWarning,
                stacklevel=2,
            )

        return current_val
    
    def neumann_kernel_midpoint(self,
                            A: torch.Tensor,
                            B: torch.Tensor,
                            C: torch.Tensor,
                            D: torch.Tensor) -> torch.Tensor:
        """

        Midpoint quadrature for Neumann mutual-inductance kernel over STRAIGHT segments

        flux = (B-A)*(D-C) / |mid(AB) - mid(CD)|

        Args:
        -----
        A : torch.Tensor
            Starting point of line segment AB 
        B : torch.Tensor
            Ending point of line segment AB
        C : torch.Tensor
            Starting point of line segment CD
        D : torch.Tensor
            Ending point of line segment CD
        
        Returns:
        --------
        torch.Tensor:
            Flux between two arbitrary 3-D wires
        """
        d1 = B - A # [..., 3]
        d2 = D - C # [..., 3]
        dot12 = (d1 * d2).sum(dim=-1) # [...]

        mid1 = (A + B) / 2 # [..., 3]
        mid2 = (C + D) / 2 # [..., 3]

        R = (mid1 - mid2).norm(dim=-1) # [...]
        R = R.clamp(min=1e-14) # guard against self-interaction

        return dot12 / R
    
    @staticmethod
    def segment_self_inductance(length, r_wire, frequency='dc'):
        """
        Rosa-Neumann self-inductance for a straight cylindrical segment.

            L = mu_0 * l / 2pi * (ln(2l/r) - k) 
            L = (mu0 l / 2pi) * (ln(2l/r) - k))

        Args:
        -----
        length : torch.Tensor
            Segment length.
        r_wire : float
            Wire radius.
        frequency : str
            Current profile mode.

        Returns:
        --------
        torch.Tensor
            Segment self-inductance.
        """
        k = 0.75 if frequency == 'dc' else 1.0
        mu0 = torch.tensor(MU0, dtype=length.dtype, device=length.device)
        return (mu0 * length / (2 * torch.pi)) * (
            torch.log(2 * length / r_wire) - k
        )
