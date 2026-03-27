import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Optional

from .inductance_kernels import base_inductance
from .bfield_kernels     import base_bfield

# Optional: analytic/parametric ellipse kernels (full-loop Gauss quadrature)
from ..filament.inductance_kernels import _gauss_legendre, _tanh_sinh
from ..ellipse.bfield import elliptical_bfield as _elliptical_bfield_kernel
from ...core.constants import MU0, EPSILON_STABILITY

class parametric_wire(base_inductance, base_bfield):
    """
    Parametric path defined by a set of points in 3D space.
    """
    
    def __init__(self,
                 wire_pts: torch.Tensor,
                 wire_rad: float = 0.5e-3,
                 closed: bool = False,
                 wire_seg_batch: Optional[int] = None,
                 verbose: bool = True):
        """ 
        Args:
        -----
        wire_pts : torch.Tensor
            Wire spatial points with shape (N, 3) in units [m]
        wire_rad : float
            Radius of wire
        closed : Optional[bool]
            If True, wire is closed loop.
        wire_seg_batch : Optional[int]
            Batching over the N wire segments for calculations
        verbose : Optional[bool]
            If True, prints progress during calculations.

        Returns:
        --------
        None
            Initializes wire object state.
        """
        self.wire_pts = wire_pts
        self.closed = closed
        self.wire_rad = wire_rad
        self.verbose = verbose
        if wire_seg_batch is None:
            self.wire_seg_batch = wire_pts.shape[0]
        else:
            self.wire_seg_batch = wire_seg_batch
       
    @property
    def cross_sectional_area(self):
        """
        ANALYTIC

        Args:
        -----
        None
            Property getter with no input arguments.

        Returns:
        --------
        float
            Cross sectional area of wire [m^2]
        """
        return torch.pi * self.wire_rad**2 

    @property
    def length(self) -> float:
        """
        DISCRETE

        Calculate arc length from discrete points.
        Subclasses can override with analytical formulas.

        Args:
        -----
        None
            Property getter with no input arguments.

        Returns:
        --------
        float
            Total arc length [m]
        """
        if self.closed:
            # Closed loop: N segments including closing segment
            next_pts = torch.roll(self.wire_pts, shifts=-1, dims=0)  # [N, 3]
            segments = next_pts - self.wire_pts  # [N, 3]
        else:
            # Open wire: N-1 segments
            segments = self.wire_pts[1:] - self.wire_pts[:-1]  # [N-1, 3]
        
        return segments.norm(dim=-1).sum().item()
     
    def calc_resistance(self,
                        resistivity: float) -> float:
        """
        ANALYTIC 

        Calculates the resistance of the wire using the formula:
        R = restivity * length / cross sectional area

        Args:        
        -----
        resistivity : float
            Resistivity of the wire material [Ohm m]

        Returns:
        --------
        float
            Resistance of the full wire [Ohm]
        """
        return resistivity * self.length / self.cross_sectional_area

    def _build_segments(self):
        """
        Build segment endpoints from stored wire points.

        Args:
        -----
        None
            Uses object state only.

        Returns:
        --------
        tuple[torch.Tensor, torch.Tensor, int]
            Segment starts, segment ends, and number of segments.
        """
        pts = self.wire_pts
        N   = pts.shape[0]
        if self.closed:
            return pts, torch.roll(pts, shifts=-1, dims=0), N
        return pts[:-1], pts[1:], N - 1

    def _build_adjacency(self, lim: int) -> torch.Tensor:
        """
        Build adjacency mask for segment pairs.

        Args:
        -----
        lim : int
            Number of segments.

        Returns:
        --------
        torch.Tensor
            Boolean mask [lim, lim] for adjacent segment pairs.
        """
        idx         = torch.arange(lim, device=self.wire_pts.device)
        dist_ij     = (idx[:, None] - idx[None, :]).abs()
        is_adjacent = dist_ij == 1
        if self.closed:
            is_adjacent |= dist_ij == (lim - 1)
        return is_adjacent

    # -- Main inductance -------------------------------------------------------

    def calc_self_inductance(self,
                             n_quad:            int            = 10,
                             n_adj:             int            = 40,
                             rtol:              Optional[float] = None,
                             atol:              Optional[float] = None,
                             frequency:         str            = 'dc',
                             kernel:            str            = 'gauss',
                             analytic_adjacent: bool           = True,
                             warn_angle_deg:    float          = 30.0) -> float:
        """
        Total self-inductance via Neumann's formula:

            L = self_inductance + mutual-inductance

        Args:
        -----
        n_quad : int
            GL / tanh-sinh order for well-separated pairs
        n_adj : int
            Quadrature order for adjacent (near-singular) pairs.
            Ignored when analytic_adjacent=True or when rtol/atol are set.
        rtol : Optional[float]
            Relative tolerance for adaptive doubling. Applied to all
            non-adjacent pairs (adjacent still use analytic when enabled).
        atol : Optional[float]
            Absolute tolerance for adaptive doubling.
        frequency : str
            'dc' or 'hf' - controls self-term coefficient k (0.75 or 1.0)
        kernel : str
            'gauss'     - Gauss-Legendre quadrature (good general default)
            'tanh_sinh' - tanh-sinh quadrature
            'midpoint'  - midpoint rule (fast, low accuracy)
        analytic_adjacent : bool
            If True, adjacent segment pairs (those sharing a vertex) are
            evaluated with the exact Grover analytic formula instead of
            quadrature. This eliminates the O(1/n^2) error floor on adjacent
            pairs entirely:
                Collinear (straight wire)  - machine-precision exact
                Non-collinear (curved wire) - error is O(sin^2 theta) where theta is
                the inter-segment angle; small for fine meshes on smooth curves
            Set to False to use the selected quadrature kernel for all pairs,
            which is useful for benchmarking.
        warn_angle_deg : float
            When analytic_adjacent=True, warn if any adjacent pair's
            inter-segment angle exceeds this value. Set to 180 to silence.

        Returns:
        --------
        float:
            Self-inductance of the wire [H]
        """
        seg_start, seg_end, lim = self._build_segments()
        seg_lengths = (seg_end - seg_start).norm(dim=-1)   # [lim]

        # -- Diagonal: self-inductance per segment -------------------------
        L_self = self.segment_self_inductance(
            seg_lengths, self.wire_rad, frequency=frequency
        ).sum()

        # -- Off-diagonal: mutual inductance -------------------------------
        As = seg_start[:, None, :]   # [lim, 1, 3]
        Bs = seg_end  [:, None, :]
        Cs = seg_start[None, :, :]   # [1, lim, 3]
        Ds = seg_end  [None, :, :]

        kernel_map = {
            'gauss'     : self.neumann_kernel_gauss,
            'tanh_sinh' : self.neumann_kernel_tanh_sinh,
            'midpoint'  : self.neumann_kernel_midpoint,
        }
        if kernel not in kernel_map:
            raise ValueError(
                f"kernel must be one of {list(kernel_map)}, got '{kernel}'"
            )
        kernel_fn   = kernel_map[kernel]
        adaptive    = rtol is not None or atol is not None
        is_adjacent = self._build_adjacency(lim)

        if kernel == 'midpoint':
            # no n_pts / tolerance arguments
            kernels = kernel_fn(As, Bs, Cs, Ds)

        elif adaptive:
            # single pass - tolerance handles all pairs
            kernels = kernel_fn(As, Bs, Cs, Ds,
                                n_pts=n_quad, rtol=rtol, atol=atol)
        else:
            # two passes: normal order for well-separated, higher for adjacent
            kernels = kernel_fn(As, Bs, Cs, Ds, n_pts=n_quad)
            if is_adjacent.any() and n_adj > n_quad:
                kernels_adj = kernel_fn(As, Bs, Cs, Ds, n_pts=n_adj)
                kernels     = torch.where(is_adjacent, kernels_adj, kernels)

        # -- Override adjacent pairs with exact analytic formula -----------
        # Replaces the O(1/n^2) quadrature floor with machine precision for
        # collinear pairs and O(sin^2 theta) error for non-collinear pairs.
        # Applied after the quadrature pass so non-adjacent pairs are unchanged.
        if analytic_adjacent and is_adjacent.any():
            kernels_analytic = self.neumann_kernel_analytic(
                As, Bs, Cs, Ds, warn_angle_deg=warn_angle_deg
            )
            kernels = torch.where(is_adjacent, kernels_analytic, kernels)

        upper = torch.triu(
            torch.ones(lim, lim, dtype=torch.bool,
                       device=self.wire_pts.device),
            diagonal=1
        )
        mu0      = torch.tensor(MU0,
                                dtype=self.wire_pts.dtype,
                                device=self.wire_pts.device)
        L_mutual = (mu0 / (4 * torch.pi)) * kernels[upper].sum()

        return (2 * L_mutual + L_self).item()

    # -- Legacy midpoint (benchmarking only) -----------------------------------

    def calc_self_inductance_midpoint(self) -> float:
        """
        Original midpoint implementation - kept for benchmarking only.
        Equivalent to calc_self_inductance(kernel='midpoint') but uses
        the original batched loop with tqdm progress bar.
        
        DISCRETIZED 

        Calculates the inductance of the wire using the formula:
        L = mu0 / 4pi * int_loop1, int_loop2 dl1 cdot dl2 / |r1 - r2|

        Self Inductance would have loop1 == loop2

        Args:
        -----
        None
            Uses object state only.

        Returns:
        --------
        float
            Inductance of the wire [H]
        """
        wire_pts = self.wire_pts
        N        = wire_pts.shape[0]
        lim      = N if self.closed else N - 1

        inductance = torch.tensor(0.0)

        if self.closed:
            next_pts  = torch.roll(wire_pts, shifts=-1, dims=0)
            dx        = next_pts - wire_pts
            midpoints = (wire_pts + next_pts) / 2
        else:
            dx        = torch.diff(wire_pts, dim=0)
            midpoints = (wire_pts[:-1] + wire_pts[1:]) / 2

        for n1 in tqdm(range(0, lim, self.wire_seg_batch),
                       desc='Inductance (midpoint legacy)',
                       disable=not self.verbose):
            n2        = min(n1 + self.wire_seg_batch, lim)
            dx1_batch = dx[n1:n2]
            x1_batch  = midpoints[n1:n2]

            dist     = torch.cdist(x1_batch, midpoints)
            dot_prod = torch.einsum('ni,mi->nm', dx1_batch, dx)

            idxs = torch.arange(n1, n2, device=wire_pts.device)
            dot_prod[idxs - n1, idxs] = 0.0
            dist    [idxs - n1, idxs] = 1.0

            inductance += MU0 * (dot_prod / dist).sum() / (4 * torch.pi)

            lx1         = dx1_batch.norm(dim=-1)
            inductance += (MU0 * lx1 *
                           (torch.log(2 * lx1 / self.wire_rad) - 0.75) /
                           (2 * torch.pi)).sum()

        return inductance.item()

    # -- Magnetic vector potential ---------------------------------------------

    def calc_mag_potential(self,
                           spatial_crds: torch.Tensor) -> torch.Tensor:
        """
        Magnetic vector potential produced by unit current via Biot-Savart:

            A(r) = (mu0/4pi) * int dl / |r - r'|

        Batches over wire segments and accumulates contributions from
        potential_kernel_midpoint.

        Args:
        -----
        spatial_crds : torch.Tensor
            Spatial coordinates with shape (..., 3) [m]
            
        Returns:
        --------
        torch.Tensor:
            Magnetic vector potential with shape (..., 3) [V*s/m]
        """
        wire_pts = self.wire_pts
        N        = wire_pts.shape[0]
        crds     = spatial_crds.reshape((-1, 3))

        # Placeholder for output
        mag_potential = torch.zeros((crds.shape[0], 3), device=crds.device, dtype=crds.dtype)

        next_pts  = torch.roll(wire_pts, shifts=-1, dims=0)
        ds        = next_pts - wire_pts       # [N, 3]
        midpoints = (next_pts + wire_pts) / 2 # [N, 3]
        lim       = N if self.closed else N - 1

        # Batch over wire segments
        for n1 in tqdm(range(0, lim, self.wire_seg_batch),
                       'Calculating magnetic potential',
                       disable=not self.verbose):
            n2 = min(n1 + self.wire_seg_batch, lim)

            ds_batch  = ds[n1:n2]        # [B, 3]
            mid_batch = midpoints[n1:n2] # [B, 3]

            mag_potential += self.potential_kernel_midpoint(ds_batch, mid_batch, crds)

        return mag_potential.reshape(spatial_crds.shape[:-1] + (3,))

    # -- Magnetic field --------------------------------------------------------

    def calc_bfield(self,
                    spatial_crds: torch.Tensor,
                    mode:         int = 0,
                    kernel:       str = 'midpoint',
                    n_pts:        int = 10) -> torch.Tensor:
        """
        Magnetic field produced by unit current via Biot-Savart:

            B(r) = (mu0/4pi) * int (dl x r_hat) / |r - r'|^2

        Batches over wire segments and accumulates contributions from
        the selected bfield kernel.

        Args:
        -----
        spatial_crds : torch.Tensor
            Spatial coordinates with shape (..., 3) [m]
        mode : int
            Integration method:
                0 - Standard Biot-Savart (midpoint or gauss depending on kernel)
                1 - Midpoint rule with shifted vertices, O(h^4) convergence
                    (not yet implemented)
                2 - Trapezoidal rule, O(h^2)/exponential for periodic paths
                    (not yet implemented)
        kernel : str
            'midpoint' - midpoint rule, O(h^2), no quadrature order
            'gauss'    - Gauss-Legendre quadrature, exponentially convergent
                         in n_pts for field points away from the wire
        n_pts : int
            Number of Gauss-Legendre quadrature points. Only used when
            kernel='gauss'.

        Returns:
        --------
        torch.Tensor:
            Magnetic field with shape (..., 3) [T/A]
        """
        kernel_map = {
            'midpoint' : lambda ds, mid, crds: self.bfield_kernel_midpoint(ds, mid, crds),
            'gauss'    : lambda ds, mid, crds: self.bfield_kernel_gauss(ds, mid, crds, n_pts=n_pts),
        }
        if kernel not in kernel_map:
            raise ValueError(
                f"kernel must be one of {list(kernel_map)}, got '{kernel}'"
            )
        kernel_fn = kernel_map[kernel]

        wire_pts = self.wire_pts
        N        = wire_pts.shape[0]
        crds     = spatial_crds.reshape((-1, 3))

        # Placeholder for output
        bfield = torch.zeros((crds.shape[0], 3), device=crds.device, dtype=crds.dtype)

        if mode == 0:
            # Batch over wire segments
            for n1 in tqdm(range(0, N - 1, self.wire_seg_batch),
                           'Calculating B-field',
                           disable=not self.verbose):
                n2 = min(n1 + self.wire_seg_batch, N - 1)

                ds_batch  = wire_pts[n1+1:n2+1] - wire_pts[n1:n2]       # [B, 3]
                mid_batch = (wire_pts[n1+1:n2+1] + wire_pts[n1:n2]) / 2 # [B, 3]

                bfield += kernel_fn(ds_batch, mid_batch, crds)

        elif mode == 1:
            pass
            # wire_pts = self._shifted_vertices(wire_pts)

        elif mode == 2:
            pass
            #### IMPORTANT:
            # We assume that sampled points are uniform sampled in phi.
            ####
            # Angular coordinate transform
            # r' = (a*cos(phi),b*sin(phi),0); dl = (-a*sin(phi), b*cos(phi),0)*dphi

            # transformed r'= Rr' + t WE ARE GIVEN (x,y,z) and phi_n ordering is implicit by creation
            # transformed dl = R@dl WE MUST SAVE or recompute using FFT

            # eval integrand at every point
            # dl = dl(phi_i) #r'(phi_i) can be written as weighted permutation 
            # rdiff = r-r'(phi_i) 
            # f(phi_i) = dl(phi_i) cross rdiff  / L2(rdiff)^3

            # first term is numerical dphi
            # Integral = 2*pi/N * (1/2 * f(phi_0) + 1/2 * f(phi_n) + sum_i=1:n-1 f(phi_i))

            # for n1 in tqdm(range(0, N-1, self.wire_seg_batch),
            #             'Calculating B-field Trapezoidal rule',
            #             disable=not self.verbose): 
            #     n2 = min(n1 + self.wire_seg_batch, N-1)

        return bfield.reshape(spatial_crds.shape[:-1] + (3,))

    # -- Magnetic field Jacobian -----------------------------------------------

    def calc_bfield_jacobian(self,
                             spatial_crds: torch.Tensor,
                             kernel:       str = 'midpoint',
                             n_pts:        int = 10) -> torch.Tensor:
        """
        Jacobian of the magnetic field produced by unit current:

            J_ij = dB_i/dx_j

        Batches over wire segments and accumulates contributions from
        the selected jacobian kernel.

        Args:
        -----
        spatial_crds : torch.Tensor
            Spatial coordinates with shape (..., 3) [m]
        kernel : str
            'midpoint' - midpoint rule, O(h^2), no quadrature order
            'gauss'    - Gauss-Legendre quadrature, exponentially convergent
                         in n_pts for field points away from the wire
        n_pts : int
            Number of Gauss-Legendre quadrature points. Only used when
            kernel='gauss'.
        
        Returns:
        --------
        torch.Tensor:
            Field Jacobian with shape (..., 3, 3) [T/(A*m)]
            Index convention: out[..., i, j] = dB_i/dx_j
        """
        kernel_map = {
            'midpoint' : lambda ds, mid, crds: self.jacobian_kernel_midpoint(ds, mid, crds),
            'gauss'    : lambda ds, mid, crds: self.jacobian_kernel_gauss(ds, mid, crds, n_pts=n_pts),
        }
        if kernel not in kernel_map:
            raise ValueError(
                f"kernel must be one of {list(kernel_map)}, got '{kernel}'"
            )
        kernel_fn = kernel_map[kernel]

        wire_pts = self.wire_pts
        N        = wire_pts.shape[0]
        crds     = spatial_crds.reshape((-1, 3))

        # Placeholder for output
        bfield_jacobian = torch.zeros((crds.shape[0], 3, 3), device=crds.device, dtype=crds.dtype)

        # Batch over wire segments
        for n1 in tqdm(range(0, N - 1, self.wire_seg_batch),
                       'Calculating B-field Jacobian',
                       disable=not self.verbose):
            n2 = min(n1 + self.wire_seg_batch, N - 1)

            ds_batch  = wire_pts[n1+1:n2+1] - wire_pts[n1:n2]       # [B, 3]
            mid_batch = (wire_pts[n1+1:n2+1] + wire_pts[n1:n2]) / 2 # [B, 3]

            bfield_jacobian += kernel_fn(ds_batch, mid_batch, crds)

        return bfield_jacobian.reshape(spatial_crds.shape[:-1] + (3, 3))

    # -- Visualisation ---------------------------------------------------------

    def show_wire(self, fig=None, ax=None):
        """
        Display wire points on a 3D axis.

        Args:
        -----
        fig : Optional[matplotlib.figure.Figure]
            Existing figure handle.
        ax : Optional[matplotlib.axes.Axes]
            Existing 3D axis handle.

        Returns:
        --------
        tuple
            Figure and axis used for plotting.
        """
        wire_pts = self.wire_pts.cpu() * 1e2 # cm
        if fig is None or ax is None:
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection='3d')
        ax.plot(wire_pts[:, 0], wire_pts[:, 1], wire_pts[:, 2],
                alpha=.8, linewidth=0.5, color='red')
        ax.set_xlabel('X [cm]')
        ax.set_ylabel('Y [cm]')
        ax.set_zlabel('Z [cm]')
        plt.axis('equal')
        return fig, ax


class elliptical_wire(parametric_wire):
    """Class defining Elliptical wire as one can use it's known derivatives"""
    def __init__(self, 
                 num_pts: int,
                 a: float,
                 b: float,
                 dphi: Optional[torch.Tensor] = None, 
                 R: Optional[torch.Tensor] = None,
                 t: Optional[torch.Tensor] = None,
                 wire_seg_batch: Optional[int] = None,
                 verbose: Optional[bool] = None,
                 ):
        """ 
        Assumptions:
        -----
        Ellipse before rot + trans is parameterized (a*cos(phi), b*sin(phi), 0)
        Thus, semi-major axis(a) aligns with x-axis and points are placed counter-clockwise(intersecting +y-axis first)
        -----
        
        Args:
        -----
        num_pts : int 
            Number of points to describe the ellipse 
        a : float
            Radius of semi-major axis of ellipse [m]
        b : float
            Radius of semi-minor axis of ellipse [m]
        dphi: Optional[torch.Tensor]
            Sampling locations of num_pts w.r.t phi [rads] 
        R : Optional[torch.Tensor]
            Rotation applied on wire points and derivative vectors [Lie Group SO(3)]
        t : Optional[torch.Tensor]
            Translation applied on wire points [m] 
        wire_seg_batch : Optional[int]
            Batching over the N wire segments for calculations
        verbose : Optional[bool]
            If True, prints progress during calculations.

        Returns:
        --------
        None
            Initializes elliptical wire geometry and kernels.
        """
        if dphi is None:
            t = torch.as_tensor(t, dtype=torch.float64)   # or whatever dtype you use
            dphi = (2 * torch.pi) * torch.arange(num_pts, device=t.device, dtype=t.dtype) / num_pts
        if R is None:
            R = torch.eye(3)
        if t is None:
            t = torch.zeros(3)

        # Store transform so we can use the non-discretized ellipse kernels.
        self.R = R
        self.t = t

        self.a    = a
        self.b    = b
        self.dphi = dphi
        
        x = self.a * torch.cos(dphi) # [N,]
        y = self.b * torch.sin(dphi) # [N,]
        z = torch.zeros(num_pts)     # [N,]

        dphi_x = -a/b * y # [N,]
        dphi_y =  b/a * x # [N,]
        dphi_z = z         # [N,]

        pos      = torch.stack([x, y, z],                 dim=-1) # [N, 3]
        tangents = torch.stack([dphi_x, dphi_y, dphi_z], dim=-1) # [N, 3]

        # Einsum for clarity.
        transformed_pos    = torch.einsum('ni,ji->nj', pos,      R) + t
        self.wire_tangents = torch.einsum('ni,ji->nj', tangents, R)
        
        super().__init__(transformed_pos,
                         wire_seg_batch=wire_seg_batch,
                         verbose=verbose)
    
        # --- Non-discretized ellipse kernels ---------------------------------
        # Use full-loop Gauss quadrature in phi rather than summing straight segments.
        # This is typically much more accurate per function evaluation for smooth fields
        # away from the filament.
        self._ellipse_kernel = _elliptical_bfield_kernel()

    def calc_mag_potential(self,
                           spatial_crds: torch.Tensor,
                           kernel: str = 'ellipse_gauss',
                           n_pts: int = 64) -> torch.Tensor:
        """
        Compute magnetic vector potential for an elliptical wire.

        Args:
        -----
        spatial_crds : torch.Tensor
            Evaluation coordinates with shape (..., 3).
        kernel : str
            Integration kernel name.
        n_pts : int
            Quadrature points for ellipse_gauss kernel.

        Returns:
        --------
        torch.Tensor
            Magnetic vector potential with shape (..., 3).
        """
        if kernel in ('ellipse_gauss', 'ellipse', 'parametric_gauss'):
            return self._ellipse_kernel.full_potential_kernel_gauss(
                spatial_crds, a=self.a, b=self.b, center=self.t, R=self.R, n_pts=n_pts
            )
        return super().calc_mag_potential(spatial_crds)

    def calc_bfield(self,
                    spatial_crds: torch.Tensor,
                    mode: int = 0,
                    kernel: str = 'ellipse_gauss',
                    n_pts: int = 64) -> torch.Tensor:
        """
        Compute magnetic field for an elliptical wire.

        Args:
        -----
        spatial_crds : torch.Tensor
            Evaluation coordinates with shape (..., 3).
        mode : int
            Parent integration mode when non-ellipse kernel is used.
        kernel : str
            Integration kernel name.
        n_pts : int
            Quadrature points for ellipse_gauss kernel.

        Returns:
        --------
        torch.Tensor
            Magnetic field with shape (..., 3).
        """
        if kernel in ('ellipse_gauss', 'ellipse', 'parametric_gauss'):
            return self._ellipse_kernel.full_bfield_kernel_gauss(
                spatial_crds, a=self.a, b=self.b, center=self.t, R=self.R, n_pts=n_pts
            )
        return super().calc_bfield(spatial_crds, mode=mode, kernel=kernel, n_pts=n_pts)

    def calc_bfield_jacobian(self,
                             spatial_crds: torch.Tensor,
                             kernel: str = 'ellipse_gauss',
                             n_pts: int = 64) -> torch.Tensor:
        """
        Compute magnetic field Jacobian for an elliptical wire.

        Args:
        -----
        spatial_crds : torch.Tensor
            Evaluation coordinates with shape (..., 3).
        kernel : str
            Integration kernel name.
        n_pts : int
            Quadrature points for ellipse_gauss kernel.

        Returns:
        --------
        torch.Tensor
            Field Jacobian with shape (..., 3, 3).
        """
        if kernel in ('ellipse_gauss', 'ellipse', 'parametric_gauss'):
            return self._ellipse_kernel.full_jacobian_kernel_gauss(
                spatial_crds, a=self.a, b=self.b, center=self.t, R=self.R, n_pts=n_pts
            )
        return super().calc_bfield_jacobian(spatial_crds, kernel=kernel, n_pts=n_pts)

    @property
    def area(self):
        """
        NON-DISCRETIZED
        Exact Area enclosed by the elliptical wire [m^2]

        Args:
        -----
        None
            Property getter with no input arguments.

        Returns:
        --------
        torch.Tensor
            Exact enclosed area.
        """
        return torch.pi * self.a * self.b
    
    @property
    def length(self):
        """
        NON-DISCRETIZED
        Ramanujan's approximate perimeter of ellipse [m].
        Converges to circle exactly.
        If eccentricity(b/a) is 0.01 -> 0.01% accuracy(good enough)

        Args:
        -----
        None
            Property getter with no input arguments.

        Returns:
        --------
        torch.Tensor
            Approximate ellipse perimeter.
        """
        h = ((self.a - self.b)**2) / ((self.a + self.b)**2)
        return torch.pi * (self.a + self.b) * (1 + 3*h / (10 + torch.sqrt(4 - 3*h)))
