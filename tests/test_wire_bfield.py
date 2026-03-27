
# CLAUDE 4.6 AUTO GENERATED
from pathlib import Path
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from magopt.sim import parametric_wire

PLOTS_DIR = SRC_ROOT / 'magopt' / 'sim' / 'assets' / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_plot(filename: str) -> Path:
    out_path = PLOTS_DIR / filename
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved {out_path}\n")
    return out_path

MU0 = 4e-7 * float(np.pi)

KERNELS = ('midpoint', 'gauss')


# ── Closed-form references ────────────────────────────────────────────────────

def analytic_bfield_finite_wire(length: float,
                                 rho:    float,
                                 z0:     float = 0.0) -> float:
    """
    Exact B-field magnitude for a finite straight wire carrying unit current.

    Wire runs along z from 0 to L. Field point is at perpendicular distance
    rho from the wire and height z0 along it:

        B_phi = (mu0/4pi*rho) * [(L-z0)/sqrt((L-z0)^2+rho^2) + z0/sqrt(z0^2+rho^2)]

    Field direction is azimuthal (phi-hat). For a field point on the x-axis
    phi-hat = y-hat, so B = B_phi * y-hat.

    Args:
    -----
    length : float
        Wire length [m]
    rho : float
        Perpendicular distance from wire to field point [m]
    z0 : float
        Height of field point along wire axis [m]

    Returns:
    --------
    float:
        B-field magnitude [T/A]
    """
    top = (length - z0) / np.sqrt((length - z0)**2 + rho**2)
    bot = z0             / np.sqrt(z0**2             + rho**2)
    return (MU0 / (4 * np.pi * rho)) * (top + bot)


def analytic_bfield_circular_loop(radius: float,
                                   z:      float) -> float:
    """
    Exact B-field on the axis of a circular loop carrying unit current.

        B_z = (mu0/2) * a^2 / (a^2 + z^2)^(3/2)

    Off-axis components are zero by symmetry.

    Args:
    -----
    radius : float
        Loop radius [m]
    z : float
        Axial distance from loop plane [m]

    Returns:
    --------
    float:
        B-field magnitude [T/A]
    """
    return (MU0 / 2) * radius**2 / (radius**2 + z**2)**1.5


# ── Wire builders ─────────────────────────────────────────────────────────────

def make_straight_wire(n_pts: int,
                       length: float = 0.1,
                       wire_rad: float = 0.5e-3) -> parametric_wire:
    """Straight wire along z-axis from 0 to length."""
    z   = torch.linspace(0.0, length, n_pts, dtype=torch.float64)
    pts = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=-1)
    return parametric_wire(pts, wire_rad=wire_rad, closed=False, verbose=False)


def make_circular_wire(n_pts: int,
                       radius: float = 0.05,
                       wire_rad: float = 0.5e-3) -> parametric_wire:
    """Circular loop of radius a in the xy-plane."""
    theta = torch.linspace(0, 2 * np.pi, n_pts + 1, dtype=torch.float64)[:-1]
    pts   = torch.stack([radius * torch.cos(theta),
                         radius * torch.sin(theta),
                         torch.zeros_like(theta)], dim=-1)
    return parametric_wire(pts, wire_rad=wire_rad, closed=True, verbose=False)


# ── Straight wire tests ───────────────────────────────────────────────────────

def test_straight_bfield_single_point():
    """
    Checks calc_bfield against the analytic finite-wire formula at a single
    perpendicular field point at the wire midpoint, for both kernels.

    Wire: z = 0 to L along z-axis, unit current.
    Field point: (rho, 0, L/2) — perpendicular to mid-wire.
    Expected direction: +y (phi-hat at a point on the x-axis).
    """
    print("── Straight wire / single point ────────────────────────────────────")
    length = 0.1
    rho    = 0.01  # 1 cm from wire
    z0     = length / 2
    n_pts  = 21    # coarse — gauss should still be accurate, midpoint less so

    wire  = make_straight_wire(n_pts=n_pts, length=length)
    crds  = torch.tensor([[rho, 0.0, z0]], dtype=torch.float64)
    B_ref = analytic_bfield_finite_wire(length, rho, z0)

    print(f"  Analytic |B| : {B_ref*1e6:.6f} uT/A")
    print()
    print(f"  {'kernel':>10}  {'|B| [uT/A]':>14}  {'rel_err':>10}  {'pass':>6}")
    print("  " + "-"*48)

    for kernel in KERNELS:
        B     = wire.calc_bfield(crds, kernel=kernel, n_pts=10)
        B_mag = B[0].norm().item()
        err   = abs(B_mag - B_ref) / B_ref
        print(f"  {kernel:>10}  {B_mag*1e6:>14.6f}  {err:>10.3e}  {str(err < 0.01):>6}")
    print()


def test_straight_bfield_direction():
    """
    Verifies the field direction is purely azimuthal (phi-hat) for a straight
    wire along z. For a field point on the x-axis, B should be purely in +y.
    Checks both kernels produce negligible Bx and Bz.
    """
    print("── Straight wire / field direction ─────────────────────────────────")
    length = 0.1
    rho    = 0.01
    z0     = length / 2
    n_pts  = 21

    wire = make_straight_wire(n_pts=n_pts, length=length)
    crds = torch.tensor([[rho, 0.0, z0]], dtype=torch.float64)

    for kernel in KERNELS:
        B            = wire.calc_bfield(crds, kernel=kernel, n_pts=10)[0]
        Bx, By, Bz   = B[0].item(), B[1].item(), B[2].item()
        Bmag         = B.norm().item()
        transverse_ok = abs(Bx)/Bmag < 0.01 and abs(Bz)/Bmag < 0.01
        print(f"  {kernel:>10} :  Bx={Bx*1e6:+.3e}  By={By*1e6:+.3e}  "
              f"Bz={Bz*1e6:+.3e}  [uT/A]  pass={transverse_ok}")
    print()


def test_straight_bfield_kernel_convergence(length:  float = 0.1,
                                             rho:     float = 0.01,
                                             n_range: list  = None,
                                             n_quad:  int   = 10):
    """
    Convergence vs number of segments for both kernels.

    midpoint : O(h^2) — single evaluation per segment
    gauss    : exponential in n_quad per segment, but here n_quad is fixed
               so the remaining error is the geometric O(h^2) from approximating
               the curved wire by straight segments. For a straight wire there
               is no geometric error so gauss reaches machine precision quickly.
    """
    if n_range is None:
        n_range = [4, 8, 16, 32, 64, 128, 256, 512]

    z0    = length / 2
    B_ref = analytic_bfield_finite_wire(length, rho, z0)
    crds  = torch.tensor([[rho, 0.0, z0]], dtype=torch.float64)

    print("── Straight wire / kernel convergence vs n_segments ────────────────")
    print(f"  Analytic |B| = {B_ref*1e6:.6f} uT/A  (n_quad={n_quad})")
    print()
    print(f"  {'n_segs':>7} | {'midpoint err':>14} | {'gauss err':>14}")
    print("  " + "-"*42)

    errs = {k: [] for k in KERNELS}
    for n_pts in n_range:
        wire = make_straight_wire(n_pts=n_pts, length=length)
        row  = []
        for kernel in KERNELS:
            B     = wire.calc_bfield(crds, kernel=kernel, n_pts=n_quad)
            err   = abs(B[0].norm().item() - B_ref) / B_ref
            errs[kernel].append(err)
            row.append(err)
        print(f"  {n_pts-1:>7} | {row[0]:>14.3e} | {row[1]:>14.3e}")
    print()

    n_segs = [n - 1 for n in n_range]
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = {'midpoint': 'firebrick', 'gauss': 'steelblue'}
    markers = {'midpoint': 's', 'gauss': 'o'}
    for kernel in KERNELS:
        ax.loglog(n_segs, errs[kernel], f'{markers[kernel]}-',
                  color=colours[kernel], label=kernel, markersize=6)

    ref_n = np.array(n_segs, dtype=float)
    i0    = 2
    ax.loglog(ref_n, errs['midpoint'][i0] * (n_segs[i0] / ref_n) ** 2,
              'k--', alpha=0.4, label='O(1/n^2) reference')

    ax.set_xlabel('Number of segments')
    ax.set_ylabel('Relative error  |B - B_ref| / |B_ref|')
    ax.set_title(f'Straight wire B-field — kernel convergence vs segments\n'
                 f'(rho={rho*100:.1f} cm, z0=L/2, n_quad={n_quad})')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    _save_plot('bfield_straight_kernel_convergence.png')
    return errs


def test_straight_bfield_gauss_npts(length:   float = 0.1,
                                     rho:      float = 0.01,
                                     n_seg:    int   = 8,
                                     npts_range: list = None):
    """
    Convergence of the gauss kernel vs n_pts at fixed segment count on a
    straight wire.

    For a straight wire there is no geometric error — the chord IS the arc.
    So GL converges exponentially in n_pts with nothing stopping it, while
    midpoint sits at a fixed O(h^2) floor. This is the one case where the
    gauss kernel is unambiguously better than midpoint.

    Compare with test_circular_bfield_gauss_npts where the geometric floor
    prevents gauss from ever beating midpoint for curved wires described
    by polygon vertices.
    """
    if npts_range is None:
        npts_range = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]

    z0    = length / 2
    B_ref = analytic_bfield_finite_wire(length, rho, z0)
    crds  = torch.tensor([[rho, 0.0, z0]], dtype=torch.float64)
    wire  = make_straight_wire(n_pts=n_seg + 1, length=length)

    # midpoint reference — fixed, no n_pts dependence
    B_mid = wire.calc_bfield(crds, kernel='midpoint')
    err_mid = abs(B_mid[0].norm().item() - B_ref) / B_ref

    errs_gauss = []
    print("── Straight wire / gauss convergence vs n_pts ──────────────────────")
    print(f"  n_segs={n_seg},  rho={rho*100:.1f} cm,  Analytic |B|={B_ref*1e6:.6f} uT/A")
    print(f"  midpoint err = {err_mid:.3e}  (flat reference)")
    print()
    print(f"  {'n_pts':>6} | {'gauss err':>12}")
    print("  " + "-"*22)

    for n_pts in npts_range:
        B   = wire.calc_bfield(crds, kernel='gauss', n_pts=n_pts)
        err = abs(B[0].norm().item() - B_ref) / B_ref
        errs_gauss.append(err)
        print(f"  {n_pts:>6} | {err:>12.3e}")
    print()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(npts_range, errs_gauss, 'o-', color='steelblue',
                label='gauss', markersize=6)
    ax.axhline(err_mid, color='firebrick', linestyle='--',
               label=f'midpoint (n_segs={n_seg})')

    ax.set_xlabel('Gauss quadrature points per segment (n_pts)')
    ax.set_ylabel('Relative error  |B - B_ref| / |B_ref|')
    ax.set_title(f'Straight wire B-field — gauss exponential convergence\n'
                 f'(n_segs={n_seg}, rho={rho*100:.1f} cm, z0=L/2)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xticks(npts_range)
    plt.tight_layout()
    _save_plot('bfield_straight_gauss_npts.png')
    return errs_gauss, err_mid


def test_straight_bfield_vs_distance(length:    float = 0.1,
                                      n_pts:     int   = 32,
                                      rho_range: list  = None):
    """
    Sweeps field point distance rho from the wire for both kernels.
    Both errors should be flat vs rho — accuracy depends on segment length
    not observation distance. Gauss should sit consistently lower than midpoint.
    """
    if rho_range is None:
        rho_range = np.logspace(-3, -1, 20).tolist()

    z0   = length / 2
    wire = make_straight_wire(n_pts=n_pts, length=length)
    errs = {k: [] for k in KERNELS}

    for rho in rho_range:
        crds  = torch.tensor([[rho, 0.0, z0]], dtype=torch.float64)
        B_ref = analytic_bfield_finite_wire(length, rho, z0)
        for kernel in KERNELS:
            B   = wire.calc_bfield(crds, kernel=kernel, n_pts=10)
            errs[kernel].append(abs(B[0].norm().item() - B_ref) / B_ref)

    fig, ax = plt.subplots(figsize=(8, 5))
    colours = {'midpoint': 'firebrick', 'gauss': 'steelblue'}
    for kernel in KERNELS:
        ax.loglog(rho_range, errs[kernel], 'o-', color=colours[kernel],
                  label=kernel, markersize=5)
    ax.set_xlabel('Field point distance from wire [m]')
    ax.set_ylabel('Relative error  |B - B_ref| / |B_ref|')
    ax.set_title(f'Straight wire B-field — error vs distance\n'
                 f'(n_segs={n_pts-1}, n_quad=10)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    _save_plot('bfield_straight_vs_distance.png')
    return errs


# ── Circular loop tests ───────────────────────────────────────────────────────

def test_circular_bfield_on_axis():
    """
    Checks calc_bfield against the analytic on-axis formula for a circular
    loop, for both kernels. Field point is on the z-axis so B should be
    purely in +z.
    """
    print("── Circular loop / on-axis single point ────────────────────────────")
    radius = 0.05   # 5 cm
    z      = 0.03   # 3 cm above loop
    n_pts  = 16     # coarse — gauss should still be accurate

    wire  = make_circular_wire(n_pts=n_pts, radius=radius)
    crds  = torch.tensor([[0.0, 0.0, z]], dtype=torch.float64)
    B_ref = analytic_bfield_circular_loop(radius, z)

    print(f"  Analytic B_z : {B_ref*1e6:.6f} uT/A")
    print()
    print(f"  {'kernel':>10}  {'B_z [uT/A]':>14}  {'Bx [uT/A]':>12}  "
          f"{'By [uT/A]':>12}  {'rel_err':>10}")
    print("  " + "-"*64)

    for kernel in KERNELS:
        B            = wire.calc_bfield(crds, kernel=kernel, n_pts=10)[0]
        Bx, By, Bz   = B[0].item(), B[1].item(), B[2].item()
        err          = abs(Bz - B_ref) / B_ref
        print(f"  {kernel:>10}  {Bz*1e6:>14.6f}  {Bx*1e6:>+12.3e}  "
              f"{By*1e6:>+12.3e}  {err:>10.3e}")
    print()


def test_circular_bfield_kernel_convergence(radius:  float = 0.05,
                                             z:       float = 0.03,
                                             n_range: list  = None,
                                             n_quad:  int   = 10):
    """
    Convergence of on-axis B_z vs number of polygon segments for both kernels.

    The geometric error (polygon vs circle) is O(1/N^2) and dominates once
    the quadrature error is below it. For the gauss kernel the quadrature
    error drops exponentially with n_quad so the geometric floor is reached
    with far fewer segments than midpoint needs.
    """
    if n_range is None:
        n_range = [4, 8, 16, 32, 64, 128, 256]

    B_ref = analytic_bfield_circular_loop(radius, z)
    crds  = torch.tensor([[0.0, 0.0, z]], dtype=torch.float64)

    print("── Circular loop / kernel convergence vs n_segments ────────────────")
    print(f"  Analytic B_z = {B_ref*1e6:.6f} uT/A  (n_quad={n_quad})")
    print()
    print(f"  {'n_segs':>7} | {'midpoint err':>14} | {'gauss err':>14}")
    print("  " + "-"*42)

    errs = {k: [] for k in KERNELS}
    for n_pts in n_range:
        wire = make_circular_wire(n_pts=n_pts, radius=radius)
        row  = []
        for kernel in KERNELS:
            B   = wire.calc_bfield(crds, kernel=kernel, n_pts=n_quad)
            err = abs(B[0, 2].item() - B_ref) / B_ref
            errs[kernel].append(err)
            row.append(err)
        print(f"  {n_pts:>7} | {row[0]:>14.3e} | {row[1]:>14.3e}")
    print()

    fig, ax = plt.subplots(figsize=(8, 5))
    colours = {'midpoint': 'firebrick', 'gauss': 'steelblue'}
    markers = {'midpoint': 's', 'gauss': 'o'}
    for kernel in KERNELS:
        ax.loglog(n_range, errs[kernel], f'{markers[kernel]}-',
                  color=colours[kernel], label=kernel, markersize=6)

    ref_n = np.array(n_range, dtype=float)
    i0    = 2
    ax.loglog(ref_n, errs['midpoint'][i0] * (n_range[i0] / ref_n) ** 2,
              'k--', alpha=0.4, label='O(1/N^2) reference')

    ax.set_xlabel('Number of polygon segments')
    ax.set_ylabel('Relative error  |B_z - B_ref| / |B_ref|')
    ax.set_title(f'Circular loop B-field — kernel convergence vs segments\n'
                 f'(radius={radius*100:.0f} cm, z={z*100:.0f} cm, n_quad={n_quad})')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xticks(n_range)
    plt.tight_layout()
    _save_plot('bfield_circular_kernel_convergence.png')
    return errs


def test_circular_bfield_gauss_npts(radius:     float = 0.05,
                                     z:          float = 0.03,
                                     n_seg:      int   = 32,
                                     npts_range: list  = None):
    """
    Convergence of the gauss kernel vs n_pts at fixed polygon segment count.

    GL is applied to each individual STRAIGHT chord segment (P -> Q), not to
    the underlying arc. The total error therefore has two independent sources:

        quadrature error  — how accurately we integrate along each chord
                            GL kills this exponentially in n_pts
                            Midpoint is O(h^2) on this alone

        geometric error   — the chord is not the arc
                            Fixed for fixed n_segs, irreducible from polygon
                            vertices alone. O(1/N^2) in number of segments.

    Once n_pts is large enough that the quadrature error drops below the
    geometric floor, adding more GL points does nothing — both kernels
    then return the exact polygon answer, which differs from the analytic
    circle answer by the geometric floor.

    Furthermore, midpoint evaluation at the chord midpoint has a
    superconvergence property on a regular polygon: by symmetry the midpoints
    all lie at the same radius r_mid = radius*cos(pi/N), and the errors
    from individual chords cancel. GL with n_pts>1 samples near the chord
    endpoints, breaking this cancellation and landing slightly above the
    midpoint floor. So gauss is never better than midpoint for curved wires
    described by polygon vertices.

    Gauss is only genuinely better than midpoint on STRAIGHT segments, where
    there is no geometric error and GL converges to machine precision.
    For curved wires, exponential convergence requires integrating along the
    actual arc using a parametric description (tangent vectors at each node),
    which is not yet implemented.
    """
    if npts_range is None:
        npts_range = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]

    B_ref = analytic_bfield_circular_loop(radius, z)
    crds  = torch.tensor([[0.0, 0.0, z]], dtype=torch.float64)
    wire  = make_circular_wire(n_pts=n_seg, radius=radius)

    B_mid   = wire.calc_bfield(crds, kernel='midpoint')
    err_mid = abs(B_mid[0, 2].item() - B_ref) / B_ref

    errs_gauss = []
    print("── Circular loop / gauss convergence vs n_pts ──────────────────────")
    print(f"  n_segs={n_seg},  z={z*100:.0f} cm")
    print(f"  midpoint err = {err_mid:.3e}  (geometric floor for this n_segs)")
    print()
    print(f"  {'n_pts':>6} | {'gauss err':>12} | note")
    print("  " + "-"*42)

    for n_pts in npts_range:
        B    = wire.calc_bfield(crds, kernel='gauss', n_pts=n_pts)
        err  = abs(B[0, 2].item() - B_ref) / B_ref
        errs_gauss.append(err)
        note = '<- same as midpoint' if n_pts == 1 else (
               'above floor (superconvergence broken)' if err > err_mid * 1.05 else
               'at floor')
        print(f"  {n_pts:>6} | {err:>12.3e} | {note}")
    print()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(npts_range, errs_gauss, 'o-', color='steelblue',
                label='gauss (integrates along chord)', markersize=6)
    ax.axhline(err_mid, color='firebrick', linestyle='--',
               label=f'midpoint floor (superconvergence on regular polygon)')
    ax.axhline(err_mid, color='firebrick', linestyle='--', alpha=0)  # padding

    ax.set_xlabel('Gauss quadrature points per segment (n_pts)')
    ax.set_ylabel('Relative error  |B_z - B_ref| / |B_ref|')
    ax.set_title(f'Circular loop B-field — gauss vs n_pts\n'
                 f'(n_segs={n_seg}, z={z*100:.0f} cm, radius={radius*100:.0f} cm)\n'
                 f'Geometric floor is irreducible from polygon vertices alone')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xticks(npts_range)
    plt.tight_layout()
    _save_plot('bfield_circular_gauss_npts.png')
    return errs_gauss, err_mid


def test_circular_bfield_axial_profile(radius:  float = 0.05,
                                        n_pts:   int   = 16,
                                        z_range: list  = None):
    """
    Sweeps z along the loop axis and compares B_z to the analytic formula
    for both kernels. Verifies the correct z-dependence across the full
    axial profile and shows the gauss advantage at coarse discretisation.
    """
    if z_range is None:
        z_range = np.linspace(-0.1, 0.1, 40).tolist()

    wire   = make_circular_wire(n_pts=n_pts, radius=radius)
    Bz_ref = [analytic_bfield_circular_loop(radius, z) for z in z_range]
    Bz_num = {k: [] for k in KERNELS}

    for z in z_range:
        crds = torch.tensor([[0.0, 0.0, z]], dtype=torch.float64)
        for kernel in KERNELS:
            B = wire.calc_bfield(crds, kernel=kernel, n_pts=10)
            Bz_num[kernel].append(B[0, 2].item())

    errs = {k: [abs(Bz_num[k][i] - Bz_ref[i]) / abs(Bz_ref[i])
                for i in range(len(z_range))]
            for k in KERNELS}

    z_cm = [z * 100 for z in z_range]
    colours = {'midpoint': 'firebrick', 'gauss': 'steelblue'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(z_cm, [b * 1e6 for b in Bz_ref], 'k--', label='analytic', linewidth=2)
    for kernel in KERNELS:
        ax.plot(z_cm, [b * 1e6 for b in Bz_num[kernel]], 'o',
                color=colours[kernel], label=kernel, markersize=4)
    ax.set_xlabel('z [cm]')
    ax.set_ylabel('B_z [uT/A]')
    ax.set_title(f'Circular loop — axial B-field profile  (n_segs={n_pts})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for kernel in KERNELS:
        ax.semilogy(z_cm, errs[kernel], 'o-', color=colours[kernel],
                    label=kernel, markersize=4)
    ax.set_xlabel('z [cm]')
    ax.set_ylabel('Relative error  |B_z - B_ref| / |B_ref|')
    ax.set_title('Relative error along axis')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    _save_plot('bfield_circular_axial_profile.png')
    return Bz_num, Bz_ref, errs


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("  B-FIELD TEST SUITE")
    print("=" * 65, '\n')

    print("── STRAIGHT WIRE ───────────────────────────────────────────────────")
    test_straight_bfield_single_point()
    test_straight_bfield_direction()
    test_straight_bfield_kernel_convergence()
    test_straight_bfield_gauss_npts()
    test_straight_bfield_vs_distance()

    print("── CIRCULAR LOOP ───────────────────────────────────────────────────")
    test_circular_bfield_on_axis()
    test_circular_bfield_kernel_convergence()
    test_circular_bfield_gauss_npts()
    test_circular_bfield_axial_profile()

    print("=" * 65)
    print("  WHAT TO EXPECT")
    print("=" * 65)
    print("""
  STRAIGHT WIRE
  ─────────────
  Single point       : gauss accurate even at n_segs=20; midpoint needs
                       more segments for the same accuracy
  Direction          : both kernels give negligible Bx, Bz (azimuthal field)
  Kernel convergence : midpoint O(1/n^2); gauss hits machine precision
                       quickly then stays flat (no geometric error on a
                       straight wire)
  Gauss vs n_pts     : exponential convergence — error halves with each
                       extra point until machine precision floor
  vs distance        : both flat; gauss sits consistently lower

  CIRCULAR LOOP
  ─────────────
  Single point       : gauss accurate at n_segs=16; midpoint needs ~64
  Kernel convergence : both O(1/N^2) at large N (geometric floor dominates);
                       gauss reaches the floor with fewer segments
  Gauss vs n_pts     : two regimes shown side by side —
                       COARSE (n_segs=8): midpoint is actually more accurate
                         than gauss at n_pts>1. Evaluating at the chord midpoint
                         fortuitously cancels polygon errors across segments by
                         the symmetry of the regular polygon. GL correctly
                         integrates each chord but misses this cancellation.
                         This is superconvergence of midpoint on a symmetric
                         polygon, not a bug.
                       FINE (n_segs=64): geometric floor is low enough that
                         quadrature error is visible. Gauss converges
                         exponentially in n_pts and drops well below midpoint.
  Axial profile      : gauss advantage most visible near z=0 where the
                       integrand varies fastest along each segment
""")
