from pathlib import Path
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings

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


# ── Closed-form references ────────────────────────────────────────────────────

def analytic_straight(length: float, wire_rad: float, frequency='dc') -> float:
    """
    Rosa-Neumann self-inductance for a straight wire.
        L = (μ₀l/2π)(ln(2l/r) - k)
    k = 0.75 (dc), k = 1.0 (hf)
    """
    k = 0.75 if frequency == 'dc' else 1.0
    return (MU0 * length / (2 * np.pi)) * (np.log(2 * length / wire_rad) - k)


def analytic_circular(radius: float, wire_rad: float, frequency='dc') -> float:
    """
    Neumann self-inductance for a circular loop.
        L = μ₀a(ln(8a/r) - 2 + Y/2)
    Y = 0.5 (dc), Y = 0.0 (hf). Valid when a >> r (thin wire).
    """
    Y = 0.5 if frequency == 'dc' else 0.0
    return MU0 * radius * (np.log(8 * radius / wire_rad) - 2 + Y / 2)


def analytic_adjacent_mutual(l: float) -> float:
    """
    Exact mutual inductance between two adjacent collinear equal segments.
    Grover f(x) = x*ln(x) with A=0, B=l, C=l, D=2l:
        M = (μ₀/4π) · [f(2l) + f(0) - f(l) - f(l)]
          = (μ₀/4π) · 2l·ln(2)
    """
    return (MU0 / (4 * np.pi)) * 2 * l * np.log(2)


# ── Wire builders ─────────────────────────────────────────────────────────────

def make_straight_wire(n_pts: int,
                       length: float = 0.1,
                       wire_rad: float = 0.5e-3) -> parametric_wire:
    """Straight wire along x-axis with n_pts points (n_pts-1 segments)."""
    x   = torch.linspace(0.0, length, n_pts, dtype=torch.float64)
    pts = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1)
    return parametric_wire(pts, wire_rad=wire_rad, closed=False, verbose=False)


def make_circular_wire(n_pts: int,
                       radius: float = 0.05,
                       wire_rad: float = 0.5e-3) -> parametric_wire:
    """Circular loop in the xy-plane approximated by n_pts polygon vertices."""
    theta = torch.linspace(0, 2 * np.pi, n_pts + 1, dtype=torch.float64)[:-1]
    pts   = torch.stack([radius * torch.cos(theta),
                         radius * torch.sin(theta),
                         torch.zeros_like(theta)], dim=-1)
    return parametric_wire(pts, wire_rad=wire_rad, closed=True, verbose=False)


# ── Straight wire tests ───────────────────────────────────────────────────────

def test_straight_single_segment():
    """
    1 segment = no mutual term at all. Both analytic_adjacent=True/False
    must return the Rosa-Neumann self-term exactly (to floating point).
    """
    print("── Straight / single segment ───────────────────────────────────────")
    length, wire_rad = 0.1, 0.5e-3
    L_ref  = analytic_straight(length, wire_rad, 'dc')
    wire   = make_straight_wire(n_pts=2, length=length, wire_rad=wire_rad)

    for aa in (True, False):
        L   = wire.calc_self_inductance(frequency='dc', analytic_adjacent=aa,
                                         warn_angle_deg=180)
        err = abs(L - L_ref) / abs(L_ref)
        print(f"  analytic_adjacent={str(aa):5s} : {L*1e9:.8f} nH  "
              f"(ref {L_ref*1e9:.8f} nH)  rel_err={err:.2e}")

    print()


def test_straight_decomposition_invariance():
    """
    A straight wire split into N collinear segments must give the same
    inductance regardless of N when analytic_adjacent=True, because the
    analytic kernel is exact for collinear pairs.

    With analytic_adjacent=False the result drifts with N due to the O(1/n²)
    quadrature floor on adjacent pairs.
    """
    print("── Straight / decomposition invariance ─────────────────────────────")
    length, wire_rad = 0.1, 0.5e-3
    L_ref  = analytic_straight(length, wire_rad, 'dc')
    n_list = [2, 3, 5, 10, 20, 50]

    print(f"  Analytic reference: {L_ref*1e9:.8f} nH")
    print()
    print(f"  {'n_segs':>7} | {'analytic_adj=True':>20} err | "
          f"{'analytic_adj=False':>20} err")
    print("  " + "-"*68)

    for n_pts in n_list:
        wire = make_straight_wire(n_pts=n_pts, length=length, wire_rad=wire_rad)

        L_on  = wire.calc_self_inductance(kernel='gauss', n_quad=10,
                                           frequency='dc', analytic_adjacent=True,
                                           warn_angle_deg=180)
        L_off = wire.calc_self_inductance(kernel='gauss', n_quad=10,
                                           frequency='dc', analytic_adjacent=False,
                                           warn_angle_deg=180)

        err_on  = abs(L_on  - L_ref) / abs(L_ref)
        err_off = abs(L_off - L_ref) / abs(L_ref)

        print(f"  {n_pts-1:>7} | {L_on*1e9:>16.8f} nH  {err_on:.2e} | "
              f"{L_off*1e9:>16.8f} nH  {err_off:.2e}")

    print()


def test_straight_kernel_independence():
    """
    With analytic_adjacent=True, adjacent pairs are always handled analytically.
    The quadrature kernel only affects the non-adjacent pairs.

    For a straight wire, non-adjacent pairs are well-separated collinear
    segments with a smooth integrand — no singularity. GL converges
    exponentially fast so n_quad=10 is effectively exact for those pairs.
    tanh_sinh behaves the same way on smooth integrands.

    midpoint is different: it is a single-point O(h²) rule per pair and
    does not converge to the same answer as GL at any fixed n_quad. So:
        gauss   ~= tanh_sinh  (both accurate on smooth non-adjacent pairs)
        midpoint != gauss      (crude approximation of non-adjacent pairs)

    The test confirms this split and also shows that the absolute error
    for gauss/tanh_sinh is ~machine precision — non-adjacent pairs are
    not the bottleneck once adjacent pairs are handled analytically.
    """
    print("── Straight / kernel independence ──────────────────────────────────")
    length, wire_rad = 0.1, 0.5e-3
    L_ref  = analytic_straight(length, wire_rad, 'dc')
    wire   = make_straight_wire(n_pts=6, length=length, wire_rad=wire_rad)

    results = {}
    for k in ('gauss', 'tanh_sinh', 'midpoint'):
        L = wire.calc_self_inductance(kernel=k, n_quad=10, frequency='dc',
                                       analytic_adjacent=True, warn_angle_deg=180)
        results[k] = L

    print(f"  Analytic reference: {L_ref*1e9:.8f} nH")
    print()
    for k, L in results.items():
        err = abs(L - L_ref) / abs(L_ref)
        print(f"  {k:12s} : {L*1e9:.8f} nH  rel_err={err:.2e}")

    # gauss and tanh_sinh should agree to near machine precision
    spread_quadrature = abs(results['gauss'] - results['tanh_sinh']) / L_ref
    # midpoint will differ — it is crude on the non-adjacent pairs
    spread_vs_mid = abs(results['gauss'] - results['midpoint']) / L_ref
    print(f"\n  gauss vs tanh_sinh spread : {spread_quadrature:.2e}  "
          f"(should be ~machine precision)")
    print(f"  gauss vs midpoint  spread : {spread_vs_mid:.2e}  "
          f"(midpoint is O(h²) on non-adjacent pairs — expected to differ)\n")


def test_straight_convergence(length: float = 0.1,
                               wire_rad: float = 0.5e-3,
                               n_range = None,
                               n_quad: int = 10):
    """
    Convergence vs number of segments with and without analytic_adjacent.
    With analytic_adjacent=True: flat at machine precision for a straight wire.
    With analytic_adjacent=False: O(1/n²) floor from adjacent quadrature error.
    """
    if n_range is None:
        n_range = [2, 3, 4, 5, 8, 11, 16, 21, 32, 51, 64]

    L_ref   = analytic_straight(length, wire_rad, 'dc')
    n_segs  = [n - 1 for n in n_range]
    err_on, err_off = [], []

    for n_pts in n_range:
        wire = make_straight_wire(n_pts=n_pts, length=length, wire_rad=wire_rad)
        L_on  = wire.calc_self_inductance(kernel='gauss', n_quad=n_quad,
                                           frequency='dc', analytic_adjacent=True,
                                           warn_angle_deg=180)
        L_off = wire.calc_self_inductance(kernel='gauss', n_quad=n_quad,
                                           frequency='dc', analytic_adjacent=False,
                                           warn_angle_deg=180)
        err_on .append(abs(L_on  - L_ref) / abs(L_ref))
        err_off.append(abs(L_off - L_ref) / abs(L_ref))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(n_segs, err_on,  'o-', color='steelblue',
                label='analytic_adjacent=True  (Grover exact)', markersize=6)
    ax.semilogy(n_segs, err_off, 's--', color='firebrick',
                label='analytic_adjacent=False (GL quadrature)', markersize=6)

    # O(1/n^2) slope for reference
    ref_n = np.array(n_segs, dtype=float)
    i0    = 3
    ax.semilogy(ref_n, err_off[i0] * (n_segs[i0] / ref_n) ** 2,
                'k:', alpha=0.4, label='O(1/n²) reference')

    ax.set_xlabel('Number of segments')
    ax.set_ylabel('Relative error  |L - L_ref| / |L_ref|')
    ax.set_title(f'Straight wire — convergence vs segments\n'
                 f'(n_quad={n_quad}, length={length*100:.0f} cm, '
                 f'r={wire_rad*1e3:.1f} mm)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xticks(n_segs)
    plt.tight_layout()
    _save_plot('straight_convergence.png')
    return err_on, err_off


# ── Circular wire tests ───────────────────────────────────────────────────────

def test_circular_single_segment():
    """
    Sanity check: a single-segment closed loop (degenerate — just one segment
    connecting back to itself) has no mutual term, so both modes agree.
    Not physically meaningful but confirms the code path is correct.
    """
    print("── Circular / 4-segment coarse check ──────────────────────────────")
    radius, wire_rad = 0.05, 0.5e-3
    L_ref = analytic_circular(radius, wire_rad, 'dc')

    # 4 segments — 90° angles, expect angle warning with analytic_adjacent
    for n_pts in (4, 8, 16, 32):
        wire = make_circular_wire(n_pts=n_pts, radius=radius, wire_rad=wire_rad)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            L_on = wire.calc_self_inductance(kernel='gauss', n_quad=10,
                                              frequency='dc',
                                              analytic_adjacent=True,
                                              warn_angle_deg=30.0)

        L_off = wire.calc_self_inductance(kernel='gauss', n_quad=10,
                                           frequency='dc',
                                           analytic_adjacent=False,
                                           warn_angle_deg=180)

        err_on  = abs(L_on  - L_ref) / abs(L_ref)
        err_off = abs(L_off - L_ref) / abs(L_ref)
        angle   = 360.0 / n_pts  # inter-segment angle [deg]
        warned  = any('neumann_kernel_analytic' in str(w.message) for w in caught)

        print(f"  n_pts={n_pts:3d}  angle={angle:5.1f}°  "
              f"on={err_on:.3e}  off={err_off:.3e}  "
              f"angle_warning={'yes' if warned else 'no '}")

    print()


def test_circular_convergence(radius: float = 0.05,
                               wire_rad: float = 0.5e-3,
                               n_range = None,
                               n_quad: int = 16):
    """
    With analytic_adjacent=True the error budget becomes purely geometric:
    polygon-approximates-circle, which converges as O(1/N²).

    With analytic_adjacent=False the error floor from adjacent quadrature
    dominates until N is large enough that geometric error takes over.
    """
    if n_range is None:
        n_range = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]

    L_ref    = analytic_circular(radius, wire_rad, 'dc')
    err_on   = []
    err_off  = []

    for n_pts in n_range:
        wire = make_circular_wire(n_pts=n_pts, radius=radius, wire_rad=wire_rad)
        L_on  = wire.calc_self_inductance(kernel='gauss', n_quad=n_quad,
                                           frequency='dc', analytic_adjacent=True,
                                           warn_angle_deg=180)
        L_off = wire.calc_self_inductance(kernel='gauss', n_quad=n_quad,
                                           frequency='dc', analytic_adjacent=False,
                                           warn_angle_deg=180)
        err_on .append(abs(L_on  - L_ref) / abs(L_ref))
        err_off.append(abs(L_off - L_ref) / abs(L_ref))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(n_range, err_on,  'o-', color='steelblue',
                label='analytic_adjacent=True  (geometric error only)', markersize=6)
    ax.semilogy(n_range, err_off, 's--', color='firebrick',
                label='analytic_adjacent=False (quadrature + geometric)', markersize=6)

    # O(1/N^2) slope anchored to the on-curve at N=32
    ref_n = np.array(n_range, dtype=float)
    i0    = n_range.index(32)
    ax.semilogy(ref_n, err_on[i0] * (n_range[i0] / ref_n) ** 2,
                'k:', alpha=0.4, label='O(1/N²) reference')

    ax.set_xlabel('Number of polygon segments')
    ax.set_ylabel('Relative error  |L - L_ref| / |L_ref|')
    ax.set_title(f'Circular loop — convergence vs segments\n'
                 f'(n_quad={n_quad}, r={radius*100:.0f} cm, '
                 f'r_wire={wire_rad*1e3:.1f} mm)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xticks(n_range)
    plt.tight_layout()
    _save_plot('circular_convergence.png')
    return err_on, err_off


def test_circular_angle_warning():
    """
    Confirms that neumann_kernel_analytic emits a RuntimeWarning when the
    inter-segment angle exceeds warn_angle_deg, and stays silent when it
    does not. Also checks the default threshold of 30°.

    For a regular N-gon the inter-segment angle is 360/N degrees.
    """
    print("── Circular / angle warning behaviour ──────────────────────────────")
    radius, wire_rad = 0.05, 0.5e-3

    # N=8: angle=45°  → should warn at default 30°, silent at 60°
    # N=16: angle=22.5° → silent at default 30°
    cases = [
        (8,  30.0, True,  "45° > 30°  → warn"),
        (8,  60.0, False, "45° < 60°  → silent"),
        (16, 30.0, False, "22.5° < 30° → silent"),
        (4,  30.0, True,  "90° > 30°  → warn"),
    ]

    for n_pts, threshold, expect_warn, label in cases:
        wire = make_circular_wire(n_pts=n_pts, radius=radius, wire_rad=wire_rad)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wire.calc_self_inductance(kernel='gauss', n_quad=10,
                                       frequency='dc', analytic_adjacent=True,
                                       warn_angle_deg=threshold)

        got_warn = any('neumann_kernel_analytic' in str(w.message)
                       for w in caught)
        status   = 'PASS' if got_warn == expect_warn else 'FAIL'
        print(f"  [{status}]  n_pts={n_pts:3d}  threshold={threshold:4.0f}°  "
              f"warned={str(got_warn):5s}  — {label}")

    print()


def test_circular_vs_radius(wire_rad: float = 0.5e-3,
                             n_pts: int = 32,
                             n_quad: int = 16):
    """
    Sweeps loop radius from 1 cm to 1 m.
    With analytic_adjacent=True the relative error should stay roughly flat
    (geometric error only, set by n_pts). Without it the error floor is
    higher and doesn't improve with radius.
    """
    radii   = np.logspace(-2, 0, 20).tolist()
    L_ref   = [analytic_circular(r, wire_rad, 'dc') for r in radii]
    err_on  = []
    err_off = []

    for i, r in enumerate(radii):
        wire = make_circular_wire(n_pts=n_pts, radius=r, wire_rad=wire_rad)
        L_on  = wire.calc_self_inductance(kernel='gauss', n_quad=n_quad,
                                           frequency='dc', analytic_adjacent=True,
                                           warn_angle_deg=180)
        L_off = wire.calc_self_inductance(kernel='gauss', n_quad=n_quad,
                                           frequency='dc', analytic_adjacent=False,
                                           warn_angle_deg=180)
        err_on .append(abs(L_on  - L_ref[i]) / abs(L_ref[i]))
        err_off.append(abs(L_off - L_ref[i]) / abs(L_ref[i]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(radii, err_on,  'o-', color='steelblue',
              label='analytic_adjacent=True', markersize=5)
    ax.loglog(radii, err_off, 's--', color='firebrick',
              label='analytic_adjacent=False', markersize=5)
    ax.set_xlabel('Loop radius [m]')
    ax.set_ylabel('Relative error  |L - L_ref| / |L_ref|')
    ax.set_title(f'Circular loop — relative error vs radius\n'
                 f'(n_segs={n_pts}, n_quad={n_quad})')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    _save_plot('circular_vs_radius.png')
    return err_on, err_off


# ── Side-by-side comparison ───────────────────────────────────────────────────

def test_side_by_side_summary():
    """
    Compact table comparing analytic_adjacent=True/False for the gauss kernel.
    midpoint is excluded here — its error comes from the non-adjacent pairs
    (O(h²) single-point rule) not the adjacent pairs, so it would obscure
    the improvement from analytic_adjacent.
    """
    print("── Side-by-side summary (gauss kernel) ─────────────────────────────")
    print()

    length,  wire_rad_s = 0.1,  0.5e-3
    radius,  wire_rad_c = 0.05, 0.5e-3

    rows = []

    # Straight wire at several discretisations
    for n_pts in (2, 4, 8, 16):
        wire  = make_straight_wire(n_pts=n_pts, length=length,
                                    wire_rad=wire_rad_s)
        L_ref = analytic_straight(length, wire_rad_s, 'dc')
        L_on  = wire.calc_self_inductance(kernel='gauss', n_quad=10,
                                           frequency='dc',
                                           analytic_adjacent=True,
                                           warn_angle_deg=180)
        L_off = wire.calc_self_inductance(kernel='gauss', n_quad=10,
                                           frequency='dc',
                                           analytic_adjacent=False,
                                           warn_angle_deg=180)
        rows.append(('straight', n_pts - 1,
                     abs(L_on  - L_ref) / abs(L_ref),
                     abs(L_off - L_ref) / abs(L_ref)))

    # Circular wire at several discretisations
    for n_pts in (8, 16, 32, 64):
        wire  = make_circular_wire(n_pts=n_pts, radius=radius,
                                    wire_rad=wire_rad_c)
        L_ref = analytic_circular(radius, wire_rad_c, 'dc')
        L_on  = wire.calc_self_inductance(kernel='gauss', n_quad=16,
                                           frequency='dc',
                                           analytic_adjacent=True,
                                           warn_angle_deg=180)
        L_off = wire.calc_self_inductance(kernel='gauss', n_quad=16,
                                           frequency='dc',
                                           analytic_adjacent=False,
                                           warn_angle_deg=180)
        rows.append(('circular', n_pts,
                     abs(L_on  - L_ref) / abs(L_ref),
                     abs(L_off - L_ref) / abs(L_ref)))

    print(f"  {'geometry':>10}  {'n_segs':>7}  "
          f"{'adj=True':>12}  {'adj=False':>12}  {'improvement':>12}")
    print("  " + "-"*60)
    for geom, n, e_on, e_off in rows:
        improvement = e_off / e_on if e_on > 1e-16 else float('inf')
        print(f"  {geom:>10}  {n:>7}  "
              f"{e_on:>12.3e}  {e_off:>12.3e}  {improvement:>10.1f}x")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("  ANALYTIC ADJACENT — TEST SUITE")
    print("=" * 65, '\n')

    print("── STRAIGHT WIRE ───────────────────────────────────────────────────")
    test_straight_single_segment()
    test_straight_decomposition_invariance()
    test_straight_kernel_independence()
    err_on_s, err_off_s = test_straight_convergence()

    print("── CIRCULAR WIRE ───────────────────────────────────────────────────")
    test_circular_single_segment()
    test_circular_angle_warning()
    err_on_c, err_off_c = test_circular_convergence()
    test_circular_vs_radius()

    print("── SUMMARY ─────────────────────────────────────────────────────────")
    test_side_by_side_summary()

    print("=" * 65)
    print("  WHAT TO EXPECT")
    print("=" * 65)
    print("""
  STRAIGHT WIRE
  ─────────────
  analytic_adjacent=True  : error < 1e-13 for any N — collinear pairs
                            are handled by the exact Grover formula, so
                            the result is independent of discretisation.
  analytic_adjacent=False : O(1/n²) floor from GL quadrature on adjacent
                            pairs — typically 1e-4 to 1e-3 depending on N.

  CIRCULAR WIRE
  ─────────────
  analytic_adjacent=True  : error is purely geometric (polygon vs circle),
                            converges as O(1/N²). Roughly 10-100x better
                            than without analytic adjacent for N < 64.
  analytic_adjacent=False : two competing error sources — quadrature floor
                            on adjacent pairs AND geometric discretisation.
                            The floor dominates until N is large enough
                            that geometric error drops below it.

  ANGLE WARNING
  ─────────────
  The analytic formula is exact only for collinear segments. For curved
  wires the error scales as O(sin²θ) where θ is the inter-segment angle
  (360/N degrees for a regular N-gon). The warning fires by default when
  θ > 30°, i.e. N < 12 segments. Suppress with warn_angle_deg=180.
""")
