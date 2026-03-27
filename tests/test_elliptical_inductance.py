from pathlib import Path
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from magopt.sim.physics.ellipse.inductance import elliptical_inductance

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

def wien_self_inductance(radius: float,
                         r_wire: float,
                         frequency: str = 'dc') -> float:
    """
    Wien's exact self-inductance for a thin circular loop:

        L = μ₀ R [ ln(8R/r) − k ]

    k = 7/4 (DC, uniform current — includes internal inductance)
    k = 2   (HF, skin effect — external inductance only)

    Assumes r_wire << radius.
    """
    k = 1.75 if frequency == 'dc' else 2.0
    return MU0 * radius * (np.log(8 * radius / r_wire) - k)


def maxwell_mutual_inductance(R1: float,
                               R2: float,
                               d: float) -> float:
    """
    Maxwell's exact mutual inductance between two coaxial circular loops
    of radii R1, R2 separated by axial distance d:

        M = μ₀ √(R1 R2) [ (2/k − k) K(k) − (2/k) E(k) ]

    where k² = 4 R1 R2 / ((R1+R2)² + d²), and K, E are complete
    elliptic integrals of the first and second kind.
    """
    from scipy.special import ellipk, ellipe

    k2 = 4 * R1 * R2 / ((R1 + R2)**2 + d**2)
    k = np.sqrt(k2)

    K = ellipk(k2)
    E = ellipe(k2)

    return MU0 * np.sqrt(R1 * R2) * ((2.0 / k - k) * K - (2.0 / k) * E)


# ── Helpers ───────────────────────────────────────────────────────────────────

solver = elliptical_inductance()


def _rotation_about_x(angle_deg: float,
                      dtype=torch.float64,
                      device=torch.device('cpu')) -> torch.Tensor:
    """Rotation matrix for angle about x-axis."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return torch.tensor([[1, 0, 0],
                         [0, c, -s],
                         [0, s,  c]], dtype=dtype, device=device)


def _rotation_about_z(angle_deg: float,
                      dtype=torch.float64,
                      device=torch.device('cpu')) -> torch.Tensor:
    """Rotation matrix for angle about z-axis."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return torch.tensor([[c, -s, 0],
                         [s,  c, 0],
                         [0,  0, 1]], dtype=dtype, device=device)


# ── Self-inductance tests ─────────────────────────────────────────────────────

def test_circular_self_inductance_single():
    """
    Checks self_inductance_gauss for a circular loop (a == b) against
    Wien's exact formula at a single set of parameters.

    Wien:  L = μ₀R [ln(8R/r) − k]
        DC:  k = 7/4  (includes internal inductance)
        HF:  k = 2    (external only)
    """
    print("── Circular self-inductance / single point ─────────────────────────")
    radius = 0.05
    r_wire = 0.5e-3

    for freq in ('dc', 'hf'):
        L_ref = wien_self_inductance(radius, r_wire, frequency=freq)
        L_num = solver.self_inductance_gauss(
            a=radius, b=radius, r_wire=r_wire,
            n_pts=80, frequency=freq).item()
        err = abs(L_num - L_ref) / abs(L_ref)
        print(f"  {freq:>3}:  Wien = {L_ref*1e9:.4f} nH    "
              f"Gauss = {L_num*1e9:.4f} nH    rel_err = {err:.3e}    "
              f"pass = {err < 0.01}")
    print()


def test_circular_self_inductance_vs_builtin():
    """
    Cross-checks self_inductance_gauss against circular_self_inductance_exact
    (both are methods on the solver). This validates internal consistency.
    """
    print("── Circular self-inductance / gauss vs exact method ────────────────")
    radii   = [0.01, 0.02, 0.05, 0.10, 0.20]
    r_wire  = 0.5e-3

    print(f"  {'radius [cm]':>12}  {'L_exact [nH]':>14}  {'L_gauss [nH]':>14}  "
          f"{'rel_err':>10}")
    print("  " + "-" * 58)

    for radius in radii:
        L_exact = solver.circular_self_inductance_exact(
            radius, r_wire).item()
        L_gauss = solver.circular_self_inductance_gauss(
            radius, r_wire, n_pts=80).item()
        err = abs(L_gauss - L_exact) / abs(L_exact)
        print(f"  {radius*100:>12.1f}  {L_exact*1e9:>14.4f}  "
              f"{L_gauss*1e9:>14.4f}  {err:>10.3e}")
    print()


def test_self_inductance_convergence_npts(radius:     float = 0.05,
                                          r_wire:     float = 0.5e-3,
                                          npts_range: list  = None):
    """
    Convergence of self_inductance_gauss vs n_pts for a circular loop.

    With singularity subtraction the smooth remainder is handled by GL
    which should converge exponentially. Without subtraction the peak
    at Δ=0 is of width δ/R and would need O(R/δ) points — thousands
    for typical thin-wire parameters.
    """
    if npts_range is None:
        npts_range = [10, 20, 30, 40, 60, 80, 100, 120, 160, 200]

    L_ref = wien_self_inductance(radius, r_wire)
    errs = []

    print("── Circular self-inductance / convergence vs n_pts ─────────────────")
    print(f"  radius = {radius*100:.1f} cm,  r_wire = {r_wire*1e3:.2f} mm")
    print(f"  Wien L = {L_ref*1e9:.4f} nH")
    print()
    print(f"  {'n_pts':>6} | {'L [nH]':>12} | {'rel_err':>12}")
    print("  " + "-" * 36)

    for n_pts in npts_range:
        L = solver.self_inductance_gauss(
            a=radius, b=radius, r_wire=r_wire, n_pts=n_pts).item()
        err = abs(L - L_ref) / abs(L_ref)
        errs.append(err)
        print(f"  {n_pts:>6} | {L*1e9:>12.6f} | {err:>12.3e}")
    print()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(npts_range, errs, 'o-', color='steelblue', markersize=6)
    ax.set_xlabel('Gauss-Legendre points (n_pts)')
    ax.set_ylabel('Relative error  |L − L_Wien| / |L_Wien|')
    ax.set_title(f'Circular self-inductance convergence vs n_pts\n'
                 f'(R = {radius*100:.1f} cm, r_wire = {r_wire*1e3:.2f} mm, '
                 f'singularity subtraction)')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    _save_plot('inductance_self_convergence_npts.png')
    return errs


def test_self_inductance_vs_wire_radius(radius:       float = 0.05,
                                        r_wire_range: list  = None):
    """
    Sweeps wire radius to verify the logarithmic dependence L ~ ln(1/r_wire).

    Wien's formula gives L = μ₀R[ln(8R/r) − 7/4], so plotting L vs ln(r)
    should be linear with slope −μ₀R. This tests that the GMD regularisation
    parameter correctly tracks the wire radius.

    NOTE: Wien assumes r_wire << radius. For thick wires the thin-wire
    approximation breaks down and the Gauss result will deviate.
    """
    if r_wire_range is None:
        # Keep r_wire < radius/10 to stay in thin-wire regime
        r_wire_range = np.logspace(-4, np.log10(radius / 10), 15).tolist()

    L_ref_list  = []
    L_num_list  = []
    errs        = []

    print("── Circular self-inductance / vs wire radius ───────────────────────")
    print(f"  radius = {radius*100:.1f} cm,  n_pts = 80")
    print()
    print(f"  {'r_wire [mm]':>12} | {'L_Wien [nH]':>14} | {'L_gauss [nH]':>14} | "
          f"{'rel_err':>10}")
    print("  " + "-" * 60)

    for r_wire in r_wire_range:
        L_ref = wien_self_inductance(radius, r_wire)
        L_num = solver.self_inductance_gauss(
            a=radius, b=radius, r_wire=r_wire, n_pts=80).item()
        err = abs(L_num - L_ref) / abs(L_ref)
        L_ref_list.append(L_ref)
        L_num_list.append(L_num)
        errs.append(err)
        print(f"  {r_wire*1e3:>12.4f} | {L_ref*1e9:>14.4f} | "
              f"{L_num*1e9:>14.4f} | {err:>10.3e}")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    log_r = [np.log(r) for r in r_wire_range]
    ax.plot(log_r, [L * 1e9 for L in L_ref_list], 'k--',
            label='Wien', linewidth=2)
    ax.plot(log_r, [L * 1e9 for L in L_num_list], 'o',
            color='steelblue', label='Gauss', markersize=5)
    ax.set_xlabel('ln(r_wire)')
    ax.set_ylabel('L [nH]')
    ax.set_title('Self-inductance vs ln(r_wire)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy([r * 1e3 for r in r_wire_range], errs, 'o-',
                color='steelblue', markersize=5)
    ax.set_xlabel('r_wire [mm]')
    ax.set_ylabel('Relative error')
    ax.set_title('Error vs wire radius')
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    _save_plot('inductance_self_vs_wire_radius.png')
    return errs


def test_self_inductance_vs_loop_radius(r_wire:       float = 0.5e-3,
                                        radius_range: list  = None):
    """
    Sweeps loop radius at fixed wire radius. Checks that the inductance
    scales correctly with loop size.

    Keeps radius/r_wire > 10 to stay in the thin-wire regime where
    Wien's formula is valid.
    """
    if radius_range is None:
        # Start at 10× wire radius to stay in thin-wire regime
        radius_range = np.linspace(max(0.01, 10 * r_wire), 0.20, 15).tolist()

    L_ref_list = []
    L_num_list = []
    errs       = []

    print("── Circular self-inductance / vs loop radius ───────────────────────")
    print(f"  r_wire = {r_wire*1e3:.2f} mm,  n_pts = 80")
    print()
    print(f"  {'radius [cm]':>12} | {'R/r_wire':>8} | {'L_Wien [nH]':>14} | "
          f"{'L_gauss [nH]':>14} | {'rel_err':>10}")
    print("  " + "-" * 70)

    for radius in radius_range:
        L_ref = wien_self_inductance(radius, r_wire)
        L_num = solver.self_inductance_gauss(
            a=radius, b=radius, r_wire=r_wire, n_pts=80).item()
        err = abs(L_num - L_ref) / abs(L_ref)
        L_ref_list.append(L_ref)
        L_num_list.append(L_num)
        errs.append(err)
        print(f"  {radius*100:>12.2f} | {radius/r_wire:>8.0f} | {L_ref*1e9:>14.4f} | "
              f"{L_num*1e9:>14.4f} | {err:>10.3e}")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    r_cm = [r * 100 for r in radius_range]
    ax.plot(r_cm, [L * 1e9 for L in L_ref_list], 'k--',
            label='Wien', linewidth=2)
    ax.plot(r_cm, [L * 1e9 for L in L_num_list], 'o',
            color='steelblue', label='Gauss', markersize=5)
    ax.set_xlabel('Loop radius [cm]')
    ax.set_ylabel('L [nH]')
    ax.set_title('Self-inductance vs loop radius')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(r_cm, errs, 'o-', color='steelblue', markersize=5)
    ax.set_xlabel('Loop radius [cm]')
    ax.set_ylabel('Relative error')
    ax.set_title('Error vs loop radius')
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    _save_plot('inductance_self_vs_loop_radius.png')
    return errs


def test_self_inductance_dc_vs_hf():
    """
    Compares DC and HF self-inductance.

    DC includes the internal inductance of the wire cross-section,
    so L_dc > L_hf. The difference is:

        L_dc − L_hf = μ₀R · 1/4   (internal inductance of loop)

    Wien:  L_dc = μ₀R[ln(8R/r) − 7/4]
           L_hf = μ₀R[ln(8R/r) − 2]
           Δ = μ₀R · 1/4
    """
    print("── Circular self-inductance / DC vs HF ─────────────────────────────")
    radius = 0.05
    r_wire = 0.5e-3

    L_dc_ref = wien_self_inductance(radius, r_wire, 'dc')
    L_hf_ref = wien_self_inductance(radius, r_wire, 'hf')
    delta_ref = L_dc_ref - L_hf_ref
    delta_exact = MU0 * radius * 0.25

    L_dc = solver.self_inductance_gauss(
        a=radius, b=radius, r_wire=r_wire, n_pts=80, frequency='dc').item()
    L_hf = solver.self_inductance_gauss(
        a=radius, b=radius, r_wire=r_wire, n_pts=80, frequency='hf').item()
    delta_num = L_dc - L_hf

    print(f"  Wien:  L_dc = {L_dc_ref*1e9:.4f} nH,  L_hf = {L_hf_ref*1e9:.4f} nH,  "
          f"Δ = {delta_ref*1e9:.4f} nH")
    print(f"  Gauss: L_dc = {L_dc*1e9:.4f} nH,  L_hf = {L_hf*1e9:.4f} nH,  "
          f"Δ = {delta_num*1e9:.4f} nH")
    print(f"  Exact Δ = μ₀R/4 = {delta_exact*1e9:.4f} nH")
    print(f"  L_dc > L_hf : {L_dc > L_hf}")
    err = abs(delta_num - delta_exact) / abs(delta_exact)
    print(f"  Δ rel_err = {err:.3e}")
    print()


# ── Mutual inductance tests ──────────────────────────────────────────────────

def test_coaxial_mutual_inductance_single():
    """
    Checks mutual_inductance_gauss for two coaxial circular loops against
    Maxwell's exact formula using elliptic integrals.

    Two loops: both radius 5 cm, separated by 3 cm along z.
    """
    print("── Coaxial mutual inductance / single point ────────────────────────")
    R1, R2 = 0.05, 0.05
    d = 0.03

    M_ref = maxwell_mutual_inductance(R1, R2, d)
    center2 = torch.tensor([0.0, 0.0, d], dtype=torch.float64)
    M_num = solver.mutual_inductance_gauss(
        a1=R1, b1=R1, a2=R2, b2=R2,
        center2=center2, n_pts=120).item()

    err = abs(M_num - M_ref) / abs(M_ref)
    print(f"  R1 = R2 = {R1*100:.0f} cm,  d = {d*100:.0f} cm")
    print(f"  Maxwell   : M = {M_ref*1e9:.6f} nH")
    print(f"  Gauss     : M = {M_num*1e9:.6f} nH")
    print(f"  rel_err   = {err:.3e}    pass = {err < 0.001}")
    print()


def test_mutual_inductance_convergence_npts(R1:         float = 0.05,
                                             R2:         float = 0.05,
                                             d:          float = 0.03,
                                             npts_range: list  = None):
    """
    Convergence of mutual_inductance_gauss vs n_pts for coaxial circular loops.
    """
    if npts_range is None:
        npts_range = [10, 20, 30, 40, 60, 80, 100, 120, 160, 200]

    M_ref = maxwell_mutual_inductance(R1, R2, d)
    center2 = torch.tensor([0.0, 0.0, d], dtype=torch.float64)
    errs = []

    print("── Coaxial mutual inductance / convergence vs n_pts ────────────────")
    print(f"  R1 = R2 = {R1*100:.0f} cm,  d = {d*100:.0f} cm")
    print(f"  Maxwell M = {M_ref*1e9:.6f} nH")
    print()
    print(f"  {'n_pts':>6} | {'M [nH]':>14} | {'rel_err':>12}")
    print("  " + "-" * 38)

    for n_pts in npts_range:
        M = solver.mutual_inductance_gauss(
            a1=R1, b1=R1, a2=R2, b2=R2,
            center2=center2, n_pts=n_pts).item()
        err = abs(M - M_ref) / abs(M_ref)
        errs.append(err)
        print(f"  {n_pts:>6} | {M*1e9:>14.6f} | {err:>12.3e}")
    print()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(npts_range, errs, 'o-', color='steelblue', markersize=6)
    ax.set_xlabel('Gauss-Legendre points (n_pts)')
    ax.set_ylabel('Relative error  |M − M_Maxwell| / |M_Maxwell|')
    ax.set_title(f'Coaxial mutual inductance convergence vs n_pts\n'
                 f'(R1 = R2 = {R1*100:.0f} cm, d = {d*100:.0f} cm)')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    _save_plot('inductance_mutual_convergence_npts.png')
    return errs


def test_mutual_inductance_vs_separation(R1:      float = 0.05,
                                          R2:      float = 0.05,
                                          d_range: list  = None):
    """
    Sweeps axial separation d and compares mutual inductance against Maxwell.
    """
    if d_range is None:
        d_range = np.linspace(0.005, 0.30, 25).tolist()

    M_ref_list = []
    M_num_list = []
    errs       = []

    print("── Coaxial mutual inductance / vs separation ───────────────────────")
    print(f"  R1 = R2 = {R1*100:.0f} cm,  n_pts = 120")
    print()
    print(f"  {'d [cm]':>8} | {'M_Maxwell [nH]':>16} | {'M_gauss [nH]':>14} | "
          f"{'rel_err':>10}")
    print("  " + "-" * 58)

    for d in d_range:
        M_ref = maxwell_mutual_inductance(R1, R2, d)
        center2 = torch.tensor([0.0, 0.0, d], dtype=torch.float64)
        M_num = solver.mutual_inductance_gauss(
            a1=R1, b1=R1, a2=R2, b2=R2,
            center2=center2, n_pts=120).item()
        err = abs(M_num - M_ref) / abs(M_ref)
        M_ref_list.append(M_ref)
        M_num_list.append(M_num)
        errs.append(err)
        print(f"  {d*100:>8.2f} | {M_ref*1e9:>16.6f} | "
              f"{M_num*1e9:>14.6f} | {err:>10.3e}")
    print()

    d_cm = [d * 100 for d in d_range]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(d_cm, [M * 1e9 for M in M_ref_list], 'k--',
            label='Maxwell', linewidth=2)
    ax.plot(d_cm, [M * 1e9 for M in M_num_list], 'o',
            color='steelblue', label='Gauss', markersize=5)
    ax.set_xlabel('Separation d [cm]')
    ax.set_ylabel('M [nH]')
    ax.set_title('Mutual inductance vs separation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(d_cm, errs, 'o-', color='steelblue', markersize=5)
    ax.set_xlabel('Separation d [cm]')
    ax.set_ylabel('Relative error')
    ax.set_title('Error vs separation')
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    _save_plot('inductance_mutual_vs_separation.png')
    return errs


def test_mutual_inductance_vs_radius_ratio(R1:        float = 0.05,
                                            d:         float = 0.03,
                                            R2_range:  list  = None):
    """
    Sweeps the radius of the second loop at fixed R1 and separation d.
    """
    if R2_range is None:
        R2_range = np.linspace(0.01, 0.15, 15).tolist()

    errs = []

    print("── Coaxial mutual inductance / vs radius ratio ─────────────────────")
    print(f"  R1 = {R1*100:.0f} cm,  d = {d*100:.0f} cm,  n_pts = 120")
    print()
    print(f"  {'R2 [cm]':>8} | {'R2/R1':>6} | {'M_Maxwell [nH]':>16} | "
          f"{'M_gauss [nH]':>14} | {'rel_err':>10}")
    print("  " + "-" * 68)

    center2 = torch.tensor([0.0, 0.0, d], dtype=torch.float64)

    for R2 in R2_range:
        M_ref = maxwell_mutual_inductance(R1, R2, d)
        M_num = solver.mutual_inductance_gauss(
            a1=R1, b1=R1, a2=R2, b2=R2,
            center2=center2, n_pts=120).item()
        err = abs(M_num - M_ref) / abs(M_ref)
        errs.append(err)
        print(f"  {R2*100:>8.2f} | {R2/R1:>6.2f} | {M_ref*1e9:>16.6f} | "
              f"{M_num*1e9:>14.6f} | {err:>10.3e}")
    print()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy([R2 / R1 for R2 in R2_range], errs, 'o-',
                color='steelblue', markersize=6)
    ax.set_xlabel('R₂ / R₁')
    ax.set_ylabel('Relative error')
    ax.set_title(f'Mutual inductance error vs radius ratio\n'
                 f'(R₁ = {R1*100:.0f} cm, d = {d*100:.0f} cm)')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    _save_plot('inductance_mutual_vs_radius_ratio.png')
    return errs


def test_mutual_inductance_tilted_loops(R1:  float = 0.05,
                                         R2:  float = 0.05,
                                         d:   float = 0.05):
    """
    Mutual inductance when the second loop is tilted about the x-axis.

    At 0° (coplanar, coaxial) M should match Maxwell. At 90° the loops
    are perpendicular and M should be zero by symmetry.
    """
    print("── Mutual inductance / tilted loops ────────────────────────────────")
    print(f"  R1 = R2 = {R1*100:.0f} cm,  d = {d*100:.0f} cm,  n_pts = 120")
    print()

    angles = np.linspace(0, 90, 19).tolist()
    M_list = []
    center2 = torch.tensor([0.0, 0.0, d], dtype=torch.float64)

    M_maxwell_0 = maxwell_mutual_inductance(R1, R2, d)

    print(f"  {'angle [°]':>10} | {'M [nH]':>14}")
    print("  " + "-" * 28)

    for angle in angles:
        R2_rot = _rotation_about_x(angle)
        M = solver.mutual_inductance_gauss(
            a1=R1, b1=R1, a2=R2, b2=R2,
            center2=center2, R2=R2_rot, n_pts=120).item()
        M_list.append(M)
        print(f"  {angle:>10.1f} | {M*1e9:>14.6f}")

    err_0  = abs(M_list[0] - M_maxwell_0) / abs(M_maxwell_0)
    err_90 = abs(M_list[-1]) / abs(M_maxwell_0)

    print()
    print(f"  At 0°:  M = {M_list[0]*1e9:.6f} nH  vs Maxwell {M_maxwell_0*1e9:.6f} nH  "
          f"(err = {err_0:.3e})")
    print(f"  At 90°: M = {M_list[-1]*1e9:.6f} nH  "
          f"(should be ≈ 0, |M/M₀| = {err_90:.3e})")
    print()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(angles, [M * 1e9 for M in M_list], 'o-', color='steelblue',
            markersize=6)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Tilt angle [°]')
    ax.set_ylabel('M [nH]')
    ax.set_title(f'Mutual inductance vs tilt angle\n'
                 f'(R₁ = R₂ = {R1*100:.0f} cm, d = {d*100:.0f} cm)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_plot('inductance_mutual_tilted.png')
    return M_list, angles


# ── Elliptical-specific tests ─────────────────────────────────────────────────

def test_ellipse_self_inductance_eccentricity(r_wire:    float = 0.5e-3,
                                               perimeter: float = 2 * np.pi * 0.05,
                                               n_pts:     int   = 80):
    """
    Self-inductance of an ellipse vs eccentricity at FIXED perimeter.

    At b/a = 1 the loop is circular and can be checked against Wien.
    """
    print("── Ellipse self-inductance / vs eccentricity (fixed perimeter) ─────")
    print(f"  perimeter = {perimeter*100:.2f} cm,  r_wire = {r_wire*1e3:.2f} mm")
    print()

    R_circ = perimeter / (2 * np.pi)
    L_wien = wien_self_inductance(R_circ, r_wire)

    ratios = np.linspace(0.2, 1.0, 17).tolist()
    L_list = []
    a_list = []
    b_list = []

    from scipy.special import ellipe as scipy_ellipe
    from scipy.optimize import brentq

    def ellipse_perimeter(a, b):
        if a < b:
            a, b = b, a
        e2 = 1 - (b / a)**2
        return 4 * a * scipy_ellipe(e2)

    print(f"  {'b/a':>6} | {'a [cm]':>8} | {'b [cm]':>8} | {'L [nH]':>12}")
    print("  " + "-" * 42)

    for ratio in ratios:
        def residual(a_val):
            return ellipse_perimeter(a_val, ratio * a_val) - perimeter

        a_val = brentq(residual, 0.001, 1.0)
        b_val = ratio * a_val
        a_list.append(a_val)
        b_list.append(b_val)

        L = solver.self_inductance_gauss(
            a=a_val, b=b_val, r_wire=r_wire, n_pts=n_pts).item()
        L_list.append(L)
        print(f"  {ratio:>6.3f} | {a_val*100:>8.3f} | {b_val*100:>8.3f} | "
              f"{L*1e9:>12.4f}")

    err_circle = abs(L_list[-1] - L_wien) / abs(L_wien)
    print()
    print(f"  At b/a=1.0: L_gauss = {L_list[-1]*1e9:.4f} nH  vs  "
          f"L_Wien = {L_wien*1e9:.4f} nH  (err = {err_circle:.3e})")
    print()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ratios, [L * 1e9 for L in L_list], 'o-', color='steelblue',
            markersize=6)
    ax.axhline(L_wien * 1e9, color='firebrick', linestyle='--',
               label=f'Wien (circle), L = {L_wien*1e9:.2f} nH')
    ax.set_xlabel('Aspect ratio b/a')
    ax.set_ylabel('L [nH]')
    ax.set_title(f'Ellipse self-inductance vs eccentricity\n'
                 f'(fixed perimeter = {perimeter*100:.2f} cm)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_plot('inductance_self_eccentricity.png')
    return L_list, ratios


def test_rotation_invariance(radius: float = 0.05,
                              r_wire: float = 0.5e-3):
    """
    Self-inductance should be invariant under rigid rotation and translation.
    """
    print("── Self-inductance / rotation & translation invariance ─────────────")

    L_base = solver.self_inductance_gauss(
        a=radius, b=radius, r_wire=r_wire, n_pts=80).item()

    R = _rotation_about_x(37.0) @ _rotation_about_z(53.0)
    center = torch.tensor([1.0, -2.0, 3.5], dtype=torch.float64)

    L_rot = solver.self_inductance_gauss(
        a=radius, b=radius, r_wire=r_wire, center=center, R=R, n_pts=80).item()

    err = abs(L_rot - L_base) / abs(L_base)
    print(f"  L (origin, I)     = {L_base*1e9:.6f} nH")
    print(f"  L (shifted, rot)  = {L_rot*1e9:.6f} nH")
    print(f"  rel_err           = {err:.3e}    pass = {err < 1e-10}")
    print()


def test_reciprocity(R1: float = 0.05,
                     R2: float = 0.03,
                     d:  float = 0.04):
    """
    M(loop1, loop2) should equal M(loop2, loop1).
    """
    print("── Mutual inductance / reciprocity ─────────────────────────────────")

    center2 = torch.tensor([0.0, 0.0, d], dtype=torch.float64)

    M_12 = solver.mutual_inductance_gauss(
        a1=R1, b1=R1, a2=R2, b2=R2,
        center2=center2, n_pts=120).item()

    center1_swap = torch.tensor([0.0, 0.0, -d], dtype=torch.float64)
    M_21 = solver.mutual_inductance_gauss(
        a1=R2, b1=R2, a2=R1, b2=R1,
        center1=center1_swap, n_pts=120).item()

    err = abs(M_12 - M_21) / abs(M_12)
    print(f"  R1 = {R1*100:.0f} cm,  R2 = {R2*100:.0f} cm,  d = {d*100:.0f} cm")
    print(f"  M(1→2) = {M_12*1e9:.6f} nH")
    print(f"  M(2→1) = {M_21*1e9:.6f} nH")
    print(f"  rel_err = {err:.3e}    pass = {err < 1e-10}")
    print()


def test_adaptive_convergence():
    """
    Checks that the adaptive methods converge and match fixed-order results.
    """
    print("── Adaptive convergence ────────────────────────────────────────────")
    radius = 0.05
    r_wire = 0.5e-3
    d = 0.03

    # Self
    L_fixed = solver.self_inductance_gauss(
        a=radius, b=radius, r_wire=r_wire, n_pts=160).item()
    L_adapt = solver.self_inductance_adaptive(
        a=radius, b=radius, r_wire=r_wire, rtol=1e-10).item()
    err_self = abs(L_adapt - L_fixed) / abs(L_fixed)

    print(f"  Self-inductance:")
    print(f"    Fixed (n=160) = {L_fixed*1e9:.6f} nH")
    print(f"    Adaptive      = {L_adapt*1e9:.6f} nH")
    print(f"    rel_err       = {err_self:.3e}")

    # Mutual
    center2 = torch.tensor([0.0, 0.0, d], dtype=torch.float64)
    M_fixed = solver.mutual_inductance_gauss(
        a1=radius, b1=radius, a2=radius, b2=radius,
        center2=center2, n_pts=320).item()
    M_adapt = solver.mutual_inductance_adaptive(
        a1=radius, b1=radius, a2=radius, b2=radius,
        center2=center2, rtol=1e-10).item()
    err_mut = abs(M_adapt - M_fixed) / abs(M_fixed)

    print(f"  Mutual inductance:")
    print(f"    Fixed (n=320) = {M_fixed*1e9:.6f} nH")
    print(f"    Adaptive      = {M_adapt*1e9:.6f} nH")
    print(f"    rel_err       = {err_mut:.3e}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("  INDUCTANCE TEST SUITE")
    print("=" * 65, '\n')

    print("── SELF-INDUCTANCE ─────────────────────────────────────────────────")
    test_circular_self_inductance_single()
    test_circular_self_inductance_vs_builtin()
    test_self_inductance_convergence_npts()
    test_self_inductance_vs_wire_radius()
    test_self_inductance_vs_loop_radius()
    test_self_inductance_dc_vs_hf()
    test_rotation_invariance()

    print("── MUTUAL INDUCTANCE ───────────────────────────────────────────────")
    test_coaxial_mutual_inductance_single()
    test_mutual_inductance_convergence_npts()
    test_mutual_inductance_vs_separation()
    test_mutual_inductance_vs_radius_ratio()
    test_mutual_inductance_tilted_loops()
    test_reciprocity()

    print("── ADAPTIVE ────────────────────────────────────────────────────────")
    test_adaptive_convergence()

    print("── ELLIPTICAL ──────────────────────────────────────────────────────")
    test_ellipse_self_inductance_eccentricity()

    print("=" * 65)
    print("  WHAT TO EXPECT")
    print("=" * 65)
    print("""
  SELF-INDUCTANCE  (singularity subtraction)
  ──────────────────────────────────────────
  Single point       : Gauss at n_pts=80 matches Wien to < 0.1%
  Convergence        : exponential in n_pts — the smooth remainder
                       after subtracting the analytic singular part
                       is well-resolved by GL even at moderate n_pts
  vs wire radius     : linear in ln(r_wire), tracks Wien across decades
                       (stays in thin-wire regime r << R)
  vs loop radius     : smooth increase, matches Wien at all sizes
  DC vs HF           : L_dc > L_hf by exactly μ₀R/4 (internal inductance)
                       DC: k=7/4, HF: k=2 in Wien's formula
  Rotation invariance: L is geometric — invariant to machine precision

  MUTUAL INDUCTANCE
  ─────────────────
  Single point       : matches Maxwell (elliptic integrals) to < 0.1%
  Convergence        : exponential in n_pts — smooth integrand
  vs separation      : M → 0 at large d (dipole-dipole 1/d³ decay)
  vs radius ratio    : handles R1 ≠ R2 correctly
  Tilted loops       : M = Maxwell at 0°, M = 0 at 90° (symmetry)
  Reciprocity        : M(1,2) = M(2,1) to machine precision

  ELLIPTICAL
  ──────────
  vs eccentricity    : at b/a=1 matches Wien (circular limit);
                       inductance increases as ellipse becomes more
                       eccentric at fixed perimeter
""")
