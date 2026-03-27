from pathlib import Path
import sys
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from magopt.sim.physics.ellipse.bfield import elliptical_bfield
from magopt.sim.physics.filament.wire_model import parametric_wire, elliptical_wire

PLOTS_DIR = SRC_ROOT / "magopt" / "sim" / "assets" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_plot(filename: str) -> Path:
    out_path = PLOTS_DIR / filename
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}\n")
    return out_path


def _set_integer_xticks(ax, ticks: list[int]) -> None:
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(tick) for tick in ticks]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="x", labelrotation=45)


def _rotation_z(angle_rad: float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return torch.tensor([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=dtype, device=device)


def _ellipse_polyline_points(
    a: float,
    b: float,
    center: torch.Tensor,
    rotation: torch.Tensor,
    n_seg: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Polygonal ellipse with the first point repeated at the end so the
    segment-based calc_bfield path evaluates the full closed loop.
    """
    phi = torch.linspace(0.0, 2.0 * math.pi, n_seg + 1, dtype=dtype, device=device)
    pts_local = torch.stack([
        a * torch.cos(phi),
        b * torch.sin(phi),
        torch.zeros_like(phi),
    ], dim=-1)
    return pts_local @ rotation.T + center[None, :]


def test_elliptical_bfield_kernel_convergence(
    a: float = 0.12,
    b: float = 0.07,
    midpoint_n_seg_range: list[int] | None = None,
    ellipse_n_quad_range: list[int] | None = None,
    n_ref: int = 600,
):
    """
    Make two convergence plots for the elliptical wire B-field:
    1. general filament midpoint vs polygon segment count
    2. specialized ellipse kernel vs ellipse quadrature order
    """
    if midpoint_n_seg_range is None:
        midpoint_n_seg_range = [4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128]
    if ellipse_n_quad_range is None:
        ellipse_n_quad_range = [4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128]

    dtype = torch.float64
    device = torch.device("cpu")

    center = torch.tensor([0.015, -0.020, 0.010], dtype=dtype, device=device)
    rotation = _rotation_z(0.55, dtype=dtype, device=device)

    pts_eval = torch.tensor([
        [0.180,  0.090, 0.110],
        [-0.140, 0.120, 0.160],
        [0.070, -0.210, 0.190],
        [0.220, -0.020, 0.080],
    ], dtype=dtype, device=device)

    ref_kernel = elliptical_bfield()
    b_ref = ref_kernel.full_bfield_kernel_gauss(
        pts_eval,
        a=a,
        b=b,
        center=center,
        R=rotation,
        n_pts=n_ref,
    )

    special_wire = elliptical_wire(
        num_pts=32,
        a=a,
        b=b,
        R=rotation,
        t=center,
        verbose=False,
    )

    errs = {
        "midpoint_general": [],
        "ellipse_specialized": [],
    }

    print("── Elliptical wire / B-field convergence ───────────────────────────")
    print(f"  Reference: exact ellipse Gauss kernel (n_ref={n_ref})")
    print()
    print(f"  {'n_segs':>7} | {'general midpoint':>18}")
    print("  " + "-" * 31)

    for n_seg in midpoint_n_seg_range:
        wire_pts = _ellipse_polyline_points(
            a,
            b,
            center,
            rotation,
            n_seg,
            dtype=dtype,
            device=device,
        )
        general_wire = parametric_wire(wire_pts, closed=False, verbose=False)

        b_mid = general_wire.calc_bfield(pts_eval, kernel="midpoint")
        rel_mid = (b_mid - b_ref).norm(dim=-1) / b_ref.norm(dim=-1).clamp(min=1e-30)
        err_mid = rel_mid.max().item()
        errs["midpoint_general"].append(err_mid)

        print(f"  {n_seg:>7} | {err_mid:>18.3e}")
    print()

    print(f"  {'n_quad':>7} | {'specialized ellipse':>20}")
    print("  " + "-" * 33)
    for n_quad in ellipse_n_quad_range:
        b_special = special_wire.calc_bfield(pts_eval, kernel="ellipse_gauss", n_pts=n_quad)
        rel_special = (b_special - b_ref).norm(dim=-1) / b_ref.norm(dim=-1).clamp(min=1e-30)
        err_special = rel_special.max().item()
        errs["ellipse_specialized"].append(err_special)
        print(f"  {n_quad:>7} | {err_special:>20.3e}")
    print()

    fig_mid, ax_mid = plt.subplots(figsize=(8, 5))
    ax_mid.loglog(
        midpoint_n_seg_range,
        errs["midpoint_general"],
        "s-",
        color="firebrick",
        label="general filament midpoint",
        markersize=6,
    )

    ref_n = np.asarray(midpoint_n_seg_range, dtype=float)
    i0 = min(3, len(midpoint_n_seg_range) - 1)
    ax_mid.loglog(
        ref_n,
        errs["midpoint_general"][i0] * (midpoint_n_seg_range[i0] / ref_n) ** 2,
        "k--",
        alpha=0.4,
        label="O(1/N^2) reference",
    )
    ax_mid.set_xlabel("Number of polygon segments")
    ax_mid.set_ylabel("Max relative vector error  ||B - B_ref|| / ||B_ref||")
    ax_mid.set_title(
        "Elliptical wire B-field convergence\n"
        f"General filament midpoint (a={a:.2f} m, b={b:.2f} m)"
    )
    ax_mid.legend()
    ax_mid.grid(True, which="both", alpha=0.3)
    _set_integer_xticks(ax_mid, midpoint_n_seg_range)
    plt.tight_layout()
    _save_plot("bfield_ellipse_midpoint_convergence.png")
    plt.close(fig_mid)

    fig_ellipse, ax_ellipse = plt.subplots(figsize=(8, 5))
    ax_ellipse.semilogy(
        ellipse_n_quad_range,
        errs["ellipse_specialized"],
        "o-",
        color="steelblue",
        label="specialized ellipse_gauss",
        markersize=6,
    )
    ax_ellipse.set_xlabel("Ellipse quadrature points (n_pts)")
    ax_ellipse.set_ylabel("Max relative vector error  ||B - B_ref|| / ||B_ref||")
    ax_ellipse.set_title(
        "Elliptical wire B-field convergence\n"
        f"Specialized ellipse class (a={a:.2f} m, b={b:.2f} m)"
    )
    ax_ellipse.legend()
    ax_ellipse.grid(True, which="both", alpha=0.3)
    _set_integer_xticks(ax_ellipse, ellipse_n_quad_range)
    plt.tight_layout()
    _save_plot("bfield_ellipse_specialized_convergence.png")
    plt.close(fig_ellipse)

    assert errs["midpoint_general"][-1] < errs["midpoint_general"][0]
    assert errs["ellipse_specialized"][-1] < errs["ellipse_specialized"][0]

    return errs


if __name__ == "__main__":
    torch.set_printoptions(precision=6, sci_mode=True)
    test_elliptical_bfield_kernel_convergence()
