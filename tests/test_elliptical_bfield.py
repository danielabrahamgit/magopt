
import math
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from magopt.sim.physics.ellipse.bfield import elliptical_bfield
from magopt.sim.physics.filament.bfield_kernels import base_bfield


def _make_ellipse_points(a: float, b: float, center: torch.Tensor, R: torch.Tensor, n: int, dtype, device):
    """Uniform phi samples, return points with first repeated at end."""
    phi = torch.linspace(0.0, 2.0 * math.pi, n + 1, dtype=dtype, device=device)
    c = torch.cos(phi)
    s = torch.sin(phi)
    pts_local = torch.stack([a * c, b * s, torch.zeros_like(c)], dim=-1)  # [n+1,3]
    return pts_local @ R.T + center[None, :]


def _polyline_bfield_gauss(crds: torch.Tensor, pts: torch.Tensor, n_pts_seg: int = 12, batch: int = 4096):
    """
    Reference B-field from a polyline with Gauss-Legendre integration on each segment.
    Uses base_bfield.bfield_kernel_gauss.
    """
    bb = base_bfield()
    ds = pts[1:] - pts[:-1]              # [M,3]
    mid = 0.5 * (pts[1:] + pts[:-1])     # [M,3]
    flat = crds.reshape(-1, 3)
    out = torch.zeros((flat.shape[0], 3), dtype=crds.dtype, device=crds.device)

    for i in range(0, ds.shape[0], batch):
        out += bb.bfield_kernel_gauss(ds[i:i+batch], mid[i:i+batch], flat, n_pts=n_pts_seg)

    return out.reshape(crds.shape[:-1] + (3,))


def test_circle_on_axis_matches_analytic():
    dtype = torch.float64
    device = torch.device('cpu')

    R0 = 0.12
    a = b = R0
    center = torch.zeros(3, dtype=dtype, device=device)
    Rot = torch.eye(3, dtype=dtype, device=device)

    ell = elliptical_bfield()

    z = torch.tensor([0.02, 0.05, 0.10, 0.25], dtype=dtype, device=device)
    pts = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=-1)

    B = ell.full_bfield_kernel_gauss(pts, a=a, b=b, center=center, R=Rot, n_pts=200)
    Bz = B[:, 2]

    mu0 = 4e-7 * math.pi
    Bz_true = mu0 * (R0**2) / (2.0 * (R0**2 + z.cpu().numpy()**2)**1.5)
    Bz_true = torch.tensor(Bz_true, dtype=dtype, device=device)

    rel_err = (Bz - Bz_true).abs() / Bz_true.abs().clamp(min=1e-30)
    assert torch.all(rel_err < 5e-6), f"rel_err={rel_err}"


def test_ellipse_matches_fine_polyline_reference():
    dtype = torch.float64
    device = torch.device('cpu')

    a = 0.12
    b = 0.07
    center = torch.tensor([0.01, -0.02, 0.03], dtype=dtype, device=device)

    angle = 0.7
    ca, sa = math.cos(angle), math.sin(angle)
    Rot = torch.tensor([[ca, -sa, 0.0],
                        [sa,  ca, 0.0],
                        [0.0, 0.0, 1.0]], dtype=dtype, device=device)

    ell = elliptical_bfield()

    pts_eval = torch.tensor([
        [0.25,  0.10,  0.15],
        [-0.20, 0.18,  0.22],
        [0.12, -0.30,  0.40],
        [0.35, -0.05,  0.08],
    ], dtype=dtype, device=device)

    B_ell = ell.full_bfield_kernel_gauss(pts_eval, a=a, b=b, center=center, R=Rot, n_pts=260)

    n_poly = 8000
    poly_pts = _make_ellipse_points(a, b, center, Rot, n_poly, dtype, device)

    B_ref = _polyline_bfield_gauss(pts_eval, poly_pts, n_pts_seg=10, batch=2048)

    rel = (B_ell - B_ref).norm(dim=-1) / B_ref.norm(dim=-1).clamp(min=1e-30)
    assert torch.all(rel < 5e-3), f"rel={rel}\nB_ell={B_ell}\nB_ref={B_ref}"


if __name__ == "__main__":
    torch.set_printoptions(precision=6, sci_mode=True)
    test_circle_on_axis_matches_analytic()
    test_ellipse_matches_fine_polyline_reference()
    print("All tests passed.")
