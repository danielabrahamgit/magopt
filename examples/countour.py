import torch
import numpy as np

import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

def contours_skimage_theta_z(
    theta: torch.Tensor,   # (Nt,)
    z: torch.Tensor,       # (Nz,)
    phi: torch.Tensor,     # (Nz, Nt)
    level: float,
    periodic_theta: bool = True,
    min_points: int = 10,
):
    """
    Returns: list of polylines, each polyline is (K,2) torch tensor [theta, z]
    """
    from skimage import measure

    th = theta.detach().cpu().numpy()
    zz = z.detach().cpu().numpy()
    ff = phi.detach().cpu().numpy()

    # seam fix for periodic theta: duplicate first column at end
    if periodic_theta:
        ff = np.concatenate([ff, ff[:, :1]], axis=1)   # (Nz, Nt+1)
        th = np.concatenate([th, th[:1] + 2*np.pi])    # (Nt+1,)

    # marching squares in index space: returns list of (K,2) arrays [row, col]
    contours_rc = measure.find_contours(ff, level=level)

    polylines = []
    for rc in contours_rc:
        if rc.shape[0] < min_points:
            continue

        r = rc[:, 0]  # row index in [0, Nz-1]
        c = rc[:, 1]  # col index in [0, Nt] if periodic_theta else [0, Nt-1]

        # Convert row/col (float) -> physical coordinates with 1D linear interpolation
        # r maps along z, c maps along theta
        z_pts = np.interp(r, np.arange(len(zz)), zz)
        th_pts = np.interp(c, np.arange(len(th)), th)

        verts = np.stack([th_pts, z_pts], axis=1)  # (K,2) [theta,z]
        polylines.append(torch.from_numpy(verts).to(phi.device, dtype=phi.dtype))

    return polylines


def contours_skimage_3d(
    theta: torch.Tensor,
    z: torch.Tensor,
    phi: torch.Tensor,
    levels,
    map_fn,                 # (theta_pts, z_pts) -> (x,y,z) each (K,)
    periodic_theta: bool = True,
    resample: int | None = None,
):
    if isinstance(levels, (float, int)):
        levels = [float(levels)]
    else:
        levels = [float(l) for l in levels]

    out = []
    for lvl in levels:
        tz_polys = contours_skimage_theta_z(theta, z, phi, lvl, periodic_theta=periodic_theta)
        xyz_polys = []
        for tz in tz_polys:
            th_pts = tz[:, 0]
            z_pts = tz[:, 1]
            x, y, z3 = map_fn(th_pts, z_pts)
            xyz = torch.stack([x, y, z3], dim=-1)
            if resample is not None and xyz.shape[0] >= 2:
                xyz = _resample_polyline_by_arclength(xyz, resample)
            xyz_polys.append(xyz)
        out.append(xyz_polys)
    return out


def _resample_polyline_by_arclength(xyz: torch.Tensor, K: int) -> torch.Tensor:
    d = torch.linalg.norm(xyz[1:] - xyz[:-1], dim=-1)
    s = torch.cat([torch.zeros(1, device=xyz.device, dtype=xyz.dtype),
                   torch.cumsum(d, dim=0)])
    total = s[-1].clamp_min(torch.finfo(xyz.dtype).eps)
    t = torch.linspace(0, 1, K, device=xyz.device, dtype=xyz.dtype) * total
    idx = torch.searchsorted(s, t, right=True).clamp(1, s.numel()-1)
    s0, s1 = s[idx-1], s[idx]
    w = (t - s0) / (s1 - s0).clamp_min(torch.finfo(xyz.dtype).eps)
    return xyz[idx-1] + w.unsqueeze(-1) * (xyz[idx] - xyz[idx-1])

Nz, Nt = 256, 256
device = "cpu"

theta = torch.linspace(0, 2*torch.pi, Nt, device=device)
z = torch.linspace(-1, 1, Nz, device=device)

# Example scalar field on grid (Nz, Nt)
TH, ZZ = torch.meshgrid(theta, z, indexing="ij")
vec = torch.stack([TH, ZZ], dim=-1)
center = torch.tensor([2.0, 0], device=device)
phi = torch.exp(-(vec - center).norm(dim=-1)**2 / 0.4**2)
phi = phi.rot90()

ret = contours_skimage_theta_z(theta, z, phi, level=0.4, periodic_theta=True, min_points=1)

for r in ret:
    plt.plot(r[:, 0].cpu(), r[:, 1].cpu(), color="red")
plt.ylim(z.min().item(), z.max().item())
plt.xlim(theta.min().item(), theta.max().item())

plt.imshow(phi.cpu(), extent=[theta.min().item(), theta.max().item(), z.min().item(), z.max().item()], aspect="auto")

plt.show()