import torch
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from magopt.gradient_coils import matrix_coil
from mr_recon.utils import gen_grd
from magopt.sim.analytic import calc_bfield_loop 
from magopt.sim.elip import EllipELookup, EllipKLookup
from magopt.optim_admm import admm_min_peak_norm_plus_quadtratic

# Generate points on surface
def gen_pts_sphere(N):
    pts = []
    N_count = 0
    a = 4 * torch.pi / N
    d = a ** 0.5
    M_theta = round(torch.pi / d)
    d_theta = torch.pi / M_theta
    d_phi = a / d_theta
    while N_count < N:
        for m in range(M_theta):
            theta = torch.pi * (m + 0.5) / M_theta
            M_phi = round(2 * torch.pi * np.sin(theta) / d_phi)
            for n in range(M_phi):
                phi = 2 * torch.pi * n / M_phi
                pts.append([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
                N_count += 1
    pts = torch.tensor(pts).type(torch.float32)
    return pts

def gen_pts_cylinder(Nrad, Nz):
    thetas = torch.arange(Nrad) / Nrad * 2 * torch.pi
    zs = torch.linspace(-0.5, 0.5, Nz)
    thetas, zs = torch.meshgrid(thetas, zs, indexing='ij')
    pts = torch.stack((torch.cos(thetas), torch.sin(thetas), zs), dim=-1)
    pts = pts.reshape(-1, 3)
    pts -= pts.mean(dim=0, keepdim=True)
    return pts

# Params
torch_dev = torch.device(5)
fov = 0.22 # m (field of view)
R = 0.05 # m (radius of loop)
Rbody = 0.1 # m (radius of body)
Zbody = 0.2 # m (height of body)
Rsurface = Rbody + 0.03 # m (radius of surface)
# im_size = (1, 100, 100) 
dsv = 0.16 # Diameter of spherical volume to optimize over
im_size = (51, 51, 1) 
d = len(im_size)
Nrad = 6
Nz = 3
Ncoils = Nrad * Nz

# Coordinates of interest
crds = gen_grd(im_size, (fov,)*d).to(torch_dev) + 1.23e-5
crds_flt = crds.reshape(-1, 3)
mask_dsv = (crds_flt.norm(dim=-1) <= dsv / 2) * 1.0
crds_grad = crds_flt[mask_dsv > 0]

# Coordinates of body (surface only)
angles = torch.linspace(0, 2 * torch.pi, 100, device=torch_dev)
zs = torch.linspace(-Zbody / 2, Zbody / 2, 100, device=torch_dev)
thetas, zs = torch.meshgrid(angles, zs, indexing='ij')
crds_body = torch.stack([Rbody * torch.cos(thetas),
                        Rbody * torch.sin(thetas),
                        zs], dim=-1)
crds_body = crds_body.reshape(-1, 3)

# Coil geometry
centers = gen_pts_cylinder(Nrad, Nz).to(torch_dev)
centers[:, :2] *= Rsurface
centers[:, 2] *= fov
normals = centers.clone()
normals[..., -1] = 0
normals = normals / normals.norm(dim=-1, keepdim=True)
radii = centers[:, 0] * 0 + R
thetas_phis = torch.stack([torch.atan2(normals[..., 1], normals[..., 0]),
                           torch.arccos(normals[..., 2]), ], dim=-1)

# Build matrix coil, relevant matrices
mxc = matrix_coil(radii=radii, 
                  centers=centers, 
                  thetas_phis=thetas_phis)
G, B, E = mxc.build_field_matrices(crds_gfield=crds_grad, crds_bfield=crds_body, crds_efield=crds_body)
L = mxc.build_magnetic_energy_matrix() / Ncoils
print(E.shape)

# Solve via covex optimization
eff = 1e-3 # mT / m / A
dct = admm_min_peak_norm_plus_quadtratic(T=E.moveaxis(0, 1),
                                         L=L,
                                         A=G[-1],
                                         b_lower=eff,
                                         b_upper=None,
                                         lamda=1,
                                         rho_adapt=True,
                                         admm_iters=500)

# Plot diagnostics
plt.figure()
plt.plot(dct['r_pri'], label='Primal Residual')
plt.plot(dct['s_dual'], label='Dual Residual')
plt.legend()

# Show Field pattern

plt.show()