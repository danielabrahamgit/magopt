import torch
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from magopt.gradient_coils import matrix_coil, circular_z_coil, elliptical_frustum
from mr_recon.utils import gen_grd
from magopt.sim.analytic import calc_bfield_loop 
from magopt.sim.elip import EllipELookup, EllipKLookup
from magopt.optim_admm import (
    admm_min_peak_norm_plus_quadtratic, 
    admm_min_quadratic, 
    admm_min_peak_norm,
    admm_general,
    unrolled_admm_general,
)
from einops import einsum

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
Rsurface = 0.13 # m (radius of surface)
Zsurface = 0.2 # m (height of surface)s
dsv = 0.16 # Diameter of spherical volume to optimize over
im_size = (51, 51, 51) 
grad_dir = 1 # 0 --> x, 1 --> y, 2 --> z
d = len(im_size)
Nrad = 6
Nz = 6
Ncoils = Nrad * Nz

# Coordinates of interest
crds = gen_grd(im_size, (fov,)*d).to(torch_dev) + 1.23e-5
crds_flt = crds.reshape(-1, 3)
mask_dsv = (crds_flt.norm(dim=-1) <= dsv / 2) * 1.0
crds_grad = crds_flt[mask_dsv > 0]

# Coordinates of body (surface only)
angles = torch.linspace(0, 2 * torch.pi, 30, device=torch_dev)
zs = torch.linspace(-Zbody / 2, Zbody / 2, 50, device=torch_dev)
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

# Build gradient coil, relevant matrices
grad_coil = matrix_coil(radii=radii, 
                        centers=centers, 
                        thetas_phis=thetas_phis)
# zs_spline = torch.linspace(-Zsurface / 2, Zsurface / 2, 10, device=torch_dev)
# rs_spline = torch.ones_like(zs_spline) * Rsurface
# zs = torch.linspace(-Zsurface / 2, Zsurface / 2, 40, device=torch_dev)
# grad_coil = circular_z_coil(zs=zs, zs_spline=zs_spline, rs_spline=rs_spline,
#                             M_fourier_modes=10)
def G_theta(thetas):
    G, _, _ = grad_coil.build_field_matrices(crds_gfield=crds_grad, crds_bfield=crds_body[:1], crds_efield=crds_body[:1])
    return G[grad_dir] * 1e3 # T/m -> mT/m
def L_theta(thetas):
    L =  grad_coil.build_magnetic_energy_matrix()
    if isinstance(grad_coil, matrix_coil):
        L = L / Ncoils
    Lfact = torch.linalg.cholesky(L + 1e-12 * torch.eye(L.shape[0], device=torch_dev)).T
    return Lfact
def E_theta(thetas):
    _, _, E = grad_coil.build_field_matrices(crds_gfield=crds_grad[:1], crds_bfield=crds_body[:1], crds_efield=crds_body)
    E = E * 1e3 / 1e-3 # s V/m -> mV/m/kHz
    return E.moveaxis(0, 1) # 3 N Ncoeff -> N 3 Ncoeff
def C_theta(thetas):
    return None
G = G_theta([])
L = L_theta([])
E = E_theta([])

# ADMM arguments
admm_kwargs = {
    # 'lamdaG': 1e1,
    'Gmin': 1,
    # 'lamdaL': 1e-3,
    'Lmax': 100_000e-6,
    'lamdaE': 1e0,
    # 'Emax': 8,
    # 'linearity_pcnt': 0.1,
    'rho_adapt': True,
}
surface_opt = False

if not surface_opt:
    dct = admm_general(G=G, L=L, E=E,
                       admm_iters=500,
                       **admm_kwargs)
else:
    grad_coil.radii.requires_grad = True
    grad_coil.centers.requires_grad = True
    grad_coil.thetas_phis.requires_grad = True
    # grad_coil.spline.coeff.requires_grad = True
    # grad_coil.zofs.requires_grad = True
    thetas = [
        grad_coil.radii,
        grad_coil.centers,
        grad_coil.thetas_phis,
        # grad_coil.spline.coeff,
        # grad_coil.zofs,
    ]
    dct = unrolled_admm_general(thetas, G_theta, L_theta, E_theta, C_theta,
                                admm_iters=50,
                                epochs=100*2,
                                lr=[1e-3, 1e-3, 1e-2],
                                **admm_kwargs)  
    grad_coil.radii.requires_grad = False
    grad_coil.centers.requires_grad = False
    grad_coil.thetas_phis.requires_grad = False  
    # grad_coil.spline.coeff.requires_grad = False
    # grad_coil.zofs.requires_grad = False
Gtarg = dct['Gmin']
Gtarg_actual = (G_theta([]) @ dct['x']).min().item()
print(f"Target Gradient = {Gtarg:.2f}mT/m")
print(f"Actual Gradient = {Gtarg_actual:.2f}mT/m")
Epeak = dct['Emax']
Epeak_actual = (E_theta([]) @ dct['x']).norm(dim=-1).max().item()
print(f"Target Peak Efield = {Epeak:.2f}mV/m/kHz")
print(f"Actual Peak Efield = {Epeak_actual:.2f}mV/m/kHz")
Lpeak = dct['Lmax']
print(f"Inductance = {2e6*Lpeak:.2f} uH")

# Plot diagnostics
plt.figure()
plt.plot(dct['r_pri'], label='Primal Residual')
plt.plot(dct['s_dual'], label='Dual Residual')
plt.legend()

# Show Coil
fig, ax, axl = grad_coil.show_design(coeffs=dct['x'])

# Show fields
G, B, E = grad_coil.build_field_matrices(crds_gfield=crds_flt, crds_bfield=crds_flt, crds_efield=crds_flt)
Gz = einsum(G[grad_dir], dct['x'], 'N X, X -> N').reshape(im_size).cpu()
Bfield = einsum(B[-1], dct['x'], 'N X, X -> N').reshape(im_size).cpu()
Efield = einsum(E, dct['x'], 'd N X, X -> d N').norm(dim=0).reshape(im_size).cpu()
grad_axis = ['x', 'y', 'z'][grad_dir]
nslices = 3
if grad_dir == 0 or grad_dir == 1:
    crds_1d = crds[0, 0, :, 2].cpu()
    slice_name = 'Axial (X-Y)'
    slice_axis = 'Z'
    xlabel = 'X [cm]'
    ylabel = 'Y [cm]'
    extent = [1e2*crds[..., 0].min().cpu().item(), 1e2*crds[..., 0].max().cpu().item(), 1e2*crds[..., 1].min().cpu().item(), 1e2*crds[..., 1].max().cpu().item()]
    crds_desired = torch.linspace(-dsv/3, dsv/3, nslices)
    idxs = torch.searchsorted(crds_1d, crds_desired)
    tupk = lambda k : (slice(None), slice(None), k)
elif grad_dir == 2:
    crds_1d = crds[0, :, 0, 1].cpu()
    slice_name = 'Corronal (X-Z)'
    slice_axis = 'Y'
    xlabel = 'X [cm]'
    ylabel = 'Z [cm]'
    extent = [1e2*crds[..., 0].min().cpu().item(), 1e2*crds[..., 0].max().cpu().item(), 1e2*crds[..., 2].min().cpu().item(), 1e2*crds[..., 2].max().cpu().item()]
    crds_desired = torch.linspace(-dsv/3, dsv/3, nslices)
    idxs = torch.searchsorted(crds_1d, crds_desired)
    tupk = lambda k : (slice(None), k, slice(None))
for k, idx in enumerate(idxs):
    plt.figure(figsize=(14,7))
    plt.suptitle(f'{slice_name} at {slice_axis} = {1e2 * crds_1d[idx].cpu().item():.2f} cm')
    plt.subplot(131)
    plt.title('Gradient Field (mT/m)')
    plt.imshow(Gz[tupk(idx)].rot90() * 1e3, 
               vmin=Gtarg*0.5, vmax=Gtarg*1.5, 
               cmap='RdBu_r', extent=extent)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(location='bottom')
    plt.subplot(132)
    plt.title('Magnetic Field (mT)')
    plt.imshow(Bfield[tupk(idx)].rot90() * 1e3,
               vmin=-Gtarg*Zbody/2, vmax=Gtarg*Zbody/2,
               cmap='RdBu_r', extent=extent)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(location='bottom')
    plt.subplot(133)
    plt.title('Electric Field (mV/m/kHz)')
    plt.imshow(Efield[tupk(idx)].rot90() * 1e3 / 1e-3, 
               vmin=0, vmax=10, 
               extent=extent, cmap='viridis')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(location='bottom')

plt.show()