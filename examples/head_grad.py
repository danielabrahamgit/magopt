import torch
import numpy as np
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from magopt.viz import show_3d_slices
from magopt.utils import gen_pts_sphere_surf, gen_pts_ellip_surf_random, gen_grd
from magopt.pns.charge_model import body_charge_model
from magopt.gradient_coils.gradient_surfaces import elliptical_frustum, planar_surface
from magopt.gradient_coils import (
    stream_func_coil, 
    matrix_coil, 
    circular_z_coil,
    stream_func_surface_coil,
)
from magopt.optim_admm import (
    admm_general, 
    unrolled_admm_general, 
    quasi_convex_min_ratio, 
    admm_general_cvxpy
)
from typing import Optional
from einops import einsum

# Parameters
torch_dev = torch.device(5)
grd_size = (51,)*3
dsv = 0.0835 # * 2.4 # m (diameter of spherical volume to optimize over)
grad_dir = 2 # 0 --> x, 1 --> y, 2 --> z
GRAD_UNITS = 1e3 # T -> mT
BFIELD_UNITS = 1e3 # T -> mT
EFIELD_UNITS = 1e3 # V/m -> mV/m
FREQ_UNITS = 1e-3 # Hz -> kHz
INDUCTANCE_UNITS = 1e3 # H -> mH
max_ni_matrix = 20
min_wire_spacing = 1e-3 # m (minimum wire spacing)
dstream = 1 # A (stream function step size AKA current through coil)
opt_surface = False
maxwell_pair = False
use_head_surf = False
# gradient_type = 'stream'
# gradient_type = 'matrix'
gradient_type = 'circular_z'
admm_kwargs = {
    # 'lamdaG': 1e0,
    'Gmin': 1.0,
    # 'lamdaL': 3e1,
    'Lmax': 0.15/2, # mH
    'lamdaE': 1e0,
    # 'Emax': 5,
    # 'linearity_pcnt': 0.1,
    'rho': 1e-2,
    'rho_adapt': False,
    'log_data': True,
    'verbose': not opt_surface,
}

# Setup coordinates for PNS surface
eps = 1e-6
u = torch.linspace(0, 2 * torch.pi, 50, device=torch_dev)
v = torch.linspace(0.2, 1, 50, device=torch_dev) ** 0.5
v = torch.clamp(v, eps, 1-eps)
bcm = body_charge_model(u, v, M_fourier_modes=7*2, N_hat_modes=100)
crds_pns = bcm.xyz_crds
# Echarge_mat = bcm.calc_efield_charge_matrix(crds_pns, 
#                                             urange=(u.min().item(), u.max().item()),
#                                             vrange=(v.min().item(), v.max().item()))
# torch.save(Echarge_mat.cpu(), './Echarge_mat.pt')
Echarge_mat = torch.load('./Echarge_mat.pt', map_location=torch_dev, weights_only=True)
# Echarge_rs = Echarge_mat.reshape(Echarge_mat.shape[0], bcm.M, bcm.N, 3)
# charge_mat = bcm.calc_charge_matrix()
# charge_mat = charge_mat.reshape(charge_mat.shape[0], bcm.M, bcm.N)
# breakpoint()

# Coordinates 
# xrange = 1.1 * (crds_pns[:, 0].max().item() - crds_pns[:, 0].min().item())
# yrange = 1.1 * (crds_pns[:, 1].max().item() - crds_pns[:, 1].min().item())
# zrange = 2 * 1.1 * crds_pns[:, 2].abs().max().item()
xrange = 0.4
yrange = 0.4
zrange = 0.4
fovs = (xrange, yrange, zrange)
crds = gen_grd(grd_size, fovs) + 2.32e-4
dx = (crds[1, 0, 0, 0] - crds[0, 0, 0, 0]).item()
dy = (crds[0, 1, 0, 1] - crds[0, 0, 0, 1]).item() 
dz = (crds[0, 0, 1, 2] - crds[0, 0, 0, 2]).item()
crds = crds.reshape((-1, 3)).to(torch_dev)

# Gradient linearity region
idx_dsv = torch.argwhere(crds.norm(dim=-1) < dsv/2)[:, 0]
crds_dsv = crds[idx_dsv]

# -------------------- Build gradient coil --------------------
grad_radius = 0.15
z_start = -0.15
z_end = 0.15

# Build gradient coil
fdir = '/local_mount/space/mayday/data/users/abrahamd/processing_scripts/insert_gradient/gradient_coil/'
pth = 'designs/surfopt=True_bmax=0.155_emax=4.79_L=192.2_eff=1.00_gmax=None.pt'
data = torch.load(fdir + pth, map_location=torch_dev)
zs_opt = data['zs']
as_opt = data['rs']
if not use_head_surf:
	zs_opt = torch.linspace(z_start, z_end, len(zs_opt), device=torch_dev)
	as_opt = torch.ones(len(zs_opt), device=torch_dev) * grad_radius
if gradient_type == 'stream':
    # nspline = 5
    # grad_length = z_end - z_start
    # zs_spline=torch.linspace(z_start, z_end, nspline, device=torch_dev).flip(dims=[0])
    # as_spline=torch.ones(nspline, device=torch_dev) * grad_radius
    # nspline = 15
    # z_end = grad_radius * 0.95
    # z_start = grad_radius * np.cos(torch.pi * 120 / 180)
    # zs_spline = torch.linspace(z_start, z_end, nspline, device=torch_dev).flip(dims=[0])
    # phis = torch.arccos(zs_spline / grad_radius)
    # as_spline = grad_radius * torch.sin(phis)
    # grad_coil = stream_func_coil(zs_spline=zs_opt,
    #                              as_spline=as_opt,
    # 							 num_z_modes=7,
    # 							 num_theta_modes=7,
    # 							#  z_bases='fourier',
    # 							 z_bases='chebyshev',
    # 							 theta_bases='fourier',)
    surf = elliptical_frustum(zs_spline=zs_opt, 
                              as_spline=as_opt,
                              bs_spline=as_opt)
    grad_coil = stream_func_surface_coil(surf, 
                                         num_v_modes=7,
                                         num_u_modes=7,
                                         u_bases='fourier',
                                         v_bases='chebyshev',
                                         device=torch_dev,
                                         dtype=torch.float32)
    Ncoeffs = grad_coil.M * grad_coil.K
elif gradient_type == 'matrix':
    loop_radius = .08
    nrad = 6
    nz = 4
    # z_start = -0.15
    # z_end = 0.24
    # grad_length = z_end - z_start
    # centers = matrix_coil.gen_pts_cylinder(nrad, nz).to(torch_dev)
    # centers[:, :2] *= grad_radius
    # centers[:, 2] *= (grad_length + 2 * grad_radius) / 2
    # centers[:, 2] += (z_start + z_end) / 2
    # normals = centers.clone()
    # normals[:, 2] = 0 # REMOVEME
    loop_radius = 0.05
    centers = matrix_coil.fibonacci_spherical_cap(nrad*nz, phi_max=torch.pi * 120 / 180).to(torch_dev)
    centers *= grad_radius
    normals = centers.clone()
    normals = normals / normals.norm(dim=-1, keepdim=True)
    radii = centers[:, 0] * 0 + loop_radius
    thetas_phis = torch.stack([torch.atan2(normals[..., 1], normals[..., 0]),
                            torch.arccos(normals[..., 2]), ], dim=-1)
    grad_coil = matrix_coil(radii, centers, thetas_phis)
    Ncoeffs = len(centers)
elif gradient_type == 'circular_z':
    grad_coil = circular_z_coil(
        # zs=torch.linspace(z_start, z_end, 50, device=torch_dev),
        # zs_spline=torch.linspace(z_start, z_end, 5, device=torch_dev),
        # rs_spline=torch.ones(5, device=torch_dev) * grad_radius,
        zs=torch.linspace(zs_opt.min().item(), zs_opt.max().item(), len(zs_opt), device=torch_dev),
        zs_spline=zs_opt,
        rs_spline=as_opt,
        lamda_spline=1e-2,
        M_fourier_modes=10,
        maxwell_pair=maxwell_pair,
    )
    Ncoeffs = grad_coil.Imat.shape[1]

# Define matrices
def G_theta(thetas):
    _, G, _ = grad_coil.build_field_matrices(crds_bfield=crds_pns[:1], crds_gfield=crds_dsv, crds_efield=crds_pns[:1])
    return G[..., grad_dir] * GRAD_UNITS / dstream # T/m -> mT/m/A
def L_theta(thetas):
    W = grad_coil.build_magnetic_energy_matrix() * INDUCTANCE_UNITS
    L = 0.5 * W / (dstream ** 2) # Inductance from energy formula
    eps = 1e-12
    L = torch.linalg.cholesky(L + eps * torch.eye(W.shape[0], device=torch_dev)).T
    return L
def E_theta(thetas):
    _, _, E = grad_coil.build_field_matrices(crds_bfield=crds_pns[:1], crds_gfield=crds_dsv[:1], crds_efield=crds_pns)
    return E.moveaxis(-1, 1) * EFIELD_UNITS / FREQ_UNITS / dstream # s V/m/Hz/A -> mV/m/kHz/A
def E_theta_total(thetas):
    _, _, Ecoil_mat = grad_coil.build_field_matrices(crds_bfield=crds_pns[:1], crds_gfield=crds_dsv[:1], crds_efield=crds_pns)
    E = bcm.build_efield_matrix(Echarge_mat, Ecoil_mat, l2_reg=1e-8)
    return E.moveaxis(-1, 1) * EFIELD_UNITS / FREQ_UNITS / dstream # s V/m/Hz/A -> mV/m/kHz/A
def P_theta(thetas):
    if gradient_type == 'stream':
        P = grad_coil.build_winding_tolerance_matrix()
        return P.moveaxis(-1, 1) * min_wire_spacing / dstream
    return None
def F_theta(thetas):
    if gradient_type == 'stream':
        F = grad_coil.build_current_boundary_matrix(verbose=True)
        return F
    return torch.zeros((1,Ncoeffs), device=torch_dev)
def C_theta(thetas):
    if gradient_type == 'matrix':
        return torch.eye(Ncoeffs, device=torch_dev)
    return None
def d_theta(thetas):
    if gradient_type == 'matrix':
        return torch.ones((Ncoeffs,), device=torch_dev) * max_ni_matrix
    return None
def loss_theta(thetas):
    if gradient_type == 'circular_z':
        rs = grad_coil.interp_rs(grad_coil.get_coil_zs())
        print(rs)
    return 0
G = G_theta([])
L = L_theta([])
# E = E_theta_total([])
E = E_theta([])
F = F_theta([])
g = torch.zeros(F.shape[0], device=torch_dev)
P = P_theta([])
C = C_theta([])
d = d_theta([])
def surf_opt(state_dict=None):
    # Solve via unrolled ADMM
    if gradient_type == 'stream':
        grad_coil.as_spline.coeff.requires_grad = True
        thetas = [
            grad_coil.as_spline.coeff,
        ]
        lrs = [1e-3]
    elif gradient_type == 'matrix':
        grad_coil.radii.requires_grad = True
        grad_coil.centers.requires_grad = True
        grad_coil.thetas_phis.requires_grad = True
        thetas = [
            grad_coil.radii,
            grad_coil.centers,
            grad_coil.thetas_phis,
        ]
        lrs = [1e-3, 1e-3, 1e-2]
    elif gradient_type == 'circular_z':
        grad_coil.spline.coeff.requires_grad = True
        grad_coil.zofs.requires_grad = True
        thetas = [
            grad_coil.spline.coeff,
            grad_coil.zofs,
        ]
        lrs = [1e-3]*2
    dct = unrolled_admm_general(thetas=thetas, 
                                G_theta=G_theta, L_theta=L_theta, 
                                C_theta=C_theta, E_theta=E_theta, 
                                P_theta=P_theta, F_theta=F_theta, 
                                Pmax=1,
                                d=d, g=g,
                                loss_theta=loss_theta,
                                admm_iters=50,
                                epochs=100*5,
                                state_dict=state_dict,
                                lr=lrs,
                                **admm_kwargs)
    if gradient_type == 'stream':
        grad_coil.as_spline.coeff.requires_grad = False
    elif gradient_type == 'matrix':
        grad_coil.radii.requires_grad = False
        grad_coil.centers.requires_grad = False
        grad_coil.thetas_phis.requires_grad = False  
    elif gradient_type == 'circular_z':
        grad_coil.spline.coeff.requires_grad = False
        grad_coil.zofs.requires_grad = False
    return dct
def coeff_opt(state_dict=None):
    # Solve via ADMM
    dct = admm_general(G=G, L=L, E=E, 
                       C=C, d=d,
                       F=F, g=g,
                       P=P, Pmax=1.0,
                    #    t=100.0,
                       state_dict=state_dict,
                       admm_iters=5_000,
                       **admm_kwargs)
    # admm_kwargs.pop('rho')
    # admm_kwargs.pop('rho_adapt')
    # admm_kwargs.pop('log_data')
    # dct = admm_general_cvxpy(G=G, L=L, E=E, 
    #                          C=C, d=d,
    #                          F=F, g=g,
    #                          P=P, Pmax=1,
    #                          admm_iters=5_000,
    #                          **admm_kwargs)
    
    # dct = quasi_convex_min_ratio(G=G, E=E, tstart=10,
    #                              L=L, Lmax=admm_kwargs['Lmax'], # mH
    #                              C=C, d=d,
    #                              F=F, g=g,
    #                              P=P, Pmax=1,
    #                              state_dict=state_dict,
    #                              bisection_iters=7,
    #                              admm_iters=5_000,
    #                              admm_iters_reduced=5_000,
    #                             #  linearity_pcnt=admm_kwargs['linearity_pcnt'],
    #                              rho=admm_kwargs['rho'],
    #                              rho_adapt=admm_kwargs['rho_adapt'],
    #                              verbose=admm_kwargs['verbose'],
    #                              log_data=admm_kwargs['log_data'],)
    
    # # Rescale
    # dct['x'] = dct['x'] * (admm_kwargs['Lmax'] / (L @ dct['x']).norm().square().item()) ** 0.5
    return dct

# Recompute E at shifted coordinates
print('SHIFTING in X')
crds_pns_old = crds_pns.clone()
crds_pns[..., 2] += 3e-2
E = E_theta_total([])
crds_pns = crds_pns_old.clone()

if opt_surface:
    dct = surf_opt()
    dct = coeff_opt(state_dict=dct)
else:
    dct = coeff_opt()
Gmin_actual = (G @ dct["x"]).min().item()
Lmax_actual = (L @ dct["x"]).norm().square().item()
Emax_actual = (E @ dct["x"]).norm(dim=-1).max().item()
print(f'Gmin: {Gmin_actual}')
print(f'Lmax: {Lmax_actual}')
print(f'Emax: {Emax_actual}')
print(F'E/G:  {Emax_actual / Gmin_actual:.2f}')
if P is not None:
    Pmax_actual = (P @ dct["x"]).norm(dim=-1).max().item()
    print(f'Pmax: {Pmax_actual}')
if F is not None:
    Fmax_actual = (F @ dct["x"] - g).abs().max().item()
    print(f'Fmax: {Fmax_actual}')

# Show rhos
for key in dct:
	if 'rho' in key:
		print(f'{key}: {dct[key]:1.3e}')

# # Plot G distribution
# plt.hist((G @ dct['x']).cpu(), bins=torch.linspace(0, 6, 100))
try:
    gmin_targ = dct['Gmin'][-1].item()
except:
    gmin_targ = admm_kwargs['Gmin']
    
# plt.axvline(gmin_targ, color='r', linestyle='--')

# # Plot diagnostics
# plt.figure()
# plt.semilogy(dct['r_pri'], label='Primal Residual')
# plt.semilogy(dct['s_dual'], label='Dual Residual')
# plt.legend()

# plt.figure(figsize=(14, 7))
# for i, key in enumerate(['Gmin', 'Lmax', 'Emax']):
# 	plt.subplot(3, 1, i+1)
# 	plt.plot(torch.tensor(dct[key]).cpu())
# 	plt.ylabel(key)
# plt.legend()

# # Show winding 
# if gradient_type == 'stream':
#     fig, ax = grad_coil.show_countour(stream_coeffs=dct['x'], dstream=dstream)
#     # ax.scatter(crds_pns[:, 0].cpu() * 1e2, crds_pns[:, 1].cpu() * 1e2, crds_pns[:, 2].cpu() * 1e2, color='orange', marker='.', alpha=0.1)
#     # ax.scatter(crds_dsv[:, 0].cpu() * 1e2, crds_dsv[:, 1].cpu() * 1e2, crds_dsv[:, 2].cpu() * 1e2, color='green', marker='.', alpha=0.05)
#     # fig, ax = grad_coil.show_current_density(stream_coeffs=dct['x'], num_theta=50, num_z=50)
#     # ax.axis('off')
#     bcm.show_surface(alpha=0.5, 
#                     ax=ax, fig=fig,)
# elif gradient_type == 'matrix':
#     fig, ax, axl = grad_coil.show_design(coeffs=dct['x'])
#     bcm.show_surface(alpha=0.5, 
#                     ax=ax, fig=fig,)
# elif gradient_type == 'circular_z':
#     fig, ax, axl = grad_coil.show_design(coeffs=dct['x'])
#     bcm.show_surface(alpha=0.5, 
#                     ax=ax, fig=fig,)


# # Show PNS surface
# efield_surf = E @ dct['x']
# fig, ax = bcm.show_surface(alpha=1.0, 
#                  fields=efield_surf, 
#                 #  ax=ax, fig=fig,
#                  colorbar_label=r'$||E||_2$ Per Unit Slew (mV/m / (T/m/s))',
#                  vmin=0, vmax=5)
# ax.axis('off')

# # Show stream function
# if gradient_type == 'stream':
#     fig, ax = grad_coil.show_design(coeffs=dct['x'],
#                                     show_1d=False,
#                                     colorbar=False)
#     ax.axis('off')

# Compute field using stream function 
crds_plt = gen_grd((101, 1, 101), (xrange, yrange, zrange)).to(torch_dev)[:, 0, :]
# crds_plt = gen_grd((220, 1, 220), (0.22, 0.22, 0.22)).to(torch_dev)[:, 0, :]
crds_plt -= crds_plt.reshape((-1,3)).mean(dim=0)
bfield, gfield, efield = grad_coil.evaluate_fields(coeffs=dct['x'], crds_bfield=crds_plt, crds_gfield=crds_plt, crds_efield=crds_plt)
linpcnt = admm_kwargs['linearity_pcnt'] if 'linearity_pcnt' in admm_kwargs else None
torch.save(bfield[..., -1].cpu(), f'./designs/bfield_linpcnt={linpcnt}_dsv={dsv*1e2:.0f}cm.pt')
# quit()
# bfield, gfield, efield = grad_coil.build_field_matrices(crds_bfield=crds_plt, crds_gfield=crds_plt, crds_efield=crds_plt)
# bfield = einsum(bfield, dct['x'], '... C d, C -> ... d')
# gfield = einsum(gfield, dct['x'], '... C d, C -> ... d')
# efield = einsum(efield, dct['x'], '... C d, C -> ... d')

# # Show gradient slices
# xrange = 0.15 * 2.3
# yrange = 0.15 * 2.3
# zrange = 0.15 * 2.3
# xz_crds = gen_grd((101, 1, 101), (xrange, yrange, zrange)).to(torch_dev)[:, 0, :]
# xz_crds -= xz_crds.reshape((-1,3)).mean(dim=0)
# xy_crds = gen_grd((101, 101, 1), (xrange, yrange, zrange)).to(torch_dev)[:, :, 0]
# xy_crds -= xy_crds.reshape((-1,3)).mean(dim=0)
# yz_crds = gen_grd((1, 101, 101), (xrange, yrange, zrange)).to(torch_dev)[0, :, :]
# yz_crds -= yz_crds.reshape((-1,3)).mean(dim=0)
# bfield_xz, gfield_xz, efield_xs = grad_coil.evaluate_fields(stream_coeffs=dct['x'], crds_bfield=xz_crds, crds_gfield=xz_crds, crds_efield=xz_crds)
# bfield_xy, gfield_xy, efield_xy = grad_coil.evaluate_fields(stream_coeffs=dct['x'], crds_bfield=xy_crds, crds_gfield=xy_crds, crds_efield=xy_crds)
# bfield_yz, gfield_yz, efield_yz = grad_coil.evaluate_fields(stream_coeffs=dct['x'], crds_bfield=yz_crds, crds_gfield=yz_crds, crds_efield=yz_crds)
# # xy = bfield_xy[..., -1] * BFIELD_UNITS
# # xz = bfield_xz[..., -1] * BFIELD_UNITS
# # yz = bfield_yz[..., -1] * BFIELD_UNITS
# xy = gfield_xy[..., grad_dir] * GRAD_UNITS
# xz = gfield_xz[..., grad_dir] * GRAD_UNITS
# yz = gfield_yz[..., grad_dir] * GRAD_UNITS
# fig = show_3d_slices(xy, xz, yz, xy_crds, xz_crds, yz_crds,
#                         #  vmin=-Gmin_actual * dsv/2, vmax=Gmin_actual * dsv/2, 
#                          vmin=Gmin_actual * 0.5, vmax=Gmin_actual * 1.5, 
#                          colorscale='RdBu_r')
# fig.show()
# quit()

# Compute fields
fields = [[bfield[..., -1].rot90() * BFIELD_UNITS / dstream, 
              gfield[..., grad_dir].rot90() * GRAD_UNITS / dstream, 
              efield.norm(dim=-1).rot90() * EFIELD_UNITS / FREQ_UNITS / dstream]]
if gradient_type == 'stream':
    from magopt.sim.wire_fields import parametric_wire
    # xyz_countours, dstream = grad_coil.stream_to_winding(stream_coeffs=dct['x'], dstream=dstream)
    xyz_countours, dstream = grad_coil.stream_to_contour(stream_coeffs=dct['x'], dstream=dstream)
    bfield_cnt = bfield * 0
    gfield_cnt = gfield * 0
    efield_cnt = efield * 0
    Iwire = dstream
    for i, xyz_contour in enumerate(xyz_countours):
        wire = parametric_wire(wire_pts=xyz_contour[:-1].to(torch_dev), verbose=False)
        bfield_cnt += wire.calc_bfield(crds_plt) * Iwire
        gfield_cnt += wire.calc_bfield_jacobian(crds_plt)[..., -1, :] * Iwire
        efield_cnt += wire.calc_mag_potential(crds_plt) * Iwire
    # bfield_cnt *= bfield.norm() / bfield_cnt.norm()
    # gfield_cnt *= gfield.norm() / gfield_cnt.norm()
    # efield_cnt *= efield.norm() / efield_cnt.norm()
    fields += [[bfield_cnt[..., -1].rot90() * BFIELD_UNITS / dstream, 
                gfield_cnt[..., grad_dir].rot90() * GRAD_UNITS / dstream, 
                efield_cnt.norm(dim=-1).rot90() * EFIELD_UNITS / FREQ_UNITS / dstream]]
# torch.save(xyz_countours, './designs/head_tight_v0/wire_path_Gx.pt')
# quit()

# Setup plotting
# gmin_targ = 0.5
titles = ['B-field (mT)', 'G-field (mT/m)', 'E-field (mV/m/kHz)']
# vmins = [-gmin_targ * dsv/2, gmin_targ * 0.5, 0]
# vmaxs = [+gmin_targ * dsv/2, gmin_targ * 1.5, 10]
vmins = [-gmin_targ * dsv/2, 0, 0]
vmaxs = [+gmin_targ * dsv/2, 1.0, 10]
# vmins = [None, None, None]
# vmaxs = [None, None, None]
cmaps = ['jet', 'RdBu_r', 'Reds']
extent = [crds_plt[..., 0].min().item() * 1e2, crds_plt[..., 0].max().item() * 1e2,
		  crds_plt[..., 2].min().item() * 1e2, crds_plt[..., 2].max().item() * 1e2]

# Coil cross section
zs = grad_coil.get_coil_zs()
rs = grad_coil.interp_rs(zs)

# Plot fields
I = len(fields)
plt.figure(figsize=(14, 7))
for k in range(3):
	for i in range(I):
		plt.subplot(2, 3, i*3+k+1)
		if vmins[k] is None or vmaxs[k] is None:
			vmin = fields[i][k].median() - 1 * fields[i][k].std()
			vmax = fields[i][k].median() + 1 * fields[i][k].std()
		else:
			vmin = vmins[k]	
			vmax = vmaxs[k]
		plt.imshow(fields[i][k].cpu(), cmap=cmaps[k], vmin=vmin, vmax=vmax, extent=extent)
		plt.colorbar(shrink=0.6)
		plt.title(titles[k])
		plt.xlabel('X [cm]')
		plt.ylabel('Z [cm]')
		plt.plot(rs.cpu() * 1e2, zs.cpu() * 1e2, color='black')
		plt.plot(-rs.cpu() * 1e2, zs.cpu() * 1e2, color='black')
plt.tight_layout()
plt.show()

