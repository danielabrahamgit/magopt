import torch
import numpy as np
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from magopt.viz import show_3d_slices
from magopt.utils import gen_pts_sphere_surf, gen_pts_ellip_surf_random, gen_grd
from magopt.gradient_coils import stream_func_coil, matrix_coil, circular_z_coil, stream_func_surface_coil
from magopt.gradient_coils.gradient_surfaces import (
    elliptical_frustum, 
    planar_surface,
    planar_curved_surface,
)
from magopt.pns.charge_model import body_charge_model
from magopt.pns.pediatric_model import body_model
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
dsv = 0.16 # m (diameter of spherical volume to optimize over)
dsvx = 0.2
dsvy = 0.15
dsvz = 0.2
grad_dir = 1 # 0 --> x, 1 --> y, 2 --> z
GRAD_UNITS = 1e3 # T -> mT
BFIELD_UNITS = 1e3 # T -> mT
EFIELD_UNITS = 1e3 # V/m -> mV/m
FREQ_UNITS = 1e-3 # Hz -> kHz
INDUCTANCE_UNITS = 1e3 # H -> mH
max_ni_matrix = 25
min_wire_spacing = 5e-3 # m (minimum wire spacing)
dstream = 1 # A (stream function step size AKA current through coil)
opt_surface = False
gradient_type = 'stream'
# gradient_type = 'matrix'
admm_kwargs = {
    'lamdaG': 1e0,
    # 'Gmin': 0.5,
    # 'lamdaL': 3e1,
    'Lmax': 0.3, # mH
    # 'lamdaE': 1e0,
    'Emax': 15,
    # 'linearity_pcnt': 0.1,
    'rho': 1e-2,
    'rho_adapt': False,
    'log_data': True,
    'verbose': not opt_surface,
}

# Setup coordinates from shreyas
zs = torch.linspace(-0.1, 0.1, torch_dev) # 20cm in Z
zs -= zs.mean()
rxs = torch.ones_like(zs) * 0.15 # 30cm in X
rys = torch.ones_like(zs) * 0.10 # 20cm in Y
thetas = torch.linspace(0, 2 * torch.pi, 50, device=torch_dev)
thetas, zs = torch.meshgrid(thetas, zs, indexing='ij')
crds_pns = torch.stack([rxs * torch.cos(thetas),
                        rys * torch.sin(thetas),
                        zs], dim=-1)
crds_pns = crds_pns.reshape(-1, 3)

# Coordinates 
xrange = 1.1 * (crds_pns[:, 0].max().item() - crds_pns[:, 0].min().item())
yrange = 1.1 * (crds_pns[:, 1].max().item() - crds_pns[:, 1].min().item())
zrange = 1.1 * (crds_pns[:, 2].max().item() - crds_pns[:, 2].min().item())
fovs = (xrange, yrange, zrange)
crds = gen_grd(grd_size, fovs) + 2.32e-4
dx = (crds[1, 0, 0, 0] - crds[0, 0, 0, 0]).item()
dy = (crds[0, 1, 0, 1] - crds[0, 0, 0, 1]).item() 
dz = (crds[0, 0, 1, 2] - crds[0, 0, 0, 2]).item()
crds = crds.reshape((-1, 3)).to(torch_dev)

# Gradient linearity region
dsv_vec = torch.tensor([dsvx, dsvy, dsvz], device=torch_dev)
idx_dsv = torch.argwhere((crds/dsv_vec).norm(dim=-1) < 1/2)[:, 0]
crds_dsv = crds[idx_dsv]

# -------------------- Build gradient coil --------------------
if gradient_type == 'stream':
    zs_spline = torch.linspace(zs.min().item(), zs.max().item(), 5, device=torch_dev)
    as_spline = torch.ones(5, device=torch_dev) * rxs.max().item() * 1.2
    bs_spline = torch.ones(5, device=torch_dev) * rys.max().item() * 1.2
    
    
    # surf = elliptical_frustum(zs_spline=zs_spline, 
    #                           as_spline=as_spline,
    #                           bs_spline=bs_spline,
    #                           device=torch_dev)
    # grad_coil = stream_func_surface_coil(surf, 
    #                                      num_v_modes=7,
    #                                      num_u_modes=7,
    #                                      u_bases='fourier',
    #                                      v_bases='chebyshev',
    #                                      device=torch_dev,
    #                                      dtype=torch.float32)
    surf = planar_surface(u_axis=torch.tensor([1, 0, 0], device=torch_dev, dtype=torch.float32),
                          v_axis=torch.tensor([0, 0, 1], device=torch_dev, dtype=torch.float32),
                          center=torch.tensor([0, rys.abs().max().item()*1.2, 0], device=torch_dev, dtype=torch.float32),
                          width_u=rxs.abs().max().item() * 2.5,
                          width_v=zs.max().item() - zs.min().item(),
                          device=torch_dev)
    # surf = planar_curved_surface(u_axis=torch.tensor([1, 0, 0], device=torch_dev, dtype=torch.float32),
    #                              v_axis=torch.tensor([0, 0, 1], device=torch_dev, dtype=torch.float32),
    #                              center=torch.tensor([0, rys.abs().max().item()*1.2, 0], device=torch_dev, dtype=torch.float32),
    #                              width_u=rxs.abs().max().item() * 2.5,
    #                              width_v=zs.max().item() - zs.min().item(),
    #                              height_curve=0.15,
    #                              poly_degree=6,
    #                              device=torch_dev)
    # grad_coil = stream_func_surface_coil(surf, 
    #                                      num_v_modes=9,
    #                                      num_u_modes=9,
    #                                      u_bases='chebyshev', 
    #                                      v_bases='chebyshev', 
    #                                      device=torch_dev,
    #                                      dtype=torch.float32)
    grad_coil = stream_func_coil(zs_spline=zs_spline,
                                 as_spline=as_spline,
                                 bs_spline=bs_spline,
    							 num_z_modes=7,
    							 num_theta_modes=7,
    							#  z_bases='fourier',
    							 z_bases='chebyshev',
    							 theta_bases='fourier',)
    Ncoeffs = grad_coil.M * grad_coil.K
elif gradient_type == 'matrix':
    loop_radius = .05
    nrad = 10
    nz = 4
    z_start = zs.min().item()
    z_end = zs.max().item()
    grad_length = z_end - z_start
    centers = matrix_coil.gen_pts_cylinder(nrad, nz).to(torch_dev)
    centers[:, 0] *= rxs.max().item() * 1.2
    centers[:, 1] *= rys.max().item() * 1.2
    centers[:, 2] *= (grad_length) / 2
    centers[:, 2] += (z_start + z_end) / 2
    normals = centers.clone()
    normals[:, 2] = 0
    normals = normals / normals.norm(dim=-1, keepdim=True)
    radii = centers[:, 0] * 0 + loop_radius
    thetas_phis = torch.stack([torch.atan2(normals[..., 1], normals[..., 0]),
                            torch.arccos(normals[..., 2]), ], dim=-1)
    grad_coil = matrix_coil(radii, centers, thetas_phis)
    Ncoeffs = len(centers)

# Define matrices
def G_theta(thetas):
    _, G, _ = grad_coil.build_field_matrices(crds_bfield=crds_pns[:1], crds_gfield=crds_dsv, crds_efield=crds_pns[:1])
    return G[..., grad_dir] * GRAD_UNITS / dstream # T/m -> mT/m/A
def L_theta(thetas):
    if gradient_type == 'stream':
        W = grad_coil.build_magnetic_energy_matrix() * INDUCTANCE_UNITS
        L = 0.5 * W / (dstream ** 2) # Inductance from energy formula
        eps = 1e-12
        L = torch.linalg.cholesky(L + eps * torch.eye(W.shape[0], device=torch_dev)).T
        return L
    return torch.eye(Ncoeffs, device=torch_dev) * 1e-9
def E_theta(thetas):
    _, _, E = grad_coil.build_field_matrices(crds_bfield=crds_pns[:1], crds_gfield=crds_dsv[:1], crds_efield=crds_pns)
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

# Plot G distribution
plt.hist((G @ dct['x']).cpu(), bins=torch.linspace(0, 6, 100))
try:
    gmin_targ = dct['Gmin'][-1].item()
except:
    gmin_targ = admm_kwargs['Gmin']
    
plt.axvline(gmin_targ, color='r', linestyle='--')

# Plot diagnostics
plt.figure()
plt.semilogy(dct['r_pri'], label='Primal Residual')
plt.semilogy(dct['s_dual'], label='Dual Residual')
plt.legend()

plt.figure(figsize=(14, 7))
for i, key in enumerate(['Gmin', 'Lmax', 'Emax']):
	plt.subplot(3, 1, i+1)
	plt.plot(torch.tensor(dct[key]).cpu())
	plt.ylabel(key)
plt.legend()

# Show winding 
if gradient_type == 'stream':
    fig, ax = grad_coil.show_countour(stream_coeffs=dct['x'], dstream=dstream)
elif gradient_type == 'matrix':
    fig, ax, axl = grad_coil.show_design(coeffs=dct['x'])
    ax.scatter(crds_pns[:, 0].cpu() * 1e2, crds_pns[:, 1].cpu() * 1e2, crds_pns[:, 2].cpu() * 1e2, color='orange', marker='.', alpha=0.1)
    ax.scatter(crds_dsv[:, 0].cpu() * 1e2, crds_dsv[:, 1].cpu() * 1e2, crds_dsv[:, 2].cpu() * 1e2, color='green', marker='.', alpha=0.05)
    ax.set_aspect('equal')


# Show PNS surface
Rx = rxs.abs().max().item()
Ry = rys.abs().max().item()
thetas = torch.linspace(0, 2 * torch.pi, 100, device=torch_dev)
zs = torch.linspace(zs.min().item(), zs.max().item(), 100, device=torch_dev)
thetas, zs = torch.meshgrid(thetas, zs, indexing='ij')
X = Rx * torch.cos(thetas)
Y = Ry * torch.sin(thetas)
Z = zs
ax.plot_surface(X.cpu() * 1e2, Y.cpu() * 1e2, Z.cpu() * 1e2, color='orange', alpha=0.2)

# Show DSV surface
thetas = torch.linspace(0, 2 * torch.pi, 100, device=torch_dev)
phis = torch.linspace(0, torch.pi, 100, device=torch_dev)
thetas, phis = torch.meshgrid(thetas, phis, indexing='ij')
X = dsvx * torch.cos(thetas) * torch.sin(phis) / 2
Y = dsvy * torch.sin(thetas) * torch.sin(phis) / 2
Z = dsvz * torch.cos(phis) / 2
ax.plot_surface(X.cpu() * 1e2, Y.cpu() * 1e2, Z.cpu() * 1e2, color='green', alpha=0.1)
# ax.scatter(crds_pns[:, 0].cpu() * 1e2, crds_pns[:, 1].cpu() * 1e2, crds_pns[:, 2].cpu() * 1e2, color='orange', marker='.', alpha=0.1)
# ax.scatter(crds_dsv[:, 0].cpu() * 1e2, crds_dsv[:, 1].cpu() * 1e2, crds_dsv[:, 2].cpu() * 1e2, color='green', marker='.', alpha=0.05)
ax.set_xlabel('X [cm]')
ax.set_ylabel('Y [cm]')
ax.set_zlabel('Z [cm]')
ax.set_aspect('equal')
ax.axis('off')

# Show stream function
if gradient_type == 'stream':
    fig, ax = grad_coil.show_design(coeffs=dct['x'],
                                    show_1d=False,
                                    colorbar=False)
    ax.axis('off')

# Compute field using stream function 
if grad_dir == 0:
    crds_plt = gen_grd((101, 1, 101), (xrange, yrange, zrange)).to(torch_dev)[:, 0, :]
elif grad_dir == 1:
    crds_plt = gen_grd((1, 101, 101), (xrange, yrange, zrange)).to(torch_dev)[0, :, :]
elif grad_dir == 2:
    crds_plt = gen_grd((101, 1, 101), (xrange, yrange, zrange)).to(torch_dev)[:, 0, :]
crds_plt -= crds_plt.reshape((-1,3)).mean(dim=0)
bfield, gfield, efield = grad_coil.evaluate_fields(coeffs=dct['x'], crds_bfield=crds_plt, crds_gfield=crds_plt, crds_efield=crds_plt)
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
# gmin_targ = 0.4
titles = ['B-field (mT)', 'G-field (mT/m)', 'E-field (mV/m/kHz)']
vmins = [-gmin_targ * dsv/2, gmin_targ * 0.5, 0]
vmaxs = [+gmin_targ * dsv/2, gmin_targ * 1.5, 10]
# vmins = [None, None, None]
# vmaxs = [None, None, None]
cmaps = ['jet', 'RdBu_r', 'Reds']
if grad_dir == 1:
    extent = [crds_plt[..., 1].min().item() * 1e2, crds_plt[..., 1].max().item() * 1e2,
              crds_plt[..., 2].min().item() * 1e2, crds_plt[..., 2].max().item() * 1e2]
else:
    extent = [crds_plt[..., 0].min().item() * 1e2, crds_plt[..., 0].max().item() * 1e2,
              crds_plt[..., 2].min().item() * 1e2, crds_plt[..., 2].max().item() * 1e2]

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
		plt.xlabel('Y [cm]')
		plt.ylabel('Z [cm]')
plt.tight_layout()
plt.show()

