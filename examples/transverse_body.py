import torch

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from magopt.utils import gen_pts_sphere_surf, gen_pts_ellip_surf_random, gen_grd
from magopt.gradient_coils import stream_func_coil
from magopt.optim_admm import admm_general
from typing import Optional
from einops import einsum

# Parameters
torch_dev = torch.device(4)
grad_dir = 0 # 0 --> x, 1 --> y, 2 --> z

# Setup coordinates for PNS surfaace
pns_x = 0.1 # m (surface ellipsoid diameter)
pns_y = 0.1 # m (surface ellipsoid diameter)
pns_z = 0.15 # m (Cylinder height)
thetas = torch.linspace(0, 2 * torch.pi, 100, device=torch_dev)
zs = torch.linspace(-pns_z / 2, pns_z / 2, 20, device=torch_dev)
thetas, zs = torch.meshgrid(thetas, zs, indexing='ij')
crds_pns = torch.stack([pns_x * torch.cos(thetas) / 2,
						pns_y * torch.sin(thetas) / 2,
						zs], dim=-1)
crds_pns = crds_pns.reshape(-1, 3)

# Setup coordinates for gradient region
dsv = 0.1 # m (diameter of spherical volume to optimize over)
dsv_vec = torch.tensor([dsv, dsv, dsv], device=torch_dev)
crds = gen_grd((20,)*3, dsv_vec.tolist()).to(torch_dev)
crds_flt = crds.reshape(-1, 3)
crds_dsv = crds[(2 * crds / dsv_vec).norm(dim=-1) <= 1.0]

# Build gradient coil
grad_radius = 0.15 / 2
grad_length = 0.2
nspline = 5
grad_coil = stream_func_coil(zs_spline=torch.linspace(-1/2, 1/2, nspline, device=torch_dev) * grad_length, 
							 as_spline=torch.ones(nspline, device=torch_dev) * grad_radius, 
							 bs_spline=torch.ones(nspline, device=torch_dev) * grad_radius,
							 num_z_modes=11,
							 num_theta_modes=3,
							 z_bases='fourier',
							#  z_bases='chebyshev',
							 theta_bases='fourier',)

# # Show all basis functions
# fig = plt.figure(figsize=(10, 10))
# K = grad_coil.K // 2
# M = grad_coil.M // 2
# for k in range(K):
#     for m in range(M):
#         ax = fig.add_subplot(K, M, k * M + m + 1, projection='3d')
#         coeffs = torch.zeros(grad_coil.K * grad_coil.M, device=torch_dev)
#         coeffs[k * grad_coil.M + m] = 1.0
#         grad_coil.show_design(stream_coeffs=coeffs, show_1d=False, ax=ax, fig=fig, 
#                               colorbar=False,
#                               num_theta=100,
#                               num_z=50)
#         ax.axis('off')
#         ax.set_axis_off()
#         ax.margins(0)
# fig.subplots_adjust(hspace=0, wspace=0)
# # plt.tight_layout()
# plt.show()
# quit()
        
# Define gradient, inductance, and efield matrices
# TODO why does E-field scaling effect things?
bmat, gmat, emat = grad_coil.build_field_matrices(crds_bfield=crds_pns, crds_gfield=crds_dsv, crds_efield=crds_pns)
Jmat = grad_coil.build_current_boundary_matrix(verbose=True)
dphi_mat, upper_bound = grad_coil.build_winding_tolerance_matrix(dstream=1, min_wire_spacing=2e-3, num_theta=50, num_zs=50)
G = gmat[:, :, grad_dir] * 1e3 # T/m -> mT/m
E = emat.moveaxis(-1, 1) * 1e3 / 1e-3 # s V/m -> mV/m/kHz
dP = dphi_mat.moveaxis(-1, 1)
E = dP / upper_bound.max().item()
upper_bound = upper_bound / upper_bound.max().item()
# E = bmat.moveaxis(-1, 1) * 1e6 # T -> uT
# W = grad_coil.build_magnetic_energy_matrix(batch_size=2**4, debug_method=True)
# L = torch.linalg.cholesky(W + 1e-6 * torch.eye(W.shape[0], device=torch_dev))
L = torch.eye(G.shape[-1], device=torch_dev)
F = Jmat
g = torch.zeros(Jmat.shape[0], device=torch_dev)

# Solve via ADMM
dct = admm_general(G=G, L=L, E=E, 
				   F=F, g=g,
				   lamdaG=1e0, 
				  #  Gmin=1.0,
				#    lamdaL=1e0,
				   Lmax=1e4,
				  #  lamdaE=1e0,
				#    Emax=5, 
				   Emax=upper_bound.max().item(),
				  #  rho=1e-2,
				   admm_iters=5_000,
				   rho_adapt=False,
				   log_data=True,
				   verbose=True)
Gmin_actual = (G @ dct["x"]).min().item()
Lmax_actual = (L @ dct["x"]).norm().square().item()
Emax_actual = (E @ dct["x"]).norm(dim=-1).max().item()
Fmax_actual = (F @ dct["x"] - g).abs().max().item()
print(f'Gmin: {Gmin_actual}')
print(f'Lmax: {Lmax_actual}')
print(f'Emax: {Emax_actual}')
print(f'Fmax: {Fmax_actual}')

# Show rhos
for key in dct:
	if 'rho' in key:
		print(f'{key}: {dct[key]:1.3e}')

# Plot G distribution
plt.hist((G @ dct['x']).cpu(), bins=torch.linspace(0, 6, 100))
plt.axvline(dct['Gmin'][-1].item(), color='r', linestyle='--')

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

# Show winding and gradient/pns regions
fig, ax = grad_coil.show_countour(stream_coeffs=dct['x'], dstream=1)
ax.scatter(crds_pns[:, 0].cpu() * 1e2, crds_pns[:, 1].cpu() * 1e2, crds_pns[:, 2].cpu() * 1e2, color='orange', marker='.', alpha=0.1)
ax.scatter(crds_dsv[:, 0].cpu() * 1e2, crds_dsv[:, 1].cpu() * 1e2, crds_dsv[:, 2].cpu() * 1e2, color='green', marker='.', alpha=0.05)
ax.set_xlabel('X [cm]')
ax.set_ylabel('Y [cm]')
ax.set_zlabel('Z [cm]')
# grad_coil.show_current_density(stream_coeffs=dct['x'], num_theta=100, num_z=50,
#                                ax=ax, fig=fig)
fig, ax = grad_coil.show_design(stream_coeffs=dct['x'], 
                                colorbar=False, 
                                show_1d=False) # Show stream function too
ax.axis('off')

# Compute field using stream function 
crds_plt = gen_grd((101, 1, 101), (grad_radius*2, grad_radius*2, grad_length)).to(torch_dev)[:, 0, :]
crds_plt -= crds_plt.reshape((-1,3)).mean(dim=0)
bfield, gfield, efield = grad_coil.evaluate_fields(stream_coeffs=dct['x'], crds_bfield=crds_plt, crds_gfield=crds_plt, crds_efield=crds_plt)
# bfield, gfield, efield = grad_coil.build_field_matrices(crds_bfield=crds_plt, crds_gfield=crds_plt, crds_efield=crds_plt)
# bfield = einsum(bfield, dct['x'], '... C d, C -> ... d')
# gfield = einsum(gfield, dct['x'], '... C d, C -> ... d')
# efield = einsum(efield, dct['x'], '... C d, C -> ... d')

# Compute fields using contours of stream function 
from magopt.sim.wire_fields import parametric_wire
xyz_countours, dstream = grad_coil.stream_to_contour(stream_coeffs=dct['x'], dstream=1)
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

# Setup plotting
fields_stm = [bfield[..., -1].rot90() * 1e3, gfield[..., grad_dir].rot90() * 1e3, efield.norm(dim=-1).rot90() * 1e6]
fields_cnt = [bfield_cnt[..., -1].rot90() * 1e3, gfield_cnt[..., grad_dir].rot90() * 1e3, efield_cnt.norm(dim=-1).rot90() * 1e6]
fields = [fields_stm, fields_cnt]
titles = ['B-field (mT)', 'G-field (mT/m)', 'E-field (mV/m/kHz)']
vmins = [-Gmin_actual * dsv/2, Gmin_actual * 0.5, 0]
vmaxs = [+Gmin_actual * dsv/2, Gmin_actual * 1.5, 10]
# vmins = [None, None, None]
# vmaxs = [None, None, None]
cmaps = ['jet', 'RdBu_r', 'Reds']
extent = [crds_plt[..., 0].min().item() * 1e2, crds_plt[..., 0].max().item() * 1e2,
		  crds_plt[..., 2].min().item() * 1e2, crds_plt[..., 2].max().item() * 1e2]

# Plot fields
plt.figure(figsize=(14, 7))
for k in range(3):
	for i in range(2):
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
plt.tight_layout()
plt.show()

