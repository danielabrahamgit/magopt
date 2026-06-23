import torch
import numpy as np
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from magopt.viz import show_3d_slices
from magopt.utils import gen_pts_sphere_surf, gen_pts_ellip_surf_random, gen_grd
from magopt.sim.analytic import calc_inductance_loop, _transform_coordinates
from magopt.gradient_coils import (
    stream_func_coil, 
    matrix_coil, 
    circular_z_coil, 
    stream_func_surface_coil,
    multi_stream,
)
from magopt.gradient_coils.gradient_surfaces import (
    elliptical_frustum, 
    planar_surface,
    planar_curved_surface,
    planar_arb_srface,
    parametric_cyl_surface,
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
torch_dev = torch.device(4)
grd_size = (41,)*3
dsvx = 0.3 * 0 + 0.26# * 0 + 0.16
dsvy = 0.2 * 0 + 0.16# * 0 + 0.11
dsvz = 0.2 * 0 + 0.16# * 0 + 0.11
grad_dir = 2 # 0 --> x, 1 --> y, 2 --> z
GRAD_UNITS = 1e3 # T -> mT
BFIELD_UNITS = 1e3 # T -> mT
EFIELD_UNITS = 1e3 # V/m -> mV/m
FREQ_UNITS = 1e-3 # Hz -> kHz
INDUCTANCE_UNITS = 1e3 # H -> mH
coil_z_ofs = -0.05
admm_iters = 5_000 * 3
max_ni_matrix = 20
min_wire_spacing = 1e-3 * 1 # m (minimum wire spacing)
dstream = 1 # A (stream function step size AKA current through coil)
opt_surface = False
show_pyvista = False
# gradient_type = 'stream_gx'
# gradient_type = 'stream_half_cyl'
gradient_type = 'matrix'
admm_kwargs = {
    'lamdaG': 1e0,
    # 'Gmin': 0.1,
    # 'lamdaL': 3e1,
    'Lmax': 0.15*2, # mH
    # 'lamdaE': 1e0,
    'Emax': 15,
    # 'linearity_pcnt': 0.1,
    'rho': 1e-2,
    'rho_adapt': False,
    'log_data': True,
    'verbose': not opt_surface,
}

# Setup coordinates for PNS surface as an ellipsoid
# age = '7 yr'
# dct = body_model[age][1]
# zs = torch.linspace(-dsvz/2, dsvz/2, grd_size[0], device=torch_dev) * 1.5
# rxs = torch.ones_like(zs) * dsvx / 2
# rys = torch.ones_like(zs) * dsvy / 2
# thetas = torch.linspace(0, 2 * torch.pi, grd_size[0], device=torch_dev)
# thetas, zs = torch.meshgrid(thetas, zs, indexing='ij')
# crds_pns = torch.stack([rxs * torch.cos(thetas),
#                         rys * torch.sin(thetas),
#                         zs], dim=-1)
# crds_pns = crds_pns.reshape(-1, 3)
eps = 1e-6
u = torch.linspace(0, 2 * torch.pi, 50, device=torch_dev)
v = torch.linspace(0.0, 1, 50, device=torch_dev)
v = torch.clamp(v, eps, 1-eps)
bcm = body_charge_model(u, v, M_fourier_modes=7*2, N_hat_modes=100,
                        ofs=torch.tensor([0, 0, 0.25], device=torch_dev))
crds_pns = bcm.xyz_crds

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
# idx_dsv = torch.argwhere((crds/dsv_vec).norm(dim=-1) < 1/2)[:, 0]
idx_dsv = torch.argwhere(torch.logical_and(
    (crds/dsv_vec)[..., :2].norm(dim=-1) <= 1/2,
    crds[..., 2].abs() <= dsvz/2,
))[:, 0]
crds_dsv = crds[idx_dsv]

# -------------------- Build gradient coil --------------------
cyl_width = 0.34
cyl_height = 0.22
cyl_length = 0.4
if gradient_type == 'stream_gx':
    cyl_height = 0.27
    cyl_length = 0.4
    cyl_width = 0.32
    # Construct two parallel plates separate by gx_width
    planar_kwargs = {'u_axis': torch.tensor([0, 1, 0], 
                                          device=torch_dev, dtype=torch.float32),
                    'v_axis': torch.tensor([0, 0, 1], 
                                         device=torch_dev, dtype=torch.float32),
                    'width_u': cyl_height,
                    'width_v': cyl_length,
                    'height_curve': -0.1*0,
                    'poly_degree': 4,
                    'device': torch_dev}
    planar_kwargs2 = planar_kwargs.copy()
    planar_kwargs2['height_curve'] = -planar_kwargs['height_curve']
    stream_func_kwargs = {'num_v_modes': 5,
                          'num_u_modes': 5,
                          'u_bases': 'chebyshev',
                          'v_bases': 'chebyshev',
                          'device': torch_dev,
                          'dtype': torch.float32}
    surf1 = planar_curved_surface(center=torch.tensor([cyl_width/2, 0, coil_z_ofs], 
                                                      device=torch_dev, 
                                                      dtype=torch.float32),
                                  **planar_kwargs)
    grad1 = stream_func_surface_coil(surf1,
                                      **stream_func_kwargs)
    surf2 = planar_curved_surface(center=torch.tensor([-cyl_width/2, 0, coil_z_ofs], 
                                                      device=torch_dev, 
                                                      dtype=torch.float32),
                                  **planar_kwargs2)
    grad2 = stream_func_surface_coil(surf2,
                                      **stream_func_kwargs)
    grad_coil = multi_stream([grad1, grad2])
elif gradient_type == 'stream_cyl':
    # Spline parameters
    zs_spline = torch.linspace(-cyl_length/2, cyl_length/2, 5, device=torch_dev)
    as_spline = torch.ones(5, device=torch_dev) * rxs.max().item() * 1.2
    bs_spline = torch.ones(5, device=torch_dev) * rys.max().item() * 1.2
    
    # Surface
    surf = elliptical_frustum(zs_spline=zs_spline,
                              as_spline=as_spline,
                              bs_spline=bs_spline,
                              device=torch_dev)
    
    # Coil
    grad_coil = stream_func_surface_coil(surf, 
                                         num_v_modes=5,
                                         num_u_modes=5,
                                         u_bases='fourier', 
                                         v_bases='fourier', 
                                         device=torch_dev,
                                         dtype=torch.float32)
elif gradient_type == 'stream_half_cyl':
    # bases = 'fourier'
    bases = 'chebyshev'
    # bases = 'triangular'
    # bases = 'gauss'
    num_u_modes = 6
    num_v_modes = 6
    u_axis = torch.tensor([1, 0, 0], device=torch_dev, dtype=torch.float32)
    v_axis = torch.tensor([0, 0, 1], device=torch_dev, dtype=torch.float32)
    center = torch.tensor([0, cyl_height/2, coil_z_ofs], device=torch_dev, dtype=torch.float32)
    def func(u):
        return (-1) * cyl_height * (1 - (2*u - 1) ** 2) ** 0.5
    def dfdu(u):
        return (-1) * -cyl_height * (4 * u - 2) / ((1 - (2*u - 1) ** 2) ** 0.5)
    # center1 = center.clone() * -1
    # surf1 = planar_arb_srface(func=func, dfdu=dfdu,
    #                           u_axis=u_axis,
    #                           v_axis=v_axis,
    #                           center=center1,
    #                           width_u=cyl_width,
    #                           width_v=cyl_length,
    #                           device=torch_dev,)
    wedge = 120 # degrees
    theta_start = 3*torch.pi/2 - wedge/2 * torch.pi / 180
    theta_end = -torch.pi/2 + wedge/2 * torch.pi / 180
    b1_axis = u_axis.clone()
    b2_axis = b1_axis * 0
    b2_axis[1] = 1
    scale_b1 = cyl_width / 2
    h0 = 1 + np.cos(wedge * torch.pi / 360)
    scale_b2 = cyl_height / h0
    center_circ = center.clone() * 0
    center_circ[1] = -scale_b2 * np.cos(wedge * torch.pi / 360)/2
    center_circ[2] = coil_z_ofs
    def b1_of_u(u):
        u_scaled = theta_start + u * (theta_end - theta_start)
        return torch.cos(u_scaled) * scale_b1
    def b2_of_u(u):
        u_scaled = theta_start + u * (theta_end - theta_start)
        return torch.sin(u_scaled) * scale_b2
    def db1_du(u):
        u_scaled = theta_start + u * (theta_end - theta_start)
        return -torch.sin(u_scaled) * (theta_end - theta_start) * scale_b1
    def db2_du(u):
        u_scaled = theta_start + u * (theta_end - theta_start)
        return torch.cos(u_scaled) * (theta_end - theta_start) * scale_b2
    surf1 = parametric_cyl_surface(b1_of_u=b1_of_u,
                                   b2_of_u=b2_of_u,
                                   db1_du=db1_du,
                                   db2_du=db2_du,
                                   b1_axis=b1_axis,
                                   b2_axis=b2_axis,
                                   v_axis=v_axis,
                                   center=center_circ,
                                   width_v=cyl_length,
                                   device=torch_dev,)
    # surf1 = planar_curved_surface(u_axis=u_axis,
    #                               v_axis=v_axis,
    #                               center=center,
    #                               width_u=cyl_width,
    #                               width_v=cyl_length,
    #                               height_curve=cyl_height/2,
    #                               poly_degree=4,
    #                               device=torch_dev,)
    grad1 = stream_func_surface_coil(surf1,
                                     num_v_modes=num_v_modes,
                                     num_u_modes=num_u_modes,
                                     u_bases=bases,
                                     v_bases=bases,
                                     device=torch_dev,
                                     dtype=torch.float32)
    center2 = center.clone() * 0
    center2[1] -= cyl_height/2
    center2[2] = coil_z_ofs
    surf2 = planar_surface(u_axis=u_axis,
                           v_axis=v_axis,
                           center=center2,
                           width_u=cyl_width,
                           width_v=cyl_length,
                           device=torch_dev,)
    # center2 = torch.zeros_like(center)
    # center2[1] = -cyl_height/2
    # surf2 = planar_curved_surface(u_axis=u_axis,
    #                               v_axis=v_axis,
    #                               center=center2,
    #                               width_u=cyl_width*1.1,
    #                               width_v=cyl_length,
    #                               height_curve=-cyl_height/2,
    #                               poly_degree=4,
    #                               device=torch_dev,)
    grad2 = stream_func_surface_coil(surf2,
                                     num_v_modes=num_v_modes,
                                     num_u_modes=num_u_modes,
                                     u_bases=bases,
                                     v_bases=bases,
                                     device=torch_dev,
                                     dtype=torch.float32)
    grad_coil = multi_stream([grad1, grad2])
elif gradient_type == 'matrix':
    # Setup half cylinder surfaces
    u_axis = torch.tensor([1, 0, 0], device=torch_dev, dtype=torch.float32)
    v_axis = torch.tensor([0, 0, 1], device=torch_dev, dtype=torch.float32)
    center = torch.tensor([0, cyl_height/2, coil_z_ofs], device=torch_dev, dtype=torch.float32)
    wedge = 120 # degrees
    theta_start = 3*torch.pi/2 - wedge/2 * torch.pi / 180
    theta_end = -torch.pi/2 + wedge/2 * torch.pi / 180
    b1_axis = u_axis.clone()
    b2_axis = b1_axis * 0
    b2_axis[1] = 1
    scale_b1 = cyl_width / 2
    h0 = 1 + np.cos(wedge * torch.pi / 360)
    scale_b2 = cyl_height / h0
    center_circ = center.clone() * 0
    center_circ[1] = -scale_b2 * np.cos(wedge * torch.pi / 360)/2
    center_circ[2] = coil_z_ofs
    def b1_of_u(u):
        u_scaled = theta_start + u * (theta_end - theta_start)
        return torch.cos(u_scaled) * scale_b1
    def b2_of_u(u):
        u_scaled = theta_start + u * (theta_end - theta_start)
        return torch.sin(u_scaled) * scale_b2
    def db1_du(u):
        u_scaled = theta_start + u * (theta_end - theta_start)
        return -torch.sin(u_scaled) * (theta_end - theta_start) * scale_b1
    def db2_du(u):
        u_scaled = theta_start + u * (theta_end - theta_start)
        return torch.cos(u_scaled) * (theta_end - theta_start) * scale_b2
    surf1 = parametric_cyl_surface(b1_of_u=b1_of_u,
                                   b2_of_u=b2_of_u,
                                   db1_du=db1_du,
                                   db2_du=db2_du,
                                   b1_axis=b1_axis,
                                   b2_axis=b2_axis,
                                   v_axis=v_axis,
                                   center=center_circ,
                                   width_v=cyl_length,
                                   device=torch_dev,)
    center2 = center.clone() * 0
    center2[1] -= cyl_height/2
    center2[2] = coil_z_ofs
    surf2 = planar_surface(u_axis=u_axis,
                           v_axis=v_axis,
                           center=center2,
                           width_u=cyl_width,
                           width_v=cyl_length,
                           device=torch_dev,)
    
    # Matrix gradient coil on surface
    loop_radius = .075 * 0 + 0.05
    u1s = torch.linspace(0, 1, 8, device=torch_dev) # cyl
    v1s = torch.linspace(0, 1, 4, device=torch_dev) # cyl
    u1s, v1s = torch.meshgrid(u1s, v1s, indexing='ij')
    u2s = torch.linspace(0, 1, 6, device=torch_dev) # backplate
    v2s = torch.linspace(0, 1, 4, device=torch_dev) # backplate
    u2s, v2s = torch.meshgrid(u2s, v2s, indexing='ij')
    centers1 = surf1.to_xyz(u1s, v1s)
    du1, dv1 = surf1.dxyz_duv(u1s, v1s)
    normals1 = torch.cross(du1, dv1, dim=-1)
    normals1 = normals1 / normals1.norm(dim=-1, keepdim=True)
    centers2 = surf2.to_xyz(u2s, v2s)
    du2, dv2 = surf2.dxyz_duv(u2s, v2s)
    normals2 = torch.cross(du2, dv2, dim=-1)
    normals2 = normals2 / normals2.norm(dim=-1, keepdim=True)
    centers = torch.cat([centers1.reshape(-1, 3), 
                         centers2.reshape(-1, 3)], dim=0)
    normals = torch.cat([normals1.reshape(-1, 3), 
                         normals2.reshape(-1, 3)], dim=0)  
    
    # # cylinder
    # centers = matrix_coil.gen_pts_cylinder(10, 6)
    # centers = centers.to(torch_dev)
    # centers[..., 0] *= cyl_width / 2
    # centers[..., 1] *= cyl_height / 2
    # centers[..., 2] *= cyl_length
    # centers[..., 2] += coil_z_ofs
    # normals = centers.clone()
    # normals[:, 2] = 0
    # normals = normals / normals.norm(dim=-1, keepdim=True)
    
    # Matrix gradient coil from PNS surface
    delta_out = 3 # cm from surface
    usurf = torch.arange(0, 2 * torch.pi, 2 * torch.pi / 8, device=torch_dev)
    vsurf = torch.linspace(0.4, 0.7, 4, device=torch_dev)
    _, xyz_crds, normals, _ = bcm._gen_surface_pts(usurf, vsurf)
    normals /= normals.norm(dim=-1, keepdim=True)
    centers = xyz_crds + normals * delta_out * 1e-2
      
    radii = centers[:, 0] * 0 + loop_radius
    thetas_phis = torch.stack([torch.atan2(normals[..., 1], normals[..., 0]),
                            torch.arccos(normals[..., 2]), ], dim=-1)
    grad_coil = matrix_coil(radii, centers, thetas_phis)
Ncoeffs = grad_coil.get_num_coeffs()

# fig, ax = surf1.visualize_surface()
# fig, ax = surf2.visualize_surface(fig=fig, ax=ax)
# # fig, ax = surf.visualize_surface()
# ax.scatter(crds_dsv[:, 0].cpu() * 1e2, crds_dsv[:, 1].cpu() * 1e2, crds_dsv[:, 2].cpu() * 1e2, color='green', marker='.', alpha=0.05)
# ax.scatter(crds_pns[:, 0].cpu() * 1e2, crds_pns[:, 1].cpu() * 1e2, crds_pns[:, 2].cpu() * 1e2, color='orange', marker='.', alpha=0.1)
# ax.view_init(elev=90, azim=-90, roll=0)
# plt.show()
# quit()

# Define matrices
def G_theta(thetas):
    _, G, _ = grad_coil.build_field_matrices(crds_bfield=crds_pns[:1], crds_gfield=crds_dsv, crds_efield=crds_pns[:1])
    return G[..., grad_dir] * GRAD_UNITS / dstream # T/m -> mT/m/A
def L_theta(thetas):
    if 'stream' in gradient_type:
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
    if 'stream' in gradient_type:
        P = grad_coil.build_winding_tolerance_matrix()
        return P.moveaxis(-1, 1) * min_wire_spacing / dstream
    return None
def F_theta(thetas):
    if 'stream' in gradient_type:
        F = grad_coil.build_current_boundary_matrix()
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

# # TODO: REMOVE THIS
# print('SKETCH.')
# Gmax = 40
# scale = Gmax ** 2 * loop_radius * 2 * torch.pi * rho
# E = (torch.eye(Ncoeffs, device=torch_dev) * scale * 1e-3) ** 0.5
# # print('Total Power = Emax^2 / Nturns ... ')
# # E = E[None, ...] # Sum of power per channel
# print('Power/Chan_max = Emax^2 / Nturns ... ')
# E = E[:, None, :] # Power per channel

def surf_opt(state_dict=None):
    # Solve via unrolled ADMM
    if 'stream' in gradient_type:
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
    if 'stream' in gradient_type:
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
                       admm_iters=admm_iters,
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
Gmin_10 = torch.quantile((G @ dct["x"]).flatten(), 0.1).item()
Gmin_20 = torch.quantile((G @ dct["x"]).flatten(), 0.2).item()
Lmax_actual = (L @ dct["x"]).norm().square().item()
Emax_actual = (E @ dct["x"]).norm(dim=-1).max().item()
print(f'\n' + '-'*100)
dir_to_str = ['x', 'y', 'z']
print(f'Gradient Type: {gradient_type} G{dir_to_str[grad_dir]}')
print(f'Number of coefficients: {Ncoeffs}')
print(f'Gmin: {Gmin_actual}')
print(f'G10%: {Gmin_10}')
print(f'G20%: {Gmin_20}')
print(f'Lmax: {Lmax_actual}')
print(f'Emax: {Emax_actual}')
print(F'E/G:  {Emax_actual / Gmin_actual:.2f}')
if 'matrix' in gradient_type:
    loop_radius = torch.tensor(loop_radius)
    inductance_single = calc_inductance_loop(loop_radius).item() * 1e3 # mH
    Vmax = 40 * max_ni_matrix * inductance_single / 8 / Gmin_10
    Imax = 40 * max_ni_matrix / Gmin_10
    print(f'Vmax: {Vmax:.3f}kV/Turn @ 20kHz, 40mT/m')
    print(f'Imax: {Imax:.2f}A-Turns @ 40mT/m')
else:
    Vmax = 40 * Lmax_actual / Gmin_10 / 8
    Imax = 40 / Gmin_10
    print(f'Vmax: {Vmax:.2f}kV @ 20kHz, 40mT/m')
    print(f'Imax: {Imax:.2f}A @ 40mT/m')
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

# Draw PNS and DSV surfaces
def draw_pns_dsv(fig, ax, alpha=0.2):
    # # PNS surface
    # Rx = rxs.abs().max().item()
    # Ry = rys.abs().max().item()
    # thetas = torch.linspace(0, 2 * torch.pi, 100, device=torch_dev)
    # zmin = crds_pns[:, 2].min().item()
    # zmax = crds_pns[:, 2].max().item()
    # zs_lin = torch.linspace(zmin, zmax, 100, device=torch_dev)
    # THETA, ZS = torch.meshgrid(thetas, zs_lin, indexing='ij')
    # X = Rx * torch.cos(THETA)
    # Y = Ry * torch.sin(THETA)
    # Z = ZS
    # ax.plot_surface(X.cpu() * 1e2, Y.cpu() * 1e2, Z.cpu() * 1e2, color='orange', alpha=alpha)
    bcm.show_surface(alpha=0.2, ax=ax, fig=fig)

    # Show DSV surface
    thetas = torch.linspace(0, 2 * torch.pi, 100, device=torch_dev)
    zs = torch.linspace(-dsvz/2, dsvz/2, 100, device=torch_dev)
    thetas, Z = torch.meshgrid(thetas, zs, indexing='ij')
    X = dsvx * torch.cos(thetas) / 2
    Y = dsvy * torch.sin(thetas) / 2
    ax.plot_surface(X.cpu() * 1e2, Y.cpu() * 1e2, Z.cpu() * 1e2, color='green', alpha=alpha)

    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_zlabel('Z [cm]')
    ax.set_aspect('equal')
    ax.axis('off')

# Show winding 
if 'stream' in gradient_type:
    fig, ax = grad_coil.show_countour(coeffs=dct['x'], dstream=dstream)
elif gradient_type == 'matrix':
    fig, ax, axl = grad_coil.show_design(coeffs=dct['x'])
draw_pns_dsv(fig, ax, alpha=0.3)
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=-30, azim=48, roll=65)
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_zlim(-25, 25)

if show_pyvista:
    import pyvista as pv
    import numpy as np
    p = pv.Plotter()
    # PNS Surface
    eps = 1e-6
    u = torch.linspace(0, 2 * torch.pi, 50, device=torch_dev)
    v = torch.linspace(0.0, 1, 200, device=torch_dev)
    v = torch.clamp(v, eps, 1-eps)
    bcm = body_charge_model(u, v, M_fourier_modes=7*2, N_hat_modes=100,
                            ofs=torch.tensor([0, 0, 0.25], device=torch_dev))
    X = bcm.xyz_crds[:, 0].cpu().numpy().reshape(bcm.num_us, bcm.num_vs) * 1e2
    Y = bcm.xyz_crds[:, 1].cpu().numpy().reshape(bcm.num_us, bcm.num_vs) * 1e2
    Z = bcm.xyz_crds[:, 2].cpu().numpy().reshape(bcm.num_us, bcm.num_vs) * 1e2
    surf = pv.StructuredGrid(X, Y, Z)
    p.add_mesh(
        surf,
        color="navajowhite",
        opacity=0.9,
        smooth_shading=True,
    )

    # DSV surface
    thetas = np.linspace(0, 2 * torch.pi, 100)
    zs = np.linspace(-dsvz/2, dsvz/2, 100) * 1e2
    thetas, Z = np.meshgrid(thetas, zs, indexing='ij')
    X = dsvx * np.cos(thetas) * 1e2 / 2
    Y = dsvy * np.sin(thetas) * 1e2 / 2
    surf = pv.StructuredGrid(X, Y, Z)
    p.add_mesh(
        surf,
        color="green",
        opacity=0.5,
        smooth_shading=True,
    )

    # Winding pattern
    if 'stream' in gradient_type:
        countours, dstr = grad_coil.stream_to_contour(coeffs=dct['x'], dstream=dstream)
        for pts in countours:
            line = pv.lines_from_points(pts.cpu().numpy() * 1e2)
            tube = line.tube(radius=0.1)  # adjust radius to your geometry scale
            p.add_mesh(
                tube,
                color="red",
                smooth_shading=True,
            )
    else:
        thetas = torch.linspace(0, 2 * torch.pi, 100, device=torch_dev)
        xs = torch.cos(thetas)
        ys = torch.sin(thetas)
        zs = torch.zeros_like(thetas)
        crds_loop = torch.stack([xs, ys, zs], dim=-1)
        normals = grad_coil._get_normals()
        for i in range(len(grad_coil.radii)):
            r = grad_coil.radii[i]
            c = grad_coil.centers[i]
            n = normals[i]
            crds_loop_new = _transform_coordinates(r * crds_loop, c[None, :], n[None, :], flip_order=True)[0]
            crds_loop_new = crds_loop_new.cpu().numpy() * 1e2
            line = pv.lines_from_points(crds_loop_new)
            tube = line.tube(radius=0.1)  # adjust radius to your geometry scale
            p.add_mesh(
                tube,
                color="red",
                smooth_shading=True,
            )

    # Config
    p.set_background("white")
    p.show_axes()      # remove this line if you want no axes
    p.enable_lightkit()
    M = 1.5
    p.camera_position = [
        (128 / M, 250 / M, 40 / M),   # camera location
        (0, 0, 0),   # focal point
        (0, 0, 1)     # up direction
    ]
    p.export_html("scene.html")
    quit()

# Show stream function
if 'stream' in gradient_type:
    fig, ax = grad_coil.show_design(coeffs=dct['x'],
                                    show_1d=False,
                                    colorbar=False)
    ax.axis('off')
    draw_pns_dsv(fig, ax, alpha=1.0)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=-30, azim=48, roll=65)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_zlim(-25, 25)

# Compute field using stream function 
if grad_dir == 0:
    crds_plt = gen_grd((101, 1, 101), (xrange, yrange, zrange)).to(torch_dev)[:, 0, :]
elif grad_dir == 1:
    crds_plt = gen_grd((1, 101, 101), (xrange, yrange, zrange)).to(torch_dev)[0, :, :]
elif grad_dir == 2:
    crds_plt = gen_grd((101, 1, 101), (xrange, yrange, zrange)).to(torch_dev)[:, 0, :]
crds_plt -= crds_plt.reshape((-1,3)).mean(dim=0)
bfield, gfield, efield = grad_coil.evaluate_fields(coeffs=dct['x'], crds_bfield=crds_plt, crds_gfield=crds_plt, crds_efield=crds_plt)

# Compute fields
fields = [[bfield[..., -1].rot90() * BFIELD_UNITS / dstream, 
              gfield[..., grad_dir].rot90() * GRAD_UNITS / dstream, 
              efield.norm(dim=-1).rot90() * EFIELD_UNITS / FREQ_UNITS / dstream]]
if 'stream' in gradient_type:
    from magopt.sim.wire_fields import parametric_wire
    # xyz_countours, dstream = grad_coil.stream_to_winding(coeffs=dct['x'], dstream=dstream)
    xyz_countours, dstream = grad_coil.stream_to_contour(coeffs=dct['x'], dstream=dstream)
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
gmin_plot = gmin_targ
delta = gmin_plot * 0.5
vmins = [fields[0][0].mean().item() - gmin_plot * dsvy, gmin_plot - delta, 0]
vmaxs = [fields[0][0].mean().item() + gmin_plot * dsvy, gmin_plot + delta, 10]
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
		if grad_dir == 1:
			plt.xlabel('Y [cm]')
			# plt.scatter(crds_dsv[:, 1].cpu() * 1e2, crds_dsv[:, 2].cpu() * 1e2, 
			# 			color='green', marker='.', alpha=0.01)
		else:
			plt.xlabel('X [cm]')
			# plt.scatter(crds_dsv[:, 0].cpu() * 1e2, crds_dsv[:, 2].cpu() * 1e2, 
			# 			color='green', marker='.', alpha=0.01)
		plt.ylabel('Z [cm]')
plt.tight_layout()
plt.show()

