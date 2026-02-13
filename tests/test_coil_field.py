import torch

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from mr_recon.utils import gen_grd
from magopt.gradient_coils import matrix_coil
from magopt.sim.analytic import calc_bfield_loop, _transform_coordinates
from magopt.sim import parametric_wire
from magopt.sim.elip import EllipELookup, EllipKLookup


# Crds to sample field at 
torch.manual_seed(42)
torch_dev = torch.device(5)
im_size = (51, 51, 3)
fov = (0.22,)*3
crds = gen_grd(im_size, fov).to(torch_dev) + 1.23e-4
crds_flt = crds.reshape(-1, 3)

# Coil geometry
radii = torch.ones((1,), device=torch_dev) * 0.04
centers = torch.zeros((1,3), device=torch_dev)
centers = torch.randn_like(centers) * 0.1
normals = torch.tensor([0, 1, 1], device=torch_dev, dtype=torch.float32)[None,:]
normals /= normals.norm(dim=-1, keepdim=True)

# Use biot savart law
thetas = torch.linspace(0, 2 * torch.pi, 200, device=torch_dev)
xs = torch.cos(thetas)
ys = torch.sin(thetas)
zs = torch.zeros_like(thetas)
crds_loop = torch.stack([xs, ys, zs], dim=-1)[None,] * radii[:, None, None]
crds_loop_new = _transform_coordinates(crds_loop, 
                                       centers[:, None, :], 
                                       normals[:, None, :],
                                       flip_order=True)[0]
pw = parametric_wire(wire_pts=crds_loop_new[0])
bfield = pw.calc_bfield(crds_flt).reshape((*im_size, 3)).cpu()

# Use analytic formula
ellipe = EllipELookup().to(torch_dev)
ellipk = EllipKLookup().to(torch_dev)
bfield_analytic = calc_bfield_loop(crds_flt[None, :, :], 
                                   radii[:, None], 
                                   centers[:, None, :], 
                                   normals[:, None, :],
                                   ellipe=ellipe,
                                   ellipk=ellipk)[0].reshape((*im_size, 3)).cpu()


# Show slices 
for i in range(im_size[2]):
    plt.figure(figsize=(12, 6))
    zslc = crds[0, 0, i, 2].item() * 1e2
    axes = ['X', 'Y', 'Z']
    plt.suptitle(f'Z = {zslc:.2f} cm')
    for d in range(3):
        vmin = bfield_analytic[..., i, d].median() - 3 * bfield_analytic[..., i, d].std()
        vmax = bfield_analytic[..., i, d].median() + 3 * bfield_analytic[..., i, d].std()
        plt.subplot(2, 3, d+1)
        plt.imshow(bfield[..., i, d].cpu(), vmin=vmin, vmax=vmax, cmap='RdBu_r')
        plt.title(f'B-field {axes[d]}')
        plt.axis('off')
        plt.subplot(2, 3, d+4)
        plt.imshow(bfield_analytic[..., i, d].cpu(), vmin=vmin, vmax=vmax, cmap='RdBu_r')
        plt.title(f'B-field {d} (analytic)')
        plt.axis('off')
    plt.tight_layout()
        
# 3D scatter crds loop
fig = plt.figure()
crds_loop_new = crds_loop_new.cpu()[0] * 1e2
crds_loop = crds_loop.cpu()[0] * 1e2
ax = fig.add_subplot(111, projection='3d')
ax.scatter(crds_loop_new[:, 0], crds_loop_new[:, 1], crds_loop_new[:, 2], label='New', alpha=0.5)
ax.scatter(crds_loop[:, 0], crds_loop[:, 1], crds_loop[:, 2], label='Original', alpha=0.5)
ax.legend()
ax.set_xlabel('X [cm]')
ax.set_ylabel('Y [cm]')
ax.set_zlabel('Z [cm]')
plt.axis('equal')
plt.show()