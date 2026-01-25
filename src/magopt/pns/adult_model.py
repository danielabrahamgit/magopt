"""
Python implementation of the body model in:
"Electric field calculation and peripheral nerve stimulationprediction for head and body gradient coils"

The surface is simplified as X and Y ellipse radii as a funnction of Z-position. 

The body is chunked into 5 anatomical regions which are (superior to inferior):
Body end, Body, Body/neck transition, Head/Neck transition, and Top of Head
"""

import torch

anatomy_params = {
    "Rx0": {  # x ellipse radii shoulder
        "Male": {"2.5": 203.0, "50": 225.0, "97.5": 246.5, 'safe': 246.5},
        "Female": {"2.5": 183.0, "50": 203.0, "97.5": 225.0},
    },
    "Rx1": {  # x ellipse radii neck
        "Male": {"2.5": 53.5, "50": 58.5, "97.5": 63.5, 'safe': 90.0},
        "Female": {"2.5": 51.7, "50": 55.9, "97.5": 60.2},
    },
    "Rx2": {  # x ellipse radii head
        "Male": {"2.5": 72.5, "50": 77.5, "97.5": 82.5, 'safe': 110.0},
        "Female": {"2.5": 72.5, "50": 72.5, "97.5": 77.5},
    },
    "Ry0": {  # y ellipse radii torso
        "Male": {"2.5": 98.0, "50": 114.5, "97.5": 136.0, 'safe': 136.0},
        "Female": {"2.5": 97.0, "50": 107.0, "97.5": 117.0},
    },
    "Ry1": {  # y ellipse radii neck
        "Male": {"2.5": 57.2, "50": 61.2, "97.5": 66.7, 'safe': 90.0},
        "Female": {"2.5": 51.7, "50": 55.9, "97.5": 60.2},
    },
    "Ry2": {  # y ellipse radii head
        "Male": {"2.5": 92.5, "50": 98.0, "97.5": 104.0, 'safe': 110.0},
        "Female": {"2.5": 91.5, "50": 92.5, "97.5": 98.0},
    },
    "L0": {  # z lengths body endcap
        "Male": {"2.5": 50.0, "50": 50.0, "97.5": 50.0, 'safe': 50.0},
        "Female": {"2.5": 50.0, "50": 50.0, "97.5": 50.0},
    },
    "L1": {  # z lengths body straight section
        "Male": {"2.5": 252.3, "50": 235.0, "97.5": 219.0, 'safe': 219.0},
        "Female": {"2.5": 267.3, "50": 249.0, "97.5": 231.8},
    },
    "L2": {  # z lengths body/neck transition
        "Male": {"2.5": 150.5, "50": 164.0, "97.5": 173.0, 'safe': 173.0},
        "Female": {"2.5": 142.5, "50": 157.0, "97.5": 170.5},
    },
    "L3": {  # z lengths neck/head transition
        "Male": {"2.5": 97.3, "50": 101.0, "97.5": 108.0, 'safe': 108.0},
        "Female": {"2.5": 90.3, "50": 94.0, "97.5": 97.8},
    },
    "L4": {  # z lengths top of head
        "Male": {"2.5": 97.3, "50": 101.0, "97.5": 108.0, 'safe': 108.0},
        "Female": {"2.5": 90.3, "50": 94.0, "97.5": 97.8},
    },
    "ZBC": {  # z brain center (relative to top of head)
        "Male": {"2.5": -97.3, "50": -101.0, "97.5": -108.0, 'safe': -108.0},
        "Female": {"2.5": -90.3, "50": -94.0, "97.5": -97.8},
    },
    "ZEC": {  # z eye center (relative to top of head)
        "Male": {"2.5": -102.0, "50": -112.0, "97.5": -119.0, 'safe': -119.0},
        "Female": {"2.5": -102.0, "50": -112.0, "97.5": -119.0},
    },
}

# Grab parameters for specific gender and percentile
gender = "Male"
# percentile = "97.5"
percentile = "safe"
Rx0 = anatomy_params['Rx0'][gender][percentile]
Rx1 = anatomy_params['Rx1'][gender][percentile]
Rx2 = anatomy_params['Rx2'][gender][percentile]
Ry0 = anatomy_params['Ry0'][gender][percentile]
Ry1 = anatomy_params['Ry1'][gender][percentile]
Ry2 = anatomy_params['Ry2'][gender][percentile]
L0 = anatomy_params['L0'][gender][percentile]
L1 = anatomy_params['L1'][gender][percentile]
L2 = anatomy_params['L2'][gender][percentile]
L3 = anatomy_params['L3'][gender][percentile]
L4 = anatomy_params['L4'][gender][percentile]
ZBC = anatomy_params['ZBC'][gender][percentile]
ZEC = anatomy_params['ZEC'][gender][percentile]
 
def dxyz_duv(u: torch.Tensor, 
             v: torch.Tensor,) -> torch.Tensor:
    """
    Computes the surface tangents at u, v coordinates.
    
    Args
    ----
    u : torch.Tensor
        shape (...) representing the u coordinates.
    v : torch.Tensor
        shape (...) representing the v coordinates. 
        
    Returns
    -------
    torch.Tensor:
        shape (..., 2, 3) representing the surface tangents at the u, v coordinates, units in mm.
    """
    # ------- Build dcrd/dv -------
    # z = alpha v + beta --> dz/dv = alpha
    alpha = (L0 + L1 + L2 + L3 + L4)
    beta = -(L0 + L1 + L2 + L3)
    z = alpha * v + beta
    dz_dv = alpha * torch.ones_like(v)
    
    # x,y = rx,y(z) * cos,sin(u) --> dx,y/dv = drx,y/dz * dz/dv * cos,sin(u)
    drx_dz, dry_dz = get_dradii_dz(z)
    dx_dv = drx_dz * dz_dv * torch.cos(u)
    dy_dv = dry_dz * dz_dv * torch.sin(u)
    
    # Stack
    dxyz_dv = torch.stack([dx_dv, dy_dv, dz_dv], dim=-1)
    
    # ------- Build dcrd/du -------
    # x,y = rx,y(z) * cos,sin(u) --> dx,y/du = rx,y(z) * -sin,+cos(u)
    rx, ry = get_radii(z)
    dx_du = -rx * torch.sin(u)
    dy_du = ry * torch.cos(u)
    
    # Stack
    dxyz_du = torch.stack([dx_du, dy_du, torch.zeros_like(u)], dim=-1)
    
    return torch.stack([dxyz_du, dxyz_dv], dim=-2)
    
def uv_to_normals(u: torch.Tensor, 
                  v: torch.Tensor,) -> torch.Tensor:
    """
    Computes the surface normals at u, v coordinates
    
    Args
    ----
    u : torch.Tensor
        shape (...) representing the u coordinates.
    v : torch.Tensor
        shape (...) representing the v coordinates. 
        
    Returns
    -------
    torch.Tensor:
        shape (..., 3) representing the surface normals at the u, v coordinates.
    """
    xyz = uv_to_xyz(u, v)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    # normals = get_normals(x, y, z)
    jacob = dxyz_duv(u, v)
    normals = torch.cross(jacob[..., 0, :], jacob[..., 1, :], dim=-1)
    return normals

def uv_to_xyz(u: torch.Tensor, 
              v: torch.Tensor,) -> torch.Tensor:
    """
    Converts u, v coordinates to x, y, z
    
    U ranges from [0, 2pi] representing the angle in the X-Y plane, 
    V ranges from [0, 1] representing the Z-position.
    
    Args:
    -----
    u : torch.Tensor 
        shape (...) representing the u coordinates.
    v : torch.Tensor
        shape (...) representing the v coordinates.

    Returns:
    --------
    torch.Tensor:
        shape (..., 3) representing the x, y, z coordinates in mm
    """
    # Convert v to z
    z = v * (L0 + L1 + L2 + L3 + L4) # [0, L]
    z -= L0 + L1 + L2 + L3 # Centered about the brain

    # Get radii at z
    radii = get_radii(z)
    rx, ry = radii[0], radii[1]

    # Convert to Cartesian coordinates
    x = rx * torch.cos(u)
    y = ry * torch.sin(u)

    return torch.stack([x, y, z], dim=-1)

def get_normals(x: torch.Tensor, 
                y: torch.Tensor, 
                z: torch.Tensor,) -> torch.Tensor:
    # Get radii at z
    z = z + L0 + L1 + L2 + L3
    z = torch.clamp(z, 0, L0 + L1 + L2 + L3 + L4)
    mask_L0 = (z < L0)[..., None]
    mask_L1 = ((z >= L0) & (z < L0 + L1))[..., None]
    mask_L2 = ((z >= L0 + L1) & (z < L0 + L1 + L2))[..., None]
    mask_L3 = ((z >= L0 + L1 + L2) & (z < L0 + L1 + L2 + L3))[..., None]
    mask_L4 = (z >= L0 + L1 + L2 + L3)[..., None]
    normals = torch.zeros((*z.shape, 3), dtype=z.dtype, device=z.device)
    normals += normal_body_end(x, y, z) * mask_L0
    normals += normal_body(x, y, z) * mask_L1
    normals += normal_body_neck(x, y, z) * mask_L2
    normals += normal_neck_head(x, y, z) * mask_L3
    normals += normal_top_head(x, y, z) * mask_L4
    return normals

def get_dradii_dz(z: torch.Tensor) -> torch.Tensor:
    # Shift to bottom of body
    z = z + L0 + L1 + L2 + L3
    z = torch.clamp(z, 0, L0 + L1 + L2 + L3 + L4)
    mask_L0 = z < L0
    mask_L1 = (z >= L0) & (z < L0 + L1)
    mask_L2 = (z >= L0 + L1) & (z < L0 + L1 + L2)
    mask_L3 = (z >= L0 + L1 + L2) & (z < L0 + L1 + L2 + L3)
    mask_L4 = z >= L0 + L1 + L2 + L3
    dradii_dz = torch.zeros((2, *z.shape), dtype=z.dtype, device=z.device)
    dradii_dz += radii_body_end(z, calc_deriv=True) * mask_L0
    dradii_dz += radii_body(z, calc_deriv=True) * mask_L1
    dradii_dz += radii_body_neck(z, calc_deriv=True) * mask_L2
    dradii_dz += radii_neck_head(z, calc_deriv=True) * mask_L3
    dradii_dz += radii_top_head(z, calc_deriv=True) * mask_L4
    return dradii_dz

def get_radii(z: torch.Tensor) -> torch.Tensor:
    # Shift to bottom of body
    z = z + L0 + L1 + L2 + L3
    z = torch.clamp(z, 0, L0 + L1 + L2 + L3 + L4)
    mask_L0 = z < L0
    mask_L1 = (z >= L0) & (z < L0 + L1)
    mask_L2 = (z >= L0 + L1) & (z < L0 + L1 + L2)
    mask_L3 = (z >= L0 + L1 + L2) & (z < L0 + L1 + L2 + L3)
    mask_L4 = z >= L0 + L1 + L2 + L3
    radii = torch.zeros((2, *z.shape), dtype=z.dtype, device=z.device)
    radii += radii_body_end(z) * mask_L0
    radii += radii_body(z) * mask_L1
    radii += radii_body_neck(z) * mask_L2
    radii += radii_neck_head(z) * mask_L3
    radii += radii_top_head(z) * mask_L4
    return radii

def normal_body_end(x, y, z):
    z0 = L0 
    return torch.stack([
        2 * x / Rx0**2,
        2 * y / Ry0**2,
        2 * (z - z0) / L0**2,
    ], dim=-1)

def radii_body_end(z, calc_deriv=False):
    z0 = L0 
    zterm = (1 - ((z - z0) / L0) ** 2) ** 0.5
    zterm = zterm.nan_to_num()
    
    if calc_deriv:
        drx_dz = -Rx0 * (z - z0) / (L0**2 * zterm).clamp(min=1e-9)
        dry_dz = -Ry0 * (z - z0) / (L0**2 * zterm).clamp(min=1e-9)
        return torch.stack([drx_dz, dry_dz], dim=0)
    else:
        rx = Rx0 * zterm
        ry = Ry0 * zterm
        return torch.stack([rx, ry], dim=0)

def normal_body(x, y, z):
    return torch.stack([
        2 * x / Rx0**2,
        2 * y / Ry0**2,
        0 * z,
    ], dim=-1)

def radii_body(z, calc_deriv=False):
    if calc_deriv:
        return torch.stack([z * 0, z * 0], dim=0)
    else:
        rx = Rx0 * torch.ones_like(z)
        ry = Ry0 * torch.ones_like(z)
        return torch.stack([rx, ry], dim=0)

def normal_body_neck(x, y, z):
    dRx_dz, dRy_dz = radii_body_neck(z, calc_deriv=True)
    Rx, Ry = radii_body_neck(z)
    return torch.stack([
        2 * x / Rx**2,
        2 * y / Ry**2,
        -2 * dRx_dz * x**2 / Rx**3 - 2 * dRy_dz * y**2 / Ry**3,
    ], dim=-1)

def radii_body_neck(z, calc_deriv=False):
    z2 = L1 + L0
    if calc_deriv:
        dRx_dz = (Rx1 - Rx0)/2 * torch.sin(torch.pi * (z - z2) / L2) * torch.pi / L2
        dRy_dz = (Ry1 - Ry0)/2 * torch.sin(torch.pi * (z - z2) / L2) * torch.pi / L2
        return torch.stack([dRx_dz, dRy_dz], dim=0)
    else:    
        Rx = (Rx1 + Rx0)/2 - (Rx1 - Rx0)/2 * torch.cos(torch.pi * (z - z2) / L2)
        Ry = (Ry1 + Ry0)/2 - (Ry1 - Ry0)/2 * torch.cos(torch.pi * (z - z2) / L2)
        return torch.stack([Rx, Ry], dim=0)

def normal_neck_head(x, y, z):
    dRx_dz, dRy_dz = radii_neck_head(z, calc_deriv=True)
    Rx, Ry = radii_neck_head(z)
    return torch.stack([
        2 * x / Rx**2,
        2 * y / Ry**2,
        -2 * dRx_dz * x**2 / Rx**3 - 2 * dRy_dz * y**2 / Ry**3,
    ], dim=-1)

def radii_neck_head(z, calc_deriv=False):
    z3 = L0 + L1 + L2
    if calc_deriv:
        dRx_dz = (Rx2 - Rx1)/2 * torch.sin(torch.pi * (z - z3) / L3) * torch.pi / L3
        dRy_dz = (Ry2 - Ry1)/2 * torch.sin(torch.pi * (z - z3) / L3) * torch.pi / L3
        return torch.stack([dRx_dz, dRy_dz], dim=0)
    else:
        Rx = (Rx2 + Rx1)/2 - (Rx2 - Rx1)/2 * torch.cos(torch.pi * (z - z3) / L3)
        Ry = (Ry2 + Ry1)/2 - (Ry2 - Ry1)/2 * torch.cos(torch.pi * (z - z3) / L3)
        return torch.stack([Rx, Ry], dim=0)

def normal_top_head(x, y, z):
    z4 = L0 + L1 + L2 + L3
    return torch.stack([
        2 * x / Rx2**2,
        2 * y / Ry2**2,
        2 * (z - z4) / L4**2,
    ], dim=-1)

def radii_top_head(z, calc_deriv=False):
    z4 = L0 + L1 + L2 + L3
    zterm = (1 - ((z - z4) / L4) ** 2) ** 0.5
    zterm = zterm.nan_to_num()

    if calc_deriv:
        drx_dz = -Rx2 * (z - z4) / (L4**2 * zterm).clamp(min=1e-9)
        dry_dz = -Ry2 * (z - z4) / (L4**2 * zterm).clamp(min=1e-9)
        return torch.stack([drx_dz, dry_dz], dim=0)
    else:
        return torch.stack([Rx2 * zterm, Ry2 * zterm], dim=0)

# import matplotlib
# matplotlib.use('webagg')
# import matplotlib.pyplot as plt

# from mr_recon.utils import gen_grd

# # z = torch.linspace(-(L0+L1+L2+L3), L4, 500)
# # radii = get_radii(z)
# # plt.figure(figsize=(14, 7))
# # plt.plot(z, radii[0], label='Rx')
# # plt.plot(z, radii[1], label='Ry')
# # kwargs = {'color': 'black', 'linestyle': '--'}
# # plt.axvline(-(L0+L1+L2+L3), **kwargs)
# # plt.axvline(-(L3+L2+L1), **kwargs)
# # plt.axvline(-(L3+L2), **kwargs)
# # plt.axvline(-(L3), **kwargs)
# # plt.axvline(0, **kwargs)
# # plt.axvline(L4, **kwargs)
# # plt.xlabel('Z')
# # plt.ylabel('Radius')
# # plt.legend()
# # plt.show()

# u = torch.linspace(0, 2 * torch.pi, 100)
# # u = torch.tensor([0, torch.pi/2])
# v = torch.linspace(0, 1, 100) #** 0.5
# us, vs = torch.meshgrid(u, v, indexing='ij')
# xyzs = uv_to_xyz(us, vs)
# normals = uv_to_normals(us, vs)
# normals /= torch.norm(normals, dim=-1, keepdim=True)
# xs, ys, zs = xyzs[..., 0], xyzs[..., 1], xyzs[..., 2]
# nx, ny, nz = normals[..., 0], normals[..., 1], normals[..., 2]

# # # 1D plot
# # plt.figure(figsize=(14,7))
# # p = plt.plot(zs[0], xs[0])
# # plt.quiver(zs[0], xs[0], nz[0], nx[0], color=p[0].get_color())
# # p = plt.plot(zs[1], ys[1])
# # plt.quiver(zs[1], ys[1], nz[1], ny[1], color=p[0].get_color())
# # plt.axis('equal')
# # plt.show()
# # quit()

# fig = plt.figure(figsize=(14,7))
# ax = fig.add_subplot(111, projection='3d')
# # ax.plot_surface(xs, ys, zs)
# # plt.xlim(-250, 250)
# # plt.ylim(-250, 250)
# # plt.zlim(-400, 100)

# # Plot normals
# skip = 3
# slc = (slice(None, None, skip), slice(None, None, skip))
# ax.quiver(xs[slc], ys[slc], zs[slc],
#           nx[slc], ny[slc], nz[slc],
#           length=40, color='k', normalize=True)
# plt.axis('equal')
# plt.show()

