import sys
import json
import torch
import trimesh
import numpy as np


import matplotlib
matplotlib.use('WebAgg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm
from shapely.geometry import Polygon

def vr_interp_R(zs, rs):
    """Piecewise-linear R(z) with flat extrapolation."""
    zs = np.asarray(zs, float).reshape(-1)
    rs = np.asarray(rs, float).reshape(-1)
    order = np.argsort(zs)
    zs = zs[order]; rs = rs[order]
    def R(zq):
        zq = np.asarray(zq, float)
        r = np.interp(zq, zs, rs)
        r[zq < zs[0]] = rs[0]
        r[zq > zs[-1]] = rs[-1]
        return r
    return R

def vr_resample_profile(zs, rs, M=200):
    """Uniform z-grid + interpolated radii for smooth shell rings."""
    zmin, zmax = float(np.min(zs)), float(np.max(zs))
    z_grid = np.linspace(zmin, zmax, int(M))
    R = vr_interp_R(zs, rs)
    return z_grid, R(z_grid)

def vr_build_shell(z_grid, r_out, wall_thickness_m=0.003, theta_sections=256, cap_ends=False, solid=False):
    """
    Variable-radius shell. If solid=True, builds a filled body (outer+caps only).
    Otherwise builds hollow shell with inner surface r_in = r_out - wall_thickness_m.
    """
    th = np.linspace(0.0, 2*np.pi, int(theta_sections), endpoint=False)

    def rings_for_radius(rad, zval):
        x = rad*np.cos(th); y = rad*np.sin(th); z = np.full_like(x, zval)
        return np.stack((x, y, z), axis=-1)

    def stitch(rings, flip=False):
        K = rings[0].shape[0]
        V = np.vstack(rings)
        F = []
        for i in range(len(rings)-1):
            b0 = i*K; b1 = (i+1)*K
            for j in range(K):
                a = b0 + j
                b = b0 + ((j+1) % K)
                c = b1 + j
                d = b1 + ((j+1) % K)
                if not flip:
                    F.append([a,b,c]); F.append([b,d,c])
                else:
                    F.append([a,c,b]); F.append([b,c,d])
        return V, np.asarray(F, np.int64)

    if solid:
        # Outer surface only; caps make it solid
        outer_rings = [rings_for_radius(r, z) for r, z in zip(r_out, z_grid)]
        V, F = stitch(outer_rings, flip=False)
        mesh = trimesh.Trimesh(vertices=V, faces=F, process=True)
        if cap_ends:
            K = outer_rings[0].shape[0]
            def cap_faces(ring_idx, outward=True):
                start = ring_idx * K
                ring = np.arange(start, start+K)
                centroid = mesh.vertices[ring].mean(axis=0)
                ci = len(mesh.vertices)
                V_new = np.vstack([mesh.vertices, centroid[None,:]])
                tris = []
                for j in range(K):
                    a = ring[j]; b = ring[(j+1)%K]
                    tris.append([a,b,ci] if outward else [b,a,ci])
                F_new = np.vstack([mesh.faces, np.asarray(tris, np.int64)])
                return V_new, F_new
            V, F = cap_faces(0, outward=False)
            mesh = trimesh.Trimesh(vertices=V, faces=F, process=True)
            V, F = cap_faces(len(z_grid)-1, outward=True)
            mesh = trimesh.Trimesh(vertices=V, faces=F, process=True)
        return mesh

    # Hollow shell: outer + inner surfaces
    r_in = r_out - wall_thickness_m
    if np.any(r_in <= 0):
        raise ValueError("Inner radius became non-positive somewhere; reduce wall thickness.")

    outer_rings = [rings_for_radius(r, z) for r, z in zip(r_out, z_grid)]
    inner_rings = [rings_for_radius(r, z) for r, z in zip(r_in,  z_grid)]

    V_out, F_out = stitch(outer_rings, flip=False)  # outward normals
    V_in,  F_in  = stitch(inner_rings,  flip=True)  # inward normals for inner cavity
    V = np.vstack([V_out, V_in])
    F = np.vstack([F_out, F_in + len(V_out)])

    mesh = trimesh.Trimesh(vertices=V, faces=F, process=True)

    if cap_ends:
        # Cap all four edges to get a watertight volume (good for booleans)
        K = outer_rings[0].shape[0]
        def cap(V_curr, F_curr, offset, ring_idx, outward):
            ring = np.arange(offset + ring_idx*K, offset + (ring_idx+1)*K)
            centroid = V_curr[ring].mean(axis=0)
            ci = len(V_curr)
            Vn = np.vstack([V_curr, centroid[None,:]])
            tris = []
            for j in range(K):
                a = ring[j]; b = ring[(j+1)%K]
                tris.append([a,b,ci] if outward else [b,a,ci])
            Fn = np.vstack([F_curr, np.asarray(tris, np.int64)])
            return Vn, Fn

        # outer caps
        V, F = cap(V, F, offset=0,            ring_idx=0,            outward=False)
        V, F = cap(V, F, offset=0,            ring_idx=len(z_grid)-1, outward=True)
        # inner caps (reverse orientation)
        V, F = cap(V, F, offset=len(V_out),   ring_idx=0,            outward=True)
        V, F = cap(V, F, offset=len(V_out),   ring_idx=len(z_grid)-1, outward=False)
        mesh = trimesh.Trimesh(vertices=V, faces=F, process=True)

    return mesh

def vr_map_wires(wire_crds, coord='cartesian', R=None, project=False):
    """
    Map input wire coordinates to (x,y,z) on r=R(z) surface.
    - coord='cartesian': wire_crds is (N,3) xyz; if project=True, snap radially to r=R(z)
    - coord='cyl_tz'   : (theta,z,*) -> r = R(z), x=r cos th, y=r sin th
    - coord='cyl_rtz'  : (r,theta,z) -> direct map (no change)
    """
    arr = np.asarray(wire_crds, float)
    if coord == 'cartesian':
        if not project or R is None:
            return arr
        # project onto r=R(z)
        x,y,z = arr[:,0], arr[:,1], arr[:,2]
        r = np.hypot(x,y); th = np.arctan2(y,x)
        r_tgt = R(z)
        eps = 1e-12
        scale = np.where(r>eps, r_tgt/r, 0.0)
        x = x*scale; y = y*scale
        x[r<=eps] = r_tgt[r<=eps]; y[r<=eps] = 0.0
        return np.stack([x,y,z], axis=-1)
    elif coord == 'cyl_tz':
        th = arr[:,0]; z = arr[:,1]
        if R is None:
            raise ValueError("R(z) is required for coord='cyl_tz'")
        r = R(z)
        x = r*np.cos(th); y = r*np.sin(th)
        return np.stack([x,y,z], axis=-1)
    elif coord == 'cyl_rtz':
        r, th, z = arr[:,0], arr[:,1], arr[:,2]
        x = r*np.cos(th); y = r*np.sin(th)
        return np.stack([x,y,z], axis=-1)
    else:
        raise ValueError(f"Unknown coord type: {coord}")

def remove_end_caps(mesh, zmin=None, zmax=None, tol=1e-6):
    """
    Remove faces that lie entirely on the end planes (opening the tube).
    - If zmin/zmax are None, they’re inferred from the mesh extents.
    - tol is in the same units as your mesh (meters if you worked in meters).
    """
    V = mesh.vertices
    F = mesh.faces
    z = V[:, 2]

    if zmin is None:
        zmin = float(z.min())
    if zmax is None:
        zmax = float(z.max())

    fz = z[F]  # (num_faces, 3) z-values per face
    is_min_cap = np.all(np.abs(fz - zmin) < tol, axis=1)
    is_max_cap = np.all(np.abs(fz - zmax) < tol, axis=1)
    keep = ~(is_min_cap | is_max_cap)

    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    return mesh

# Conversion factor from meters to millimeters -- stl files are in mm
M_TO_MM = 1e3 

# Print params
cyl_thickness = 7.5 # mm
radius_wire_mm = 1 # mm

# Load design
design_pth = './designs/head_tight_v0/'
surface_pth = design_pth + 'surfopt=True_bmax=0.155_emax=4.79_L=192.2_eff=1.00_gmax=None.pt'
wire_pts_pth = design_pth + 'wire_path_Gx.pt'
dct = torch.load(surface_pth, weights_only=True, map_location='cpu')
rs = dct['rs'].numpy()
zs = dct['zs'].numpy()
wire_pts_list = torch.load(wire_pts_pth, weights_only=True, map_location='cpu')
wire_pts_Z = dct['wire_crds']

# Build the surface
# cyl = trimesh.creation.annulus(r_min=M_TO_MM *  dct['radius'][0] - cyl_thickness, 
#                                r_max=M_TO_MM * (dct['radius'][0] + .7 * dct['radius_wire']),
#                                height=M_TO_MM * dct['length'][0], 
#                                sections=100)

z_grid, r_out = vr_resample_profile(zs, rs, M=300)
cyl = vr_build_shell(z_grid, r_out, 
                     wall_thickness_m=cyl_thickness / M_TO_MM,
                     theta_sections=300,
                     cap_ends=True,
                     solid=False)
cyl.apply_scale(M_TO_MM)


# # Load wire points and create grooves
# wire_pts = wire_crds
# wire_len = wire_pts.diff(dim=0).norm(dim=-1).sum().item()
# wire_pts = wire_pts.numpy() * M_TO_MM
# print(f'Wire length: {wire_len:.2f} m')
# quit()

# Build a tube along wire_pts
thetas = np.linspace(0, 2 * np.pi, 30)
circle_pts = np.stack([
    radius_wire_mm * np.cos(thetas),
    radius_wire_mm * np.sin(thetas),
])

# Iterate over wire points isocontrours 
grooved_cyl = cyl
for wire_pts in tqdm(wire_pts_list, desc='Building X-Gradient grooves'):
    
    # X-Gradient grooves
    wire_pts = wire_pts.numpy() * M_TO_MM
    normal_dir = wire_pts.copy()
    normal_dir[..., -1] = 0
    normal_dir = normal_dir / np.linalg.norm(normal_dir, axis=-1)[..., None]
    wire_pts -= normal_dir * radius_wire_mm * 0.7
    groove = trimesh.creation.sweep_polygon(
        path=wire_pts,
        polygon=Polygon(circle_pts.T),
    )
    
    # Subtract the groove from the cylinder
    grooved_cyl = grooved_cyl.difference(groove, engine='manifold')
    
# Z-Gradient grooves
print(f'Building Z-Gradient grooves ... ', end='')
idx_start = 300
wire_pts = wire_pts_Z.numpy() * M_TO_MM
normal_dir = wire_pts.copy()
normal_dir[..., -1] = 0
normal_dir = normal_dir / np.linalg.norm(normal_dir, axis=-1)[..., None]
wire_pts += normal_dir * radius_wire_mm * 0.7 - normal_dir * cyl_thickness 
groove = trimesh.creation.sweep_polygon(
    path=wire_pts,
    polygon=Polygon(circle_pts.T),
)
grooved_cyl = grooved_cyl.difference(groove, engine='manifold')
print('done.')

# Fix issues
grooved_cyl = remove_end_caps(grooved_cyl, tol=1e-3)
if not grooved_cyl.is_watertight:
    trimesh.repair.fill_holes(grooved_cyl)      # plug small leaks
    grooved_cyl.nondegenerate_faces()       # drop zero-area faces
    grooved_cyl.remove_unreferenced_vertices()  # clean up verts
    grooved_cyl.merge_vertices()                # merge dupes
    grooved_cyl.fix_normals()  

# Save
grooved_cyl.export(design_pth + 'Gx_Gz.stl')
