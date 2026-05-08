import torch

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from matplotlib import cm, colors

def show_3d_slices(xy_slice, xz_slice, yz_slice, xy_crds, xz_crds, yz_crds,
                   vmin=None, vmax=None, cmap=None):
    
    # To cpu
    xy_slice = xy_slice.cpu()
    xz_slice = xz_slice.cpu()
    yz_slice = yz_slice.cpu()
    xy_crds = xy_crds.cpu()
    xz_crds = xz_crds.cpu()
    yz_crds = yz_crds.cpu()
     
    # Min max
    if vmin is None:
        vmin = min(xy_slice.min().item(), xz_slice.min().item(), yz_slice.min().item())
    if vmax is None:
        vmax = max(xy_slice.max().item(), xz_slice.max().item(), yz_slice.max().item())
    
    # Colormap normalization shared across all slices
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    if cmap is None:
        cmap = cm.viridis
    elif isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    elif isinstance(cmap, matplotlib.colors.Colormap):
        cmap = cmap
    else:
        raise ValueError(f'Invalid cmap: {cmap}')
    face_yz = cmap(norm(yz_slice))
    face_xz = cmap(norm(xz_slice))
    face_xy = cmap(norm(xy_slice))

    # 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    plot_surface_kwargs = {
        'shade': False,
        'rstride': 1,
        'cstride': 1,
        'linewidth': 0,
        'antialiased': False,
        'alpha': 0.5,
    }

    # --- yz plane at x = x[ix]
    ax.plot_surface(
        yz_crds[..., 0], yz_crds[..., 1], yz_crds[..., 2],
        facecolors=face_yz,
        **plot_surface_kwargs
    )

    # --- xz plane at y = y[iy]
    ax.plot_surface(
        xz_crds[..., 0], xz_crds[..., 1], xz_crds[..., 2],
        facecolors=face_xz,
        **plot_surface_kwargs
    )

    # --- xy plane at z = z[iz]
    ax.plot_surface(
        xy_crds[..., 0], xy_crds[..., 1], xy_crds[..., 2],
        facecolors=face_xy,
        **plot_surface_kwargs
    )

    # Optional colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.08)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal')
    
    return fig, ax



def show_3d_slices(xy_slice, xz_slice, yz_slice, xy_crds, xz_crds, yz_crds,
                   vmin=None, vmax=None, colorscale="RdBu_r"):
    
    # To cpu
    xy_slice = xy_slice.cpu()
    xz_slice = xz_slice.cpu()
    yz_slice = yz_slice.cpu()
    xy_crds = xy_crds.cpu()
    xz_crds = xz_crds.cpu()
    yz_crds = yz_crds.cpu()
     
    # Min max
    if vmin is None:
        vmin = min(xy_slice.min().item(), xz_slice.min().item(), yz_slice.min().item())
    if vmax is None:
        vmax = max(xy_slice.max().item(), xz_slice.max().item(), yz_slice.max().item())
        
    # 3D plotly figure
    fig = go.Figure()    
    plot_surface_kwargs = {
        'cmin': vmin,
        'cmax': vmax,
        'colorscale': colorscale,
    }

    # # --- yz plane at x = x[ix]
    # fig.add_trace(go.Surface(
    #     x=yz_crds[..., 0],
    #     y=yz_crds[..., 1],
    #     z=yz_crds[..., 2],
    #     surfacecolor=yz_slice,
    #     **plot_surface_kwargs
    # ))

    # --- xz plane at y = y[iy]
    fig.add_trace(go.Surface(
        x=xz_crds[..., 0],
        y=xz_crds[..., 1],
        z=xz_crds[..., 2],
        surfacecolor=xz_slice,
        **plot_surface_kwargs
    ))

    # # --- xy plane at z = z[iz]
    # fig.add_trace(go.Surface(
    #     x=xy_crds[..., 0],
    #     y=xy_crds[..., 1],
    #     z=xy_crds[..., 2],
    #     surfacecolor=xy_slice,
    #     **plot_surface_kwargs
    # ))

    fig.update_layout(
        scene=dict(
            # xaxis_title="x (cm)",
            # yaxis_title="y (cm)",
            # zaxis_title="z (cm)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
    )
    
    theta = torch.linspace(0, 2*torch.pi, 30)
    z = torch.linspace(-0.15, 0.24, 20)
    # ------------------------------------------------
    # Vertical lines (constant theta)
    # ------------------------------------------------
    for t in theta:
        x = 0.15 * torch.cos(t) * torch.ones_like(z)
        y = 0.15 * torch.sin(t) * torch.ones_like(z)

        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ))

    # ------------------------------------------------
    # Horizontal rings (constant z)
    # ------------------------------------------------
    theta_dense = torch.linspace(0, 2*torch.pi, 200)

    for zi in z:
        x = 0.15 * torch.cos(theta_dense)
        y = 0.15 * torch.sin(theta_dense)
        z_ring = torch.full_like(theta_dense, zi)

        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z_ring,
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ))
        
    return fig