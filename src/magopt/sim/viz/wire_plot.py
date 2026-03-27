import matplotlib.pyplot as plt


def show_wire(wire_pts, fig=None, ax=None):
    """
    Plot wire coordinates in 3D.

    Args:
    -----
    wire_pts : torch.Tensor
        Wire points with shape [N, 3] in meters.
    fig : Optional[matplotlib.figure.Figure]
        Existing figure handle.
    ax : Optional[matplotlib.axes.Axes]
        Existing 3D axis handle.

    Returns:
    --------
    tuple
        Figure and axis used for plotting.
    """
    wire_pts = wire_pts.cpu() * 1e2
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.plot(wire_pts[:, 0], wire_pts[:, 1], wire_pts[:, 2],
            alpha=.8, linewidth=0.5, color='red')
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_zlabel('Z [cm]')
    plt.axis('equal')
    return fig, ax
