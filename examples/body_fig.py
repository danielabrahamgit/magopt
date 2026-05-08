import numpy as np
import matplotlib as mpl
mpl.use('WebAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch

def pediatric_body_z_gradient_schematic(
    L=120.0, R=30.0,
    pns_len_frac=0.78, pns_rad_frac=0.72, pns_x0_frac=0.10,
    dsv_a_frac=0.33, dsv_b_frac=0.55, dsv_xc_frac=0.5,
    n_loops=22, loop_rx_frac=0.018,
    title="Pediatric Body Design",
):
    """
    Simple 2D side-view schematic:
      - Outer coil former: black
      - Windings: red loops
      - PNS surface proxy: orange rounded rectangle (elliptical cylinder in side view)
      - DSV: green ellipse
    Coordinates: x ~ Z (head-to-toe), y ~ Y (posterior-anterior)
    """

    # ----- Derived geometry -----
    x0, x1 = 0.0, L

    # PNS proxy region (rounded rectangle)
    Lpns = pns_len_frac * L
    Rpns = pns_rad_frac * R
    xpns0 = x0 + pns_x0_frac * L
    xpns1 = xpns0 + Lpns

    # DSV ellipse
    a_dsv = dsv_a_frac * L           # semi-axis along x (Z)
    b_dsv = dsv_b_frac * Rpns        # semi-axis along y
    xc_dsv = x0 + dsv_xc_frac * L
    yc_dsv = 0.0

    # Windings
    th = np.linspace(0, 2*np.pi, 400)
    loop_rx = loop_rx_frac * L
    loop_ry = 1.00 * R
    x_loops = np.linspace(x0 + 0.06*L, x1 - 0.06*L, n_loops)

    # Colors (tweak to match your palette)
    col_red    = (0.90, 0.10, 0.10)
    col_orange = (1.00, 0.70, 0.40)
    col_green  = (0.55, 0.85, 0.55)

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
    ax.set_aspect("equal")
    ax.axis("off")

    # Outer coil former (black)
    ax.plot([x0, x1], [ R,  R], color="black", lw=4)
    ax.plot([x0, x1], [-R, -R], color="black", lw=4)
    ax.plot([x0, x0], [-R,  R], color="black", lw=4)

    # End-cap ellipse (right side)
    endcap_rx = 0.06 * L
    endcap = Ellipse((x1, 0), width=2*endcap_rx, height=2*R,
                     fill=False, ec="black", lw=4)
    ax.add_patch(endcap)

    # # Red windings (stylized loops)
    # for xi in x_loops:
    #     xloop = xi + loop_rx * np.cos(th)
    #     yloop =       loop_ry * np.sin(th)
    #     ax.plot(xloop, yloop, color=col_red, lw=2)

    # PNS region: rounded rectangle fill + outline
    # FancyBboxPatch uses corner rounding in data coords; pick rounding ~ Rpns
    pns_patch = FancyBboxPatch(
        (xpns0, -Rpns), xpns1 - xpns0, 2*Rpns,
        boxstyle=f"round,pad=0,rounding_size={Rpns}",
        facecolor=col_orange, edgecolor="black", lw=2, alpha=0.45
    )
    ax.add_patch(pns_patch)

    # DSV ellipse: fill + outline
    dsv = Ellipse((xc_dsv, yc_dsv), width=2*a_dsv, height=2*b_dsv,
                  facecolor=col_green, edgecolor="black", lw=1.5, alpha=0.60)
    ax.add_patch(dsv)

    # Labels
    ax.text(xpns0 + 0.08*Lpns, -0.55*Rpns, r"$\Omega_{pns}$", fontsize=16)
    ax.text(xc_dsv - 0.10*a_dsv, -0.10*b_dsv, r"$\Omega_{dsv}$", fontsize=16)

    # Title
    ax.set_title(title, fontsize=20, fontweight="bold", pad=10)

    # View limits
    ax.set_xlim(x0 - 0.12*L, x1 + 0.18*L)
    ax.set_ylim(-1.25*R, 1.25*R)

    return fig, ax

if __name__ == "__main__":
    fig, ax = pediatric_body_z_gradient_schematic()
    # Save if you want
    # fig.savefig("peds_body_design.png", dpi=300, bbox_inches="tight")
    plt.show()
