"""s3 — CIGS chalcopyrite structure schematic + bandgap–composition tunability."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from figures._common import configure, FIG_OUT, ACCENT, INK, BODY, SECONDARY


CU_COLOR = "#c08030"     # copper
INGA_COLOR = "#7a8aa8"   # In/Ga (gray-blue)
SE_COLOR = "#d8b020"     # Se (yellow-ochre)


def _cube_edges(ax, a, c, color="#888", lw=1.0):
    """Draw the wireframe of a tetragonal box of dimensions (a, a, c)."""
    verts = np.array([
        [0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0],
        [0, 0, c], [a, 0, c], [a, a, c], [0, a, c],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        xs, ys, zs = zip(verts[i], verts[j])
        ax.plot(xs, ys, zs, color=color, lw=lw)


def _atom(ax, x, y, z, color, size=140, label=None):
    ax.scatter([x], [y], [z], color=color, s=size,
               edgecolor="white", linewidth=0.8, depthshade=True)


def _draw_chalcopyrite(ax, a=5.78, c=11.62):
    """Schematic chalcopyrite Cu(In,Ga)Se2 unit cell.

    Chalcopyrite I-42d is a doubled zincblende: cations Cu / (In,Ga) sit on
    alternating Wyckoff sites along c, anions Se on the tetrahedral sites.
    Coordinates here are illustrative (one of the four formula units shown).
    """
    _cube_edges(ax, a, c)

    # Cu at (0,0,0), (a/2, a/2, c/4), and z+c/2 partners
    cu_sites = [
        (0.0, 0.0, 0.0),
        (a / 2, a / 2, c / 4),
        (a / 2, 0.0, c / 2),
        (0.0, a / 2, 3 * c / 4),
    ]
    for s in cu_sites:
        _atom(ax, *s, CU_COLOR, size=160)

    # In/Ga interleaved on the other cation sublattice
    inga_sites = [
        (a / 2, a / 2, 0.0),
        (0.0, 0.0, c / 4),
        (0.0, a / 2, c / 2),
        (a / 2, 0.0, 3 * c / 4),
    ]
    for s in inga_sites:
        _atom(ax, *s, INGA_COLOR, size=160)

    # Se on the tetrahedral anion sublattice (four representative sites)
    se_sites = [
        (a / 4, a / 4, c / 8),
        (3 * a / 4, 3 * a / 4, c / 8),
        (a / 4, 3 * a / 4, 3 * c / 8),
        (3 * a / 4, a / 4, 3 * c / 8),
        (a / 4, a / 4, 5 * c / 8),
        (3 * a / 4, 3 * a / 4, 5 * c / 8),
        (a / 4, 3 * a / 4, 7 * c / 8),
        (3 * a / 4, a / 4, 7 * c / 8),
    ]
    for s in se_sites:
        _atom(ax, *s, SE_COLOR, size=110)


def main():
    configure()
    fig = plt.figure(figsize=(11, 4.5))

    # Left panel: 3D unit-cell schematic
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    _draw_chalcopyrite(ax)
    ax.set_xlabel("a (Å)")
    ax.set_ylabel("a (Å)")
    ax.set_zlabel("c (Å)")
    ax.set_title("Cu(In,Ga)Se$_2$ chalcopyrite · I-4$\\bar{2}$d",
                 fontsize=11, color=INK, pad=2)
    ax.view_init(elev=18, azim=-58)
    # legend dots
    for color, label in [(CU_COLOR, "Cu"), (INGA_COLOR, "In / Ga"),
                          (SE_COLOR, "Se")]:
        ax.scatter([], [], [], color=color, s=80, edgecolor="white",
                   linewidth=0.6, label=label)
    ax.legend(loc="upper left", frameon=False, fontsize=10,
              bbox_to_anchor=(0.0, 1.0))

    # Right panel: Eg(x) tunability, linear from CIS (1.04) to CGS (1.68)
    ax2 = fig.add_subplot(1, 2, 2)
    x = np.linspace(0.0, 1.0, 100)
    eg = 1.04 + (1.68 - 1.04) * x  # near-linear empirical
    ax2.plot(x, eg, color=ACCENT, lw=2.2)
    # SQ optimum band (1.34 ± 0.10 eV)
    ax2.axhspan(1.24, 1.44, color=SECONDARY, alpha=0.10,
                label="SQ optimum window")
    ax2.axhline(1.34, color=SECONDARY, lw=0.8, ls="--")
    ax2.scatter([0.0, 1.0], [1.04, 1.68], color=ACCENT, s=60,
                edgecolor="white", linewidth=0.8, zorder=3)
    ax2.text(0.0, 0.96, "CuInSe$_2$\n1.04 eV", ha="left", va="top",
             fontsize=10, color=INK)
    ax2.text(1.0, 1.74, "CuGaSe$_2$\n1.68 eV", ha="right", va="bottom",
             fontsize=10, color=INK)
    ax2.set_xlabel(r"x = Ga / (In + Ga)")
    ax2.set_ylabel(r"$E_g$ (eV)")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(0.9, 1.85)
    ax2.set_title("Direct-gap tunability with Ga/In ratio",
                  fontsize=11, color=INK, pad=2)
    ax2.legend(frameon=False, loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_OUT / "s3_cigs_structure.png")
    print(f"wrote {FIG_OUT / 's3_cigs_structure.png'}")


if __name__ == "__main__":
    main()
