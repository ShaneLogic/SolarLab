"""s4 — ABX3 perovskite structure + Goldschmidt tolerance-factor map."""
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from figures._common import configure, FIG_OUT, ACCENT, INK, BODY, SECONDARY


A_COLOR = "#1c2533"      # A cation (large, dark)
B_COLOR = "#888888"       # B cation (small, gray)
X_COLOR = "#a830a0"       # X anion (purple — halide)


def _cube_edges(ax, a):
    verts = np.array(list(product([0, a], repeat=3)))
    edges = []
    for i, v1 in enumerate(verts):
        for j, v2 in enumerate(verts):
            if j <= i:
                continue
            d = v2 - v1
            if (d != 0).sum() == 1:
                edges.append((v1, v2))
    for v1, v2 in edges:
        ax.plot(*zip(v1, v2), color="#888", lw=1.0)


def _atom(ax, x, y, z, color, size=160):
    ax.scatter([x], [y], [z], color=color, s=size,
               edgecolor="white", linewidth=0.8, depthshade=True)


def _draw_octahedron(ax, center, half, color=B_COLOR, alpha=0.12):
    """Faint BX6 octahedron at center, half-edge half (= a/2)."""
    cx, cy, cz = center
    # six vertices: ±half along each axis from the center
    verts = np.array([
        [cx + half, cy, cz], [cx - half, cy, cz],
        [cx, cy + half, cz], [cx, cy - half, cz],
        [cx, cy, cz + half], [cx, cy, cz - half],
    ])
    faces = [
        (0, 2, 4), (0, 2, 5), (0, 3, 4), (0, 3, 5),
        (1, 2, 4), (1, 2, 5), (1, 3, 4), (1, 3, 5),
    ]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    poly = Poly3DCollection([verts[list(f)] for f in faces],
                             alpha=alpha, facecolor=color,
                             edgecolor=color, linewidths=0.6)
    ax.add_collection3d(poly)


def _draw_perovskite(ax, a=6.3):
    """Cubic Pm-3m perovskite ABX3 with A at center, B at corners, X at edges."""
    _cube_edges(ax, a)

    # B cations at the 8 cube corners
    for v in product([0, a], repeat=3):
        _atom(ax, *v, B_COLOR, size=110)

    # X anions at the 12 edge midpoints
    edge_mids = []
    for axis in range(3):
        for fixed in product([0, a], repeat=2):
            mid = list(fixed)
            mid.insert(axis, a / 2)
            edge_mids.append(mid)
    for m in edge_mids:
        _atom(ax, *m, X_COLOR, size=130)

    # A cation at the cube body center
    _atom(ax, a / 2, a / 2, a / 2, A_COLOR, size=240)

    # Faint BX6 octahedron around one corner B atom to visualise the framework
    # Move the central B by half-cell, octahedron centered there:
    _draw_octahedron(ax, (0, 0, 0), a / 2, color="#cccccc", alpha=0.10)


def main():
    configure()
    fig = plt.figure(figsize=(11, 4.5))

    # Left panel: 3D unit-cell schematic
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    _draw_perovskite(ax)
    ax.set_xlabel("a (Å)")
    ax.set_ylabel("a (Å)")
    ax.set_zlabel("a (Å)")
    ax.set_title(r"ABX$_3$ perovskite · Pm-3$\bar{m}$ cubic parent",
                 fontsize=11, color=INK, pad=2)
    ax.view_init(elev=18, azim=-58)
    for color, label in [(A_COLOR, "A: Cs / MA / FA"),
                          (B_COLOR, "B: Pb / Sn"),
                          (X_COLOR, "X: I / Br / Cl")]:
        ax.scatter([], [], [], color=color, s=80, edgecolor="white",
                   linewidth=0.6, label=label)
    ax.legend(loc="upper left", frameon=False, fontsize=9,
              bbox_to_anchor=(0.0, 1.0))

    # Right panel: Goldschmidt tolerance-factor regimes
    ax2 = fig.add_subplot(1, 2, 2)
    # Show example cations on the t-axis for MAPbI3, FAPbI3, CsPbI3
    points = [
        ("CsPbI$_3$", 0.81),
        ("MAPbI$_3$", 0.91),
        ("FAPbI$_3$", 0.99),
        ("FAPbBr$_3$", 1.01),
    ]
    # color regions
    ax2.axvspan(0.0, 0.71, color="#dddddd", alpha=0.6)
    ax2.axvspan(0.71, 0.80, color="#f5d0d0", alpha=0.6)
    ax2.axvspan(0.80, 0.90, color="#fce4b3", alpha=0.6)
    ax2.axvspan(0.90, 1.00, color="#cfe8d0", alpha=0.6)
    ax2.axvspan(1.00, 1.10, color="#fcd0d0", alpha=0.6)
    ax2.text(0.55, 0.82, "non-perovskite",
             ha="center", fontsize=9, color="#666",
             transform=ax2.get_xaxis_transform())
    ax2.text(0.755, 0.82, "tilted /\northo.",
             ha="center", fontsize=9, color="#666",
             transform=ax2.get_xaxis_transform())
    ax2.text(0.85, 0.82, "tetragonal",
             ha="center", fontsize=9, color="#666",
             transform=ax2.get_xaxis_transform())
    ax2.text(0.95, 0.82, "cubic",
             ha="center", fontsize=9, color="#444", weight="bold",
             transform=ax2.get_xaxis_transform())
    ax2.text(1.05, 0.82, "hex.",
             ha="center", fontsize=9, color="#666",
             transform=ax2.get_xaxis_transform())

    # Alternate label rows so closely spaced points (e.g. FAPbI3 at t = 0.99
    # vs FAPbBr3 at t = 1.01) do not clash. A short tick line connects each
    # marker to its label.
    label_rows = [0.28, 0.14]   # alternating y positions
    for i, (label, t) in enumerate(points):
        y_label = label_rows[i % 2]
        ax2.scatter([t], [0.42], color=ACCENT, s=80,
                    edgecolor="white", linewidth=0.8, zorder=3)
        ax2.plot([t, t], [0.40, y_label + 0.03],
                 color=BODY, lw=0.6, zorder=2)
        ax2.text(t, y_label, label, ha="center", va="top",
                 fontsize=9, color=INK)

    ax2.set_xlim(0.55, 1.10)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlabel(r"Goldschmidt tolerance factor $t$")
    ax2.set_title(r"$t = (r_A + r_X) \,/\, \sqrt{2}\,(r_B + r_X)$",
                  fontsize=11, color=INK, pad=2)
    ax2.spines["left"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_OUT / "s4_perovskite_structure.png")
    print(f"wrote {FIG_OUT / 's4_perovskite_structure.png'}")


if __name__ == "__main__":
    main()
