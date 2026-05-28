from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "radau_transient_solve_overview.png"
SVG_PATH = OUT_DIR / "radau_transient_solve_overview.svg"

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "dejavusans"


COLORS = {
    "bg": "#F7F9FC",
    "navy": "#162033",
    "slate": "#334155",
    "muted": "#64748B",
    "line": "#CBD5E1",
    "blue": "#2563EB",
    "blue_light": "#DBEAFE",
    "green": "#16803C",
    "green_light": "#DCFCE7",
    "amber": "#A16207",
    "amber_light": "#FEF3C7",
    "purple": "#6D28D9",
    "purple_light": "#EDE9FE",
    "red": "#B42318",
    "red_light": "#FEE4E2",
    "white": "#FFFFFF",
}


def box(ax, xy, wh, title, body, fill, edge, title_color=None, body_size=12.5):
    x, y = xy
    w, h = wh
    patch = Rectangle(
        (x, y),
        w,
        h,
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.4,
        joinstyle="round",
    )
    ax.add_patch(patch)
    ax.add_patch(Rectangle((x, y), 0.08, h, facecolor=edge, edgecolor=edge, linewidth=0))
    ax.text(
        x + 0.20,
        y + h - 0.18,
        title,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=title_color or edge,
    )
    ax.text(
        x + 0.20,
        y + h - 0.48,
        body,
        ha="left",
        va="top",
        fontsize=body_size,
        color=COLORS["navy"],
        linespacing=1.22,
    )


def arrow(ax, start, end, color=None, lw=1.8, curve=0.0, label=None, label_offset=(0, 0)):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=lw,
        color=color or COLORS["muted"],
        connectionstyle=f"arc3,rad={curve}",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arr)
    if label:
        mx = 0.5 * (start[0] + end[0]) + label_offset[0]
        my = 0.5 * (start[1] + end[1]) + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=10.5, color=color or COLORS["muted"])


def make_figure() -> None:
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=180)
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 13.33)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    ax.text(
        0.55,
        7.04,
        "Radau Transient Solve: Implicit Time Integration of the Semi-Discrete System",
        fontsize=22,
        fontweight="bold",
        color=COLORS["navy"],
        ha="left",
        va="top",
    )
    ax.text(
        0.56,
        6.62,
        "Outer solver for one fixed applied voltage $V_{app}$; the inner RHS operator $F(Y)$ is evaluated repeatedly.",
        fontsize=12.2,
        color=COLORS["slate"],
        ha="left",
        va="top",
    )

    # Main three-stage workflow
    box(
        ax,
        (0.55, 4.85),
        (2.55, 1.05),
        "INITIAL-VALUE PROBLEM",
        "$\\dfrac{dY}{dt}=F(Y;V_{app})$\n$Y(0)=Y_0$",
        COLORS["blue_light"],
        COLORS["blue"],
        body_size=14,
    )
    box(
        ax,
        (4.03, 4.48),
        (3.72, 1.62),
        "IMPLICIT RADAU IIA STEP",
        "$K_i=F(t_n+c_i h,\\;Y_n+h\\sum_j a_{ij}K_j)$\n$Y_{n+1}=Y_n+h\\sum_i b_iK_i$",
        COLORS["purple_light"],
        COLORS["purple"],
        body_size=12.6,
    )
    box(
        ax,
        (8.62, 4.85),
        (2.28, 1.05),
        "SETTLED STATE",
        "$Y(t_{end})$\nquasi-steady carrier field",
        COLORS["green_light"],
        COLORS["green"],
        body_size=12.8,
    )
    box(
        ax,
        (11.34, 4.85),
        (1.45, 1.05),
        "OUTPUT",
        "$J(V_{app})$\nterminal current",
        COLORS["amber_light"],
        COLORS["amber"],
        body_size=12.4,
    )
    arrow(ax, (3.12, 5.38), (4.00, 5.38), COLORS["muted"])
    arrow(ax, (7.78, 5.38), (8.60, 5.38), COLORS["muted"])
    arrow(ax, (10.94, 5.38), (11.32, 5.38), COLORS["muted"])

    # Time-axis schematic
    ax.plot([0.72, 12.55], [3.78, 3.78], color=COLORS["line"], linewidth=2.1)
    t_positions = [1.05, 3.25, 5.45, 7.65, 9.85, 12.05]
    labels = ["$t_0$", "$t_1$", "$t_2$", "$\\cdots$", "$t_N$", "$t_{end}$"]
    for x, lab in zip(t_positions, labels):
        ax.plot([x, x], [3.66, 3.90], color=COLORS["slate"], linewidth=1.5)
        ax.text(x, 3.43, lab, ha="center", va="top", fontsize=11.5, color=COLORS["slate"])

    # Highlight one implicit step with stage nodes.
    ax.add_patch(Rectangle((3.95, 3.18), 3.92, 0.96, facecolor="#FFFFFF", edgecolor=COLORS["purple"], linewidth=1.3))
    ax.text(
        4.10,
        4.00,
        "one implicit step",
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["purple"],
    )
    ax.plot([4.45, 7.40], [3.57, 3.57], color=COLORS["purple"], linewidth=1.7)
    for i, x in enumerate([4.95, 5.85, 6.95], start=1):
        ax.scatter([x], [3.57], s=135, color=COLORS["purple_light"], edgecolors=COLORS["purple"], linewidths=1.5, zorder=5)
        ax.text(x, 3.57, f"$K_{i}$", ha="center", va="center", fontsize=10.2, color=COLORS["purple"], zorder=6)
    ax.text(
        5.90,
        3.28,
        "nonlinear coupled stage equations",
        ha="center",
        va="center",
        fontsize=10.0,
        color=COLORS["slate"],
    )

    # Inner RHS operator, shown as a callable kernel beneath the implicit step.
    box(
        ax,
        (3.98, 1.52),
        (3.84, 1.15),
        "INNER RHS OPERATOR  $F(Y)$",
        "Poisson solve  +  generation/recombination\n+ SG flux divergence  +  contact BCs",
        COLORS["white"],
        COLORS["blue"],
        body_size=11.8,
    )
    arrow(ax, (5.88, 4.45), (5.88, 2.72), COLORS["blue"], lw=1.8, curve=0.0, label="repeated calls", label_offset=(0.72, 0.00))
    arrow(ax, (7.05, 2.70), (7.05, 4.43), COLORS["blue"], lw=1.4, curve=0.20)

    # Robustness / solver controls side panel.
    box(
        ax,
        (8.55, 1.38),
        (4.25, 1.42),
        "WHY RADAU?",
        "Stiff carrier-transport system\nstrong electrostatic coupling\nsharp heterointerface gradients\nimplicit method improves stability",
        COLORS["red_light"],
        COLORS["red"],
        body_size=11.3,
    )
    ax.text(
        0.65,
        1.22,
        "Implementation mapping:  run_transient_2d(...) wraps solve_ivp(..., method=\"Radau\"); each RHS call invokes assemble_rhs_2d(...).",
        fontsize=10.7,
        color=COLORS["muted"],
        ha="left",
        va="center",
    )
    ax.text(
        0.65,
        0.68,
        "Takeaway: RHS assembly defines the instantaneous operator F(Y); Radau repeatedly evaluates F(Y) to advance the state toward a voltage-specific quasi-steady solution.",
        fontsize=12.0,
        fontweight="bold",
        color=COLORS["navy"],
        ha="left",
        va="center",
    )

    fig.savefig(PNG_PATH, bbox_inches="tight", facecolor=COLORS["bg"])
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)


if __name__ == "__main__":
    print(font_manager.findfont("Arial", fallback_to_default=False))
    make_figure()
    print(PNG_PATH)
    print(SVG_PATH)
