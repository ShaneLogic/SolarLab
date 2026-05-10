"""Schematic device-stack diagram — matches the SolarLab frontend layer builder.

Used as an inset in s18 (1D vs 2D fidelity) and s19 (predicted device J-V) so
viewers see the actual layer configuration that produced the curves.

Reference stack: spiro_HTL (200 nm, top) / MAPbI₃ (400 nm, absorber, mid) /
TiO₂_ETL (100 nm, bottom). AM1.5G illumination from the HTL side. Robin
selective-contact velocities annotated at the two interfaces.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figures._common import INK, BODY, ACCENT


HTL_COLOR = "#f3d6e3"      # pale pink
ABS_COLOR = "#3a1c4a"      # deep purple
ETL_COLOR = "#dbe6f0"      # pale blue


def draw_mapbi3_stack(ax, *, show_gb=False, title=None):
    """Render a MAPbI3 reference device stack on the given Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target Axes (will have its limits, ticks, spines reset).
    show_gb : bool, optional
        If True, draw a vertical grain-boundary line through the absorber
        (used on the 2D fidelity slide).
    title : str, optional
        Title rendered above the stack.
    """
    ax.set_xlim(-0.05, 1.40)
    ax.set_ylim(-0.05, 1.10)
    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=10, color=INK, pad=2, loc="left")

    # Layer thicknesses (nm) — drawn in proportion (200 / 400 / 100 = 700 nm).
    layers = [
        ("spiro_HTL",  "200 nm · HTL",      HTL_COLOR, 200),
        ("MAPbI$_3$",  "400 nm · absorber", ABS_COLOR, 400),
        ("TiO$_2$_ETL","100 nm · ETL",      ETL_COLOR, 100),
    ]
    total_thickness = sum(t for *_, t in layers)
    box_w = 1.0
    x0 = 0.05
    y_top = 0.95
    cumulative_y = y_top

    layer_box_y = []   # remembered for annotation
    for name, sub, color, thick in layers:
        h = (thick / total_thickness) * 0.85
        y = cumulative_y - h
        box = FancyBboxPatch((x0, y), box_w, h,
                             boxstyle="round,pad=0.005,rounding_size=0.012",
                             linewidth=0.6,
                             edgecolor="#888",
                             facecolor=color)
        ax.add_patch(box)
        # text colour matches contrast against the layer fill
        text_color = "white" if color == ABS_COLOR else INK
        ax.text(x0 + 0.04, y + h - 0.04, name,
                fontsize=10, color=text_color, weight="bold",
                ha="left", va="top")
        ax.text(x0 + 0.04, y + 0.03, sub,
                fontsize=8, color=text_color,
                ha="left", va="bottom")
        layer_box_y.append((y, y + h))
        cumulative_y = y

    # Grain-boundary vertical line through absorber (2D fidelity slide).
    if show_gb:
        gb_x = x0 + 0.55 * box_w
        gb_y0, gb_y1 = layer_box_y[1]
        ax.plot([gb_x, gb_x], [gb_y0, gb_y1],
                color=ACCENT, lw=1.6, alpha=0.85)
        ax.text(gb_x + 0.02, (gb_y0 + gb_y1) / 2.0, "GB",
                fontsize=9, color=ACCENT, weight="bold", va="center")

    # Robin selective-contact annotations between layers.
    htl_abs = layer_box_y[0][0]   # bottom of HTL = top of absorber
    abs_etl = layer_box_y[1][0]   # bottom of absorber = top of ETL

    ax.text(x0 + box_w + 0.03, htl_abs,
            r"$S_n=3{\times}10^{5}$, $S_p=10^{-1}$ m/s",
            fontsize=8, color=BODY, va="center", ha="left")
    ax.text(x0 + box_w + 0.03, abs_etl,
            r"$S_n=10^{-1}$, $S_p=3{\times}10^{5}$ m/s",
            fontsize=8, color=BODY, va="center", ha="left")

    # AM1.5G illumination arrows above the HTL.
    arrow_y0 = y_top + 0.07
    arrow_y1 = y_top + 0.005
    n_arrows = 5
    for i in range(n_arrows):
        ax_x = x0 + 0.15 + i * (box_w - 0.30) / (n_arrows - 1)
        arrow = FancyArrowPatch((ax_x, arrow_y0), (ax_x, arrow_y1),
                                 arrowstyle="-|>", mutation_scale=8,
                                 color=ACCENT, lw=1.0)
        ax.add_patch(arrow)
    ax.text(x0 + box_w / 2.0, arrow_y0 + 0.02, "AM1.5G",
            fontsize=9, color=ACCENT, ha="center", va="bottom",
            weight="bold")
