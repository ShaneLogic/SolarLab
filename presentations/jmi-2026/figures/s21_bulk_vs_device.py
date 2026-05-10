"""s21 — bulk DFT descriptors vs device-level performance dependencies."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figures._common import configure, FIG_OUT, INK, BODY, ACCENT, HAIRLINE


def _box(ax, x, y, w, h, *, fc, ec, text, text_color=INK,
         fontsize=10, weight="normal"):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.01,rounding_size=0.04",
                         linewidth=0.8, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            color=text_color, fontsize=fontsize, weight=weight)


def main():
    configure()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Left column — bulk material descriptors from DFT.
    _box(ax, 0.3, 5.0, 3.4, 0.7,
         fc="#ecf3fb", ec="#1a73d6",
         text="DFT — bulk descriptors", text_color="#0f3a78",
         fontsize=11, weight="bold")

    bulk = [
        (r"$E_g$ — bandgap"),
        (r"$m^{*}$ — effective mass"),
        (r"$\varepsilon$ — dielectric constant"),
        (r"$E_b$ — exciton binding energy"),
    ]
    for i, txt in enumerate(bulk):
        _box(ax, 0.3, 4.0 - i * 0.85, 3.4, 0.65,
             fc="white", ec=HAIRLINE, text=txt, fontsize=10)

    # Right column — additional device-level inputs needed.
    _box(ax, 6.3, 5.0, 3.4, 0.7,
         fc="#fdecea", ec=ACCENT,
         text="Device — additional inputs", text_color=ACCENT,
         fontsize=11, weight="bold")

    device = [
        "stack geometry (layers, d)",
        "contact selectivity (S$_n$, S$_p$)",
        "coherent TMM optics",
        "recombination kinetics",
        "mobile-ion transport (2D)",
    ]
    for i, txt in enumerate(device):
        _box(ax, 6.3, 4.0 - i * 0.7, 3.4, 0.55,
             fc="white", ec=HAIRLINE, text=txt, fontsize=9.5)

    # Bridging arrow with caption.
    arrow = FancyArrowPatch((3.85, 3.0), (6.15, 3.0),
                            arrowstyle="-|>", mutation_scale=18,
                            color=INK, lw=1.4)
    ax.add_patch(arrow)
    ax.text(5.0, 3.30, "device-scale\nsimulation",
            ha="center", va="bottom",
            color=INK, fontsize=10, style="italic")
    ax.text(5.0, 2.62, "drift-diffusion +\nPoisson + TMM",
            ha="center", va="top",
            color=BODY, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIG_OUT / "s21_bulk_vs_device.png")
    print(f"wrote {FIG_OUT / 's21_bulk_vs_device.png'}")


if __name__ == "__main__":
    main()
