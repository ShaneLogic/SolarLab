"""s26 — Robin selective-contact schematic with surface-recombination flux arrows."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figures._common import configure, FIG_OUT, INK, BODY, ACCENT, SECONDARY, HAIRLINE


HTL_COLOR = "#f3d6e3"
ABS_COLOR = "#3a1c4a"
ETL_COLOR = "#dbe6f0"


def main():
    configure()
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Three-layer device — HTL | absorber | ETL.
    layers = [
        ("HTL",       1.0, 2.0, HTL_COLOR, INK),
        ("absorber",  3.0, 4.0, ABS_COLOR, "white"),
        ("ETL",       7.0, 2.0, ETL_COLOR, INK),
    ]
    for label, x, w, fc, tc in layers:
        box = FancyBboxPatch((x, 1.5), w, 3.0,
                             boxstyle="round,pad=0.01,rounding_size=0.05",
                             linewidth=0.8, edgecolor="#888", facecolor=fc)
        ax.add_patch(box)
        ax.text(x + w / 2, 4.20, label,
                ha="center", va="top", color=tc,
                fontsize=11, weight="bold")

    # Left contact (HTL side) — extracts holes, blocks electrons.
    # J_p flux arrow leaving the absorber to the left, with high S_p.
    ax.add_patch(FancyArrowPatch((3.05, 3.5), (2.45, 3.5),
                                 arrowstyle="-|>", mutation_scale=18,
                                 color=ACCENT, lw=1.8))
    ax.text(2.75, 3.75, r"$J_p$", color=ACCENT,
            fontsize=11, weight="bold", ha="center")
    ax.text(2.75, 3.05,
            r"$S_{p,L} = 3{\times}10^{5}$ m/s",
            color=ACCENT, fontsize=8.5, ha="center")

    # J_n flux essentially blocked at the HTL — small low-amplitude arrow.
    ax.add_patch(FancyArrowPatch((3.05, 2.2), (2.85, 2.2),
                                 arrowstyle="-|>", mutation_scale=10,
                                 color=SECONDARY, lw=1.0, alpha=0.45))
    ax.text(2.75, 1.85,
            r"$S_{n,L} = 10^{-1}$ m/s — blocked",
            color=SECONDARY, fontsize=8.5, ha="center")

    # Right contact (ETL side) — extracts electrons, blocks holes.
    ax.add_patch(FancyArrowPatch((6.95, 3.5), (7.55, 3.5),
                                 arrowstyle="-|>", mutation_scale=18,
                                 color=SECONDARY, lw=1.8))
    ax.text(7.25, 3.75, r"$J_n$", color=SECONDARY,
            fontsize=11, weight="bold", ha="center")
    ax.text(7.25, 3.05,
            r"$S_{n,R} = 3{\times}10^{5}$ m/s",
            color=SECONDARY, fontsize=8.5, ha="center")

    ax.add_patch(FancyArrowPatch((6.95, 2.2), (7.15, 2.2),
                                 arrowstyle="-|>", mutation_scale=10,
                                 color=ACCENT, lw=1.0, alpha=0.45))
    ax.text(7.25, 1.85,
            r"$S_{p,R} = 10^{-1}$ m/s — blocked",
            color=ACCENT, fontsize=8.5, ha="center")

    # Robin equation summary panel below the diagram.
    ax.text(5.0, 0.95,
            r"$J = \pm\,q\,S\,(c - c_{\mathrm{eq}})$",
            ha="center", va="center",
            color=INK, fontsize=12)
    ax.text(5.0, 0.45,
            r"$S\to\infty$: Dirichlet pinning  ·  $S\to 0$: blocking (Neumann)",
            ha="center", va="center",
            color=BODY, fontsize=9.5, style="italic")

    fig.tight_layout()
    fig.savefig(FIG_OUT / "s26_robin_contacts.png")
    print(f"wrote {FIG_OUT / 's26_robin_contacts.png'}")


if __name__ == "__main__":
    main()
