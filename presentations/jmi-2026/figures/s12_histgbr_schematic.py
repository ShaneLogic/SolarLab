"""s12 — HistGBR stage-wise additive boosting schematic."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figures._common import configure, FIG_OUT, INK, BODY, ACCENT, SECONDARY, HAIRLINE


def main():
    configure()
    fig = plt.figure(figsize=(7.6, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.30)
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    # Left panel — boxes representing successive trees, residual feeds forward.
    ax_left.set_xlim(0, 10)
    ax_left.set_ylim(0, 6)
    ax_left.axis("off")

    box_w = 1.6
    box_h = 1.4
    gap = 0.30
    y_box = 3.5
    x_starts = [0.4, 0.4 + box_w + gap, 0.4 + 2 * (box_w + gap),
                0.4 + 3 * (box_w + gap)]
    labels = [r"$h_1(x)$", r"$h_2(x)$", r"$h_3(x)$", r"$h_M(x)$"]
    for i, (x, lab) in enumerate(zip(x_starts, labels)):
        if i == 3:
            ax_left.text(x - 0.30, y_box + box_h / 2, "…",
                         ha="center", va="center",
                         color=BODY, fontsize=14)
        box = FancyBboxPatch((x, y_box), box_w, box_h,
                             boxstyle="round,pad=0.01,rounding_size=0.05",
                             linewidth=0.8, edgecolor=ACCENT,
                             facecolor="#fdecea")
        ax_left.add_patch(box)
        ax_left.text(x + box_w / 2, y_box + box_h / 2, lab,
                     ha="center", va="center",
                     color=ACCENT, fontsize=11, weight="bold")
        if i < 3:
            ax_left.add_patch(FancyArrowPatch(
                (x + box_w, y_box + box_h / 2),
                (x_starts[i + 1], y_box + box_h / 2),
                arrowstyle="-|>", mutation_scale=12,
                color=INK, lw=1.0))
        ax_left.text(x + box_w / 2, y_box - 0.30, "stage " + str(i + 1) if i < 3 else "stage M",
                     ha="center", va="top",
                     color=BODY, fontsize=8, style="italic")

    # Recurrence equation rendered below the box row.
    ax_left.text(5.0, 1.7,
                 r"$F_m(x) = F_{m-1}(x) + \nu\,h_m(x)$",
                 ha="center", va="center",
                 color=INK, fontsize=11)
    ax_left.text(5.0, 0.85,
                 r"$h_m \leftarrow$ negative gradient of loss at $F_{m-1}$",
                 ha="center", va="center",
                 color=BODY, fontsize=9, style="italic")
    ax_left.set_title("Stage-wise additive ensemble",
                      loc="left", fontsize=10, color=INK, pad=2)

    # Right panel — log1p target transform compressing the heavy-tailed PV score.
    y = np.linspace(0.0, 0.6, 400)
    z = np.log1p(y / 0.08)
    ax_right.plot(y, z, color=SECONDARY, lw=1.8)
    ax_right.set_xlabel(r"PV score $y$", fontsize=9)
    ax_right.set_ylabel(r"$\log_1 p(y/0.08)$", fontsize=9)
    ax_right.tick_params(axis="both", labelsize=8)
    ax_right.set_title("Heavy-tailed target compression",
                       loc="left", fontsize=10, color=INK, pad=2)
    ax_right.axvline(0.5, color=BODY, lw=0.6, ls=":")
    ax_right.text(0.50, 0.10, "peak", color=BODY, fontsize=8,
                  ha="center", va="bottom", style="italic")
    ax_right.axvline(0.01, color=BODY, lw=0.6, ls=":")
    ax_right.text(0.01, 0.10, "  bulk", color=BODY, fontsize=8,
                  ha="left", va="bottom", style="italic")

    fig.tight_layout()
    fig.savefig(FIG_OUT / "s12_histgbr_schematic.png")
    print(f"wrote {FIG_OUT / 's12_histgbr_schematic.png'}")


if __name__ == "__main__":
    main()
