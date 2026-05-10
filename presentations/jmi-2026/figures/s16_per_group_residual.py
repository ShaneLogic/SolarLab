"""s16 — per-group residual audit bar chart with priority highlight."""
import numpy as np
import matplotlib.pyplot as plt
from figures._common import configure, FIG_OUT, INK, BODY, ACCENT, SECONDARY


GROUPS = [
    # (label, n, RMSE, top20_total, top20_missed, priority_score)
    ("Se",            182, 0.042, 17, 9, "highest"),
    ("y > 0.1",        14, 0.145, 14, 7, "high"),
    ("Pnma (SG 62)",  182, 0.034, 10, 4, "mid"),
    (r"AB$_2$C$_4$",   91, 0.045, 7, 1, "low"),
    (r"ABC$_3$",       56, 0.045, 6, 3, "high"),
]


def main():
    configure()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0),
                              gridspec_kw={"width_ratios": [1.0, 1.0],
                                           "wspace": 0.35})
    ax_rmse, ax_recall = axes

    labels = [g[0] for g in GROUPS]
    rmses = np.array([g[2] for g in GROUPS])
    n_arr = np.array([g[1] for g in GROUPS])
    miss = np.array([g[4] for g in GROUPS])
    total = np.array([g[3] for g in GROUPS])
    miss_rate = miss / total

    priority = [g[5] for g in GROUPS]
    color_map = {"highest": ACCENT, "high": "#e07c4a",
                 "mid": SECONDARY, "low": "#1aa37a"}
    bar_colors = [color_map[p] for p in priority]

    y = np.arange(len(labels))[::-1]   # top-of-axis goes first
    ax_rmse.barh(y, rmses, color=bar_colors, edgecolor=INK, linewidth=0.4)
    for yi, r, n in zip(y, rmses, n_arr):
        ax_rmse.text(r + 0.003, yi, f"  RMSE {r:.3f} · n={n}",
                     va="center", ha="left",
                     color=BODY, fontsize=8)
    ax_rmse.set_yticks(y)
    ax_rmse.set_yticklabels(labels, fontsize=9)
    ax_rmse.set_xlim(0, 0.20)
    ax_rmse.set_xlabel("RMSE on PV score")
    ax_rmse.set_title("Residual size per group",
                      loc="left", fontsize=10, color=INK, pad=2)

    ax_recall.barh(y, miss_rate * 100, color=bar_colors,
                   edgecolor=INK, linewidth=0.4)
    for yi, mr, m, t in zip(y, miss_rate, miss, total):
        ax_recall.text(mr * 100 + 1.5, yi,
                       f"  {m}/{t} missed",
                       va="center", ha="left",
                       color=BODY, fontsize=8)
    ax_recall.set_yticks(y)
    ax_recall.set_yticklabels([])
    ax_recall.set_xlim(0, 70)
    ax_recall.set_xlabel("Top-20 miss rate (%)")
    ax_recall.set_title("Ranking gap at the high-score tail",
                        loc="left", fontsize=10, color=INK, pad=2)

    fig.suptitle(
        r"Score $= \mathrm{RMSE}\sqrt{n} + 0.02\,\cdot\,(\mathrm{missed\ top}\text{-}20)$"
        " · drives the next active-learning batch",
        fontsize=9.5, color=BODY, y=1.00, style="italic")

    fig.tight_layout()
    fig.savefig(FIG_OUT / "s16_per_group_residual.png")
    print(f"wrote {FIG_OUT / 's16_per_group_residual.png'}")


if __name__ == "__main__":
    main()
