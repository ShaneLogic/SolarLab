#!/usr/bin/env python3
"""f6_survival.png — Adversarial 3-lens survival summary (lollipop chart)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab/docs/partner/ppt_root_cause/figures/f6_survival.png"

# Verified findings: (label, lenses held / 3, sublabel)
# Ordered bottom-to-top so the survivor sits at the top of the chart.
rows = [
    ("J$_{sc}$ -2% optical loss",            0, "physics / numerics / data"),
    ("N$_{t,PVK\\!/\\!ETL}$ interface channel", 0, "physics / numerics / data"),
    ("N$_{t,C}$/N$_{t,V}$ bulk insensitivity", 1, "physics only"),
    ("ΔE$_C$ (CHI$_{ETL}$) CBO sweep",   1, "physics only"),
    ("PCE / FF self-consistency",            1, "reporting artifact"),
    ("V$_{oc}$ -96 mV deficit",              1, "mechanism contested"),
    ("N$_{d,ETL}$ donor-doping reversal",    3, "ALL THREE lenses"),
]

labels   = [r[0] for r in rows]
counts   = [r[1] for r in rows]
sublabels= [r[2] for r in rows]
y        = list(range(len(rows)))

SURV   = "#1a7f37"   # survivor green
FAIL   = "#b0b6bd"   # grey for non-survivors
ACCENT = "#c2410c"   # threshold line / annotation

fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for yi, c in zip(y, counts):
    is_surv = c >= 2
    col = SURV if is_surv else FAIL
    lw  = 4.5 if is_surv else 3.0
    ms  = 17  if is_surv else 12
    # stem
    ax.plot([0, c], [yi, yi], color=col, lw=lw, solid_capstyle="round", zorder=2)
    # marker
    ax.plot(c, yi, "o", color=col, ms=ms, zorder=3,
            markeredgecolor="white", markeredgewidth=1.3)
    # count text inside/next to marker
    ax.text(c + 0.12, yi, f"{c}/3", va="center", ha="left",
            fontsize=12, fontweight="bold", color=col, zorder=4)

# Threshold line at 2/3 (majority)
ax.axvline(2, color=ACCENT, ls="--", lw=1.8, zorder=1)
ax.text(1.96, 2.5, "majority threshold (2/3)",
        color=ACCENT, fontsize=11, fontweight="bold",
        va="center", ha="right", rotation=90)

# Highlight band behind the survivor row
ax.axhspan(len(rows) - 1.45, len(rows) - 0.55, color=SURV, alpha=0.07, zorder=0)
ax.annotate("SURVIVES majority review",
            xy=(3, len(rows) - 1), xytext=(2.28, len(rows) - 1.75),
            fontsize=11.5, fontweight="bold", color=SURV,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=SURV, lw=1.8,
                            connectionstyle="arc3,rad=0.25"))

# Axes cosmetics
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=12)
# sublabels as faint right-side annotations
for yi, sl, c in zip(y, sublabels, counts):
    col = SURV if c >= 2 else "#7a7f86"
    ax.text(3.92, yi, sl, va="center", ha="right",
            fontsize=9.5, style="italic", color=col)

ax.set_xlim(0, 4.0)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xticks([0, 1, 2, 3])
ax.set_xlabel("Adversarial lenses held (out of 3:  physics / numerics / data)",
              fontsize=12)
ax.set_title("Only 1 of 7 discrepancies survives 3-lens adversarial review",
             fontsize=14, fontweight="bold", pad=12)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")
ax.tick_params(axis="both", labelsize=11)
ax.grid(axis="x", color="#e8e8e8", lw=0.8, zorder=0)
ax.set_axisbelow(True)

# Legend
legend_elems = [
    Line2D([0], [0], marker="o", color=SURV, lw=4.5, ms=14,
           markeredgecolor="white", label="Survives (≥ 2/3 lenses held)"),
    Line2D([0], [0], marker="o", color=FAIL, lw=3.0, ms=11,
           markeredgecolor="white", label="Refuted / contested (< 2/3)"),
    Line2D([0], [0], color=ACCENT, ls="--", lw=1.8, label="Majority threshold"),
]
leg = ax.legend(handles=legend_elems, loc="lower center",
                bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=10.5,
                frameon=True, framealpha=0.95, edgecolor="#cccccc")

fig.tight_layout()
fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
print("WROTE", OUT, os.path.getsize(OUT), "bytes")
