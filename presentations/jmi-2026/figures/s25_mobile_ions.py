"""s25 — mobile-ion redistribution under bias and the resulting field screening."""
import numpy as np
import matplotlib.pyplot as plt
from figures._common import configure, FIG_OUT, INK, BODY, ACCENT, SECONDARY, HAIRLINE


def main():
    configure()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), sharey=False)
    ax_n, ax_b = axes

    x = np.linspace(0, 1.0, 400)

    # Equilibrium ion-vacancy density profile P(x) — uniform across absorber.
    P_eq = np.where((x >= 0.05) & (x <= 0.95), 1.0, 0.0)

    # Under forward bias — ions drift, accumulate at one contact (steric cap).
    P_bias = np.where(
        (x >= 0.05) & (x <= 0.95),
        0.30 + 1.65 * np.exp(-((x - 0.92) / 0.15) ** 2),
        0.0,
    )
    P_bias = np.clip(P_bias, 0, 1.95)   # steric P_lim cap

    for ax, title, mode in [(ax_n, "equilibrium", "eq"),
                            (ax_b, "under forward bias", "bias")]:
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 2.2)
        ax.set_xticks([])
        ax.set_xlabel("position x")
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")

        # Layer separators (HTL / absorber / ETL).
        for xb in (0.05, 0.95):
            ax.axvline(xb, color=HAIRLINE, lw=0.6, ls="--")
        ax.text(0.025, 2.0, "HTL", ha="center", color=BODY, fontsize=8.5)
        ax.text(0.50, 2.0, "absorber", ha="center", color=BODY, fontsize=9)
        ax.text(0.975, 2.0, "ETL", ha="center", color=BODY, fontsize=8.5)

    ax_n.plot(x, P_eq, color=SECONDARY, lw=1.8,
              label=r"$P(x)$ — vacancy density")
    ax_n.fill_between(x, 0, P_eq, color=SECONDARY, alpha=0.18, lw=0)
    ax_n.set_ylabel(r"normalised $P / P_{\mathrm{lim}}$")
    ax_n.legend(loc="lower right", frameon=False, fontsize=8.5)

    ax_b.plot(x, P_bias, color=ACCENT, lw=1.8,
              label=r"$P(x)$ after drift")
    ax_b.fill_between(x, 0, P_bias, color=ACCENT, alpha=0.18, lw=0)

    # Steric-cap reference line.
    ax_b.axhline(1.95, color=BODY, lw=0.7, ls=":")
    ax_b.text(0.05, 1.97, r"steric cap $P_{\mathrm{lim}}$",
              color=BODY, fontsize=8.5, va="bottom")

    # Bias arrow indicating the direction of ion drift.
    ax_b.annotate("",
                  xy=(0.85, 0.85), xytext=(0.20, 0.85),
                  arrowprops=dict(arrowstyle="-|>",
                                  color=ACCENT, lw=1.4))
    ax_b.text(0.50, 0.95, r"$\mu_{\mathrm{ion}}\,P\,\nabla\varphi$",
              ha="center", va="bottom",
              color=ACCENT, fontsize=9.5)

    ax_b.legend(loc="upper left", frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIG_OUT / "s25_mobile_ions.png")
    print(f"wrote {FIG_OUT / 's25_mobile_ions.png'}")


if __name__ == "__main__":
    main()
