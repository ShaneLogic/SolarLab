"""s14 — UCB acquisition function with the next-batch picks highlighted."""
import numpy as np
import matplotlib.pyplot as plt
from figures._common import configure, FIG_OUT, INK, BODY, ACCENT, SECONDARY


def main():
    configure()
    fig, ax = plt.subplots(figsize=(7.6, 4.0))

    rng = np.random.default_rng(11)
    x = np.linspace(0, 1.0, 500)
    x_train = np.array([0.10, 0.18, 0.25, 0.55, 0.62, 0.72, 0.95])
    y_train = np.array([0.12, 0.22, 0.18, 0.42, 0.45, 0.40, 0.30])

    # Posterior mean — taken from the HistGBR predictor, monotone interpolation.
    mu = np.interp(x, x_train, y_train)
    # Posterior std — taken from the GPR + RF stack; growing between training points.
    dist = np.array([np.min(np.abs(xi - x_train)) for xi in x])
    sigma = 0.05 + 0.30 * (1 - np.exp(-(dist / 0.13) ** 2))

    kappa = 2.0
    ucb = mu + kappa * sigma

    ax.fill_between(x, mu - sigma, mu + sigma,
                    color=SECONDARY, alpha=0.15, lw=0,
                    label=r"$\mu \pm \sigma$")
    ax.plot(x, mu, color=SECONDARY, lw=1.8, label=r"$\mu(x)$ — HistGBR")
    ax.plot(x, ucb, color=ACCENT, lw=1.8, ls="--",
            label=r"$\alpha_{UCB} = \mu + \kappa\sigma$ ($\kappa = 2$)")
    ax.scatter(x_train, y_train, s=22, color=INK, zorder=5,
               label="labelled")

    # Top-3 acquisition picks for the next active-learning batch.
    top_idx = np.argsort(ucb)[-3:]
    ax.scatter(x[top_idx], ucb[top_idx],
               s=110, marker="*", color=ACCENT, edgecolor=INK,
               linewidth=0.8, zorder=6, label="top-3 picks")
    for idx in top_idx:
        ax.annotate("", xy=(x[idx], ucb[idx] + 0.02),
                    xytext=(x[idx], ucb[idx] + 0.18),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=ACCENT, lw=1.2))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.10)
    ax.set_xlabel("descriptor x (a.u.)")
    ax.set_ylabel("acquisition / score")
    ax.legend(loc="lower center", frameon=False, fontsize=8.5,
              ncol=4, bbox_to_anchor=(0.5, -0.30))
    ax.set_title(r"Top-N candidates by $\alpha_{UCB}$ "
                 "feed the next DFT batch",
                 loc="left", fontsize=10, color=INK, pad=2,
                 style="italic")

    fig.tight_layout()
    fig.savefig(FIG_OUT / "s14_ucb_acquisition.png")
    print(f"wrote {FIG_OUT / 's14_ucb_acquisition.png'}")


if __name__ == "__main__":
    main()
