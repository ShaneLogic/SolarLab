"""s13 — GP + RF uncertainty stack: mean and confidence band on a 1-D toy."""
import numpy as np
import matplotlib.pyplot as plt
from figures._common import configure, FIG_OUT, INK, BODY, ACCENT, SECONDARY, HAIRLINE


def main():
    configure()
    fig = plt.figure(figsize=(7.6, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.30)
    ax = fig.add_subplot(gs[0])
    ax_k = fig.add_subplot(gs[1])

    rng = np.random.default_rng(7)

    # Toy training points — sparse; the GP underconfidence regions sit between them.
    x_train = np.array([0.10, 0.18, 0.25, 0.55, 0.62, 0.72, 0.95])
    y_train = np.array([0.12, 0.22, 0.18, 0.42, 0.45, 0.40, 0.30])

    x = np.linspace(0.0, 1.0, 400)
    # GP-like posterior mean — interpolating spline through the training points.
    mu_gp = np.interp(x, x_train, y_train)
    # GP-like posterior std — small near training points, larger between them.
    dist = np.array([np.min(np.abs(xi - x_train)) for xi in x])
    sigma_gp = 0.05 + 0.45 * (1 - np.exp(-(dist / 0.10) ** 2))

    # RF predictor — slightly smoother, less confident in interpolation.
    mu_rf = mu_gp + 0.02 * np.sin(8 * x)
    sigma_rf = 0.10 + 0.25 * (1 - np.exp(-(dist / 0.20) ** 2))

    # Stack — non-negative ridge mixing weights (toy values).
    w_gp, w_rf = 0.65, 0.35
    mu_stack = w_gp * mu_gp + w_rf * mu_rf
    sigma_stack = np.sqrt((w_gp * sigma_gp) ** 2 + (w_rf * sigma_rf) ** 2)

    ax.fill_between(x, mu_stack - sigma_stack, mu_stack + sigma_stack,
                    color=SECONDARY, alpha=0.18, lw=0,
                    label=r"$\mu \pm \sigma$ — stack")
    ax.plot(x, mu_stack, color=SECONDARY, lw=1.8,
            label=r"$\mu_{stack}$")
    ax.plot(x, mu_gp, color=ACCENT, lw=1.0, ls="--",
            label=r"$\mu_{GPR}$")
    ax.plot(x, mu_rf, color="#1aa37a", lw=1.0, ls=":",
            label=r"$\mu_{RF}$")
    ax.scatter(x_train, y_train, s=22, color=INK,
               zorder=5, label="training")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("descriptor x (a.u.)")
    ax.set_ylabel(r"PV score $y$")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5,
              ncol=2, columnspacing=0.6)
    ax.set_title("Stack mean and σ — calibrated by non-negative ridge",
                 loc="left", fontsize=10, color=INK, pad=2)

    # Right panel — the Matérn-5/2 kernel shape.
    r = np.linspace(0, 4, 400)
    k = (1 + np.sqrt(5) * r + (5 / 3) * r ** 2) * np.exp(-np.sqrt(5) * r)
    ax_k.plot(r, k, color=ACCENT, lw=1.8)
    ax_k.fill_between(r, 0, k, color=ACCENT, alpha=0.15, lw=0)
    ax_k.set_xlabel(r"$r = \|x - x'\|/\ell$")
    ax_k.set_ylabel(r"$k(r)/\sigma_f^{2}$")
    ax_k.set_xlim(0, 4)
    ax_k.set_ylim(0, 1.05)
    ax_k.set_title("Matérn-5/2 kernel",
                   loc="left", fontsize=10, color=INK, pad=2)

    fig.tight_layout()
    fig.savefig(FIG_OUT / "s13_gp_rf_uncertainty.png")
    print(f"wrote {FIG_OUT / 's13_gp_rf_uncertainty.png'}")


if __name__ == "__main__":
    main()
