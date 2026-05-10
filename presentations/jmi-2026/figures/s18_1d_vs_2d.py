"""s18 hero — 1D vs 2D J-V curves with reference-stack inset."""
import json
import matplotlib.pyplot as plt
from figures._common import configure, FIG_OUT, FIG_DATA, ACCENT, INK
from figures.stack_diagram import draw_mapbi3_stack


def main():
    configure()
    data = json.loads((FIG_DATA / "r5_jv_curves.json").read_text())
    fig = plt.figure(figsize=(11.0, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.30)
    ax = fig.add_subplot(gs[0])
    ax_stack = fig.add_subplot(gs[1])

    # J-V curves — convert A/m² → mA/cm² (×0.1) and apply the standard
    # photovoltaic sign convention: photocurrent is plotted as negative so the
    # familiar fourth-quadrant operating curve sits below the V axis.
    voc_1d = data["metrics_1d"]["voc_V"]
    voc_2d = data["metrics_2d"]["voc_V"]
    jsc_1d = data["metrics_1d"]["jsc_A_per_m2"] / 10.0     # mA/cm²
    v_max = 1.05 * max(voc_1d, voc_2d)

    # Truncate the curves to the operating region — past V_oc the diode
    # current shoots up by orders of magnitude and is uninformative.
    def _trim(v_list, j_list, v_cut):
        return zip(*[(v, -j / 10.0) for v, j in zip(v_list, j_list) if v <= v_cut])
    v_1d_trim, j_1d = _trim(data["v_1d"], data["j_1d"], v_max)
    v_2d_trim, j_2d = _trim(data["v_2d"], data["j_2d"], v_max)

    ax.plot(v_1d_trim, j_1d, color=ACCENT, lw=2.2,
            label=rf"1D · $V_{{oc}}$ = {voc_1d:.3f} V")
    ax.plot(v_2d_trim, j_2d, color=INK, lw=2.2,
            label=rf"2D · $V_{{oc}}$ = {voc_2d:.3f} V")
    ax.axhline(0, color="#999", lw=0.6)
    ax.axvline(voc_1d, color=ACCENT, lw=0.6, ls=":", alpha=0.6)
    ax.axvline(voc_2d, color=INK, lw=0.6, ls=":", alpha=0.6)

    ax.set_xlim(0.0, v_max)
    ax.set_ylim(-1.20 * jsc_1d, 0.10 * jsc_1d)

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel(r"J (mA/cm$^{2}$)")
    delta_voc_mV = (voc_2d - voc_1d) * 1000.0
    ax.set_title(rf"$\Delta V_{{oc}}$ = {delta_voc_mV:+.0f} mV"
                 r" — grain-boundary recombination resolved only in 2D",
                 fontsize=10.5, color="#666", style="italic", loc="left")
    ax.legend(frameon=False, loc="lower left")

    # Reference-stack inset — vertical layer schematic w/ grain boundary.
    draw_mapbi3_stack(ax_stack, show_gb=True, title="Reference stack (2D run)")

    fig.tight_layout()
    fig.savefig(FIG_OUT / "s18_1d_vs_2d.png")
    print(f"wrote {FIG_OUT / 's18_1d_vs_2d.png'}")


if __name__ == "__main__":
    main()
