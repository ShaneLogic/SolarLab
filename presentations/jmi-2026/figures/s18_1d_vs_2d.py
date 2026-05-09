"""s18 hero — 1D vs 2D J-V."""
import json
import matplotlib.pyplot as plt
from figures._common import configure, FIG_OUT, FIG_DATA, ACCENT, INK

def main():
    configure()
    data = json.loads((FIG_DATA / "r5_jv_curves.json").read_text())
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(data["v_1d"], [j / 10 for j in data["j_1d"]],   # A/m² → mA/cm²
            color=ACCENT, lw=2.0, label=f"1D · PCE {data['metrics_1d']['pce_pct']:.1f}%")
    ax.plot(data["v_2d"], [j / 10 for j in data["j_2d"]],
            color=INK, lw=2.0, label=f"2D · PCE {data['metrics_2d']['pce_pct']:.1f}%")
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("J (mA/cm²)")
    delta = data['metrics_1d']['pce_pct'] - data['metrics_2d']['pce_pct']
    ax.set_title(f"ΔPCE = {delta:+.2f} % (grain-boundary recombination)",
                 fontsize=11, color="#666", style="italic", loc="left")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "s18_1d_vs_2d.png")
    print(f"wrote {FIG_OUT / 's18_1d_vs_2d.png'}")

if __name__ == "__main__":
    main()
