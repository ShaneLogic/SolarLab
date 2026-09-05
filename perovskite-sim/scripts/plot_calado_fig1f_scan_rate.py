#!/usr/bin/env python3
"""Scan-rate ladder on the Calado 2016 Fig 1f preset: the hysteresis bell.

Runs the hysteretic stack of ``configs/calado2016_fig1f.yaml`` through the
protocol of ``scripts/plot_calado_fig1f.py`` at every rate in ``--rates``.
The +1.2 V dwell is removed by default (``--hold 0``): with a hold, the
reverse branch always starts from the +1.2 V-polarised ion state and the
loop never closes at the frozen-ion limit. Without it both branches share
one ion history when ions cannot follow, so the loop closes at both ends —
the scan-rate bell of Tress 2015 / Calado 2016. The paper-protocol point
(40 mV/s with the 3 s hold, from ``Calado16Fig1fMetrics<STAMP>.json``) is
overlaid as a separate marker.

Outputs (SolarLab root ``docs/manual/figures/``, suffix ``STAMP``):

    Calado16Fig1fScanRate<STAMP>.png        (a) J-V loops at three rates,
                                            (b) HI (paper definition) vs rate
    Calado16Fig1fScanRate<STAMP>.json       per-rate branch metrics and HI
    Calado16Fig1fScanRate<STAMP>/rate_<r>.npz   per-rate curves (resume, --replot)

Rates run in parallel processes (``--jobs``); a rate whose ``.npz`` exists is
skipped, so an interrupted ladder resumes where it stopped.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import dataclasses
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_calado_fig1f as base  # noqa: E402

STAMP = base.STAMP
OUT_DIR = base.OUT_DIR
DEFAULT_RATES = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 10.0, 100.0)
DEFAULT_HOLD_S = 0.0
# Sequential single-hue ramp for the ordered (slow -> fast) rates in panel (a).
RATE_COLOURS = ("#6baed6", "#2171b5", "#08306b")


def rate_tag(rate: float) -> str:
    return f"{rate:g}".replace(".", "p").replace("-", "m")


def select_display_rates(hi_by_rate: dict[float, float]) -> tuple[float, ...]:
    """Slowest, peak-HI, fastest rate — deduplicated, ascending."""
    rates = sorted(hi_by_rate)
    peak = max(rates, key=lambda r: hi_by_rate[r])
    return tuple(sorted({rates[0], peak, rates[-1]}))


def _run_one(rate: float, n_grid: int, hold_s: float, npz_path: str) -> float:
    stack = base.load_device_from_yaml(str(base.CONFIG))
    t0 = time.time()
    result = base.run_protocol(
        stack, n_grid, log=lambda _msg: None, scan_rate=rate, hold_s=hold_s
    )
    np.savez(
        npz_path,
        V_fwd=result.V_fwd, J_fwd=result.J_fwd, V_rev=result.V_rev, J_rev=result.J_rev,
        failures=np.array([f"{V:.3f}: {msg}" for V, msg in result.failures]),
    )
    return time.time() - t0


def _load(npz_path: Path) -> base.ProtocolResult:
    d = np.load(npz_path)
    failures = tuple(
        (float(item.split(":", 1)[0]), item.split(":", 1)[1].strip())
        for item in d["failures"].tolist()
    )
    return base.ProtocolResult(d["V_fwd"], d["J_fwd"], d["V_rev"], d["J_rev"], failures)


def run_ladder(rates, n_grid, hold_s, jobs, curve_dir: Path) -> None:
    curve_dir.mkdir(parents=True, exist_ok=True)
    pending = {
        rate: curve_dir / f"rate_{rate_tag(rate)}.npz"
        for rate in rates
        if not (curve_dir / f"rate_{rate_tag(rate)}.npz").exists()
    }
    print(f"{len(rates) - len(pending)} rate(s) cached, {len(pending)} to run "
          f"with {jobs} worker(s)", flush=True)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(_run_one, rate, n_grid, hold_s, str(path)): rate
            for rate, path in pending.items()
        }
        for fut in as_completed(futures):
            rate = futures[fut]
            elapsed = fut.result()   # re-raises a worker crash
            print(f"  rate {rate:g} V/s done [{elapsed:.0f} s]", flush=True)


def paper_protocol_point() -> dict | None:
    path = OUT_DIR / f"Calado16Fig1fMetrics{STAMP}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "scan_rate_V_s": payload["protocol"]["scan_rate_V_s"],
        "hold_s": payload["protocol"]["hold_s"],
        "HI_paper": payload["cases"]["hysteretic"]["HI_paper"],
    }


def plot(curves: dict[float, base.ProtocolResult], metrics: dict[float, dict],
         paper_point: dict | None, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hi = {rate: m["HI_paper"] for rate, m in metrics.items()}
    shown = select_display_rates(hi)
    fig, (ax, ab) = plt.subplots(1, 2, figsize=(8.5, 5.0), dpi=200,
                                 gridspec_kw={"width_ratios": [1.15, 1.0]})
    for rate, colour in zip(shown, RATE_COLOURS):
        c = curves[rate]
        ax.plot(c.V_fwd, c.J_fwd / 10.0, color=colour, lw=2.0, label=f"{rate:g} V/s, forward")
        ax.plot(c.V_rev, c.J_rev / 10.0, color=colour, lw=2.0, ls="--", label=f"{rate:g} V/s, reverse")
    ax.axhline(0.0, color="0.55", lw=0.8)
    ax.axvline(0.0, color="0.55", lw=0.8)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-5.0, 22.0)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current density (mA cm$^{-2}$)")
    ax.set_title("(a)  J–V at three scan rates", fontsize=10.5)
    ax.grid(color="0.92", lw=0.6)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)

    rates = sorted(hi)
    ab.plot(rates, [hi[r] for r in rates], color="#2171b5", lw=2.0, marker="o", ms=6,
            label="SolarLab, no hold")
    ab.axhline(base.PAPER["HI_paper_fig1f"], color="#c0392b", lw=1.2, ls=":",
               label=f"paper Fig 1f, {base.PAPER['HI_paper_fig1f']:.2f} at 40 mV/s")
    if paper_point is not None:
        ab.plot([paper_point["scan_rate_V_s"]], [paper_point["HI_paper"]], marker="D", ms=8,
                mfc="white", mec="#c0392b", mew=2.0, ls="none",
                label=f"paper protocol, {paper_point['hold_s']:g} s hold")
    peak = max(rates, key=lambda r: hi[r])
    ab.annotate(f"peak HI = {hi[peak]:.0f} at {peak:g} V/s", (peak, hi[peak]),
                textcoords="offset points", xytext=(26, -56), fontsize=8.5, color="0.25",
                arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.8})
    ab.set_xscale("log")
    ab.set_yscale("log")
    ab.set_ylim(5e-4, 1e2)
    ab.set_xlabel("scan rate (V s$^{-1}$)")
    ab.set_ylabel("hysteresis index  $P_{max,rev}/P_{max,fwd} - 1$")
    ab.set_title("(b)  the scan-rate bell", fontsize=10.5)
    ab.grid(color="0.92", lw=0.6, which="both")
    ab.legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rates", type=float, nargs="+", default=list(DEFAULT_RATES))
    ap.add_argument("--hold", type=float, default=DEFAULT_HOLD_S,
                    help="dwell at +1.2 V before the reverse scan [s]")
    ap.add_argument("--n-grid", type=int, default=base.N_GRID)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--replot", action="store_true",
                    help="only plot from the cached per-rate .npz files")
    args = ap.parse_args()
    curve_dir = args.out_dir / f"Calado16Fig1fScanRate{STAMP}"
    out_png = args.out_dir / f"Calado16Fig1fScanRate{STAMP}.png"
    out_json = args.out_dir / f"Calado16Fig1fScanRate{STAMP}.json"

    if not args.replot:
        run_ladder(args.rates, args.n_grid, args.hold, args.jobs, curve_dir)

    curves = {rate: _load(curve_dir / f"rate_{rate_tag(rate)}.npz") for rate in args.rates}
    metrics = {rate: base.summarise(c) for rate, c in curves.items()}
    paper_point = paper_protocol_point()
    payload = {
        "config": str(base.CONFIG.relative_to(base.ROOT)),
        "n_grid": args.n_grid,
        "hold_s": args.hold,
        "protocol": "plot_calado_fig1f.run_protocol at each rate",
        "paper_protocol_point": paper_point,
        "rates": {f"{rate:g}": m for rate, m in sorted(metrics.items())},
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot(curves, metrics, paper_point, out_png)

    failed = 0
    for rate in sorted(metrics):
        m = metrics[rate]
        failed += len(m["failed_steps"])
        print(f"rate {rate:>7g} V/s  HI_paper={m['HI_paper']:.3f}  "
              f"Pmax_fwd={m['forward']['P_max']:.1f} Pmax_rev={m['reverse']['P_max']:.1f} W/m2  "
              f"Voc_fwd={m['forward']['V_oc']:.3f} Voc_rev={m['reverse']['V_oc']:.3f} V  "
              f"failed_steps={len(m['failed_steps'])}")
    print(f"wrote {out_png}\n      {out_json}\n      {curve_dir}/")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
