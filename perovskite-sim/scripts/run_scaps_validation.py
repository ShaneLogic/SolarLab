#!/usr/bin/env python
"""SCAPS-mirror validation runner.

Drives the SCAPS parameter sweeps documented in the partner 1D-SCAPS PDF
(``1D-SCAPS 模拟.pdf`` + ``Parameters(1).xlsx``) through SolarLab via
``perovskite_sim.scaps_compat.load_scaps_yaml`` + the existing
``device_parameter_sweep.apply_sweep_point`` helper, then emits CSV +
PNG overlays and a Markdown report comparing SolarLab vs SCAPS metrics
sweep-by-sweep.

The script keeps the SCAPS reference data inline (small enough that an
external CSV would just add friction) and uses matplotlib for the
plots. One JV per point at ``N_grid=30, n_points=20, v_rate=5.0,
V_max=1.6`` -- the same envelope as the integration tests.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import math
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perovskite_sim.experiments.jv_sweep import compute_metrics, run_jv_sweep
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.sweeps.device_parameter_sweep import (
    SweepPoint,
    apply_sweep_point,
)


# SCAPS reference data extracted from the partner PDF (page numbers
# noted alongside) and the Parameters(1).xlsx sheets.
SCAPS_BASE = {
    "V_oc_V": 1.1676,
    "J_sc_mA_cm2": 26.2820,
    "FF_percent": 86.99,
    "PCE_percent": 26.69,
}

SCAPS_REF: dict[str, dict[float, dict[str, float]]] = {
    "cbo_delta_ec_eV": {  # PDF p3
        -1.00: {"PCE": 5.8685, "Voc": 0.332353, "FF": 67.1265, "Jsc": 26.30454},
        -0.75: {"PCE": 11.8314, "Voc": 0.580953, "FF": 77.4679, "Jsc": 26.28894},
        -0.50: {"PCE": 18.0802, "Voc": 0.830850, "FF": 82.7910, "Jsc": 26.28441},
        -0.25: {"PCE": 24.4188, "Voc": 1.080493, "FF": 85.9878, "Jsc": 26.28239},
        -0.16: {"PCE": 26.6926, "Voc": 1.167550, "FF": 86.9876, "Jsc": 26.28199},
        -0.10: {"PCE": 28.1546, "Voc": 1.217203, "FF": 88.0108, "Jsc": 26.28154},
        0.00: {"PCE": 29.6512, "Voc": 1.247696, "FF": 90.4240, "Jsc": 26.28148},
        0.10: {"PCE": 29.8159, "Voc": 1.250391, "FF": 90.7313, "Jsc": 26.28116},
        0.20: {"PCE": 29.8035, "Voc": 1.250158, "FF": 90.7096, "Jsc": 26.28141},
        0.30: {"PCE": 29.7103, "Voc": 1.249937, "FF": 90.4411, "Jsc": 26.28168},
    },
    "etl_doping_cm3": {  # PDF p5
        1e10: {"PCE": 17.5945, "Voc": 1.10020, "FF": 60.8285, "Jsc": 26.2905},
        1e12: {"PCE": 20.4893, "Voc": 1.13245, "FF": 68.8235, "Jsc": 26.2889},
        1e14: {"PCE": 23.2512, "Voc": 1.14125, "FF": 77.5039, "Jsc": 26.2871},
        1e16: {"PCE": 24.7579, "Voc": 1.14637, "FF": 82.1615, "Jsc": 26.2857},
        1e18: {"PCE": 26.6926, "Voc": 1.16755, "FF": 86.9876, "Jsc": 26.2820},
        1e20: {"PCE": 29.2855, "Voc": 1.23726, "FF": 90.0630, "Jsc": 26.2812},
    },
    "absorber_doping_cm3": {  # PDF p6 -- PVK donor (n-type)
        1e8: {"PCE": 26.6926, "Voc": 1.16755, "FF": 86.9875, "Jsc": 26.2820},
        1e12: {"PCE": 26.6926, "Voc": 1.16755, "FF": 86.9876, "Jsc": 26.2820},
        1e14: {"PCE": 26.6957, "Voc": 1.16758, "FF": 86.9960, "Jsc": 26.2818},
        1e16: {"PCE": 26.8630, "Voc": 1.17073, "FF": 87.3175, "Jsc": 26.2783},
        1e18: {"PCE": 16.5663, "Voc": 1.20157, "FF": 65.8337, "Jsc": 20.9425},
    },
    "absorber_defect_density_cm3": {  # PDF p9 -- PVK-CB bulk N_t
        1e9: {"PCE": 26.6934, "Voc": 1.16765, "FF": 86.9829, "Jsc": 26.2820},
        1e11: {"PCE": 26.6934, "Voc": 1.16764, "FF": 86.9834, "Jsc": 26.2820},
        1e12: {"PCE": 26.6926, "Voc": 1.16755, "FF": 86.9876, "Jsc": 26.2820},
        1e13: {"PCE": 26.6853, "Voc": 1.16668, "FF": 87.0283, "Jsc": 26.2820},
        1e14: {"PCE": 26.6172, "Voc": 1.15956, "FF": 87.3396, "Jsc": 26.2819},
        1e15: {"PCE": 26.1403, "Voc": 1.12908, "FF": 88.0927, "Jsc": 26.2814},
    },
    "interface_trap_density_cm3": {  # PDF p12 -- PVK/ETL interface N_t
        1e9: {"PCE": 29.8175, "Voc": 1.24947, "FF": 90.7999, "Jsc": 26.2820},
        1e11: {"PCE": 28.4621, "Voc": 1.23572, "FF": 87.6373, "Jsc": 26.2820},
        1e12: {"PCE": 26.6926, "Voc": 1.16755, "FF": 86.9876, "Jsc": 26.2820},
        1e13: {"PCE": 25.0476, "Voc": 1.08851, "FF": 87.5539, "Jsc": 26.2820},
        1e14: {"PCE": 23.4575, "Voc": 1.02281, "FF": 87.2634, "Jsc": 26.2818},
        1e15: {"PCE": 21.9278, "Voc": 0.96776, "FF": 86.2206, "Jsc": 26.2796},
    },
    "absorber_defect_depth_eV": {  # PDF p15 -- PVK-CB E_t
        0.01: {"PCE": 26.6934, "Voc": 1.16765, "FF": 86.9831, "Jsc": 26.2820},
        0.10: {"PCE": 26.6926, "Voc": 1.16755, "FF": 86.9876, "Jsc": 26.2820},
        0.20: {"PCE": 26.6814, "Voc": 1.16725, "FF": 86.9736, "Jsc": 26.2820},
        0.30: {"PCE": 26.6754, "Voc": 1.16722, "FF": 86.9563, "Jsc": 26.2820},
        0.50: {"PCE": 26.6752, "Voc": 1.16722, "FF": 86.9558, "Jsc": 26.2820},
    },
}


SWEEP_LABELS = {
    "cbo_delta_ec_eV": ("ETL/PVK conduction band offset", "DeltaE_C (eV)", False),
    "etl_doping_cm3": ("ETL donor doping", "N_D,ETL (cm^-3)", True),
    "absorber_doping_cm3": ("PVK donor doping", "N_D,PVK (cm^-3)", True),
    "absorber_defect_density_cm3": ("PVK-CB bulk defect density", "N_t (cm^-3)", True),
    "interface_trap_density_cm3": ("PVK/ETL interface defect density", "N_t (cm^-3)", True),
    "absorber_defect_depth_eV": ("PVK-CB bulk defect energy", "E_t below CB (eV)", False),
}


# SCAPS_REF axis keys to apply_sweep_point.updates keys (mostly the
# same; CBO uses an "etl_" prefix in the sweep helper).
_AXIS_TO_UPDATE_KEY = {
    "cbo_delta_ec_eV": "etl_delta_ec_eV",
    "etl_doping_cm3": "etl_doping_cm3",
    "absorber_doping_cm3": "absorber_doping_cm3",
    "absorber_defect_density_cm3": "absorber_defect_density_cm3",
    "interface_trap_density_cm3": "interface_trap_density_cm3",
    "absorber_defect_depth_eV": "absorber_defect_depth_eV",
}


def _build_sweep_point(axis: str, x: float) -> SweepPoint:
    update_key = _AXIS_TO_UPDATE_KEY[axis]
    updates = {update_key: x}
    if axis == "absorber_doping_cm3":
        # SCAPS PVK is donor-doped (N_D = 1e14 at base); pass through accordingly.
        updates["absorber_doping_type"] = "donor"
    return SweepPoint("p", axis, f"{x:.3e}", updates)


def run_axis(
    base_path: Path, axis: str, points: Iterable[float], jv_kwargs: dict,
) -> list[dict[str, float]]:
    """Run a 1-D sweep and return per-point metric records."""
    base_stack = load_scaps_yaml(base_path)
    records = []
    for x in points:
        pt = _build_sweep_point(axis, x)
        try:
            stack = apply_sweep_point(base_stack, pt)
            result = run_jv_sweep(stack, **jv_kwargs)
            m = result.metrics_fwd
            records.append(
                {
                    "x": x,
                    "V_oc_V": m.V_oc,
                    "J_sc_A_m2": m.J_sc,
                    "FF": m.FF,
                    "PCE": m.PCE,
                    "voc_bracketed": int(m.voc_bracketed),
                    "status": "ok",
                }
            )
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    "x": x, "V_oc_V": math.nan, "J_sc_A_m2": math.nan,
                    "FF": math.nan, "PCE": math.nan, "voc_bracketed": 0,
                    "status": f"fail: {type(exc).__name__}",
                }
            )
    return records


def write_csv(path: Path, records: list[dict[str, float]], axis_label: str) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["x", "V_oc_V", "J_sc_A_m2", "FF", "PCE", "voc_bracketed", "status"],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def write_plot(
    path: Path, axis: str, records: list[dict[str, float]],
    scaps_ref: dict[float, dict[str, float]],
) -> None:
    title, xlabel, log_x = SWEEP_LABELS[axis]
    metrics = [
        ("PCE", "PCE (%)", lambda r: r["PCE"] * 100.0, "PCE"),
        ("Voc", "V_oc (V)", lambda r: r["V_oc_V"], "Voc"),
        ("FF", "FF (%)", lambda r: r["FF"] * 100.0, "FF"),
        ("Jsc", "J_sc (mA/cm^2)", lambda r: r["J_sc_A_m2"] / 10.0, "Jsc"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    xs_solar = [r["x"] for r in records if r["status"] == "ok"]
    for ax, (key, ylabel, fn, scaps_key) in zip(axes, metrics):
        ys_solar = [fn(r) for r in records if r["status"] == "ok"]
        xs_scaps = sorted(scaps_ref.keys())
        ys_scaps = [scaps_ref[x][scaps_key] for x in xs_scaps]
        if log_x:
            ax.set_xscale("log")
        ax.plot(xs_solar, ys_solar, "o-", label="SolarLab", color="C0")
        ax.plot(xs_scaps, ys_scaps, "s--", label="SCAPS", color="C3")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_base_jv(base_path: Path, jv_kwargs: dict) -> dict[str, float]:
    stack = load_scaps_yaml(base_path)
    result = run_jv_sweep(stack, **jv_kwargs)
    m = result.metrics_fwd
    return {
        "V_oc_V": m.V_oc,
        "J_sc_A_m2": m.J_sc,
        "FF": m.FF,
        "PCE": m.PCE,
        "voc_bracketed": int(m.voc_bracketed),
    }


def write_report(
    out_dir: Path, base: dict[str, float], sweep_summaries: list[dict],
) -> None:
    lines = []
    lines.append("# SCAPS-mirror validation report\n")
    lines.append("Auto-generated by `scripts/run_scaps_validation.py`.\n\n")
    lines.append("Compares SolarLab (`configs/scaps_mirror.yaml` via "
                 "`scaps_compat.load_scaps_yaml`) against the partner SCAPS "
                 "reference data extracted from the 1D-SCAPS PDF + "
                 "`Parameters(1).xlsx`.\n\n")
    lines.append("## Base J-V\n\n")
    lines.append("| Metric | SolarLab | SCAPS | Δ |\n|---|---|---|---|\n")
    lines.append(
        f"| V_oc (V) | {base['V_oc_V']:.4f} | {SCAPS_BASE['V_oc_V']:.4f} | "
        f"{(base['V_oc_V'] - SCAPS_BASE['V_oc_V']) * 1000:+.0f} mV |\n"
    )
    lines.append(
        f"| J_sc (mA/cm^2) | {base['J_sc_A_m2'] / 10:.2f} | "
        f"{SCAPS_BASE['J_sc_mA_cm2']:.2f} | "
        f"{(base['J_sc_A_m2'] / 10 - SCAPS_BASE['J_sc_mA_cm2']):+.2f} |\n"
    )
    lines.append(
        f"| FF (%) | {base['FF'] * 100:.2f} | {SCAPS_BASE['FF_percent']:.2f} | "
        f"{(base['FF'] * 100 - SCAPS_BASE['FF_percent']):+.2f} |\n"
    )
    lines.append(
        f"| PCE (%) | {base['PCE'] * 100:.2f} | {SCAPS_BASE['PCE_percent']:.2f} | "
        f"{(base['PCE'] * 100 - SCAPS_BASE['PCE_percent']):+.2f} |\n\n"
    )
    lines.append("## Per-sweep summaries\n\n")
    lines.append(
        "| Sweep | SCAPS V_oc range | SolarLab V_oc range | "
        "SCAPS PCE range | SolarLab PCE range | Direction |\n"
        "|---|---|---|---|---|---|\n"
    )
    for s in sweep_summaries:
        lines.append(
            f"| {s['title']} | "
            f"{s['scaps_voc_range_mV']:.0f} mV | "
            f"{s['solarlab_voc_range_mV']:.0f} mV | "
            f"{s['scaps_pce_range_pct']:.1f} pp | "
            f"{s['solarlab_pce_range_pct']:.1f} pp | "
            f"{s['direction']} |\n"
        )
    lines.append("\n## Sweep plots\n\n")
    for s in sweep_summaries:
        lines.append(f"### {s['title']}\n\n![{s['title']}]({s['plot_filename']})\n\n")
    (out_dir / "report.md").write_text("".join(lines))


def _direction_label(records, scaps_ref) -> str:
    """Compare sign of metric trend across the sweep -- match / mismatch / flat."""
    xs = sorted(scaps_ref.keys())
    if len(xs) < 2:
        return "n/a"
    sc_first, sc_last = scaps_ref[xs[0]]["Voc"], scaps_ref[xs[-1]]["Voc"]
    by_x = {r["x"]: r for r in records if r["status"] == "ok"}
    if xs[0] not in by_x or xs[-1] not in by_x:
        # Map nearest SolarLab x.
        sl_xs = sorted(by_x)
        sl_first, sl_last = by_x[sl_xs[0]]["V_oc_V"], by_x[sl_xs[-1]]["V_oc_V"]
    else:
        sl_first, sl_last = by_x[xs[0]]["V_oc_V"], by_x[xs[-1]]["V_oc_V"]
    d_sc = sc_last - sc_first
    d_sl = sl_last - sl_first
    if abs(d_sc) < 0.01:
        return "flat both" if abs(d_sl) < 0.01 else "SCAPS flat, SolarLab non-flat"
    return "match" if (d_sc * d_sl > 0) else "mismatch"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/scaps_mirror.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--N-grid", type=int, default=30)
    parser.add_argument("--n-points", type=int, default=20)
    parser.add_argument("--v-rate", type=float, default=5.0)
    parser.add_argument("--V-max", type=float, default=1.6)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jv_kwargs = dict(
        N_grid=args.N_grid, n_points=args.n_points,
        v_rate=args.v_rate, V_max=args.V_max,
    )

    print("== base J-V ==", flush=True)
    base = run_base_jv(args.config, jv_kwargs)
    print(
        f"   V_oc={base['V_oc_V']:.4f} V  "
        f"J_sc={base['J_sc_A_m2']/10:.2f} mA/cm^2  "
        f"FF={base['FF']*100:.2f}%  PCE={base['PCE']*100:.2f}%",
        flush=True,
    )

    sweep_summaries = []
    for axis, scaps_ref in SCAPS_REF.items():
        title = SWEEP_LABELS[axis][0]
        print(f"== sweep: {title} ==", flush=True)
        records = run_axis(args.config, axis, sorted(scaps_ref.keys()), jv_kwargs)
        csv_path = args.out_dir / f"sweep_{axis}.csv"
        plot_path = args.out_dir / f"sweep_{axis}.png"
        write_csv(csv_path, records, axis)
        write_plot(plot_path, axis, records, scaps_ref)
        ok_records = [r for r in records if r["status"] == "ok"]
        if ok_records:
            voc_range = (max(r["V_oc_V"] for r in ok_records)
                         - min(r["V_oc_V"] for r in ok_records))
            pce_range = (max(r["PCE"] for r in ok_records)
                         - min(r["PCE"] for r in ok_records))
        else:
            voc_range = pce_range = math.nan
        scaps_voc = [scaps_ref[x]["Voc"] for x in scaps_ref]
        scaps_pce = [scaps_ref[x]["PCE"] for x in scaps_ref]
        sweep_summaries.append(
            {
                "title": title,
                "axis": axis,
                "plot_filename": plot_path.name,
                "scaps_voc_range_mV": (max(scaps_voc) - min(scaps_voc)) * 1000,
                "solarlab_voc_range_mV": voc_range * 1000,
                "scaps_pce_range_pct": max(scaps_pce) - min(scaps_pce),
                "solarlab_pce_range_pct": pce_range * 100,
                "direction": _direction_label(records, scaps_ref),
            }
        )
        print(
            f"   SolarLab V_oc range: {voc_range*1000:.0f} mV, "
            f"SCAPS V_oc range: {(max(scaps_voc) - min(scaps_voc))*1000:.0f} mV, "
            f"direction: {sweep_summaries[-1]['direction']}",
            flush=True,
        )

    write_report(args.out_dir, base, sweep_summaries)
    print(f"== done == report: {args.out_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
