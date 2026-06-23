"""Reproduce the two SCAPS 2D scans on the Perovskite/ETL interface in SolarLab,
on the validated parity config (scaps_mirror_v2: DOS band potentials + de-spike
0.53), via the transient J-V driver (robust through the cliff/spike where the SS
Newton diverges — matches SCAPS's quasi-steady-state per-bias result).

Scan 1: interface Nt (1e9..1e15) x interface Et (0.01..0.6 eV)  -> 7 x 8 = 56
Scan 2: interface Nt (1e9..1e15) x ETL/PVK dEc (-1.0..0.70 eV)  -> 7 x 16 = 112

FoM recorded per point: PCE, Voc, FF, Jsc (NaN where no V_oc crossing — SCAPS's
white cells). Results cached to JSON; 4-panel heatmaps per scan match the SCAPS
figure layout (log10 Nt on y, the second axis on x, viridis).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, time, traceback
from pathlib import Path
from multiprocessing import Pool
import numpy as np

REPO = Path("/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab/perovskite-sim")
OUT = REPO.parent / "outputs" / "scan2d"
OUT.mkdir(parents=True, exist_ok=True)

NT = [1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]                                   # cm^-2 (SCAPS labels cm^-3)
ET = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]                                  # eV, 8
DEC = [-1.0, -0.75, -0.5, -0.25, -0.16, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # eV, 16


def _fom(updates):
    """Run one J-V on scaps_mirror_v2 + updates; return (PCE%, Voc, FF%, Jsc mA/cm2) or NaNs."""
    from perovskite_sim.scaps_compat import load_scaps_yaml
    from perovskite_sim.experiments.jv_sweep import run_jv_sweep
    from perovskite_sim.sweeps.device_parameter_sweep import SweepPoint, apply_sweep_point
    base = load_scaps_yaml(REPO / "configs" / "scaps_mirror_v2.yaml")
    try:
        sw = apply_sweep_point(base, SweepPoint("p", "pair", "x", updates))
        m = run_jv_sweep(sw, N_grid=30, n_points=40, v_rate=5.0, V_max=1.6).metrics_fwd
        if not m.voc_bracketed:
            return (float("nan"), float("nan"), float("nan"), float(m.J_sc) / 10.0)
        return (float(m.PCE) * 100, float(m.V_oc), float(m.FF) * 100, float(m.J_sc) / 10.0)
    except Exception:
        return (float("nan"), float("nan"), float("nan"), float("nan"))


def _work(task):
    scan, i, j, updates = task
    t0 = time.time()
    pce, voc, ff, jsc = _fom(updates)
    return (scan, i, j, pce, voc, ff, jsc, time.time() - t0)


def build_tasks():
    tasks = []
    for i, nt in enumerate(NT):
        for j, et in enumerate(ET):
            tasks.append(("ntet", i, j, {"interface_defect_N_t_cm2": nt, "interface_defect_E_t_eV": et}))
    for i, nt in enumerate(NT):
        for j, dec in enumerate(DEC):
            tasks.append(("ntcbo", i, j, {"interface_defect_N_t_cm2": nt, "etl_delta_ec_eV": dec}))
    return tasks


if __name__ == "__main__":
    tasks = build_tasks()
    grids = {
        "ntet": {k: np.full((len(NT), len(ET)), np.nan) for k in ("PCE", "Voc", "FF", "Jsc")},
        "ntcbo": {k: np.full((len(NT), len(DEC)), np.nan) for k in ("PCE", "Voc", "FF", "Jsc")},
    }
    print(f"[{time.strftime('%H:%M:%S')}] {len(tasks)} points (56 Nt-Et + 112 Nt-CBO), Pool(8)", flush=True)
    t0 = time.time(); done = 0
    with Pool(8) as pool:
        for scan, i, j, pce, voc, ff, jsc, dt in pool.imap_unordered(_work, tasks):
            grids[scan]["PCE"][i, j] = pce; grids[scan]["Voc"][i, j] = voc
            grids[scan]["FF"][i, j] = ff; grids[scan]["Jsc"][i, j] = jsc
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"[{time.strftime('%H:%M:%S')}] {done}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    # cache
    cache = {"NT": NT, "ET": ET, "DEC": DEC,
             "ntet": {k: grids["ntet"][k].tolist() for k in grids["ntet"]},
             "ntcbo": {k: grids["ntcbo"][k].tolist() for k in grids["ntcbo"]}}
    (OUT / "scan2d_results.json").write_text(json.dumps(cache))
    print(f"[{time.strftime('%H:%M:%S')}] cached -> {OUT/'scan2d_results.json'}", flush=True)

    # ---- plot 4-panel heatmaps per scan (match SCAPS layout) ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
    logn = [9, 10, 11, 12, 13, 14, 15]

    def heat(scan, xvals, xlabel, title, fname):
        g = grids[scan]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
        for ax, key, lab in zip(axes.ravel(), ("PCE", "Voc", "FF", "Jsc"),
                                ("PCE (%)", "Voc (V)", "FF (%)", "Jsc (mA/cm2)")):
            im = ax.pcolormesh(np.arange(len(xvals) + 1), np.arange(len(logn) + 1), g[key],
                               cmap="viridis", shading="flat")
            ax.set_xticks(np.arange(len(xvals)) + 0.5); ax.set_xticklabels([f"{v:g}" for v in xvals], rotation=45, fontsize=7)
            ax.set_yticks(np.arange(len(logn)) + 0.5); ax.set_yticklabels(logn, fontsize=8)
            ax.set_xlabel(xlabel); ax.set_ylabel("log10 interface Nt (cm^-2)"); ax.set_title(lab, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"SolarLab — {title} (Perovskite/ETL, scaps_mirror_v2, transient)", fontsize=11)
        fig.tight_layout(); fig.savefig(OUT / fname, dpi=110); plt.close(fig)
        print(f"  wrote {OUT/fname}", flush=True)

    heat("ntet", ET, "Defect energy E_t (eV)", "interface Nt x Et 2D scan", "solarlab_ntet.png")
    heat("ntcbo", DEC, "Delta E_C (eV)", "interface Nt x dEc 2D scan", "solarlab_ntcbo.png")
    print(f"[{time.strftime('%H:%M:%S')}] DONE ({time.time()-t0:.0f}s total)", flush=True)
