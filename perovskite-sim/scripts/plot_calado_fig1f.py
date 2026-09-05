#!/usr/bin/env python3
"""Reproduce Calado 2016 Fig 1e/1f under the paper's scan protocol and plot it.

Target: Calado et al., Nat. Commun. 7, 13831 (2016), Fig 1e (control, no
contact SRH) and Fig 1f (hysteretic, contact SRH tau = 2e-15 s), both at
40 mV/s. The device is ``configs/calado2016_fig1f.yaml`` (the paper's SI
Table 1 toy stack); the control is that stack with the contact-layer SRH
lifetime raised to ``CONTROL_TAU_S``.

Protocol (paper Methods / SI Note 2), driven with the primitives
``run_jv_sweep`` itself uses (a staircase of ``_integrate_step`` calls at
fixed bias, terminal current from ``_compute_current``):

    dark 0 V settle (SETTLE_S) -> soft light-on at 0 V -> walk to -1 V ->
    forward -1 V -> +1.2 V at 40 mV/s -> hold +1.2 V (HOLD_S) ->
    reverse +1.2 V -> -1 V at 40 mV/s.

The paper's uniform i-layer generation (2.5e21 cm^-3 s^-1) replaces the
preset's Beer-Lambert stand-in. Every ``_integrate_step`` span is relative
([0, dwell]): Radau's minimum step scales with eps*|t|, and absolute-time
spans fail at tau <= 1e-14 s (2026-09-04 measurement).

Outputs (SolarLab root ``docs/manual/figures/``, suffix ``STAMP``):

    Calado16Fig1fJV<STAMP>.png       forward/reverse J-V, control + hysteretic
    Calado16Fig1fMetrics<STAMP>.json J_sc, V_oc, P_max, FF per branch; HI in
                                     the paper (P_rev/P_fwd - 1) and SolarLab
                                     ((P_rev - P_fwd)/P_rev) definitions
    Calado16Fig1fCurves<STAMP>.npz   raw curves; ``--replot`` reuses them

Registered result (2026-09-05, N_grid 100, 20 mV steps): control HI 0.007;
hysteretic HI 0.44 against the paper's 1.84. The reverse branch is
quantitative (P_max,rev 81 vs ~85 W/m^2, V_oc,rev 0.78 vs ~0.73 V); the
forward collapse is about 2x too shallow (V_oc,fwd 0.59 V), cause open
(see the preset header).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import time

# The stiff, small tridiagonal solves run fastest single-threaded.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from perovskite_sim.experiments import jv_sweep as jv  # noqa: E402
from perovskite_sim.models.config_loader import load_device_from_yaml  # noqa: E402
from perovskite_sim.models.device import DeviceStack  # noqa: E402

CONFIG = ROOT / "configs" / "calado2016_fig1f.yaml"
OUT_DIR = ROOT.parent / "docs" / "manual" / "figures"
STAMP = "260905"

# Scan protocol (paper Methods).
SCAN_RATE_V_S = 0.04
V_START = -1.0
V_TOP = 1.2
DV = 0.02
HOLD_S = 3.0
SETTLE_S = 120.0
# Entry into the scan: a single dark-0 V -> light/-1 V shock defeats Radau
# from the ion-screened state, so illumination ramps up at 0 V and the bias
# then walks down to V_START under full light.
SOFT_START_FRACTIONS = (0.01, 0.1, 0.3, 1.0)
SOFT_START_S = 0.1
WALK_DOWN_STEPS = 20
WALK_DOWN_DWELL_S = 0.2

N_GRID = 100
RTOL, ATOL = 1e-4, 1e-6
G_UNIFORM = 2.5e27          # m^-3 s^-1  (SI Table 1: 2.5e21 cm^-3 s^-1)
CONTROL_TAU_S = 1.0         # contact SRH switched off (Fig 1e control)

# Paper figures of merit at 40 mV/s. HI values are quoted in the text; the
# others are read off Fig 1e/1f.
PAPER = {
    "J_sc_mA_cm2": 16.0,
    "V_oc_rev_V": 0.73,
    "P_max_rev_W_m2": 85.0,
    "HI_paper_fig1f": 1.84,
    "HI_paper_fig1e": 0.0,
}

# (key, label, colour) — categorical hues in fixed order.
CASES = (
    ("control", "no contact SRH (Fig 1e)", "#2166ac"),
    ("hysteretic", r"contact SRH $\tau$ = 2e-15 s (Fig 1f)", "#c0392b"),
)


@dataclasses.dataclass(frozen=True)
class BranchMetrics:
    J_sc: float      # A/m^2, active-cell sign (J > 0 at V = 0)
    V_oc: float      # V
    P_max: float     # W/m^2
    V_mp: float      # V
    FF: float


@dataclasses.dataclass(frozen=True)
class ProtocolResult:
    V_fwd: np.ndarray
    J_fwd: np.ndarray
    V_rev: np.ndarray
    J_rev: np.ndarray
    failures: tuple[tuple[float, str], ...]


def branch_metrics(V: np.ndarray, J: np.ndarray) -> BranchMetrics:
    """J_sc, V_oc, P_max, V_mp, FF of one scan branch (NaN points skipped)."""
    V = np.asarray(V, dtype=float)
    J = np.asarray(J, dtype=float)
    ok = np.isfinite(J)
    order = np.argsort(V[ok])
    V, J = V[ok][order], J[ok][order]
    J_sc = float(np.interp(0.0, V, J))
    pos = V > 0.0
    Vp, Jp = V[pos], J[pos]
    crossings = np.nonzero((Jp[:-1] > 0.0) & (Jp[1:] <= 0.0))[0]
    if crossings.size:
        i = int(crossings[0])
        V_oc = float(Vp[i] + (Vp[i + 1] - Vp[i]) * Jp[i] / (Jp[i] - Jp[i + 1]))
    else:
        V_oc = float("nan")
    P = Vp * Jp
    k = int(np.argmax(P))
    P_max, V_mp = float(P[k]), float(Vp[k])
    denom = J_sc * V_oc
    FF = P_max / denom if np.isfinite(denom) and denom > 0.0 else float("nan")
    return BranchMetrics(J_sc=J_sc, V_oc=V_oc, P_max=P_max, V_mp=V_mp, FF=FF)


def despike_current(J: np.ndarray, scale: float, window: int = 5,
                    threshold: float = 0.3) -> tuple[np.ndarray, int]:
    """Blank isolated single-step spikes: |J - running median| > threshold*scale.

    The external staircase bypasses run_jv_sweep's wrong-branch rejection, so a
    Radau step that lands on the injection branch shows up as one or two
    points off the curve (measured at 10 V/s on the Fig 1f ladder). A 5-point
    running median is blind to smooth S-shapes and plateaus; only isolated
    outliers are set to NaN. Returns the cleaned copy and the count removed.
    """
    J = np.asarray(J, dtype=float)
    half = window // 2
    padded = np.pad(J, half, mode="edge")
    median = np.array([np.nanmedian(padded[i:i + window]) for i in range(J.size)])
    spike = np.isfinite(J) & (np.abs(J - median) > threshold * abs(scale))
    cleaned = np.where(spike, np.nan, J)
    return cleaned, int(spike.sum())


def hysteresis_index_paper(p_fwd: float, p_rev: float) -> float:
    """Calado 2016 definition: P_max,rev / P_max,fwd - 1."""
    return p_rev / p_fwd - 1.0


def hysteresis_index_solarlab(p_fwd: float, p_rev: float) -> float:
    """run_jv_sweep definition: (P_max,rev - P_max,fwd) / P_max,rev."""
    return (p_rev - p_fwd) / p_rev


def control_stack(stack: DeviceStack) -> DeviceStack:
    """Same stack with the contact-layer SRH sink switched off (Fig 1e)."""
    layers = tuple(
        layer
        if layer.role == "absorber" or layer.params is None
        else dataclasses.replace(
            layer,
            params=dataclasses.replace(
                layer.params, tau_n=CONTROL_TAU_S, tau_p=CONTROL_TAU_S
            ),
        )
        for layer in stack.layers
    )
    return dataclasses.replace(stack, layers=layers)


def uniform_generation(x: np.ndarray, stack: DeviceStack) -> np.ndarray:
    """G_UNIFORM inside the absorber, zero elsewhere (paper's optical model)."""
    x0 = 0.0
    for layer in stack.layers:
        if layer.params is None:
            continue
        if layer.role == "absorber":
            x1 = x0 + layer.thickness
            return np.where((x >= x0) & (x <= x1), G_UNIFORM, 0.0)
        x0 += layer.thickness
    raise ValueError("stack has no absorber layer")


def _integrate(x, y, stack, mat, V, dwell, illuminated=True):
    return jv._integrate_step(
        x, y, stack, mat, float(V), 0.0, float(dwell), RTOL, ATOL,
        illuminated=illuminated,
    )


def _staircase(x, y, stack, mat, V_from, V_to, failures, scan_rate=SCAN_RATE_V_S):
    n = int(round(abs(V_to - V_from) / DV)) + 1
    Vs = np.linspace(V_from, V_to, n)
    dwell = abs(V_to - V_from) / (scan_rate * (n - 1))
    Js = np.full(n, np.nan)
    V_prev = None
    for i, V in enumerate(Vs):
        y_prev = y
        try:
            y = _integrate(x, y, stack, mat, V, dwell)
            Js[i] = jv._compute_current(
                x, y, stack, float(V), y_prev=y_prev, dt=dwell, mat=mat,
                V_app_prev=V_prev,
            )
        except RuntimeError as exc:
            # Diagnostic grade: record the point as NaN and keep the chain
            # alive on the last good state. Reported in the output.
            failures.append((float(V), str(exc)[:120]))
        V_prev = float(V)
    return Vs, Js, y


def run_protocol(
    stack: DeviceStack,
    n_grid: int,
    log=print,
    *,
    scan_rate: float = SCAN_RATE_V_S,
    hold_s: float = HOLD_S,
) -> ProtocolResult:
    """Paper protocol at ``scan_rate`` [V/s]; ``hold_s`` = 0 skips the +V_TOP dwell."""
    x = jv.build_electrical_grid(stack, n_grid)
    mat = dataclasses.replace(
        jv.build_material_arrays(x, stack), G_optical=uniform_generation(x, stack)
    )
    failures: list[tuple[float, str]] = []
    t0 = time.time()
    y = jv.solve_equilibrium(x, stack)      # algebraic seed: ions uniform
    y = _integrate(x, y, stack, mat, 0.0, SETTLE_S, illuminated=False)
    log(f"  dark 0 V settle {SETTLE_S:.0f} s done [{time.time() - t0:.0f} s]")
    for frac in SOFT_START_FRACTIONS:
        mat_frac = dataclasses.replace(mat, G_optical=mat.G_optical * frac)
        y = _integrate(x, y, stack, mat_frac, 0.0, SOFT_START_S)
    for V_pre in np.linspace(0.0, V_START, WALK_DOWN_STEPS + 1)[1:]:
        y = _integrate(x, y, stack, mat, V_pre, WALK_DOWN_DWELL_S)
    V_fwd, J_fwd, y = _staircase(
        x, y, stack, mat, V_START, V_TOP, failures, scan_rate=scan_rate
    )
    log(f"  forward scan done [{time.time() - t0:.0f} s]")
    if hold_s > 0.0:
        try:
            y = _integrate(x, y, stack, mat, V_TOP, hold_s)
        except RuntimeError as exc:
            failures.append((V_TOP, "hold: " + str(exc)[:120]))
    V_rev, J_rev, _ = _staircase(
        x, y, stack, mat, V_TOP, V_START, failures, scan_rate=scan_rate
    )
    log(f"  reverse scan done [{time.time() - t0:.0f} s], "
        f"failed steps {len(failures)}")
    return ProtocolResult(V_fwd, J_fwd, V_rev, J_rev, tuple(failures))


def photocurrent_scale(V: np.ndarray, J: np.ndarray) -> float:
    """|J| at V = 0 on a branch — the spike threshold's reference, not the
    injection current at +V_TOP, which can be 10-100x larger."""
    V = np.asarray(V, dtype=float)
    J = np.asarray(J, dtype=float)
    ok = np.isfinite(J)
    order = np.argsort(V[ok])
    return abs(float(np.interp(0.0, V[ok][order], J[ok][order])))


def summarise(result: ProtocolResult) -> dict:
    scale = photocurrent_scale(result.V_fwd, result.J_fwd)
    J_fwd, n_fwd = despike_current(result.J_fwd, scale)
    J_rev, n_rev = despike_current(result.J_rev, scale)
    fwd = branch_metrics(result.V_fwd, J_fwd)
    rev = branch_metrics(result.V_rev, J_rev)
    return {
        "forward": dataclasses.asdict(fwd),
        "reverse": dataclasses.asdict(rev),
        "HI_paper": hysteresis_index_paper(fwd.P_max, rev.P_max),
        "HI_solarlab": hysteresis_index_solarlab(fwd.P_max, rev.P_max),
        "failed_steps": [list(f) for f in result.failures],
        "spikes_removed": {"forward": n_fwd, "reverse": n_rev},
    }


def _deviation(sim: float, ref: float) -> str:
    return f"{100.0 * (sim - ref) / ref:+.0f} %" if ref else f"{sim - ref:+.3f}"


def plot(curves: dict[str, ProtocolResult], metrics: dict[str, dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, at) = plt.subplots(
        1, 2, figsize=(11.2, 4.7), dpi=200, gridspec_kw={"width_ratios": [1.3, 1.0]}
    )
    for key, label, color in CASES:
        c = curves[key]
        ax.plot(c.V_fwd, c.J_fwd / 10.0, color=color, lw=2.0, label=f"{label}, forward")
        ax.plot(c.V_rev, c.J_rev / 10.0, color=color, lw=2.0, ls="--",
                label=f"{label}, reverse")
    ax.axhline(0.0, color="0.55", lw=0.8)
    ax.axvline(0.0, color="0.55", lw=0.8)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-5.0, 22.0)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current density (mA cm$^{-2}$)")
    ax.set_title("SolarLab: Calado 2016 SI Table 1 stack, 40 mV s$^{-1}$", fontsize=10.5)
    ax.grid(color="0.92", lw=0.6)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    h, cnt = metrics["hysteretic"], metrics["control"]
    rows = (
        ("$J_{sc}$ (mA cm$^{-2}$)", f"{h['reverse']['J_sc'] / 10.0:.1f}",
         f"≈ {PAPER['J_sc_mA_cm2']:.0f}",
         _deviation(h["reverse"]["J_sc"] / 10.0, PAPER["J_sc_mA_cm2"])),
        ("$V_{oc}$, reverse (V)", f"{h['reverse']['V_oc']:.2f}",
         f"≈ {PAPER['V_oc_rev_V']:.2f}",
         _deviation(h["reverse"]["V_oc"], PAPER["V_oc_rev_V"])),
        ("$P_{max}$, reverse (W m$^{-2}$)", f"{h['reverse']['P_max']:.0f}",
         f"≈ {PAPER['P_max_rev_W_m2']:.0f}",
         _deviation(h["reverse"]["P_max"], PAPER["P_max_rev_W_m2"])),
        ("HI, hysteretic (Fig 1f)", f"{h['HI_paper']:.2f}",
         f"{PAPER['HI_paper_fig1f']:.2f}",
         _deviation(h["HI_paper"], PAPER["HI_paper_fig1f"])),
        ("HI, control (Fig 1e)", f"{cnt['HI_paper']:.3f}",
         f"{PAPER['HI_paper_fig1e']:.2f}",
         _deviation(cnt["HI_paper"], PAPER["HI_paper_fig1e"])),
    )
    at.axis("off")
    table = at.table(
        cellText=[r[1:] for r in rows],
        rowLabels=[r[0] for r in rows],
        colLabels=("SolarLab", "Calado 2016", "Deviation"),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.55)
    for cell in table.get_celld().values():
        cell.set_edgecolor("0.8")
    at.set_title("Figures of merit, hysteretic cell vs paper Fig 1f", fontsize=10.5)
    at.text(
        0.0, 0.06,
        "HI = $P_{max,rev}\\,/\\,P_{max,fwd} - 1$ (paper definition)\n"
        "protocol: dark 0 V settle, light on, $-1 \\to +1.2$ V, 3 s hold, back",
        transform=at.transAxes, fontsize=8, color="0.3", va="top",
    )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _save_curves(path: Path, curves: dict[str, ProtocolResult]) -> None:
    flat = {}
    for key, c in curves.items():
        for field in ("V_fwd", "J_fwd", "V_rev", "J_rev"):
            flat[f"{key}_{field}"] = getattr(c, field)
    np.savez(path, **flat)


def _load_curves(path: Path) -> dict[str, ProtocolResult]:
    data = np.load(path)
    return {
        key: ProtocolResult(
            data[f"{key}_V_fwd"], data[f"{key}_J_fwd"],
            data[f"{key}_V_rev"], data[f"{key}_J_rev"], (),
        )
        for key, _, _ in CASES
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--n-grid", type=int, default=N_GRID)
    ap.add_argument("--replot", action="store_true",
                    help="reuse the cached curves .npz instead of re-running the solver")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.out_dir / f"Calado16Fig1fJV{STAMP}.png"
    out_json = args.out_dir / f"Calado16Fig1fMetrics{STAMP}.json"
    out_npz = args.out_dir / f"Calado16Fig1fCurves{STAMP}.npz"

    if args.replot:
        curves = _load_curves(out_npz)
    else:
        base = load_device_from_yaml(str(CONFIG))
        stacks = {"control": control_stack(base), "hysteretic": base}
        curves = {}
        for key, label, _ in CASES:
            print(f"{key}: {label}", flush=True)
            curves[key] = run_protocol(stacks[key], args.n_grid, log=print)
        _save_curves(out_npz, curves)

    metrics = {key: summarise(curves[key]) for key, _, _ in CASES}
    payload = {
        "config": str(CONFIG.relative_to(ROOT)),
        "n_grid": args.n_grid,
        "protocol": {
            "scan_rate_V_s": SCAN_RATE_V_S, "V_start": V_START, "V_top": V_TOP,
            "dV": DV, "hold_s": HOLD_S, "dark_settle_s": SETTLE_S,
            "G_uniform_m3_s": G_UNIFORM,
        },
        "paper": PAPER,
        "cases": metrics,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot(curves, metrics, out_png)

    for key, _, _ in CASES:
        m = metrics[key]
        print(
            f"{key:10s} HI_paper={m['HI_paper']:.3f} HI_solarlab={m['HI_solarlab']:.3f} "
            f"Jsc_rev={m['reverse']['J_sc'] / 10.0:.1f} mA/cm2 "
            f"Voc_rev={m['reverse']['V_oc']:.3f} V "
            f"Pmax_fwd={m['forward']['P_max']:.1f} Pmax_rev={m['reverse']['P_max']:.1f} W/m2 "
            f"failed_steps={len(m['failed_steps'])}"
        )
        for V, msg in m["failed_steps"][:6]:
            print(f"  FAIL V={V:.3f}: {msg}")
    print(f"wrote {out_png}\n      {out_json}\n      {out_npz}")
    return 1 if any(metrics[k]["failed_steps"] for k, _, _ in CASES) else 0


if __name__ == "__main__":
    sys.exit(main())
