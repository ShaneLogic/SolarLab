"""End-to-End Parity Test: Notebook (direct library) vs API (FastAPI).

Runs identical simulations through two channels:

  1. Notebook channel — imports `perovskite_sim` and calls `run_jv_sweep`,
     `run_impedance`, `run_degradation` directly.
  2. API channel — uses FastAPI's TestClient to call `/api/jv`,
     `/api/impedance`, `/api/degradation` endpoints (no real server needed).

Asserts that both channels produce byte-identical numerical results for
J-V sweeps, impedance spectra, and degradation curves. This guarantees the
two entry points remain in parity whenever the underlying library changes.
"""
from __future__ import annotations

import os
import sys
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.experiments.impedance import run_impedance
from perovskite_sim.experiments.degradation import run_degradation
from backend.main import app


CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "configs", "ionmonger_benchmark.yaml")
)

# Tight tolerances for bit-level reproducibility. The two channels share
# the same Python kernel, so any divergence is a bug, not round-off.
RTOL = 1e-12
ATOL = 1e-12

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    tag = "[PASS]" if ok else "[FAIL]"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  {tag} {name}" + (f"  ({detail})" if detail else ""))


def arrays_match(a: np.ndarray, b: np.ndarray) -> tuple[bool, str]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False, f"shape mismatch {a.shape} vs {b.shape}"
    if not np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True):
        diff = np.max(np.abs(a - b))
        return False, f"max |Δ| = {diff:.3e}"
    return True, ""


client = TestClient(app)

print("=" * 68)
print(" END-TO-END PARITY TEST — Notebook channel vs API channel")
print("=" * 68)
print(f"Config: {CONFIG_PATH}")
print()

# ─── 1. J-V Sweep ──────────────────────────────────────────────────────────
print("[1] J-V sweep (fast scan, v_rate=1.0 V/s)")

JV_PARAMS = dict(N_grid=45, n_points=20, v_rate=1.0, V_max=1.4)

stack = load_device_from_yaml(CONFIG_PATH)
nb_jv = run_jv_sweep(stack, **JV_PARAMS)

resp = client.post(
    "/api/jv",
    json={"config_path": CONFIG_PATH, **JV_PARAMS},
)
assert resp.status_code == 200, f"API error: {resp.text}"
api_jv = resp.json()["result"]

ok, det = arrays_match(nb_jv.V_fwd, api_jv["V_fwd"])
check("V_fwd arrays identical", ok, det)

ok, det = arrays_match(nb_jv.J_fwd, api_jv["J_fwd"])
check("J_fwd arrays identical", ok, det)

ok, det = arrays_match(nb_jv.V_rev, api_jv["V_rev"])
check("V_rev arrays identical", ok, det)

ok, det = arrays_match(nb_jv.J_rev, api_jv["J_rev"])
check("J_rev arrays identical", ok, det)

nb_m = nb_jv.metrics_fwd
api_m = api_jv["metrics_fwd"]
for k in ("V_oc", "J_sc", "FF", "PCE"):
    ok = abs(getattr(nb_m, k) - api_m[k]) < 1e-12
    check(f"metric[{k}] matches", ok, f"nb={getattr(nb_m, k):.6g} api={api_m[k]:.6g}")

print(f"    Notebook V_oc = {nb_m.V_oc:.4f} V, API V_oc = {api_m['V_oc']:.4f} V")

# ─── 2. Impedance Spectroscopy ──────────────────────────────────────────────
print("\n[2] Impedance spectroscopy")

IS_PARAMS = dict(N_grid=30, V_dc=0.9, n_freq=8, f_min=10.0, f_max=1e4)

freq_arr = np.logspace(np.log10(IS_PARAMS["f_min"]),
                       np.log10(IS_PARAMS["f_max"]),
                       IS_PARAMS["n_freq"])
nb_is = run_impedance(stack, freq_arr, V_dc=IS_PARAMS["V_dc"], N_grid=IS_PARAMS["N_grid"])

resp = client.post(
    "/api/impedance",
    json={"config_path": CONFIG_PATH, **IS_PARAMS},
)
assert resp.status_code == 200, f"API error: {resp.text}"
api_is = resp.json()["result"]

ok, det = arrays_match(nb_is.frequencies, api_is["frequencies"])
check("frequencies identical", ok, det)

ok, det = arrays_match(np.real(nb_is.Z), api_is["Z_real"])
check("Re(Z) identical", ok, det)

ok, det = arrays_match(np.imag(nb_is.Z), api_is["Z_imag"])
check("Im(Z) identical", ok, det)

# ─── 3. Degradation ─────────────────────────────────────────────────────────
print("\n[3] Degradation")

DEG_PARAMS = dict(N_grid=30, V_bias=0.9, t_end=10.0, n_snapshots=5)

nb_deg = run_degradation(stack, **DEG_PARAMS)

resp = client.post(
    "/api/degradation",
    json={"config_path": CONFIG_PATH, **DEG_PARAMS},
)
assert resp.status_code == 200, f"API error: {resp.text}"
api_deg = resp.json()["result"]

ok, det = arrays_match(nb_deg.t, api_deg["times"])
check("time array identical", ok, det)

ok, det = arrays_match(nb_deg.PCE, api_deg["PCE"])
check("PCE(t) identical", ok, det)

ok, det = arrays_match(nb_deg.V_oc, api_deg["V_oc"])
check("V_oc(t) identical", ok, det)

ok, det = arrays_match(nb_deg.J_sc, api_deg["J_sc"])
check("J_sc(t) identical", ok, det)

# ─── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print(f" RESULT: {PASS} passed, {FAIL} failed")
print("=" * 68)
sys.exit(0 if FAIL == 0 else 1)
