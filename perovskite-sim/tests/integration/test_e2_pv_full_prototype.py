"""Phase E2 Sprint 5 — full Pauwels-Vanhoutte 1978 interface SRH prototype.

After three single-knob failures (BBD, thin-shell, single-sided PV), this
prototype implements the FULL Pauwels-Vanhoutte form with three missing
ingredients restored:

1. **V_1, V_2 charge-balance partition** (not heavy-doping V_1=0). Cached
   on MaterialArrays per interface. Formula:
       V_1 + V_2 = V_bi − V_app
       V_1 / V_2 = (ε_2 · N_2) / (ε_1 · N_1)
   For heavily-doped ETL (N_1) and lightly-doped PVK (N_2), V_2 → V_bi−V_app
   and V_1 → 0 — matching paper eq (30) heavy-doping limit but reproducing
   it from charge balance, not by ad-hoc truncation.

2. **Two-sided Shockley-Read SRH** (paper eq 7): R = R_s1 + R_s2 where
       R_s1 uses (n_1s, p_2s) — ETL-side electron capture, PVK-side hole
       R_s2 uses (n_2s, p_1s) — PVK-side electron capture, ETL-side hole

3. **All four interface-plane densities** (paper eqs 8, 9, 11):
       n_1s = n[bulk_ETL] · exp(−V_1/V_T)    # ETL-side electron, depleted
       p_1s = p[bulk_ETL] · exp(+V_1/V_T)    # ETL-side hole, accumulated
       n_2s = n[bulk_PVK] · exp(+V_2/V_T)    # PVK-side electron, accumulated
       p_2s = p[bulk_PVK] · exp(−V_2/V_T)    # PVK-side hole, depleted

Activated by env var ``SOLARLAB_PV_FULL=1``. Default unset → legacy E1.5
cross-carrier path, bit-identical.

Contract pinned by this test file:
1. env unset → V_oc bit-identical to current main baseline (1.0694 V).
2. env=0 / non-canonical → legacy V_oc (only literal "1" activates).
3. env=1 → V_oc moves ≥1 mV (proves wiring).
4. env=1 → JV arrays finite (no NaN/Inf).
5. env=1 → V_oc within physical envelope [0.8, 1.3] V.
6. MaterialArrays.interface_V_partition_2 cached + populated correctly:
   - non-empty when interface defects present
   - partition_2 ∈ [0, 1]
   - SCAPS-mirror (PVK 1e14 vs ETL 1e18) → partition_2 ≥ 0.99 (heavy-doping)
"""
from __future__ import annotations

import math

import pytest

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.models.device import electrical_layers


_LEGACY_V_OC = 1.0694  # current main-branch scaps_mirror.yaml baseline


def _voc(stack) -> float:
    r = run_jv_sweep(stack, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
    assert r.metrics_fwd.voc_bracketed, "V_oc must bracket on scaps_mirror"
    return float(r.metrics_fwd.V_oc)


def _build_scaps_mat():
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    elec = electrical_layers(stack)
    layers_grid = [Layer(thickness=L.thickness, N=30 // len(elec)) for L in elec]
    x = multilayer_grid(layers_grid)
    mat = build_material_arrays(x, stack)
    return stack, mat


def test_pv_full_env_unset_legacy(monkeypatch):
    """env unset → legacy E1.5 V_oc 1.0694 ±5 mV."""
    monkeypatch.delenv("SOLARLAB_PV_FULL", raising=False)
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    assert _voc(stack) == pytest.approx(_LEGACY_V_OC, abs=5.0e-3)


def test_pv_full_env_zero_legacy(monkeypatch):
    """env=0 → legacy V_oc (only literal "1" activates)."""
    monkeypatch.setenv("SOLARLAB_PV_FULL", "0")
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    assert _voc(stack) == pytest.approx(_LEGACY_V_OC, abs=5.0e-3)


def test_pv_full_env_malformed_legacy(monkeypatch):
    """env=non-canonical → legacy V_oc (defensive)."""
    monkeypatch.setenv("SOLARLAB_PV_FULL", "true")
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    assert _voc(stack) == pytest.approx(_LEGACY_V_OC, abs=5.0e-3)


def test_pv_full_env_active_voc_moves(monkeypatch):
    """env=1 → V_oc moves measurably from legacy.

    Full PV restores two-sided rate + charge-balance V_1/V_2 partition.
    Expected motion: O(10-50 mV). Direction not pinned because two-sided
    rate can either suppress (if V_2 small) or amplify (if V_1 nontrivial).
    """
    monkeypatch.setenv("SOLARLAB_PV_FULL", "1")
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    voc = _voc(stack)
    assert 0.8 <= voc <= 1.3, f"V_oc {voc} outside physical envelope"
    assert abs(voc - _LEGACY_V_OC) >= 1.0e-3, (
        f"full PV V_oc {voc:.4f} did not move from legacy {_LEGACY_V_OC}; "
        "activation path may be silently bypassed."
    )


def test_pv_full_env_active_finite_jv(monkeypatch):
    """env=1 JV completes without NaN/Inf."""
    monkeypatch.setenv("SOLARLAB_PV_FULL", "1")
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    r = run_jv_sweep(stack, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
    for v in r.V_fwd:
        assert math.isfinite(v), f"non-finite V {v}"
    for j in r.J_fwd:
        assert math.isfinite(j), f"non-finite J {j}"


def test_pv_full_material_arrays_carries_v_partition():
    """MaterialArrays.interface_V_partition_2 is populated for scaps_mirror.

    Required for full PV runtime path. SCAPS-mirror has one PVK/ETL
    interface defect → tuple length 1. Partition is the fraction of
    band-bending V_bi−V_app absorbed by the PVK (light-doped) side.
    """
    _, mat = _build_scaps_mat()
    assert hasattr(mat, "interface_V_partition_2"), (
        "MaterialArrays must expose interface_V_partition_2 for full PV"
    )
    # Non-empty because scaps_mirror.yaml declares one InterfaceDefect.
    assert len(mat.interface_V_partition_2) >= 1
    for p in mat.interface_V_partition_2:
        assert 0.0 <= p <= 1.0, f"partition {p} outside [0, 1]"


def test_pv_full_heavy_doping_partition_approaches_unity():
    """SCAPS-mirror partition_2 ≥ 0.99 in heavy-doping ETL limit.

    PVK N_A = 1e14 cm⁻³ vs ETL N_D = 1e18 cm⁻³ → ratio 1e-4 → partition_2
    is ε_1·N_1 / (ε_1·N_1 + ε_2·N_2). Assuming ε_1 ≈ ε_2 (perovskite-like
    relative permittivities), partition_2 ≈ N_1 / (N_1 + N_2) ≈ 1e18 /
    (1e18 + 1e14) = 0.9999.
    """
    _, mat = _build_scaps_mat()
    # PVK/ETL interface is the last in the chain (after HTL/PVK).
    p2_pvk_etl = mat.interface_V_partition_2[-1]
    assert p2_pvk_etl >= 0.99, (
        f"PVK/ETL partition_2={p2_pvk_etl} should be ≥ 0.99 in heavy-doping "
        f"ETL limit (N_ETL=1e18 ≫ N_PVK=1e14)"
    )
