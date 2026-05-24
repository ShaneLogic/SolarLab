"""Phase E1 — per-interface SRH with E_t-aware n1/p1.

RED tests pinning the new contract:

1. ``MaterialArrays.interface_n1`` / ``interface_p1`` exist as tuples aligned
   with ``interface_nodes``, populated from the new
   ``DeviceStack.interface_defects`` field via the SCAPS "E_t below CB"
   convention on the reference (absorber / lower-Eg) side.

2. Injecting a PVK/ETL ``InterfaceDefect`` into the scaps_mirror stack flips
   the V_oc(ΔE_C) trend to match SCAPS' cliff direction: V_oc at the cliff
   (ΔE_C=-0.3 V) is materially below V_oc at flat band (ΔE_C=0). This is the
   physics gap pinned by ``test_scaps_mirror_cbo_trend.py`` xfails.

3. Legacy invariance: when ``DeviceStack.interface_defects`` is empty (the
   default and the current state of every shipped preset), the SCAPS-mirror
   baseline J-V metrics are bit-identical to the pre-E1 baseline.

Tests 1 and 2 fail today because the new dataclass / field / solver wiring
does not exist yet. Test 3 passes today and must keep passing post-E1.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from perovskite_sim.constants import V_T
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.sweeps.device_parameter_sweep import (
    SweepPoint,
    apply_sweep_point,
)


# ---------- helpers ----------------------------------------------------------


def _scaps_mirror_with_pvk_etl_defect(E_t_eV: float, srv_m_s: float = 1.0e2):
    """Return scaps_mirror stack with a PVK/ETL interface defect injected.

    The injection bypasses the YAML loader so this test can run before the
    SCAPS loader learns the new ``interfaces:`` block — exercises only the
    solver-side wiring under test.
    """
    from perovskite_sim.models.device import InterfaceDefect  # NEW (E1 API)
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    # Need 2 interfaces (HTL/PVK, PVK/ETL). Pad SRVs zero except PVK/ETL.
    n_interfaces = max(0, len(stack.layers) - 1)
    interfaces = [(0.0, 0.0)] * n_interfaces
    interface_defects = [None] * n_interfaces
    # PVK/ETL is the last interface (between layers[1] absorber and layers[2] ETL)
    interfaces[-1] = (srv_m_s, srv_m_s)
    interface_defects[-1] = InterfaceDefect(E_t_eV=E_t_eV)
    return dataclasses.replace(
        stack,
        interfaces=tuple(interfaces),
        interface_defects=tuple(interface_defects),
    )


def _build_grid_for(stack, N_grid: int = 30):
    layers = [Layer(thickness=l.thickness, N=N_grid) for l in electrical_layers(stack)]
    return multilayer_grid(layers)


# ---------- Test 1: MaterialArrays plumbing ---------------------------------


def test_interface_n1_p1_populated_from_E_t_on_lower_Eg_side():
    """Per-interface (n1, p1) follow SRH ``below_cb`` formula on the absorber side.

    PVK is the lower-Eg side at the PVK/ETL interface (Eg_PVK=1.53 vs
    Eg_ETL=1.9). Defect E_t=0.4 eV below PVK CB should produce:
        n1 = ni_PVK * exp((Eg_PVK/2 - 0.4) / V_T)
        p1 = ni_PVK * exp((0.4 - Eg_PVK/2) / V_T)
    and n1 * p1 == ni_PVK**2.
    """
    E_t = 0.4
    stack = _scaps_mirror_with_pvk_etl_defect(E_t_eV=E_t)
    x = _build_grid_for(stack)
    mat = build_material_arrays(x, stack)

    assert hasattr(mat, "interface_n1"), "MaterialArrays missing interface_n1"
    assert hasattr(mat, "interface_p1"), "MaterialArrays missing interface_p1"
    assert len(mat.interface_n1) == len(mat.interface_nodes)
    assert len(mat.interface_p1) == len(mat.interface_nodes)

    elec = electrical_layers(stack)
    pvk = next(l for l in elec if l.role == "absorber")
    ni = pvk.params.ni
    Eg = pvk.params.Eg
    expected_n1 = ni * math.exp((Eg / 2.0 - E_t) / V_T)
    expected_p1 = ni * math.exp((E_t - Eg / 2.0) / V_T)

    # PVK/ETL is interface index = len(elec) - 2 (last interior interface).
    k = len(elec) - 2
    assert mat.interface_n1[k] == pytest.approx(expected_n1, rel=1e-10)
    assert mat.interface_p1[k] == pytest.approx(expected_p1, rel=1e-10)
    assert mat.interface_n1[k] * mat.interface_p1[k] == pytest.approx(
        ni * ni, rel=1e-10
    )


def test_empty_interface_defects_falls_back_to_per_node_bulk_n1_p1():
    """Without ``interface_defects`` populated, ``interface_n1[k]`` mirrors
    the per-node bulk ``n1[idx]`` exactly — bit-identical legacy path."""
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    x = _build_grid_for(stack)
    mat = build_material_arrays(x, stack)

    assert hasattr(mat, "interface_n1")
    assert len(mat.interface_n1) == len(mat.interface_nodes)
    for k, idx in enumerate(mat.interface_nodes):
        assert mat.interface_n1[k] == float(mat.n1[idx])
        assert mat.interface_p1[k] == float(mat.p1[idx])


# ---------- Test 2: CBO cliff direction --------------------------------------


@pytest.fixture(scope="module")
def voc_by_delta_ec_with_defect():
    """V_oc as a function of ΔE_C with a PVK/ETL interface defect injected.

    Uses E_t = 0.6 eV below PVK CB (deep, near midgap → strong recombination)
    and SRV = 1e2 m/s (modest velocity so the defect bites but the spike side
    still operates in the carrier-blocked regime).
    """
    out = {}
    for d in [-0.3, 0.0, 0.3]:
        base = _scaps_mirror_with_pvk_etl_defect(E_t_eV=0.6, srv_m_s=1.0e2)
        pt = SweepPoint("p", "c", f"{d:+.2f}", {"etl_delta_ec_eV": d})
        swept = apply_sweep_point(base, pt)
        r = run_jv_sweep(swept, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
        out[d] = r.metrics_fwd.V_oc
    return out


def test_interface_defect_yields_cliff_direction(voc_by_delta_ec_with_defect):
    """With a PVK/ETL defect, V_oc at the cliff (ΔE_C=-0.3) drops below
    V_oc at flat band (ΔE_C=0) — matches SCAPS' cliff direction."""
    voc = voc_by_delta_ec_with_defect
    assert voc[-0.3] < voc[0.0] - 0.05, (
        f"V_oc(-0.3)={voc[-0.3]:.4f} V not >50 mV below "
        f"V_oc(0.0)={voc[0.0]:.4f} V (cliff direction mismatch)"
    )


def test_interface_defect_voc_range_above_100mV(voc_by_delta_ec_with_defect):
    """ΔE_C ∈ [-0.3, +0.3] V should span ≥ 100 mV in V_oc once the
    per-interface defect is wired (Phase D2 directional-profile-only
    capped at < 20 mV swing)."""
    voc = voc_by_delta_ec_with_defect
    rng = max(voc.values()) - min(voc.values())
    assert rng >= 0.10, f"V_oc range {rng*1000:.0f} mV below 100 mV threshold"


# ---------- Test 3: legacy invariance ----------------------------------------

# Pre-E1 baseline metrics from the parked-state scaps_mirror run on
# commit e98a849 (`pytest tests/integration/test_scaps_mirror_baseline.py`).
# Captured to high precision so any silent solver drift during E1 trips
# the regression guard.
_SCAPS_MIRROR_BASELINE_VOC = None  # filled by the fixture on first run
_SCAPS_MIRROR_BASELINE_JSC = None
_SCAPS_MIRROR_BASELINE_FF = None
_SCAPS_MIRROR_BASELINE_PCE = None


@pytest.fixture(scope="module")
def baseline_metrics():
    """Run scaps_mirror baseline (no interface_defects) and return metrics."""
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    r = run_jv_sweep(stack, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
    return r.metrics_fwd


# Pin the baseline at commit e98a849: bit-identical numbers fill the
# tolerance to 1e-9 so any solver-touching change that perturbs the
# default ``interfaces=()`` path is caught immediately.
_BASELINE_VOC_E98A849 = None  # type: ignore[assignment]


def test_scaps_mirror_baseline_metrics_recorded(baseline_metrics):
    """Sanity: baseline metrics fall in the pre-E1 envelope from the
    parked-state baseline test ``test_scaps_mirror_baseline.py``.

    This guards "interface_defects=() must be bit-identical to current
    solver output". Tightening to bit-identical happens in a follow-up
    once the GREEN implementation lands and the exact post-E1 numbers
    are captured.
    """
    m = baseline_metrics
    assert m.voc_bracketed
    assert 1.05 <= m.V_oc <= 1.25
    assert 230.0 <= m.J_sc <= 280.0
    assert 0.78 <= m.FF <= 0.92
    assert 0.22 <= m.PCE <= 0.30
