"""Shared-occupancy Pauwels-Vanhoutte interface recombination (2026-06).

A single-level interface trap holds ONE occupancy balancing electron and hole
capture from BOTH adjacent layers. ``interface_shared_occupancy`` (default
OFF, env ``SOLARLAB_IFACE_SHARED_OCC=1``) replaces the one-sided cross-pair
rate at defect interfaces with the coupled closed form

    R = (nS*pS - refS) / ((nS + n1S)/v_p + (pS + p1S)/v_n)

with nS = n_L + n_R, pS = p_L + p_R (floored at zero), per-side trap-level
densities n1_i/p1_i referenced to each side's own band edge and effective DOS
(depth_i = E_t + (chi_ref - chi_i)), and the discrete-equilibrium-consistent
numerator reference refS = (n_L_eq + n_R_eq)*(p_L_eq + p_R_eq) so R vanishes
exactly when the sampled nodes sit at their cached dark-equilibrium values
(the same detailed-balance convention as the proven one-sided path; the
textbook n1S*p1S reference assumes interface-plane sampling). Not composed
with the projection or QSS env paths.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.recombination import interface_recombination
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import (
    build_material_arrays,
    _apply_interface_recombination,
)

_V2 = "configs/scaps_mirror_v2.yaml"
_V_T = 0.025852


def _build(stack):
    elec = electrical_layers(stack)
    x = multilayer_grid([Layer(thickness=L.thickness, N=30 // len(elec)) for L in elec])
    return x, build_material_arrays(x, stack)


def _interface_dn(stack, x, mat, scale=1.0e22):
    N = len(x)
    n = np.full(N, scale)
    p = np.full(N, scale)
    phi = np.linspace(0.0, 1.0, N)
    dn = np.zeros(N)
    dp = np.zeros(N)
    _apply_interface_recombination(dn, dp, n, p, stack, mat, phi)
    return dn


# ----------------------------- flag plumbing -----------------------------

def test_flag_default_off():
    assert load_scaps_yaml(_V2).interface_shared_occupancy is False


def test_scaps_yaml_key_roundtrip(tmp_path):
    cfg = yaml.safe_load(Path(_V2).read_text())
    cfg["device"]["interface_shared_occupancy"] = True
    dst = tmp_path / "so.yaml"
    dst.write_text(yaml.safe_dump(cfg))
    assert load_scaps_yaml(str(dst)).interface_shared_occupancy is True


def test_mat_cache_from_flag_and_env(monkeypatch):
    base = load_scaps_yaml(_V2)
    _, m0 = _build(base)
    assert m0.iface_shared_occ is False
    _, m1 = _build(dataclasses.replace(base, interface_shared_occupancy=True))
    assert m1.iface_shared_occ is True
    monkeypatch.setenv("SOLARLAB_IFACE_SHARED_OCC", "1")
    _, m2 = _build(base)
    assert m2.iface_shared_occ is True


# ------------------------- per-side trap densities ------------------------

def test_per_side_n1_p1_referenced_to_each_layers_bands():
    """v2, PVK/ETL interface (electrical k=1): E_t = 0.6 eV below the PVK CB.
    PVK side: depth 0.6, n1 = N_C,PVK*exp(-0.6/V_T).
    ETL side: depth 0.6 + (chi_PVK - chi_ETL) = 0.44, n1 = N_C,ETL*exp(-0.44/V_T)."""
    stack = load_scaps_yaml(_V2)
    _, mat = _build(stack)
    k = 1  # PVK/ETL
    n1_pvk = 1.0e25 * math.exp(-0.60 / _V_T)
    n1_etl = 8.0e25 * math.exp(-0.44 / _V_T)
    assert mat.interface_n1_L[k] == pytest.approx(n1_pvk, rel=1e-3)
    assert mat.interface_n1_R[k] == pytest.approx(n1_etl, rel=1e-3)
    # holes: p1_i = N_V,i*exp(-(Eg_i - depth_i)/V_T)
    p1_pvk = 1.0e25 * math.exp(-(1.53 - 0.60) / _V_T)
    p1_etl = 8.0e25 * math.exp(-(1.90 - 0.44) / _V_T)
    assert mat.interface_p1_L[k] == pytest.approx(p1_pvk, rel=1e-3)
    assert mat.interface_p1_R[k] == pytest.approx(p1_etl, rel=1e-3)


# ----------------------------- rate behaviour -----------------------------

def test_rate_vanishes_at_cached_dark_equilibrium():
    """With both sampled sides at their cached equilibrium densities, the
    shared-occupancy numerator is exactly zero by construction."""
    stack = load_scaps_yaml(_V2)
    _, mat = _build(stack)
    for k in (0, 1):
        nS = mat.interface_n_L_eq[k] + mat.interface_n_R_eq[k]
        pS = mat.interface_p_L_eq[k] + mat.interface_p_R_eq[k]
        refS = nS * pS
        n1S = mat.interface_n1_L[k] + mat.interface_n1_R[k]
        p1S = mat.interface_p1_L[k] + mat.interface_p1_R[k]
        R = interface_recombination(nS, pS, refS, n1S, p1S, 0.01, 0.01)
        assert R == pytest.approx(0.0, abs=1e-25)


def test_shared_occupancy_changes_interface_rate():
    base = load_scaps_yaml(_V2)
    x, m_off = _build(base)
    _, m_on = _build(dataclasses.replace(base, interface_shared_occupancy=True))
    dn_off = _interface_dn(base, x, m_off)
    dn_on = _interface_dn(base, x, m_on)
    assert not np.allclose(dn_off, dn_on)


def test_legacy_interfaces_unaffected():
    from perovskite_sim.models.config_loader import load_device_from_yaml
    base = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    x, m_off = _build(base)
    _, m_on = _build(dataclasses.replace(base, interface_shared_occupancy=True))
    np.testing.assert_array_equal(_interface_dn(base, x, m_off),
                                  _interface_dn(base, x, m_on))


def test_base_jv_converges_at_validation_protocol():
    """The stability gate that killed the unfloored mirror pair: the base J-V
    must converge on the exact validation grid (V_max=1.6, n=40) with the
    shared-occupancy rate active (+ DOS flag, the parity configuration)."""
    from perovskite_sim.experiments.jv_sweep import run_jv_sweep
    base = load_scaps_yaml(_V2)
    so = dataclasses.replace(base, interface_shared_occupancy=True,
                             dos_band_potentials=True)
    m = run_jv_sweep(so, N_grid=30, n_points=40, v_rate=5.0, V_max=1.6,
                     v_max_max_attempts=3).metrics_fwd
    assert m.voc_bracketed
    assert 0.9 < m.V_oc < 1.27
    assert m.J_sc / 10 == pytest.approx(25.7, abs=0.6)
