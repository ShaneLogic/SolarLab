"""Two-sided Pauwels-Vanhoutte interface capture (2026-06).

SCAPS's interface recombination couples the trap to BOTH adjacent layers
(electron and hole capture from each side — the jn1/jn2/jp1/jp2 currents).
SolarLab's E1.5 formulation samples ONE cross-carrier pair (n from the
transport-side interior node, p from the absorber-side node). The
``interface_two_sided`` flag (default OFF, env ``SOLARLAB_IFACE_TWOSIDED=1``)
adds the Newton-safe mirror pair:

  R_A = PV(n[idx+1], p[idx-1]; ni_A² = n_R_eq·p_L_eq)   (existing, unchanged)
  R_B = PV(n[idx-1], p[idx+1]; ni_B² = n_L_eq·p_R_eq)   (mirror pair)

Each pair carries its own detailed-balance reference (cached by the Phase-E3
block), vanishes at dark equilibrium by construction, and is clamped
non-negative individually. Pair B activates only where an interface defect is
present (cross-carrier eval nodes), so legacy interfaces are bit-identical.
"""
from __future__ import annotations

import dataclasses
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
    stack = load_scaps_yaml(_V2)
    assert stack.interface_two_sided is False


def test_scaps_yaml_key_roundtrip(tmp_path):
    cfg = yaml.safe_load(Path(_V2).read_text())
    cfg["device"]["interface_two_sided"] = True
    dst = tmp_path / "ts.yaml"
    dst.write_text(yaml.safe_dump(cfg))
    assert load_scaps_yaml(str(dst)).interface_two_sided is True


def test_mat_cache_from_stack_flag():
    stack = dataclasses.replace(load_scaps_yaml(_V2), interface_two_sided=True)
    _, mat = _build(stack)
    assert mat.iface_two_sided is True


def test_mat_cache_from_env(monkeypatch):
    monkeypatch.setenv("SOLARLAB_IFACE_TWOSIDED", "1")
    _, mat = _build(load_scaps_yaml(_V2))
    assert mat.iface_two_sided is True


def test_mat_cache_default_off():
    _, mat = _build(load_scaps_yaml(_V2))
    assert mat.iface_two_sided is False


# ----------------------------- behaviour ---------------------------------

def test_two_sided_adds_mirror_channel():
    base = load_scaps_yaml(_V2)
    x, m_off = _build(base)
    _, m_on = _build(dataclasses.replace(base, interface_two_sided=True))
    dn_off = _interface_dn(base, x, m_off)
    dn_on = _interface_dn(base, x, m_on)
    assert not np.allclose(dn_off, dn_on), "mirror pair must change the rate"
    # the added channel is recombination (more negative dn at interface nodes)
    assert dn_on.sum() <= dn_off.sum() + 1e-30


def test_pair_b_vanishes_at_dark_equilibrium():
    """The mirror pair's detailed-balance reference makes R_B exactly zero
    when both sides sit at their cached equilibrium densities."""
    stack = load_scaps_yaml(_V2)
    _, mat = _build(stack)
    for k in range(len(mat.interface_nodes)):
        n_L_eq = mat.interface_n_L_eq[k]
        p_R_eq = mat.interface_p_R_eq[k]
        ni_B = n_L_eq * p_R_eq
        R = interface_recombination(
            n_L_eq, p_R_eq, ni_B,
            mat.interface_n1[k], mat.interface_p1[k], 0.01, 0.01,
        )
        assert R == pytest.approx(0.0, abs=1e-20)


def test_legacy_interfaces_unaffected():
    """Interfaces without an InterfaceDefect (eval nodes == idx) must not
    receive the mirror pair — flag on == flag off on such stacks."""
    from perovskite_sim.models.config_loader import load_device_from_yaml
    base = load_device_from_yaml("configs/nip_MAPbI3.yaml")  # SRVs, no defects
    x, m_off = _build(base)
    _, m_on = _build(dataclasses.replace(base, interface_two_sided=True))
    np.testing.assert_array_equal(_interface_dn(base, x, m_off),
                                  _interface_dn(base, x, m_on))


def test_base_jv_runs_and_stays_sane():
    """Full J-V with the mirror pair: converges, brackets, V_oc within a
    bounded shift of the one-sided result (pair B is minority-limited at the
    base point)."""
    from perovskite_sim.experiments.jv_sweep import run_jv_sweep
    base = load_scaps_yaml(_V2)
    ts = dataclasses.replace(base, interface_two_sided=True)
    kw = dict(N_grid=30, n_points=40, v_rate=5.0, V_max=1.3, v_max_max_attempts=2)
    m0 = run_jv_sweep(base, **kw).metrics_fwd
    m1 = run_jv_sweep(ts, **kw).metrics_fwd
    assert m1.voc_bracketed
    assert abs(m1.V_oc - m0.V_oc) < 0.06
    assert abs(m1.J_sc - m0.J_sc) / m0.J_sc < 0.02
