"""QSS interface-plane closure (2026-06) — physics/interface_plane.py + wiring.

The closure evaluates defect-interface recombination on TRUE plane densities
solved from a local implicit 2x2 flux balance (supply-limited, reduced
interface gap, trap-level-visible) instead of sampling bulk nodes — the
structural fix identified by the four bulk-node falsifications (two-sided,
shared-occupancy, projection, Gaussian-E_t; see CLAUDE.md).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.interface_plane import (
    S_SUPPLY,
    build_plane_params,
    plane_rate,
    solve_plane_densities,
)
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import (
    build_material_arrays,
    _apply_interface_recombination,
)

_V2 = "configs/scaps_mirror_v2.yaml"
_VT = 0.025852
# PVK(L)/ETL(R)-like parameter set: electron edge on the ETL side, hole
# edge on the PVK side -> reduced interface gap Eg_s = Eg_PVK - |dE_C|.
_PRM = build_plane_params(
    chi_L=3.94, chi_R=4.10, ceg_L=3.94 + 1.53, ceg_R=4.10 + 3.20,
    Nc_ref=1.0e25, Nv_ref=1.0e25, chi_ref=3.94, E_t_eV=0.6, V_T=_VT,
)


def _dos_stack():
    return dataclasses.replace(load_scaps_yaml(_V2), dos_band_potentials=True)


def _build(stack):
    elec = electrical_layers(stack)
    x = multilayer_grid([Layer(thickness=L.thickness, N=30 // len(elec)) for L in elec])
    return x, build_material_arrays(x, stack)


# --------------------------- pure closure physics ---------------------------

def test_plane_params_consistency():
    """n1_s * p1_s == ni_s^2 (detailed balance of the trap level) and the
    reduced interface gap shrinks with a conduction-band cliff."""
    assert _PRM.n1_s * _PRM.p1_s == pytest.approx(_PRM.ni_s_sq, rel=1e-9)
    cliff = build_plane_params(3.94, 4.60, 3.94 + 1.53, 4.60 + 3.20,
                               1.0e25, 1.0e25, 3.94, 0.6, _VT)
    # 0.5 eV deeper cliff -> plane gap shrinks by 0.5 eV -> ni_s^2 grows
    # by exp(0.5/V_T) (the SCAPS cliff mechanism)
    assert cliff.ni_s_sq / _PRM.ni_s_sq == pytest.approx(
        math.exp(0.5 / _VT), rel=1e-6)


def test_rate_nogen_clamped_below_plane_equilibrium():
    n_s = p_s = math.sqrt(_PRM.ni_s_sq) * 0.5
    assert plane_rate(n_s, p_s, _PRM, 0.01, 0.01) == 0.0
    assert plane_rate(2.0 * n_s, 4.0 * p_s, _PRM, 0.01, 0.01) > 0.0


def test_solver_detailed_balance_at_equilibrium():
    """Supplies consistent with one flat Fermi level -> R = 0 exactly."""
    # E_F such that plane n_s_eq = Nc*exp(-(chi_s-chi)/...) — construct via
    # the b factors: pick the favourable-edge density n_eq, back out sides.
    n_eq = 1.0e12
    p_eq = _PRM.ni_s_sq / n_eq
    n_L = n_eq / _PRM.bn_L if _PRM.bn_L > 0 else 0.0
    n_R = n_eq / _PRM.bn_R if _PRM.bn_R > 1e-200 else 0.0
    p_L = p_eq / _PRM.bp_L if _PRM.bp_L > 1e-200 else 0.0
    p_R = p_eq / _PRM.bp_R if _PRM.bp_R > 1e-200 else 0.0
    # guard absurd back-projections (blocked side contributes ~nothing)
    n_R = min(n_R, 1e30); p_L = min(p_L, 1e30)
    n_s, p_s, R = solve_plane_densities(n_L, n_R, p_L, p_R, _PRM, 0.01, 0.01)
    assert R == pytest.approx(0.0, abs=S_SUPPLY * (n_eq + p_eq) * 1e-9)


def test_supply_limitation():
    """R saturates at the delivery flux: 4 decades of SRV give far less
    than 4 decades of rate, and R never exceeds the inflow bound."""
    n_L, n_R, p_L, p_R = 1e20, 1e22, 1e21, 1e10
    nb = _PRM.bn_L * n_L + _PRM.bn_R * n_R
    pb = _PRM.bp_L * p_L + _PRM.bp_R * p_R
    # saturation needs v >> S_SUPPLY (2.5e4 m/s): test inside that regime
    _, _, R_mid = solve_plane_densities(n_L, n_R, p_L, p_R, _PRM, 1e6, 1e6)
    _, _, R_hi = solve_plane_densities(n_L, n_R, p_L, p_R, _PRM, 1e8, 1e8)
    assert R_hi <= S_SUPPLY * min(nb, pb) * (1 + 1e-9)
    # in the saturated regime two further decades of SRV change R < 2x
    assert R_hi / max(R_mid, 1e-300) < 2.0


def test_solver_finite_on_extremes():
    for args in ((1e30, 1e30, 1e30, 1e30), (0.0, 0.0, 0.0, 0.0),
                 (1e28, 0.0, 0.0, 1e28), (1e-5, 1e-5, 1e25, 1e25)):
        n_s, p_s, R = solve_plane_densities(*args, _PRM, 10.0, 10.0)
        assert math.isfinite(n_s) and math.isfinite(p_s) and math.isfinite(R)
        assert R >= 0.0


# ------------------------------- plumbing -----------------------------------

def test_flag_default_off_and_yaml(tmp_path):
    import yaml
    from pathlib import Path
    assert load_scaps_yaml(_V2).interface_plane_closure is False
    cfg = yaml.safe_load(Path(_V2).read_text())
    cfg["device"]["interface_plane_closure"] = True
    dst = tmp_path / "pc.yaml"
    dst.write_text(yaml.safe_dump(cfg))
    assert load_scaps_yaml(str(dst)).interface_plane_closure is True


def test_mat_caches_require_parity_configuration():
    """prm tuples built only under dos_band_potentials + reference DOS."""
    base = load_scaps_yaml(_V2)
    _, m_no_dos = _build(dataclasses.replace(base, interface_plane_closure=True))
    assert all(p is None for p in m_no_dos.interface_plane_prm)
    _, m_full = _build(dataclasses.replace(
        base, interface_plane_closure=True, dos_band_potentials=True))
    assert m_full.iface_plane_closure is True
    built = [p for p in m_full.interface_plane_prm if p is not None]
    assert len(built) == 2  # HTL/PVK + PVK/ETL defects


def test_env_var_activates(monkeypatch):
    monkeypatch.setenv("SOLARLAB_IFACE_PLANE", "1")
    _, m = _build(_dos_stack())
    assert m.iface_plane_closure is True


# ------------------------------ integration ---------------------------------

def _interface_dn(stack, x, mat, scale=1.0e22):
    N = len(x)
    n = np.full(N, scale)
    p = np.full(N, scale)
    phi = np.linspace(0.0, 1.0, N)
    dn = np.zeros(N)
    dp = np.zeros(N)
    _apply_interface_recombination(dn, dp, n, p, stack, mat, phi)
    return dn


def test_closure_changes_interface_rate():
    base = _dos_stack()
    x, m_off = _build(base)
    _, m_on = _build(dataclasses.replace(base, interface_plane_closure=True))
    assert not np.allclose(_interface_dn(base, x, m_off),
                           _interface_dn(base, x, m_on))


def test_legacy_preset_unaffected():
    from perovskite_sim.models.config_loader import load_device_from_yaml
    base = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    x, m_off = _build(base)
    _, m_on = _build(dataclasses.replace(base, interface_plane_closure=True))
    np.testing.assert_array_equal(_interface_dn(base, x, m_off),
                                  _interface_dn(base, x, m_on))


def test_base_jv_converges_at_validation_protocol():
    """The stability gate: full J-V on the exact validation grid with the
    closure active. Bracketed V_oc inside the physical window."""
    from perovskite_sim.experiments.jv_sweep import run_jv_sweep
    pc = dataclasses.replace(_dos_stack(), interface_plane_closure=True)
    m = run_jv_sweep(pc, N_grid=30, n_points=40, v_rate=5.0, V_max=1.6,
                     v_max_max_attempts=3).metrics_fwd
    assert m.voc_bracketed
    assert 0.95 < m.V_oc < 1.27
    assert m.J_sc / 10 == pytest.approx(25.7, abs=0.8)
