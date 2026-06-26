"""Plumbing + bit-identical contract for continuous bandgap grading.

The flag is default-OFF and a flat grade (endpoints equal) must reproduce the
ungraded scalar-broadcast MaterialArrays byte-for-byte; LEGACY tier must force
grading off even when a layer declares a real back endpoint.
"""
import dataclasses

import numpy as np
import pytest

from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.models.device import DeviceStack, LayerSpec, electrical_layers
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.solver.mol import build_material_arrays


def _mat(**kw) -> MaterialParams:
    base = dict(
        eps_r=10.0, mu_n=1e-3, mu_p=1e-3, D_ion=0.0, P_lim=1e24, P0=1e24,
        ni=1e12, tau_n=1e-7, tau_p=1e-7, n1=1e12, p1=1e12, B_rad=1e-17,
        C_n=0.0, C_p=0.0, alpha=1e7, N_A=0.0, N_D=0.0,
    )
    base.update(kw)
    return MaterialParams(**base)


def _stack(absorber_params: MaterialParams, *, band_grading: bool, mode: str = "full") -> DeviceStack:
    htl = _mat(chi=2.0, Eg=3.0, N_A=1e24)
    etl = _mat(chi=4.0, Eg=3.0, N_D=1e24)
    layers = (
        LayerSpec("HTL", 50e-9, htl, "HTL"),
        LayerSpec("ABS", 400e-9, absorber_params, "absorber"),
        LayerSpec("ETL", 50e-9, etl, "ETL"),
    )
    return DeviceStack(layers=layers, V_bi=1.0, band_grading=band_grading, mode=mode)


def _grid(stack: DeviceStack, N: int = 20) -> np.ndarray:
    elec = electrical_layers(stack)
    return multilayer_grid([Layer(l.thickness, N) for l in elec], alpha=3.0)


# --- bit-identical contracts ------------------------------------------------

def test_flag_off_is_default():
    assert DeviceStack(layers=(LayerSpec("a", 1e-7, _mat(), "absorber"),)).band_grading is False


def test_flat_grade_byte_identical_to_uniform():
    """Eg_back==Eg, chi_back==chi, bowing=0, flag ON → identical arrays."""
    abs_flat = _mat(chi=3.9, Eg=1.6, Eg_back=1.6, chi_back=3.9, grading_bowing=0.0)
    abs_plain = _mat(chi=3.9, Eg=1.6)
    st_on = _stack(abs_flat, band_grading=True)
    st_off = _stack(abs_plain, band_grading=False)
    x = _grid(st_on)
    m_on = build_material_arrays(x, st_on)
    m_off = build_material_arrays(x, st_off)
    for field in ("chi", "Eg", "ni_sq", "n1", "p1", "B_rad"):
        a = getattr(m_on, field)
        b = getattr(m_off, field)
        assert np.array_equal(a, b), f"{field} not byte-identical under flat grade"


def test_legacy_tier_forces_grading_off():
    """A real back endpoint under LEGACY must still produce uniform arrays."""
    abs_graded = _mat(chi=3.9, Eg=1.6, Eg_back=1.9)
    abs_plain = _mat(chi=3.9, Eg=1.6)
    st_graded = _stack(abs_graded, band_grading=True, mode="legacy")
    st_plain = _stack(abs_plain, band_grading=False, mode="legacy")
    x = _grid(st_graded)
    m_graded = build_material_arrays(x, st_graded)
    m_plain = build_material_arrays(x, st_plain)
    for field in ("chi", "Eg", "ni_sq", "n1", "p1"):
        assert np.array_equal(getattr(m_graded, field), getattr(m_plain, field)), field


def test_flag_off_ignores_back_endpoint():
    """Back endpoint set but flag OFF → uniform (flag is the master gate)."""
    abs_graded = _mat(chi=3.9, Eg=1.6, Eg_back=1.9)
    abs_plain = _mat(chi=3.9, Eg=1.6)
    x = _grid(_stack(abs_plain, band_grading=False))
    m_graded_off = build_material_arrays(x, _stack(abs_graded, band_grading=False))
    m_plain = build_material_arrays(x, _stack(abs_plain, band_grading=False))
    assert np.array_equal(m_graded_off.Eg, m_plain.Eg)


# --- real grade behaviour ---------------------------------------------------

def test_real_linear_grade_monotone_and_detailed_balance():
    abs_graded = _mat(chi=3.9, Eg=1.6, Eg_back=1.9, grading_profile="linear")
    st = _stack(abs_graded, band_grading=True)
    x = _grid(st)
    m = build_material_arrays(x, st)
    # Absorber nodes, dropping the trailing shared interface node which the
    # build assigns to the downstream ETL (last-writer-wins per-node fill).
    idx = np.where((x >= 50e-9 - 1e-12) & (x <= 450e-9 + 1e-12))[0][:-1]
    Eg_abs = m.Eg[idx]
    # No Nc300/Nv300 → DOS fold is a no-op, so Eg is the pure graded gap.
    assert Eg_abs[0] == pytest.approx(1.6, abs=1e-9)   # absorber front face
    assert Eg_abs[-1] < 1.9 and Eg_abs[-1] > 1.6        # last interior node, below back endpoint
    assert np.all(np.diff(Eg_abs) > 0)                  # widening front→back
    # ni²(x) decreasing as gap widens; n1·p1 == ni²(x) per node.
    ni_sq_abs = m.ni_sq[idx]
    assert np.all(np.diff(ni_sq_abs) < 0)
    assert np.allclose(m.n1[idx] * m.p1[idx], ni_sq_abs, rtol=1e-10)


def test_back_to_front_direction_flips_profile():
    fwd = _mat(chi=3.9, Eg=1.6, Eg_back=1.9, grading_direction="front_to_back")
    rev = _mat(chi=3.9, Eg=1.6, Eg_back=1.9, grading_direction="back_to_front")
    x = _grid(_stack(fwd, band_grading=True))
    # Absorber nodes minus the trailing ETL-owned interface node.
    idx = np.where((x >= 50e-9 - 1e-12) & (x <= 450e-9 + 1e-12))[0][:-1]
    m_fwd = build_material_arrays(x, _stack(fwd, band_grading=True))
    m_rev = build_material_arrays(x, _stack(rev, band_grading=True))
    # forward rises front→back; reverse falls
    assert m_fwd.Eg[idx][0] < m_fwd.Eg[idx][-1]
    assert m_rev.Eg[idx][0] > m_rev.Eg[idx][-1]
