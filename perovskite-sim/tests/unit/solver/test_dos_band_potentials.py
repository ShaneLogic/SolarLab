"""Effective-DOS band potentials for heterojunction transport (2026-06).

The Scharfetter-Gummel flux drives carriers with phi_n = phi + chi and
phi_p = phi + chi + Eg (continuity.py). With Boltzmann statistics the correct
heterostructure form also carries the effective-DOS terms V_T*ln(N_C) /
V_T*ln(N_V); omitting them imposes a spurious quasi-Fermi-level step of
kT*ln(DOS ratio) at every DOS-contrast heterojunction (measured 84 meV at
HTL/PVK = kT*ln 25 and 53 meV at PVK/ETL = kT*ln 8 on scaps_mirror_v2 — the
root cause of the SolarLab-vs-SCAPS V_oc gap).

The fix folds the DOS corrections into the cached chi/Eg ARRAYS used by the
flux + TE (never into ni / n1 / p1 / boundary densities), gated by the
default-OFF ``DeviceStack.dos_band_potentials`` flag (or SOLARLAB_DOS_BAND=1),
with the absorber layer as the reference (only ratios matter physically).
Legacy configs carry no Nc300/Nv300 data, so the flag is a no-op there.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import build_material_arrays

_V2 = "configs/scaps_mirror_v2.yaml"
_V_T = 0.025852  # 300 K

# scaps_mirror_v2 DOS (m^-3): HTL 2.5e26, PVK 1e25, ETL 8e25 (N_C == N_V per layer)
_RATIO_HTL = 25.0
_RATIO_ETL = 8.0


def _build(stack):
    elec = electrical_layers(stack)
    layers_grid = [Layer(thickness=L.thickness, N=30 // len(elec)) for L in elec]
    x = multilayer_grid(layers_grid)
    return x, build_material_arrays(x, stack)


def _node_in_layer(x, stack, name):
    """Any node index strictly inside the named electrical layer."""
    elec = electrical_layers(stack)
    off = 0.0
    for L in elec:
        if name in str(L.name):
            mid = off + L.thickness / 2.0
            return int(np.argmin(np.abs(x - mid)))
        off += L.thickness
    raise AssertionError(f"layer {name} not found")


# ----------------------------- loader -----------------------------

def test_scaps_loader_populates_dos_fields():
    stack = load_scaps_yaml(_V2)
    for L in electrical_layers(stack):
        assert L.params.Nc300 is not None and L.params.Nc300 > 0
        assert L.params.Nv300 is not None and L.params.Nv300 > 0


def test_devicestack_dos_flag_default_off():
    stack = load_scaps_yaml(_V2)
    assert stack.dos_band_potentials is False


def test_scaps_yaml_dos_key_roundtrip(tmp_path):
    cfg = yaml.safe_load(Path(_V2).read_text())
    cfg["device"]["dos_band_potentials"] = True
    dst = tmp_path / "dos.yaml"
    dst.write_text(yaml.safe_dump(cfg))
    assert load_scaps_yaml(str(dst)).dos_band_potentials is True


# ----------------------------- build -----------------------------

def test_flag_off_chi_eg_unchanged():
    stack = load_scaps_yaml(_V2)
    x, mat = _build(stack)
    i_etl = _node_in_layer(x, stack, "ETL")
    i_htl = _node_in_layer(x, stack, "HTL")
    assert mat.chi[i_etl] == pytest.approx(4.1)
    assert mat.chi[i_htl] == pytest.approx(2.4)


def test_flag_on_shifts_chi_eg_by_dos_ratios():
    stack = dataclasses.replace(load_scaps_yaml(_V2), dos_band_potentials=True)
    x, mat = _build(stack)
    i_htl = _node_in_layer(x, stack, "HTL")
    i_pvk = _node_in_layer(x, stack, "PVK")
    i_etl = _node_in_layer(x, stack, "ETL")
    # absorber is the reference: untouched
    assert mat.chi[i_pvk] == pytest.approx(3.94)
    assert mat.Eg[i_pvk] == pytest.approx(1.53)
    # chi_eff = chi + V_T*ln(N_C/N_C_ref)
    assert mat.chi[i_htl] == pytest.approx(2.4 + _V_T * math.log(_RATIO_HTL), abs=1e-4)
    assert mat.chi[i_etl] == pytest.approx(4.1 + _V_T * math.log(_RATIO_ETL), abs=1e-4)
    # (chi+Eg)_eff = chi + Eg - V_T*ln(N_V/N_V_ref)
    assert mat.chi[i_htl] + mat.Eg[i_htl] == pytest.approx(
        5.65 - _V_T * math.log(_RATIO_HTL), abs=1e-4)
    assert mat.chi[i_etl] + mat.Eg[i_etl] == pytest.approx(
        6.0 - _V_T * math.log(_RATIO_ETL), abs=1e-4)


def test_flag_on_leaves_ni_and_boundaries_unchanged():
    base = load_scaps_yaml(_V2)
    x, m_off = _build(base)
    _, m_on = _build(dataclasses.replace(base, dos_band_potentials=True))
    np.testing.assert_array_equal(m_off.ni_sq, m_on.ni_sq)
    np.testing.assert_array_equal(m_off.n1, m_on.n1)
    np.testing.assert_array_equal(m_off.p1, m_on.p1)
    assert m_off.n_L == m_on.n_L and m_off.p_L == m_on.p_L
    assert m_off.n_R == m_on.n_R and m_off.p_R == m_on.p_R


def test_env_var_enables_dos(monkeypatch):
    monkeypatch.setenv("SOLARLAB_DOS_BAND", "1")
    stack = load_scaps_yaml(_V2)  # flag stays False; env drives it
    x, mat = _build(stack)
    i_etl = _node_in_layer(x, stack, "ETL")
    assert mat.chi[i_etl] == pytest.approx(4.1 + _V_T * math.log(_RATIO_ETL), abs=1e-4)


def test_flag_noop_without_dos_data():
    """Legacy configs (no Nc300/Nv300) are bit-identical under the flag."""
    base = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    x, m_off = _build(base)
    _, m_on = _build(dataclasses.replace(base, dos_band_potentials=True))
    np.testing.assert_array_equal(m_off.chi, m_on.chi)
    np.testing.assert_array_equal(m_off.Eg, m_on.Eg)
