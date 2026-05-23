"""Tests for the SCAPS-style YAML loader.

The loader reads a SCAPS-parameter YAML (microscopic sigma/N_t/E_t triplets,
N_C/N_V instead of ni, cgs units optional) and returns the same frozen
``DeviceStack`` the existing SolarLab loader produces, so the solver consumes
it without any code path change.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest


_MINIMAL_SCAPS_YAML = dedent(
    """
    device:
      V_bi: 1.15
      Phi: 2.5e21
      mode: fast
    layers:
      - name: HTL_test
        role: HTL
        thickness_nm: 20.0
        E_g_eV: 3.25
        chi_eV: 2.40
        eps_r: 11.75
        mu_n_cm2: 1.0e-3
        mu_p_cm2: 1.0e-3
        N_C_cm3: 2.5e20
        N_V_cm3: 2.5e20
        N_A_cm3: 1.0e18
        N_D_cm3: 0.0
        v_th_cm_s: 1.0e7
      - name: PVK_test
        role: absorber
        thickness_nm: 800.0
        E_g_eV: 1.53
        chi_eV: 3.94
        eps_r: 31.0
        mu_n_cm2: 20.0
        mu_p_cm2: 20.0
        N_C_cm3: 1.0e19
        N_V_cm3: 1.0e19
        N_D_cm3: 1.0e14
        N_A_cm3: 0.0
        v_th_cm_s: 1.0e7
        bulk_defect:
          sigma_n_cm2: 1.0e-15
          sigma_p_cm2: 1.0e-15
          N_t_cm3: 1.0e12
          E_t_eV_below_cb: 0.1
      - name: ETL_test
        role: ETL
        thickness_nm: 25.0
        E_g_eV: 1.9
        chi_eV: 4.10
        eps_r: 4.2
        mu_n_cm2: 1.0e-2
        mu_p_cm2: 1.0e-5
        N_C_cm3: 8.0e19
        N_V_cm3: 8.0e19
        N_D_cm3: 1.0e18
        N_A_cm3: 0.0
        v_th_cm_s: 1.0e7
    """
)


def _write_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "scaps_minimal.yaml"
    p.write_text(_MINIMAL_SCAPS_YAML)
    return p


def test_load_scaps_yaml_returns_devicestack(tmp_path):
    from perovskite_sim.models.device import DeviceStack
    from perovskite_sim.scaps_compat import load_scaps_yaml

    stack = load_scaps_yaml(_write_yaml(tmp_path))
    assert isinstance(stack, DeviceStack)


def test_load_scaps_yaml_assigns_three_electrical_layers_in_order(tmp_path):
    from perovskite_sim.models.device import electrical_layers
    from perovskite_sim.scaps_compat import load_scaps_yaml

    stack = load_scaps_yaml(_write_yaml(tmp_path))
    layers = electrical_layers(stack)
    assert [layer.role for layer in layers] == ["HTL", "absorber", "ETL"]


def test_load_scaps_yaml_converts_cgs_to_si_for_absorber(tmp_path):
    from perovskite_sim.models.device import electrical_layers
    from perovskite_sim.scaps_compat import load_scaps_yaml

    stack = load_scaps_yaml(_write_yaml(tmp_path))
    absorber = next(L for L in electrical_layers(stack) if L.role == "absorber")
    # thickness 800 nm = 8e-7 m
    assert absorber.thickness == pytest.approx(8.0e-7)
    # mu 20 cm^2/V s = 2e-3 m^2/V s
    assert absorber.params.mu_n == pytest.approx(2.0e-3)
    # PVK SCAPS bulk defect tau = 1 / (sigma * v_th * N_t)
    # SI: sigma = 1e-19 m^2, v_th = 1e5 m/s, N_t = 1e18 m^-3 -> tau = 1e-4 s
    assert absorber.params.tau_n == pytest.approx(1.0e-4, rel=1.0e-3)


def test_load_scaps_yaml_raises_when_required_field_missing(tmp_path):
    from perovskite_sim.scaps_compat import load_scaps_yaml

    broken = "device:\n  V_bi: 1.0\nlayers: []\n"
    path = tmp_path / "broken.yaml"
    path.write_text(broken)
    with pytest.raises(ValueError):
        load_scaps_yaml(path)
