"""Phase E1.10 — SCAPS YAML loader parses Robin contact S fields.

Mirrors `perovskite_sim.models.config_loader.load_device_from_yaml`'s
support for flat top-level ``device.S_n_left`` / ``S_p_left`` / ``S_n_right``
/ ``S_p_right`` fields so SCAPS configs can declare Robin / selective
outer contacts (Phase 3.3) directly. Without this extension, SCAPS-shape
YAMLs default to ohmic Dirichlet contacts regardless of YAML content
— silent placebo.

Contract:
1. SCAPS YAML without S fields → all four ``DeviceStack.S_*`` are None
   (legacy bit-identical with pre-E1.10).
2. SCAPS YAML with subset of S fields → only those slots populated, the
   rest stay None.
3. ``S = 0`` is the Neumann blocking sentinel (NOT None — None means
   "field absent, use Dirichlet ohmic default").
4. Numeric values pass through unchanged as float m/s.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest


_LAYERS_BLOCK = dedent(
    """
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
).lstrip()


_BASELINE_DEVICE_BLOCK = dedent(
    """
    device:
      V_bi: 1.15
      Phi: 2.5e21
      mode: fast
    """
).lstrip()


def _yaml_with_device_extras(extra_lines: str) -> str:
    """Compose a valid SCAPS YAML with extra device-block lines inserted
    immediately after the required keys. ``extra_lines`` should already
    carry the 2-space indent that puts the keys under ``device:``."""
    return _BASELINE_DEVICE_BLOCK + extra_lines + _LAYERS_BLOCK


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "scaps_minimal.yaml"
    p.write_text(body)
    return p


def test_scaps_yaml_without_S_fields_leaves_robin_fields_none(tmp_path):
    """Legacy bit-identical: omitting the S fields produces a stack with
    all four ``DeviceStack.S_*`` = None — pre-E1.10 behaviour."""
    from perovskite_sim.scaps_compat import load_scaps_yaml

    stack = load_scaps_yaml(_write_yaml(tmp_path, _yaml_with_device_extras("")))
    assert stack.S_n_left is None
    assert stack.S_p_left is None
    assert stack.S_n_right is None
    assert stack.S_p_right is None


def test_scaps_yaml_with_all_four_S_fields_populates_robin(tmp_path):
    """All four S fields specified → all four DeviceStack S fields equal
    the YAML value as a float."""
    from perovskite_sim.scaps_compat import load_scaps_yaml

    extras = (
        "  S_n_left: 1.0e-4\n"
        "  S_p_left: 1.0e5\n"
        "  S_n_right: 1.0e5\n"
        "  S_p_right: 1.0e-4\n"
    )
    stack = load_scaps_yaml(_write_yaml(tmp_path, _yaml_with_device_extras(extras)))
    assert stack.S_n_left == pytest.approx(1.0e-4)
    assert stack.S_p_left == pytest.approx(1.0e5)
    assert stack.S_n_right == pytest.approx(1.0e5)
    assert stack.S_p_right == pytest.approx(1.0e-4)


def test_scaps_yaml_partial_S_fields_only_populates_specified(tmp_path):
    """Only some S fields specified → those populated, the rest stay None."""
    from perovskite_sim.scaps_compat import load_scaps_yaml

    extras = "  S_n_right: 1.0e5\n"
    stack = load_scaps_yaml(_write_yaml(tmp_path, _yaml_with_device_extras(extras)))
    assert stack.S_n_left is None
    assert stack.S_p_left is None
    assert stack.S_n_right == pytest.approx(1.0e5)
    assert stack.S_p_right is None


def test_scaps_yaml_S_zero_is_neumann_blocking_not_absent(tmp_path):
    """``S = 0`` is the perfectly-blocking Neumann sentinel, distinct
    from absent / None. The loader must preserve the zero value rather
    than coerce it to None."""
    from perovskite_sim.scaps_compat import load_scaps_yaml

    extras = "  S_n_left: 0.0\n"
    stack = load_scaps_yaml(_write_yaml(tmp_path, _yaml_with_device_extras(extras)))
    assert stack.S_n_left == 0.0
    assert stack.S_n_left is not None
    assert stack.S_p_left is None  # unspecified stays None
