"""Phase E6.3 — multi-defect + Gaussian + above-VB loader extension.

Extends `scaps_compat.loader` to consume the schema introduced in
`configs/scaps_mirror_v2.yaml`:

  - `bulk_defects: list` — multi-defect parallel SRH per layer.
    Combined via 1/tau_total = sum(1/tau_i); n1/p1 weighted by
    recombination rate (`n1_eff = sum(n1_i / tau_n_i) * tau_n_total`).
  - `E_t_eV_above_vb` — alternative to `E_t_eV_below_cb` for traps
    referenced from the valence-band edge (mutex with `below_cb`).
  - `distribution: single | gaussian` — Gaussian variant carries
    optional `E_char_eV` + `N_peak_cm3` informational fields; the
    loader uses `N_t_cm3` (bulk) or `N_t_cm2` (interface) directly.
  - Unknown defect keys raise loudly so partner-side schema drift
    can never silently disable a defect (Phase E1.5 regression class:
    `calibration_factor` had hidden a 4-order sigma error for months).

Backward compatibility:
  - Singular `bulk_defect: { ... }` continues to work and is treated
    as equivalent to `bulk_defects: [{ ... }]` with one entry.
  - Interface entries without `distribution:` default to `single`
    (legacy behaviour — bit-identical with pre-E6.3 loads).
"""
from __future__ import annotations

import math
import textwrap
from pathlib import Path
from textwrap import dedent

import pytest

from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.scaps_compat.defects import srh_lifetime
from perovskite_sim.sweeps.device_parameter_sweep import (
    cm3_to_m3,
    cms_to_ms,
    srh_n1_p1_from_trap_depth,
)


_BASE_LAYERS = dedent(
    """
    device:
      V_bi: 1.30
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
    """
)

_ETL_LAYER = """\
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


def _write(tmp_path: Path, pvk_extra: str = "", ifc_block: str = "") -> Path:
    """Compose a minimal HTL/PVK/ETL YAML.

    `pvk_extra` is auto-indented 4 spaces to land inside the PVK layer
    fields (the same column as `role:` / `thickness_nm:`). `ifc_block`
    is appended at root level after the ETL layer.
    """
    if pvk_extra:
        pvk_extra = textwrap.indent(dedent(pvk_extra), "    ")
    body = _BASE_LAYERS + pvk_extra + _ETL_LAYER + (ifc_block or "")
    p = tmp_path / "scaps.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return p


# -------------------- bulk_defects (plural) --------------------

def test_bulk_defects_singular_dict_still_loads(tmp_path):
    """Backward compat: `bulk_defect: {...}` must behave as before E6.3."""
    extra = dedent(
        """
        bulk_defect:
          sigma_n_cm2: 1.0e-15
          sigma_p_cm2: 1.0e-15
          N_t_cm3: 1.0e12
          E_t_eV_below_cb: 0.1
        """
    )
    stack = load_scaps_yaml(_write(tmp_path, extra))
    pvk = stack.layers[1]
    # Pre-E6.3 baseline: tau ≈ 1e-4 s for sigma·v_th·N_t = 1e-15·1e7·1e12 (in cgs).
    expected_tau = 1.0 / (1.0e-15 * 1.0e-4 * cms_to_ms(1.0e7) * cm3_to_m3(1.0e12))
    assert math.isclose(pvk.params.tau_n, expected_tau, rel_tol=1e-9)
    assert math.isclose(pvk.params.tau_p, expected_tau, rel_tol=1e-9)


def test_bulk_defects_list_single_entry_matches_singular(tmp_path):
    """`bulk_defects: [one]` produces same tau/n1/p1 as `bulk_defect: one`."""
    extra_plural = dedent(
        """
        bulk_defects:
          - sigma_n_cm2: 1.0e-15
            sigma_p_cm2: 1.0e-15
            N_t_cm3: 1.0e12
            E_t_eV_below_cb: 0.1
        """
    )
    extra_singular = dedent(
        """
        bulk_defect:
          sigma_n_cm2: 1.0e-15
          sigma_p_cm2: 1.0e-15
          N_t_cm3: 1.0e12
          E_t_eV_below_cb: 0.1
        """
    )
    s_plural = load_scaps_yaml(_write(tmp_path / "a", extra_plural))
    s_singular = load_scaps_yaml(_write(tmp_path / "b", extra_singular))
    for attr in ("tau_n", "tau_p", "n1", "p1"):
        a = getattr(s_plural.layers[1].params, attr)
        b = getattr(s_singular.layers[1].params, attr)
        assert math.isclose(a, b, rel_tol=1e-12), f"{attr}: plural={a} singular={b}"


def test_bulk_defects_two_identical_halve_tau(tmp_path):
    """Two parallel defects with identical kinetics ⇒ tau_total = tau_single / 2."""
    extra = dedent(
        """
        bulk_defects:
          - sigma_n_cm2: 1.0e-15
            sigma_p_cm2: 1.0e-15
            N_t_cm3: 1.0e12
            E_t_eV_below_cb: 0.1
          - sigma_n_cm2: 1.0e-15
            sigma_p_cm2: 1.0e-15
            N_t_cm3: 1.0e12
            E_t_eV_above_vb: 0.1
        """
    )
    stack = load_scaps_yaml(_write(tmp_path, extra))
    pvk = stack.layers[1]
    single_tau = 1.0 / (1.0e-15 * 1.0e-4 * cms_to_ms(1.0e7) * cm3_to_m3(1.0e12))
    assert math.isclose(pvk.params.tau_n, single_tau / 2.0, rel_tol=1e-9)
    assert math.isclose(pvk.params.tau_p, single_tau / 2.0, rel_tol=1e-9)


def test_bulk_defects_two_identical_n1_p1_swapped_pair_yields_midgap(tmp_path):
    """CB-Et=0.1-below-CB and VB-Et=0.1-above-VB swap n1/p1 between defects.

    The recombination-weighted average then yields n1_eff = p1_eff = (n1_CB + n1_VB)/2
    (both taus equal). Pinned for the SCAPS PVK case so the v2 mirror is
    well-defined regardless of which defect is listed first.
    """
    extra = dedent(
        """
        bulk_defects:
          - sigma_n_cm2: 1.0e-15
            sigma_p_cm2: 1.0e-15
            N_t_cm3: 1.0e12
            E_t_eV_below_cb: 0.1
          - sigma_n_cm2: 1.0e-15
            sigma_p_cm2: 1.0e-15
            N_t_cm3: 1.0e12
            E_t_eV_above_vb: 0.1
        """
    )
    stack = load_scaps_yaml(_write(tmp_path, extra))
    pvk = stack.layers[1]
    ni = pvk.params.ni
    n1_cb, p1_cb = srh_n1_p1_from_trap_depth(ni, 1.53, 0.1, reference="below_cb")
    n1_vb, p1_vb = srh_n1_p1_from_trap_depth(ni, 1.53, 0.1, reference="above_vb")
    # Both defects have identical tau → uniform weighting.
    n1_expected = (n1_cb + n1_vb) / 2.0
    p1_expected = (p1_cb + p1_vb) / 2.0
    assert math.isclose(pvk.params.n1, n1_expected, rel_tol=1e-9)
    assert math.isclose(pvk.params.p1, p1_expected, rel_tol=1e-9)


def test_bulk_defects_and_singular_mutex(tmp_path):
    extra = dedent(
        """
        bulk_defect:
          sigma_n_cm2: 1.0e-15
          sigma_p_cm2: 1.0e-15
          N_t_cm3: 1.0e12
          E_t_eV_below_cb: 0.1
        bulk_defects:
          - sigma_n_cm2: 1.0e-15
            sigma_p_cm2: 1.0e-15
            N_t_cm3: 1.0e12
            E_t_eV_below_cb: 0.1
        """
    )
    with pytest.raises(ValueError, match="bulk_defect.*bulk_defects"):
        load_scaps_yaml(_write(tmp_path, extra))


def test_bulk_defect_empty_list_falls_back_to_defect_free(tmp_path):
    extra = "bulk_defects: []\n"
    stack = load_scaps_yaml(_write(tmp_path, extra))
    pvk = stack.layers[1]
    assert pvk.params.tau_n == pytest.approx(1.0e-3)
    assert pvk.params.tau_p == pytest.approx(1.0e-3)


# -------------------- above_vb reference --------------------

def test_above_vb_reference_loads(tmp_path):
    extra = dedent(
        """
        bulk_defects:
          - sigma_n_cm2: 1.0e-15
            sigma_p_cm2: 1.0e-15
            N_t_cm3: 1.0e12
            E_t_eV_above_vb: 0.1
        """
    )
    stack = load_scaps_yaml(_write(tmp_path, extra))
    pvk = stack.layers[1]
    n1, p1 = srh_n1_p1_from_trap_depth(pvk.params.ni, 1.53, 0.1, reference="above_vb")
    assert math.isclose(pvk.params.n1, n1, rel_tol=1e-9)
    assert math.isclose(pvk.params.p1, p1, rel_tol=1e-9)


def test_both_below_cb_and_above_vb_rejected(tmp_path):
    extra = dedent(
        """
        bulk_defects:
          - sigma_n_cm2: 1.0e-15
            sigma_p_cm2: 1.0e-15
            N_t_cm3: 1.0e12
            E_t_eV_below_cb: 0.1
            E_t_eV_above_vb: 0.1
        """
    )
    with pytest.raises(ValueError, match="exactly one of"):
        load_scaps_yaml(_write(tmp_path, extra))


def test_neither_below_cb_nor_above_vb_rejected(tmp_path):
    extra = dedent(
        """
        bulk_defects:
          - sigma_n_cm2: 1.0e-15
            sigma_p_cm2: 1.0e-15
            N_t_cm3: 1.0e12
        """
    )
    with pytest.raises(ValueError, match="exactly one of"):
        load_scaps_yaml(_write(tmp_path, extra))


# -------------------- distribution field --------------------

def test_bulk_distribution_single_default(tmp_path):
    """Omitting `distribution` defaults to single (= legacy)."""
    extra = dedent(
        """
        bulk_defects:
          - sigma_n_cm2: 1.0e-15
            sigma_p_cm2: 1.0e-15
            N_t_cm3: 1.0e12
            E_t_eV_below_cb: 0.1
        """
    )
    stack = load_scaps_yaml(_write(tmp_path, extra))
    pvk = stack.layers[1]
    expected_tau = 1.0 / (1.0e-15 * 1.0e-4 * cms_to_ms(1.0e7) * cm3_to_m3(1.0e12))
    assert math.isclose(pvk.params.tau_n, expected_tau, rel_tol=1e-9)


def test_bulk_distribution_gaussian_uses_N_t_directly(tmp_path):
    """`distribution: gaussian` documents but does not recompute N_t.

    PDF column-header units are ambiguous (5.64e8 cm^-3 peak vs 1e12 cm^-3
    total — ratio 5.64e-4 does not match any standard Gaussian
    normalisation). Loader trusts the SCAPS-input value `N_t_cm3` directly;
    `E_char_eV` and `N_peak_cm3` are accepted as informational metadata
    so the YAML remains a transparent partner-facing record.
    """
    extra = dedent(
        """
        bulk_defects:
          - sigma_n_cm2: 1.0e-15
            sigma_p_cm2: 1.0e-15
            N_t_cm3: 1.0e12
            E_t_eV_below_cb: 0.1
            distribution: gaussian
            E_char_eV: 0.1
            N_peak_cm3: 5.64e8
        """
    )
    stack = load_scaps_yaml(_write(tmp_path, extra))
    pvk = stack.layers[1]
    expected_tau = 1.0 / (1.0e-15 * 1.0e-4 * cms_to_ms(1.0e7) * cm3_to_m3(1.0e12))
    assert math.isclose(pvk.params.tau_n, expected_tau, rel_tol=1e-9)


def test_bulk_distribution_unknown_value_rejected(tmp_path):
    extra = dedent(
        """
        bulk_defects:
          - sigma_n_cm2: 1.0e-15
            sigma_p_cm2: 1.0e-15
            N_t_cm3: 1.0e12
            E_t_eV_below_cb: 0.1
            distribution: pancake
        """
    )
    with pytest.raises(ValueError, match="distribution.*pancake"):
        load_scaps_yaml(_write(tmp_path, extra))


# -------------------- loud-fail on unknown keys --------------------

def test_bulk_defect_unknown_key_rejected(tmp_path):
    extra = dedent(
        """
        bulk_defects:
          - sigma_n_cm2: 1.0e-15
            sigma_p_cm2: 1.0e-15
            N_t_cm3: 1.0e12
            E_t_eV_below_cb: 0.1
            calibration_facor: 1.0   # typo of `calibration_factor`
        """
    )
    with pytest.raises(ValueError, match="unknown.*calibration_facor|calibration_facor.*unknown"):
        load_scaps_yaml(_write(tmp_path, extra))


# -------------------- interface distribution field --------------------

def test_interface_distribution_single_default(tmp_path):
    """Pre-E6.3 YAMLs with no `distribution:` key still load."""
    extra = dedent(
        """
        bulk_defects:
          - sigma_n_cm2: 1.0e-15
            sigma_p_cm2: 1.0e-15
            N_t_cm3: 1.0e12
            E_t_eV_below_cb: 0.1
        """
    )
    ifc = dedent(
        """
        interfaces:
          - target: PVK/ETL
            sigma_n_cm2: 1.0e-15
            sigma_p_cm2: 1.0e-15
            N_t_cm2: 1.0e12
            v_th_cm_s: 1.0e7
            E_t_eV_below_cb: 0.6
        """
    )
    stack = load_scaps_yaml(_write(tmp_path, extra, ifc))
    # Two interfaces (HTL/PVK and PVK/ETL); only the second populated.
    assert stack.interface_defects[0] is None
    assert stack.interface_defects[1] is not None
    assert stack.interface_defects[1].E_t_eV == pytest.approx(0.6)


def test_interface_distribution_gaussian_uses_N_t_directly(tmp_path):
    """Gaussian interface defect uses N_t_cm2 as the SCAPS-input value."""
    extra = dedent(
        """
        bulk_defects:
          - sigma_n_cm2: 1.0e-15
            sigma_p_cm2: 1.0e-15
            N_t_cm3: 1.0e12
            E_t_eV_below_cb: 0.1
        """
    )
    ifc = dedent(
        """
        interfaces:
          - target: PVK/ETL
            sigma_n_cm2: 1.0e-19
            sigma_p_cm2: 1.0e-19
            N_t_cm2: 1.0e12
            v_th_cm_s: 1.0e7
            E_t_eV_below_cb: 0.6
            distribution: gaussian
            E_char_eV: 0.1
            N_peak_cm3: 5.64e8
        """
    )
    stack = load_scaps_yaml(_write(tmp_path, extra, ifc))
    v_n, v_p = stack.interfaces[1]
    # v = sigma_si · v_th_si · N_t_areal_si
    expected = 1.0e-19 * 1.0e-4 * cms_to_ms(1.0e7) * (1.0e12 * 1.0e4)
    assert math.isclose(v_n, expected, rel_tol=1e-9)
    assert math.isclose(v_p, expected, rel_tol=1e-9)


def test_interface_unknown_key_rejected(tmp_path):
    extra = dedent(
        """
        bulk_defects:
          - sigma_n_cm2: 1.0e-15
            sigma_p_cm2: 1.0e-15
            N_t_cm3: 1.0e12
            E_t_eV_below_cb: 0.1
        """
    )
    ifc = dedent(
        """
        interfaces:
          - target: PVK/ETL
            sigma_n_cm2: 1.0e-15
            sigma_p_cm2: 1.0e-15
            N_t_cm2: 1.0e12
            v_th_cm_s: 1.0e7
            E_t_eV_below_cb: 0.6
            calibration_facor: 1.0   # typo
        """
    )
    with pytest.raises(ValueError, match="unknown.*calibration_facor|calibration_facor.*unknown"):
        load_scaps_yaml(_write(tmp_path, extra, ifc))


# -------------------- end-to-end: scaps_mirror_v2 loads --------------------

def test_scaps_mirror_v2_loads_with_finite_tau():
    """scaps_mirror_v2.yaml must produce finite PVK tau and two interface defects."""
    path = Path("configs/scaps_mirror_v2.yaml")
    if not path.exists():
        pytest.skip("scaps_mirror_v2.yaml not present in this checkout")
    stack = load_scaps_yaml(path)
    pvk = next(L for L in stack.layers if L.role == "absorber")
    # Two PVK defects in parallel → tau halved vs single-defect baseline.
    single_tau = 1.0 / (1.0e-15 * 1.0e-4 * cms_to_ms(1.0e7) * cm3_to_m3(1.0e12))
    assert math.isclose(pvk.params.tau_n, single_tau / 2.0, rel_tol=1e-9)
    assert math.isclose(pvk.params.tau_p, single_tau / 2.0, rel_tol=1e-9)
    # Both interfaces populated.
    assert stack.interface_defects[0] is not None, "HTL/PVK interface missing"
    assert stack.interface_defects[1] is not None, "PVK/ETL interface missing"
