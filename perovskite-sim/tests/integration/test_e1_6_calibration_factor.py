"""Phase E1.6 (Option B-2) — explicit InterfaceDefect.calibration_factor.

Phase A + A2 (2026-05-26) probed three face-density formulations on
scaps_mirror.yaml and concluded:
- SG-Selberherr face density (originally planned Phase B) loses cliff
  direction because p at the heavily-doped ETL side stays at machine
  epsilon under both Dirichlet and Robin contacts.
- E1.5 cross-carrier sampling IS the right cliff-direction formula.
  Its magnitude error is a separate, smaller problem.

Option B-2 keeps E1.5 cross-carrier sampling and adds an explicit
per-interface attenuation factor that multiplies v_n, v_p before the
SRH rate computation. The factor lives on the ``InterfaceDefect``
dataclass so partner sees the calibration ratio in the YAML (not
hidden in a validation-script constant).

Contract:
1. ``InterfaceDefect`` exposes a ``calibration_factor: float = 1.0``
   field (default = legacy bit-identical with pre-E1.6 behaviour).
2. Solver multiplies v_n, v_p by ``calibration_factor`` at the
   interface SRH evaluation. Mathematically equivalent to scaling N_t
   areal density, so a YAML with ``N_t_cm2: 1e13 + calibration_factor: 1e-5``
   produces the same V_oc as ``N_t_cm2: 1e8 + calibration_factor: 1.0``.
3. SCAPS YAML loader parses the field from each ``interfaces:`` entry
   (default 1.0 if absent).
4. ``backend/main.py:stack_from_dict`` parses the field from inline
   device dicts (frontend live-editor round-trip).
"""
from __future__ import annotations

import dataclasses

import pytest

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.models.device import InterfaceDefect
from perovskite_sim.scaps_compat import load_scaps_yaml


def test_interface_defect_carries_calibration_factor_default_1():
    """Default value of ``calibration_factor`` is 1.0 — legacy bit-identical."""
    d = InterfaceDefect(E_t_eV=0.6)
    assert d.calibration_factor == 1.0


def test_interface_defect_calibration_factor_settable():
    """Can construct with explicit calibration_factor."""
    d = InterfaceDefect(E_t_eV=0.6, calibration_factor=1.0e-5)
    assert d.calibration_factor == pytest.approx(1.0e-5)


def test_n_t_calibration_factor_equivalence_on_scaps_mirror():
    """SCAPS-mirror with N_t_cm2=1e13 + calibration_factor=1e-5 produces
    the same V_oc as N_t_cm2=1e8 + calibration_factor=1.0 because the
    solver multiplies v_n, v_p by the calibration factor (mathematically
    equivalent to scaling N_t). This is the key acceptance check for
    Option B-2: SCAPS direct N_t values become useable as long as the
    calibration factor is supplied."""
    # Current scaps_mirror.yaml: N_t_cm2=1e8, calibration_factor=1.0 (default)
    baseline_stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    baseline = run_jv_sweep(
        baseline_stack, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6,
    )
    voc_baseline = baseline.metrics_fwd.V_oc

    # Equivalent stack: N_t_cm2=1e13 + calibration_factor=1e-5 → same v_eff.
    # Build via dataclasses.replace so we do not touch the YAML in the test.
    n_iface = len(baseline_stack.layers) - 1
    interfaces = list(baseline_stack.interfaces)
    defects = list(baseline_stack.interface_defects)
    # SRV currently = σ · v_th · N_t_cm2 · 1e-2 = 1e-15 · 1e7 · 1e8 · 1e-2 = 1e-2 m/s.
    # New equivalent: SRV_direct = 1e3 m/s · calibration_factor 1e-5 = 1e-2 m/s.
    # Same effective velocity → same V_oc.
    interfaces[-1] = (1.0e3, 1.0e3)  # SCAPS direct SRV
    defects[-1] = InterfaceDefect(
        E_t_eV=defects[-1].E_t_eV, calibration_factor=1.0e-5,
    )
    equiv_stack = dataclasses.replace(
        baseline_stack,
        interfaces=tuple(interfaces),
        interface_defects=tuple(defects),
    )
    equiv = run_jv_sweep(
        equiv_stack, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6,
    )
    voc_equiv = equiv.metrics_fwd.V_oc

    assert voc_equiv == pytest.approx(voc_baseline, abs=1.0e-3), (
        f"V_oc mismatch: baseline={voc_baseline:.4f} V, "
        f"equiv={voc_equiv:.4f} V (expected within 1 mV — same effective SRV)"
    )


def test_scaps_mirror_baseline_unchanged_post_e1_6_migration():
    """Migrating scaps_mirror.yaml from ``N_t_cm2: 1e8`` (empirical)
    to ``N_t_cm2: 1e13 + calibration_factor: 1e-5`` (SCAPS direct +
    explicit attenuation) must produce a V_oc inside the SCAPS window
    [1.05, 1.25] V. The two declarations are mathematically equivalent
    via ``SRV_eff = σ · v_th · N_t · 1e-2 · cf``, so the baseline
    physics is unchanged — only the partner-facing data model presentation."""
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    # Post-E1.6 explicit value (was implicit 1.0 default pre-E1.6).
    # Aligned to SCAPS PDF baseline N_t=1e12; cf=1e-4 produces the same
    # effective SRV (1e-2 m/s) as the pre-E1.6 empirical N_t=1e8 + cf=1.0.
    assert stack.interface_defects[-1].calibration_factor == pytest.approx(1.0e-4)
    r = run_jv_sweep(stack, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
    # Window pinned by test_scaps_mirror_baseline.py
    assert 1.05 <= r.metrics_fwd.V_oc <= 1.25


def test_scaps_yaml_parses_calibration_factor(tmp_path):
    """SCAPS YAML loader picks up the ``calibration_factor`` field from
    each ``interfaces:`` entry (absent → 1.0)."""
    from textwrap import dedent

    body = dedent(
        """
        device:
          V_bi: 1.15
          Phi: 2.5e21
          mode: fast
        layers:
          - name: HTL
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
          - name: PVK
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
          - name: ETL
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
        interfaces:
          - target: PVK/ETL
            sigma_n_cm2: 1.0e-15
            sigma_p_cm2: 1.0e-15
            N_t_cm2: 1.0e13
            v_th_cm_s: 1.0e7
            E_t_eV_below_cb: 0.6
            calibration_factor: 1.0e-5
        """
    )
    p = tmp_path / "scaps_calib.yaml"
    p.write_text(body)
    stack = load_scaps_yaml(p)
    assert stack.interface_defects[-1].calibration_factor == pytest.approx(1.0e-5)


def test_backend_stack_from_dict_parses_calibration_factor():
    """``backend/main.py:stack_from_dict`` plumbs ``calibration_factor``
    from the inline device dict (live editor / API caller round-trip)."""
    from backend.main import stack_from_dict

    cfg = {
        "device": {
            "V_bi": 1.1,
            "Phi": 2.5e21,
            "mode": "full",
            "interface_defects": [
                None,
                {
                    "sigma_n_cm2": 1.0e-15,
                    "sigma_p_cm2": 1.0e-15,
                    "N_t_cm2": 1.0e13,
                    "v_th_cm_s": 1.0e7,
                    "E_t_eV_below_cb": 0.6,
                    "calibration_factor": 1.0e-5,
                },
            ],
        },
        "layers": [
            _minimal_layer("HTL", "HTL", chi=2.4, Eg=3.25),
            _minimal_layer("PVK", "absorber", chi=3.94, Eg=1.53),
            _minimal_layer("ETL", "ETL", chi=4.10, Eg=1.9),
        ],
    }
    stack = stack_from_dict(cfg)
    assert stack.interface_defects[-1].calibration_factor == pytest.approx(1.0e-5)


def _minimal_layer(name: str, role: str, chi: float, Eg: float) -> dict:
    return {
        "name": name, "role": role,
        "thickness": 1.0e-7, "eps_r": 10.0,
        "mu_n": 1.0e-4, "mu_p": 1.0e-4,
        "ni": 1.0e10, "N_D": 1.0e22, "N_A": 0.0,
        "D_ion": 0.0, "P_lim": 1.0e30, "P0": 0.0,
        "tau_n": 1.0e-6, "tau_p": 1.0e-6,
        "n1": 1.0e10, "p1": 1.0e10,
        "B_rad": 0.0, "C_n": 0.0, "C_p": 0.0, "alpha": 0.0,
        "chi": chi, "Eg": Eg,
    }
