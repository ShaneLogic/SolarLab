"""Phase E1.8 — ``backend/main.py:stack_from_dict`` parses ``device.interface_defects``.

The frontend live editor lets users declare per-interface SCAPS-style
defect fields (σ, v_th, N_t, E_t) under FULL tier. The dict the frontend
sends as ``device:`` in /api/jobs payloads must round-trip through
``stack_from_dict`` so the resulting ``DeviceStack`` carries:

  - ``interfaces[k] = (srv, srv)`` computed via σ·v_th·N_t_areal (SCAPS
    kinetic identity, cgs→SI conversion identical to scaps_compat loader)
  - ``interface_defects[k] = InterfaceDefect(E_t_eV=…)`` for the
    E_t-aware n1/p1 derivation in solver/mol.py

CLAUDE.md flags this as a recurring bug class — UI fields that do not
survive the backend boundary are silent placebos. The integration test
guards against that regression by checking BOTH ``stack.interfaces[k]``
and ``stack.interface_defects[k]`` after a round-trip.

Contract:
1. ``device.interface_defects`` absent → ``stack.interface_defects = ()``
   (legacy bit-identical with pre-E1.8 behaviour).
2. ``device.interface_defects = [null, null]`` (all slots None) →
   ``stack.interface_defects[k] is None`` for every k; ``stack.interfaces``
   unchanged from legacy parsing.
3. Populated slot → ``stack.interfaces[k]`` SRV computed,
   ``stack.interface_defects[k]`` is an ``InterfaceDefect`` instance with
   matching ``E_t_eV``.
4. Mixed (some slots populated, others null) → independent per-slot
   handling.
"""
from __future__ import annotations

import pytest

from perovskite_sim.models.device import InterfaceDefect


def _minimal_layer_dict(name: str, role: str, chi: float, Eg: float) -> dict:
    """Minimal LayerConfig-shape dict matching `MaterialParams` fields."""
    return {
        "name": name,
        "role": role,
        "thickness": 1.0e-7,
        "eps_r": 10.0,
        "mu_n": 1.0e-4,
        "mu_p": 1.0e-4,
        "ni": 1.0e10,
        "N_D": 1.0e22,
        "N_A": 0.0,
        "D_ion": 0.0,
        "P_lim": 1.0e30,
        "P0": 0.0,
        "tau_n": 1.0e-6,
        "tau_p": 1.0e-6,
        "n1": 1.0e10,
        "p1": 1.0e10,
        "B_rad": 0.0,
        "C_n": 0.0,
        "C_p": 0.0,
        "alpha": 0.0,
        "chi": chi,
        "Eg": Eg,
    }


def _three_layer_cfg(extra_device: dict | None = None) -> dict:
    """3-layer HTL/PVK/ETL stack with two electrical interfaces (k=0 HTL/PVK,
    k=1 PVK/ETL) for testing interface_defects parsing."""
    return {
        "device": {
            "V_bi": 1.1,
            "Phi": 2.5e21,
            "mode": "full",
            **(extra_device or {}),
        },
        "layers": [
            _minimal_layer_dict("HTL", "HTL", chi=2.4, Eg=3.25),
            _minimal_layer_dict("PVK", "absorber", chi=3.94, Eg=1.53),
            _minimal_layer_dict("ETL", "ETL", chi=4.10, Eg=1.9),
        ],
    }


def test_absent_interface_defects_field_round_trips_as_empty_tuple():
    """No ``interface_defects`` key on the device dict → legacy bit-identical
    state (``stack.interface_defects = ()``)."""
    from backend.main import stack_from_dict

    stack = stack_from_dict(_three_layer_cfg())
    assert stack.interface_defects == ()


def test_all_null_interface_defects_yields_none_per_slot():
    """``interface_defects: [null, null]`` → tuple of two ``None`` values
    aligned with the two electrical interfaces."""
    from backend.main import stack_from_dict

    cfg = _three_layer_cfg({"interface_defects": [None, None]})
    stack = stack_from_dict(cfg)
    assert stack.interface_defects == (None, None)


def test_populated_pvk_etl_defect_sets_interfaces_srv_and_defect():
    """SCAPS-style PVK/ETL defect on the second interface (k=1) is
    converted to SRV via σ·v_th·N_t_areal and stored as InterfaceDefect."""
    from backend.main import stack_from_dict

    pvk_etl_defect = {
        "sigma_n_cm2": 1.0e-15,
        "sigma_p_cm2": 1.0e-15,
        "N_t_cm2": 1.0e8,           # areal
        "v_th_cm_s": 1.0e7,
        "E_t_eV_below_cb": 0.6,
    }
    cfg = _three_layer_cfg({"interface_defects": [None, pvk_etl_defect]})
    stack = stack_from_dict(cfg)

    # SRV [m/s] = σ_cm2 · v_th_cm_s · N_t_cm2 · 1e-2
    #          = 1e-15 · 1e7 · 1e8 · 1e-2 = 1e-2
    expected_srv = 1.0e-2
    assert stack.interfaces[0] == (0.0, 0.0)  # HTL/PVK untouched
    assert stack.interfaces[1][0] == pytest.approx(expected_srv, rel=1e-12)
    assert stack.interfaces[1][1] == pytest.approx(expected_srv, rel=1e-12)
    assert stack.interface_defects[0] is None
    assert isinstance(stack.interface_defects[1], InterfaceDefect)
    assert stack.interface_defects[1].E_t_eV == pytest.approx(0.6)


def test_legacy_interfaces_array_still_works_when_interface_defects_absent():
    """The legacy ``interfaces: [[v_n, v_p], ...]`` SRV-pairs schema (used
    by the workstation Interface Recombination panel that pre-dates E1.5)
    must keep working when ``interface_defects`` is absent. Both schemas
    coexist; ``interface_defects`` takes precedence per-slot when both
    declare data on the same k."""
    from backend.main import stack_from_dict

    cfg = _three_layer_cfg({"interfaces": [[0.0, 0.0], [1.0e3, 1.0e3]]})
    stack = stack_from_dict(cfg)
    assert stack.interfaces[1] == (1.0e3, 1.0e3)
    assert stack.interface_defects == ()


def test_interface_defects_overrides_legacy_interfaces_when_both_present():
    """When both ``interfaces`` (SRV pairs) and ``interface_defects`` (SCAPS
    σ·v_th·N_t) declare data on the same interface index, the new
    ``interface_defects`` schema wins. Same SCAPS PVK/ETL defect injected
    on top of a legacy SRV=1e3 → the SCAPS-computed SRV=0.01 replaces
    the legacy 1e3."""
    from backend.main import stack_from_dict

    pvk_etl_defect = {
        "sigma_n_cm2": 1.0e-15,
        "sigma_p_cm2": 1.0e-15,
        "N_t_cm2": 1.0e8,
        "v_th_cm_s": 1.0e7,
        "E_t_eV_below_cb": 0.6,
    }
    cfg = _three_layer_cfg({
        "interfaces": [[0.0, 0.0], [1.0e3, 1.0e3]],
        "interface_defects": [None, pvk_etl_defect],
    })
    stack = stack_from_dict(cfg)
    assert stack.interfaces[1][0] == pytest.approx(1.0e-2, rel=1e-12)
    assert stack.interface_defects[1].E_t_eV == pytest.approx(0.6)


def test_asymmetric_sigma_n_vs_sigma_p_uses_each_for_its_carrier():
    """SCAPS schema separates σ_n and σ_p so v_n and v_p can differ
    per defect. The SRV pair on the heterojunction interface must
    reflect both: v_n = σ_n·v_th·N_t; v_p = σ_p·v_th·N_t."""
    from backend.main import stack_from_dict

    defect = {
        "sigma_n_cm2": 1.0e-15,
        "sigma_p_cm2": 5.0e-16,    # different from σ_n
        "N_t_cm2": 1.0e9,
        "v_th_cm_s": 1.0e7,
        "E_t_eV_below_cb": 0.4,
    }
    cfg = _three_layer_cfg({"interface_defects": [None, defect]})
    stack = stack_from_dict(cfg)
    v_n = 1.0e-15 * 1.0e7 * 1.0e9 * 1.0e-2
    v_p = 5.0e-16 * 1.0e7 * 1.0e9 * 1.0e-2
    assert stack.interfaces[1][0] == pytest.approx(v_n, rel=1e-12)
    assert stack.interfaces[1][1] == pytest.approx(v_p, rel=1e-12)
