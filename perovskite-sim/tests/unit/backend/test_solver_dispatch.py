"""Backend wiring for the steady-state Newton solver + inline-device physics flags.

Covers two fixes:
  1. stack_from_dict (the inline ``device:`` path the frontend always uses)
     must plumb the same five physics flags the YAML loader does, instead of
     silently dropping them.
  2. A ``solver`` selector routes the J-V job to the steady-state Newton
     driver (run_jv_sweep_ss) when requested; default stays the transient
     Radau sweep, bit-identical to before.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import backend.main as bm


def _min_cfg() -> dict:
    """Smallest cfg dict stack_from_dict accepts: one absorber layer."""
    return {
        "device": {"mode": "full"},
        "layers": [
            {
                "name": "ABS",
                "role": "absorber",
                "thickness": 5e-7,
                "eps_r": 10.0,
                "mu_n": 1e-4,
                "mu_p": 1e-4,
                "D_ion": 0.0,
                "P_lim": 1e26,
                "P0": 1e24,
                "ni": 1e15,
                "tau_n": 1e-9,
                "tau_p": 1e-9,
                "n1": 1e15,
                "p1": 1e15,
                "B_rad": 0.0,
                "C_n": 0.0,
                "C_p": 0.0,
                "alpha": 0.0,
                "N_A": 0.0,
                "N_D": 0.0,
            }
        ],
    }


# --- fix #1: inline-device flag plumbing ------------------------------------

def test_stack_from_dict_plumbs_physics_flags():
    cfg = _min_cfg()
    cfg["device"].update(
        dos_band_potentials=True,
        het_recomb_despike=0.53,
        flat_band_contacts=True,
        interface_plane_closure=True,
        interface_plane_projection=True,
    )
    s = bm.stack_from_dict(cfg)
    assert s.dos_band_potentials is True
    assert s.het_recomb_despike == 0.53
    assert s.flat_band_contacts is True
    assert s.interface_plane_closure is True
    assert s.interface_plane_projection is True


def test_stack_from_dict_flag_defaults():
    """Absent flags → defaults. dos_band_potentials defaults ON (2026-06,
    correct heterojunction transport; a no-op without per-layer DOS data, so
    inline-device submissions stay bit-identical). The rest default off / 0.0."""
    s = bm.stack_from_dict(_min_cfg())
    assert s.dos_band_potentials is True
    assert s.het_recomb_despike == 0.0
    assert s.flat_band_contacts is False
    assert s.interface_plane_closure is False
    assert s.interface_plane_projection is False


def test_stack_from_dict_flag_string_truthiness():
    """YAML/JSON may carry the flag as a string; mirror the loader's parsing."""
    cfg = _min_cfg()
    cfg["device"]["dos_band_potentials"] = "on"
    assert bm.stack_from_dict(cfg).dos_band_potentials is True


# --- fix #2: solver dispatch ------------------------------------------------

def test_dispatch_steady_state_maps_to_hysteresis_free_jvresult(monkeypatch):
    captured = {}

    def fake_ss(stack, *, N_grid, n_points, V_max, illuminated,
                iface_states, progress=None):
        captured.update(
            N_grid=N_grid, n_points=n_points, V_max=V_max,
            illuminated=illuminated, iface_states=iface_states,
        )
        return SimpleNamespace(
            V=np.array([0.0, 0.5, 1.0]),
            J=np.array([200.0, 180.0, 0.0]),
            metrics="MET",
        )

    monkeypatch.setattr(bm, "run_jv_sweep_ss", fake_ss)
    res = bm._run_jv_dispatch(
        stack=None, N_grid=30, n_points=20, v_rate=1.0, V_max=1.3,
        illuminated=True, solver="steady_state", iface_states=True,
    )
    # The steady-state limit has no hysteresis: forward == reverse, index 0.
    assert res.hysteresis_index == 0.0
    assert np.array_equal(res.V_fwd, res.V_rev)
    assert np.array_equal(res.J_fwd, res.J_rev)
    assert res.metrics_fwd == "MET" and res.metrics_rev == "MET"
    assert captured["iface_states"] is True
    assert captured["V_max"] == 1.3


def test_dispatch_steady_state_defaults_v_max_when_none(monkeypatch):
    captured = {}

    def fake_ss(stack, *, N_grid, n_points, V_max, illuminated,
                iface_states, progress=None):
        captured["V_max"] = V_max
        return SimpleNamespace(V=np.zeros(2), J=np.zeros(2), metrics="MET")

    monkeypatch.setattr(bm, "run_jv_sweep_ss", fake_ss)
    bm._run_jv_dispatch(
        stack=None, N_grid=30, n_points=20, v_rate=1.0, V_max=None,
        illuminated=True, solver="steady_state", iface_states=False,
    )
    assert captured["V_max"] == 1.25  # SS driver default when caller omits


def test_dispatch_default_calls_transient(monkeypatch):
    called = {}

    def fake_transient(stack, *, N_grid, n_points, v_rate, V_max,
                       illuminated, progress=None):
        called["hit"] = True
        return "TRANSIENT_RESULT"

    monkeypatch.setattr(bm.jv_sweep, "run_jv_sweep", fake_transient)
    out = bm._run_jv_dispatch(
        stack=None, N_grid=10, n_points=5, v_rate=1.0, V_max=None,
        illuminated=True,  # solver omitted → transient
    )
    assert out == "TRANSIENT_RESULT" and called["hit"]


def test_dispatch_unknown_solver_raises():
    import pytest

    with pytest.raises(ValueError):
        bm._run_jv_dispatch(
            stack=None, N_grid=10, n_points=5, v_rate=1.0, V_max=None,
            illuminated=True, solver="bogus",
        )
