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
import pytest

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


def test_stack_from_dict_plumbs_jv_solver_policy():
    cfg = _min_cfg()
    cfg["device"]["jv_solver_policy"] = "cancellation_safe_qf_required"
    stack = bm.stack_from_dict(cfg)
    assert stack.jv_solver_policy == "cancellation_safe_qf_required"
    assert (
        bm._stack_to_config_dict(stack)["device"]["jv_solver_policy"]
        == "cancellation_safe_qf_required"
    )


def test_stack_from_dict_defaults_to_general_jv_solver_policy():
    assert bm.stack_from_dict(_min_cfg()).jv_solver_policy == "general"


def test_stack_from_dict_carries_all_material_params():
    """The inline path must plumb the SAME layer fields the YAML loader does. It
    silently dropped 17 — TE Richardson constants (A_star), the effective DOS
    (Nc300/Nv300, which the default-ON dos_band_potentials fold needs), trap
    profiles, the 2nd ion species, and temperature scaling — so a UI-built device
    lost that physics. Pinned via delegation to the shared loader parser."""
    cfg = _min_cfg()
    cfg["layers"][0].update(
        A_star_n=5.0e5, A_star_p=6.0e5,
        Nc300=2.2e25, Nv300=1.8e25,
        D_ion_neg=3e-17, P0_neg=1e24, P_lim_neg=5e29,
        mu_T_gamma=-2.0, E_a_ion=0.4, B_rad_T_gamma=-1.0,
        varshni_alpha=4.7e-4, varshni_beta=636.0,
        trap_N_t_interface=1e22, trap_N_t_bulk=1e20,
        trap_decay_length=2e-8, trap_profile_shape="gaussian", trap_edge="left",
    )
    p = bm.stack_from_dict(cfg).layers[0].params
    assert (p.A_star_n, p.A_star_p) == (5.0e5, 6.0e5)
    assert (p.Nc300, p.Nv300) == (2.2e25, 1.8e25)
    assert (p.D_ion_neg, p.P0_neg, p.P_lim_neg) == (3e-17, 1e24, 5e29)
    assert (p.mu_T_gamma, p.E_a_ion, p.B_rad_T_gamma) == (-2.0, 0.4, -1.0)
    assert (p.varshni_alpha, p.varshni_beta) == (4.7e-4, 636.0)
    assert (p.trap_N_t_interface, p.trap_N_t_bulk) == (1e22, 1e20)
    assert p.trap_decay_length == 2e-8
    assert (p.trap_profile_shape, p.trap_edge) == ("gaussian", "left")


def test_stack_from_dict_layer_parity_with_loader():
    """Inline path and YAML loader build byte-identical MaterialParams from the
    same layer dict — the regression guard against the two parsers drifting."""
    import dataclasses
    from perovskite_sim.models import config_loader as cl

    layer = {
        **_min_cfg()["layers"][0],
        "Nc300": 2.5e25, "A_star_p": 7e5, "trap_N_t_bulk": 1e21,
        "varshni_alpha": 3e-4, "v_sat_n": 1e5, "Eg_back": 1.7,
    }
    cfg = {"device": {"mode": "full"}, "layers": [layer]}
    inline = bm.stack_from_dict(cfg).layers[0].params
    shared = cl.material_params_from_dict(layer)
    assert dataclasses.asdict(inline) == dataclasses.asdict(shared)


def test_stack_from_dict_nested_contacts_block():
    """Device-level: the nested ``contacts:`` block (not just flat S_* keys) must
    be honoured, mirroring the YAML loader."""
    cfg = _min_cfg()
    cfg["device"]["contacts"] = {
        "left": {"S_n": 1e3, "S_p": 2e3},
        "right": {"S_n": 3e3, "S_p": 4e3},
    }
    s = bm.stack_from_dict(cfg)
    assert (s.S_n_left, s.S_p_left) == (1e3, 2e3)
    assert (s.S_n_right, s.S_p_right) == (3e3, 4e3)


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


def test_stack_from_dict_plumbs_optical_material():
    """Inline-device path must carry TMM optics (optical_material / n_optical /
    incoherent); without them wavelength-resolved EQE/EL raise 'requires
    optical_material' even on a *_tmm preset (the frontend always submits the
    device inline). Mirrors config_loader.py."""
    cfg = _min_cfg()
    cfg["layers"][0].update(
        optical_material="MAPbI3", n_optical=2.4, incoherent=True
    )
    p = bm.stack_from_dict(cfg).layers[0].params
    assert p.optical_material == "MAPbI3"
    assert p.n_optical == 2.4
    assert p.incoherent is True


def test_stack_from_dict_optical_defaults():
    """Absent optics → Beer-Lambert sentinels (None / None / False),
    bit-identical to the pre-fix inline path."""
    p = bm.stack_from_dict(_min_cfg()).layers[0].params
    assert p.optical_material is None
    assert p.n_optical is None
    assert p.incoherent is False


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
            point_acceptance=(
                "residual_converged",
                "current_bounded_stall",
                "transient_assisted",
            ),
            point_residual=np.array([1.0e-9, 2.0e-7, 3.0e-6]),
            point_continuity_current_bound=np.array([
                1.0e-8, 2.0e-6, np.nan,
            ]),
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
    assert [status.reason_code for status in res.status_fwd] == [
        "residual_converged",
        "current_bounded_stall",
        "transient_assisted",
    ]
    assert [status.branch for status in res.status_rev] == [
        "jv_reverse",
        "jv_reverse",
        "jv_reverse",
    ]
    assert res.status_fwd[0].max_normalized_residual == pytest.approx(1.0e-9)
    assert (
        res.status_fwd[1].electron_continuity_bound_A_m2
        == pytest.approx(2.0e-6)
    )
    assert (
        res.status_fwd[1].hole_continuity_bound_A_m2
        == pytest.approx(2.0e-6)
    )
    assert res.status_fwd[0].valid is True
    assert res.status_fwd[1].valid is True
    assert res.status_fwd[2].valid is False
    assert res.certified is False


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


def test_dispatch_steady_state_missing_metadata_is_compatible_but_uncertified(
    monkeypatch,
):
    def fake_ss(stack, *, N_grid, n_points, V_max, illuminated,
                iface_states, progress=None):
        return SimpleNamespace(
            V=np.array([0.0, 0.5]),
            J=np.array([200.0, 100.0]),
            metrics="MET",
        )

    monkeypatch.setattr(bm, "run_jv_sweep_ss", fake_ss)
    result = bm._run_jv_dispatch(
        stack=None,
        N_grid=30,
        n_points=2,
        v_rate=1.0,
        V_max=0.5,
        illuminated=True,
        solver="steady_state",
    )

    assert len(result.status_fwd) == len(result.V_fwd)
    assert all(s.reason_code == "not_reported" for s in result.status_fwd)
    assert all(not s.valid for s in result.status_fwd)
    assert result.certified is False


def test_dispatch_steady_state_misaligned_metadata_fails_closed(monkeypatch):
    def fake_ss(stack, *, N_grid, n_points, V_max, illuminated,
                iface_states, progress=None):
        return SimpleNamespace(
            V=np.array([0.0, 0.5]),
            J=np.array([200.0, 100.0]),
            metrics="MET",
            point_acceptance=("residual_converged",),
            point_residual=np.array([1.0e-9, 2.0e-9]),
            point_current_bound=np.array([1.0e-8, 2.0e-8]),
        )

    monkeypatch.setattr(bm, "run_jv_sweep_ss", fake_ss)
    result = bm._run_jv_dispatch(
        stack=None,
        N_grid=30,
        n_points=2,
        v_rate=1.0,
        V_max=0.5,
        illuminated=True,
        solver="steady_state",
    )

    assert [status.reason_code for status in result.status_fwd] == [
        "residual_converged",
        "not_reported",
    ]
    assert all(not status.valid for status in result.status_fwd)
    assert all(
        "incomplete or misaligned" in status.message
        for status in result.status_fwd
    )


def test_dispatch_default_calls_transient(monkeypatch):
    called = {}

    def fake_transient(
        stack,
        *,
        N_grid,
        n_points,
        v_rate,
        V_max,
        illuminated,
        experiment_protocol=None,
        protocol_mode="compatibility",
        progress=None,
    ):
        called["hit"] = True
        called["experiment_protocol"] = experiment_protocol
        called["protocol_mode"] = protocol_mode
        return "TRANSIENT_RESULT"

    monkeypatch.setattr(bm.jv_sweep, "run_jv_sweep", fake_transient)
    out = bm._run_jv_dispatch(
        stack=None, N_grid=10, n_points=5, v_rate=1.0, V_max=None,
        illuminated=True,  # solver omitted → transient
    )
    assert out == "TRANSIENT_RESULT" and called["hit"]
    assert called["experiment_protocol"] is None
    assert called["protocol_mode"] == "compatibility"


def test_dispatch_quasi_fermi_returns_certificate_bearing_jvresult(monkeypatch):
    captured = {}

    def point(voltage, current):
        return SimpleNamespace(
            V_app=voltage,
            current_A_m2=current,
            certified=True,
            max_normalized_cell_residual=1.0e-12,
            electron_continuity_bound_A_m2=2.0e-8,
            hole_continuity_bound_A_m2=3.0e-8,
            face_current_spread_A_m2=4.0e-8,
            poisson_residual=5.0e-12,
        )

    points = (point(0.0, 200.0), point(0.5, 100.0), point(0.6, -1.0))

    def fake_qf(
        x,
        stack,
        voltages,
        *,
        interface_boundary,
        interface_transport_model,
        stop_after_voc,
    ):
        captured.update(
            node_count=len(x),
            voltages=np.asarray(voltages),
            interface_boundary=interface_boundary,
            interface_transport_model=interface_transport_model,
            stop_after_voc=stop_after_voc,
        )
        return SimpleNamespace(
            voltages_V=np.array([p.V_app for p in points]),
            currents_A_m2=np.array([p.current_A_m2 for p in points]),
            points=points,
            metrics="MET",
        )

    monkeypatch.setattr(bm, "solve_quasi_fermi_jv_sweep", fake_qf)
    result = bm._run_jv_dispatch(
        stack=bm.stack_from_dict(_min_cfg()),
        N_grid=10,
        n_points=8,
        v_rate=1.0,
        V_max=0.7,
        illuminated=True,
        solver="quasi_fermi",
        interface_boundary=True,
        interface_transport_model="scaps_thermionic",
    )

    assert captured["node_count"] > 0
    assert captured["voltages"] == pytest.approx(np.linspace(0.0, 0.7, 8))
    assert captured["stop_after_voc"] is True
    assert captured["interface_boundary"] is True
    assert captured["interface_transport_model"] == "scaps_thermionic"
    assert result.certified
    assert result.hysteresis_index == 0.0
    assert result.metrics_fwd == result.metrics_rev == "MET"
    assert all(s.solver == "quasi_fermi" for s in result.status_fwd)
    assert result.status_fwd[-1].face_current_spread_A_m2 == 4.0e-8
    assert result.status_fwd[-1].poisson_residual == 5.0e-12
    assert result.bulk_defect_evidence is None


def _defect_diagnostics(
    *,
    digest="a" * 64,
    minimum_occupancy=0.1,
    maximum_occupancy=0.8,
    minimum_denominator=4.0,
    charge=(-2.0, 3.0),
    recombination=(-5.0, 7.0),
    distribution_kinds=(),
    source_energy_orders=(),
    spatial_profile_sha256s=(),
    minimum_density_multipliers=(),
    maximum_density_multipliers=(),
):
    return SimpleNamespace(
        model_identity_sha256=digest,
        species_identifiers=("V_I", "I_i"),
        charge_transitions=("acceptor", "donor"),
        minimum_occupancy=minimum_occupancy,
        maximum_occupancy=maximum_occupancy,
        minimum_kinetic_denominator_s1=minimum_denominator,
        total_charge_density_C_m3=np.asarray(charge),
        total_recombination_rate_m3_s=np.asarray(recombination),
        distribution_kinds=distribution_kinds,
        source_energy_orders=source_energy_orders,
        spatial_profile_sha256s=spatial_profile_sha256s,
        minimum_density_multipliers=minimum_density_multipliers,
        maximum_density_multipliers=maximum_density_multipliers,
    )


def test_qf_bulk_defect_evidence_preserves_identity_and_global_extrema():
    points = (
        SimpleNamespace(bulk_defect_diagnostics=_defect_diagnostics()),
        SimpleNamespace(bulk_defect_diagnostics=_defect_diagnostics(
            minimum_occupancy=0.02,
            maximum_occupancy=0.95,
            minimum_denominator=2.0,
            charge=(-11.0, 1.0),
            recombination=(-3.0, 13.0),
        )),
    )
    evidence = bm._summarize_qf_bulk_defect_evidence(points)
    assert evidence.model == "monovalent-device-mb-qf-dc-v1"
    assert evidence.model_identity_sha256 == "a" * 64
    assert evidence.species_identifiers == ("V_I", "I_i")
    assert evidence.charge_transitions == ("acceptor", "donor")
    assert evidence.points_completed == 2
    assert evidence.minimum_occupancy == pytest.approx(0.02)
    assert evidence.maximum_occupancy == pytest.approx(0.95)
    assert evidence.minimum_kinetic_denominator_s1 == pytest.approx(2.0)
    assert evidence.maximum_absolute_charge_density_C_m3 == pytest.approx(11.0)
    assert evidence.maximum_absolute_recombination_rate_m3_s == pytest.approx(13.0)


def test_qf_bulk_defect_evidence_preserves_distributed_source_protocol():
    diagnostics = _defect_diagnostics(
        distribution_kinds=("gaussian", "conduction_band_tail"),
        source_energy_orders=(16, 16),
    )
    evidence = bm._summarize_qf_bulk_defect_evidence(
        (SimpleNamespace(bulk_defect_diagnostics=diagnostics),)
    )

    assert evidence.distribution_kinds == (
        "gaussian",
        "conduction_band_tail",
    )
    assert evidence.source_energy_orders == (16, 16)


def test_qf_bulk_defect_evidence_preserves_spatial_profile_identity():
    diagnostics = _defect_diagnostics(
        spatial_profile_sha256s=("c" * 64, None),
        minimum_density_multipliers=(0.5, 1.0),
        maximum_density_multipliers=(1.5, 1.0),
    )
    evidence = bm._summarize_qf_bulk_defect_evidence(
        (SimpleNamespace(bulk_defect_diagnostics=diagnostics),)
    )

    assert evidence.spatial_closure == "layer-density-profile-v1"
    assert evidence.spatial_profile_sha256s == ("c" * 64, None)
    assert evidence.minimum_density_multipliers == (0.5, 1.0)
    assert evidence.maximum_density_multipliers == (1.5, 1.0)


@pytest.mark.parametrize(
    "points, message",
    [
        (
            (
                SimpleNamespace(bulk_defect_diagnostics=_defect_diagnostics()),
                SimpleNamespace(bulk_defect_diagnostics=None),
            ),
            "incomplete",
        ),
        (
            (
                SimpleNamespace(bulk_defect_diagnostics=_defect_diagnostics()),
                SimpleNamespace(bulk_defect_diagnostics=_defect_diagnostics(digest="b" * 64)),
            ),
            "identity changed",
        ),
        (
            (
                SimpleNamespace(
                    bulk_defect_diagnostics=_defect_diagnostics(
                        spatial_profile_sha256s=("c" * 64, None),
                        minimum_density_multipliers=(0.5, 1.0),
                        maximum_density_multipliers=(1.5, 1.0),
                    )
                ),
                SimpleNamespace(
                    bulk_defect_diagnostics=_defect_diagnostics(
                        spatial_profile_sha256s=("d" * 64, None),
                        minimum_density_multipliers=(0.5, 1.0),
                        maximum_density_multipliers=(1.5, 1.0),
                    )
                ),
            ),
            "identity changed",
        ),
    ],
)
def test_qf_bulk_defect_evidence_fails_closed_on_incomplete_or_mixed_identity(
    points,
    message,
):
    with pytest.raises(ValueError, match=message):
        bm._summarize_qf_bulk_defect_evidence(points)


def test_dispatch_quasi_fermi_rejects_dark_jv():
    with pytest.raises(ValueError, match="illuminated J-V only"):
        bm._run_jv_dispatch(
            stack=None,
            N_grid=10,
            n_points=8,
            v_rate=1.0,
            V_max=0.7,
            illuminated=False,
            solver="quasi_fermi",
        )


def test_dispatch_rejects_interface_boundary_on_other_solvers():
    with pytest.raises(ValueError, match="requires solver='quasi_fermi'"):
        bm._run_jv_dispatch(
            stack=None,
            N_grid=10,
            n_points=8,
            v_rate=1.0,
            V_max=0.7,
            illuminated=True,
            solver="steady_state",
            interface_boundary=True,
        )


def test_dispatch_rejects_inactive_nondefault_interface_model():
    with pytest.raises(ValueError, match="requires interface_boundary=true"):
        bm._run_jv_dispatch(
            stack=None,
            N_grid=10,
            n_points=8,
            v_rate=1.0,
            V_max=0.7,
            illuminated=True,
            solver="quasi_fermi",
            interface_transport_model="scaps_thermionic",
        )


def test_jv_api_returns_422_for_policy_rejected_driver():
    cfg = _min_cfg()
    cfg["device"]["jv_solver_policy"] = "cancellation_safe_qf_required"
    request = bm.JVRequest(
        device=cfg,
        N_grid=10,
        n_points=2,
        V_max=0.1,
        solver="transient",
    )

    with pytest.raises(bm.HTTPException) as exc_info:
        bm.run_jv(request)
    assert exc_info.value.status_code == 422
    assert "solver='quasi_fermi'" in str(exc_info.value.detail)


def test_dispatch_unknown_solver_raises():
    with pytest.raises(ValueError):
        bm._run_jv_dispatch(
            stack=None, N_grid=10, n_points=5, v_rate=1.0, V_max=None,
            illuminated=True, solver="bogus",
        )
