from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import perovskite_sim.experiments.ion_aware_dc as dc
from perovskite_sim.experiments.jv_sweep import (
    build_electrical_grid,
    compute_ionic_current_components,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.solver.numerical_diagnostics import (
    DensityMinima,
    NegativeEntryCounts,
    NumericalDiagnosticsReport,
)


def _stack_grid(n_grid: int = 8):
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    return stack, build_electrical_grid(stack, n_grid)


def _report(*, passed: bool = True) -> NumericalDiagnosticsReport:
    minima = DensityMinima(
        n=1.0,
        p=1.0,
        positive_ion_active=1.0,
        negative_ion_active=None,
        interface_state=None,
    )
    violations = () if passed else ("terminal_p_not_above_floor",)
    return NumericalDiagnosticsReport(
        mode="observe",
        solver_success=True,
        trial_evaluations=1,
        negative_trial_evaluations=0,
        negative_trial_entries=NegativeEntryCounts(),
        nonfinite_trial_evaluations=0,
        nonfinite_rhs_evaluations=0,
        minimum_trial_density_m3=minima,
        final_minimum_density_m3=minima,
        minimum_bulk_srh_denominator_s_m3=1.0,
        minimum_interface_srh_denominator_s_m4=None,
        terminal_density_floor_m3=0.0,
        bulk_srh_denominator_floor_s_m3=0.0,
        interface_srh_denominator_floor_s_m4=0.0,
        violations=violations,
        would_pass_strict=passed,
    )


def test_protocol_round_trip_and_hash_are_canonical():
    stack, _ = _stack_grid()
    protocol = dc.build_ion_aware_dc_protocol(
        stack,
        V_dc=0.9,
        illuminated=True,
        settle_end_times_s=(0.1, 1.0, 10.0),
    )

    rebuilt = dc.IonAwareDCProtocol.from_json(protocol.canonical_json())

    assert rebuilt == protocol
    assert rebuilt.protocol_hash == protocol.protocol_hash
    assert len(protocol.protocol_hash) == 64


def test_protocol_rejects_unknown_or_missing_fields():
    stack, _ = _stack_grid()
    payload = dc.build_ion_aware_dc_protocol(
        stack, V_dc=0.9, illuminated=True
    ).to_dict()
    payload["claim"] = "certified"
    with pytest.raises(ValueError, match="extra"):
        dc.IonAwareDCProtocol.from_dict(payload)

    payload.pop("claim")
    payload.pop("V_dc")
    with pytest.raises(ValueError, match="missing"):
        dc.IonAwareDCProtocol.from_dict(payload)


def test_user_supplied_state_identity_is_required_and_hashed():
    stack, x = _stack_grid()
    state = solve_equilibrium(x, stack)
    digest = dc.ion_aware_dc_state_sha256(state)

    protocol = dc.build_ion_aware_dc_protocol(
        stack,
        V_dc=0.9,
        illuminated=True,
        initial_state_source="user_supplied_state",
        initial_state_sha256=digest,
    )

    assert protocol.initial_state_sha256 == digest
    assert len(digest) == 64
    with pytest.raises(ValueError, match="requires a lowercase SHA-256"):
        dc.build_ion_aware_dc_protocol(
            stack,
            V_dc=0.9,
            illuminated=True,
            initial_state_source="user_supplied_state",
        )
    with pytest.raises(ValueError, match="cannot carry"):
        dc.build_ion_aware_dc_protocol(
            stack,
            V_dc=0.9,
            illuminated=True,
            initial_state_sha256=digest,
        )


def test_state_hash_rejects_nonfinite_or_nonvector_input():
    with pytest.raises(ValueError, match="finite 1-D"):
        dc.ion_aware_dc_state_sha256(np.array([[1.0]]))
    with pytest.raises(ValueError, match="finite 1-D"):
        dc.ion_aware_dc_state_sha256(np.array([np.nan]))


def test_state_hash_normalizes_signed_zero():
    assert dc.ion_aware_dc_state_sha256(np.array([0.0, 1.0])) == (
        dc.ion_aware_dc_state_sha256(np.array([-0.0, 1.0]))
    )


def test_protocol_records_effective_legacy_temperature():
    stack, _ = _stack_grid()
    legacy = replace(stack, mode="legacy", T=345.0)

    protocol = dc.build_ion_aware_dc_protocol(
        legacy, V_dc=0.9, illuminated=True
    )

    assert protocol.temperature_K == pytest.approx(300.0)


@pytest.mark.parametrize(
    "times",
    [(), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (np.inf,)],
)
def test_protocol_rejects_invalid_settle_ladders(times):
    stack, _ = _stack_grid()
    with pytest.raises((TypeError, ValueError)):
        dc.build_ion_aware_dc_protocol(
            stack,
            V_dc=0.9,
            illuminated=True,
            settle_end_times_s=times,
        )


def test_dark_protocol_has_no_illumination_source():
    stack, _ = _stack_grid()
    protocol = dc.build_ion_aware_dc_protocol(
        stack, V_dc=0.0, illuminated=False
    )
    assert protocol.illumination_source is None


def test_assessment_uses_per_species_ionic_current_not_cancelled_net(
    monkeypatch,
):
    stack, x = _stack_grid()
    mat = build_material_arrays(x, stack)
    state = solve_equilibrium(x, stack)
    protocol = dc.build_ion_aware_dc_protocol(
        stack,
        V_dc=0.0,
        illuminated=False,
        max_ionic_face_current_A_m2=0.5,
        max_carrier_area_rate_A_m2=1.0,
        max_ion_area_rate_A_m2=1.0,
        max_dc_face_current_spread_A_m2=1.0,
    )
    monkeypatch.setattr(dc, "assemble_rhs", lambda *_args, **_kwargs: np.zeros_like(state))
    monkeypatch.setattr(
        dc,
        "compute_current_components",
        lambda *_args, **_kwargs: SimpleNamespace(
            J_total=np.zeros(len(x) - 1)
        ),
    )
    monkeypatch.setattr(
        dc,
        "compute_ionic_current_components",
        lambda *_args, **_kwargs: SimpleNamespace(
            J_positive=np.ones(len(x) - 1),
            J_negative=-np.ones(len(x) - 1),
            J_total=np.zeros(len(x) - 1),
        ),
    )

    certificate = dc.assess_ion_aware_dc_state(
        x, state, state, stack, protocol, mat=mat
    )

    assert certificate.max_ionic_face_current_A_m2 == pytest.approx(1.0)
    assert "ionic_face_current_exceeds_limit" in certificate.numerical_reasons
    assert not certificate.numerically_certified


def test_ionic_current_unpack_preserves_dual_ion_with_interface_tail():
    stack, x = _stack_grid()
    layers = []
    for layer in stack.layers:
        params = layer.params
        if layer.role == "absorber":
            params = replace(
                params,
                D_ion_neg=3.2e-18,
                P0_neg=1.6e25,
                P_lim_neg=1.6e27,
            )
        layers.append(replace(layer, params=params))
    dual_stack = replace(stack, layers=tuple(layers))
    material = build_material_arrays(x, dual_stack)
    assert material.has_dual_ions
    state = solve_equilibrium(x, dual_stack)
    state_with_tail = np.concatenate((state, np.ones(4)))
    material_with_tail = replace(material, N_iface_state=1)

    current = compute_ionic_current_components(
        x,
        state_with_tail,
        dual_stack,
        0.0,
        mat=material_with_tail,
    )

    assert current.J_negative is not None
    assert current.J_negative.shape == (len(x) - 1,)


def test_solver_requires_protocol_initial_state_to_match_y0():
    stack, x = _stack_grid()
    protocol = dc.build_ion_aware_dc_protocol(
        stack,
        V_dc=0.9,
        illuminated=True,
        settle_end_times_s=(1.0,),
        required_consecutive_passes=1,
    )
    with pytest.raises(ValueError, match="supplying y0"):
        dc.solve_ion_aware_dc(
            x,
            stack,
            protocol,
            y0=solve_equilibrium(x, stack),
        )


def test_solver_rejects_user_state_that_does_not_match_protocol_hash():
    stack, x = _stack_grid()
    state = solve_equilibrium(x, stack)
    protocol = dc.build_ion_aware_dc_protocol(
        stack,
        V_dc=0.9,
        illuminated=True,
        initial_state_source="user_supplied_state",
        initial_state_sha256=dc.ion_aware_dc_state_sha256(state),
        settle_end_times_s=(1.0,),
        required_consecutive_passes=1,
    )
    changed = state.copy()
    changed[0] = np.nextafter(changed[0], np.inf)

    with pytest.raises(ValueError, match="initial_state_sha256"):
        dc.solve_ion_aware_dc(x, stack, protocol, y0=changed)


def test_solver_requires_mobile_ions():
    stack = load_device_from_yaml("configs/cSi_homojunction.yaml")
    x = build_electrical_grid(stack, 200)
    protocol = dc.build_ion_aware_dc_protocol(
        stack,
        V_dc=0.0,
        illuminated=False,
        settle_end_times_s=(1.0,),
        required_consecutive_passes=1,
    )
    with pytest.raises(dc.IonAwareDCCapabilityError, match="mobile-ion"):
        dc.solve_ion_aware_dc(x, stack, protocol)


def test_solver_counts_only_consecutive_state_and_diagnostic_passes(monkeypatch):
    stack, x = _stack_grid()
    protocol = dc.build_ion_aware_dc_protocol(
        stack,
        V_dc=0.9,
        illuminated=True,
        settle_end_times_s=(1.0, 2.0, 4.0, 8.0),
        required_consecutive_passes=2,
    )
    diagnostics = iter((_report(passed=False), _report(), _report()))
    state_passes = iter((True, True, True))
    durations: list[float] = []

    def fake_transient(_x, y0, span, t_eval, *_args, **_kwargs):
        durations.append(span[1] - span[0])
        assert t_eval[0] == pytest.approx(durations[-1])
        return SimpleNamespace(
            success=True,
            y=np.asarray(y0)[:, None],
            numerical_diagnostics=next(diagnostics),
            nfev=7,
            njev=2,
            nlu=3,
        )

    def fake_assess(*_args, **_kwargs):
        return SimpleNamespace(
            numerically_certified=next(state_passes),
            thermodynamically_certified=False,
            contact_thermodynamics=SimpleNamespace(status="compatible_unverified"),
        )

    monkeypatch.setattr(dc, "run_transient", fake_transient)
    monkeypatch.setattr(dc, "assess_ion_aware_dc_state", fake_assess)

    result = dc.solve_ion_aware_dc(x, stack, protocol)

    assert durations == pytest.approx([1.0, 1.0, 2.0])
    assert [step.accepted_for_closure for step in result.steps] == [False, True, True]
    assert result.consecutive_certified_steps == 2
    assert result.numerically_certified
    assert not result.certified
    assert result.total_settle_time_s == pytest.approx(4.0)
    assert result.steps[-1].nfev == 7


def test_solver_exhaustion_returns_evidence_on_error(monkeypatch):
    stack, x = _stack_grid()
    protocol = dc.build_ion_aware_dc_protocol(
        stack,
        V_dc=0.9,
        illuminated=True,
        settle_end_times_s=(1.0, 2.0),
        required_consecutive_passes=2,
    )
    initial = solve_equilibrium(x, stack)
    monkeypatch.setattr(
        dc,
        "run_transient",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=True,
            y=initial[:, None],
            numerical_diagnostics=_report(),
        ),
    )
    monkeypatch.setattr(
        dc,
        "assess_ion_aware_dc_state",
        lambda *_args, **_kwargs: SimpleNamespace(
            numerically_certified=False,
            thermodynamically_certified=False,
            contact_thermodynamics=SimpleNamespace(status="compatible_unverified"),
        ),
    )

    with pytest.raises(dc.IonAwareDCCertificationError) as caught:
        dc.solve_ion_aware_dc(x, stack, protocol)

    assert len(caught.value.result.steps) == 2
    assert not caught.value.result.numerically_certified


def test_solver_records_radau_failure_before_bdf_recovery(monkeypatch):
    stack, x = _stack_grid()
    protocol = dc.build_ion_aware_dc_protocol(
        stack,
        V_dc=0.9,
        illuminated=True,
        settle_end_times_s=(1.0,),
        required_consecutive_passes=1,
    )
    initial = solve_equilibrium(x, stack)
    methods: list[str] = []

    def fake_transient(_x, _y0, _span, _t_eval, *_args, **kwargs):
        method = kwargs["method"]
        methods.append(method)
        if method == "Radau":
            failed = _report(passed=False)
            return SimpleNamespace(
                success=False,
                message="max_nfev exhausted",
                y=np.empty((initial.size, 0)),
                numerical_diagnostics=failed,
                nfev=20_001,
            )
        return SimpleNamespace(
            success=True,
            message="success",
            y=initial[:, None],
            numerical_diagnostics=_report(),
            nfev=80,
        )

    monkeypatch.setattr(dc, "run_transient", fake_transient)
    monkeypatch.setattr(
        dc,
        "assess_ion_aware_dc_state",
        lambda *_args, **_kwargs: SimpleNamespace(
            numerically_certified=True,
            thermodynamically_certified=False,
            contact_thermodynamics=SimpleNamespace(status="compatible_unverified"),
        ),
    )

    result = dc.solve_ion_aware_dc(x, stack, protocol)
    step = result.steps[0]

    assert methods == ["Radau", "BDF"]
    assert step.accepted_method == "BDF"
    assert [attempt.success for attempt in step.attempts] == [False, True]
    assert step.nfev == 20_081


def test_solver_method_exhaustion_retains_attempt_evidence(monkeypatch):
    stack, x = _stack_grid()
    protocol = dc.build_ion_aware_dc_protocol(
        stack,
        V_dc=0.9,
        illuminated=True,
        settle_end_times_s=(1.0,),
        required_consecutive_passes=1,
    )

    def fake_transient(*_args, **kwargs):
        return SimpleNamespace(
            success=False,
            message=f"{kwargs['method']} failed",
            numerical_diagnostics=_report(passed=False),
            nfev=11,
        )

    monkeypatch.setattr(dc, "run_transient", fake_transient)

    with pytest.raises(dc.IonAwareDCSolverError) as caught:
        dc.solve_ion_aware_dc(x, stack, protocol)

    assert caught.value.target_time_s == pytest.approx(1.0)
    assert [attempt.method for attempt in caught.value.attempts] == [
        "Radau",
        "BDF",
    ]
    assert all(attempt.nfev == 11 for attempt in caught.value.attempts)
    assert all(attempt.numerical_diagnostics is not None for attempt in caught.value.attempts)


def test_contact_gate_remains_separate_from_numerical_closure(monkeypatch):
    stack, x = _stack_grid()
    protocol = dc.build_ion_aware_dc_protocol(
        stack,
        V_dc=0.9,
        illuminated=True,
        settle_end_times_s=(1.0,),
        required_consecutive_passes=1,
    )
    initial = solve_equilibrium(x, stack)
    monkeypatch.setattr(
        dc,
        "run_transient",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=True,
            y=initial[:, None],
            numerical_diagnostics=_report(),
        ),
    )
    monkeypatch.setattr(
        dc,
        "assess_ion_aware_dc_state",
        lambda *_args, **_kwargs: SimpleNamespace(
            numerically_certified=True,
            thermodynamically_certified=False,
            contact_thermodynamics=SimpleNamespace(status="compatible_unverified"),
        ),
    )

    result = dc.solve_ion_aware_dc(
        x, stack, protocol, require_contact_certificate=False
    )
    assert result.numerically_certified
    assert not result.certified

    with pytest.raises(dc.IonAwareDCCertificationError, match="contact"):
        dc.solve_ion_aware_dc(
            x, stack, protocol, require_contact_certificate=True
        )
