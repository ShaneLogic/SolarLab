from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.experiments.jv_sweep import (
    JVAcceptedSolveDiagnostics,
    JVMetrics,
    JVPointStatus,
)
from perovskite_sim.experiments.impedance_frequency import (
    FrequencyWindowAssessment,
)
from perovskite_sim.experiments.ion_aware_impedance import (
    FrequencyPerturbationAssessment,
    IonAwareImpedanceCertificate,
    IonAwareImpedanceFrequencyCertificate,
    PerturbationStepAssessment,
)
from perovskite_sim.discretization.grid import GridResolutionError
from perovskite_sim.physics.contacts import ContactThermodynamicCertificate
from perovskite_sim.solver.numerical_diagnostics import (
    NumericalDiagnosticsMonitor,
    StateLayout,
)
from perovskite_sim.solver.mol import StateVec
from perovskite_sim.solver.tolerances import ComponentwiseAtol
from perovskite_sim.validation import refinement_executors as executors
from perovskite_sim.validation.numerical_certificate import (
    LaneDefinition,
    MatrixPoint,
    ObservableGate,
    content_sha256,
    load_refinement_registry,
)


ROOT = Path(__file__).resolve().parents[3]


def _lane(*, options=None, grid_values=(4, 7)):
    return LaneDefinition(
        lane_id="adapter-smoke",
        claim_level="internal-numerical-candidate",
        config_path="device.yaml",
        config_sha256="0" * 64,
        grid_parameter="N_grid",
        grid_values=grid_values,
        tolerance_parameter="factor",
        tolerance_factors=(1.0, 0.1),
        observables=(ObservableGate("response", "absolute_linf", 1.0),),
        quality_gates=(),
        options_json=json.dumps(options or {}),
    )


def _metric_dict(measurement, *, quality=False):
    values = measurement.quality if quality else measurement.observables
    return {item.name: item for item in values}


def _assert_protocol(measurement):
    metadata = json.loads(measurement.metadata_json)
    assert metadata["protocol"]["schema_version"] == metadata["protocol_schema"]
    assert content_sha256(metadata["protocol"]) == metadata["protocol_hash"]


def _assert_registry_contract(measurement, lane_id):
    registry = load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    )
    lane = registry.lane(lane_id)
    observables = _metric_dict(measurement)
    quality = _metric_dict(measurement, quality=True)
    assert set(observables) == {gate.metric for gate in lane.observables}
    assert set(quality) == {gate.metric for gate in lane.quality_gates}
    assert all(
        observables[gate.metric].units == gate.units for gate in lane.observables
    )
    assert all(quality[gate.metric].units == gate.units for gate in lane.quality_gates)


def _accepted_numerical_diagnostics(*, nonfinite_rhs: bool = False):
    state = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 0.0])
    monitor = NumericalDiagnosticsMonitor(
        StateLayout(2, positive_ion_active=(True, False))
    )
    monitor.observe_trial_state(state)
    monitor.observe_srh_denominator("bulk", np.array([7.0]))
    monitor.observe_srh_denominator("interface", np.array([8.0]))
    rhs = np.zeros_like(state)
    if nonfinite_rhs:
        rhs[0] = np.nan
    monitor.observe_rhs(rhs)
    report = monitor.finalize(state, solver_success=True)
    return JVAcceptedSolveDiagnostics(report=report, nfev=9, njev=2, nlu=2)


def test_frozen_ion_steady_executor_smoke(monkeypatch):
    lane = _lane(
        options={"V_max_V": 1.0, "voltage_points": 3},
        grid_values=(10, 20),
    )
    calls = []
    monkeypatch.setattr(executors, "_load_stack", lambda *_: object())
    monkeypatch.setattr(
        executors,
        "build_electrical_grid",
        lambda *_: np.linspace(0.0, 1.0, 11),
    )

    def solve(_x, _stack, voltage, **kwargs):
        calls.append((voltage, kwargs["tol"], kwargs["tol_step"]))
        return SimpleNamespace(
            y=np.ones(33),
            residual=1.0e-8,
            continuity_current_bound=1.0e-8,
            acceptance="residual_converged",
            iterations=2,
        )

    monkeypatch.setattr(executors, "solve_steady_state", solve)
    monkeypatch.setattr(
        executors,
        "_compute_current_ss_with_spread",
        lambda _x, _y, _stack, voltage, **_kwargs: (1.0 - 2.0 * voltage, 1e-8),
    )

    measurement = executors.run_frozen_ion_steady_jv(
        lane,
        MatrixPoint(10, 0.1),
        Path("."),
    )
    _assert_protocol(measurement)
    _assert_registry_contract(measurement, "scaps-mirror-frozen-ion-ss")
    assert [item[0] for item in calls] == [0.0, 0.5, 1.0]
    assert all(item[1] == pytest.approx(1.0e-7) for item in calls)
    assert all(item[2] == pytest.approx(1.0e-9) for item in calls)
    assert _metric_dict(measurement)["voc_V"].values[0] == pytest.approx(0.5)
    assert _metric_dict(measurement, quality=True)[
        "all_points_residual_converged"
    ].values == (1.0,)
    protocol = json.loads(measurement.metadata_json)["protocol"]
    assert protocol["settle"]["max_continuity_current_error_A_m2"] == pytest.approx(
        0.1
    )
    assert "max_current_spread_A_m2" not in protocol["settle"]


def test_mobile_ion_transient_executor_smoke(monkeypatch):
    lane = _lane(options={"V_max_V": 1.0, "voltage_points": 3})
    monkeypatch.setattr(
        executors,
        "_load_stack",
        lambda *_: SimpleNamespace(T=300.0),
    )
    captured = {}

    def run_jv(_stack, **kwargs):
        captured.update(kwargs)
        snapshots = tuple(
            SimpleNamespace(
                P=np.array([1.0, 1.0]),
                x=np.array([0.0, 1.0]),
            )
            for _ in range(6)
        )
        metrics = JVMetrics(0.5, 1.0, 0.5, 0.025)
        statuses = tuple(
            JVPointStatus(
                branch="jv_forward" if index < 3 else "jv_reverse",
                index=index % 3,
                voltage=float(index % 3) / 2.0,
                numerical_diagnostics=(_accepted_numerical_diagnostics(),),
                nfev=9,
                njev=2,
                nlu=2,
            )
            for index in range(6)
        )
        return SimpleNamespace(
            V_fwd=np.array([0.0, 0.5, 1.0]),
            J_fwd=np.array([1.0, 0.0, -1.0]),
            V_rev=np.array([1.0, 0.5, 0.0]),
            J_rev=np.array([-1.0, 0.0, 1.0]),
            metrics_rev=metrics,
            hysteresis_index=0.0,
            snapshots_fwd=snapshots[:3],
            snapshots_rev=snapshots[3:],
            status_fwd=statuses[:3],
            status_rev=statuses[3:],
            initial_numerical_diagnostics=SimpleNamespace(
                numerical_diagnostics=(
                    _accepted_numerical_diagnostics().report
                )
            ),
            certified=True,
        )

    monkeypatch.setattr(executors, "run_jv_sweep", run_jv)
    measurement = executors.run_mobile_ion_transient_jv(
        lane,
        MatrixPoint(4, 0.1),
        Path("."),
    )
    _assert_protocol(measurement)
    _assert_registry_contract(measurement, "ionmonger-mobile-ion-transient")

    assert isinstance(captured["atol"], ComponentwiseAtol)
    assert captured["atol"].refinement_factor == pytest.approx(0.1)
    assert captured["protocol_mode"] == "research_strict"
    assert captured["collect_numerical_diagnostics"] is True
    assert _metric_dict(measurement)["voc_reverse_V"].values == (0.5,)
    assert _metric_dict(measurement, quality=True)[
        "max_positive_ion_inventory_relative_drift"
    ].values == (0.0,)
    quality = _metric_dict(measurement, quality=True)
    assert quality["diagnostics_complete"].values == (1.0,)
    assert quality["terminal_densities_positive"].values == (1.0,)
    assert quality["nonfinite_rhs_evaluations"].values == (0.0,)
    assert quality["zero_floor_diagnostics_pass"].values == (1.0,)
    metadata = json.loads(measurement.metadata_json)
    assert metadata["accepted_solver_segment_count"] == 6
    assert metadata["negative_trial_evaluations"] == 0
    assert metadata["minimum_bulk_srh_denominator_s_m3"] == 7.0
    assert metadata["minimum_interface_srh_denominator_s_m4"] == 8.0


def test_mobile_ion_executor_fails_quality_when_diagnostics_are_missing(
    monkeypatch,
):
    lane = _lane(options={"V_max_V": 1.0, "voltage_points": 2})
    monkeypatch.setattr(
        executors,
        "_load_stack",
        lambda *_: SimpleNamespace(T=300.0),
    )
    snapshots = tuple(
        SimpleNamespace(P=np.ones(2), x=np.array([0.0, 1.0]))
        for _ in range(4)
    )
    metrics = JVMetrics(0.5, 1.0, 0.5, 0.025)
    incomplete_statuses = tuple(
        JVPointStatus(
            branch="jv_forward" if index < 2 else "jv_reverse",
            index=index % 2,
            voltage=float(index % 2),
            numerical_diagnostics=(
                (_accepted_numerical_diagnostics(),) if index != 3 else ()
            ),
        )
        for index in range(4)
    )
    monkeypatch.setattr(
        executors,
        "run_jv_sweep",
        lambda *_args, **_kwargs: SimpleNamespace(
            V_fwd=np.array([0.0, 1.0]),
            J_fwd=np.array([1.0, -1.0]),
            V_rev=np.array([1.0, 0.0]),
            J_rev=np.array([-1.0, 1.0]),
            metrics_rev=metrics,
            hysteresis_index=0.0,
            snapshots_fwd=snapshots[:2],
            snapshots_rev=snapshots[2:],
            status_fwd=incomplete_statuses[:2],
            status_rev=incomplete_statuses[2:],
            initial_numerical_diagnostics=SimpleNamespace(
                numerical_diagnostics=(
                    _accepted_numerical_diagnostics().report
                )
            ),
            certified=True,
        ),
    )

    measurement = executors.run_mobile_ion_transient_jv(
        lane,
        MatrixPoint(4, 0.1),
        Path("."),
    )

    quality = _metric_dict(measurement, quality=True)
    assert quality["diagnostics_complete"].values == (0.0,)
    assert quality["terminal_densities_positive"].values == (0.0,)


def test_ion_aware_dc_executor_smoke(monkeypatch):
    lane = _lane(
        options={
            "V_dc_V": 0.9,
            "settle_end_times_s": [1.0, 2.0],
            "required_consecutive_passes": 2,
        }
    )
    stack = SimpleNamespace(T=300.0)
    monkeypatch.setattr(executors, "_load_stack", lambda *_args: stack)
    monkeypatch.setattr(
        executors,
        "build_electrical_grid",
        lambda *_args: np.linspace(0.0, 1.0, 5),
    )
    monkeypatch.setattr(executors, "build_material_arrays", lambda *_args: object())
    contact = ContactThermodynamicCertificate(
        status="compatible_unverified",
        built_in_potential_mode="legacy_manual",
        tolerance_eV=0.005,
        fermi_level_span_eV=None,
        potential_mismatch_V=0.0,
        metal_work_function_mismatch_eV=None,
        contact_quasi_fermi_levels_eV=(),
        message="endpoint DOS unavailable",
    )
    report = _accepted_numerical_diagnostics().report
    certificate = SimpleNamespace(
        dc_current_density_A_m2=200.0,
        maximum_site_occupancy_fraction=0.01,
        positive_ion_inventory=SimpleNamespace(
            terminal_centroid_fraction=0.6,
        ),
        minimum_electron_density_m3=1.0,
        minimum_hole_density_m3=2.0,
        minimum_positive_ion_density_m3=3.0,
        minimum_negative_ion_density_m3=None,
        contact_thermodynamics=contact,
        carrier_area_rate_A_m2=1.0e-4,
        dc_face_current_spread_A_m2=2.0e-4,
        ion_area_rate_A_m2=3.0e-8,
        max_ion_inventory_relative_drift=4.0e-13,
        max_ionic_face_current_A_m2=5.0e-9,
    )
    captured = {}

    def fake_solve(_grid, _stack, protocol, **kwargs):
        captured["protocol"] = protocol
        captured.update(kwargs)
        attempt = SimpleNamespace(
            success=True,
            numerical_diagnostics=report,
        )
        steps = tuple(
            SimpleNamespace(
                numerical_diagnostics=report,
                diagnostics_passed=True,
                target_time_s=target,
                accepted_method="Radau",
                attempts=(attempt,),
                nfev=10,
            )
            for target in (1.0, 2.0)
        )
        return SimpleNamespace(
            state_certificate=certificate,
            steps=steps,
            numerically_certified=True,
            thermodynamically_certified=False,
            total_settle_time_s=2.0,
        )

    monkeypatch.setattr(executors, "solve_ion_aware_dc", fake_solve)

    measurement = executors.run_ion_aware_dc_operating_point(
        lane, MatrixPoint(4, 0.1), Path(".")
    )

    _assert_protocol(measurement)
    _assert_registry_contract(measurement, "ionmonger-ion-aware-dc-v1")
    assert isinstance(captured["atol"], ComponentwiseAtol)
    assert captured["atol"].refinement_factor == pytest.approx(0.1)
    assert captured["require_numerical_certificate"] is False
    assert captured["method_ladder"] == ("Radau", "BDF")
    assert _metric_dict(measurement)["dc_current_density_A_m2"].values == (
        200.0,
    )
    quality = _metric_dict(measurement, quality=True)
    assert quality["required_consecutive_passes_met"].values == (1.0,)
    assert quality["contact_not_inconsistent"].values == (1.0,)
    metadata = json.loads(measurement.metadata_json)
    assert metadata["thermodynamically_certified"] is False
    assert metadata["total_settle_time_s"] == pytest.approx(2.0)


def test_ion_aware_impedance_executor_smoke(monkeypatch):
    registry = load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    )
    lane = registry.lane("ionmonger-ion-aware-impedance-resolved-v1")
    stack = SimpleNamespace(T=300.0, mode="full")
    grid = np.linspace(0.0, 1.0, 5)
    monkeypatch.setattr(executors, "_load_stack", lambda *_args: stack)
    monkeypatch.setattr(
        executors,
        "build_electrical_grid",
        lambda *_args: grid,
    )
    monkeypatch.setattr(executors, "build_material_arrays", lambda *_args: object())

    contact = ContactThermodynamicCertificate(
        status="compatible_unverified",
        built_in_potential_mode="legacy_manual",
        tolerance_eV=0.005,
        fermi_level_span_eV=None,
        potential_mismatch_V=0.0,
        metal_work_function_mismatch_eV=None,
        contact_quasi_fermi_levels_eV=(),
        message="endpoint DOS unavailable",
    )
    dc_certificate = SimpleNamespace(
        contact_thermodynamics=contact,
        max_ion_inventory_relative_drift=2.0e-13,
        maximum_site_occupancy_fraction=0.02,
    )
    captured = {}

    def fake_dc(_grid, _stack, protocol, **kwargs):
        captured["dc_atol"] = kwargs["atol"]
        captured["dc_require"] = kwargs["require_numerical_certificate"]
        return SimpleNamespace(
            y=np.ones(15),
            protocol_hash=protocol.protocol_hash,
            numerically_certified=True,
            thermodynamically_certified=False,
            state_certificate=dc_certificate,
        )

    monkeypatch.setattr(executors, "solve_ion_aware_dc", fake_dc)

    def fake_build_impedance(dc_state, frequencies, **kwargs):
        captured["frequencies"] = np.asarray(frequencies)
        captured["state_step"] = kwargs["state_step"]
        captured["voltage_step"] = kwargs["voltage_step"]
        payload = {
            "dc_protocol_sha256": dc_state.protocol_hash,
            "schema_version": "test-ion-aware-impedance-protocol-v1",
            "state_step": kwargs["state_step"],
            "voltage_step": kwargs["voltage_step"],
        }
        return SimpleNamespace(
            protocol_hash="b" * 64,
            to_dict=lambda: payload,
        )

    monkeypatch.setattr(
        executors,
        "build_ion_aware_impedance_protocol",
        fake_build_impedance,
    )

    def fake_impedance(_grid, _stack, protocol, **kwargs):
        captured["impedance_kwargs"] = kwargs
        frequencies = captured["frequencies"]
        frequency_assessments = tuple(
            FrequencyPerturbationAssessment(
                frequency_Hz=float(frequency),
                coarse_factor=0.5,
                fine_factor=0.25,
                impedance_magnitude_relative_change=1.0e-4,
                impedance_phase_change_deg=2.0e-3,
                passed=True,
            )
            for frequency in frequencies
        )
        step = PerturbationStepAssessment(
            coarse_factor=0.5,
            fine_factor=0.25,
            max_impedance_magnitude_relative_change=1.0e-4,
            max_impedance_phase_change_deg=2.0e-3,
            passed=True,
            frequency_assessments=frequency_assessments,
        )
        points = tuple(
            IonAwareImpedanceFrequencyCertificate(
                frequency_Hz=float(frequency),
                numerically_certified=True,
                max_relative_face_spread=1.0e-6,
                reciprocal_condition=0.1,
                backward_error=1.0e-13,
                positive_ion_inventory_response_relative=2.0e-12,
                negative_ion_inventory_response_relative=0.0,
                current_decomposition_relative_error=3.0e-12,
                electron_storage_response_F_m2=1.0e-5 + 2.0e-6j,
                hole_storage_response_F_m2=2.0e-5 + 3.0e-6j,
                positive_ion_storage_response_F_m2=3.0e-5 + 4.0e-6j,
                negative_ion_storage_response_F_m2=None,
                net_charge_storage_response_F_m2=4.0e-5 + 5.0e-6j,
                perturbation_assessments=(frequency_assessments[0],),
                reasons=(),
            )
            for frequency in frequencies
        )
        certificate = IonAwareImpedanceCertificate(
            numerically_certified=True,
            thermodynamically_certified=False,
            certified=False,
            max_relative_face_spread=1.0e-6,
            max_backward_error=1.0e-13,
            minimum_reciprocal_condition=0.1,
            max_mass_diagonal_relative_error=4.0e-12,
            max_mass_off_diagonal_relative=5.0e-12,
            max_ion_inventory_response_relative=2.0e-12,
            max_current_decomposition_relative_error=3.0e-12,
            frequency_window_certified=True,
            perturbation_assessments=(step,),
            frequency_point_certificates=points,
            reasons=(),
        )
        window = FrequencyWindowAssessment(
            f_min_Hz=float(frequencies[0]),
            f_max_Hz=float(frequencies[-1]),
            has_mobile_ions=True,
            ionic_branch_covered=True,
            max_observed_sampling_gap_decades=0.25,
        )
        return SimpleNamespace(
            Z=np.asarray([10.0 - 1.0j] * len(frequencies)),
            certificate=certificate,
            frequency_window=window,
            protocol_hash=protocol.protocol_hash,
        )

    monkeypatch.setattr(executors, "run_ion_aware_impedance", fake_impedance)

    measurement = executors.run_ion_aware_impedance_frequency_domain(
        lane,
        MatrixPoint(60, 0.5),
        Path("."),
    )

    _assert_protocol(measurement)
    _assert_registry_contract(
        measurement,
        "ionmonger-ion-aware-impedance-resolved-v1",
    )
    assert captured["dc_atol"].refinement_factor == pytest.approx(1.0)
    assert captured["dc_require"] is False
    assert captured["state_step"] == pytest.approx(5.0e-6)
    assert captured["voltage_step"] == pytest.approx(5.0e-6)
    assert captured["impedance_kwargs"]["require_numerical_certificate"] is False
    quality = _metric_dict(measurement, quality=True)
    assert quality["all_frequency_points_certified"].values == (1.0,)
    assert quality["frequency_window_certified"].values == (1.0,)
    metadata = json.loads(measurement.metadata_json)
    assert metadata["actual_nodes"] == 5
    assert metadata["external_finite_difference_step_factor"] == pytest.approx(0.5)
    assert metadata["raw_impedance_ohm_m2"][0] == {
        "imag": -1.0,
        "real": 10.0,
    }


def test_csi_qf_frequency_executor_smoke(monkeypatch):
    lane = _lane(
        options={
            "biases_V": [-0.2, -0.1, 0.0],
            "frequencies_Hz": [1.0e4, 1.0e5, 1.0e6],
        }
    )
    monkeypatch.setattr(
        executors,
        "_load_stack",
        lambda *_: SimpleNamespace(T=300.0),
    )
    monkeypatch.setattr(
        executors,
        "build_electrical_grid",
        lambda *_: np.linspace(0.0, 1.0, 5),
    )
    guard_calls = []
    monkeypatch.setattr(
        executors,
        "require_thick_layer_interface_resolution",
        lambda grid, stack, *, N_grid: guard_calls.append(
            (grid.copy(), stack, N_grid)
        ),
    )
    monkeypatch.setattr(executors, "build_material_arrays", lambda *_: object())
    calls = []

    def impedance(_grid, _stack, frequencies, *, V_dc, **kwargs):
        calls.append((V_dc, kwargs["state_step"], kwargs["voltage_step"]))
        capacitance = 2.0e-4 + (V_dc + 0.2) * 1.0e-4
        admittance = 1j * 2.0 * np.pi * frequencies * capacitance
        return SimpleNamespace(
            Z=1.0 / admittance,
            max_relative_face_spread=np.full(3, 1.0e-6),
            backward_error=np.full(3, 1.0e-12),
            reciprocal_condition=np.full(3, 0.1),
            dc_state=SimpleNamespace(
                certified=True,
                max_normalized_cell_residual=1.0e-12,
            ),
        )

    monkeypatch.setattr(executors, "run_quasi_fermi_impedance", impedance)
    monkeypatch.setattr(
        executors,
        "_fit_mott_schottky",
        lambda *_args, **_kwargs: (0.8, 1.0e22, -0.2, 0.0),
    )
    measurement = executors.run_csi_qf_frequency_domain(
        lane,
        MatrixPoint(4, 0.1),
        Path("."),
    )
    _assert_protocol(measurement)
    _assert_registry_contract(measurement, "csi-qf-frequency-domain")

    assert len(calls) == 3
    assert len(guard_calls) == 1
    assert guard_calls[0][2] == 4
    assert all(call[1] == pytest.approx(1.0e-6) for call in calls)
    assert all(call[2] == pytest.approx(1.0e-6) for call in calls)
    assert _metric_dict(measurement)["mott_intercept_V"].values == (0.8,)
    assert _metric_dict(measurement, quality=True)["dc_residual_certified"].values == (
        1.0,
    )


@pytest.mark.parametrize(
    "capacitance",
    (0.0, -2.0e-4, np.nan),
    ids=("zero", "negative", "nonfinite"),
)
def test_csi_qf_frequency_executor_rejects_invalid_capacitance(
    monkeypatch,
    capacitance,
):
    lane = _lane(
        options={
            "biases_V": [-0.2, -0.1, 0.0],
            "frequencies_Hz": [1.0e4, 1.0e5, 1.0e6],
        }
    )
    monkeypatch.setattr(
        executors,
        "_load_stack",
        lambda *_: SimpleNamespace(T=300.0),
    )
    monkeypatch.setattr(
        executors,
        "build_electrical_grid",
        lambda *_: np.linspace(0.0, 1.0, 5),
    )
    monkeypatch.setattr(
        executors,
        "require_thick_layer_interface_resolution",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(executors, "build_material_arrays", lambda *_: object())

    def impedance(_grid, _stack, frequencies, **_kwargs):
        admittance = 1j * 2.0 * np.pi * frequencies * capacitance
        impedance_values = (
            np.full(frequencies.shape, 1.0e300 + 0.0j)
            if capacitance == 0.0
            else 1.0 / admittance
        )
        return SimpleNamespace(
            Z=impedance_values,
            max_relative_face_spread=np.full(3, 1.0e-6),
            backward_error=np.full(3, 1.0e-12),
            reciprocal_condition=np.full(3, 0.1),
            dc_state=SimpleNamespace(
                certified=True,
                max_normalized_cell_residual=1.0e-12,
            ),
        )

    monkeypatch.setattr(executors, "run_quasi_fermi_impedance", impedance)
    monkeypatch.setattr(
        executors,
        "_fit_mott_schottky",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid capacitance reached Mott-Schottky fitting"
        ),
    )

    with pytest.raises(ValueError, match="capacitance must be finite and positive"):
        executors.run_csi_qf_frequency_domain(
            lane,
            MatrixPoint(4, 0.1),
            Path("."),
        )


def test_csi_minimum_roadmap_lane_fails_at_the_registered_grid_guard():
    registry = load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    )
    lane = registry.lane("csi-qf-frequency-domain")

    with pytest.raises(GridResolutionError, match="under-resolved electrical grid"):
        executors.run_csi_qf_frequency_domain(
            lane,
            MatrixPoint(100, 1.0),
            ROOT,
        )


@dataclass
class _FakeGrid:
    x: np.ndarray
    y: np.ndarray

    @property
    def Nx(self):
        return len(self.x)

    @property
    def Ny(self):
        return len(self.y)


def test_twod_uniform_executor_smoke(monkeypatch):
    lane = _lane(
        options={
            "base_lateral_intervals": 4,
            "base_vertical_intervals_per_layer": 5,
            "V_max_V": 1.0,
            "V_step_V": 0.5,
            "settle_time_s": 1.0e-6,
        },
        grid_values=(1, 2),
    )
    fake_layers = [SimpleNamespace(thickness=1.0) for _ in range(3)]
    monkeypatch.setattr(
        executors,
        "_load_stack",
        lambda *_: SimpleNamespace(T=300.0),
    )
    monkeypatch.setattr(executors, "_freeze_ions", lambda stack: stack)
    monkeypatch.setattr(executors, "electrical_layers", lambda _stack: fake_layers)
    metrics = JVMetrics(0.5, 1.0, 0.5, 0.025)
    captured = {}

    def run_jv(*_args, **kwargs):
        captured["jv"] = kwargs
        snapshots = tuple(
            SimpleNamespace(
                x=np.linspace(0.0, 1.0, 3),
                n=np.ones(3),
                p=np.ones(3),
                P=np.ones(3),
                V_app=voltage,
            )
            for voltage in (0.0, 0.5, 1.0)
        )
        return SimpleNamespace(
            V_fwd=np.array([0.0, 0.5, 1.0]),
            # Deliberately includes a large non-conduction contribution.  The
            # uniform-limit adapter must compare Jn + Jp from snapshots instead.
            J_fwd=np.array([101.0, 100.0, 99.0]),
            metrics_fwd=metrics,
            snapshots_fwd=snapshots,
            certified=True,
        )

    monkeypatch.setattr(executors, "run_jv_sweep", run_jv)
    monkeypatch.setattr(executors, "build_material_arrays", lambda *_: object())

    def current_components(_x, _state, _stack, voltage, **_kwargs):
        half = 0.5 * (1.0 - 2.0 * voltage)
        return SimpleNamespace(
            J_n=np.full(2, half),
            J_p=np.full(2, half),
        )

    monkeypatch.setattr(
        executors,
        "compute_current_components",
        current_components,
        raising=False,
    )
    grid = _FakeGrid(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 4))

    def build_grid(layers, **kwargs):
        captured["layers"] = layers
        captured["grid"] = kwargs
        return grid

    monkeypatch.setattr(executors, "build_grid_2d", build_grid)
    monkeypatch.setattr(
        executors,
        "solve_illuminated_ss",
        lambda *_args, **_kwargs: StateVec.pack(np.ones(4), np.ones(4), np.ones(4)),
    )
    material = SimpleNamespace(has_radiative_reabsorption_2d=False)
    monkeypatch.setattr(
        executors,
        "build_material_arrays_2d",
        lambda *_args, **_kwargs: material,
    )
    monkeypatch.setattr(
        executors,
        "_settle_2d_with_tolerance",
        lambda state, *_args, **_kwargs: state,
    )

    def snapshot(_state, _material, *, V_app):
        return SimpleNamespace(
            V=V_app,
            n=np.ones((4, 3)),
            p=np.ones((4, 3)),
            Jy_n=np.full((3, 3), 0.5),
            Jy_p=np.full((3, 3), 0.5),
        )

    monkeypatch.setattr(executors, "extract_snapshot_2d", snapshot)
    monkeypatch.setattr(
        executors,
        "compute_terminal_current_2d",
        lambda snap: -(1.0 - 2.0 * snap.V),
    )
    monkeypatch.setattr(executors, "_poisson_relative_residual", lambda *_: 1e-12)

    measurement = executors.run_twod_uniform_limit(
        lane,
        MatrixPoint(2, 0.1),
        Path("."),
    )
    _assert_protocol(measurement)
    _assert_registry_contract(measurement, "twod-uniform-limit")
    assert captured["jv"]["N_grid"] == 31
    assert captured["jv"]["v_rate"] == pytest.approx(5.0e5)
    assert captured["jv"]["save_snapshots"] is True
    assert captured["grid"]["Nx"] == 8
    assert {layer.N for layer in captured["layers"]} == {10}
    assert _metric_dict(measurement)["voc_2d_V"].values == (0.5,)
    assert _metric_dict(measurement)["jv_2d_to_1d_normalized_difference"].values == (
        0.0,
        0.0,
        0.0,
    )
    assert _metric_dict(measurement, quality=True)[
        "frozen_ion_scope_declared"
    ].values == (1.0,)
    assert _metric_dict(measurement, quality=True)[
        "max_abs_2d_to_1d_normalized_difference"
    ].values == (0.0,)
    protocol = json.loads(measurement.metadata_json)["protocol"]
    assert protocol["current_composition"] == {
        "compared": "electron_plus_hole_conduction",
        "excluded": ["ionic", "displacement"],
        "one_d_source": "saved_forward_snapshots_contact_face",
        "two_d_source": "terminal_contact_carrier_flux",
    }


def test_twod_absolute_parity_preserves_a_stable_offset():
    difference, max_abs = executors._normalized_current_parity(
        np.array([1.2, 0.2, -0.8]),
        np.array([1.0, 0.0, -1.0]),
        scale=1.0,
    )
    lane = load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    ).lane("twod-uniform-limit")
    parity_gate = next(
        gate
        for gate in lane.quality_gates
        if gate.metric == "max_abs_2d_to_1d_normalized_difference"
    )

    assert difference == pytest.approx(np.full(3, 0.2))
    assert max_abs == pytest.approx(0.2)
    assert parity_gate.operator == "le"
    assert parity_gate.limit == pytest.approx(0.005)
    assert max_abs > parity_gate.limit


def test_twod_lateral_uniformity_includes_hole_variation():
    snapshot = SimpleNamespace(
        n=np.ones((2, 3)),
        p=np.array([[1.0, 2.0, 1.0], [4.0, 8.0, 4.0]]),
    )

    assert executors._max_lateral_carrier_variation_relative(snapshot) == pytest.approx(
        1.0
    )


def test_interface_charge_off_executor_smoke(monkeypatch):
    lane = _lane(options={"V_max_V": 1.0, "voltage_points": 3})
    defect = SimpleNamespace(
        calibration_factor=1.0,
        iface_state_calibration_factor=1.0,
    )
    stack = SimpleNamespace(
        interface_charge_closure="off",
        interface_charge_rebaseline_acknowledged=True,
        het_recomb_despike=0.0,
        flat_band_contacts=False,
        flat_band_metal_contacts=False,
        contact_phi_B_eV=0.0,
        interface_defects=(defect,),
    )
    monkeypatch.setattr(executors, "_load_stack", lambda *_: stack)
    monkeypatch.setattr(
        executors,
        "build_electrical_grid",
        lambda *_: np.linspace(0.0, 1.0, 3),
    )
    monkeypatch.setattr(executors, "build_two_sided_trace_grid", lambda x, _: x)
    base_material = SimpleNamespace(iface_state_charge=0.0)
    monkeypatch.setattr(executors, "build_material_arrays", lambda *_: base_material)
    material = SimpleNamespace(iface_qss_left_nodes=(0,))
    monkeypatch.setattr(
        executors,
        "_prepare_two_sided_material",
        lambda *_: material,
    )
    contact = ContactThermodynamicCertificate(
        status="certified",
        built_in_potential_mode="semiconductor_work_function",
        tolerance_eV=5.0e-3,
        fermi_level_span_eV=0.0,
        potential_mismatch_V=0.0,
        metal_work_function_mismatch_eV=None,
        contact_quasi_fermi_levels_eV=(0.0, 0.0, 0.0, 0.0),
        message="test",
    )
    monkeypatch.setattr(
        executors,
        "require_contact_thermodynamic_certificate",
        lambda *_: contact,
    )
    state = StateVec.pack(
        np.ones(3),
        np.ones(3),
        np.ones(3),
    )
    point = SimpleNamespace(
        y=state,
        phi=np.zeros(3),
        certified=True,
        electron_continuity_bound_A_m2=1.0e-8,
        hole_continuity_bound_A_m2=2.0e-8,
        face_current_spread_A_m2=3.0e-8,
        interface_local_residual=4.0e-9,
        interface_topology=executors.TWO_SIDED_TRACE,
        max_normalized_cell_residual=5.0e-9,
        poisson_residual=6.0e-10,
    )
    monkeypatch.setattr(
        executors, "solve_quasi_fermi_steady_state", lambda *_a, **_k: point
    )
    metrics = executors.compute_metrics(
        np.array([0.0, 0.5, 1.0]),
        np.array([1.0, 0.0, -1.0]),
    )
    sweep = SimpleNamespace(
        points=(point, point, point),
        currents_A_m2=np.array([1.0, 0.0, -1.0]),
        metrics=metrics,
        certified=True,
    )
    monkeypatch.setattr(
        executors,
        "solve_quasi_fermi_jv_sweep",
        lambda *_a, **_k: sweep,
    )
    qss = SimpleNamespace(
        capture_flux_m2_s=np.array([2.0, 1.0, 3.0, 4.0]),
        state_flux_m2_s=np.zeros(4),
        occupancy=np.array([0.4]),
    )
    monkeypatch.setattr(
        executors, "solve_material_two_sided_interfaces_qss", lambda *_a, **_k: qss
    )

    measurement = executors.run_interface_recombination_charge_off(
        lane,
        MatrixPoint(4, 0.1),
        Path("."),
    )
    _assert_protocol(measurement)
    _assert_registry_contract(
        measurement,
        "interface-recombination-charge-off",
    )
    assert _metric_dict(measurement)["voc_V"].values == (0.5,)
    assert _metric_dict(measurement)["interface_flux_A_m2"].shape == (3,)
    assert _metric_dict(measurement, quality=True)[
        "trap_electrostatic_charge_enabled"
    ].values == (0.0,)
    assert _metric_dict(measurement, quality=True)[
        "max_interface_state_residual_A_m2"
    ].values == (0.0,)
    protocol = json.loads(measurement.metadata_json)["protocol"]
    assert protocol["interface"] == {
        "charge_closure": "off",
        "cross_transmission": 1.0,
        "rebaseline_acknowledged": True,
        "topology": executors.TWO_SIDED_TRACE,
        "transport_model": executors.FERMI_DIRAC_RICHARDSON,
    }
    assert protocol["solver"]["base_newton_residual_tolerance"] == pytest.approx(4.0e-7)


def test_two_sided_interface_evidence_checks_carrier_and_state_balance():
    result = SimpleNamespace(
        capture_flux_m2_s=np.array([2.0, 1.0, 3.0, 4.0]),
        state_flux_m2_s=np.array([1.0, -2.0, 3.0, -4.0]),
        occupancy=np.array([0.25]),
    )

    flux, carrier_balance, state_residual, occupancy = (
        executors._two_sided_interface_evidence(result, interface_count=1)
    )

    assert flux == pytest.approx([5.0 * executors.Q])
    assert carrier_balance == pytest.approx(0.0)
    assert state_residual == pytest.approx(4.0 * executors.Q)
    assert occupancy == pytest.approx([0.25])


def test_two_sided_interface_evidence_rejects_unbounded_occupancy():
    result = SimpleNamespace(
        capture_flux_m2_s=np.zeros(4),
        state_flux_m2_s=np.zeros(4),
        occupancy=np.array([1.01]),
    )

    with pytest.raises(RuntimeError, match="occupancy left"):
        executors._two_sided_interface_evidence(result, interface_count=1)
