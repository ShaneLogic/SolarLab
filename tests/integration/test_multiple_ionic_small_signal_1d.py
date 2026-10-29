"""Two-domain AC algebra/port/source checks; native evidence is read only."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d import multiple_ionic_impedance as example
from examples.research_1d.multiple_ionic_domains import DOMAINS, INTERVALS, prepare_multiple_ionic_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_multiple_ionic_1d import compile_multiple_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.multiple_ionic_1d import MultipleIonic1DProblem
from solarlab.equations.multiple_ionic_small_signal import build_multiple_ionic_small_signal
from solarlab.experiments.ionic_impedance import IonicImpedanceOptions
from solarlab.experiments.multiple_ionic_impedance import (
    MultipleIonicImpedanceResult, _checks, read_multiple_ionic_impedance_result, write_multiple_ionic_impedance_result,
)
from solarlab.experiments.ionic_impedance import IonicImpedanceAttempt
from solarlab.solvers.nonlinear.problem import GateOutcome
from solarlab.solvers.linear import FrequencySolution

Q = 1.602176634e-19


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def case():
    device = compile_multiple_ionic_device_1d(prepare_multiple_ionic_device(),
        intervals_per_layer=INTERVALS, ion_domains=DOMAINS)
    u = device.initial_state(voltage_V=.002).u.copy()
    N = device.node_count
    u[:N] += .01 * np.sin(np.linspace(0, 2 * np.pi, N))
    u[N:2 * N] = np.linspace(-.15, .2, N)
    u[2 * N:3 * N] = np.linspace(.1, -.25, N)
    u[3 * N:] = np.linspace(-.12, .13, device.state_size - 3 * N)
    state = device.state(u, voltage_V=.002)
    return device, state, build_multiple_ionic_small_signal(device, state)


def derivative(function, u, direction, h=1e-5):
    return (8 * (function(u + h * direction) - function(u - h * direction))
            - (function(u + 2 * h * direction) - function(u - 2 * h * direction))) / (12 * h)


def test_one_production_electronic_block_and_distinct_zero_hz_inventories(case):
    device, state, operator = case
    problem = MultipleIonic1DProblem(device, state.voltage_V)
    assert (device.node_count, device.state_size) == (13, 47)
    np.testing.assert_array_equal(operator.jacobian.toarray(), problem.jacobian(state.u).toarray())
    np.testing.assert_array_equal(operator.storage_jacobian.toarray(), problem.storage_jacobian(state.u).toarray())
    system = operator.system(0.0, voltage_phasor_V=2e-6 + 1e-6j)
    assert system.constraint_rows == {"absorber_left": {"positive": 39}, "absorber_right": {"positive": 43}}
    assert system.original_rows_enforced == tuple(range(47))
    for name, blocks in device.ion_species_slices.items():
        row, block = system.constraint_rows[name]["positive"], blocks["positive"]
        gradient = problem.inventory_gradients(state.u)[name]["positive"]
        np.testing.assert_array_equal(system.jacobian[row, :].toarray()[0], gradient)
        np.testing.assert_array_equal(np.flatnonzero(gradient), np.arange(block.start, block.stop))
        assert system.storage_jacobian[row, :].nnz == 0
        assert system.rhs[row] == system.rhs_rate[row] == 0.0
    finite = operator.system(1.0)
    assert finite.constraint_rows == {}
    np.testing.assert_array_equal(finite.jacobian.toarray(), operator.jacobian.toarray())
    np.testing.assert_array_equal(finite.storage_jacobian.toarray(), operator.storage_jacobian.toarray())


def test_full_J_K_and_actual_moving_voltage_drive_directions(case):
    device, state, operator = case
    N = device.node_count
    problem = MultipleIonic1DProblem(device, state.voltage_V)
    rng = np.random.default_rng(1521)
    direction = rng.normal(size=device.state_size) * .03
    direction[:N] *= device.electronic_component.thermal_voltage_V
    scale = np.maximum(abs(operator.jacobian) @ abs(direction), 1e-20)
    assert np.max(abs(derivative(problem.residual, state.u, direction) - operator.jacobian @ direction) / scale) < 2e-7
    storage = derivative(problem.physical_storage, state.u, direction)
    scale_K = np.maximum(abs(operator.storage_jacobian) @ abs(direction), 1e-20)
    assert np.max(abs(storage + operator.storage_jacobian @ direction) / scale_K) < 2e-7
    lift = operator.coordinate_lift_per_V
    def primal(parameter):
        return MultipleIonic1DProblem(device, state.voltage_V + parameter[0]).residual(state.u + parameter[0] * lift)
    partial, rate = operator._moving_voltage_partial(lift)
    scale = np.maximum(abs(operator.jacobian) @ (abs(lift) + abs(operator.reference_direction_per_V)
        + np.r_[np.full(N, .02), np.ones(device.state_size - N)]), 1e-20)
    assert np.max(abs(derivative(primal, np.zeros(1), np.ones(1)) - partial) / scale) < 2e-7
    np.testing.assert_allclose(rate, operator.storage_jacobian @ (operator.reference_direction_per_V - lift), rtol=2e-14, atol=1e-12)
    np.testing.assert_array_equal(partial[3 * N:], operator.jacobian[3 * N:, :N] @ lift[:N])


def test_complex_local_ion_and_carrier_tangents_and_no_gap_flux(case):
    device, state, operator = case
    N = device.node_count
    problem = MultipleIonic1DProblem(device, state.voltage_V)
    rng = np.random.default_rng(1522)
    real, imag = rng.normal(size=device.state_size) * .01, rng.normal(size=device.state_size) * .01
    real[:N] *= .02
    imag[:N] *= .02
    out = operator.observe(real + 1j * imag, 1.0, coordinate_form="fixed_reference_v1")
    for index, species in enumerate(("electron", "hole")):
        def flux(u):
            return getattr(problem.electronic_problem.fluxes(u[:3 * N]), species).flux
        numeric = derivative(flux, state.u, real) + 1j * derivative(flux, state.u, imag)
        np.testing.assert_allclose(out["carrier_particle_fluxes_m2_s"][index, 1:-1], numeric, rtol=2e-7, atol=1e-12)
    for domain in device.domains:
        name = domain.domain_layer_id
        def flux(u):
            return problem.ion_fluxes(u, domain_id=name)
        numeric = derivative(flux, state.u, real) + 1j * derivative(flux, state.u, imag)
        full = out["domains"][name]["species"]["positive"]["particle_fluxes_m2_s"]
        np.testing.assert_allclose(full[domain.ion_node_indices[:-1] + 1], numeric, rtol=2e-7, atol=1e-12)
        allowed = np.zeros(N + 1, dtype=bool)
        allowed[domain.ion_node_indices[:-1] + 1] = True
        np.testing.assert_array_equal(full[~allowed], np.zeros(np.sum(~allowed)))
    np.testing.assert_array_equal(out["ionic_conduction_current_density_A_m2"][[0, 1, 2, 6, 7, 11, 12, 13]], np.zeros(8))


def test_global_charge_ports_and_Maxwell_accounted_once(case):
    device, state, operator = case
    V, f = 2e-6 + 1e-6j, 1.0
    x = np.random.default_rng(1523).normal(size=device.state_size) * (1e-4 + 2e-4j)
    out = operator.observe(x, f, voltage_phasor_V=V)
    particle = out["carrier_particle_fluxes_m2_s"]
    current = Q * (particle[1] - particle[0])
    charge = Q * device.electronic_component.geometry.cell_measures_m * (
        out["hole_density_perturbation_m3"] - out["electron_density_perturbation_m3"])
    for domain in device.domains:
        values = out["domains"][domain.domain_layer_id]["species"]["positive"]
        current += Q * values["particle_fluxes_m2_s"]
        charge[domain.ion_node_indices] += Q * (domain.ion_geometry.cell_measures_m * values["density_perturbation_m3"])
    np.testing.assert_array_equal(out["conduction_current_density_A_m2"], current)
    np.testing.assert_array_equal(out["charge_inventory_perturbation_C_m2"], charge)
    np.testing.assert_array_equal(out["total_current_density_A_m2"], current + out["displacement_current_density_A_m2"])
    assert out["total_current_density_A_m2"].shape == (14,)
    scale = max(np.max(abs(out["total_current_density_A_m2"])), 2 * np.pi * np.max(abs(charge)), np.finfo(float).tiny)
    assert np.max(abs(out["maxwell_identity_error_A_m2"])) < 256 * np.finfo(float).eps * scale
    fixed = operator.observe(x + (operator.coordinate_lift_per_V - operator.reference_direction_per_V) * V,
        f, voltage_phasor_V=V, coordinate_form="fixed_reference_v1")
    np.testing.assert_allclose(fixed["total_current_density_A_m2"], out["total_current_density_A_m2"], rtol=3e-12, atol=1e-12)
    factor = .4 - .7j
    other = operator.observe(x * factor, f, voltage_phasor_V=V * factor)
    np.testing.assert_allclose(other["total_current_density_A_m2"], out["total_current_density_A_m2"] * factor, rtol=3e-12, atol=1e-12)


def test_total_inventory_cancellation_does_not_pass_either_domain(case):
    device, _state, _operator = case
    domains = {}
    changes = []
    for index, domain in enumerate(device.domains):
        density = np.zeros(domain.ion_node_count, dtype=complex)
        density[0] = (1.0 if index == 0 else -1.0) * 1e8 / domain.ion_geometry.cell_measures_m[0]
        changes.append(np.sum(domain.ion_geometry.cell_measures_m * density))
        domains[domain.domain_layer_id] = {"species": {"positive": {"density_perturbation_m3": density}}}
    assert abs(sum(changes)) < 1e-7
    observed = {"domains": domains, "total_current_density_A_m2": np.zeros(14, dtype=complex),
                "original_componentwise_backward_error": 0.0}
    linear = FrequencySolution(0.0, np.zeros(47, dtype=complex), np.zeros(47, dtype=complex), 0.0, 0)
    checks = _checks(device, observed, linear, 1.0, IonicImpedanceOptions())
    assert all(not check.passed for check in checks if "inventory" in check.name)
    assert sum("inventory" in check.name for check in checks) == 2


@pytest.mark.parametrize("kind,immobile", [("independent", False), ("shared", False), ("positive", True)])
def test_unsupported_populations_refused_without_assembling(kind, immobile, monkeypatch):
    device = compile_multiple_ionic_device_1d(prepare_multiple_ionic_device(kind=kind, right_immobile=immobile),
        intervals_per_layer=INTERVALS, ion_domains=DOMAINS)
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported scope reached operator assembly")
    monkeypatch.setattr(MultipleIonic1DProblem, "jacobian", forbidden)
    with pytest.raises(CapabilityError):
        build_multiple_ionic_small_signal(device, device.initial_state())


def test_mode_contacts_shape_identity_and_changed_source_refused(case, monkeypatch, tmp_path):
    device, state, operator = case
    with pytest.raises(CapabilityError, match="only mobile"):
        operator.system(1.0, ion_mode="frozen")
    with pytest.raises(ValueError):
        build_multiple_ionic_small_signal(device, replace(state, device_sha256="wrong"))
    with pytest.raises(ValueError):
        operator.observe(np.zeros(48), 1.0)
    from solarlab.equations import multiple_ionic_1d
    changed = tmp_path / "changed.py"
    changed.write_text("changed source\n")
    monkeypatch.setattr(multiple_ionic_1d, "__file__", str(changed))
    with pytest.raises(RuntimeError, match="physical source changed"):
        operator.observe(np.zeros(47), 1.0)


def test_nonfixed_contacts_and_CT_refused_before_operator(monkeypatch):
    for variant in ("contacts", "CT"):
        prepared = prepare_multiple_ionic_device()
        data = prepared.to_input().editing_data()
        if variant == "contacts":
            data["settings"]["flat_band_contacts"] = True
        else:
            data["layers"][1]["parameters"].update(v_sat_n=1e4, ct_beta_n=2.0)
        prepared = resolve_device(DeviceInput.model_validate(data), prepared.defaults, prepared.resources, sources=prepared.sources)
        device = compile_multiple_ionic_device_1d(prepared, intervals_per_layer=INTERVALS, ion_domains=DOMAINS)
        def forbidden(*args, **kwargs):
            raise AssertionError("unsupported contact or CT reached assembly")
        monkeypatch.setattr(MultipleIonic1DProblem, "jacobian", forbidden)
        with pytest.raises(CapabilityError):
            build_multiple_ionic_small_signal(device, device.initial_state())


def test_saved_native_result_reads_without_compile_or_solver(monkeypatch):
    path, sha = os.environ.get("SOLARLAB_MULTIPLE_AC_RESULT"), os.environ.get("SOLARLAB_MULTIPLE_AC_SHA256")
    if path is None or sha is None:
        pytest.skip("bind the sole saved native result; never repeat a physics solve")
    def forbidden(*args, **kwargs):
        raise AssertionError("saved reading reached compilation or a solver")
    monkeypatch.setattr(example, "compile_multiple_ionic_device_1d", forbidden)
    monkeypatch.setattr(example, "solve_multiple_ionic_impedance", forbidden)
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
    record = read_multiple_ionic_impedance_result(path, sha)
    assert record["completed"] and len(record["result"]["attempts"]) == 4
    assert record["result"]["initial_provenance"]["source_sha256"] == "858ef2f39be16018049d49cfe171d3f79f03891c1259b0eff7701945a78ef7f3"
    for attempt in record["result"]["attempts"]:
        assert attempt["accepted"] and all(item["passed"] for item in attempt["checks"])
        out = attempt["observation"]
        assert set(out["domains"]) == {"absorber_left", "absorber_right"}
        np.testing.assert_array_equal(out["total_current_density_A_m2"],
            out["conduction_current_density_A_m2"] + out["displacement_current_density_A_m2"])
        assert tuple(out["original_rows_enforced"]) == tuple(range(47))
    assert record["result"]["physical_certificate_status"] == "not_evaluated"


def test_solver_free_saved_prefix_retains_first_failure(case, monkeypatch, tmp_path):
    """Synthetic codec/prefix fixture, never a claimed physical response."""
    device, _state, _operator = case
    requested = np.asarray([0.0, .01, 1.0, 1000.0])
    x = np.zeros(device.state_size, dtype=complex)
    solved = FrequencySolution(0.0, x, x.copy(), 0.0, 0)
    observed = {"coordinate_form": "moving_boundary_lift_v1", "solve_coordinate_perturbation": x,
                "synthetic_prefix_fixture": True}
    attempts = (IonicImpedanceAttempt(0.0, True, "converged", solved, observed,
                    (GateOutcome("synthetic_codec_check", True, 0.0, 1.0),), 0.0,
                    coordinate_representation="moving_boundary_lift_v1"),
                IonicImpedanceAttempt(.01, False, "synthetic_failure", None, None, (), 0.0, "fixture only"))
    from dataclasses import asdict
    result = MultipleIonicImpedanceResult(requested, 2e-6 + 1e-6j, attempts,
        {"role": "synthetic codec/prefix fixture only"}, device.identity, device.original_prepared_sha256,
        device.binding_manifest, {}, asdict(IonicImpedanceOptions()), 2, 0.0)
    path = tmp_path / "Prefix.json"
    digest = write_multiple_ionic_impedance_result(result, path)
    monkeypatch.setattr(example, "solve_multiple_ionic_impedance", lambda *args, **kwargs: pytest.fail("reader solved"))
    record = read_multiple_ionic_impedance_result(path, digest)
    assert not record["completed"] and len(record["result"]["attempts"]) == 2
    assert record["result"]["attempts"][-1]["termination"] == "synthetic_failure"
    np.testing.assert_array_equal(record["result"]["requested_frequencies_Hz"], requested)
    with pytest.raises(ValueError, match="SHA-256"):
        read_multiple_ionic_impedance_result(path, "wrong")
    with pytest.raises(FileExistsError):
        write_multiple_ionic_impedance_result(result, path)
    raw = json.loads(path.read_text())
    raw["completed"] = True
    bad = tmp_path / "Contradictory.json"
    bad.write_text(json.dumps(raw) + "\n")
    with pytest.raises(ValueError, match="completion"):
        read_multiple_ionic_impedance_result(bad, hashlib.sha256(bad.read_bytes()).hexdigest())
