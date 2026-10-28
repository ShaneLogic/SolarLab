"""Actual shared-site AC, saved-state admission and bounded failure records.

Default CI makes one genuine steady fixture and reuses it. Author/reviewer
waves can explicitly supply a pinned initial manifest and saved AC responses,
so checking evidence never implicitly regenerates a historical operating point.
"""

from copy import copy, deepcopy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_impedance import compile_case, load_manifest_case
from examples.research_1d.ionic_steady import controls as steady_controls
from examples.research_1d.transient import json_value
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.equations import ionic_small_signal as core
from solarlab.experiments import ionic_impedance as driver
from solarlab.experiments.ionic_transient import load_ionic_steady_initial
from solarlab.solvers.linear.frequency_domain import FrequencySolution
from solarlab.solvers.nonlinear.ionic_1d import solve_ionic_steady_state


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    manifest = os.environ.get("SOLARLAB_IONIC_AC_INITIAL_MANIFEST")
    if manifest:
        return load_manifest_case(manifest, "shared")
    compiled = compile_case("shared")
    controls = steady_controls(compiled.ion_node_count)
    point = solve_ionic_steady_state(compiled, voltage_V=0.002, generation_scale=0.0, options=controls)
    path = tmp_path_factory.mktemp("ionic_ac_initial") / "Steady.json"
    summary = {"accepted": point.accepted, "compiled_identity": compiled.identity,
               "voltage_V": 0.002, "generation_scale": 0.0}
    raw = (json.dumps(json_value({"summary": summary, "result": point}), allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    assert point.accepted, (point.termination, point.error, str(path))
    initial = load_ionic_steady_initial(compiled, path, hashlib.sha256(raw).hexdigest(), controls=controls)
    return compiled, initial


def recorded(path, digest):
    return driver.read_ionic_impedance_result(path, digest)["result"]


@pytest.fixture(scope="module")
def responses(case, tmp_path_factory):
    compiled, initial = case
    manifest_path = os.environ.get("SOLARLAB_IONIC_AC_RESPONSE_MANIFEST")
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text())
        items = manifest["responses"]
        results = {name: recorded(items[name]["path"], items[name]["sha256"])
                   for name in ("shared_mobile", "shared_frozen", "shared_scaled")}
    else:
        directory = tmp_path_factory.mktemp("ionic_ac_responses")
        results = {}
        for name, mode, voltage, frequencies in (
            ("shared_mobile", "mobile", 1.0, [0.0, 0.01, 1.0, 1000.0]),
            ("shared_frozen", "frozen", 1.0, [0.0, 0.01, 1.0, 1000.0]),
            ("shared_scaled", "mobile", 2e-6 + 1e-6j, [1.0]),
        ):
            result = driver.solve_ionic_impedance(compiled, initial, frequencies,
                ion_mode=mode, voltage_phasor_V=voltage)
            path = directory / (name + ".json")
            digest = driver.write_ionic_impedance_result(result, path)
            results[name] = recorded(path, digest)
            assert result.completed, (name, [(a.frequency_Hz, a.termination, a.error,
                [(check.name, check.value, check.limit) for check in a.checks if not check.passed])
                for a in result.failed_attempts], str(path))
    for result in results.values():
        assert result["compiled_identity"] == compiled.identity
        assert result["initial_provenance"]["source_sha256"] == initial.source_sha256
        assert result["physical_certificate_status"] == "not_evaluated"
        assert all(attempt["accepted"] for attempt in result["attempts"])
    return results


def test_actual_mobile_response_keeps_all_requested_points_ions_and_inventory(case, responses):
    compiled, initial = case
    result = responses["shared_mobile"]
    assert (compiled.node_count, compiled.ion_node_count, compiled.state_size) == (37, 13, 137)
    np.testing.assert_array_equal(result["requested_frequencies_Hz"], [0.0, 0.01, 1.0, 1000.0])
    assert result["linear_solver_calls"] == 4 and len(result["attempts"]) == 4
    assert result["phasor_convention"] == "exp(+i omega t)"
    assert result["wall_s"] < result["solver_controls"]["wall_s"]
    for attempt in result["attempts"]:
        observed = attempt["observation"]
        assert attempt["accepted"] and all(check["passed"] for check in attempt["checks"])
        assert attempt["coordinate_representation"] == observed["coordinate_form"]
        np.testing.assert_array_equal(observed["solve_coordinate_perturbation"], attempt["linear_solution"]["solution"])
        assert tuple(observed["original_rows_enforced"]) == tuple(range(compiled.state_size))
        assert attempt["linear_solution"]["componentwise_backward_error"] <= 1e-11
        assert observed["original_componentwise_backward_error"] <= 1e-11
        assert set(observed["inventory_constraint_rows"]) == (set(compiled.ion_species_names)
            if attempt["frequency_Hz"] == 0.0 else set())
        for species in compiled.ion_species_names:
            density = observed["ion_density_perturbation_m3"][species]
            assert density.shape == (13,) and np.max(abs(density)) > 0.0
            terms = compiled.ion_geometry.cell_measures_m * density
            inventory = complex(math.fsum(terms.real.tolist()), math.fsum(terms.imag.tolist()))
            ratio = abs(inventory) / initial.state.domain_inventories_m2[species]
            assert ratio <= 1e-12
            np.testing.assert_array_equal(observed["base_ion_density_m3"][species],
                getattr(initial.state, species + "_ion_density_m3"))


def test_all_face_signed_current_displacement_and_endpoint_gauss_accounting(case, responses):
    compiled, _ = case
    component = compiled.electronic_component
    for name in ("shared_mobile", "shared_frozen"):
        for attempt in responses[name]["attempts"]:
            o = attempt["observation"]
            ion = Q * (o["ion_particle_fluxes_m2_s"]["positive"] - o["ion_particle_fluxes_m2_s"]["negative"])
            np.testing.assert_allclose(o["ionic_conduction_current_density_A_m2"], ion, rtol=3e-15, atol=1e-30)
            conduction = Q * (o["carrier_particle_fluxes_m2_s"][1] - o["carrier_particle_fluxes_m2_s"][0]) + ion
            np.testing.assert_allclose(o["conduction_current_density_A_m2"], conduction, rtol=3e-15, atol=1e-30)
            internal_D = -component.poisson.face_capacitance_F_m2 * np.diff(o["potential_perturbation_V"])
            charge = o["charge_inventory_perturbation_C_m2"]
            full_D = np.r_[internal_D[0] - charge[0], internal_D, internal_D[-1] + charge[-1]]
            np.testing.assert_array_equal(o["displacement_current_density_A_m2"],
                2j * np.pi * attempt["frequency_Hz"] * full_D)
            total = o["conduction_current_density_A_m2"] + o["displacement_current_density_A_m2"]
            np.testing.assert_array_equal(total, o["total_current_density_A_m2"])
            assert total.shape == (38,)
            np.testing.assert_allclose(total, np.full(38, total[0]), rtol=1e-10, atol=1e-8)
            port = total[0 if component.voltage_polarity > 0 else -1]
            assert o["input_current_density_A_m2"] == component.voltage_polarity * port
            assert o["admittance_density_S_m2"] == o["input_current_density_A_m2"] / o["voltage_phasor_V"]
            assert np.max(abs(o["maxwell_identity_error_A_m2"])) < 128 * np.finfo(float).eps * max(1.0, np.max(abs(total)))


def test_explicit_frozen_response_retains_charge_and_exposes_unenforced_ion_forces(case, responses):
    compiled, _ = case
    mobile, frozen = responses["shared_mobile"], responses["shared_frozen"]
    assert frozen["ion_mode"] == "frozen" and len(frozen["attempts"]) == 4
    for active, clamp in zip(mobile["attempts"], frozen["attempts"]):
        a, b = active["observation"], clamp["observation"]
        np.testing.assert_array_equal(b["base_ion_charge_density_C_m3"], a["base_ion_charge_density_C_m3"])
        assert np.max(abs(b["base_ion_charge_density_C_m3"])) > 0.0
        assert tuple(b["original_rows_enforced"]) == tuple(range(3 * compiled.node_count))
        assert b["inventory_constraint_rows"] == {}
        np.testing.assert_array_equal(b["fixed_coordinate_perturbation"][3 * compiled.node_count:], 0.0)
        np.testing.assert_array_equal(b["ionic_conduction_current_density_A_m2"], 0.0)
        for species in compiled.ion_species_names:
            np.testing.assert_array_equal(b["base_ion_density_m3"][species], a["base_ion_density_m3"][species])
            np.testing.assert_array_equal(b["ion_density_perturbation_m3"][species], 0.0)
            assert np.max(abs(b["original_ion_force_residual"][species])) > 0.0


def test_complex_voltage_is_a_linear_normalization(case, responses):
    compiled, _ = case
    base = responses["shared_mobile"]["attempts"][2]["observation"]
    scaled = responses["shared_scaled"]["attempts"][0]["observation"]
    voltage = 2e-6 + 1e-6j
    assert scaled["frequency_Hz"] == base["frequency_Hz"] == 1.0
    assert scaled["voltage_phasor_V"] == voltage
    np.testing.assert_allclose(scaled["admittance_density_S_m2"], base["admittance_density_S_m2"], rtol=2e-10)
    for key in ("potential_perturbation_V", "electron_density_perturbation_m3", "hole_density_perturbation_m3",
                "conduction_current_density_A_m2", "displacement_current_density_A_m2", "total_current_density_A_m2"):
        scale = max(float(np.max(abs(base[key]))), np.finfo(float).tiny)
        assert np.max(abs(scaled[key] / voltage - base[key])) / scale < 2e-10
    for species in compiled.ion_species_names:
        a, b = base["ion_density_perturbation_m3"][species], scaled["ion_density_perturbation_m3"][species]
        assert np.max(abs(b / voltage - a)) / np.max(abs(a)) < 2e-10


@pytest.mark.parametrize("frequencies", [[], [-1.0], [np.nan], [np.inf], [1.0, 1.0], [[1.0]], [1 + 0j], list(range(33))])
def test_invalid_frequency_requests_refuse_before_a_solve(case, monkeypatch, frequencies):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid frequencies reached the solver")
    monkeypatch.setattr(driver, "solve_frequency_systems", forbidden)
    with pytest.raises(ValueError, match="frequenc"):
        driver.solve_ionic_impedance(*case, frequencies)


@pytest.mark.parametrize("voltage", [0.0, np.nan, np.inf, 1 + complex(0, np.inf)])
def test_invalid_amplitudes_refuse_before_a_solve(case, voltage):
    with pytest.raises(ValueError, match="finite and nonzero"):
        driver.solve_ionic_impedance(*case, [1.0], voltage_phasor_V=voltage)


def test_unsupported_mode_contacts_and_foreign_device_refuse(case):
    compiled, initial = case
    with pytest.raises(ValueError, match="ion_mode"):
        driver.solve_ionic_impedance(compiled, initial, [1.0], ion_mode="relaxed")
    foreign = compile_case("positive")
    with pytest.raises(ValueError, match="different compiled device"):
        driver.solve_ionic_impedance(foreign, initial, [1.0])
    modified, electronic = copy(compiled), copy(compiled.electronic_component)
    channel = replace(electronic.channels[0], velocity_m_s=1.0)
    object.__setattr__(electronic, "channels", (channel,) + electronic.channels[1:])
    object.__setattr__(modified, "electronic_component", electronic)
    with pytest.raises(ValueError, match="fixed-reservoir"):
        driver.solve_ionic_impedance(modified, initial, [1.0])


def test_initial_checksum_failed_state_and_rebinding_are_rejected(case):
    compiled, initial = case
    with pytest.raises(ValueError, match="pinned accepted steady input"):
        driver.solve_ionic_impedance(compiled, None, [1.0])
    with pytest.raises(ValueError, match="SHA-256"):
        driver.solve_ionic_impedance(compiled, replace(initial, source_sha256="0" * 64), [1.0])
    with pytest.raises(ValueError, match="disagrees"):
        driver.solve_ionic_impedance(compiled, replace(initial, generation_scale=1.0), [1.0])
    bad = json.loads(initial.evidence_json)
    bad["result"]["accepted"] = False
    text = json.dumps(bad)
    rejected = replace(initial, evidence_json=text, source_sha256=hashlib.sha256(text.encode()).hexdigest())
    with pytest.raises(ValueError, match="accepted steady solve"):
        driver.solve_ionic_impedance(compiled, rejected, [1.0])
    wrong = copy(initial.state)
    object.__setattr__(wrong, "device_sha256", "0" * 64)
    with pytest.raises(ValueError, match="disagrees"):
        driver.solve_ionic_impedance(compiled, replace(initial, state=wrong), [1.0])


def saved_operator(case, responses):
    """A cached real first response for execution-control tests; no new AC solve."""
    compiled, _ = case
    sample = responses["shared_mobile"]["attempts"][0]
    system = SimpleNamespace(jacobian=None, storage_jacobian=None, rhs=None, rhs_rate=None, fixed_coordinates={},
        original_rows_enforced=tuple(range(compiled.state_size)),
        constraint_rows=sample["observation"]["inventory_constraint_rows"],
        coordinate_form=sample["coordinate_representation"])
    operator = SimpleNamespace(system=lambda *args, **kwargs: system,
        observe=lambda *args, **kwargs: deepcopy(sample["observation"]))
    return operator, FrequencySolution(**sample["linear_solution"])


def test_first_failure_stops_sweep_and_preserves_accepted_prefix(case, responses, monkeypatch, tmp_path):
    operator, first = saved_operator(case, responses)
    monkeypatch.setattr(core, "build_ionic_small_signal", lambda *args, **kwargs: operator)
    seen = []
    def controlled(J, K, rhs, frequencies, **kwargs):
        seen.extend(frequencies)
        if len(seen) == 1:
            return (first,)
        raise ArithmeticError("injected second-frequency refusal")
    monkeypatch.setattr(driver, "solve_frequency_systems", controlled)
    result = driver.solve_ionic_impedance(*case, [0.0, 1.0, 1000.0])
    assert seen == [0.0, 1.0] and result.linear_solver_calls == 2
    assert not result.completed and result.attempts[0].accepted and len(result.failed_attempts) == 1
    failure = result.failed_attempts[0]
    assert failure.frequency_Hz == 1.0 and "injected second-frequency refusal" in failure.error
    path = tmp_path / "partial.json"
    digest = driver.write_ionic_impedance_result(result, path)
    decoded = driver.read_ionic_impedance_result(path, digest)
    assert not decoded["completed"] and len(decoded["result"]["attempts"]) == 2


def test_cooperative_wall_budget_exhaustion_does_not_start_a_frequency(case, responses, monkeypatch):
    operator, _ = saved_operator(case, responses)
    monkeypatch.setattr(core, "build_ionic_small_signal", lambda *args, **kwargs: operator)
    ticks = iter((0.0, 1.0, 31.0, 32.0, 33.0))
    monkeypatch.setattr(driver, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    result = driver.solve_ionic_impedance(*case, [0.0, 1.0])
    assert result.linear_solver_calls == 0 and not result.completed
    assert len(result.attempts) == 1 and result.attempts[0].termination == "wall_budget_exhausted"


def test_source_change_stops_before_a_solve(case, responses, monkeypatch):
    operator, _ = saved_operator(case, responses)
    monkeypatch.setattr(core, "build_ionic_small_signal", lambda *args, **kwargs: operator)
    pins = iter(({"source": "original"}, {"source": "changed"}))
    monkeypatch.setattr(driver, "_source_pins", lambda: next(pins))
    result = driver.solve_ionic_impedance(*case, [0.0, 1.0])
    assert result.linear_solver_calls == 0 and len(result.failed_attempts) == 1
    assert "source changed" in result.failed_attempts[0].error


def test_result_reading_roundtrips_complex_arrays_without_any_solver(case, responses, monkeypatch, tmp_path):
    result = responses["shared_scaled"]
    raw = (json.dumps({"schema": driver.RESULT_SCHEMA, "completed": True, "result": driver._encoded(result)}) + "\n").encode()
    path = tmp_path / "read_only.json"
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    def forbidden(*args, **kwargs):
        pytest.fail("result reading called a solver, operator or admission")
    monkeypatch.setattr(driver, "solve_frequency_systems", forbidden)
    monkeypatch.setattr(core, "build_ionic_small_signal", forbidden)
    monkeypatch.setattr(driver, "_admit_evidence", forbidden)
    actual = driver.read_ionic_impedance_result(path, digest)
    a, b = actual["result"]["attempts"][0], result["attempts"][0]
    np.testing.assert_array_equal(a["linear_solution"]["solution"], b["linear_solution"]["solution"])
    np.testing.assert_array_equal(a["observation"]["total_current_density_A_m2"], b["observation"]["total_current_density_A_m2"])
    assert not a["linear_solution"]["solution"].flags.writeable
    with pytest.raises(ValueError, match="SHA-256"):
        driver.read_ionic_impedance_result(path, "0" * 64)
    inconsistent = deepcopy(result)
    inconsistent["attempts"][0]["coordinate_representation"] = "wrong_representation"
    path.write_text(json.dumps({"schema": driver.RESULT_SCHEMA, "completed": True,
        "result": driver._encoded(inconsistent)}))
    with pytest.raises(ValueError, match="representation"):
        driver.read_ionic_impedance_result(path, hashlib.sha256(path.read_bytes()).hexdigest())
    altered = json.loads(raw)
    altered["completed"] = False
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="completion"):
        driver.read_ionic_impedance_result(path, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("kind", ["positive", "independent"])
def test_optional_historical_species_branch_from_pinned_saved_evidence(kind):
    initial_manifest = os.environ.get("SOLARLAB_IONIC_AC_INITIAL_MANIFEST")
    response_manifest = os.environ.get("SOLARLAB_IONIC_AC_RESPONSE_MANIFEST")
    if not initial_manifest or not response_manifest:
        pytest.skip("optional historical branch checks require explicit saved input/response manifests")
    compiled, initial = load_manifest_case(initial_manifest, kind)
    item = json.loads(Path(response_manifest).read_text())["responses"][kind + "_mobile"]
    result = recorded(item["path"], item["sha256"])
    assert result["compiled_identity"] == compiled.identity
    assert result["initial_provenance"]["source_sha256"] == initial.source_sha256
    np.testing.assert_array_equal(result["requested_frequencies_Hz"], [0.0, 1.0])
    assert len(result["attempts"]) == 2 and result["linear_solver_calls"] == 2
    for attempt in result["attempts"]:
        assert attempt["accepted"] and all(check["passed"] for check in attempt["checks"])
        assert set(attempt["observation"]["ion_density_perturbation_m3"]) == set(compiled.ion_species_names)
