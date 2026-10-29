"""Native time rows, reference-aware storage, Maxwell identity and saved prefix."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Iterator
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.multiple_ionic_domains import DOMAINS, INTERVALS, prepare_multiple_ionic_device
from examples.research_1d.multiple_ionic_steady import prepare_mixed_immobile_device
from examples.research_1d.ionic_steady import controls as steady_controls
from solarlab.device.compile_multiple_ionic_1d import CompiledMultipleIonicDevice1D, compile_multiple_ionic_device_1d
from solarlab.equations.multiple_ionic_1d import MultipleIonic1DProblem
from solarlab.experiments.multiple_ionic_transient import load_multiple_ionic_steady_initial, read_multiple_ionic_transient
from solarlab.experiments.transient_results import _decode, _encode
from solarlab.solvers.time.backward_euler import _storage_mask
from solarlab.solvers.time.ionic_backward_euler import IonicBackwardEuler1DStep
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.solvers.time.multiple_ionic_backward_euler import MultipleIonicBackwardEuler1DStep, multiple_ionic_step_accounting
from solarlab.solvers.nonlinear.multiple_ionic_1d import solve_multiple_ionic_steady_state

STEADY_SHA = "858ef2f39be16018049d49cfe171d3f79f03891c1259b0eff7701945a78ef7f3"
STEADY_PATH = Path("/Users/bytedance/Library/Mobile Documents/com~apple~CloudDocs/projects/solarlab/test/ArchitectureRefactor/Stage04/P04CoreV1/Core1D/MultipleIonDomainsV1/Solvers/Steady.json")
D0_SHA = "52fbe8cad99b739698459daaf428cba17acf5a8a70ea19de8a4fd9c4852651aa"
D0_FIXTURE = Path(__file__).with_name("fixtures") / "multiple_ionic_d0_steady.json.gz"


@pytest.fixture(scope="module")
def d0_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Materialize the exact reviewed initial state without a new steady solve."""
    payload = gzip.decompress(D0_FIXTURE.read_bytes())
    assert hashlib.sha256(payload).hexdigest() == D0_SHA
    path = tmp_path_factory.mktemp("multiple-ionic-d0") / "MixedSteady.json"
    path.write_bytes(payload)
    return path


@pytest.fixture(scope="module")
def current_d0_initial(d0_path: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """Validate one current-source initial from historical coordinates as a seed.

    Configuration source IDs include the checkout path. The archived accepted
    result is kept intact; it cannot be relabelled as a current-platform result.
    """
    original = _decode(json.loads(d0_path.read_text()))["result"]
    device = compile_multiple_ionic_device_1d(prepare_mixed_immobile_device(),
        intervals_per_layer=INTERVALS, ion_domains=DOMAINS)
    voltage = original["state"]["electronic_state"]["voltage_V"]
    seed = device.state(original["state"]["u"], voltage_V=voltage)
    result = solve_multiple_ionic_steady_state(device, voltage_V=voltage, generation_scale=0.0,
        options=steady_controls(4), initial_state=seed)
    assert result.accepted, (result.termination, result.error, result.checks)
    path = tmp_path_factory.mktemp("current-multiple-ionic-d0") / "Steady.json"
    document = {"schema": "solarlab.multiple_ionic_steady.v1", "result": result,
                "seed_provenance": {"archived_sha256": D0_SHA, "use": "unverified_coordinates_only"},
                "qualification": False}
    path.write_text(json.dumps(_encode(document), indent=2, allow_nan=False) + "\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module", autouse=True)
def blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


def case(kind: str = "positive") -> tuple[CompiledMultipleIonicDevice1D, MultipleIonicBackwardEuler1DStep, np.ndarray]:
    device = compile_multiple_ionic_device_1d(prepare_multiple_ionic_device(kind=kind),
        intervals_per_layer=INTERVALS, ion_domains=DOMAINS)
    old = device.initial_state(voltage_V=0.002).u.copy()
    old[3 * device.node_count:] = np.linspace(-0.12, 0.13, device.state_size - 3 * device.node_count)
    previous = device.state(old, voltage_V=0.002)
    endpoint = MultipleIonic1DProblem(device, voltage_V=0.003, generation_scale=0.0)
    step = MultipleIonicBackwardEuler1DStep(endpoint, previous, 0.25)
    y = step.initial_coordinates()
    y[3 * device.node_count:] += np.linspace(-0.01, 0.02, device.state_size - 3 * device.node_count)
    return device, step, y


@pytest.mark.parametrize("kind", ["positive", "independent", "shared"])
def test_every_native_domain_time_row_matches_existing_physical_secant(kind: str) -> None:
    device, step, y = case(kind)
    actual = step.residual(y)
    for domain, previous in zip(device.domains, step.previous.domain_component_states):
        local = IonicBackwardEuler1DStep(Ionic1DProblem(domain, 0.003, 0.0), previous, step.dt_s)
        expected = local.residual(device.local_coordinates(y, domain.domain_layer_id))
        np.testing.assert_array_equal(actual[device.domain_slices[domain.domain_layer_id]], expected[3 * device.node_count:])
    mask = _storage_mask(device.electronic_component)
    carrier = step.electronic_step.residual(y[:3 * device.node_count])[device.node_count:].reshape(2, device.node_count)
    np.testing.assert_array_equal(actual[device.node_count:3 * device.node_count].reshape(2, device.node_count), carrier)
    assert np.count_nonzero(~mask) == 4


@pytest.mark.parametrize("kind", ["positive", "independent", "shared"])
def test_full_BE_jacobian_with_nonzero_old_coordinates_and_changed_voltage(kind: str) -> None:
    device, step, y = case(kind)
    rng = np.random.default_rng(81459)
    direction = rng.uniform(-0.1, 0.1, device.state_size)
    direction[:device.node_count] *= device.electronic_component.thermal_voltage_V
    J = step.jacobian(y)
    h = 1e-5
    finite = (step.residual(y + h * direction) - step.residual(y - h * direction)) / (2 * h)
    error = float(np.max(abs(finite - J @ direction) / np.maximum(abs(J) @ abs(direction), np.finfo(float).tiny)))
    print(f"native two-domain BE {kind}: directional error {error:.9g}")
    assert error < 2e-7
    np.testing.assert_array_equal(J.toarray(),
        (step.endpoint.jacobian(y) + step.endpoint.storage_jacobian(y) / step.dt_s).toarray())


@pytest.mark.parametrize("kind", ["positive", "shared"])
def test_harmonic_seed_preserves_populations_and_maxwell_charge_accounting(kind: str) -> None:
    device, step, y = case(kind)
    seed = step.initial_coordinates()
    current = device.state(seed, voltage_V=step.endpoint.voltage_V)
    for before, after in zip(step.previous.domain_component_states, current.domain_component_states):
        np.testing.assert_array_equal(before.positive_ion_density_m3, after.positive_ion_density_m3)
        np.testing.assert_array_equal(before.negative_ion_density_m3, after.negative_ion_density_m3)
    for name in ("electron_density_m3", "hole_density_m3"):
        np.testing.assert_allclose(getattr(current.electronic_state, name),
                                   getattr(step.previous.electronic_state, name), rtol=2e-14, atol=0.0)
    data = multiple_ionic_step_accounting(step, y)
    scale = max(1.0, float(np.max(abs(data["total_current_density_A_m2"]))))
    assert np.max(abs(data["maxwell_identity_error_A_m2"])) / scale < 1e-12
    assert np.max(abs(data["readback_maxwell_identity_error_A_m2"])) / scale < 1e-12
    np.testing.assert_array_equal(data["total_current_density_A_m2"],
        data["conduction_current_density_A_m2"] + data["displacement_current_density_A_m2"])
    np.testing.assert_array_equal(data["ionic_conduction_current_A_m2"][[0, 1, 2, 6, 7, 11, 12, 13]], np.zeros(8))
    for record in data["domains"].values():
        for values in record["species"].values():
            np.testing.assert_array_equal(values["native_BE_rows_m2_s"], values["balance_m2_s"])
            assert abs(values["inventory_balance_identity_error_m2_s"]) <= values["inventory_balance_roundoff_bound_m2_s"]


def test_step_rejects_wrong_identity_bad_dt_and_shape() -> None:
    device, step, y = case()
    for dt in (True, 0.0, -1.0, float("inf")):
        with pytest.raises(ValueError):
            MultipleIonicBackwardEuler1DStep(step.endpoint, step.previous, dt)
    with pytest.raises(ValueError):
        MultipleIonicBackwardEuler1DStep(step.endpoint, replace(step.previous, device_sha256="wrong"), 0.25)
    with pytest.raises(ValueError):
        step.residual(np.r_[y, 0.0])


def test_solver_free_saved_reader_preserves_failed_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    device, step, _ = case()
    state = step.previous
    requested = np.asarray([0.0, 0.25, 0.5])
    document = {"schema": "solarlab.multiple_ionic_transient.v1", "completed": False, "result": {
        "protocol": {"times_s": requested}, "accepted_times_s": requested[:1], "states": (state,),
        "attempts": ({"start_s": 0.0, "end_s": 0.25, "accepted": False,
                      "termination": "synthetic_failure", "error": "synthetic saved prefix"},),
        "compiled_identity": device.identity, "solver_controls": {"inventory_row_replacement": False}}}
    path = tmp_path / "Failed.json"
    path.write_text(json.dumps(_encode(document), indent=2) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr("solarlab.experiments.multiple_ionic_transient.solve_newton",
                        lambda *args, **kwargs: pytest.fail("reader invoked Newton"))
    loaded = read_multiple_ionic_transient(path, digest)
    assert not loaded["completed"] and len(loaded["result"]["states"]) == 1
    with pytest.raises(ValueError, match="SHA256"):
        read_multiple_ionic_transient(path, "wrong")
    cli = subprocess.run([sys.executable, "-B", "-m", "examples.research_1d.multiple_ionic_transient",
                          "read", "--result", str(path), "--sha256", digest],
                         capture_output=True, text=True, timeout=10)
    assert cli.returncode == 2, cli.stderr
    output = json.loads(cli.stdout)
    assert output["completed"] is False
    assert output["first_failed_step"]["end_s"] == 0.25
    assert output["first_failed_step"]["termination"] == "synthetic_failure"


def test_actual_four_step_history_and_reuse_saved_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One real regression from supplied saved DC evidence; never a cold solve."""
    from examples.research_1d.multiple_ionic_transient import run_case
    if not os.environ.get("PYTEST_MULTIPLE_TRANSIENT_INITIAL"):
        pytest.skip("requires explicit PYTEST_MULTIPLE_TRANSIENT_INITIAL bound to this source; historical absolute paths are not portable")
    path = Path(os.environ["PYTEST_MULTIPLE_TRANSIENT_INITIAL"])
    if not path.exists():
        pytest.skip("requires the reviewed saved two-domain DC artifact; no cold substitute")
    output = Path(os.environ.get("SOLARLAB_MULTIPLE_TRANSIENT_OUTPUT", str(tmp_path / "History.json")))
    result = run_case(path, STEADY_SHA, output)
    assert result.completed, [(a.termination, a.error) for a in result.failed_attempts]
    assert len(result.states) == 5 and len(result.attempts) == 4
    np.testing.assert_array_equal(result.accepted_times_s, [0.0, 0.25, 0.5, 0.75, 1.0])
    for attempt in result.attempts:
        assert attempt.accepted and attempt.newton is not None and all(check.passed for check in attempt.checks)
        assert attempt.accounting is not None
        for domain in attempt.accounting["domains"].values():
            for values in domain["species"].values():
                assert values["balance_l1_m2_s"] <= 1e6
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    monkeypatch.setattr("solarlab.experiments.multiple_ionic_transient.solve_newton",
                        lambda *args, **kwargs: pytest.fail("saved read reran the history"))
    assert read_multiple_ionic_transient(output, digest)["completed"]


def test_D0_bound_initial_and_full_native_shared_mass(tmp_path: Path, current_d0_initial: tuple[Path, str]) -> None:
    d0_path, d0_sha = current_d0_initial
    device = compile_multiple_ionic_device_1d(prepare_mixed_immobile_device(),
        intervals_per_layer=INTERVALS, ion_domains=DOMAINS)
    initial = load_multiple_ionic_steady_initial(device, d0_path, d0_sha, controls=steady_controls(4))
    step = MultipleIonicBackwardEuler1DStep(MultipleIonic1DProblem(device, 0.0025, 0.0), initial.state, 0.25)
    y = step.initial_coordinates()
    domain = device.domains[1]
    plus, minus = device.ion_species_slices["absorber_right"].values()
    previous = initial.state.domain_component_states[1]
    positive = previous.positive_ion_density_m3 * 1.0001
    negative = previous.negative_ion_density_m3
    capacity = domain.positive_ion_site_limit_m3
    vacancy = 1.0 - (positive + negative) / capacity
    y[plus] = np.log(positive / capacity) - np.log(vacancy) - domain.positive_ion_logit_reference
    y[minus] = np.log(negative / capacity) - np.log(vacancy) - domain.negative_ion_logit_reference
    assert np.any(y[minus] != initial.state.u[minus])
    np.testing.assert_allclose(device.state(y, voltage_V=0.0025).domain_component_states[1].negative_ion_density_m3,
                               negative, rtol=2e-15, atol=0.0)
    assert step.endpoint.storage_jacobian(y)[minus, plus].nnz == 4
    assert np.max(abs(step.residual(y)[minus])) < 1e6
    raw = _decode(json.loads(d0_path.read_text()))
    raw["result"]["solver_controls"]["immobile_profile_binding"]["absorber_right"]["species"]["negative"]["target_density_m3"][0] *= 1.1
    bad = tmp_path / "WrongBinding.json"
    bad.write_text(json.dumps(_encode(raw), indent=2) + "\n")
    with pytest.raises(ValueError, match="D0 profile binding"):
        load_multiple_ionic_steady_initial(device, bad, hashlib.sha256(bad.read_bytes()).hexdigest(),
                                         controls=steady_controls(4))


def test_actual_D0_four_step_history(tmp_path: Path, current_d0_initial: tuple[Path, str]) -> None:
    from examples.research_1d.multiple_ionic_transient import run_case
    d0_path, d0_sha = current_d0_initial
    output = Path(os.environ.get("SOLARLAB_MULTIPLE_D0_TIME_OUTPUT", str(tmp_path / "D0History.json")))
    result = run_case(d0_path, d0_sha, output, mixed_immobile=True)
    assert result.completed, [(a.termination, a.error) for a in result.failed_attempts]
    assert len(result.states) == 5 and len(result.attempts) == 4
    for attempt in result.attempts:
        assert attempt.accounting is not None and attempt.accepted
        data = attempt.accounting["domains"]["absorber_right"]["species"]["negative"]
        np.testing.assert_array_equal(data["domain_flux_m2_s"], np.zeros(5))
        assert data["immobile_profile"]["profile_relative_error"] <= 1e-12
        assert data["balance_l1_m2_s"] <= 1e6
        assert all(check.passed for check in attempt.checks)
    assert read_multiple_ionic_transient(output, hashlib.sha256(output.read_bytes()).hexdigest())["completed"]
