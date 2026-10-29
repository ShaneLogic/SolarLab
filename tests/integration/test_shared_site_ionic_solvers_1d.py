"""Coupled shared-capacity inventories, physical secants and a bounded real history."""

from dataclasses import replace
from decimal import Decimal, localcontext
import hashlib
import json
import math

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_steady import controls as steady_controls
from examples.research_1d.ionic_transient import controls as time_controls, protocol
from examples.research_1d.shared_site_ionic import compile_case, run_wave
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.experiments import ionic_transient as driver
from solarlab.solvers.nonlinear.ionic_1d import ConstrainedIonic1DProblem, diagnostics as ionic_diagnostics
from solarlab.solvers.time.ionic_backward_euler import IonicBackwardEuler1DStep, step_accounting


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def case():
    device = compile_case()
    original = Ionic1DProblem(device, voltage_V=0.002)
    constrained = ConstrainedIonic1DProblem(original, device.ion_node_count - 1)
    y = device.initial_state(voltage_V=0.002).u.copy()
    N, M = device.node_count, device.ion_node_count
    x, xp = np.linspace(0, np.pi, N), np.linspace(0, np.pi, M)
    y[:N] += 0.001 * np.sin(x)
    y[N:2 * N] += 0.02 * np.sin(x)
    y[2 * N:3 * N] -= 0.03 * np.sin(x)
    y[device.ion_species_slices[0]] += 0.04 * np.cos(xp)
    y[device.ion_species_slices[1]] -= 0.03 * np.cos(xp)
    return device, original, constrained, y


def directional_error(problem, y, N):
    direction = np.random.default_rng(8813).uniform(-0.05, 0.05, y.size)
    direction[:N] *= 0.025
    h = 1e-5
    matrix = problem.jacobian(y)
    measured = (problem.residual(y + h * direction) - problem.residual(y - h * direction)) / (2 * h)
    scale = np.maximum(abs(matrix) @ abs(direction), np.finfo(float).tiny)
    return float(np.max(abs(matrix @ direction - measured) / scale))


def test_each_constraint_uses_full_coupled_storage_gradient_and_preserves_other_rows(case):
    device, original, constrained, y = case
    assert (device.node_count, device.ion_node_count, device.state_size) == (37, 13, 137)
    before, after = original.residual(y), constrained.residual(y)
    old_J, J = original.jacobian(y).toarray(), constrained.jacobian(y).toarray()
    K = original.storage_jacobian(y).toarray()
    populations = device.fields_by_species(y, voltage_V=0.002)[3]
    mask = np.ones(device.state_size, dtype=bool)
    mask[list(constrained.constraint_rows.values())] = False
    np.testing.assert_array_equal(after[mask], before[mask])
    np.testing.assert_array_equal(J[mask], old_J[mask])
    for species, block, density in zip(device.ion_species_names, device.ion_species_slices, populations):
        row = constrained.constraint_rows[species]
        inventory = math.fsum((device.ion_geometry.cell_measures_m * density).tolist())
        assert after[row] == inventory - constrained.inventory_targets_m2[species]
        np.testing.assert_array_equal(J[row], -np.sum(K[block], axis=0))
        own = device.ion_species_names.index(species)
        partner = device.ion_species_slices[1 - own]
        assert np.all(J[row, block] > 0.0) and np.all(J[row, partner] < 0.0)
        expected_cross = -device.ion_geometry.cell_measures_m * populations[0] * populations[1] / device.positive_ion_site_limit_m3
        np.testing.assert_allclose(J[row, partner], expected_cross, rtol=3e-15)
    error = directional_error(constrained, y, device.node_count)
    print(f"shared constrained J directional error={error:.9g}")
    assert error < 3e-7


def test_original_dropped_rows_and_each_signed_charge_deposition_remain_visible(case):
    device, original, constrained, y = case
    info = ionic_diagnostics(constrained, y)
    for species, block in zip(device.ion_species_names, device.ion_species_slices):
        one = info["ion_species"][species]
        np.testing.assert_array_equal(one["ion_balance_m2_s"], original.residual(y)[block])
        assert one["original_dropped_balance_m2_s"] == original.residual(y)[constrained.constraint_rows[species]]
        np.testing.assert_allclose(one["deposited_ion_net_charge_C_m2"], one["physical_ion_net_charge_C_m2"],
                                   rtol=5e-13, atol=1e-18)


def shared_delta_oracle(device, before, after, node):
    """100-digit coupled softmax difference of the represented input numbers."""
    def d(value):
        return Decimal.from_float(float(value))

    with localcontext() as context:
        context.prec = 100
        references = (device.positive_ion_logit_reference[node], device.negative_ion_logit_reference[node])
        capacity = d(device.positive_ion_site_limit_m3[node])

        def densities(coordinates):
            weights = [(d(reference) + d(coordinate)).exp()
                       for reference, coordinate in zip(references, coordinates)]
            denominator = Decimal(1) + sum(weights)
            return tuple(capacity * value / denominator for value in weights)

        old, new = densities(before), densities(after)
        return tuple(float(b - a) for a, b in zip(old, new))


@pytest.mark.parametrize("changes", [
    (np.nextafter(0.4, np.inf), -0.2),
    (0.4, np.nextafter(-0.2, -np.inf)),
    (0.40000000000001, -0.19999999999999),
    (0.41, -0.23),
])
def test_physical_BE_secants_include_partner_motion_and_match_decimal_oracle(case, changes):
    device, _, _, _ = case
    blocks = device.ion_species_slices
    old_y = device.initial_state(voltage_V=0.002).u.copy()
    old_y[blocks[0]], old_y[blocks[1]] = 0.4, -0.2
    old = device.state(old_y, voltage_V=0.002)
    step = IonicBackwardEuler1DStep(Ionic1DProblem(device, voltage_V=0.004), old, 0.25)
    y = step.initial_coordinates(voltage_lift=True)
    y[blocks[0]], y[blocks[1]] = changes
    _, rates = step.inventory_rates_m2_s(y)
    for node in (0, device.ion_node_count // 2, device.ion_node_count - 1):
        expected = shared_delta_oracle(device, (0.4, -0.2), changes, node)
        measured = [rate[node] * step.dt_s / device.ion_geometry.cell_measures_m[node] for rate in rates]
        np.testing.assert_allclose(measured, expected, rtol=3e-13, atol=0.0)
    if changes[1] == -0.2:
        assert np.all(rates[0] > 0.0) and np.all(rates[1] < 0.0)
    residual, original = step.residual(y), step.endpoint.residual(y)
    for block, rate in zip(blocks, rates):
        np.testing.assert_array_equal(residual[block], original[block] - rate)
    np.testing.assert_array_equal(residual[:device.node_count], original[:device.node_count])


@pytest.fixture(scope="module")
def moving(case):
    device, _, _, old_y = case
    old = device.state(old_y, voltage_V=0.002)
    step = IonicBackwardEuler1DStep(Ionic1DProblem(device, voltage_V=0.004), old, 0.25)
    y = step.initial_coordinates(voltage_lift=True)
    for index, block in enumerate(device.ion_species_slices):
        y[block] += (index + 1) * 0.01 * np.sin(np.linspace(0, np.pi, device.ion_node_count))
    return device, step, y


def test_BE_J_has_all_coupled_storage_entries_and_no_inventory_substitution(moving):
    device, step, y = moving
    J = step.jacobian(y)
    expected = step.endpoint.jacobian(y) + step.endpoint.storage_jacobian(y) / step.dt_s
    np.testing.assert_array_equal(J.toarray(), expected.toarray())
    positive, negative = device.ion_species_slices
    storage = step.endpoint.storage_jacobian(y)
    assert storage[positive, negative].nnz == device.ion_node_count
    assert storage[negative, positive].nnz == device.ion_node_count
    error = directional_error(step, y, device.node_count)
    print(f"shared physical BE J directional error={error:.9g}")
    assert error < 2e-7


def test_shared_motion_has_signed_all_face_current_and_complete_Maxwell_identity(moving):
    device, step, y = moving
    info = step_accounting(step, y)
    species = info["ion_species"]
    expected_current = Q * (species["positive"]["ion_particle_fluxes_m2_s"]
                            - species["negative"]["ion_particle_fluxes_m2_s"])
    np.testing.assert_array_equal(info["ion_conduction_current_density_A_m2"], expected_current)
    carriers, rates = step.inventory_rates_m2_s(y)
    expected_charge_rate = Q * (carriers[1] - carriers[0])
    expected_charge_rate[device.ion_node_indices] += Q * (rates[0] - rates[1])
    np.testing.assert_array_equal(info["charge_inventory_rate_A_m2"], expected_charge_rate)
    for name, block in zip(device.ion_species_names, device.ion_species_slices):
        np.testing.assert_array_equal(species[name]["ion_balance_m2_s"], step.residual(y)[block])
    scale = max(1.0, float(np.max(abs(info["total_current_density_A_m2"]))),
                float(np.max(abs(info["predicted_maxwell_cell_jump_A_m2"]))))
    assert np.max(abs(info["maxwell_identity_error_A_m2"])) < 128 * np.finfo(float).eps * scale
    assert np.max(abs(info["readback_maxwell_identity_error_A_m2"])) < 256 * np.finfo(float).eps * scale
    assert info["total_current_density_A_m2"].shape == (38,)
    assert info["physical_certificate_status"] == "not_evaluated"


def test_shared_dynamic_norm_keeps_existing_independent_species_limits(case):
    device, _, _, _ = case
    settings, norm = driver._newton_settings(device, time_controls())
    for block in device.ion_species_slices:
        np.testing.assert_array_equal(settings.scaling.row_weights[block], np.full(13, 13 / 1e6))
        values = np.zeros(device.state_size)
        values[block] = 1.01
        assert norm(values) == 1.01
    assert not settings.gates


@pytest.fixture(scope="module")
def wave(tmp_path_factory):
    directory = tmp_path_factory.mktemp("shared_wave")
    device, steady, history = run_wave(directory)
    return device, steady, history, directory


def test_real_high_occupancy_cold_steady_keeps_both_inventories_and_capacity(wave):
    device, steady, _, _ = wave
    assert steady.accepted, (steady.termination, steady.error)
    assert steady.newton.accepted and not steady.qualification
    assert all(check.passed for check in steady.checks)
    total = steady.state.positive_ion_density_m3 + steady.state.negative_ion_density_m3
    assert np.all(total > 0.0) and np.all(total < device.positive_ion_site_limit_m3)
    assert np.max(total / device.positive_ion_site_limit_m3) > 0.5
    np.testing.assert_array_equal(steady.initial_state.u, device.initial_state(voltage_V=0.002).u)
    for species in device.ion_species_names:
        one = steady.diagnostics["ion_species"][species]
        assert one["inventory_relative_error"] <= 1e-12
        assert one["ion_balance_l1_m2_s"] <= 1e6 and one["ion_flux_max_m2_s"] <= 1e6
        assert abs(one["original_dropped_balance_m2_s"]) <= 1e6
        assert np.max(abs(getattr(steady.state, f"{species}_ion_density_m3")
                          / getattr(device, f"{species}_ion_reference_m3") - 1)) > 0.001


def test_real_four_steps_keep_all_species_active_and_inventory_balance(wave):
    device, steady, history, _ = wave
    assert steady.accepted and history is not None
    assert history.completed, [(a.termination, a.error) for a in history.failed_attempts]
    assert len(history.attempts) == 4 and len(history.states) == 5
    np.testing.assert_array_equal(history.accepted_times_s, protocol(4).times_s)
    np.testing.assert_array_equal(history.states[0].u, steady.state.u)
    for species in device.ion_species_names:
        density = f"{species}_ion_density_m3"
        assert np.max(abs(getattr(history.states[-1], density) / getattr(steady.state, density) - 1)) > 1e-6
        initial_inventory = steady.state.domain_inventories_m2[species]
        bound = 0.0
        for index, attempt in enumerate(history.attempts, 1):
            one = attempt.accounting["ion_species"][species]
            assert math.fsum(abs(one["ion_balance_m2_s"]).tolist()) <= 1e6
            bound += (attempt.end_s - attempt.start_s) * (
                math.fsum(abs(one["ion_balance_m2_s"]).tolist())
                + math.fsum(abs(one["ion_inventory_readback_error_m2_s"]).tolist()))
            assert abs(history.states[index].domain_inventories_m2[species] - initial_inventory) <= bound + 2 * index * np.spacing(initial_inventory)
    for state in history.states:
        assert np.all(state.positive_ion_density_m3 + state.negative_ion_density_m3 < device.positive_ion_site_limit_m3)
    for attempt in history.attempts:
        assert attempt.accounting["total_current_density_A_m2"].shape == (38,)
        assert np.max(abs(attempt.accounting["maxwell_identity_error_A_m2"])) < 1e-12
    assert history.physical_certificate_status == "not_evaluated"


def test_shared_initial_pin_cannot_rebind_the_site_law_or_partner_density(wave, tmp_path):
    device, steady, _, directory = wave
    assert steady.accepted
    for mutation in ("site_law", "negative_density"):
        document = json.loads((directory / "Steady.json").read_text())
        if mutation == "site_law":
            document["result"]["binding_manifest"]["ion_model"]["site_contract"] = "independent_finite_sites"
        else:
            document["result"]["state"]["negative_ion_density_m3"][0] *= 1.001
        raw = json.dumps(document, allow_nan=False).encode()
        path = tmp_path / (mutation + ".json")
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            driver.load_ionic_steady_initial(device, path, hashlib.sha256(raw).hexdigest(), controls=steady_controls(13))


def test_real_prefix_and_first_refusal_survive_without_repeating_a_solve(wave, monkeypatch):
    device, _, history, directory = wave
    assert history is not None and history.completed
    path = directory / "Steady.json"
    initial = driver.load_ionic_steady_initial(device, path, hashlib.sha256(path.read_bytes()).hexdigest(), controls=steady_controls(13))
    calls = []

    def first_then_stop(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return history.attempts[0].newton
        raise ValueError("injected shared second-step refusal")

    monkeypatch.setattr(driver, "solve_newton", first_then_stop)
    result = driver.solve_ionic_transient(device, initial, protocol(4), options=time_controls())
    assert len(calls) == 2 and len(result.states) == 2 and len(result.failed_attempts) == 1
    np.testing.assert_array_equal(result.states[-1].u, history.states[1].u)
    np.testing.assert_array_equal(result.accepted_times_s, [0.0, 0.25])
    assert not result.completed and "second-step refusal" in result.failed_attempts[0].error
    forged = replace(initial, state=replace(initial.state,
        negative_ion_density_m3=initial.state.negative_ion_density_m3 * 1.001))
    with pytest.raises(ValueError, match="pinned steady evidence"):
        driver.solve_ionic_transient(device, forged, protocol(4), options=time_controls())
