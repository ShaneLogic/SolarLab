"""Independent inventories, all dynamic rows, signed accounting and real dual motion."""

from dataclasses import replace
import hashlib
import json
import math

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.dual_ionic import compile_case, run_wave
from examples.research_1d.ionic_multilayer import prepare_confined_dual_ion_device
from examples.research_1d.ionic_steady import controls as steady_controls
from examples.research_1d.ionic_transient import controls as time_controls, protocol
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.experiments import ionic_transient as driver
from solarlab.solvers.nonlinear.ionic_1d import (
    ConstrainedIonic1DProblem, checks as ionic_checks, diagnostics as ionic_diagnostics, solve_ionic_steady_state,
)
from solarlab.solvers.time.ionic_backward_euler import (
    IonicBackwardEuler1DStep, finite_site_secant_m3, inventory_m2, step_accounting,
)


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def case():
    device = compile_case()
    original = Ionic1DProblem(device, voltage_V=0.002, generation_scale=0.0)
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


def test_two_particle_constraints_preserve_all_other_rows_and_own_exact_J(case):
    device, original, constrained, y = case
    assert (device.node_count, device.ion_node_count, device.state_size) == (37, 13, 137)
    before, after = original.residual(y), constrained.residual(y)
    original_J, J = original.jacobian(y).toarray(), constrained.jacobian(y).toarray()
    other = np.ones(device.state_size, dtype=bool)
    other[list(constrained.constraint_rows.values())] = False
    np.testing.assert_array_equal(after[other], before[other])
    np.testing.assert_array_equal(J[other], original_J[other])
    populations = device.fields_by_species(y, voltage_V=0.002)[3]
    for species, block, density in zip(device.ion_species_names, device.ion_species_slices, populations):
        row = constrained.constraint_rows[species]
        physical = math.fsum((device.ion_geometry.cell_measures_m * density).tolist())
        assert after[row] == physical - constrained.inventory_targets_m2[species]
        expected = np.zeros(device.state_size)
        expected[block] = -original.storage_jacobian(y).diagonal()[block]
        np.testing.assert_array_equal(J[row], expected)
        assert np.all(J[row, block] > 0.0)  # negative ions still have positive particle inventories
        assert original_J[:device.node_count, block].any()
        assert original_J[block, :device.node_count].any()


def directional_error(problem, y, N):
    direction = np.random.default_rng(7752).uniform(-0.05, 0.05, y.size)
    direction[:N] *= 0.025
    h = 1e-5
    J = problem.jacobian(y)
    difference = (problem.residual(y + h * direction) - problem.residual(y - h * direction)) / (2 * h)
    scale = np.maximum(abs(J) @ abs(direction), np.finfo(float).tiny)
    return float(np.max(abs(J @ direction - difference) / scale))


def test_constrained_directional_J_includes_both_inventory_rows(case):
    device, _, constrained, y = case
    error = directional_error(constrained, y, device.node_count)
    print(f"dual constrained J directional error={error:.9g}")
    assert error < 3e-7


def test_steady_diagnostics_keep_both_dropped_rows_and_signed_deposition(case):
    device, original, constrained, y = case
    info = ionic_diagnostics(constrained, y)
    for species, block in zip(device.ion_species_names, device.ion_species_slices):
        one = info["ion_species"][species]
        np.testing.assert_array_equal(one["ion_balance_m2_s"], original.residual(y)[block])
        assert one["original_dropped_balance_m2_s"] == original.residual(y)[constrained.constraint_rows[species]]
        assert one["ion_balance_l1_m2_s"] == math.fsum(abs(original.residual(y)[block]).tolist())
        np.testing.assert_allclose(one["deposited_ion_net_charge_C_m2"], one["physical_ion_net_charge_C_m2"],
                                   rtol=5e-13, atol=1e-18)
    np.testing.assert_array_equal(info["ionic_current_density_A_m2"],
        np.r_[0.0, Q * (original.ion_fluxes_on_electronic_faces(y)
                        - original.ion_fluxes_on_electronic_faces(y, species="negative")), 0.0])
    checks = {check.name: check for check in ionic_checks(info, steady_controls(device.ion_node_count))}
    assert checks["negative_ion_balance_l1_m2_s"].value == info["ion_species"]["negative"]["ion_balance_l1_m2_s"]
    assert checks["negative_inventory_relative_error"].limit == 1e-12


@pytest.mark.parametrize("species", ["positive", "negative"])
def test_zero_D_species_uses_local_profiles_while_mobile_species_keeps_inventory(species):
    prepared = prepare_confined_dual_ion_device(**{f"{species}_diffusivity_m2_s": 0.0})
    device = compile_ionic_device_1d(prepared, intervals_per_layer=(12, 12, 12),
        ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"))
    assert device.state_size == 137 and device.initial_state().domain_inventories_m2[species] > 0
    original = Ionic1DProblem(device)
    constrained = ConstrainedIonic1DProblem(original, device.ion_node_count - 1)
    blocks = dict(zip(device.ion_species_names, device.ion_species_slices))
    block = blocks[species]
    assert constrained.immobile_species == (species,)
    assert constrained.profile_constraint_rows == {species: tuple(range(block.start, block.stop))}
    y = device.initial_state().u.copy()
    y[block.start] += 0.02
    state = device.state(y)
    density = getattr(state, species + "_ion_density_m3")
    target = getattr(device, species + "_ion_reference_m3")
    expected = device.ion_geometry.cell_measures_m * (density - target)
    np.testing.assert_array_equal(original.residual(y)[block], 0.0)
    np.testing.assert_array_equal(constrained.residual(y)[block], expected)
    np.testing.assert_array_equal(constrained.jacobian(y)[block, :].toarray(),
                                  -original.storage_jacobian(y)[block, :].toarray())
    assert expected[0] > 0.0
    other = "negative" if species == "positive" else "positive"
    inventory = math.fsum((device.ion_geometry.cell_measures_m
                          * getattr(state, other + "_ion_density_m3")).tolist())
    assert constrained.residual(y)[constrained.constraint_rows[other]] == (
        inventory - constrained.inventory_targets_m2[other])


def test_steady_scaling_separates_species_and_rejects_wrong_negative_inventory(case, monkeypatch):
    device, _, constrained, _ = case
    settings_seen = []

    def stop_after_controls(problem, seed, settings):
        settings_seen.append(settings)
        raise RuntimeError("injected stop before any Newton work")

    monkeypatch.setattr("solarlab.solvers.nonlinear.ionic_1d.solve_newton", stop_after_controls)
    options = steady_controls(device.ion_node_count)
    result = solve_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0, options=options)
    assert not result.accepted and len(settings_seen) == 1
    assert result.evaluation_counts == {"residual": 0, "jacobian": 0}
    settings = settings_seen[0]
    for species, block in zip(device.ion_species_names, device.ion_species_slices):
        expected = np.full(device.ion_node_count, 2 * device.ion_node_count / 1e6)
        expected[-1] = 1 / (1e-12 * constrained.inventory_targets_m2[species])
        np.testing.assert_array_equal(settings.scaling.row_weights[block], expected)
        np.testing.assert_array_equal(settings.step_control.max_step[block], np.ones(device.ion_node_count))
    wrong = device.initial_state(voltage_V=0.002).u.copy()
    wrong[device.ion_species_slices[1]] += 0.01
    with pytest.raises(ValueError, match="prepared-domain inventory"):
        solve_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0, options=options,
                                 initial_state=device.state(wrong, voltage_V=0.002))
    assert len(settings_seen) == 1


@pytest.fixture(scope="module")
def moving(case):
    device, _, _, old_y = case
    old = device.state(old_y, voltage_V=0.002)
    step = IonicBackwardEuler1DStep(Ionic1DProblem(device, voltage_V=0.004), old, 0.25)
    y = step.initial_coordinates(voltage_lift=True)
    for index, block in enumerate(device.ion_species_slices):
        y[block] += (index + 1) * 0.01 * np.sin(np.linspace(0, np.pi, device.ion_node_count))
    return device, step, y


def test_BE_retains_every_species_row_and_the_original_exact_storage_J(moving):
    device, step, y = moving
    N = device.node_count
    dc, residual = step.endpoint.residual(y), step.residual(y)
    np.testing.assert_array_equal(residual[:N], dc[:N])
    for species, block in zip(device.ion_species_names, device.ion_species_slices):
        _, rate = step.inventory_rate_m2_s(y, species=species)
        expected = device.ion_geometry.cell_measures_m * finite_site_secant_m3(
            getattr(device, f"{species}_ion_site_limit_m3"), getattr(device, f"{species}_ion_logit_reference"),
            step.previous.u[block], y[block]) / step.dt_s
        np.testing.assert_array_equal(rate, expected)
        np.testing.assert_array_equal(residual[block], dc[block] - expected)
    expected_J = step.endpoint.jacobian(y) + step.endpoint.storage_jacobian(y) / step.dt_s
    np.testing.assert_array_equal(step.jacobian(y).toarray(), expected_J.toarray())
    error = directional_error(step, y, N)
    print(f"dual physical BE J directional error={error:.9g}")
    assert error < 2e-7


def test_negative_secant_retains_tiny_motion_and_rebase_preserves_both_populations(case):
    device, _, _, _ = case
    old_y = device.initial_state(voltage_V=0.002).u.copy()
    block = device.ion_species_slices[1]
    old_y[block] = 0.4
    old = device.state(old_y, voltage_V=0.002)
    step = IonicBackwardEuler1DStep(Ionic1DProblem(device, voltage_V=0.004), old, 0.25)
    y = step.initial_coordinates(voltage_lift=True)
    for species in device.ion_species_names:
        np.testing.assert_array_equal(step.inventory_rate_m2_s(y, species=species)[1], np.zeros(device.ion_node_count))
    y[block] = np.nextafter(0.4, np.inf)
    _, rate = step.inventory_rate_m2_s(y, species="negative")
    assert np.all(rate > 0.0)
    rounded = (inventory_m2(device, device.state(y, voltage_V=0.004), species="negative")[1]
               - inventory_m2(device, old, species="negative")[1]) / step.dt_s
    assert np.any(rate != rounded)
    forged = replace(old, negative_ion_density_m3=old.negative_ion_density_m3 * 1.01)
    with pytest.raises(ValueError, match="fields/inventory"):
        IonicBackwardEuler1DStep(step.endpoint, forged, step.dt_s)


def test_motion_has_signed_all_face_current_and_independent_species_accounting(moving):
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
    assert info["total_current_density_A_m2"].shape == (device.node_count + 1,)
    for name, block in zip(device.ion_species_names, device.ion_species_slices):
        np.testing.assert_array_equal(species[name]["ion_balance_m2_s"], step.residual(y)[block])
        assert np.max(abs(species[name]["ion_domain_fluxes_m2_s"])) > 1e6
    scale = max(1.0, float(np.max(abs(info["total_current_density_A_m2"]))),
                float(np.max(abs(info["predicted_maxwell_cell_jump_A_m2"]))))
    assert np.max(abs(info["maxwell_identity_error_A_m2"])) < 128 * np.finfo(float).eps * scale
    assert np.max(abs(info["readback_maxwell_identity_error_A_m2"])) < 256 * np.finfo(float).eps * scale
    assert info["physical_certificate_status"] == "not_evaluated"


def test_dynamic_norm_neither_cancels_species_nor_averages_over_2M(case):
    device, _, _, _ = case
    settings, norm = driver._newton_settings(device, time_controls())
    M = device.ion_node_count
    scaled = np.zeros(device.state_size)
    for block in device.ion_species_slices:
        np.testing.assert_array_equal(settings.scaling.row_weights[block], np.full(M, M / 1e6))
        np.testing.assert_array_equal(settings.step_control.max_step[block], np.ones(M))
    for block in device.ion_species_slices:
        scaled[:] = 0
        scaled[block] = 1.01
        assert norm(scaled) == 1.01
    scaled[device.ion_species_slices[0]] = -1.01
    assert norm(scaled) == 1.01 and not settings.gates


@pytest.fixture(scope="module")
def wave(tmp_path_factory):
    directory = tmp_path_factory.mktemp("dual_wave")
    device, steady, history = run_wave(directory)
    return device, steady, history, directory


def test_real_cold_dual_steady_redistributes_both_populations_with_original_checks(wave):
    device, steady, _, _ = wave
    assert steady.accepted, (steady.termination, steady.error)
    assert steady.newton.accepted and not steady.qualification
    np.testing.assert_array_equal(steady.initial_state.u, device.initial_state(voltage_V=0.002).u)
    assert len(steady.checks) == 10 and all(check.passed for check in steady.checks)
    for species in device.ion_species_names:
        density = getattr(steady.state, f"{species}_ion_density_m3")
        assert np.max(abs(density / getattr(device, f"{species}_ion_reference_m3") - 1)) > 0.001
        info = steady.diagnostics["ion_species"][species]
        assert info["inventory_relative_error"] <= 1e-12
        assert info["ion_balance_l1_m2_s"] <= 1e6 and info["ion_flux_max_m2_s"] <= 1e6
        assert abs(info["original_dropped_balance_m2_s"]) <= 1e6


def test_real_four_step_history_moves_both_species_and_preserves_interval_balances(wave):
    device, steady, history, _ = wave
    assert steady.accepted and history is not None
    assert history.completed, [(a.termination, a.error) for a in history.failed_attempts]
    assert len(history.attempts) == 4 and len(history.states) == 5
    np.testing.assert_array_equal(history.accepted_times_s, protocol(4).times_s)
    np.testing.assert_array_equal(history.states[0].u, steady.state.u)
    assert history.physical_certificate_status == "not_evaluated"
    for species in device.ion_species_names:
        density = f"{species}_ion_density_m3"
        assert np.max(abs(getattr(history.states[-1], density) / getattr(steady.state, density) - 1)) > 1e-6
        initial_total = steady.state.domain_inventories_m2[species]
        bound = 0.0
        for index, attempt in enumerate(history.attempts, 1):
            info = attempt.accounting["ion_species"][species]
            assert math.fsum(abs(info["ion_balance_m2_s"]).tolist()) <= 1e6
            bound += (attempt.end_s - attempt.start_s) * (
                math.fsum(abs(info["ion_balance_m2_s"]).tolist())
                + math.fsum(abs(info["ion_inventory_readback_error_m2_s"]).tolist()))
            drift = abs(history.states[index].domain_inventories_m2[species] - initial_total)
            assert drift <= bound + 2 * index * np.spacing(initial_total)
    for attempt in history.attempts:
        info = attempt.accounting
        assert info["total_current_density_A_m2"].shape == (38,)
        assert np.max(abs(info["maxwell_identity_error_A_m2"])) < 1e-12


@pytest.mark.parametrize("mutation", ["negative_density", "negative_inventory", "missing_negative", "missing_check"])
def test_dual_initial_evidence_requires_both_fields_inventories_and_checks(wave, tmp_path, mutation):
    device, steady, _, directory = wave
    assert steady.accepted
    record = json.loads((directory / "Steady.json").read_text())
    state = record["result"]["state"]
    if mutation == "negative_density":
        state["negative_ion_density_m3"][0] *= 1.001
    elif mutation == "negative_inventory":
        state["negative_domain_inventory_m2"] *= 1.001
    elif mutation == "missing_negative":
        del state["negative_ion_density_m3"]
    else:
        record["result"]["checks"].pop()
    raw = json.dumps(record, allow_nan=False).encode()
    path = tmp_path / "Rebound.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        driver.load_ionic_steady_initial(device, path, hashlib.sha256(raw).hexdigest(),
                                          controls=steady_controls(device.ion_node_count))


def test_real_prefix_and_first_injected_failure_are_retained_without_retry(wave, monkeypatch):
    device, _, history, directory = wave
    assert history is not None and history.completed
    path = directory / "Steady.json"
    initial = driver.load_ionic_steady_initial(device, path, hashlib.sha256(path.read_bytes()).hexdigest(),
                                               controls=steady_controls(device.ion_node_count))
    calls = []

    def first_result_then_refuse(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return history.attempts[0].newton
        raise ValueError("injected second-step refusal; no repeated solve")

    monkeypatch.setattr(driver, "solve_newton", first_result_then_refuse)
    result = driver.solve_ionic_transient(device, initial, protocol(4), options=time_controls())
    assert len(calls) == 2 and len(result.states) == 2 and len(result.attempts) == 2
    assert not result.completed and len(result.failed_attempts) == 1
    np.testing.assert_array_equal(result.states[-1].u, history.states[1].u)
    np.testing.assert_array_equal(result.accepted_times_s, [0.0, 0.25])
    np.testing.assert_array_equal(result.protocol.voltage_V, protocol(4).voltage_V)
    assert result.failed_attempts[0].start_s == 0.25 and "second-step refusal" in result.failed_attempts[0].error
