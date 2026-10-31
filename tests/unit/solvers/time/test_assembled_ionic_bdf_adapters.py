"""Canonical storage and every domain/species of the new ionic BDF adapters."""

from dataclasses import asdict, is_dataclass
import hashlib
import json

import numpy as np
import pytest

from examples.research_1d.dual_ionic import compile_case as dual
from examples.research_1d.shared_site_ionic import compile_case as shared
from examples.research_1d.immobile_ions import prepare_immobile_ion_device
from examples.research_1d.multiple_ionic_domains import DOMAINS, INTERVALS, prepare_multiple_ionic_device
from examples.research_1d.multiple_ionic_steady import prepare_mixed_immobile_device
from examples.research_1d.steady import load_device
from examples.research_1d.transient import controls, json_value
from examples.research_1d.ionic_residual import prepare_single_ion_device
from examples.research_1d.ionic_steady import controls as steady_ionic_controls
from examples.research_1d.ionic_transient import controls as time_ionic_controls
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.compile_multiple_ionic_1d import compile_multiple_ionic_device_1d
from solarlab.equations.assembly.ionic_1d import (
    AssembledIonic1DProblem, AssembledMultipleIonic1DProblem,
    AssembledConstrainedIonic1DProblem, AssembledConstrainedMultipleIonic1DProblem,
)
from solarlab.equations.ionic_1d import CompiledIonicDevice1D
from solarlab.experiments.ionic_transient import load_ionic_steady_initial
from solarlab.experiments.transient import TransientProtocol
from solarlab.experiments.transient_results import _encode
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state
from solarlab.solvers.nonlinear.ionic_1d import solve_ionic_steady_state
from solarlab.solvers.time.assembled_ionic_backward_euler import (
    AssembledIonicBackwardEuler1DStep, AssembledMultipleIonicBackwardEuler1DStep,
)
from solarlab.solvers.time.assembled_ionic_bdf import assembled_ionic_bdf_accounting, assembled_ionic_bdf_seed
from solarlab.solvers.time.assembled_storage import AssembledIonicStorage1D
from solarlab.solvers.time.bdf import BDFOptions, BDFStep, HistoryPoint
from solarlab.experiments.bdf import solve_electronic_bdf, solve_ionic_bdf

CASES = ("ionic_dual37", "ionic_shared37", "ionic_shared_mixed37", "ionic_shared_full37",
         "multiple_positive", "multiple_independent", "multiple_shared", "multiple_mixed")


@pytest.fixture(scope="module")
def devices():
    result = {"ionic_dual37": dual(), "ionic_shared37": shared()}
    for name, kind in (("ionic_shared_mixed37", "shared_mixed"), ("ionic_shared_full37", "shared_full")):
        result[name] = compile_ionic_device_1d(prepare_immobile_ion_device(kind),
            intervals_per_layer=(12, 12, 12), ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"))
    for name in CASES[4:]:
        prepared = prepare_mixed_immobile_device() if name == "multiple_mixed" else prepare_multiple_ionic_device(kind=name.split("_")[1])
        result[name] = compile_multiple_ionic_device_1d(prepared, intervals_per_layer=INTERVALS, ion_domains=DOMAINS)
    return result


def adapter(device, voltage=.002):
    cls = AssembledIonic1DProblem if isinstance(device, CompiledIonicDevice1D) else AssembledMultipleIonic1DProblem
    return AssembledIonicStorage1D(cls(device, voltage, 0.0))


def domains(device):
    if isinstance(device, CompiledIonicDevice1D):
        return ((device, dict(zip(device.ion_species_names, device.ion_species_slices))),)
    return tuple((d, device.ion_species_slices[d.domain_layer_id]) for d in device.domains)


def step(device, order):
    a0, a1, a2 = adapter(device, .002), adapter(device, .003), adapter(device, .004)
    u0 = np.zeros(device.state_size)
    h0 = HistoryPoint(0.0, u0, a0.storage(u0), a0)
    first = BDFStep(a1, (h0,), .2, 1)
    u1 = assembled_ionic_bdf_seed(first)
    u1[3 * device.node_count:] += .001
    h1 = HistoryPoint(.2, u1, a1.storage(u1), a1)
    result = BDFStep(a2, (h1, h0), .45, order)
    u = assembled_ionic_bdf_seed(result)
    u[3 * device.node_count:] += .002
    return result, u


@pytest.mark.parametrize("name", CASES)
def test_every_species_physical_inventory_and_algebraic_storage_rows(devices, name):
    device = devices[name]
    a = adapter(device)
    u = np.zeros(device.state_size)
    state = device.state(u, voltage_V=.002)
    physical = a.physical_storage(u)
    equation = a.storage(u)
    tangent = a.storage_tangent(u)
    n = device.node_count
    np.testing.assert_array_equal(equation[:n], np.zeros(n))
    assert tangent[:n].nnz == 0
    component = device.electronic_component
    for species, offset in (("n", n), ("p", 2 * n)):
        for side, index in (("left", 0), ("right", n - 1)):
            if component.channel(species, side).kind == "fixed_reservoir":
                assert equation[offset + index] == 0
                assert tangent[offset + index].nnz == 0
    for domain, blocks in domains(device):
        local = state if isinstance(device, CompiledIonicDevice1D) else state.domain_component_states[device.domain_ids.index(domain.domain_layer_id)]
        for species, block in blocks.items():
            expected = domain.ion_geometry.cell_measures_m * getattr(local, species + "_ion_density_m3")
            assert physical[block].tobytes() == expected.tobytes()
            assert equation[block].tobytes() == expected.tobytes()
            # D0 is not a massless transient constraint: canonical BE keeps S_u.
            assert tangent[block].nnz > 0 and np.all(expected > 0)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("order", [1, 2])
def test_full_storage_and_BDF_Jacobians_match_represented_directional_derivatives(devices, name, order):
    s, u = step(devices[name], order)
    direction = np.random.default_rng(761).normal(size=len(u))
    direction[:devices[name].node_count] *= .01
    epsilon = 2e-5
    analytic = s.jacobian(u)
    numeric = (s.residual(u + epsilon * direction) - s.residual(u - epsilon * direction)) / (2 * epsilon)
    scale = np.maximum(abs(analytic) @ abs(direction), 1.0)
    assert np.max(abs(numeric - analytic @ direction) / scale) < 3e-8
    tangent = s.endpoint.storage_tangent(u)
    difference = (s.endpoint.storage(u + epsilon * direction) - s.endpoint.storage(u - epsilon * direction)) / (2 * epsilon)
    scale = np.maximum(abs(tangent) @ abs(direction), 1.0)
    assert np.max(abs(difference - tangent @ direction) / scale) < 3e-8


@pytest.mark.parametrize("name", CASES)
def test_order_one_accounting_reuses_the_exact_assembled_BE_reader(devices, name):
    device = devices[name]
    s, u = step(device, 1)
    old = s.history[0]
    previous = device.state(old.u, voltage_V=old.problem.voltage_V)
    be = (AssembledIonicBackwardEuler1DStep(s.endpoint.problem, previous, s.dt_s)
          if isinstance(device, CompiledIonicDevice1D) else AssembledMultipleIonicBackwardEuler1DStep(s.endpoint.problem, previous, s.dt_s))
    original, actual = be.accounting(u), assembled_ionic_bdf_accounting(s, u)
    for key, value in original.items():
        assert _encode(actual[key]) == _encode(value), key
    assert be.residual(u).tobytes() == s.residual(u).tobytes()
    assert be.jacobian(u).data.tobytes() == s.jacobian(u).data.tobytes()
    assert actual["domains"].keys() == {d.domain_layer_id for d, _ in domains(device)}


@pytest.mark.parametrize("name", CASES)
def test_order_two_per_domain_rates_balances_and_Maxwell_identity(devices, name):
    device = devices[name]
    s, u = step(device, 2)
    a, old, older = s.endpoint, s.history[0], s.history[1]
    rate = (s.coefficients[0] * a.physical_increment(u, old.problem, old.u)
            - s.coefficients[2] * old.problem.physical_increment(old.u, older.problem, older.u)) / s.dt_s
    data = assembled_ionic_bdf_accounting(s, u)
    np.testing.assert_array_equal(data["bdf_physical_inventory_rate_m2_s"], rate)
    for domain, blocks in domains(device):
        for species, block in blocks.items():
            record = data["domains"][domain.domain_layer_id]["species"][species]
            np.testing.assert_array_equal(record["inventory_rate_m2_s"], rate[block])
            expected = -np.diff(record["domain_flux_m2_s"]) - rate[block]
            np.testing.assert_array_equal(record["balance_m2_s"], expected)
    scale = max(np.max(abs(data["conduction_current_density_A_m2"])), np.max(abs(data["bdf_displacement_rate_A_m2"])), 1.0)
    assert np.max(abs(data["maxwell_identity_error_A_m2"])) < 64 * np.finfo(float).eps * scale
    np.testing.assert_array_equal(data["terminal_total_current_density_A_m2"], data["total_current_density_A_m2"][[0, -1]])


@pytest.mark.parametrize("name", CASES)
def test_order_two_readback_prediction_matches_the_reported_BDF2_identity(devices, name):
    s, u = step(devices[name], 2)
    data = assembled_ionic_bdf_accounting(s, u)
    ion_balance = np.zeros(devices[name].node_count)
    for domain in data["domains"].values():
        for species, record in domain["species"].items():
            sign = 1.0 if species == "positive" else -1.0
            ion_balance[domain["electrical_node_indices"]] += sign * record["balance_m2_s"]
    carrier = data["carrier_balance_m2_s"]
    defect = Q * (carrier[1] - carrier[0] + ion_balance)
    rounded_gauss = data["rounded_gauss_defect_rate_A_m2"]
    charge_error = data["charge_readback_error_A_m2"]
    displacement_error = data["bdf_displacement_readback_error_A_m2"]
    prediction = defect + rounded_gauss + charge_error + displacement_error[:-1] - displacement_error[1:]
    np.testing.assert_array_equal(
        data["readback_predicted_maxwell_cell_jump_A_m2"].view(np.uint64), prediction.view(np.uint64))
    jump = data["maxwell_cell_jump_A_m2"]
    residual = jump - defect - rounded_gauss - charge_error - displacement_error[:-1] + displacement_error[1:]
    np.testing.assert_array_equal(data["readback_maxwell_identity_error_A_m2"].view(np.uint64), residual.view(np.uint64))
    # The stored residual keeps its original arithmetic; grouping the prediction
    # first differs only by roundoff, not by an inherited BE readback term.
    scale = np.max(abs(jump) + abs(defect) + abs(rounded_gauss) + abs(charge_error)
                   + abs(displacement_error[:-1]) + abs(displacement_error[1:]))
    assert np.max(abs(jump - prediction - residual)) <= 8 * np.finfo(float).eps * scale


@pytest.mark.parametrize("name", CASES)
def test_explicit_steady_Z_and_inventory_constraint_rows_have_zero_storage(devices, name):
    device = devices[name]
    original = adapter(device).problem
    closed = (AssembledConstrainedIonic1DProblem(original, 0) if isinstance(device, CompiledIonicDevice1D)
              else AssembledConstrainedMultipleIonic1DProblem(original, 0))
    u = np.zeros(device.state_size)
    storage, tangent = closed.system.assembler.storage(u, original.inputs)
    mask = closed.system.closure_rows
    assert np.any(mask)
    np.testing.assert_array_equal(storage[mask], np.zeros(np.count_nonzero(mask)))
    assert tangent[mask].nnz == 0


def strip_time(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {key: strip_time(item) for key, item in value.items() if key not in {"wall_s", "initial_admission_wall_s"}}
    if isinstance(value, (list, tuple)):
        return tuple(strip_time(item) for item in value)
    return value


@pytest.mark.parametrize("kind", ["electronic", "single"])
def test_existing_BDF_paths_are_bitwise_before_and_after_new_adapter_evaluation(tmp_path, devices, kind):
    bdf_options = BDFOptions(1e-7 if kind == "electronic" else .25, adaptive=False, max_order=1)
    if kind == "electronic":
        device = compile_device_1d(load_device(), intervals_per_layer=12)
        admitted = SteadyOptions()
        initial = solve_steady_state(device, voltage_V=0.0, generation_scale=1.0, options=admitted)
        assert initial.converged
        protocol = TransientProtocol([0, 1e-7, 2e-7], [0, 0, 0], [1, 1.0025, 1.01])
        def run():
            return solve_electronic_bdf(device, initial, protocol, options=controls(),
                initial_options=admitted, bdf_options=bdf_options)
    else:
        device = compile_ionic_device_1d(prepare_single_ion_device(diffusion_only=True), intervals_per_layer=12)
        admitted = steady_ionic_controls(device.ion_node_count)
        point = solve_ionic_steady_state(device, voltage_V=.002, generation_scale=0.0, options=admitted)
        assert point.accepted
        content = json.dumps(json_value({"summary": {"accepted": True, "compiled_identity": device.identity,
            "voltage_V": .002, "generation_scale": 0.0}, "result": point}), allow_nan=False).encode()
        path = tmp_path / "Initial.json"
        path.write_bytes(content)
        initial = load_ionic_steady_initial(device, path, hashlib.sha256(content).hexdigest(), controls=admitted)
        protocol = TransientProtocol([0, .25, .5], [.002, .0025, .003], [0, 0, 0])
        def run():
            return solve_ionic_bdf(device, initial, protocol, options=time_ionic_controls(),
                bdf_options=bdf_options)
    before = run()
    assert before.completed
    for name in CASES:
        s, u = step(devices[name], 2)
        assembled_ionic_bdf_accounting(s, u)
    after = run()
    assert after.completed
    assert _encode(strip_time(before)) == _encode(strip_time(after))
