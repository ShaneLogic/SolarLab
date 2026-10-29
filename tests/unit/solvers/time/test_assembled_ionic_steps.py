"""Frozen BE F/J, physical secants and accounting on eight ionic factories.

All probes are algebra states, not accepted time steps. The frozen digest
oracle is independent of the old implementation's lifetime; the original
accounting is also compared word for word, including signed zeros.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
import runpy
from typing import Any

import numpy as np
import pytest
from scipy.sparse import csc_matrix
from threadpoolctl import threadpool_limits

from solarlab.equations.assembly.ionic_1d import AssembledIonic1DProblem, AssembledMultipleIonic1DProblem
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, Ionic1DProblem
from solarlab.equations.multiple_ionic_1d import CompiledMultipleIonicDevice1D, MultipleIonic1DProblem
from solarlab.solvers.time.assembled_ionic_backward_euler import (
    AssembledIonicBackwardEuler1DStep, AssembledMultipleIonicBackwardEuler1DStep,
)
from solarlab.solvers.time.assembled_storage import AssembledIonicStorage1D
from solarlab.solvers.time.backward_euler import inventory_m2 as carrier_inventory
from solarlab.solvers.time.ionic_backward_euler import IonicBackwardEuler1DStep, step_accounting
from solarlab.solvers.time.multiple_ionic_backward_euler import MultipleIonicBackwardEuler1DStep, multiple_ionic_step_accounting
from support.assembly_parity import Extraction
from support.assembly_references import REFERENCE_FACTORIES, ReferenceCase

SINGLE = ("ionic_be:confined", "ionic_be:dual", "ionic_be:shared", "ionic_be:immobile")
MULTIPLE = ("multiple_be:positive", "multiple_be:independent", "multiple_be:shared", "multiple_be:immobile")
FACTORIES = SINGLE + MULTIPLE
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module", autouse=True)
def blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def cases() -> dict[str, ReferenceCase]:
    return {name: REFERENCE_FACTORIES[name]() for name in FACTORIES}


@pytest.fixture(scope="module")
def frozen() -> tuple[Any, dict[str, Any]]:
    import json
    oracle = runpy.run_path(str(ROOT / "tests/unit/assembly_parity/test_frozen_baselines.py"))
    document = json.loads(oracle["BASELINE_PATH"].read_bytes())
    return oracle["extraction_digest"], oracle["expand_digests"](document["factories"], document["sha256"])


def words(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, np.ascontiguousarray(value).tobytes())
    if isinstance(value, (float, np.floating)):
        return np.float64(value).tobytes()
    if isinstance(value, dict):
        return tuple((key, words(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(words(item) for item in value)
    return value


def assert_matrix(left: csc_matrix, right: csc_matrix) -> None:
    assert left.shape == right.shape
    assert np.array_equal(left.indptr, right.indptr)
    assert np.array_equal(left.indices, right.indices)
    assert left.data.tobytes() == right.data.tobytes()


def steps(case: ReferenceCase, state: Any, dt_s: float | None = None) -> tuple[Any, Any, AssembledIonicStorage1D]:
    device, dt = case.device, state.dt_s if dt_s is None else dt_s
    previous = device.state(state.previous_u, voltage_V=state.previous_voltage_V)
    if isinstance(device, CompiledIonicDevice1D):
        original = IonicBackwardEuler1DStep(Ionic1DProblem(device, state.voltage_V, state.generation_scale), previous, dt)
        problem = AssembledIonic1DProblem(device, state.voltage_V, state.generation_scale)
        candidate = AssembledIonicBackwardEuler1DStep(problem, previous, dt)
    else:
        assert isinstance(device, CompiledMultipleIonicDevice1D)
        original = MultipleIonicBackwardEuler1DStep(MultipleIonic1DProblem(device, state.voltage_V, state.generation_scale), previous, dt)
        problem = AssembledMultipleIonic1DProblem(device, state.voltage_V, state.generation_scale)
        candidate = AssembledMultipleIonicBackwardEuler1DStep(problem, previous, dt)
    return original, candidate, AssembledIonicStorage1D(problem)


def full_rate(step: Any, u: np.ndarray) -> np.ndarray:
    carrier, ions = step.inventory_rates_m2_s(u)
    if isinstance(ions, dict):
        parts = [rate for species in ions.values() for rate in species.values()]
    else:
        parts = list(ions)
    return np.r_[np.zeros(step.endpoint.device.node_count), carrier.ravel(), *parts]


def original_accounting(step: Any, u: np.ndarray) -> dict[str, Any]:
    return multiple_ionic_step_accounting(step, u) if isinstance(step, MultipleIonicBackwardEuler1DStep) else step_accounting(step, u)


@pytest.mark.parametrize("name", FACTORIES)
def test_all_32_probes_match_frozen_step_and_storage_arrays(
    name: str, cases: dict[str, ReferenceCase], frozen: tuple[Any, dict[str, Any]], request: pytest.FixtureRequest,
) -> None:
    case = cases[name]
    digest, expected = frozen
    count = 0
    for state in case.states:
        original, candidate, storage = steps(case, state)
        assembled = storage.problem.assemble(state.u, storage=True)
        captured = Extraction(assembled.F, assembled.J, case.blocks, assembled.S, assembled.S_u, assembled.M,
            extra={"R_step": candidate.residual(state.u), "J_step": candidate.jacobian(state.u)})
        actual = digest(captured)
        oracle = expected[name]["states"][state.name]["quantities"]
        assert actual == {key: oracle[key] for key in actual}
        assert words(original.residual(state.u)) == words(candidate.residual(state.u))
        assert_matrix(original.jacobian(state.u), candidate.jacobian(state.u))
        assert words(original.inventory_rates_m2_s(state.u)) == words(candidate.inventory_rates_m2_s(state.u))
        old_raw, raw_step, _ = steps(case, state, 1.0)
        assert words(full_rate(old_raw, state.u)) == words(raw_step.physical_increment(state.u))
        assert words(original.initial_coordinates()) == words(candidate.initial_coordinates())
        count += 1
    request.node.user_properties.append(("step_parity", {"factory": name, "states": count,
        "F_J_S_Su_M_Rstep_Jstep_changed_words": 0, "physical_increment_changed_words": 0,
        "rate_and_predictor_changed_words": 0}))


@pytest.mark.parametrize("name", FACTORIES)
def test_every_accounting_readout_is_bitwise(
    name: str, cases: dict[str, ReferenceCase], request: pytest.FixtureRequest,
) -> None:
    case = cases[name]
    arrays = 0
    def count_arrays(value: Any) -> int:
        if isinstance(value, np.ndarray):
            return 1
        if isinstance(value, dict):
            return sum(count_arrays(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(count_arrays(item) for item in value)
        return 0
    for state in case.states:
        original, candidate, storage = steps(case, state)
        reference, actual = original_accounting(original, state.u), candidate.accounting(state.u)
        assert words(reference) == words(actual)
        arrays += count_arrays(reference)
        physical = storage.physical_storage(state.u)
        new_state = case.device.state(state.u, voltage_V=state.voltage_V)
        carriers = carrier_inventory(case.device.electronic_component, new_state.electronic_state).ravel()
        assert physical[case.device.node_count:3 * case.device.node_count].tobytes() == carriers.tobytes()
    request.node.user_properties.append(("accounting_parity", {"factory": name,
        "states": len(case.states), "array_vectors": arrays, "all_numeric_changed_words": 0}))


@pytest.mark.parametrize("name", FACTORIES)
def test_site_secants_retain_motion_hidden_by_rounded_fields(name: str, cases: dict[str, ReferenceCase]) -> None:
    case = cases[name]
    state = case.states[0]
    _, candidate, _ = steps(case, state, 1.0)
    u = state.previous_u.copy()
    u[3 * case.device.node_count:] += 1e-18
    raw = candidate.physical_increment(u)[3 * case.device.node_count:]
    assert np.any(raw != 0.0)
    before = case.device.state(state.previous_u, voltage_V=state.previous_voltage_V)
    after = case.device.state(u, voltage_V=state.previous_voltage_V)
    if isinstance(case.device, CompiledIonicDevice1D):
        for species in case.device.ion_species_names:
            assert getattr(before, species + "_ion_density_m3").tobytes() == getattr(after, species + "_ion_density_m3").tobytes()
    else:
        for domain, old, new in zip(case.device.domains, before.domain_component_states, after.domain_component_states):
            for species in domain.ion_species_names:
                assert getattr(old, species + "_ion_density_m3").tobytes() == getattr(new, species + "_ion_density_m3").tobytes()


def test_history_admission_refuses_another_domain_binding(cases: dict[str, ReferenceCase]) -> None:
    first, second = cases[SINGLE[0]], cases[MULTIPLE[0]]
    _, step, storage = steps(first, first.states[0])
    _, _, other = steps(second, second.states[0])
    with pytest.raises(ValueError, match="same bound domain system"):
        storage.storage_increment(first.states[0].u, other, second.states[0].u)
    with pytest.raises(ValueError, match="same complete ionic device"):
        AssembledIonicBackwardEuler1DStep(step.endpoint, replace(step.previous, device_sha256="wrong"), step.dt_s)
