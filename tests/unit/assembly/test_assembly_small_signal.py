"""Small-signal analysis on the assembled ionic systems, against the originals.

The AC operators of :mod:`solarlab.equations.assembly.small_signal` run on the
assembled problems only (every evaluation method of the original problem and
AC classes raises while they run). They are compared with the original
``IonicSmallSignal1D``/``MultipleIonicSmallSignal1D`` in every binary64 word:

* the harness AC captures (operator J/M, system J/M/rhs/rate, prescribed
  coordinates, lift, constraint-row masks, the residual operators), also
  against Main's frozen digests, which outlive the original classes;
* every system at 0 to 1e8 Hz, both coordinate forms, both ion responses and
  two phasors, its frequency solve, its ``observe`` record and
  ``original_residual``.

The analytic voltage drive ``F_V + J c`` and storage rate of the moving lift
are checked by finite differences of the assembled residual and storage, and
every system records its row operations and identity.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import replace
import itertools
from typing import Any, NoReturn
from unittest import mock

import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.sparse import csc_matrix, issparse
from threadpoolctl import threadpool_limits

from solarlab.equations.assembly.ionic_1d import AssembledIonic1DProblem, AssembledMultipleIonic1DProblem
from solarlab.equations.assembly.small_signal import (
    AssembledACSystem, AssembledIonicSmallSignal1D, AssembledMultipleIonicSmallSignal1D,
)
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.equations.ionic_small_signal import IonicSmallSignal1D
from solarlab.equations.multiple_ionic_1d import CompiledMultipleIonicDevice1D, MultipleIonic1DProblem
from solarlab.equations.multiple_ionic_small_signal import MultipleIonicSmallSignal1D
from solarlab.solvers.linear.frequency_domain import solve_frequency_systems
from support.assembly_parity import Extraction, compare
from support.assembly_references import REFERENCE_FACTORIES, Probe, ReferenceCase, _mark_rows

AC = ("ionic_ac:confined", "ionic_ac:shared", "ionic_ac:immobile", "multiple_ac:positive")
FREQUENCIES = (0.0, 37.0, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e8)
PHASORS = (complex(0.01, 0.002), 1.0 + 0.0j)
FORMS = ("moving_boundary_lift_v1", "fixed_reference_v1")
SEEDED = "seeded_perturbation"
PROVENANCE = frozenset({"source_sha256", "analysis_identity", "ac_system_identity", "ac_row_operations"})
FD_STEP_V = 1.0e-4
FD_TOLERANCE = 1.0e-6


@pytest.fixture(scope="module", autouse=True)
def blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


def originals_forbidden() -> ExitStack:
    """Every evaluation of the original problem and AC classes raises."""
    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("an original evaluation ran inside the assembled AC analysis")

    stack = ExitStack()
    for cls, methods in (
        (DC1DProblem, ("residual", "jacobian", "storage_jacobian", "fluxes", "recombination_rate")),
        (Ionic1DProblem, ("residual", "jacobian", "storage_jacobian", "ion_fluxes")),
        (MultipleIonic1DProblem, ("residual", "jacobian", "storage_jacobian", "ion_fluxes")),
        (IonicSmallSignal1D, ("system", "observe", "original_residual", "_carrier_particles",
                              "moving_voltage_partial")),
        (MultipleIonicSmallSignal1D, ("system", "observe", "original_residual", "_moving_voltage_partial")),
    ):
        for method in methods:
            stack.enter_context(mock.patch.object(cls, method, forbidden))
    return stack


def assembled_operator(case: ReferenceCase, state: Probe, form: str) -> Any:
    """The assembled AC operator of one probe."""
    device = case.device
    operating = device.state(state.u, voltage_V=state.voltage_V)
    if isinstance(device, CompiledMultipleIonicDevice1D):
        return AssembledMultipleIonicSmallSignal1D(
            AssembledMultipleIonic1DProblem(device, state.voltage_V, state.generation_scale), operating, form)
    return AssembledIonicSmallSignal1D(AssembledIonic1DProblem(device, state.voltage_V, state.generation_scale),
                                       operating, form)


def operators(case: ReferenceCase, state: Probe, form: str) -> tuple[Any, Any]:
    """The original and the assembled AC operator of one probe."""
    device = case.device
    operating = device.state(state.u, voltage_V=state.voltage_V)
    if isinstance(device, CompiledMultipleIonicDevice1D):
        original: Any = MultipleIonicSmallSignal1D(
            MultipleIonic1DProblem(device, state.voltage_V, state.generation_scale), operating, form)
    else:
        original = IonicSmallSignal1D(Ionic1DProblem(device, state.voltage_V, state.generation_scale), operating, form)
    return original, assembled_operator(case, state, form)


def words(value: Any) -> Any:
    """Every leaf as binary64 words (sparse: canonical CSC structure and data)."""
    if isinstance(value, dict):
        return ("dict", tuple((key, words(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(words(item) for item in value))
    if issparse(value):
        matrix = csc_matrix(value, copy=True)
        matrix.sum_duplicates()
        matrix.sort_indices()
        return ("sparse", matrix.shape, matrix.indptr.tobytes(), matrix.indices.tobytes(), matrix.data.tobytes())
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, np.ascontiguousarray(value).tobytes())
    if isinstance(value, (complex, np.complexfloating)):
        return ("complex", np.complex128(value).tobytes())
    if isinstance(value, (float, np.floating)):
        return ("float", np.float64(value).tobytes())
    return ("value", value)


SYSTEM_FIELDS = ("frequency_Hz", "voltage_phasor_V", "ion_mode", "jacobian", "storage_jacobian", "rhs", "rhs_rate",
                 "fixed_coordinates", "constraint_rows", "original_rows_enforced", "coordinate_form",
                 "coordinate_lift_per_V")


def system_words(system: Any) -> Any:
    fields = {name: getattr(system, name) for name in SYSTEM_FIELDS}
    fields["immobile_constraint_rows"] = getattr(system, "immobile_constraint_rows", {})
    return words(fields)


def candidate(case: ReferenceCase, state: Probe) -> Extraction:
    """The harness AC capture of one probe, from the assembled operators only."""
    operator = assembled_operator(case, state, state.coordinate_form)
    problem = operator.problem
    layout = problem.binding.layout
    masks: dict[str, NDArray[np.bool_]] = {}
    for name, suffix in (("potential_fixed", "potential.contacts"), ("carrier_fixed", ".reservoirs")):
        masks[name] = np.zeros(layout.size, dtype=bool)
        for rows in layout.constraints:
            if rows.owner.endswith(suffix):
                masks[name][layout.rows(rows.block, rows.nodes)] = True
    masks["ion_boundary"] = np.zeros(layout.size, dtype=bool)
    for block in layout.blocks[3:]:
        span = layout.offsets[block.id]
        masks["ion_boundary"][[span.start, span.stop - 1]] = True
    extra: dict[str, Any] = {}
    F, S = problem.residual(state.u), problem.physical_storage(state.u)
    if state.name != SEEDED:
        extra["R_implicit"] = problem.residual(state.u, state.udot)
        extra["J_implicit"] = problem.jacobian(state.u, state.udot, alpha=state.alpha)
    mode = "frozen" if case.family == "ionic_ac" and state.name == SEEDED else "mobile"
    system = operator.system(state.frequency_Hz, voltage_phasor_V=complex(.01, .002), ion_mode=mode)
    omega = 2 * np.pi * state.frequency_Hz
    matrix = system.jacobian + 1j * omega * system.storage_jacobian
    known: NDArray[np.complex128] = np.zeros(state.u.size, dtype=complex)
    for name in ("ac_fixed", "ac_original_enforced", "ac_inventory_constraint", "ac_immobile_constraint"):
        masks[name] = np.zeros(state.u.size, dtype=bool)
    for row, value in system.fixed_coordinates.items():
        known[row], masks["ac_fixed"][row] = value, True
    masks["ac_original_enforced"][list(system.original_rows_enforced)] = True
    _mark_rows(system.constraint_rows, masks["ac_inventory_constraint"])
    _mark_rows(system.immobile_constraint_rows, masks["ac_immobile_constraint"])
    extra.update(J_ac=system.jacobian, M_ac=system.storage_jacobian, A_ac=matrix,
                 R_ac=matrix @ state.ac_perturbation - (system.rhs + 1j * omega * system.rhs_rate),
                 rhs_ac=system.rhs, rhs_rate_ac=system.rhs_rate, fixed_ac=known, lift_ac=system.coordinate_lift_per_V,
                 original_R_ac=operator.original_residual(state.ac_perturbation, state.frequency_Hz,
                                                          voltage_phasor_V=complex(.01, .002)))
    J, M = operator.jacobian, operator.storage_jacobian
    return Extraction(F, J, case.blocks, S, -M, M, extra, masks, {"candidate": type(operator).__name__})


def reference(case: ReferenceCase, state: Probe) -> Extraction:
    captured = case.extract(state)
    if state.name != SEEDED:
        return captured
    return replace(captured, extra={name: value for name, value in captured.extra.items()
                                    if name not in ("R_implicit", "J_implicit")})


@pytest.mark.parametrize("name", AC)
def test_harness_ac_captures_are_bitwise_equal(name: str) -> None:
    case = REFERENCE_FACTORIES[name]()
    for state in case.states:
        with originals_forbidden():
            assembled = candidate(case, state)
        comparison = compare(reference(case, state), assembled)
        comparison.assert_equal()
        assert all(item.bitwise_equal for item in comparison.differences)


@pytest.mark.platform_bound
@pytest.mark.parametrize("name", AC)
def test_harness_ac_captures_match_the_frozen_digests(
        name: str, frozen_digests: tuple[Any, dict[str, Any]]) -> None:
    case = REFERENCE_FACTORIES[name]()
    digest, factories = frozen_digests
    frozen = factories[name]
    for state in case.states:
        with originals_forbidden():
            assembled = candidate(case, state)
        digests = digest(assembled)
        expected = frozen["states"][state.name]["quantities"]
        if state.name == SEEDED:
            expected = {key: value for key, value in expected.items() if key not in ("R_implicit", "J_implicit")}
        assert digests == expected, name + "/" + state.name


@pytest.mark.parametrize("name", AC)
def test_every_frequency_system_solve_and_observation_is_bitwise_equal(name: str,
                                                                      request: pytest.FixtureRequest) -> None:
    case = REFERENCE_FACTORIES[name]()
    multiple = isinstance(case.device, CompiledMultipleIonicDevice1D)
    systems = 0
    for state, form in itertools.product(case.states, FORMS):
        original, assembled = operators(case, state, form)
        for attribute in ("jacobian", "storage_jacobian", "reference_direction_per_V", "coordinate_lift_per_V",
                          "working_state_digest", "working_state_identity"):
            assert words(getattr(original, attribute)) == words(getattr(assembled, attribute)), attribute
        for mode, frequency, phasor in itertools.product(("mobile",) if multiple else ("mobile", "frozen"),
                                                         FREQUENCIES, PHASORS):
            before = original.system(frequency, voltage_phasor_V=phasor, ion_mode=mode)
            with originals_forbidden():
                after = assembled.system(frequency, voltage_phasor_V=phasor, ion_mode=mode)
            assert system_words(after) == system_words(before), (state.name, form, mode, frequency)
            solved = [solve_frequency_systems(item.jacobian, item.storage_jacobian, item.rhs, [frequency],
                                              fixed_coordinates=item.fixed_coordinates, rhs_rate=item.rhs_rate)[0]
                      for item in (before, after)]
            assert words([(item.solution, item.residual, item.componentwise_backward_error, item.refinement_steps)
                          for item in solved[:1]]) == words(
                [(item.solution, item.residual, item.componentwise_backward_error, item.refinement_steps)
                 for item in solved[1:]])
            seen = original.observe(solved[0].solution, frequency, voltage_phasor_V=phasor, ion_mode=mode)
            with originals_forbidden():
                observed = assembled.observe(solved[1].solution, frequency, voltage_phasor_V=phasor, ion_mode=mode,
                                             system=after)
                plain = assembled.observe(solved[1].solution, frequency, voltage_phasor_V=phasor, ion_mode=mode)
                residual = assembled.original_residual(solved[1].solution, frequency, voltage_phasor_V=phasor)
            assert set(seen) - PROVENANCE == set(observed) - PROVENANCE == set(plain) - PROVENANCE
            for key in sorted(set(seen) - PROVENANCE):
                assert words(observed[key]) == words(seen[key]) == words(plain[key]), (state.name, form, mode,
                                                                                       frequency, key)
            assert observed["analysis_identity"] == plain["analysis_identity"] == assembled.identity
            assert observed["ac_system_identity"] == after.identity and "ac_system_identity" not in plain
            assert observed["ac_row_operations"] == tuple((item.kind, item.identity) for item in after.row_operations)
            assert words(residual) == words(original.original_residual(solved[0].solution, frequency,
                                                                        voltage_phasor_V=phasor))
            systems += 1
    request.node.user_properties.append(("ac_parity", {"factory": name, "systems_solves_observations": systems}))


@pytest.mark.parametrize("name", AC)
def test_moving_lift_drive_and_rate_match_finite_differences(name: str) -> None:
    """``F_V + J c`` and ``dS/dV`` along the lift, against the assembled residual and storage."""
    case = REFERENCE_FACTORIES[name]()
    eps = np.finfo(float).eps
    for state in case.states:
        _, operator = operators(case, state, "moving_boundary_lift_v1")
        problem = operator.problem
        lift = np.asarray(operator.coordinate_lift_per_V)
        partial, drive = operator._moving_voltage_partial(lift)
        for exact, quantity in ((partial, "residual"), (drive, "storage")):
            def along(step: float) -> NDArray[np.float64]:
                moved = problem.at(state.voltage_V + step, state.generation_scale)
                result: NDArray[np.float64] = getattr(moved, quantity)(state.u + step * lift)
                return result

            def central(step: float) -> NDArray[np.float64]:
                result: NDArray[np.float64] = (along(step) - along(-step)) / (2.0 * step)
                return result

            estimate = (4.0 * central(0.5 * FD_STEP_V) - central(FD_STEP_V)) / 3.0
            floor = 8.0 * eps * np.abs(along(0.0)) / FD_STEP_V
            for block in case.blocks:
                scale = max(float(np.max(np.abs(exact[block.indices]))), 1e-300)
                error = np.max(np.maximum(np.abs(exact - estimate)[block.indices] - floor[block.indices], 0.0)) / scale
                assert error < FD_TOLERANCE, (state.name, quantity, block.name, error)


def operation_kinds(system: AssembledACSystem) -> list[str]:
    return [operation.kind for operation in system.row_operations]


def test_row_operations_and_identities_are_recorded() -> None:
    case = REFERENCE_FACTORIES["ionic_ac:immobile"]()
    state = case.states[1]
    _, moving = operators(case, state, "moving_boundary_lift_v1")
    _, fixed = operators(case, state, "fixed_reference_v1")
    zero = moving.system(0.0)
    assert operation_kinds(zero) == ["contact_drive", "known_contact_coordinates", "moving_boundary_lift",
                                     "zero_hz_inventory_rows", "zero_hz_immobile_profile_rows"]
    assert operation_kinds(moving.system(1.0e3)) == ["contact_drive", "known_contact_coordinates",
                                                     "moving_boundary_lift"]
    assert operation_kinds(fixed.system(1.0e3)) == ["contact_drive", "known_contact_coordinates"]
    frozen = moving.system(0.0, ion_mode="frozen")
    assert operation_kinds(frozen)[-1] == "frozen_ion_rows"
    N, size = case.device.node_count, case.device.state_size
    assert frozen.row_operations[-1].rows == tuple(range(3 * N, size))
    assert (frozen.jacobian[3 * N:, :] != csc_matrix(np.eye(size))[3 * N:, :]).nnz == 0
    # The replaced zero-Hz rows are the physical storage gradients (their nonzero entries).
    inventory, profile = zero.row_operations[3:]
    storage = moving.storage_jacobian

    def entries(row: int) -> tuple[bytes, bytes]:
        part = csc_matrix(zero.jacobian[row, :]).tocsr()
        part.sort_indices()
        return part.indices.astype(np.int64).tobytes(), part.data.tobytes()

    def nonzero(gradient: NDArray[np.float64]) -> tuple[bytes, bytes]:
        columns = np.flatnonzero(gradient)
        return columns.astype(np.int64).tobytes(), gradient[columns].tobytes()

    for row in inventory.rows:
        block = next(part for part in case.device.ion_species_slices if part.start == row)
        assert entries(row) == nonzero(-np.asarray(storage[block, :].sum(axis=0)).ravel())
        assert zero.storage_jacobian[row, :].nnz == 0
    for row in profile.rows:
        assert entries(row) == nonzero(-storage[row, :].toarray().ravel())
        assert zero.storage_jacobian[row, :].nnz == 0
    identities = {system.identity for system in (zero, frozen, moving.system(1.0e3), fixed.system(1.0e3),
                                                 moving.system(1.0e3, voltage_phasor_V=2.0))}
    assert len(identities) == 5
    assert moving.system(1.0e3).identity == moving.system(1.0e3).identity
    _, other = operators(case, case.states[2], "moving_boundary_lift_v1")
    assert other.identity != moving.identity
    assert other.system(1.0e3).identity != moving.system(1.0e3).identity


def test_invalid_inputs_are_refused_as_before() -> None:
    case = REFERENCE_FACTORIES["ionic_ac:confined"]()
    state = case.states[1]
    device = case.device
    operating = device.state(state.u, voltage_V=state.voltage_V)
    problem = AssembledIonic1DProblem(device, state.voltage_V, state.generation_scale)
    with pytest.raises(ValueError, match="coordinate representation"):
        AssembledIonicSmallSignal1D(problem, operating, "lifted")
    with pytest.raises(ValueError, match="same complete device and operating voltage"):
        AssembledIonicSmallSignal1D(problem.at(state.voltage_V + 0.01, state.generation_scale), operating)
    with pytest.raises(ValueError, match="full ionic problem"):
        AssembledIonicSmallSignal1D(Ionic1DProblem(device, state.voltage_V, 0.0), operating)  # type: ignore[arg-type]
    operator = AssembledIonicSmallSignal1D(problem, operating)
    for frequency, phasor, mode in ((-1.0, 1.0, "mobile"), (1.0, 0.0, "mobile"), (1.0, 1.0, "both")):
        with pytest.raises(ValueError):
            operator.system(frequency, voltage_phasor_V=phasor, ion_mode=mode)
    perturbation = np.zeros(device.state_size, dtype=complex)
    perturbation[-1] = 1.0
    with pytest.raises(ValueError, match="frozen-ion response"):
        operator.observe(perturbation, 1.0, ion_mode="frozen")
    with pytest.raises(ValueError, match="cannot change ion coordinates"):
        operator._moving_voltage_partial(perturbation.real)
    multiple = REFERENCE_FACTORIES["multiple_ac:positive"]()
    probe = multiple.states[0]
    target = AssembledMultipleIonicSmallSignal1D(
        AssembledMultipleIonic1DProblem(multiple.device, probe.voltage_V, probe.generation_scale),
        multiple.device.state(probe.u, voltage_V=probe.voltage_V))
    from solarlab.equations.dc_1d import CapabilityError

    with pytest.raises(CapabilityError, match="only mobile positive ions"):
        target.system(1.0, ion_mode="frozen")
    # An observation records a system only if it is that solve's system.
    zero = np.zeros(device.state_size, dtype=complex)
    for other in (operator.system(2.0), operator.system(1.0, voltage_phasor_V=2.0), operator.system(1.0, ion_mode="frozen"),
                  AssembledIonicSmallSignal1D(problem, operating, "fixed_reference_v1").system(1.0)):
        with pytest.raises(ValueError, match="belongs to another"):
            operator.observe(zero, 1.0, system=other)
    record = operator.observe(operator.system(1.0), zero)
    assert record["ac_system_identity"] == operator.system(1.0).identity
