"""Finite-site ionic 1D on the shared Assembler, against the original classes.

Every ionic factory of the integrated parity harness (tests/support) is
captured from the original implementation and from the assembled bindings
at all four probes, then compared in every binary64 word, signed zeros
included: F, J, S, S_u, M, the backward-Euler step residual/Jacobian (here
in storage-difference form, ``F - dS/dt`` and ``J + M/dt``), every AC
operator of the unchanged small-signal classes, and the boundary/constraint
row masks. The sparse matrices must also share their canonical CSC
structure, so a sparse LU sees the same pattern and ordering. The udot form
(R/J_implicit) is compared at zero rate and is refused otherwise (Q2).
The masked storage increment keeps the original storage adapters' ``x * 0``
signed zeros on algebraic rows; a step residual leaves those rows at F.

Finite differences then check J and S_u of each binding independently, and
measure-scaling checks show that the Assembler applies each measure once.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import replace
import math
from typing import Any, TypeAlias
from unittest import mock

import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.sparse import csc_matrix, issparse
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_steady import controls as ionic_steady_controls
import solarlab.equations.ionic_small_signal as ionic_ac
import solarlab.equations.multiple_ionic_small_signal as multiple_ac
from solarlab.contracts.errors import ContractError
from solarlab.equations.assembly.assembler import Assembler, Mesh
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
from solarlab.equations.assembly.ionic_1d import (
    AssembledConstrainedIonic1DProblem, AssembledConstrainedMultipleIonic1DProblem, AssembledIonic1DProblem,
    AssembledMultipleIonic1DProblem, IonicSystem, ionic_system, multiple_ionic_system,
)
from solarlab.equations.assembly.terms import OperatingInputs
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.equations.ionic_small_signal import IonicSmallSignal1D
from solarlab.equations.multiple_ionic_1d import CompiledMultipleIonicDevice1D, MultipleIonic1DProblem
from solarlab.equations.multiple_ionic_small_signal import MultipleIonicSmallSignal1D
from solarlab.solvers.nonlinear.ionic_1d import ConstrainedIonic1DProblem, solve_ionic_steady_state
from solarlab.solvers.time.backward_euler import _storage_mask
from solarlab.solvers.time.ionic_backward_euler import IonicBackwardEuler1DStep
from solarlab.solvers.time.multiple_ionic_backward_euler import MultipleIonicBackwardEuler1DStep
from solarlab.solvers.nonlinear.multiple_ionic_1d import (
    ConstrainedMultipleIonic1DProblem, solve_multiple_ionic_steady_state,
)
from support.assembly_parity import Extraction, compare
from support.assembly_references import REFERENCE_FACTORIES, Probe, ReferenceCase, _mark_rows

Vector: TypeAlias = NDArray[np.float64]
Problem: TypeAlias = AssembledIonic1DProblem | AssembledMultipleIonic1DProblem
FAMILIES = ("ionic", "ionic_constrained", "multiple", "multiple_constrained",
            "ionic_be", "multiple_be", "ionic_ac", "multiple_ac")
IONIC = tuple(name for name, factory in REFERENCE_FACTORIES.items() if factory.family in FAMILIES)
SEEDED = "seeded_perturbation"
# Richardson central differences of smooth rows reach ~1e-7 of the row scale
# here (h = 1e-4: the coordinate-to-density maps round like ~1e-11 |S|, which
# a smaller h amplifies as 1/h); a missing or wrong-sign tangent term is O(1).
ROW_TOLERANCE = 1.0e-6
FD_STEP = 1.0e-4


@pytest.fixture(scope="module", autouse=True)
def blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


def assembled(case: ReferenceCase, state: Probe) -> Problem:
    device = case.device
    if isinstance(device, CompiledMultipleIonicDevice1D):
        return AssembledMultipleIonic1DProblem(device, state.voltage_V, state.generation_scale)
    return AssembledIonic1DProblem(device, state.voltage_V, state.generation_scale)


def constrained(problem: Problem, node: int) -> AssembledConstrainedIonic1DProblem | \
        AssembledConstrainedMultipleIonic1DProblem:
    if isinstance(problem, AssembledMultipleIonic1DProblem):
        return AssembledConstrainedMultipleIonic1DProblem(problem, node)
    return AssembledConstrainedIonic1DProblem(problem, node)


def substituted_ac() -> ExitStack:
    """Test-only: the unchanged AC classes accept (and build) the assembled problems."""
    stack = ExitStack()
    stack.enter_context(mock.patch.object(ionic_ac, "Ionic1DProblem", AssembledIonic1DProblem))
    stack.enter_context(mock.patch.object(multiple_ac, "Ionic1DProblem", AssembledIonic1DProblem))
    stack.enter_context(mock.patch.object(multiple_ac, "MultipleIonic1DProblem", AssembledMultipleIonic1DProblem))
    return stack


def owned(system: IonicSystem, suffix: str) -> NDArray[np.bool_]:
    """Global rows of the constraint Terms whose id ends with ``suffix``."""
    layout = system.layout
    mask: NDArray[np.bool_] = np.zeros(layout.size, dtype=bool)
    for rows in layout.constraints:
        if rows.owner.endswith(suffix):
            mask[layout.rows(rows.block, rows.nodes)] = True
    return mask


def candidate(case: ReferenceCase, state: Probe) -> Extraction:
    """The harness quantities of ``case`` at ``state``, from the assembled bindings only."""
    device, problem = case.device, assembled(case, state)
    system = problem.binding
    masks = {"potential_fixed": owned(system, "potential.contacts"),
             "carrier_fixed": owned(system, ".reservoirs"),
             "ion_boundary": np.zeros(system.layout.size, dtype=bool)}
    for block in system.layout.blocks[3:]:
        span = system.layout.offsets[block.id]
        masks["ion_boundary"][[span.start, span.stop - 1]] = True
    extra: dict[str, Any] = {}
    S = S_u = M = None
    if case.family.endswith("constrained"):
        target = constrained(problem, 1)
        masks["inventory_constraint"] = owned(target.system, ".inventory")
        masks["immobile_profile_constraint"] = owned(target.system, ".profile")
        F, J = target.residual(state.u), target.jacobian(state.u)
    else:
        F, J = problem.residual(state.u), problem.jacobian(state.u)
        S, S_u, M = problem.physical_storage(state.u), problem.storage_tangent(state.u), problem.storage_jacobian(state.u)
        if state.name != SEEDED:
            extra["R_implicit"] = problem.residual(state.u, state.udot)
            extra["J_implicit"] = problem.jacobian(state.u, state.udot, alpha=state.alpha)
    if case.family.endswith("_be"):
        previous = problem.at(state.previous_voltage_V, state.generation_scale)
        increment = problem.storage_increment(state.u, previous, state.previous_u)
        # The original step never writes its algebraic rows (R = F there), so the
        # masked increment's signed zeros never enter R.
        algebraic = system.layout.constraint_mask()
        extra.update(R_step=np.where(algebraic, F, F - increment / state.dt_s), J_step=J + M / state.dt_s)
    if case.family.endswith("_ac"):
        with substituted_ac():
            operator_class: Any = MultipleIonicSmallSignal1D if case.family == "multiple_ac" else IonicSmallSignal1D
            operator = operator_class(problem, device.state(state.u, voltage_V=state.voltage_V), state.coordinate_form)
            mode = "frozen" if case.family == "ionic_ac" and state.name == SEEDED else "mobile"
            ac = operator.system(state.frequency_Hz, voltage_phasor_V=complex(.01, .002), ion_mode=mode)
            omega = 2 * np.pi * state.frequency_Hz
            matrix = ac.jacobian + 1j * omega * ac.storage_jacobian
            known = np.zeros(state.u.size, dtype=complex)
            for name in ("ac_fixed", "ac_original_enforced", "ac_inventory_constraint", "ac_immobile_constraint"):
                masks[name] = np.zeros(state.u.size, dtype=bool)
            for row, value in ac.fixed_coordinates.items():
                known[row], masks["ac_fixed"][row] = value, True
            masks["ac_original_enforced"][list(ac.original_rows_enforced)] = True
            _mark_rows(ac.constraint_rows, masks["ac_inventory_constraint"])
            _mark_rows(getattr(ac, "immobile_constraint_rows", {}), masks["ac_immobile_constraint"])
            extra.update(J_ac=ac.jacobian, M_ac=ac.storage_jacobian, A_ac=matrix,
                         R_ac=matrix @ state.ac_perturbation - (ac.rhs + 1j * omega * ac.rhs_rate),
                         rhs_ac=ac.rhs, rhs_rate_ac=ac.rhs_rate, fixed_ac=known, lift_ac=ac.coordinate_lift_per_V,
                         original_R_ac=operator.original_residual(state.ac_perturbation, state.frequency_Hz,
                                                                  voltage_phasor_V=complex(.01, .002)))
            J, M, S_u = operator.jacobian, operator.storage_jacobian, -operator.storage_jacobian
    return Extraction(F, J, case.blocks, S, S_u, M, extra, masks, {"candidate": type(problem).__name__})


def reference(case: ReferenceCase, state: Probe) -> Extraction:
    """The original capture; the nonzero-rate udot form is not migrated (Q2)."""
    captured = case.extract(state)
    if state.name != SEEDED:
        return captured
    extra = {name: value for name, value in captured.extra.items() if name not in ("R_implicit", "J_implicit")}
    return replace(captured, extra=extra)


def canonical(matrix: Any) -> csc_matrix:
    result = csc_matrix(matrix, copy=True)
    result.sum_duplicates()
    result.sort_indices()
    return result


@pytest.mark.parametrize("name", IONIC)
def test_every_quantity_is_bitwise_equal(name: str, request: pytest.FixtureRequest) -> None:
    case = REFERENCE_FACTORIES[name]()
    words, blocks = 0, 0
    for state in case.states:
        left, right = reference(case, state), candidate(case, state)
        comparison = compare(left, right)
        comparison.assert_equal()
        assert all(item.bitwise_equal and item.changed_entries == 0 for item in comparison.differences)
        for quantity, value in left.quantities().items():
            if issparse(value):
                a, b = canonical(value), canonical(right.quantities()[quantity])
                assert (a.indptr.tobytes(), a.indices.tobytes(), a.data.tobytes()) == \
                       (b.indptr.tobytes(), b.indices.tobytes(), b.data.tobytes()), quantity
        blocks += len(comparison.differences)
        words += sum(int(np.prod(np.shape(value))) for value in left.quantities().values())
    request.node.user_properties.append(("ionic_parity", {"factory": name, "block_pairs": blocks,
                                                          "words": words, "changed_words": 0}))


def test_nonzero_rate_udot_form_is_refused() -> None:
    case = REFERENCE_FACTORIES["ionic:shared"]()
    state = next(probe for probe in case.states if probe.name == SEEDED)
    problem = assembled(case, state)
    legacy = Ionic1DProblem(case.device, state.voltage_V, state.generation_scale)
    with pytest.raises(CapabilityError):
        problem.residual(state.u, state.udot)
    with pytest.raises(CapabilityError):
        problem.jacobian(state.u, state.udot, alpha=state.alpha)
    for bad in (0.0, np.zeros(1), np.zeros(state.u.size + 1), np.full(state.u.size, np.nan)):
        with pytest.raises(ValueError):
            legacy.residual(state.u, bad)
        with pytest.raises(ValueError):
            problem.residual(state.u, bad)
    multiple = REFERENCE_FACTORIES["multiple:shared"]()
    probe = multiple.states[0]
    target = assembled(multiple, probe)
    original = MultipleIonic1DProblem(multiple.device, probe.voltage_V, probe.generation_scale)
    for rate in (np.zeros(probe.u.size, dtype=complex), np.zeros(probe.u.size + 1)):
        for function in (original.residual, target.residual):
            with pytest.raises(ValueError):
                function(probe.u, rate)
    with pytest.raises(ValueError, match="finite real scalar"):
        target.jacobian(probe.u, alpha=float("nan"))


@pytest.mark.parametrize("name", ["ionic:dual", "ionic:shared", "multiple:independent", "multiple:shared"])
def test_observation_surface_equals_the_original_classes(name: str) -> None:
    case = REFERENCE_FACTORIES[name]()
    for state in case.states:
        problem = assembled(case, state)
        if isinstance(problem, AssembledMultipleIonic1DProblem):
            original: Any = MultipleIonic1DProblem(case.device, state.voltage_V, state.generation_scale)
            selectors = [dict(domain_id=domain.domain_layer_id, species=species)
                         for domain in case.device.domains for species in domain.ion_species_names]
            gradients = original.inventory_gradients(state.u), problem.inventory_gradients(state.u)
            for domain, species in gradients[0].items():
                for key, value in species.items():
                    assert value.tobytes() == gradients[1][domain][key].tobytes()
            assert problem.physical_storage(state.u).tobytes() == original.physical_storage(state.u).tobytes()
        else:
            original = Ionic1DProblem(case.device, state.voltage_V, state.generation_scale)
            selectors = [dict(species=species) for species in case.device.ion_species_names]
        for selector in selectors:
            assert problem.ion_fluxes(state.u, **selector).tobytes() == original.ion_fluxes(state.u, **selector).tobytes()
            assert (problem.ion_fluxes_on_electronic_faces(state.u, **selector).tobytes()
                    == original.ion_fluxes_on_electronic_faces(state.u, **selector).tobytes())
        assert problem.ion_current_density_A_m2(state.u).tobytes() == original.ion_current_density_A_m2(state.u).tobytes()
        electronic = problem.electronic_problem
        assert isinstance(electronic, AssembledDC1DProblem)
        u = state.u[:3 * case.device.node_count]
        for carrier in ("electron", "hole"):
            old = getattr(original.electronic_problem.fluxes(u), carrier)
            new = getattr(electronic.fluxes(u), carrier)
            assert old.flux.tobytes() == new.flux.tobytes()
        for alpha in (0.25, -3.0e7):
            a, b = canonical(original.jacobian(state.u, alpha=alpha)), canonical(problem.jacobian(state.u, alpha=alpha))
            assert (a.indptr.tobytes(), a.indices.tobytes(), a.data.tobytes()) == \
                   (b.indptr.tobytes(), b.indices.tobytes(), b.data.tobytes())


@pytest.mark.parametrize("name", [name for name in IONIC if name.split(":")[0].endswith("constrained")])
def test_constrained_bookkeeping_equals_the_original_wrappers(name: str) -> None:
    case = REFERENCE_FACTORIES[name]()
    state = case.states[1]
    problem = assembled(case, state)
    target = constrained(problem, 1)
    names: tuple[str, ...]
    if isinstance(problem, AssembledMultipleIonic1DProblem):
        original: Any = ConstrainedMultipleIonic1DProblem(
            MultipleIonic1DProblem(case.device, state.voltage_V, state.generation_scale), 1)
        names = ("immobile_species", "profile_constraint_rows", "constraint_rows")
    else:
        original = ConstrainedIonic1DProblem(Ionic1DProblem(case.device, state.voltage_V, state.generation_scale), 1)
        names = ("constraint_row", "immobile_species", "profile_constraint_rows", "inventory_target_m2",
                 "constraint_rows", "inventory_targets_m2")
    for attribute in names:
        assert getattr(target, attribute) == getattr(original, attribute), attribute
    closure = owned(target.system, ".inventory") | owned(target.system, ".profile")
    np.testing.assert_array_equal(closure, target.system.closure_rows)
    for node in (-1, case.device.node_count * 99, True):
        with pytest.raises(ValueError, match="inventory_row_node"):
            constrained(problem, node)  # type: ignore[arg-type]


def test_inadmissible_and_invalid_inputs_are_refused_as_before() -> None:
    case = REFERENCE_FACTORIES["ionic:confined"]()
    state = case.states[0]
    problem = AssembledIonic1DProblem(case.device, state.voltage_V, state.generation_scale)
    legacy = Ionic1DProblem(case.device, state.voltage_V, state.generation_scale)
    u = np.array(state.u, copy=True)
    u[case.device.node_count] = 800.0  # n overflows
    for function in (legacy.residual, problem.residual, legacy.jacobian, problem.jacobian):
        with pytest.raises(ValueError, match="outside the finite positive domain"):
            function(u)
    with pytest.raises(ValueError, match="alpha must be finite"):
        problem.jacobian(state.u, alpha=float("inf"))
    with pytest.raises(TypeError):
        AssembledIonic1DProblem(REFERENCE_FACTORIES["multiple:positive"]().device)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AssembledIonic1DProblem(case.device, 0.0, -1.0)
    with pytest.raises(ValueError, match="original rows"):
        AssembledIonic1DProblem(case.device, 0.0, 0.0, ionic_system(case.device, inventory_row_node=1))
    with pytest.raises(ValueError, match="requested ion species"):
        problem.ion_fluxes(state.u, species="negative")


def richardson(function: Callable[[Vector], Vector], u: Vector, steps: Vector) -> NDArray[np.float64]:
    """Fourth-order central-difference Jacobian, one coordinate at a time."""
    columns = []
    for index, step in enumerate(steps):
        def central(h: float) -> Vector:
            plus, minus = u.copy(), u.copy()
            plus[index] += h
            minus[index] -= h
            result: Vector = (function(plus) - function(minus)) / (2.0 * h)
            return result
        columns.append((4.0 * central(0.5 * step) - central(step)) / 3.0)
    return np.column_stack(columns)


def block_errors(exact: NDArray[np.float64], estimate: NDArray[np.float64], case: ReferenceCase,
                 level: Vector, steps: Vector) -> dict[tuple[str, str], float]:
    """Max |exact - estimate| per block pair, beyond the difference quotient's round-off.

    The probes are off-solution, so a row value ``|F_i|`` can exceed its
    tangents by ~1e5 (constant generation); a central difference then carries
    ``~eps |F_i| / h_j`` of rounding. That floor (x8) is removed before the
    error is normalized by the row's largest exact entry.
    """
    scale = np.max(np.abs(exact), axis=1)
    scale[scale == 0.0] = 1.0
    floor = 8.0 * np.finfo(float).eps * np.abs(level)[:, None] / steps[None, :]
    normalized = np.maximum(np.abs(exact - estimate) - floor, 0.0) / scale[:, None]
    return {(row.name, column.name): float(np.max(normalized[row.indices, column.indices]))
            for row in case.blocks for column in case.blocks}


FD_CASES = ("ionic:single", "ionic:confined", "ionic:dual", "ionic:shared", "ionic:immobile",
            "ionic_constrained:confined", "ionic_constrained:shared", "ionic_constrained:immobile",
            "ionic_constrained:immobile_single", "multiple:positive", "multiple:independent", "multiple:shared",
            "multiple_constrained:independent", "multiple_constrained:shared", "multiple_constrained:immobile")


def steps_of(case: ReferenceCase) -> Vector:
    thermal = case.device.electronic_component.thermal_voltage_V
    count = case.device.node_count
    result: Vector = np.r_[np.full(count, FD_STEP * thermal), np.full(case.device.state_size - count, FD_STEP)]
    return result


def fd_states(case: ReferenceCase) -> list[tuple[str, Vector, float, float]]:
    """The four harness probes plus, with a closure, an accepted steady state and a perturbation.

    (The original solver stagnates on ``multiple_constrained:immobile``; that
    case is checked at the four probes.)
    """
    states = [(state.name, np.array(state.u, copy=True), state.voltage_V, state.generation_scale)
              for state in case.states]
    if case.family.endswith("constrained"):
        device = case.device
        multiple = isinstance(device, CompiledMultipleIonicDevice1D)
        domain = device.domains[0] if isinstance(device, CompiledMultipleIonicDevice1D) else device
        solve: Any = solve_multiple_ionic_steady_state if multiple else solve_ionic_steady_state
        point = solve(device, voltage_V=0.002, generation_scale=0.0, options=ionic_steady_controls(domain.ion_node_count))
        if not point.accepted or point.state is None:
            return states
        rng = np.random.default_rng(20261029)
        count = device.node_count
        scale = np.r_[np.full(count, device.electronic_component.thermal_voltage_V), np.ones(device.state_size - count)]
        states += [("solved", np.array(point.state.u, copy=True), 0.002, 0.0),
                   ("solved_perturbed", point.state.u + 1.0e-2 * scale * rng.standard_normal(device.state_size), 0.002, 0.0)]
    return states


@pytest.mark.parametrize("name", FD_CASES)
def test_jacobian_and_storage_tangent_match_finite_differences(name: str, request: pytest.FixtureRequest) -> None:
    case = REFERENCE_FACTORIES[name]()
    steps = steps_of(case)
    worst: dict[str, float] = {}
    for label, u, voltage, light in fd_states(case):
        problem: Problem = (AssembledMultipleIonic1DProblem(case.device, voltage, light)
                            if isinstance(case.device, CompiledMultipleIonicDevice1D)
                            else AssembledIonic1DProblem(case.device, voltage, light))
        rows: Any = constrained(problem, 1) if case.family.endswith("constrained") else problem
        quantities = [("J", rows.jacobian(u).toarray(), rows.residual)]
        if rows is problem:
            quantities.append(("S_u", problem.storage_tangent(u).toarray(), problem.storage))
        for quantity, exact, function in quantities:
            estimate = richardson(function, u, steps)
            assert np.all(np.isfinite(estimate))
            errors = block_errors(exact, estimate, case, function(u), steps)
            worst[quantity] = max(worst.get(quantity, 0.0), max(errors.values()))
            assert max(errors.values()) < ROW_TOLERANCE, (quantity, label, errors)
    request.node.user_properties.append(("ionic_fd", {"factory": name, "states": len(fd_states(case)), **worst}))


@pytest.mark.parametrize("name", ["ionic_be:dual", "ionic_be:shared", "multiple_be:shared"])
def test_storage_increment_is_the_storage_difference(name: str) -> None:
    """The site-law secant equals S(u) - S(u_old) to rounding and has slope S_u."""
    case = REFERENCE_FACTORIES[name]()
    state = case.states[3]
    problem = assembled(case, state)
    previous = problem.at(state.previous_voltage_V, state.generation_scale)
    increment = problem.storage_increment(state.u, previous, state.previous_u)
    difference = problem.storage(state.u) - previous.storage(state.previous_u)
    scale = np.maximum(np.abs(problem.storage(state.u)), np.abs(previous.storage(state.previous_u)))
    np.testing.assert_array_less(np.abs(increment - difference), 1.0e-13 * scale + 1e-300)
    steps = steps_of(case)
    slope = richardson(lambda v: problem.storage_increment(v, previous, state.previous_u),
                       np.array(state.u, copy=True), steps)
    errors = block_errors(problem.storage_tangent(state.u).toarray(), slope, case, increment, steps)
    assert max(errors.values()) < ROW_TOLERANCE, errors


def scaled(system: IonicSystem, *, volumes: float = 1.0, faces: float = 1.0) -> Assembler:
    """The same declaration on a mesh whose ion-domain measures are scaled exactly."""
    supports = []
    for support in system.assembler.mesh.supports:
        if support.node_set.startswith("ion:"):
            g = support.geometry
            support = replace(support, geometry=replace(g, volumes=volumes * g.volumes,
                                                        face_measures=faces * g.face_measures))
        supports.append(support)
    assembler = system.assembler
    return Assembler(assembler.layout, Mesh(tuple(supports)), assembler.fields, assembler.terms)


@pytest.mark.parametrize("name", ["ionic_constrained:immobile", "multiple_constrained:shared"])
def test_each_ion_measure_enters_its_kinds_exactly_once(name: str) -> None:
    """R06: doubling one measure doubles exactly the pieces that own it, bit for bit.

    Doubling is exact in binary64, so a Term that pre-multiplied a measure (or
    one weighted twice) breaks the equalities.
    """
    case = REFERENCE_FACTORIES[name]()
    state = case.states[3]
    multiple = isinstance(case.device, CompiledMultipleIonicDevice1D)
    inputs = OperatingInputs(state.voltage_V, state.generation_scale)
    for closure in (None, 1):
        system = (multiple_ionic_system if multiple else ionic_system)(case.device, inventory_row_node=closure)
        layout = system.layout
        base = system.assembler.evaluate(state.u, inputs, storage=True)
        cells = scaled(system, volumes=2.0).evaluate(state.u, inputs, storage=True)
        faces = scaled(system, faces=2.0).evaluate(state.u, inputs, storage=True)
        ion: NDArray[np.bool_] = np.zeros(layout.size, dtype=bool)
        for block in layout.blocks[3:]:
            ion[layout.offsets[block.id]] = True
        closed = system.closure_rows
        transport = ion & ~closed
        assert base.J is not None and cells.J is not None and faces.J is not None
        J, J_cells, J_faces = (matrix.toarray() for matrix in (base.J, cells.J, faces.J))
        # Ion balance rows carry only face pieces; closure rows only cell-weighted pieces.
        assert (2.0 * base.F[transport]).tobytes() == faces.F[transport].tobytes()
        assert (2.0 * J[transport]).tobytes() == J_faces[transport].tobytes()
        assert base.F[transport].tobytes() == cells.F[transport].tobytes()
        assert (2.0 * base.F[closed]).tobytes() == cells.F[closed].tobytes()
        assert (2.0 * J[closed]).tobytes() == J_cells[closed].tobytes()
        assert base.F[~transport].tobytes() == faces.F[~transport].tobytes()
        # Storage: S and S_u on ion rows are cell pieces.
        assert base.S is not None and cells.S is not None and base.S_u is not None and cells.S_u is not None
        assert (2.0 * base.S[ion & ~closed]).tobytes() == cells.S[ion & ~closed].tobytes()
        assert (2.0 * base.S_u.toarray()[ion]).tobytes() == cells.S_u.toarray()[ion].tobytes()
        # Gauss rows: ion charge tangents carry the Mesh ratio W_i/W_e (doubled);
        # rho is the compiled ledger, bound to the Mesh by the ratio contract.
        gauss: NDArray[np.bool_] = np.zeros(layout.size, dtype=bool)
        gauss[layout.offsets["potential"]] = True
        assert (2.0 * J[np.ix_(gauss, ion)]).tobytes() == J_cells[np.ix_(gauss, ion)].tobytes()
        assert J[np.ix_(gauss, ~ion)].tobytes() == J_cells[np.ix_(gauss, ~ion)].tobytes()


def test_embedded_charge_needs_the_compiled_deposition_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ledger deposits with the compiled W_i/W_e; a different Mesh ratio is refused."""
    device = REFERENCE_FACTORIES["ionic:confined"]().device
    system = ionic_system(device)
    ratio = system.assembler.embedding_ratio("ion:" + device.domain_layer_id)
    assert ratio.tobytes() == device.ion_charge_weight_ratio.tobytes()
    original = Mesh.embedding_ratio
    monkeypatch.setattr(Mesh, "embedding_ratio",
                        lambda self, layout, node_set: np.nextafter(original(self, layout, node_set), np.inf))
    with pytest.raises(CapabilityError, match="deposition weights"):
        ionic_system(device)


def test_reduction_rows_keep_the_historical_fsum_grouping() -> None:
    """K1 rows are ``fsum(W P) - fsum(W P0)``; KM rows are ``fsum(W (P - P0))``."""
    case = REFERENCE_FACTORIES["ionic_constrained:dual"]()
    multiple = REFERENCE_FACTORIES["multiple_constrained:independent"]()
    for current, form in ((case, "k1"), (multiple, "km")):
        state = current.states[3]
        target = constrained(assembled(current, state), 1)
        F = target.residual(state.u)
        for rows in target.system.layout.constraints:
            if not rows.owner.endswith(".inventory"):
                continue
            domain = next(d for d in (current.device.domains if form == "km" else (current.device,))
                          if rows.block.startswith("ion:" + d.domain_layer_id + ":"))
            species = rows.block.rsplit(":", 1)[1]
            density = target.system.assembler.state(state.u, target.original.inputs).field(rows.block)  # type: ignore[union-attr]
            reference = getattr(domain, species + "_ion_reference_m3")
            W = domain.ion_geometry.cell_measures_m
            expected = (math.fsum((W * density).tolist()) - math.fsum((W * reference).tolist()) if form == "k1"
                        else math.fsum((W * (density - reference)).tolist()))
            row = target.system.layout.rows(rows.block, rows.nodes)[0]
            assert np.float64(F[row]).tobytes() == np.float64(expected).tobytes()


def test_reduction_constraint_contract_is_enforced() -> None:
    device = REFERENCE_FACTORIES["ionic:confined"]().device
    system = ionic_system(device, inventory_row_node=1)
    layout = system.layout
    rows = next(item for item in layout.constraints if item.owner.endswith(".inventory"))
    widened = replace(rows, nodes=(1, 2))
    others = tuple(item for item in layout.constraints if item is not rows)
    bad = Assembler(replace(layout, constraints=others + (widened,)), system.assembler.mesh,
                    system.assembler.fields, system.assembler.terms)
    state = REFERENCE_FACTORIES["ionic:confined"]().states[0]
    with pytest.raises(ContractError, match="reduction_constraint_owns_one_row"):
        bad.evaluate(state.u, OperatingInputs(state.voltage_V, state.generation_scale))


def test_nonfinite_original_rows_are_refused_before_their_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    """The original wrappers check every original row, then replace the closure rows.

    On ``immobile_single`` every ion row is a profile row, so a nonfinite ion
    transport value or tangent survives only in the replaced rows.
    """
    import solarlab.equations.assembly.ionic_1d as assembled_module
    import solarlab.equations.ionic_1d as original_module

    flux_kernel, tangent_kernel = original_module.ion_face_flux, original_module.ion_face_flux_jacobian

    def poisoned_flux(*args: Any, **kwargs: Any) -> Any:
        flux = np.array(flux_kernel(*args, **kwargs), copy=True)
        flux[0] = np.nan
        return flux

    def poisoned_tangent(*args: Any, **kwargs: Any) -> Any:
        tangent = tangent_kernel(*args, **kwargs)
        derivative = np.array(tangent.potential_left_derivative, copy=True)
        derivative[0] = np.nan
        return replace(tangent, potential_left_derivative=derivative)

    case = REFERENCE_FACTORIES["ionic_constrained:immobile_single"]()
    state = case.states[1]
    for module in (original_module, assembled_module):
        monkeypatch.setattr(module, "ion_face_flux_jacobian", poisoned_tangent)
    original = ConstrainedIonic1DProblem(Ionic1DProblem(case.device, state.voltage_V, state.generation_scale), 1)
    target = AssembledConstrainedIonic1DProblem(
        AssembledIonic1DProblem(case.device, state.voltage_V, state.generation_scale), 1)
    for problem in (original, target):
        with pytest.raises(FloatingPointError, match="Jacobian is outside the finite"):
            problem.jacobian(state.u)
    system = target.assemble(state.u)
    assert system.J is not None and system.replaced_nonfinite_rows is not None
    assert np.all(np.isfinite(system.J.data)) and np.any(system.replaced_nonfinite_rows)
    for module in (original_module, assembled_module):
        monkeypatch.setattr(module, "ion_face_flux_jacobian", tangent_kernel)
        monkeypatch.setattr(module, "ion_face_flux", poisoned_flux)
    for problem in (original, target):
        with pytest.raises(FloatingPointError, match="residual is outside the finite"):
            problem.residual(state.u)
    assert np.all(np.isfinite(target.assemble(state.u, derivatives=False).F))


@pytest.mark.parametrize("name", ["ionic_be:confined", "ionic_be:dual", "ionic_be:shared", "multiple_be:shared"])
def test_masked_storage_increment_keeps_the_original_signed_zeros(name: str) -> None:
    """Increments equal the original secants with ``[carrier rows] *= storage_mask``, byte for byte."""
    case = REFERENCE_FACTORIES[name]()
    device, count = case.device, case.device.node_count
    negative = 0
    for state in case.states:
        problem = assembled(case, state)
        previous = problem.at(state.previous_voltage_V, state.generation_scale)
        increment = problem.storage_increment(state.u, previous, state.previous_u)
        old = device.state(state.previous_u, voltage_V=state.previous_voltage_V)
        ions: list[Vector]
        if isinstance(device, CompiledMultipleIonicDevice1D):
            step: Any = MultipleIonicBackwardEuler1DStep(
                MultipleIonic1DProblem(device, state.voltage_V, state.generation_scale), old, 1.0)
            carrier, rates = step.inventory_rates_m2_s(state.u)
            ions = [rates[domain][species] for domain, blocks in device.ion_species_slices.items() for species in blocks]
        else:
            step = IonicBackwardEuler1DStep(Ionic1DProblem(device, state.voltage_V, state.generation_scale), old, 1.0)
            carrier, local = step.inventory_rates_m2_s(state.u)
            ions = list(local)
        change = np.r_[np.zeros(count), carrier.ravel(), *ions]
        change[count:3 * count] *= _storage_mask(device.electronic_component).ravel()
        assert increment.tobytes() == change.tobytes()
        negative += int(np.count_nonzero(np.signbit(increment[problem.binding.layout.constraint_mask()])))
    assert negative > 0
