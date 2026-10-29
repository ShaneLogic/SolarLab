"""Bitwise DC2D migration, against frozen oracles and the unchanged readouts.

The twenty harness states are algebra probes, not accepted steady results.
Set SOLARLAB_ASSEMBLY_2D_BASELINE to the pinned gate NPZ to additionally
compare the raw arrays captured at 4c7fb353; the repository's frozen digest
oracle is checked on its capture environment. Live old-vs-new comparisons
run on every platform. No baseline is rewritten and no solver is called.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import asdict, replace
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.sparse import csc_matrix
from threadpoolctl import threadpool_limits

from examples.research_2d import dc_assembly
from examples.research_1d.optical_generation import prepare_case as prepare_optical
from solarlab.contracts.errors import ContractError
from solarlab.contracts.units import COULOMB, LENGTH, ONE, PARTICLE, SECOND, VOLT
from solarlab.device.compile_2d import compile_device_2d
from solarlab.equations.assembly.terms import STORAGE
from solarlab.equations.assembly.two_dimensional import (
    AssembledDC2DProblem, classical_dc_2d,
)
from solarlab.equations.dc_2d import CompiledDevice2D, DC2DProblem
from support.assembly_parity import Extraction
from support.assembly_references import Probe, REFERENCE_FACTORIES, ReferenceCase

FACTORIES = ("dc2d:fixed", "dc2d:contacts", "dc2d:ct", "dc2d:grain", "dc2d:composed")
BASELINE_SHA256 = "dee5eae717fbba920fa0d8ddd1dc7d657beb7d33dca34ad594d00031fd687345"


@pytest.fixture(scope="module", autouse=True)
def blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def cases() -> dict[str, ReferenceCase]:
    return {name: REFERENCE_FACTORIES[name]() for name in FACTORIES}


@pytest.fixture(scope="module")
def gate_baseline(frozen_digests: tuple[Callable[[Any], dict[str, Any]], dict[str, Any]]) -> Iterator[Any]:
    filename = os.environ.get("SOLARLAB_ASSEMBLY_2D_BASELINE")
    if filename is None:
        yield None
        return
    path = Path(filename)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == BASELINE_SHA256
    with np.load(path, allow_pickle=False) as baseline:
        yield baseline


def assert_bits(reference: Any, candidate: Any) -> None:
    left = np.ascontiguousarray(reference, dtype=np.float64)
    right = np.ascontiguousarray(candidate, dtype=np.float64)
    assert left.shape == right.shape
    assert left.tobytes() == right.tobytes()


def assert_matrix(reference: csc_matrix, candidate: csc_matrix) -> None:
    assert reference.shape == candidate.shape
    assert np.array_equal(reference.indptr, candidate.indptr)
    assert np.array_equal(reference.indices, candidate.indices)
    assert_bits(reference.data, candidate.data)


def assembled_capture(case: ReferenceCase, state: Probe) -> Extraction:
    device = case.device
    assert isinstance(device, CompiledDevice2D)
    result = AssembledDC2DProblem(device, state.voltage_V, state.generation_scale).assemble(state.u)
    potential = np.zeros(result.layout.size, dtype=bool)
    potential[:device.node_count] = result.boundary_rows[:device.node_count]
    carrier = result.boundary_rows.copy()
    carrier[:device.node_count] = False
    return Extraction(result.F, result.J, case.blocks, result.S, result.S_u, result.M,
        boundary_masks={"potential_fixed": potential, "carrier_fixed": carrier,
                        "ion_boundary": np.zeros(result.layout.size, dtype=bool)})


@pytest.mark.parametrize("name", FACTORIES)
def test_all_twenty_states_are_live_bitwise_equal(
    name: str, cases: dict[str, ReferenceCase], request: pytest.FixtureRequest,
) -> None:
    case = cases[name]
    for state in case.states:
        candidate = assembled_capture(case, state)
        assert candidate.S is None and candidate.S_u is None and candidate.M is None
        original = case.extract(state)
        assert_bits(original.F, candidate.F)
        assert_matrix(original.J, candidate.J)
    request.node.user_properties.append(("parity", {"factory": name, "states": len(case.states),
        "F_changed_words": 0, "J_changed_words": 0,
        "S_Su_M": "absent in both original and candidate"}))


@pytest.mark.parametrize("name", FACTORIES)
def test_all_twenty_states_match_frozen_original_arrays(
    name: str, cases: dict[str, ReferenceCase], gate_baseline: Any,
    frozen_digests: tuple[Callable[[Any], dict[str, Any]], dict[str, Any]],
    request: pytest.FixtureRequest,
) -> None:
    case = cases[name]
    digest, expected = frozen_digests
    for state in case.states:
        candidate = assembled_capture(case, state)
        assert candidate.S is None and candidate.S_u is None and candidate.M is None
        assert digest(candidate) == expected[name]["states"][state.name]["quantities"]
        if gate_baseline is not None:
            prefix = f"{name}__{state.name}"
            assert_bits(gate_baseline[prefix + "_u"], state.u)
            assert_bits(gate_baseline[prefix + "_F"], candidate.F)
            assert_bits(gate_baseline[prefix + "_J_data"], candidate.J.data)
            assert np.array_equal(gate_baseline[prefix + "_J_indices"], candidate.J.indices)
            assert np.array_equal(gate_baseline[prefix + "_J_indptr"], candidate.J.indptr)
    request.node.user_properties.append(("frozen_parity", {"factory": name, "states": len(case.states),
        "F_changed_words": 0, "J_changed_words": 0, "gate_NPZ": gate_baseline is not None}))


@pytest.mark.parametrize("name", FACTORIES)
def test_every_flux_and_port_current_readout_is_bitwise(
    name: str, cases: dict[str, ReferenceCase], request: pytest.FixtureRequest,
) -> None:
    case = cases[name]
    device = case.device
    assert isinstance(device, CompiledDevice2D)
    system = classical_dc_2d(device)
    vectors = fixed_ports = exchange_ports = 0
    for state in case.states:
        original = DC2DProblem(device, state.voltage_V, state.generation_scale)
        candidate = AssembledDC2DProblem(device, state.voltage_V, state.generation_scale, system)
        for name in ("displacement", "current_density", "cross_section_current_A_m",
                     "terminal_current_A_m", "terminal_current_density_A_m2"):
            assert_bits(getattr(original, name)(state.u), getattr(candidate, name)(state.u))
            vectors += 1
        old_fluxes, new_fluxes = asdict(original.fluxes(state.u)), asdict(candidate.fluxes(state.u))
        for carrier in old_fluxes:
            for key in old_fluxes[carrier]:
                assert_bits(old_fluxes[carrier][key], new_fluxes[carrier][key])
                vectors += 1
        _, n, p = device.fields(state.u, voltage_V=state.voltage_V)
        for derivatives in (False, True):
            old_rate = asdict(original.recombination_rate(n, p, derivatives=derivatives))
            new_rate = asdict(candidate.recombination_rate(n, p, derivatives=derivatives))
            for key in old_rate:
                if old_rate[key] is None:
                    assert new_rate[key] is None
                else:
                    assert_bits(old_rate[key], new_rate[key])
                    vectors += 1
        old_contact, new_contact = original.contact_particle_fluxes(state.u), candidate.contact_particle_fluxes(state.u)
        for block, old, new in zip(("electron", "hole"), old_contact, new_contact):
            assert_bits(old, new)
            vectors += 1
            for port_index, port in enumerate(("y_min", "y_max")):
                key = (block, port)
                if key in system.exchanges:
                    exchange_ports += 1
                    assert_bits(system.exchanges[key].exchange(candidate._state(state.u))[0], old[port_index])
                    with pytest.raises(ContractError, match="requires_constraint_rows"):
                        system.assembler.reaction_flux_ordered(state.u, candidate.inputs, block, port,
                            normal_faces=system.geometry.normal_faces(port),
                            transverse_faces=system.geometry.transverse_faces)
                else:
                    fixed_ports += 1
                    assert_bits(system.assembler.reaction_flux_ordered(state.u, candidate.inputs, block, port,
                        normal_faces=system.geometry.normal_faces(port),
                        transverse_faces=system.geometry.transverse_faces), old[port_index])
    request.node.user_properties.append(("readouts", {"factory": case.name, "vectors": vectors,
        "fixed_port_vectors": fixed_ports, "exchange_port_vectors": exchange_ports, "changed_words": 0}))


@pytest.mark.parametrize("name", FACTORIES)
def test_geometry_layout_binding_and_ordered_readout_identity(name: str, cases: dict[str, ReferenceCase]) -> None:
    device = cases[name].device
    assert isinstance(device, CompiledDevice2D)
    system = classical_dc_2d(device)
    g, count = device.geometry, device.node_count
    assert system.layout.block_ranges() == {"potential": (0, count), "electron": (count, 2 * count),
                                            "hole": (2 * count, 3 * count)}
    assert tuple(block.variable_unit for block in system.layout.blocks) == (VOLT, ONE, ONE)
    assert tuple(block.balance_unit for block in system.layout.blocks) == (
        COULOMB / LENGTH, PARTICLE / (LENGTH * SECOND), PARTICLE / (LENGTH * SECOND))
    support = system.geometry.mesh.support("nodes")
    assert support.geometry is g.contract_geometry
    assert_bits(support.geometry.volumes, g.contract_geometry.volumes)
    assert_bits(support.geometry.face_measures, g.contract_geometry.face_measures)
    assert np.array_equal(support.geometry.pairs, g.contract_geometry.pairs)
    ids = np.arange(count).reshape(g.shape)
    assert np.array_equal(support.geometry.pairs[:g.x_face_count],
                          np.column_stack((ids[:, :-1].ravel(), ids[:, 1:].ravel())))
    assert np.array_equal(support.geometry.pairs[g.x_face_count:],
                          np.column_stack((ids[:-1].ravel(), ids[1:].ravel())))
    assert tuple(port.id for port in support.ports) == ("y_min", "y_max")
    for port in support.ports:
        assert port.measure_unit == LENGTH
        assert_bits(port.measures, g.x_widths_m)
    assert np.array_equal(system.layout.constraint_mask()[count:].reshape(2, count), device.fixed_carrier_mask)
    assert system.recombination.device.grain_recombination is device.grain_recombination
    assert all(term.kind != STORAGE for term in system.assembler.terms)
    assert classical_dc_2d(device).identity == system.identity
    assert AssembledDC2DProblem(device, system=system).at(.002, 1.0).binding is system
    for block, carrier in (("electron", "n"), ("hole", "p")):
        for side, port in (("left", "y_min"), ("right", "y_max")):
            if device.column.channel(carrier, side).kind == "fixed_reservoir":
                observer = system.assembler.reaction_flux_ordered_identity(block, port,
                    normal_faces=system.geometry.normal_faces(port), transverse_faces=system.geometry.transverse_faces)
                assert len(observer) == 64
                assert observer == system.assembler.reaction_flux_ordered_identity(block, port,
                    normal_faces=system.geometry.normal_faces(port), transverse_faces=system.geometry.transverse_faces)
    with pytest.raises(ValueError, match="y_min or y_max"):
        system.geometry.normal_faces("x_min")
    with pytest.raises(ValueError, match="bound to this device"):
        AssembledDC2DProblem(replace(device), system=system)


@pytest.mark.parametrize("name", FACTORIES)
@pytest.mark.parametrize("axis", ("x", "y"))
def test_x_and_y_directional_jacobians(
    name: str, axis: str, cases: dict[str, ReferenceCase], request: pytest.FixtureRequest,
) -> None:
    device = cases[name].device
    assert isinstance(device, CompiledDevice2D)
    _, u = dc_assembly.coordinates(device, lateral=True)
    problem = AssembledDC2DProblem(device, dc_assembly.VOLTAGE_V, 0.0)
    matrix = problem.jacobian(u)
    g = device.geometry
    coordinate = g.x_m / g.lateral_length_m if axis == "x" else device.column.nodes_m / device.column.nodes_m[-1]
    # A nonzero offset prevents the symmetric x midpoint from making the
    # constraint-row direction (and its relative FD scale) nearly zero.
    wave = .37 + np.cos(np.pi * coordinate)
    nodal = np.tile(wave, g.ny) if axis == "x" else np.repeat(wave, g.nx)
    direction = np.r_[.02 * nodal, nodal, -.8 * nodal]
    h = dc_assembly.DIRECTIONAL_STEP
    numerical = (problem.residual(u - 2 * h * direction) - 8 * problem.residual(u - h * direction)
                 + 8 * problem.residual(u + h * direction) - problem.residual(u + 2 * h * direction)) / (12 * h)
    scale = np.maximum(abs(matrix) @ abs(direction), 1e-20)
    errors = np.abs(numerical - matrix @ direction) / scale
    assert float(np.max(errors)) < dc_assembly.DIRECTIONAL_RTOL
    # Actual off-diagonal couplings exist in the requested direction.
    node = g.nx + 1
    neighbor = node + (1 if axis == "x" else g.nx)
    for offset in (0, device.node_count, 2 * device.node_count):
        assert matrix[offset + node, offset + neighbor] != 0.0
    request.node.user_properties.append(("directional", {"factory": name, "axis": axis,
        "h": h, "unchanged_limit": dc_assembly.DIRECTIONAL_RTOL,
        "max_relative_by_block": np.max(errors.reshape(3, device.node_count), axis=1).tolist()}))


def test_original_domain_and_drive_refusals(cases: dict[str, ReferenceCase]) -> None:
    device = cases["dc2d:fixed"].device
    assert isinstance(device, CompiledDevice2D)
    with pytest.raises(TypeError, match="CompiledDevice2D"):
        classical_dc_2d(device.column)  # type: ignore[arg-type]
    for light in (-1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="generation_scale"):
            AssembledDC2DProblem(device, generation_scale=light)
    problem = AssembledDC2DProblem(device)
    u = cases["dc2d:fixed"].states[0].u.copy()
    u[device.node_count] = 1e3
    assert_bits(DC2DProblem(device).residual(u), problem.residual(u))
    with pytest.raises(FloatingPointError, match="admissible domain"):
        problem.jacobian(u)


def test_residual_jacobian_and_readouts_do_not_delegate_to_dc2d(
    cases: dict[str, ReferenceCase], monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = cases["dc2d:composed"]
    device = case.device
    assert isinstance(device, CompiledDevice2D)
    state = case.states[3]
    reference = case.extract(state)
    original = DC2DProblem(device, state.voltage_V, state.generation_scale)
    current = original.terminal_current_A_m(state.u)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the Term binding called the old DC2D implementation")

    for name in ("residual", "jacobian", "fluxes", "recombination_rate", "displacement",
                 "_contact_exchange", "contact_particle_fluxes", "terminal_current_A_m"):
        monkeypatch.setattr(DC2DProblem, name, forbidden)
    candidate = AssembledDC2DProblem(device, state.voltage_V, state.generation_scale)
    assert_bits(reference.F, candidate.residual(state.u))
    assert_matrix(reference.J, candidate.jacobian(state.u))
    assert_bits(current, candidate.terminal_current_A_m(state.u))


def test_compiled_optical_source_uses_the_same_plain_source_run(request: pytest.FixtureRequest) -> None:
    device = compile_device_2d(prepare_optical(), dc_assembly.spatial_input())
    assert device.generation_y.method == "transfer_matrix"
    assert device.geometry.shape == (13, 5)
    _, u = dc_assembly.coordinates(device, lateral=True)
    system = classical_dc_2d(device)
    for light in (0.0, .25, 1.0):
        original = DC2DProblem(device, dc_assembly.VOLTAGE_V, light)
        candidate = AssembledDC2DProblem(device, dc_assembly.VOLTAGE_V, light, system)
        assert_bits(original.residual(u), candidate.residual(u))
        assert_matrix(original.jacobian(u), candidate.jacobian(u))
        assert_bits(original.terminal_current_A_m(u), candidate.terminal_current_A_m(u))
    light = AssembledDC2DProblem(device, dc_assembly.VOLTAGE_V, 1.0, system)
    dark = light.at(dc_assembly.VOLTAGE_V, 0.0)
    increment = light.residual(u) - dark.residual(u)
    expected = (device.generation_source_m3_s * device.geometry.contract_geometry.volumes)
    # Only unconstrained carrier rows receive the integrated pair source.
    for carrier_index, part in enumerate(increment.reshape(3, device.node_count)[1:]):
        mask = ~device.fixed_carrier_mask[carrier_index]
        # This subtraction check has cancellation from F; direct source and
        # full original/candidate arithmetic above are bitwise comparisons.
        np.testing.assert_allclose(part[mask], expected[mask], rtol=1e-14, atol=0.0)
        assert np.all(part[~mask] == 0.0)
    request.node.user_properties.append(("optical", {"method": device.generation_y.method,
        "source_identity": device.generation_y.binding_sha256, "scales": [0.0, .25, 1.0],
        "F_J_ports_changed_words": 0, "qualification": False}))
