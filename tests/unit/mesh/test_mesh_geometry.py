"""Mesh geometry: grids, ownership, named control volumes and substrate alignment.

Total measures are checked against an exact rational oracle built from the
binary64 node positions.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from solarlab.contracts import LENGTH, ONE, CellSource, EquationSpec, FaceFlux, Layout, Support
from solarlab.contracts import PARTICLE, SECOND, TermSink
from solarlab.mesh import (
    GEOMETRY_NAMES, HALF_ENDPOINT_V1, INTEGER_ENDPOINT_V1, LEFT_LAYER_WINS,
    PHYSICAL_BOUNDARY_TWO_SIDED_V1, RIGHT_LAYER_WINS, TWO_SIDED_TRACE_V1, ElectricalStack,
    StackLayer, build_geometry, build_interface_stencils, multilayer_grid, node_owner,
    remove_shared_interface_nodes, tanh_grid,
)
from solarlab.mesh.geometry import (
    dual_cell_faces, half_endpoint_widths, integer_endpoint_widths, physical_cell_faces,
    two_sided_trace_widths,
)
from solarlab.physics.electrostatics import PoissonOperator1D

ROOT = Path(__file__).resolve().parents[3]
THICKNESSES = (2.0e-7, 4.0e-7, 1.5e-7)
INTERVALS = (7, 11, 5)


def same_bits(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def stack_grid(alpha=3.0):
    return multilayer_grid(list(zip(THICKNESSES, INTERVALS)), alpha=alpha)


def interface_positions():
    return ElectricalStack.from_layers(
        [StackLayer(f"L{i}", t, "absorber") for i, t in enumerate(THICKNESSES)]
    ).interface_positions_m


@pytest.mark.parametrize("alpha", [3.0, 1.5, (2.0, 3.5, 1.0)])
def test_grid_constructors_span_layers_and_validate_alpha(alpha):
    single = tanh_grid(9, 3.0e-7, 2.5)
    assert single[0] == 0.0 and single[-1] == pytest.approx(3.0e-7, rel=0.0, abs=1e-22)
    assert np.all(np.diff(single) > 0.0)
    grid = stack_grid(alpha)
    assert grid.size == sum(INTERVALS) + 1
    assert grid[0] == 0.0 and grid[-1] == pytest.approx(sum(THICKNESSES), rel=1e-15)
    assert np.all(np.diff(grid) > 0.0)
    with pytest.raises(ValueError, match="one value per layer"):
        multilayer_grid(list(zip(THICKNESSES, INTERVALS)), alpha=(1.0, 2.0))


def test_node_ownership_conventions_differ_only_on_shared_interface_nodes():
    x = stack_grid()
    owner = node_owner(x, THICKNESSES, RIGHT_LAYER_WINS)
    left_owner = node_owner(x, THICKNESSES, LEFT_LAYER_WINS)
    shared = [s.shared_boundary_node for s in build_interface_stencils(x, interface_positions())]
    differing = np.flatnonzero(owner != left_owner).tolist()
    assert differing == shared
    assert all(owner[node] == left_owner[node] + 1 for node in shared)
    with pytest.raises(ValueError, match="unknown node ownership"):
        node_owner(x, THICKNESSES, "middle")


def test_interface_stencils_and_node_removal_invariants():
    x = stack_grid()
    positions = interface_positions()
    new = build_interface_stencils(x, positions)
    assert all(s.shared_boundary_node is not None for s in new)
    reduced = remove_shared_interface_nodes(x, positions)
    assert reduced.size == x.size - len(positions)
    gap = build_interface_stencils(reduced, positions)
    assert all(s.shared_boundary_node is None and s.right_bulk_node == s.left_bulk_node + 1
               for s in gap)
    for bad in ([x[0]], [positions[1], positions[0]]):
        with pytest.raises(ValueError):
            build_interface_stencils(x, bad)


def test_control_volume_arrays_and_half_endpoint_owner_conventions():
    x = stack_grid()
    reduced = remove_shared_interface_nodes(x, interface_positions())
    np.testing.assert_allclose(half_endpoint_widths(x), np.diff(dual_cell_faces(x)), rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(integer_endpoint_widths(x)[1:-1], np.diff(dual_cell_faces(x))[1:-1], rtol=1e-13, atol=0.0)
    assert integer_endpoint_widths(x)[0] == x[1] - x[0]
    assert integer_endpoint_widths(x)[-1] == x[-1] - x[-2]
    p_faces = physical_cell_faces(reduced, interface_positions())
    assert p_faces[0] == reduced[0] and p_faces[-1] == reduced[-1]
    geometry = build_geometry(HALF_ENDPOINT_V1, x, interface_positions())
    assert geometry.material_owner_convention == RIGHT_LAYER_WINS
    assert geometry.role_owner_convention == LEFT_LAYER_WINS
    eps_layers = np.array([3.0, 25.0, 9.0])
    material_eps = eps_layers[node_owner(x, THICKNESSES, geometry.material_owner_convention)]
    role_eps = eps_layers[node_owner(x, THICKNESSES, geometry.role_owner_convention)]
    assert not same_bits(material_eps, role_eps)
    operator = PoissonOperator1D.from_harmonic_nodes(x, material_eps, geometry_name=HALF_ENDPOINT_V1)
    role_operator = PoissonOperator1D.from_harmonic_nodes(x, role_eps, geometry_name=HALF_ENDPOINT_V1)
    assert not same_bits(operator.face_capacitance_F_m2, role_operator.face_capacitance_F_m2)
    assert same_bits(geometry.cell_measures_m, half_endpoint_widths(x))


@pytest.mark.parametrize("physical", [False, True])
def test_two_sided_geometries_preserve_series_capacitance_and_measures(physical):
    positions = interface_positions()
    reduced = remove_shared_interface_nodes(stack_grid(), positions)
    owner = node_owner(reduced, THICKNESSES, RIGHT_LAYER_WINS)
    eps = np.array([3.0, 25.0, 9.0])[owner]
    name = PHYSICAL_BOUNDARY_TWO_SIDED_V1 if physical else TWO_SIDED_TRACE_V1
    geometry = build_geometry(name, reduced, positions)
    operator = PoissonOperator1D.for_geometry(geometry, eps)
    assert same_bits(operator.interior_widths_m, geometry.cell_measures_m[1:-1])
    assert same_bits(two_sided_trace_widths(reduced, positions),
                     build_geometry(TWO_SIDED_TRACE_V1, reduced, positions).cell_measures_m)


def _exact(values):
    return [Fraction(float(v)) for v in np.asarray(values, dtype=float)]


@pytest.mark.parametrize("name", GEOMETRY_NAMES)
def test_every_named_geometry_has_its_exact_total_measure(name):
    positions = interface_positions()
    x = stack_grid()
    if name in {TWO_SIDED_TRACE_V1, PHYSICAL_BOUNDARY_TWO_SIDED_V1}:
        x = remove_shared_interface_nodes(x, positions)
    geometry = build_geometry(name, x, positions)
    nodes = _exact(x)
    length = nodes[-1] - nodes[0]
    endpoint_excess = ((nodes[1] - nodes[0]) + (nodes[-1] - nodes[-2])) / 2
    expected = length + endpoint_excess if name in {INTEGER_ENDPOINT_V1, TWO_SIDED_TRACE_V1} \
        else length
    exact_sum = sum(_exact(geometry.cell_measures_m), Fraction(0))
    allowance = Fraction(8 * len(x)) * Fraction(np.spacing(float(x[-1])))
    assert abs(exact_sum - expected) <= allowance, name
    assert abs(Fraction(geometry.total_measure_m) - exact_sum) <= Fraction(np.spacing(1e-6))
    assert geometry.geometric_cells == (expected == length)
    assert geometry.name == name
    left, right = geometry.ports
    assert (left.outward_normal, right.outward_normal) == (-1, 1)
    assert (left.node, right.node) == (0, geometry.node_count - 1)
    assert left.area_m2 == right.area_m2 == 1.0 and geometry.face_normal == 1
    assert left.terminal_port().outward_normal == -1
    assert [p.position_m for p in geometry.interfaces] == list(positions)
    for plane in geometry.interfaces:
        assert plane.left_distance_m > 0 and plane.right_distance_m > 0
        assert geometry.nodes_m[plane.left_node] < plane.position_m < geometry.nodes_m[plane.right_node]
        assert (plane.shared_node is None) == (name in {TWO_SIDED_TRACE_V1,
                                                         PHYSICAL_BOUNDARY_TWO_SIDED_V1})


def test_geometry_builders_reject_mismatched_topologies():
    positions = interface_positions()
    x = stack_grid()
    reduced = remove_shared_interface_nodes(x, positions)
    with pytest.raises(ValueError, match="without shared interface nodes"):
        build_geometry(PHYSICAL_BOUNDARY_TWO_SIDED_V1, x, positions)
    with pytest.raises(ValueError, match="keeps a node on every interface"):
        build_geometry(INTEGER_ENDPOINT_V1, reduced, positions)
    with pytest.raises(ValueError, match="unknown historical geometry"):
        build_geometry("unified_endpoint", x, positions)


def test_contract_geometry_applies_lengths_once_for_a_unit_cross_section():
    geometry = build_geometry(HALF_ENDPOINT_V1, stack_grid(), interface_positions())
    contract = geometry.to_contract_geometry()
    assert contract.volume_unit == LENGTH and contract.face_unit == ONE
    n = geometry.node_count
    layout = Layout((Support("cells", "cell", (n,)), Support("faces", "face", (n - 1,))), (),
                    (EquationSpec("balance", "carriers", "cells", (n,),
                                  PARTICLE / (LENGTH * LENGTH) / SECOND),))
    sink = TermSink(layout, "balance", contract)
    sink.add(CellSource("uniform", "cells", np.ones(n), PARTICLE / (LENGTH * LENGTH * LENGTH) / SECOND))
    sink.add(FaceFlux("zero-flux", "faces", np.zeros(n - 1), PARTICLE / (LENGTH * LENGTH) / SECOND))
    assert sink.value().sum() == pytest.approx(geometry.extent_m, rel=1e-14)


@pytest.mark.parametrize("prefix", [0, 1, 2])
def test_substrate_prefix_aligns_interfaces_and_optical_offset(prefix):
    substrate = [StackLayer(f"glass{i}", 1.0e-3 * (i + 1), "substrate") for i in range(prefix)]
    electrical = [StackLayer(f"L{i}", t, role) for i, (t, role)
                  in enumerate(zip(THICKNESSES, ("HTL", "absorber", "ETL")))]
    layers = substrate + electrical
    pairs = tuple((float(i), float(10 * i)) for i in range(len(layers) - 1))
    defects = tuple(f"defect{i}" for i in range(len(layers) - 2))
    stack = ElectricalStack.from_layers(layers)
    assert [layer.name for layer in stack.electrical] == [layer.name for layer in electrical]
    assert stack.substrate_prefix == prefix
    assert stack.electrical_interface_data(pairs) == pairs[prefix:]
    padded_defects = (*defects, None)
    assert stack.electrical_interface_data(defects, pad_missing=True) == padded_defects[prefix:]
    assert stack.full_interface_index(0) == prefix
    assert stack.optical_offset_m == sum(layer.thickness_m for layer in substrate)
    assert stack.interface_positions_m == tuple(np.cumsum(THICKNESSES[:-1]).tolist())
    x = multilayer_grid([(layer.thickness_m, n) for layer, n in zip(stack.electrical, INTERVALS)])
    stencils = build_interface_stencils(x, stack.interface_positions_m)
    for stencil in stencils:
        assert stencil.shared_boundary_node is not None
        assert x[stencil.shared_boundary_node] == stencil.interface_position_m
    assert stack.optical_coordinates(x[0]) == stack.optical_offset_m


def test_substrate_must_be_a_contiguous_prefix():
    for layers in ([StackLayer("a", 1e-7, "absorber"), StackLayer("g", 1e-3, "substrate")],
                   [StackLayer("g", 1e-3, "substrate"), StackLayer("a", 1e-7, "absorber"),
                    StackLayer("g2", 1e-3, "substrate"), StackLayer("b", 1e-7, "ETL")]):
        with pytest.raises(ValueError, match="contiguous prefix"):
            ElectricalStack.from_layers(layers)
    with pytest.raises(ValueError, match="no electrical layers"):
        ElectricalStack.from_layers([StackLayer("g", 1e-3, "substrate")])
    with pytest.raises(IndexError):
        ElectricalStack.from_layers([StackLayer("a", 1e-7, "absorber")]).full_interface_index(0)
    with pytest.raises(ValueError, match="finite and positive"):
        StackLayer("bad", 0.0, "absorber")
