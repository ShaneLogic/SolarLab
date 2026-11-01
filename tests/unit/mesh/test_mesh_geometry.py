"""Mesh geometry: grids, ownership, named control volumes and substrate alignment.

M-class checks compare the migrated arrays bit-for-bit with the legacy engine
(read-only import). Total measures are checked against an exact rational
oracle built from the binary64 node positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.discretization import grid as legacy_grid
from perovskite_sim.models import device as legacy_device
from perovskite_sim.physics import generation as legacy_generation
from perovskite_sim.physics import physical_control_volume as legacy_physical
from perovskite_sim.physics import two_sided_interface as legacy_two_sided
from perovskite_sim.solver import mol as legacy_mol
from perovskite_sim.twod import poisson_2d as legacy_poisson_2d
from perovskite_sim.twod import solver_2d as legacy_solver_2d
from perovskite_sim.twod.grid_2d import Grid2D

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
def test_grid_constructors_match_legacy_bitwise(alpha):
    assert same_bits(tanh_grid(9, 3.0e-7, 2.5), legacy_grid.tanh_grid(9, 3.0e-7, 2.5))
    legacy = legacy_grid.multilayer_grid(
        [legacy_grid.Layer(t, n) for t, n in zip(THICKNESSES, INTERVALS)], alpha=alpha)
    assert same_bits(stack_grid(alpha), legacy)
    with pytest.raises(ValueError, match="one value per layer"):
        multilayer_grid(list(zip(THICKNESSES, INTERVALS)), alpha=(1.0, 2.0))


def test_node_ownership_conventions_match_their_historical_consumers():
    x = stack_grid()
    layers = [SimpleNamespace(thickness=t) for t in THICKNESSES]
    legacy_masks = legacy_mol._layer_node_masks(x, layers)
    owner = node_owner(x, THICKNESSES, RIGHT_LAYER_WINS)
    for index, mask in enumerate(legacy_masks):
        assert np.array_equal(owner == index, mask)
    stack = SimpleNamespace(layers=[SimpleNamespace(thickness=t, role=f"r{i}")
                                    for i, t in enumerate(THICKNESSES)])
    roles = legacy_solver_2d._layer_role_at_each_y(x, stack)
    left_owner = node_owner(x, THICKNESSES, LEFT_LAYER_WINS)
    assert [f"r{i}" for i in left_owner] == roles
    # The two conventions differ exactly on the shared interface nodes.
    shared = [s.shared_boundary_node for s in build_interface_stencils(x, interface_positions())]
    differing = np.flatnonzero(owner != left_owner).tolist()
    assert differing == shared
    assert all(owner[node] == left_owner[node] + 1 for node in shared)
    with pytest.raises(ValueError, match="unknown node ownership"):
        node_owner(x, THICKNESSES, "middle")


def test_interface_stencils_and_node_removal_match_legacy_bitwise():
    x = stack_grid()
    positions = interface_positions()
    new = build_interface_stencils(x, positions)
    old = legacy_two_sided.build_two_sided_interface_stencils(x, positions)
    assert [tuple(vars(s).values()) for s in new] == [tuple(vars(s).values()) for s in old]
    assert all(s.shared_boundary_node is not None for s in new)
    reduced = remove_shared_interface_nodes(x, positions)
    assert same_bits(reduced, legacy_two_sided.remove_shared_interface_nodes(x, positions))
    gap = build_interface_stencils(reduced, positions)
    old_gap = legacy_two_sided.build_two_sided_interface_stencils(reduced, positions)
    assert [tuple(vars(s).values()) for s in gap] == [tuple(vars(s).values()) for s in old_gap]
    assert all(s.shared_boundary_node is None and s.right_bulk_node == s.left_bulk_node + 1
               for s in gap)
    for bad in ([x[0]], [positions[1], positions[0]]):
        with pytest.raises(ValueError):
            build_interface_stencils(x, bad)


def test_control_volume_arrays_match_their_legacy_sources_bitwise():
    x = stack_grid()
    reduced = remove_shared_interface_nodes(x, interface_positions())
    assert same_bits(integer_endpoint_widths(x), legacy_generation.dual_cell_widths(x))
    assert same_bits(dual_cell_faces(x), legacy_generation.dual_cell_faces(x))
    assert same_bits(physical_cell_faces(reduced, interface_positions()),
                     legacy_physical.physical_cell_faces(reduced, interface_positions()))
    # 2D: the Neumann x-axis cells and interior y-cells form the Poisson cell area.
    # The legacy 2D builder extrudes the 1D (right-owned) material arrays, so eps
    # is built with the geometry's material owner, not its role owner.
    y, lateral = x, np.linspace(0.0, 4.0e-7, 6)
    geometry = build_geometry(HALF_ENDPOINT_V1, y, interface_positions())
    assert geometry.material_owner_convention == RIGHT_LAYER_WINS
    assert geometry.role_owner_convention == LEFT_LAYER_WINS
    eps_1d = np.where(node_owner(y, THICKNESSES, geometry.material_owner_convention) == 1,
                      25.0, 3.0)
    eps = np.broadcast_to(eps_1d[:, None], (y.size, lateral.size)).copy()
    factor = legacy_poisson_2d.build_poisson_2d_factor(Grid2D(x=lateral, y=y), eps, "neumann")
    expected_area = half_endpoint_widths(y)[1:-1][:, None] * half_endpoint_widths(lateral)[None, :]
    assert same_bits(expected_area, factor.cell_area)
    operator = PoissonOperator1D.from_harmonic_nodes(y, eps_1d, geometry_name=HALF_ENDPOINT_V1)
    for column in range(lateral.size):
        assert same_bits(operator.face_capacitance_F_m2, factor.C_y[:, column])


def test_half_endpoint_material_owner_reproduces_the_legacy_2d_builder_bitwise():
    """The 2D builder's eps (and so C_y) is right-owned; role tagging is left-owned."""
    from perovskite_sim.discretization.grid import Layer
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.twod.grid_2d import build_grid_2d
    from perovskite_sim.twod.microstructure import Microstructure

    preset = ROOT / "perovskite-sim/tests/fixtures/configs/nip_MAPbI3.yaml"
    stack = load_device_from_yaml(str(preset))
    layers = legacy_device.electrical_layers(stack)
    grid = build_grid_2d([Layer(layer.thickness, 6) for layer in layers], 4.0e-7, 5)
    material = legacy_solver_2d.build_material_arrays_2d(
        grid, stack, Microstructure(), lateral_bc="neumann")
    thicknesses = [layer.thickness for layer in layers]
    eps_layers = np.array([layer.params.eps_r for layer in layers])
    geometry = build_geometry(HALF_ENDPOINT_V1, grid.y,
                              np.cumsum(thicknesses)[:-1])
    material_eps = eps_layers[node_owner(grid.y, thicknesses,
                                         geometry.material_owner_convention)]
    role_eps = eps_layers[node_owner(grid.y, thicknesses, geometry.role_owner_convention)]
    assert same_bits(material.eps_r[:, 0], material_eps)
    assert not same_bits(material_eps, role_eps)  # the shared rows really differ
    operator = PoissonOperator1D.from_harmonic_nodes(grid.y, material_eps,
                                                     geometry_name=HALF_ENDPOINT_V1)
    role_operator = PoissonOperator1D.from_harmonic_nodes(grid.y, role_eps,
                                                          geometry_name=HALF_ENDPOINT_V1)
    for column in range(grid.x.size):
        assert same_bits(operator.face_capacitance_F_m2, material.poisson_factor.C_y[:, column])
        assert not same_bits(role_operator.face_capacitance_F_m2,
                             material.poisson_factor.C_y[:, column])
    roles = legacy_solver_2d._layer_role_at_each_y(grid.y, stack)
    assert roles == [layers[i].role for i in node_owner(grid.y, thicknesses,
                                                         geometry.role_owner_convention)]


@dataclass(frozen=True)
class _LegacyMaterialStub:
    """Only the fields ``_prepare_two_sided_material`` reads or replaces."""

    dx_cell: np.ndarray
    eps_r: np.ndarray
    poisson_factor: object
    interface_V_partition_2: tuple
    interface_faces: tuple = ()
    iface_qss_two_sided_trace: bool = False
    iface_qss_interface_faces: tuple = ()
    iface_qss_left_nodes: tuple = ()
    iface_qss_right_nodes: tuple = ()
    iface_qss_interface_positions_m: tuple = ()
    iface_qss_left_distances_m: tuple = ()
    iface_qss_right_distances_m: tuple = ()
    physical_cell_faces_m: object = field(default=None)


@pytest.mark.parametrize("physical", [False, True])
def test_two_sided_geometries_match_the_qf_material_preparation_bitwise(physical):
    from perovskite_sim.experiments import quasi_fermi_steady_state as legacy_qf
    from perovskite_sim.physics.poisson import factor_poisson

    positions = interface_positions()
    reduced = remove_shared_interface_nodes(stack_grid(), positions)
    owner = node_owner(reduced, THICKNESSES, RIGHT_LAYER_WINS)
    eps = np.array([3.0, 25.0, 9.0])[owner]
    stack = SimpleNamespace(layers=[SimpleNamespace(thickness=t, role="absorber")
                                    for t in THICKNESSES])
    material = _LegacyMaterialStub(legacy_generation.dual_cell_widths(reduced), eps,
                                   factor_poisson(reduced, eps), (None,) * len(positions))
    prepared = legacy_qf._prepare_two_sided_material(reduced, stack, material,
                                                     physical_boundary_volumes=physical)
    name = PHYSICAL_BOUNDARY_TWO_SIDED_V1 if physical else TWO_SIDED_TRACE_V1
    geometry = build_geometry(name, reduced, positions)
    assert same_bits(geometry.cell_measures_m, prepared.dx_cell)
    operator = PoissonOperator1D.for_geometry(geometry, eps)
    assert same_bits(operator.face_capacitance_F_m2, prepared.poisson_factor.C)
    assert same_bits(operator.interior_widths_m, prepared.poisson_factor.h_cell)
    assert tuple(p.position_m for p in geometry.interfaces) == prepared.iface_qss_interface_positions_m
    assert tuple(p.left_node for p in geometry.interfaces) == prepared.iface_qss_left_nodes
    assert tuple(p.left_distance_m for p in geometry.interfaces) == \
        prepared.iface_qss_left_distances_m
    assert tuple(p.right_distance_m for p in geometry.interfaces) == \
        prepared.iface_qss_right_distances_m
    if physical:
        assert same_bits(geometry.cell_faces_m, prepared.physical_cell_faces_m)
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
    # Each measure carries a few roundings of node-scale numbers.
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
def test_substrate_prefix_aligns_interfaces_with_the_legacy_helpers(prefix):
    substrate = [StackLayer(f"glass{i}", 1.0e-3 * (i + 1), "substrate") for i in range(prefix)]
    electrical = [StackLayer(f"L{i}", t, role) for i, (t, role)
                  in enumerate(zip(THICKNESSES, ("HTL", "absorber", "ETL")))]
    layers = substrate + electrical
    pairs = tuple((float(i), float(10 * i)) for i in range(len(layers) - 1))
    defects = tuple(f"defect{i}" for i in range(len(layers) - 2))  # one short: padding case
    stack = ElectricalStack.from_layers(layers)
    legacy_stack = SimpleNamespace(
        layers=[SimpleNamespace(name=layer.name, thickness=layer.thickness_m, role=layer.role)
                for layer in layers],
        interfaces=pairs, interface_defects=defects)
    assert [layer.name for layer in stack.electrical] == \
        [layer.name for layer in legacy_device.electrical_layers(legacy_stack)]
    assert stack.substrate_prefix == prefix
    assert stack.electrical_interface_data(pairs) == legacy_device.electrical_interfaces(legacy_stack)
    assert stack.electrical_interface_data(defects, pad_missing=True) == \
        legacy_device.electrical_interface_defects(legacy_stack)
    assert stack.full_interface_index(0) == prefix
    assert stack.optical_offset_m == sum(layer.thickness for layer in legacy_stack.layers
                                         if layer.role == "substrate")
    from perovskite_sim.experiments.quasi_fermi_steady_state import _interface_positions
    assert stack.interface_positions_m == _interface_positions(legacy_stack)
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
