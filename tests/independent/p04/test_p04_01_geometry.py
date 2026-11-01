"""P04-01 oracle: named geometries, total measures, substrate alignment, ports."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import p04_adapter as adapter
import p04_legacy as legacy
from p04_gates import gate
from p04_oracle_geometry import (
    ALL_GEOMETRIES,
    ALL_GEOMETRIES_V2,
    GEOMETRY_QF_TWO_SIDED_TRACE,
    GEOMETRY_LEGACY_1D,
    GEOMETRY_LEGACY_2D,
    GEOMETRY_R1_TWO_SIDED,
    effective_interface_positions_integer_endpoint,
    electrical_alignment,
    exact_sum,
    expected_total_measure,
    half_endpoint_widths,
    integer_endpoint_widths,
    periodic_wrap_widths,
    port_outward_particle_flux,
    right_owner_layer_index,
    two_sided_faces,
    two_sided_trace_widths,
    two_sided_widths,
)
from test_p04_01_poisson import TH, interface_node_grid, two_sided_grid


def _grids():
    rng = np.random.default_rng(5)
    yield interface_node_grid()
    yield interface_node_grid((3, 9, 2), a=1.5)
    yield np.sort(np.concatenate([[0.0, 700e-9], rng.uniform(0, 700e-9, 50)]))


def test_closed_form_total_measures():
    g = gate("P0401-MEAS")
    for x in _grids():
        L = float(x[-1] - x[0])
        h = np.diff(x)
        assert abs(exact_sum(integer_endpoint_widths(x)) - (L + 0.5 * (h[0] + h[-1]))) <= g.rel_limit * L
        assert abs(exact_sum(half_endpoint_widths(x)) - L) <= g.rel_limit * L
        assert abs(exact_sum(periodic_wrap_widths(x)) - (L + 0.5 * (h[0] + h[-1]))) <= g.rel_limit * L
    x = two_sided_grid()
    assert abs(exact_sum(two_sided_widths(x, tuple(np.cumsum(TH)[:-1]))) - sum(TH)) <= g.rel_limit * sum(TH)


def test_two_sided_material_measures_equal_thicknesses():
    g = gate("P0401-LAYER-MEAS")
    x = two_sided_grid((7, 13, 5))
    faces = two_sided_faces(x, tuple(np.cumsum(TH)[:-1]))
    bounds = np.concatenate(([0.0], np.cumsum(TH)))
    for k, d in enumerate(TH):
        inside = (faces[:-1] >= bounds[k] - 1e-21) & (faces[1:] <= bounds[k + 1] + 1e-21)
        assert abs(exact_sum(np.diff(faces)[inside]) - d) <= g.rel_limit * d


def test_two_sided_rejects_a_node_on_a_plane():
    with pytest.raises(ValueError):
        two_sided_faces(interface_node_grid(), tuple(np.cumsum(TH)[:-1]))


def test_legacy_dual_cells_are_the_integer_endpoint_geometry():
    for x in _grids():
        assert np.array_equal(legacy.generation().dual_cell_widths(x), integer_endpoint_widths(x))


def test_legacy_physical_faces_are_the_two_sided_geometry():
    x = two_sided_grid((7, 13, 5))
    planes = tuple(np.cumsum(TH)[:-1])
    assert np.array_equal(legacy.physical_control_volume().physical_cell_faces(x, planes), two_sided_faces(x, planes))


def test_legacy_node_ownership_is_right_layer_wins():
    from perovskite_sim.solver.mol import _layer_node_masks

    x = interface_node_grid()
    layers = [SimpleNamespace(thickness=d) for d in TH]
    masks = _layer_node_masks(x, layers)
    owner = right_owner_layer_index(x, TH)
    for k, m in enumerate(masks):
        assert np.array_equal(m, owner == k)
    planes = effective_interface_positions_integer_endpoint(x, TH)
    for plane, xi in zip(planes, np.cumsum(TH)[:-1]):
        node = int(np.argmin(np.abs(x - xi)))
        assert plane == 0.5 * (x[node - 1] + x[node]) < xi


@pytest.mark.parametrize("geometry", ALL_GEOMETRIES_V2)
def test_candidate_control_volumes(geometry):
    g = gate("P0401-MEAS")
    planes = tuple(np.cumsum(TH)[:-1])
    if geometry == GEOMETRY_R1_TWO_SIDED:
        x, used = two_sided_grid(), planes
        ref = two_sided_widths(x, planes)
    elif geometry == GEOMETRY_QF_TWO_SIDED_TRACE:
        x, used = two_sided_grid(), planes
        ref = two_sided_trace_widths(x, planes)
    else:
        x, used = interface_node_grid(), ()
        ref = integer_endpoint_widths(x) if geometry == GEOMETRY_LEGACY_1D else half_endpoint_widths(x)
    w = np.asarray(adapter.control_volume_widths(geometry, x, used), dtype=float)
    assert abs(exact_sum(w) - expected_total_measure(geometry, x)) <= g.rel_limit * float(x[-1])
    if geometry == GEOMETRY_QF_TWO_SIDED_TRACE:
        # The oracle forms these widths by a different expression (rounding-equal);
        # the bitwise QF comparison against the legacy engine is in MSweepV1.
        assert np.max(np.abs(w - ref)) <= 16 * np.finfo(float).eps * float(x[-1])
    else:
        # R3-1 assertion, unchanged: the oracle uses the historical expressions.
        assert np.array_equal(w, ref) or adapter.change_class("control_volumes") != "M"
    if adapter.change_class("control_volumes") == "M" and geometry != GEOMETRY_QF_TWO_SIDED_TRACE:
        legacy_ref = (legacy.physical_control_volume().physical_cell_faces(x, planes)
                      if geometry == GEOMETRY_R1_TWO_SIDED else None)
        if geometry == GEOMETRY_LEGACY_1D:
            assert np.array_equal(w, legacy.generation().dual_cell_widths(x))
        elif legacy_ref is not None:
            assert np.array_equal(w, np.diff(legacy_ref))


def test_candidate_geometry_records_planes_distances_normals_and_ports():
    """V1 P04-01 step 1: true plane, left/right distances, normals, port areas."""

    x = two_sided_grid()
    planes = tuple(np.cumsum(TH)[:-1])
    for oracle_id in (GEOMETRY_R1_TWO_SIDED, GEOMETRY_QF_TWO_SIDED_TRACE):
        geom = adapter.build_geometry(oracle_id, x, planes)
        assert len(geom.interfaces) == 2
        for plane, xi in zip(geom.interfaces, planes):
            assert plane.position_m == xi and plane.shared_node is None
            assert plane.right_node == plane.left_node + 1
            assert plane.left_distance_m == xi - x[plane.left_node]
            assert plane.right_distance_m == x[plane.right_node] - xi
        left, right = geom.ports
        assert (left.outward_normal, right.outward_normal) == (-1, 1)
        assert left.area_m2 == right.area_m2 == 1.0 and geom.face_normal == 1
    xi_grid = interface_node_grid()
    geom = adapter.build_geometry(GEOMETRY_LEGACY_1D, xi_grid, planes)
    for plane, xi in zip(geom.interfaces, planes):
        assert plane.shared_node is not None and xi_grid[plane.shared_node] == xi
    assert geom.geometric_cells is False  # full-width contact cells are not geometric


def test_candidate_names_three_distinct_geometries():
    names = adapter.geometry_names()
    assert len(set(names)) >= 3, "1D integer-endpoint, 2D half-endpoint and R1 two-sided must be distinct"


# ---------------------------------------------------------------- substrate


CASES = (
    (("HTL", "absorber", "ETL"), (200e-9, 400e-9, 100e-9)),
    (("substrate", "HTL", "absorber", "ETL"), (1e-3, 200e-9, 400e-9, 100e-9)),
    (("substrate", "substrate", "ETL", "absorber"), (1e-3, 1e-4, 50e-9, 500e-9)),
)


@pytest.mark.parametrize("roles, th", CASES)
def test_legacy_substrate_prefix_alignment(roles, th):
    g = gate("P0401-SUBSTRATE")
    ref = electrical_alignment(roles, th)
    n_full = len(roles) - 1
    tags = tuple(f"iface{k}" for k in range(n_full))
    stack = SimpleNamespace(
        layers=tuple(SimpleNamespace(role=r, thickness=t) for r, t in zip(roles, th)),
        interfaces=tags,
        interface_defects=tags,
    )
    dev = legacy.device_alignment()
    expected = tuple(f"iface{k}" for k in ref.electrical_interface_full_indices)
    assert dev.electrical_interfaces(stack) == expected
    assert dev.electrical_interface_defects(stack) == expected
    elec = dev.electrical_layers(stack)
    pos = np.cumsum([layer.thickness for layer in elec])[:-1]
    assert np.allclose(pos, ref.electrical_interface_positions_m, rtol=g.rel_limit, atol=0.0)


def test_substrate_must_be_a_contiguous_prefix():
    with pytest.raises(ValueError):
        electrical_alignment(("HTL", "substrate", "ETL"), (1.0, 1.0, 1.0))
    stack = SimpleNamespace(layers=(SimpleNamespace(role="HTL", thickness=1.0),
                                    SimpleNamespace(role="substrate", thickness=1.0)))
    with pytest.raises(ValueError):
        legacy.device_alignment().electrical_layers(stack)


@pytest.mark.parametrize("roles, th", CASES)
def test_candidate_substrate_prefix_alignment(roles, th):
    ref = electrical_alignment(roles, th)
    idx, pos = adapter.electrical_interface_alignment(roles, th)
    assert tuple(idx) == ref.electrical_interface_full_indices
    assert np.allclose(pos, ref.electrical_interface_positions_m, rtol=gate("P0401-SUBSTRATE").rel_limit, atol=0.0)


# ---------------------------------------------------------------- ports


def test_port_normals_and_conservation():
    flux = np.full(9, 3.7e21)  # divergence-free +x particle flux
    left, right = port_outward_particle_flux(flux, 1.0)
    assert left == -3.7e21 and right == 3.7e21 and left + right == 0.0
    lx = 500e-9  # 2D unit depth: the port measure is the lateral length
    left2, right2 = port_outward_particle_flux(flux, lx)
    assert left2 + right2 == 0.0 and right2 == 3.7e21 * lx
    assert GEOMETRY_LEGACY_2D in ALL_GEOMETRIES
