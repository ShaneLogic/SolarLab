"""Gauss law through the public TermSink (N class: same rows, contract accumulation).

The rows are bitwise equal to the historical residual and contact
displacements on every finite state; the only byte difference is the sign of
an exact zero, because the contract accumulator starts from ``+0.0``.
"""

from __future__ import annotations

import numpy as np
import pytest

from solarlab.mesh import (
    HALF_ENDPOINT_V1, INTEGER_ENDPOINT_V1, PHYSICAL_BOUNDARY_TWO_SIDED_V1, RIGHT_LAYER_WINS,
    TWO_SIDED_TRACE_V1, build_geometry, multilayer_grid, node_owner,
    remove_shared_interface_nodes,
)
from solarlab.physics.electrostatics import PoissonOperator1D, contact_displacement
from solarlab.physics.electrostatics.gauss_assembly import gauss_balance_with_term_sink

THICKNESSES = (2.0e-7, 4.0e-7, 1.5e-7)
EPS_LAYERS = (3.0, 25.0, 9.0)
POSITIONS = [THICKNESSES[0], THICKNESSES[0] + THICKNESSES[1]]
FULL_GRID = multilayer_grid(list(zip(THICKNESSES, (9, 14, 7))))
GEOMETRY_GRIDS = {
    INTEGER_ENDPOINT_V1: FULL_GRID,
    HALF_ENDPOINT_V1: FULL_GRID,
    TWO_SIDED_TRACE_V1: remove_shared_interface_nodes(FULL_GRID, POSITIONS),
    PHYSICAL_BOUNDARY_TWO_SIDED_V1: remove_shared_interface_nodes(FULL_GRID, POSITIONS),
}


def same_bits(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def operator_for(name):
    grid = GEOMETRY_GRIDS[name]
    geometry = build_geometry(name, grid, POSITIONS)
    eps = np.asarray(EPS_LAYERS)[node_owner(grid, THICKNESSES, RIGHT_LAYER_WINS)]
    return geometry, PoissonOperator1D.for_geometry(geometry, eps)


@pytest.mark.parametrize("name", sorted(GEOMETRY_GRIDS))
def test_term_sink_gauss_rows_equal_the_residual_and_contact_displacements_bitwise(name):
    geometry, operator = operator_for(name)
    rng = np.random.default_rng(17)
    for case in range(60):
        size = geometry.node_count
        phi = rng.normal(scale=10.0 ** rng.uniform(-6.0, 1.0), size=size)
        rho = rng.normal(scale=10.0 ** rng.uniform(-3.0, 7.0), size=size)
        if case % 10 == 0:
            phi[:] = phi[0]          # flat potential: displacement rows are exact zeros
        if case % 13 == 0:
            rho[:] = 0.0
        balance = gauss_balance_with_term_sink(operator, geometry, phi, rho)
        assert same_bits(balance[1:-1], operator.interior_residual(phi, rho))
        contacts = contact_displacement(operator.face_displacement(phi), rho,
                                        geometry.cell_measures_m)
        assert same_bits([balance[0], balance[-1]], [-contacts[0], contacts[1]])


@pytest.mark.parametrize("name", sorted(GEOMETRY_GRIDS))
def test_term_sink_gauss_differs_only_in_the_sign_of_an_exact_zero(name):
    """The N-class reason: a historical ``-0.0`` row is returned as ``+0.0``."""
    geometry, operator = operator_for(name)
    phi = np.zeros(geometry.node_count)
    phi[::2] = -0.0
    rho = np.full(geometry.node_count, -0.0)
    balance = gauss_balance_with_term_sink(operator, geometry, phi, rho)
    legacy = operator.interior_residual(phi, rho)
    np.testing.assert_array_equal(balance[1:-1], legacy)   # equal values (0 == -0)
    assert np.all(balance[1:-1] == 0.0)
    assert np.any(np.signbit(legacy)) and not np.any(np.signbit(balance[1:-1]))
