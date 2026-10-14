"""P04-01 oracle: Poisson as an algebraic residual (capacitance, charge, MMS, Gauss).

Three subjects again: the oracle FV reference (exact face data), the legacy
1D rule (``physics/poisson.py`` with right-owner nodal eps/rho, i.e. the
``legacy_1d_integer_endpoint_v1`` channel) and the candidate (adapter).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import p04_adapter as adapter
import p04_legacy as legacy
from p04_gates import gate
from p04_oracle_electrostatics import (
    EPS_0_LEGACY,
    LayeredStack,
    ManufacturedPoisson,
    exact_cell_charge,
    exact_face_capacitance,
    fv_residual,
    fv_residual_scale,
    layered_constant_charge_solution,
    legacy_rule_cell_charge,
    legacy_rule_face_capacitance,
    observed_orders,
    series_capacitance,
    shifted_stack_for_legacy_rule,
    solve_fv_dirichlet,
)
from p04_oracle_geometry import (
    GEOMETRY_LEGACY_1D,
    GEOMETRY_QF_TWO_SIDED_TRACE,
    GEOMETRY_R1_TWO_SIDED,
    right_owner_layer_index,
    two_sided_faces,
)

Q = 1.602176634e-19
TH = (200e-9, 400e-9, 100e-9)
ER = (3.0, 24.1, 10.0)


def tanh_nodes(N: int, L: float, a: float = 3.0) -> np.ndarray:
    xi = np.linspace(-1.0, 1.0, N + 1)
    return L * (1.0 + np.tanh(a * xi) / np.tanh(a)) / 2.0


def interface_node_grid(Ns=(20, 40, 15), th=TH, a=3.0) -> np.ndarray:
    """Independent re-implementation of the historical multilayer tanh grid."""

    segs, off = [], 0.0
    for k, (d, N) in enumerate(zip(th, Ns)):
        s = tanh_nodes(N, d, a) + off
        segs.append(s[1:] if k else s)
        off += d
    return np.concatenate(segs)


def two_sided_grid(Ns=(8, 16, 4), th=TH, a=2.0) -> np.ndarray:
    """Cell-centred nodes inside each layer (no node on any plane)."""

    pts, off = [0.0], 0.0
    for d, N in zip(th, Ns):
        edges = tanh_nodes(N, d, a) + off
        pts.extend(0.5 * (edges[:-1] + edges[1:]))
        off += d
    pts.append(off)
    return np.array(pts)


def dual_faces(x: np.ndarray) -> np.ndarray:
    return np.concatenate(([x[0]], 0.5 * (x[:-1] + x[1:]), [x[-1]]))


def _limit(g, phi_exact):
    return g.abs_limit * max(1.0, float(np.max(np.abs(phi_exact))))


def _legacy_solve(x, stack: LayeredStack, phi_l, phi_r):
    owner = right_owner_layer_index(x, stack.thicknesses_m)
    eps_node = np.array(stack.eps_r)[owner]
    rho_node = np.array(stack.rho())[owner]
    return legacy.poisson().solve_poisson(x, eps_node, rho_node, phi_l, phi_r)


def test_grid_reimplementation_matches_legacy_multilayer_grid():
    g = legacy.grid()
    x_legacy = g.multilayer_grid([g.Layer(d, n) for d, n in zip(TH, (20, 40, 15))])
    assert np.array_equal(interface_node_grid(), x_legacy)


def test_vacuum_permittivity_identity():
    """M-class migrations must keep the legacy eps0 bit-for-bit."""

    from perovskite_sim.constants import EPS_0

    assert EPS_0 == EPS_0_LEGACY
    eps0 = adapter.poisson_vacuum_permittivity()
    assert eps0 == EPS_0_LEGACY or adapter.change_class("poisson_1d") == "P"


# ------------------------------------------------------- series capacitor


def test_oracle_fv_reproduces_series_capacitor_exactly():
    g, gc = gate("P0401-POI-EXACT"), gate("P0401-POI-CAP")
    stack = LayeredStack(TH, ER)
    for x in (interface_node_grid(), interface_node_grid((5, 3, 7), a=1.0)):
        C = exact_face_capacitance(x, stack)
        phi = solve_fv_dirichlet(C, np.zeros(x.size), 0.0, 1.1)
        sol = layered_constant_charge_solution(stack, 0.0, 1.1)
        assert np.max(np.abs(phi - sol.potential(x))) <= _limit(g, sol.potential(x))
        c_disc = 1.0 / math.fsum(1.0 / C)
        assert abs(c_disc / series_capacitance(stack) - 1.0) <= gc.rel_limit


def test_legacy_rule_is_exact_for_the_shifted_structure_only():
    """Geometry difference, not algorithm error: planes act at x_I - h_{I-1}/2."""

    g = gate("P0401-POI-GEOM")
    stack = LayeredStack(TH, ER)
    x = interface_node_grid()
    phi = _legacy_solve(x, stack, 0.0, 1.1)
    shifted = layered_constant_charge_solution(shifted_stack_for_legacy_rule(x, stack), 0.0, 1.1)
    true = layered_constant_charge_solution(stack, 0.0, 1.1)
    lim = _limit(g, shifted.potential(x))
    assert np.max(np.abs(phi - shifted.potential(x))) <= lim
    assert np.max(np.abs(phi - true.potential(x))) >= 100.0 * lim, "test must discriminate"
    owner = right_owner_layer_index(x, TH)
    C_rule = legacy_rule_face_capacitance(np.array(ER)[owner], x)
    assert abs((1.0 / math.fsum(1.0 / C_rule)) / series_capacitance(shifted_stack_for_legacy_rule(x, stack)) - 1.0) <= 1e-13


@pytest.mark.parametrize("geometry", (GEOMETRY_LEGACY_1D, GEOMETRY_R1_TWO_SIDED, GEOMETRY_QF_TWO_SIDED_TRACE))
def test_candidate_series_capacitor(geometry):
    g = gate("P0401-POI-GEOM" if geometry == GEOMETRY_LEGACY_1D else "P0401-POI-EXACT")
    stack = LayeredStack(TH, ER)
    x = interface_node_grid() if geometry == GEOMETRY_LEGACY_1D else two_sided_grid()
    phi = np.asarray(adapter.poisson_solve(geometry, x, TH, ER, (0.0,) * 3, (0.0, 0.0), 0.0, 1.1))
    ref_stack = shifted_stack_for_legacy_rule(x, stack) if geometry == GEOMETRY_LEGACY_1D else stack
    ref = layered_constant_charge_solution(ref_stack, 0.0, 1.1).potential(x)
    assert np.max(np.abs(phi - ref)) <= _limit(g, ref)


# ------------------------------------------------------- constant charge


def test_uniform_charge_single_layer_is_exact_on_nonuniform_mesh():
    g = gate("P0401-POI-EXACT")
    stack = LayeredStack((400e-9,), (24.1,), (Q * 1e22,))
    x = tanh_nodes(37, 400e-9)
    sol = layered_constant_charge_solution(stack, 0.0, 0.9)
    phi_ref = solve_fv_dirichlet(exact_face_capacitance(x, stack), exact_cell_charge(dual_faces(x), stack), 0.0, 0.9)
    phi_leg = _legacy_solve(x, stack, 0.0, 0.9)
    exact = sol.potential(x)
    assert np.max(np.abs(exact - (0.9 * x / 400e-9))) > 0.1, "charge term must be visible"
    for phi in (phi_ref, phi_leg):
        assert np.max(np.abs(phi - exact)) <= _limit(g, exact)
    # The algebraic residual of the exact nodal solution vanishes to rounding.
    C, Qc = exact_face_capacitance(x, stack), exact_cell_charge(dual_faces(x), stack)
    r = fv_residual(exact, C, Qc)
    assert np.all(np.abs(r) <= 1e-9 * fv_residual_scale(exact, C, Qc))


def test_layered_charge_with_interface_sheets_exact_on_interface_nodes():
    """Piecewise-quadratic exactness with exact cell charge (sheets inside cells)."""

    g = gate("P0401-POI-EXACT")
    stack = LayeredStack(TH, ER, (Q * 1e23, -Q * 5e21, -Q * 2e23), (Q * 1e15, -Q * 3e15))
    x = interface_node_grid()
    sol = layered_constant_charge_solution(stack, 0.0, 1.1)
    Qc = exact_cell_charge(dual_faces(x), stack)
    phi = solve_fv_dirichlet(exact_face_capacitance(x, stack), Qc, 0.0, 1.1)
    assert np.max(np.abs(phi - sol.potential(x))) <= _limit(g, sol.potential(x))


def _with_trace_nodes(xb: np.ndarray):
    planes = tuple(np.cumsum(TH)[:-1])
    xa = np.sort(np.concatenate([xb, planes]))
    return xa, np.isin(xa, planes)


def test_gauss_jump_through_zero_volume_trace_node():
    """Two-sided geometry: each plane is its own trace unknown; D+ - D- = sigma.

    rho = 0 makes D piecewise constant, so the trace scheme is exact and the
    jump at every plane must equal the declared sheet charge.
    """

    g, ge = gate("P0401-GAUSS-JUMP"), gate("P0401-POI-EXACT")
    sig = (Q * 2e15, -Q * 1e15)
    stack = LayeredStack(TH, ER, (), sig)
    xa, is_plane = _with_trace_nodes(two_sided_grid())
    Qc = np.zeros(xa.size)
    Qc[is_plane] = sig
    C = exact_face_capacitance(xa, stack)
    phi = solve_fv_dirichlet(C, Qc, 0.0, 1.1)
    exact = layered_constant_charge_solution(stack, 0.0, 1.1).potential(xa)
    assert np.max(np.abs(phi - exact)) <= _limit(ge, exact)
    D_face = -C * np.diff(phi)
    for k, idx in enumerate(np.flatnonzero(is_plane)):
        jump = D_face[idx] - D_face[idx - 1]
        assert abs(jump - sig[k]) <= g.rel_limit * (abs(sig[k]) + abs(D_face[idx]))


@pytest.mark.parametrize("geometry", (GEOMETRY_R1_TWO_SIDED, GEOMETRY_QF_TWO_SIDED_TRACE))
def test_candidate_sheet_charge_two_sided_exact(geometry):
    g = gate("P0401-POI-EXACT")
    sig = (Q * 2e15, -Q * 1e15)
    x = two_sided_grid()
    phi = np.asarray(adapter.poisson_solve(geometry, x, TH, ER, (0.0,) * 3, sig, 0.0, 1.1))
    exact = layered_constant_charge_solution(LayeredStack(TH, ER, (), sig), 0.0, 1.1).potential(x)
    assert np.max(np.abs(phi - exact)) <= _limit(g, exact)


# ------------------------------------------------------- MMS


def _mms(eps=ER):
    return ManufacturedPoisson(TH, eps, F0=Q * 1e22 * 100e-9, F1=-Q * 3e21 * 50e-9, phi_left=0.0)


def _mms_errors(kind: str, m: ManufacturedPoisson):
    errs, hs = [], []
    for f in (1, 2, 4, 8, 16):
        if kind == "two_sided":
            x = two_sided_grid((8 * f, 16 * f, 4 * f))
            faces = two_sided_faces(x, tuple(np.cumsum(TH)[:-1]))
            phi = solve_fv_dirichlet(exact_face_capacitance(x, m.as_layered_stack()), m.cell_charge(faces),
                                     m.phi_left, m.phi_right)
        else:
            x = interface_node_grid((10 * f, 20 * f, 8 * f))
            if kind == "interface_nodes_exact":
                phi = solve_fv_dirichlet(exact_face_capacitance(x, m.as_layered_stack()),
                                         m.cell_charge(dual_faces(x)), m.phi_left, m.phi_right)
            else:  # historical rule
                owner = right_owner_layer_index(x, TH)
                phi = legacy.poisson().solve_poisson(
                    x, np.array(m.eps_r)[owner], m.rho(x), m.phi_left, m.phi_right)
        errs.append(np.max(np.abs(phi - m.potential(x))))
        hs.append(np.max(np.diff(x)))
    return np.array(hs), np.array(errs)


@pytest.mark.parametrize("kind", ("two_sided", "interface_nodes_exact"))
def test_mms_second_order_with_exact_interface_data(kind):
    g = gate("P0401-POI-MMS")
    h, e = _mms_errors(kind, _mms())
    p = observed_orders(h, e)
    assert p[-1] >= g.rel_limit, f"{kind}: orders {np.round(p, 3)}"


def test_mms_historical_rule_is_second_order_without_contrast_first_order_with_it():
    g = gate("P0401-POI-MMS")
    h, e = _mms_errors("legacy", _mms((10.0, 10.0, 10.0)))
    assert observed_orders(h, e)[-1] >= g.rel_limit
    h, e = _mms_errors("legacy", _mms())
    p = observed_orders(h, e)
    assert 0.8 <= p[-1] <= 1.5, f"historical eps-contrast order {np.round(p, 3)} (geometry shift)"


def test_candidate_mms_two_sided():
    g = gate("P0401-POI-MMS")
    m = _mms()
    errs, hs = [], []
    for f in (1, 2, 4, 8, 16):
        x = two_sided_grid((8 * f, 16 * f, 4 * f))
        faces = two_sided_faces(x, tuple(np.cumsum(TH)[:-1]))
        phi = np.asarray(adapter.poisson_solve_manufactured(
            GEOMETRY_R1_TWO_SIDED, x, TH, ER, m.rho, m.cell_charge(faces), m.phi_left, m.phi_right))
        errs.append(np.max(np.abs(phi - m.potential(x))))
        hs.append(np.max(np.diff(x)))
    assert observed_orders(np.array(hs), np.array(errs))[-1] >= g.rel_limit


def test_legacy_prefactored_path_equals_sparse_path():
    """Historical consistency of the two legacy entry points on the golden case."""

    stack = LayeredStack(TH, ER, (Q * 1e23, -Q * 5e21, -Q * 2e23))
    x = interface_node_grid()
    owner = right_owner_layer_index(x, TH)
    eps_node, rho_node = np.array(ER)[owner], np.array(stack.rho())[owner]
    poi = legacy.poisson()
    a = poi.solve_poisson(x, eps_node, rho_node, 0.0, 1.1)
    b = poi.solve_poisson_prefactored(poi.factor_poisson(x, eps_node), rho_node, 0.0, 1.1)
    assert np.max(np.abs(a - b)) <= 1e-10 * max(1.0, np.max(np.abs(a)))
    rule = solve_fv_dirichlet(legacy_rule_face_capacitance(eps_node, x), legacy_rule_cell_charge(rho_node, x), 0.0, 1.1)
    assert np.max(np.abs(rule - b)) <= 1e-10 * max(1.0, np.max(np.abs(a)))


# ------------------------------------------------------- TermSink Gauss (N)


@pytest.mark.parametrize("geometry", (GEOMETRY_LEGACY_1D, GEOMETRY_R1_TWO_SIDED, GEOMETRY_QF_TWO_SIDED_TRACE))
def test_candidate_gauss_term_sink_equals_the_algebraic_residual(geometry):
    """N class: TermSink assembly of the same Gauss law (gate P0401-GAUSS-SINK).

    Interior rows equal the oracle FV residual built from the operator's own
    face capacitances and widths to rounding.  Contact rows equal -D(0) and
    +D(L) by Gauss over the end cells; that reading is physical only where the
    contact cells are geometric (not the integer-endpoint geometry, whose
    contact weights are the full adjacent interval).
    """

    gs = gate("P0401-GAUSS-SINK")
    x = interface_node_grid() if geometry == GEOMETRY_LEGACY_1D else two_sided_grid()
    rng = np.random.default_rng(23)
    phi = rng.normal(scale=0.4, size=x.size)
    rho = rng.normal(scale=Q * 1e23, size=x.size)
    geom, _, _, op = adapter.poisson_operator(geometry, x, TH, ER)
    balance = np.asarray(adapter.gauss_balance(geometry, x, TH, ER, phi, rho))
    C = np.asarray(op.face_capacitance_F_m2)
    w = np.asarray(geom.cell_measures_m)
    Qc = rho * w
    ref = fv_residual(phi, C, Qc)
    scale = fv_residual_scale(phi, C, Qc)
    assert np.all(np.abs(balance[1:-1] - ref) <= gs.rel_limit * scale)
    d_face = -C * np.diff(phi)
    left = d_face[0] - Qc[0]
    right = d_face[-1] + Qc[-1]
    assert abs(balance[0] + left) <= gs.rel_limit * (abs(d_face[0]) + abs(Qc[0]))
    assert abs(balance[-1] - right) <= gs.rel_limit * (abs(d_face[-1]) + abs(Qc[-1]))


@pytest.mark.parametrize("geometry", (GEOMETRY_LEGACY_1D, GEOMETRY_R1_TWO_SIDED))
def test_candidate_poisson_residual_is_the_historical_expression(geometry):
    """M class: the interior residual equals the oracle FV residual bitwise-close.

    For the historical channel the face data must be the harmonic nodal mean
    and the widths the dual-cell rule (the shifted-plane structure).
    """

    x = interface_node_grid() if geometry == GEOMETRY_LEGACY_1D else two_sided_grid()
    geom, owner, eps_node, op = adapter.poisson_operator(geometry, x, TH, ER)
    if geometry == GEOMETRY_LEGACY_1D:
        assert np.array_equal(np.asarray(op.face_capacitance_F_m2), legacy_rule_face_capacitance(eps_node, x))
        assert np.array_equal(np.asarray(op.interior_widths_m), legacy_rule_cell_charge(np.ones(x.size), x)[1:-1])
    else:
        exact = exact_face_capacitance(x, LayeredStack(TH, ER))
        assert np.max(np.abs(np.asarray(op.face_capacitance_F_m2) / exact - 1.0)) <= 1e-15 * 8
