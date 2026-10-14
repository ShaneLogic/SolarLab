"""Electrostatics: shared charge ledger, Poisson algebraic residual, Gauss traces.

M class (bit-identical to the legacy engine, read-only import): Poisson face
capacitances/control widths per channel, the LAPACK elimination, the interior
residual, the space-charge expression, face/contact displacement, trace
residual/solution and the sheet-charge row weights.

Physics acceptance (V1 P04-01): layered capacitance, constant-charge analytic
potential, MMS, Gauss/port charge balance and two-sided Gauss jumps. Geometry
differences (legacy node-at-interface harmonic face) are reported separately
from algorithm error.
"""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.experiments import quasi_fermi_steady_state as legacy_qf
from perovskite_sim.physics import physical_control_volume as legacy_physical
from perovskite_sim.physics import poisson as legacy_poisson
from perovskite_sim.physics import two_sided_interface as legacy_two_sided
from perovskite_sim.solver import degenerate_equilibrium as legacy_equilibrium
from perovskite_sim.solver import mol as legacy_mol

from solarlab.contracts import VOLT, VariableSpec
from solarlab.mesh import (
    INTEGER_ENDPOINT_V1, PHYSICAL_BOUNDARY_TWO_SIDED_V1, RIGHT_LAYER_WINS, TWO_SIDED_TRACE_V1,
    build_geometry, multilayer_grid, node_owner, remove_shared_interface_nodes,
)
from solarlab.physics.electrostatics import (
    ChargeEntry, ChargeLedger, PoissonFactorization, PoissonOperator1D, SheetChargeEntry,
    TraceElectrostatics, add_sheet_charge_to_interior_residual, contact_displacement,
    electrostatic_trace_residual_and_jacobian,
    interface_trace_layout, require_carrier_trace, require_occupancy, sheet_charge_row_weights,
    solve_electrostatic_traces,
)

EPS0 = 8.854187817e-12
Q = 1.602176634e-19
THICKNESSES = (2.0e-7, 4.0e-7, 1.5e-7)
INTERVALS = (9, 14, 7)
EPS_LAYERS = (3.0, 25.0, 9.0)


def same_bits(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def positions():
    out, total = [], 0.0
    for t in THICKNESSES[:-1]:
        total += t
        out.append(total)
    return out


def stack_grid(intervals=INTERVALS, alpha=3.0):
    return multilayer_grid(list(zip(THICKNESSES, intervals)), alpha=alpha)


def nodal_eps(x):
    return np.asarray(EPS_LAYERS)[node_owner(x, THICKNESSES, RIGHT_LAYER_WINS)]


# ---------------------------------------------------------------- M class


@pytest.mark.parametrize("n_nodes", [3, 4, 5, 31])
def test_harmonic_channel_and_lapack_elimination_match_legacy_bitwise(n_nodes):
    rng = np.random.default_rng(n_nodes)
    x = np.cumsum(rng.uniform(1e-9, 3e-8, n_nodes))
    eps = rng.uniform(2.0, 30.0, n_nodes)
    rho = rng.normal(scale=1e3, size=n_nodes)
    operator = PoissonOperator1D.from_harmonic_nodes(x, eps)
    legacy_factor = legacy_poisson.factor_poisson(x, eps)
    assert same_bits(operator.face_capacitance_F_m2, legacy_factor.C)
    assert same_bits(operator.interior_widths_m, legacy_factor.h_cell)
    factor = PoissonFactorization.factor(operator)
    for name in ("dl", "d", "du", "du2"):
        assert same_bits(getattr(factor, name), getattr(legacy_factor, name))
    assert np.array_equal(factor.ipiv, legacy_factor.ipiv)
    assert same_bits(factor.solve(rho, 0.1, -0.7),
                     legacy_poisson.solve_poisson_prefactored(legacy_factor, rho, 0.1, -0.7))
    explicit = PoissonOperator1D.from_finite_volume(legacy_factor.C, legacy_factor.h_cell,
                                                    geometry_name="explicit")
    legacy_explicit = legacy_poisson.factor_poisson_from_finite_volume(legacy_factor.C,
                                                                       legacy_factor.h_cell)
    assert same_bits(PoissonFactorization.factor(explicit).solve(rho, 0.0, 1.0),
                     legacy_poisson.solve_poisson_prefactored(legacy_explicit, rho, 0.0, 1.0))


def test_interior_residual_and_displacements_match_legacy_bitwise():
    x = stack_grid()
    eps = nodal_eps(x)
    rng = np.random.default_rng(5)
    phi, rho = rng.normal(size=x.size), rng.normal(scale=1e4, size=x.size)
    operator = PoissonOperator1D.from_harmonic_nodes(x, eps)
    material = SimpleNamespace(poisson_factor=legacy_poisson.factor_poisson(x, eps))
    assert same_bits(operator.interior_residual(phi, rho),
                     legacy_equilibrium._poisson_residual(phi, rho, material))
    displacement = operator.face_displacement(phi)
    assert same_bits(displacement, -material.poisson_factor.C * np.diff(phi))
    widths = build_geometry(INTEGER_ENDPOINT_V1, x, positions()).cell_measures_m
    assert same_bits(contact_displacement(displacement, rho, widths),
                     legacy_physical.physical_contact_displacement(displacement, rho, widths))
    full = operator.residual(phi, rho, 0.25, -0.5)
    assert full[0] == phi[0] - 0.25 and full[-1] == phi[-1] + 0.5
    assert same_bits(full[1:-1], operator.interior_residual(phi, rho))


@pytest.mark.parametrize("dual_ion", [False, True])
def test_shared_charge_ledger_reproduces_the_legacy_space_charge(dual_ion):
    rng = np.random.default_rng(9)
    n_nodes = 25
    p, n = 10.0 ** rng.uniform(10, 24, n_nodes), 10.0 ** rng.uniform(10, 24, n_nodes)
    ion, ion0 = 10.0 ** rng.uniform(20, 24, n_nodes), 10.0 ** rng.uniform(20, 24, n_nodes)
    acceptors, donors = 10.0 ** rng.uniform(15, 23, n_nodes), 10.0 ** rng.uniform(15, 23, n_nodes)
    ledger = ChargeLedger(n_nodes)
    for entry in (ChargeEntry("carriers.p", "drift_diffusion", 1, p),
                  ChargeEntry("carriers.n", "drift_diffusion", -1, n),
                  ChargeEntry("ions.positive", "ions", 1, ion, ion0),
                  ChargeEntry("doping.acceptors", "doping", -1, acceptors),
                  ChargeEntry("doping.donors", "doping", 1, donors)):
        ledger = ledger.with_entry(entry)
    expected = legacy_mol._charge_density(p, n, ion, ion0, acceptors, donors)
    if dual_ion:
        neg, neg0 = 10.0 ** rng.uniform(20, 24, n_nodes), 10.0 ** rng.uniform(20, 24, n_nodes)
        ledger = ledger.with_entry(ChargeEntry("ions.negative", "ions", -1, neg, neg0,
                                               group="negative_ions"))
        expected = legacy_mol._charge_density(p, n, ion, ion0, acceptors, donors, neg, neg0)
    assert same_bits(ledger.space_charge_C_m3(), expected)
    assert ledger.space_charge_derivative("carriers.n") == -Q
    assert ledger.space_charge_derivative("ions.positive") == Q
    with pytest.raises(ValueError, match="duplicate"):
        ledger.with_entry(ChargeEntry("carriers.p", "other", 1, p))
    with pytest.raises(ValueError, match="node count"):
        ledger.with_entry(ChargeEntry("short", "other", 1, p[:-1]))
    with pytest.raises(ValueError, match="non-negative"):
        ChargeEntry("negative", "other", 1, -p)
    with pytest.raises(ValueError, match="nonzero integer"):
        ChargeEntry("neutral", "other", 0, p)


def test_trace_electrostatics_match_legacy_bitwise():
    geometry = TraceElectrostatics(3.0e-9, 7.0e-9, 4.0, 22.0, 0.031, -2.0e-4)
    legacy_geometry = legacy_two_sided.TwoSidedInterfaceGeometry(3.0e-9, 7.0e-9, 4.0, 22.0,
                                                                  0.031, -2.0e-4)
    bulk = legacy_two_sided.TwoSidedBulkState(0.12, 0.47, 1e20, 1e10, 1e15, 1e16)
    traces = np.array([0.2, 0.25])
    new_r, new_j = electrostatic_trace_residual_and_jacobian(traces, geometry, 0.12, 0.47)
    old_r, old_j = legacy_two_sided.electrostatic_trace_residual_and_jacobian(
        traces, legacy_geometry, bulk)
    assert same_bits(new_r, old_r) and same_bits(new_j, old_j)
    solved = solve_electrostatic_traces(geometry, 0.12, 0.47)
    legacy_solved = legacy_two_sided.solve_electrostatic_traces(legacy_geometry, bulk)
    assert solved == (legacy_solved.phi_left_V, legacy_solved.phi_right_V)


def test_sheet_charge_row_weights_match_the_qf_poisson_assembly_bitwise():
    reduced = remove_shared_interface_nodes(stack_grid(), positions())
    geometry = build_geometry(PHYSICAL_BOUNDARY_TWO_SIDED_V1, reduced, positions())
    eps = nodal_eps(reduced)
    sheets = np.array([3.0e-4, -1.5e-4])
    rng = np.random.default_rng(13)
    raw = rng.normal(size=reduced.size - 2)
    legacy_raw = raw.copy()
    planes = geometry.interfaces
    system = SimpleNamespace(
        node_count=reduced.size,
        mat=SimpleNamespace(
            iface_qss_left_nodes=tuple(p.left_node for p in planes),
            iface_qss_right_nodes=tuple(p.right_node for p in planes),
            iface_qss_left_distances_m=tuple(p.left_distance_m for p in planes),
            iface_qss_right_distances_m=tuple(p.right_distance_m for p in planes),
            eps_r=eps,
        ),
    )
    legacy_qf._QuasiFermiSystem._add_interface_sheet_charge_to_poisson(
        system, legacy_raw, SimpleNamespace(incremental_sheet_charge_C_m2=sheets))
    add_sheet_charge_to_interior_residual(raw, planes, eps, sheets)
    assert same_bits(raw, legacy_raw)
    w_l, w_r = sheet_charge_row_weights(eps[planes[0].left_node], planes[0].left_distance_m,
                                        eps[planes[0].right_node], planes[0].right_distance_m)
    assert w_l + w_r == pytest.approx(1.0, abs=2e-16)


# ---------------------------------------------------------------- physics


def series_capacitance_exact():
    return Fraction(1) / sum(Fraction(t) / (Fraction(EPS0) * Fraction(e))
                             for t, e in zip(THICKNESSES, EPS_LAYERS))


def test_layered_capacitance_physical_geometry_is_exact():
    """Zero charge, three dielectrics: uniform D = C_series*V on every face."""
    voltage = 0.85
    reduced = remove_shared_interface_nodes(stack_grid(), positions())
    for name in (PHYSICAL_BOUNDARY_TWO_SIDED_V1, TWO_SIDED_TRACE_V1):
        geometry = build_geometry(name, reduced, positions())
        operator = PoissonOperator1D.for_geometry(geometry, nodal_eps(reduced))
        rho = np.zeros(reduced.size)
        phi = PoissonFactorization.factor(operator).solve(rho, voltage, 0.0)
        displacement = operator.face_displacement(phi)
        expected = float(series_capacitance_exact() * Fraction(voltage))
        np.testing.assert_allclose(displacement, expected, rtol=2e-13)
        assert np.max(np.abs(operator.residual(phi, rho, voltage, 0.0))) <= 1e-12 * expected
        # Electrode charge through the declared ports: Q_left = +C V, Q_right = -C V.
        left, right = (port.terminal_port() for port in geometry.ports)
        contacts = contact_displacement(displacement, rho, geometry.cell_measures_m)
        np.testing.assert_allclose(left.charge([contacts[0]]), [expected], rtol=2e-13)
        np.testing.assert_allclose(right.charge([contacts[1]]), [-expected], rtol=2e-13)


def test_layered_capacitance_legacy_channel_is_its_discrete_series_and_converges_first_order():
    """The node-at-interface harmonic face is a geometry difference, not solver error."""
    voltage = 0.85
    exact = series_capacitance_exact()
    deviations = []
    for scale in (1, 2, 4, 8):
        x = stack_grid(tuple(scale * n for n in INTERVALS), alpha=1e-6)
        operator = PoissonOperator1D.for_geometry(build_geometry(INTEGER_ENDPOINT_V1, x,
                                                                 positions()), nodal_eps(x))
        phi = PoissonFactorization.factor(operator).solve(np.zeros(x.size), voltage, 0.0)
        displacement = operator.face_displacement(phi)
        # Independent rational oracle of this channel's discrete series capacitance.
        eps = [Fraction(float(e)) for e in nodal_eps(x)]
        nodes = [Fraction(float(v)) for v in x]
        inverse = sum((nodes[i + 1] - nodes[i]) * (eps[i] + eps[i + 1])
                      / (2 * eps[i] * eps[i + 1] * Fraction(EPS0)) for i in range(len(x) - 1))
        discrete = Fraction(1) / inverse
        np.testing.assert_allclose(displacement, float(discrete * Fraction(voltage)), rtol=2e-12)
        deviations.append(float(abs(discrete - exact) / exact))
    ratios = np.array(deviations[:-1]) / np.array(deviations[1:])
    assert deviations[0] > 1e-3  # a real geometric difference at coarse resolution
    assert np.all(np.abs(ratios - 2.0) < 0.1), ratios  # first order in the interface cell


@pytest.mark.parametrize("name", [INTEGER_ENDPOINT_V1, PHYSICAL_BOUNDARY_TWO_SIDED_V1])
def test_constant_charge_gives_the_analytic_parabola_on_a_nonuniform_grid(name):
    length, eps_r, rho0 = 5.0e-7, 12.0, 4.0e4
    x = multilayer_grid([(length, 37)], alpha=2.5)
    geometry = build_geometry(name, x, ())
    operator = PoissonOperator1D.for_geometry(geometry, np.full(x.size, eps_r))
    phi_left, phi_right = 0.2, -0.4
    phi = PoissonFactorization.factor(operator).solve(np.full(x.size, rho0), phi_left, phi_right)
    exact = (phi_left + (phi_right - phi_left) * x / length
             + rho0 / (2 * EPS0 * eps_r) * x * (length - x))
    scale = max(abs(phi_left), abs(phi_right), rho0 * length**2 / (8 * EPS0 * eps_r))
    assert np.max(np.abs(phi - exact)) <= 1e-12 * scale


def _mms_error(name, scale, two_layers):
    l1, l2 = (3e-7, 2e-7) if two_layers else (5e-7, 0.0)
    eps1, eps2 = 4.0, (20.0 if two_layers else 4.0)
    length = l1 + l2
    d0, d1 = 1e-3, 5e-4

    def displacement(x):
        return d0 + d1 * np.sin(np.pi * x / length)

    def phi_exact(x):
        def integral(a, b, eps):
            return (d0 * (b - a) - d1 * length / np.pi
                    * (np.cos(np.pi * b / length) - np.cos(np.pi * a / length))) / (EPS0 * eps)
        if not two_layers:
            return 0.3 - integral(0.0, x, eps1)
        return 0.3 - np.where(x <= l1, integral(0.0, np.minimum(x, l1), eps1),
                              integral(0.0, l1, eps1) + integral(l1, x, eps2))

    layers = [(l1, 3 * scale), (l2, 2 * scale)] if two_layers else [(l1, 5 * scale)]
    x = multilayer_grid(layers, alpha=1e-6)
    planes = [l1] if two_layers else []
    if name == PHYSICAL_BOUNDARY_TWO_SIDED_V1 and two_layers:
        x = remove_shared_interface_nodes(x, planes)
    geometry = build_geometry(name, x, planes)
    thickness = (l1, l2) if two_layers else (l1,)
    eps = np.array([eps1, eps2])[node_owner(x, thickness, RIGHT_LAYER_WINS)]
    operator = PoissonOperator1D.for_geometry(geometry, eps)
    faces = geometry.cell_faces_m
    cell_charge = displacement(faces[1:]) - displacement(faces[:-1])  # exact cell integral
    rho = np.zeros(x.size)
    rho[1:-1] = cell_charge[1:-1] / operator.interior_widths_m
    phi = PoissonFactorization.factor(operator).solve(rho, phi_exact(x[0]), phi_exact(x[-1]))
    return float(np.max(np.abs(phi - phi_exact(x))))


@pytest.mark.parametrize("name,two_layers,order", [
    (INTEGER_ENDPOINT_V1, False, 2.0),
    (PHYSICAL_BOUNDARY_TWO_SIDED_V1, False, 2.0),
    (PHYSICAL_BOUNDARY_TWO_SIDED_V1, True, 2.0),
    (INTEGER_ENDPOINT_V1, True, 1.0),  # geometry difference at the dielectric jump
])
def test_mms_observed_order(name, two_layers, order):
    errors = np.array([_mms_error(name, scale, two_layers) for scale in (8, 16, 32, 64)])
    observed = np.log2(errors[:-1] / errors[1:])
    # Asymptotic order on the two finest pairs; no pair may fall far below it.
    assert np.all(np.abs(observed[1:] - order) < 0.1), observed
    assert np.all(observed > order - 0.25), observed


def test_jacobian_agrees_with_the_direct_residual():
    reduced = remove_shared_interface_nodes(stack_grid(), positions())
    geometry = build_geometry(PHYSICAL_BOUNDARY_TWO_SIDED_V1, reduced, positions())
    operator = PoissonOperator1D.for_geometry(geometry, nodal_eps(reduced))
    rng = np.random.default_rng(17)
    phi, rho = rng.normal(scale=0.3, size=reduced.size), rng.normal(scale=3e4, size=reduced.size)
    direction = rng.normal(size=reduced.size)
    jacobian = operator.jacobian()
    base = operator.residual(phi, rho, 0.1, 0.2)
    moved = operator.residual(phi + direction, rho, 0.1, 0.2)
    np.testing.assert_allclose(jacobian @ direction, moved - base,
                               atol=1e-9 * np.max(np.abs(operator.face_capacitance_F_m2)))
    assert np.array_equal(operator.charge_jacobian()[1:-1], operator.interior_widths_m)


def test_gauss_total_charge_balances_the_electrode_charges():
    reduced = remove_shared_interface_nodes(stack_grid(), positions())
    geometry = build_geometry(PHYSICAL_BOUNDARY_TWO_SIDED_V1, reduced, positions())
    operator = PoissonOperator1D.for_geometry(geometry, nodal_eps(reduced))
    rng = np.random.default_rng(23)
    ledger = ChargeLedger(reduced.size, interface_count=2)
    ledger = ledger.with_entry(ChargeEntry("doping.donors", "doping", 1,
                                           10.0 ** rng.uniform(20, 22, reduced.size)))
    ledger = ledger.with_entry(ChargeEntry("doping.acceptors", "doping", -1,
                                           10.0 ** rng.uniform(20, 22, reduced.size)))
    rho = ledger.space_charge_C_m3()
    phi = PoissonFactorization.factor(operator).solve(rho, 0.6, 0.0)
    contacts = contact_displacement(operator.face_displacement(phi), rho, geometry.cell_measures_m)
    left, right = (port.terminal_port() for port in geometry.ports)
    electrodes = float(left.charge([contacts[0]])[0] + right.charge([contacts[1]])[0])
    device = ledger.total_charge_C_m2(geometry.cell_measures_m)
    scale = float(np.sum(np.abs(rho) * geometry.cell_measures_m))
    assert abs(device + electrodes) <= 1e-12 * scale
    with pytest.raises(ValueError, match="interface outside"):
        ledger.with_sheet(SheetChargeEntry("sheet", "interface", 2, 1e-4))


def test_two_sided_gauss_jump_equals_series_elimination_with_weighted_sheet_charge():
    reduced = remove_shared_interface_nodes(stack_grid(), positions())
    geometry = build_geometry(PHYSICAL_BOUNDARY_TWO_SIDED_V1, reduced, positions())
    eps = nodal_eps(reduced)
    operator = PoissonOperator1D.for_geometry(geometry, eps)
    rng = np.random.default_rng(29)
    phi = rng.normal(scale=0.2, size=reduced.size)
    ledger = ChargeLedger(reduced.size, interface_count=2)
    ledger = ledger.with_sheet(SheetChargeEntry("traps.0", "interface_traps", 0, 2.0e-4))
    ledger = ledger.with_sheet(SheetChargeEntry("traps.1", "interface_traps", 1, -7.0e-5))
    ledger = ledger.with_sheet(SheetChargeEntry("fixed.1", "interface_fixed", 1, 1.0e-5))
    sigma = ledger.sheet_charge_C_m2()
    assert np.array_equal(sigma, [2.0e-4, -7.0e-5 + 1.0e-5])
    # Explicit traces: solve the Gauss jump at each plane, then form each side's flux.
    folded = operator.interior_residual(phi, np.zeros(reduced.size))
    add_sheet_charge_to_interior_residual(folded, geometry.interfaces, eps, sigma)
    for plane, sheet in zip(geometry.interfaces, sigma):
        traces = TraceElectrostatics.from_plane(plane, eps[plane.left_node], eps[plane.right_node],
                                                sheet_charge_C_m2=sheet)
        phi_ls, phi_rs = solve_electrostatic_traces(traces, phi[plane.left_node],
                                                    phi[plane.right_node])
        c_l = EPS0 * eps[plane.left_node] / plane.left_distance_m
        c_r = EPS0 * eps[plane.right_node] / plane.right_distance_m
        d_left = c_l * (phi[plane.left_node] - phi_ls)     # D just left of the plane
        d_right = c_r * (phi_rs - phi[plane.right_node])   # D just right of the plane
        assert d_right - d_left == pytest.approx(sheet, rel=1e-12, abs=1e-12 * abs(d_left))
        # The folded rows see exactly the eliminated series face plus w*sigma.
        series = operator.face_capacitance_F_m2[plane.left_node]
        w_l, w_r = sheet_charge_row_weights(eps[plane.left_node], plane.left_distance_m,
                                            eps[plane.right_node], plane.right_distance_m)
        assert -d_left == pytest.approx(
            -series * (phi[plane.left_node] - phi[plane.right_node]) + w_l * sheet, rel=1e-11)
        assert d_right == pytest.approx(
            series * (phi[plane.left_node] - phi[plane.right_node]) + w_r * sheet, rel=1e-11)
    assert np.all(np.isfinite(folded))


def test_interface_traces_declare_carriers_potentials_and_occupancy_separately():
    declaration = interface_trace_layout(2)
    supports = {spec.id: spec.support for spec in declaration.layout.variables}
    assert {s.id for s in declaration.carrier_traces} == {
        "interfaces.n_left", "interfaces.p_left", "interfaces.n_right", "interfaces.p_right"}
    assert all(spec.unit == declaration.carrier_traces[0].unit and spec.lower == 0.0
               for spec in declaration.carrier_traces)
    assert all(spec.unit == VOLT for spec in declaration.potential_traces)
    occupancy = declaration.trap_occupancy
    assert (occupancy.lower, occupancy.upper) == (0.0, 1.0)
    assert supports["interfaces.n_left"] != supports["interfaces.n_right"]
    assert supports["interfaces.trap_occupancy"] not in {supports["interfaces.n_left"],
                                                          supports["interfaces.n_right"]}
    require_occupancy(occupancy)
    for carrier in declaration.carrier_traces:
        require_carrier_trace(carrier)
        with pytest.raises(ValueError, match="not a trap occupancy"):
            require_occupancy(carrier)
    with pytest.raises(ValueError, match="not a carrier trace"):
        require_carrier_trace(occupancy)
    unbounded = VariableSpec("f", "x", "interfaces.plane", (2,), occupancy.unit, lower=0.0)
    with pytest.raises(ValueError, match="not a trap occupancy"):
        require_occupancy(unbounded)
    with pytest.raises(ValueError, match="at least one interface"):
        interface_trace_layout(0)
    with pytest.raises(ValueError, match="finite and positive"):
        TraceElectrostatics(0.0, 1e-9, 3.0, 4.0)


def test_poisson_operator_rejects_invalid_channels():
    with pytest.raises(ValueError, match="positive vector"):
        PoissonOperator1D(np.array([1.0, -1.0]), np.array([1.0]), "bad")
    with pytest.raises(ValueError, match="match faces"):
        PoissonOperator1D(np.array([1.0, 1.0]), np.array([1.0, 1.0]), "bad")
    with pytest.raises(ValueError, match="name its geometry"):
        PoissonOperator1D(np.array([1.0, 1.0]), np.array([1.0]), "")
    operator = PoissonOperator1D.from_harmonic_nodes(np.array([0.0, 1.0, 2.0]), np.ones(3))
    with pytest.raises(ValueError, match="finite array"):
        operator.residual(np.ones(4), np.zeros(3), 0.0, 0.0)
