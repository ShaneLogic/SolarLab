"""Unit contracts for P08-02a dielectric capacitance, P08-02b geometry, and P08-02c band steps."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from examples.research_1d.steady import load_device
from examples.research_2d import dc_assembly as example_2d
from solarlab.device.compile_2d import compile_device_2d
from solarlab.discretization import (
    DIELECTRIC_RULES,
    LAYER_SERIES_CAPACITANCE_V2,
    NODAL_HARMONIC_DIELECTRIC_V1,
    VACUUM_PERMITTIVITY_F_M,
    series_face_capacitance,
    subface_series_capacitance,
)
from solarlab.mesh import (
    HALF_ENDPOINT_V1,
    INTEGER_ENDPOINT_V1,
    PHYSICAL_BOUNDARY_TWO_SIDED_V1,
    PHYSICAL_HALF_ENDPOINT_V2,
    RIGHT_LAYER_WINS,
    TWO_SIDED_TRACE_V1,
    areal_interface_to_cell_source,
    build_geometry,
    multilayer_grid,
    node_owner,
    physical_half_endpoint_geometry,
    remove_shared_interface_nodes,
)
from solarlab.mesh.geometry import half_endpoint_geometry
from solarlab.mesh.geometry_2d import RectangularGeometry2D
from solarlab.physics.electrostatics import (
    PoissonFactorization,
    PoissonOperator1D,
    TraceElectrostatics,
    add_interface_sheet_charge,
    contact_displacement,
    layer_permittivities_from_geometry,
    layer_series_face_capacitances,
    rectangular_2d_face_capacitances,
    solve_electrostatic_traces,
)
from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.physics.electrostatics.gauss_assembly import gauss_balance_with_term_sink
from solarlab.physics.generation.beer_lambert import BeerLambertBinding
from solarlab.physics.recombination.interface import evaluate_shared_interface_trap
from solarlab.physics.statistics.band_potentials import (
    BandStructure,
    LayerBandInput,
    layered_band_structure,
)
from solarlab.physics.transport.drift_diffusion import (
    BAND_STEP_MODES,
    HOLE_SEPARATE_INCREMENT_V1,
    LEGACY_SUMMED_POTENTIAL_V1,
    legacy_carrier_face_currents,
    legacy_carrier_face_jacobians,
)
from solarlab.physics.transport.electrochemical_flux import (
    carrier_particle_fluxes,
    carrier_transport_potential_steps,
    electron_current_from_particle_flux,
    hole_current_from_particle_flux,
)

EPS_0 = VACUUM_PERMITTIVITY_F_M
Q = ELEMENTARY_CHARGE_C


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    return arr_a.shape == arr_b.shape and arr_a.tobytes() == arr_b.tobytes()


def _solve_2d_dirichlet_gauss(
    geometry: RectangularGeometry2D,
    face_capacitance_F_m2: np.ndarray,
    dirichlet_nodes: np.ndarray,
    dirichlet_values: np.ndarray,
    rho_C_m3: np.ndarray | None = None,
) -> np.ndarray:
    """Solve 2D graph Gauss balance with arbitrary Dirichlet contact subset."""
    n = geometry.node_count
    cg = geometry.contract_geometry
    left, right = cg.pairs.T
    mat = lil_matrix((n, n), dtype=float)
    for a, b, cap in zip(left, right, face_capacitance_F_m2 * cg.face_measures):
        mat[a, a] -= cap
        mat[a, b] += cap
        mat[b, a] += cap
        mat[b, b] -= cap
    rhs: np.ndarray = np.zeros(n, dtype=np.float64)
    if rho_C_m3 is not None:
        rhs -= np.asarray(rho_C_m3, dtype=np.float64) * cg.volumes
    for node, value in zip(dirichlet_nodes, dirichlet_values):
        idx = int(node)
        mat.rows[idx] = [idx]
        mat.data[idx] = [1.0]
        rhs[idx] = float(value)
    return np.asarray(spsolve(mat.tocsc(), rhs), dtype=float)


def test_p08_02a_subface_series_capacitance_homogeneous_0_to_1_ulp() -> None:
    """Homogeneous sub-face reduction agrees with eps_0 * eps / h to 0-1 ULP."""
    assert DIELECTRIC_RULES == (NODAL_HARMONIC_DIELECTRIC_V1, LAYER_SERIES_CAPACITANCE_V2)

    rng = np.random.default_rng(20261114)
    for eps_r in (1.0, 3.7, 11.9, 24.1, 63.3):
        x = np.r_[0.0, np.cumsum(rng.uniform(1.3e-9, 4.7e-8, size=19))]
        h = np.diff(x)
        direct = EPS_0 * eps_r / h
        split_ratio = rng.uniform(0.05, 0.95, size=h.size)
        d_left = split_ratio * h
        d_right = h - d_left
        c_sub = subface_series_capacitance(d_left, eps_r, d_right, eps_r)
        assert np.all(np.abs(c_sub - direct) <= np.spacing(direct))

        # Unsplit faces (d_R == 0 or d_L == 0) match EPS_0 * eps_r / h to 0 ULPs.
        assert _same_bits(subface_series_capacitance(h, eps_r, 0.0, 99.0), direct)
        assert _same_bits(subface_series_capacitance(0.0, 99.0, h, eps_r), direct)

        geom_1d = build_geometry(INTEGER_ENDPOINT_V1, x, ())
        op_v1 = PoissonOperator1D.for_geometry(geom_1d, np.full(x.size, eps_r))
        op_v2 = PoissonOperator1D.for_geometry(
            geom_1d, np.full(x.size, eps_r), dielectric_rule=LAYER_SERIES_CAPACITANCE_V2
        )
        assert np.all(
            np.abs(op_v2.face_capacitance_F_m2 - op_v1.face_capacitance_F_m2)
            <= np.spacing(op_v1.face_capacitance_F_m2)
        )

        y_geom = half_endpoint_geometry(x, ())
        rect = RectangularGeometry2D(np.array([0.0, 1.7e-7, 5.3e-7, 1.0e-6]), y_geom)
        c2d_v1 = rectangular_2d_face_capacitances(
            rect, np.full(x.size, eps_r), dielectric_rule=NODAL_HARMONIC_DIELECTRIC_V1
        )
        c2d_v2 = rectangular_2d_face_capacitances(
            rect, np.full(x.size, eps_r), dielectric_rule=LAYER_SERIES_CAPACITANCE_V2
        )
        assert np.all(np.abs(c2d_v2 - c2d_v1) <= np.spacing(c2d_v1))

    # Scalar evaluation and domain checks.
    scalar_cap = subface_series_capacitance(4.0e-9, 3.0, 2.5e-9, 24.0)
    assert scalar_cap.shape == ()
    assert float(scalar_cap) == pytest.approx(
        series_face_capacitance(4.0e-9, 3.0, 2.5e-9, 24.0), rel=0.0, abs=0.0
    )
    with pytest.raises(ValueError, match="nonnegative"):
        subface_series_capacitance(-1.0e-9, 3.0, 2.0e-9, 4.0)
    with pytest.raises(ValueError, match="positive face spacing"):
        subface_series_capacitance(0.0, 3.0, 0.0, 4.0)
    with pytest.raises(ValueError, match="positive"):
        subface_series_capacitance(1.0e-9, 0.0, 1.0e-9, 4.0)


def test_p08_02a_default_legacy_channels_stay_bit_identical() -> None:
    """Default 1D and 2D dielectric compilation remains bit-identical."""
    thicknesses = (1.5e-7, 3.5e-7, 2.0e-7)
    interfaces = (1.5e-7, 5.0e-7)
    x_shared = multilayer_grid(list(zip(thicknesses, (5, 9, 6))), alpha=2.5)
    eps_shared = np.array([3.5, 24.0, 9.0])[node_owner(x_shared, thicknesses, RIGHT_LAYER_WINS)]

    for name in (INTEGER_ENDPOINT_V1, HALF_ENDPOINT_V1):
        geom = build_geometry(name, x_shared, interfaces)
        default_op = PoissonOperator1D.for_geometry(geom, eps_shared)
        harmonic_op = PoissonOperator1D.from_harmonic_nodes(
            x_shared, eps_shared, geometry_name=name
        )
        assert default_op.dielectric_rule == NODAL_HARMONIC_DIELECTRIC_V1
        assert _same_bits(default_op.face_capacitance_F_m2, harmonic_op.face_capacitance_F_m2)
        assert _same_bits(default_op.interior_widths_m, harmonic_op.interior_widths_m)

    x_two_sided = remove_shared_interface_nodes(x_shared, interfaces)
    eps_two_sided = np.array([3.5, 24.0, 9.0])[
        node_owner(x_two_sided, thicknesses, RIGHT_LAYER_WINS)
    ]
    for name in (TWO_SIDED_TRACE_V1, PHYSICAL_BOUNDARY_TWO_SIDED_V1):
        geom = build_geometry(name, x_two_sided, interfaces)
        default_op = PoissonOperator1D.for_geometry(geom, eps_two_sided)
        explicit_v2 = PoissonOperator1D.for_geometry(
            geom, eps_two_sided, dielectric_rule=LAYER_SERIES_CAPACITANCE_V2
        )
        assert default_op.dielectric_rule == LAYER_SERIES_CAPACITANCE_V2
        assert np.all(
            np.abs(default_op.face_capacitance_F_m2 - explicit_v2.face_capacitance_F_m2)
            <= np.spacing(default_op.face_capacitance_F_m2)
        )
        assert _same_bits(default_op.interior_widths_m, geom.cell_measures_m[1:-1])

    prepared = load_device()
    spatial = example_2d.spatial_input()
    dev_default = compile_device_2d(prepared, spatial)
    dev_explicit_v1 = compile_device_2d(
        prepared, spatial, dielectric_rule=NODAL_HARMONIC_DIELECTRIC_V1
    )
    assert dev_default.sha256 == dev_explicit_v1.sha256
    assert _same_bits(dev_default.face_capacitance_F_m2, dev_explicit_v1.face_capacitance_F_m2)

    dev_v2 = compile_device_2d(
        prepared, spatial, dielectric_rule=LAYER_SERIES_CAPACITANCE_V2
    )
    assert dev_v2.sha256 != dev_default.sha256
    assert dev_v2.dielectric_rule == LAYER_SERIES_CAPACITANCE_V2


def test_p08_02a_1d_nonuniform_two_layer_capacitor_exact_to_1e14() -> None:
    """Non-uniform 1D 2-layer and 1-interval 3-layer capacitors hit <= 1e-14 rel error."""
    l_left, l_right = 10.0e-9, 10.0e-9
    eps_left, eps_right = 3.0, 24.0
    v_applied = 1.15

    # Exact analytical series capacitance, displacement, and potential profile.
    c_exact = float(
        Fraction(1)
        / (
            Fraction.from_float(l_left) / (Fraction.from_float(EPS_0) * Fraction(eps_left))
            + Fraction.from_float(l_right) / (Fraction.from_float(EPS_0) * Fraction(eps_right))
        )
    )
    d_exact = c_exact * v_applied

    def phi_analytic(x_arr: np.ndarray) -> np.ndarray:
        return np.where(
            x_arr <= l_left,
            v_applied - d_exact * x_arr / (EPS_0 * eps_left),
            v_applied
            - d_exact * l_left / (EPS_0 * eps_left)
            - d_exact * (x_arr - l_left) / (EPS_0 * eps_right),
        )

    # 1) Shared-node strongly non-uniform grid: x_2 = X_if = 10 nm with d_L = 4 nm != d_R = 2.5 nm.
    x_shared = np.array([0.0, 6.0e-9, 10.0e-9, 12.5e-9, 20.0e-9])
    eps_shared = np.array([eps_left, eps_left, eps_right, eps_right, eps_right])
    for geom_name in (INTEGER_ENDPOINT_V1, HALF_ENDPOINT_V1):
        geom = build_geometry(geom_name, x_shared, [l_left])
        assert tuple(layer_permittivities_from_geometry(geom, eps_shared)) == (
            eps_left,
            eps_right,
        )
        op_v2 = PoissonOperator1D.for_geometry(
            geom, eps_shared, dielectric_rule=LAYER_SERIES_CAPACITANCE_V2
        )
        rho = np.zeros_like(x_shared)
        phi_v2 = PoissonFactorization.factor(op_v2).solve(rho, v_applied, 0.0)
        disp_v2 = op_v2.face_displacement(phi_v2)
        contacts_v2 = contact_displacement(disp_v2, rho, geom.cell_measures_m)
        c_geom_v2 = contacts_v2[0] / v_applied

        phi_ref = phi_analytic(x_shared)
        assert np.max(np.abs(phi_v2 - phi_ref)) / v_applied <= 1e-14
        assert np.max(np.abs(disp_v2 - d_exact)) / abs(d_exact) <= 1e-14
        assert abs(c_geom_v2 - c_exact) / c_exact <= 1e-14

        # Legacy nodal-harmonic rule exhibits O(h) dielectric shift (> 8% error here).
        op_v1 = PoissonOperator1D.for_geometry(geom, eps_shared)
        phi_v1 = PoissonFactorization.factor(op_v1).solve(rho, v_applied, 0.0)
        c_geom_v1 = op_v1.face_displacement(phi_v1)[0] / v_applied
        assert abs(c_geom_v1 - c_exact) / c_exact > 0.08

    # 2) Two-sided split-face non-uniform grid: interface at 10 nm splits [6 nm, 12.5 nm].
    x_split = np.array([0.0, 6.0e-9, 12.5e-9, 20.0e-9])
    eps_split = np.array([eps_left, eps_left, eps_right, eps_right])
    geom_split = build_geometry(PHYSICAL_BOUNDARY_TWO_SIDED_V1, x_split, [l_left])
    op_split = PoissonOperator1D.from_layer_series_capacitance(geom_split, eps_split)
    rho_split = np.zeros_like(x_split)
    phi_split = PoissonFactorization.factor(op_split).solve(rho_split, v_applied, 0.0)
    disp_split = op_split.face_displacement(phi_split)
    assert np.max(np.abs(phi_split - phi_analytic(x_split))) / v_applied <= 1e-14
    assert np.max(np.abs(disp_split - d_exact)) / abs(d_exact) <= 1e-14

    # 3) Shared-node 3-layer stack with a single-interval thin middle layer [10 nm, 12 nm].
    x_thin = np.array([0.0, 7.0e-9, 10.0e-9, 12.0e-9, 20.0e-9])
    eps_thin = np.array([3.0, 3.0, 40.0, 9.0, 9.0])
    geom_thin = build_geometry(HALF_ENDPOINT_V1, x_thin, [10.0e-9, 12.0e-9])
    caps_thin = layer_series_face_capacitances(geom_thin, eps_thin)
    expected_caps = EPS_0 * np.array([3.0, 3.0, 40.0, 9.0]) / np.diff(x_thin)
    assert np.max(np.abs(caps_thin - expected_caps) / expected_caps) <= 1e-15


def test_p08_02a_2d_nonuniform_two_layer_capacitor_stack_and_lateral_exact_to_1e14() -> None:
    """Non-uniform 2D stack (y) and lateral (x) 2-layer capacitors hit <= 1e-14 rel error."""
    v_applied = 0.95
    eps_1, eps_2 = 3.2, 25.6

    # --- Case 1: 2D stack (y) 2-layer capacitor on stretched non-uniform (x, y) mesh ---
    y_nodes = np.array([0.0, 5.5e-9, 10.0e-9, 12.2e-9, 20.0e-9])
    x_nodes = np.array([0.0, 1.3e-7, 4.1e-7, 1.0e-6])
    eps_y = np.array([eps_1, eps_1, eps_2, eps_2, eps_2])
    y_geom = half_endpoint_geometry(y_nodes, [10.0e-9])

    c_stack_exact = EPS_0 / (10.0e-9 / eps_1 + 10.0e-9 / eps_2)
    d_stack_exact = c_stack_exact * v_applied
    phi_y_exact = np.where(
        y_nodes <= 10.0e-9,
        v_applied - d_stack_exact * y_nodes / (EPS_0 * eps_1),
        v_applied
        - d_stack_exact * 10.0e-9 / (EPS_0 * eps_1)
        - d_stack_exact * (y_nodes - 10.0e-9) / (EPS_0 * eps_2),
    )

    for lateral_bc in ("neumann", "periodic"):
        rect = RectangularGeometry2D(x_nodes, y_geom, lateral_bc=lateral_bc)
        caps = rectangular_2d_face_capacitances(
            rect, eps_y, dielectric_rule=LAYER_SERIES_CAPACITANCE_V2
        )
        bottom_nodes = np.arange(rect.nx)
        top_nodes = np.arange(rect.node_count - rect.nx, rect.node_count)
        dirichlet_nodes = np.r_[bottom_nodes, top_nodes]
        dirichlet_vals = np.r_[np.full(rect.nx, v_applied), np.zeros(rect.nx)]
        phi_2d = _solve_2d_dirichlet_gauss(rect, caps, dirichlet_nodes, dirichlet_vals)
        phi_grid = phi_2d.reshape(rect.shape)

        assert np.max(np.abs(phi_grid - phi_y_exact[:, None])) / v_applied <= 1e-14
        left, right = rect.contract_geometry.pairs.T
        disp_all = -caps * (phi_2d[right] - phi_2d[left])
        disp_x = disp_all[: rect.x_face_count]
        disp_y = disp_all[rect.x_face_count :]
        assert np.max(np.abs(disp_x)) <= 1e-14 * abs(d_stack_exact)
        assert np.max(np.abs(disp_y - d_stack_exact)) / abs(d_stack_exact) <= 1e-14
        q_bottom = float(np.sum(disp_y.reshape(rect.ny - 1, rect.nx)[0] * rect.x_widths_m))
        c_geom_y = q_bottom / (rect.lateral_length_m * v_applied)
        assert abs(c_geom_y - c_stack_exact) / c_stack_exact <= 1e-14

    # --- Case 2: 2D lateral (x) 2-layer capacitor (shared-node and split-face) ---
    y_uniform = half_endpoint_geometry(np.array([0.0, 0.4e-7, 1.1e-7, 2.0e-7]), ())
    x_if = 3.0e-7
    l_x = 1.0e-6
    c_lat_exact = EPS_0 / (x_if / eps_1 + (l_x - x_if) / eps_2)
    d_lat_exact = c_lat_exact * v_applied

    def phi_x_analytic(x_arr: np.ndarray) -> np.ndarray:
        return np.where(
            x_arr <= x_if,
            v_applied - d_lat_exact * x_arr / (EPS_0 * eps_1),
            v_applied
            - d_lat_exact * x_if / (EPS_0 * eps_1)
            - d_lat_exact * (x_arr - x_if) / (EPS_0 * eps_2),
        )

    for x_case, eps_x in (
        # Shared lateral node at x_2 = 0.3 um with d_L = 0.12 um != d_R = 0.07 um
        (
            np.array([0.0, 1.8e-7, 3.0e-7, 3.7e-7, 1.0e-6]),
            np.array([eps_1, eps_1, eps_2, eps_2, eps_2]),
        ),
        # Split lateral face [0.18 um, 0.37 um] cut at X_if = 0.30 um (d_L = 0.12 != d_R = 0.07)
        (
            np.array([0.0, 1.8e-7, 3.7e-7, 1.0e-6]),
            np.array([eps_1, eps_1, eps_2, eps_2]),
        ),
    ):
        rect_lat = RectangularGeometry2D(x_case, y_uniform, lateral_bc="neumann")
        caps_lat = rectangular_2d_face_capacitances(
            rect_lat,
            np.ones(rect_lat.ny),
            dielectric_rule=LAYER_SERIES_CAPACITANCE_V2,
            lateral_eps_r_x=eps_x,
            lateral_interfaces_m=(x_if,),
        )
        ids = np.arange(rect_lat.node_count).reshape(rect_lat.shape)
        left_col = ids[:, 0]
        right_col = ids[:, -1]
        phi_lat = _solve_2d_dirichlet_gauss(
            rect_lat,
            caps_lat,
            np.r_[left_col, right_col],
            np.r_[np.full(rect_lat.ny, v_applied), np.zeros(rect_lat.ny)],
        )
        phi_lat_grid = phi_lat.reshape(rect_lat.shape)
        assert np.max(np.abs(phi_lat_grid - phi_x_analytic(x_case)[None, :])) / v_applied <= 1e-14

        left, right = rect_lat.contract_geometry.pairs.T
        disp_lat = -caps_lat * (phi_lat[right] - phi_lat[left])
        disp_lat_x = disp_lat[: rect_lat.x_face_count].reshape(rect_lat.ny, rect_lat.nx - 1)
        disp_lat_y = disp_lat[rect_lat.x_face_count :]
        assert np.max(np.abs(disp_lat_y)) <= 1e-14 * abs(d_lat_exact)
        assert np.max(np.abs(disp_lat_x - d_lat_exact)) / abs(d_lat_exact) <= 1e-14
        height_y = float(y_uniform.nodes_m[-1] - y_uniform.nodes_m[0])
        q_left = float(np.sum(disp_lat_x[:, 0] * y_uniform.cell_measures_m))
        c_geom_x = q_left / (height_y * v_applied)
        assert abs(c_geom_x - c_lat_exact) / c_lat_exact <= 1e-14


def test_p08_02a_interface_sheet_charge_jump_exact_across_nonuniform_subfaces() -> None:
    """Exact Gauss jump D_R - D_L = sigma_sheet holds across non-uniform sub-faces."""
    l_left, l_right = 12.0e-9, 18.0e-9
    eps_left, eps_right = 3.5, 22.0
    sigma_sheet = -3.75e-4  # C / m^2
    phi_left, phi_right = 0.8, -0.1

    r_left = l_left / (EPS_0 * eps_left)
    r_right = l_right / (EPS_0 * eps_right)
    # phi_L - d_L * r_L = phi_if = phi_R + d_R * r_R, and d_R - d_L = sigma_sheet
    d_left_exact = (phi_left - phi_right - sigma_sheet * r_right) / (r_left + r_right)
    d_right_exact = d_left_exact + sigma_sheet
    phi_if_exact = phi_left - d_left_exact * r_left

    def phi_piecewise(x_arr: np.ndarray) -> np.ndarray:
        return np.where(
            x_arr <= l_left,
            phi_left - d_left_exact * x_arr / (EPS_0 * eps_left),
            phi_if_exact - d_right_exact * (x_arr - l_left) / (EPS_0 * eps_right),
        )

    # 1) Shared-node non-uniform grid: x_2 = l_left = 12 nm, d_L = 4.5 nm != d_R = 2.7 nm.
    x_shared = np.array([0.0, 7.5e-9, 12.0e-9, 14.7e-9, 30.0e-9])
    eps_shared = np.array([eps_left, eps_left, eps_right, eps_right, eps_right])
    geom_shared = build_geometry(HALF_ENDPOINT_V1, x_shared, [l_left])
    op_shared = PoissonOperator1D.from_layer_series_capacitance(geom_shared, eps_shared)

    # Solve C_-(phi_{i-1}-phi_i) + C_+(phi_{i+1}-phi_i) + sigma_i = 0 via equivalent rho_i = sigma_i / w_i.
    sheet_rhs: np.ndarray = np.zeros(geom_shared.node_count - 2, dtype=np.float64)
    add_interface_sheet_charge(sheet_rhs, geom_shared, [sigma_sheet], eps_r_node=eps_shared)
    rho_equiv: np.ndarray = np.zeros(geom_shared.node_count, dtype=np.float64)
    rho_equiv[1:-1] = sheet_rhs / op_shared.interior_widths_m
    phi_sol = PoissonFactorization.factor(op_shared).solve(rho_equiv, phi_left, phi_right)

    res = op_shared.interior_residual(phi_sol, np.zeros_like(x_shared))
    add_interface_sheet_charge(res, geom_shared, [sigma_sheet], eps_r_node=eps_shared)
    assert np.max(np.abs(res)) <= 1e-14 * abs(sigma_sheet)

    disp = op_shared.face_displacement(phi_sol)
    assert abs((disp[2] - disp[1]) - sigma_sheet) / abs(sigma_sheet) <= 1e-14
    assert abs(disp[1] - d_left_exact) / abs(d_left_exact) <= 1e-14
    assert abs(disp[2] - d_right_exact) / abs(d_right_exact) <= 1e-14
    assert np.max(np.abs(phi_sol - phi_piecewise(x_shared))) <= 1e-14

    # 2) Two-sided split-face non-uniform grid: [8.3 nm, 17.2 nm] split at 12.0 nm (d_L = 3.7 != d_R = 5.2).
    x_two = np.array([0.0, 8.3e-9, 17.2e-9, 30.0e-9])
    eps_two = np.array([eps_left, eps_left, eps_right, eps_right])
    geom_two = build_geometry(PHYSICAL_BOUNDARY_TWO_SIDED_V1, x_two, [l_left])
    op_two = PoissonOperator1D.from_layer_series_capacitance(geom_two, eps_two)
    sheet_rhs_two: np.ndarray = np.zeros(geom_two.node_count - 2, dtype=np.float64)
    add_interface_sheet_charge(sheet_rhs_two, geom_two, [sigma_sheet], eps_r_node=eps_two)
    rho_two: np.ndarray = np.zeros(geom_two.node_count, dtype=np.float64)
    rho_two[1:-1] = sheet_rhs_two / op_two.interior_widths_m
    phi_two = PoissonFactorization.factor(op_two).solve(rho_two, phi_left, phi_right)

    plane = geom_two.interfaces[0]
    trace = TraceElectrostatics.from_plane(
        plane, eps_left, eps_right, sheet_charge_C_m2=sigma_sheet
    )
    phi_ls, phi_rs = solve_electrostatic_traces(
        trace, float(phi_two[plane.left_node]), float(phi_two[plane.right_node])
    )
    d_l_trace = EPS_0 * eps_left * (phi_two[plane.left_node] - phi_ls) / plane.left_distance_m
    d_r_trace = EPS_0 * eps_right * (phi_rs - phi_two[plane.right_node]) / plane.right_distance_m
    assert abs(phi_ls - phi_if_exact) <= 1e-14
    assert abs(phi_rs - phi_if_exact) <= 1e-14
    assert abs((d_r_trace - d_l_trace) - sigma_sheet) / abs(sigma_sheet) <= 1e-14
    assert np.max(np.abs(phi_two - phi_piecewise(x_two))) <= 1e-14


def test_p08_02b_physical_half_endpoint_v2_measures_gauss_and_generation_coupling() -> None:
    """PHYSICAL_HALF_ENDPOINT_V2 couples w_i = b_{i+1}-b_i across Gauss and Beer-Lambert."""
    thicknesses = (1.5e-7, 4.0e-7, 1.8e-7)
    total_length = sum(thicknesses)
    interfaces = (thicknesses[0], thicknesses[0] + thicknesses[1])
    x_shared = multilayer_grid(list(zip(thicknesses, (6, 10, 5))), alpha=2.2)
    eps_shared = np.array([3.5, 24.0, 9.0])[node_owner(x_shared, thicknesses, RIGHT_LAYER_WINS)]

    # Historical INTEGER_ENDPOINT_V1 stays bit-identical with endpoint over-count.
    geom_int = build_geometry(INTEGER_ENDPOINT_V1, x_shared, interfaces)
    dx = np.diff(x_shared)
    assert abs(geom_int.total_measure_m - (total_length + 0.5 * (dx[0] + dx[-1]))) <= 1e-21

    # Shared-node PHYSICAL_HALF_ENDPOINT_V2: exact w_i = b_{i+1} - b_i, sum(w) == L.
    geom_v2 = build_geometry(PHYSICAL_HALF_ENDPOINT_V2, x_shared, interfaces)
    assert geom_v2.geometric_cells is True
    assert _same_bits(geom_v2.cell_measures_m, np.diff(geom_v2.cell_faces_m))
    assert abs(geom_v2.total_measure_m - total_length) / total_length <= 1e-15
    assert geom_v2.material_owner_convention == RIGHT_LAYER_WINS
    assert geom_v2.role_owner_convention == RIGHT_LAYER_WINS

    op_v2 = PoissonOperator1D.for_geometry(geom_v2, eps_shared)
    assert op_v2.dielectric_rule == LAYER_SERIES_CAPACITANCE_V2
    assert _same_bits(op_v2.interior_widths_m, geom_v2.cell_measures_m[1:-1])

    rng = np.random.default_rng(20261115)
    phi = rng.normal(scale=0.4, size=x_shared.size)
    rho = rng.normal(scale=2.0e4, size=x_shared.size)
    balance = gauss_balance_with_term_sink(op_v2, geom_v2, phi, rho)
    contacts = contact_displacement(op_v2.face_displacement(phi), rho, geom_v2.cell_measures_m)
    assert _same_bits(balance[1:-1], op_v2.interior_residual(phi, rho))
    assert _same_bits(np.array([balance[0], balance[-1]]), np.array([-contacts[0], contacts[1]]))

    # Uncoupled interior widths must be refused by gauss_balance_with_term_sink.
    bad_op = PoissonOperator1D.from_finite_volume(
        op_v2.face_capacitance_F_m2,
        op_v2.interior_widths_m * 1.01,
        geometry_name=PHYSICAL_HALF_ENDPOINT_V2,
    )
    with pytest.raises(ValueError, match="interior measures"):
        gauss_balance_with_term_sink(bad_op, geom_v2, phi, rho)

    # Beer-Lambert generation coupled to PHYSICAL_HALF_ENDPOINT_V2 matches 2D y_geometry
    # and closes exact absorbed photon budget.
    alpha_uniform = np.full(x_shared.size, 8.0e6)
    incident_flux = 2.5e21
    bl = BeerLambertBinding(x_shared, alpha_uniform, incident_flux)
    prof_v2 = bl.generation_profile(geom_v2)
    prof_int = bl.generation_profile(geom_int)
    prof_2d_y = bl.generation_profile(half_endpoint_geometry(x_shared, interfaces))
    expected_absorbed = incident_flux * (1.0 - np.exp(-alpha_uniform[0] * total_length))
    assert (
        abs(float(np.sum(prof_v2.source_m3_s * geom_v2.cell_measures_m)) - expected_absorbed)
        / expected_absorbed
        <= 1e-14
    )
    # PHYSICAL_HALF_ENDPOINT_V2 fixes the 2x boundary cell dilution of INTEGER_ENDPOINT_V1.
    assert abs(prof_v2.source_m3_s[0] - prof_2d_y.source_m3_s[0]) / prof_v2.source_m3_s[0] <= 1e-14
    assert abs(prof_v2.source_m3_s[0] / prof_int.source_m3_s[0] - 2.0) < 1e-12

    # Two-sided PHYSICAL_HALF_ENDPOINT_V2 places interior faces at the true material planes.
    x_two = remove_shared_interface_nodes(x_shared, interfaces)
    geom_two_v2 = physical_half_endpoint_geometry(x_two, interfaces)
    assert _same_bits(geom_two_v2.cell_measures_m, np.diff(geom_two_v2.cell_faces_m))
    assert abs(geom_two_v2.total_measure_m - total_length) / total_length <= 1e-15
    for plane in geom_two_v2.interfaces:
        assert geom_two_v2.cell_faces_m[plane.right_node] == plane.position_m


def test_p08_02b_areal_sheet_trap_conversion_conserves_across_geometries() -> None:
    """Areal sheet-trap density and capture rates conserve identically on cell_measures_m."""
    thicknesses = (1.2e-7, 3.6e-7, 1.4e-7)
    interfaces = (thicknesses[0], thicknesses[0] + thicknesses[1])
    x_shared = multilayer_grid(list(zip(thicknesses, (5, 8, 5))), alpha=2.0)
    x_two = remove_shared_interface_nodes(x_shared, interfaces)

    trap_eval = evaluate_shared_interface_trap(
        1.2e19,
        4.5e15,
        8.0e16,
        2.3e18,
        v_n_left_m_s=1.5e2,
        v_p_left_m_s=8.0e1,
        v_n_right_m_s=2.0e2,
        v_p_right_m_s=1.1e2,
        n1_left_m3=1.0e15,
        p1_left_m3=2.0e15,
        n1_right_m3=5.0e14,
        p1_right_m3=3.0e15,
        areal_density_m2=2.0e15,
        equilibrium_occupancy=0.25,
    )
    cn_left, cp_left, cn_right, cp_right = (float(v) for v in trap_eval.capture_rates_m2_s)
    total_electron_capture = cn_left + cn_right
    total_hole_capture = cp_left + cp_right
    assert abs(total_electron_capture - total_hole_capture) <= 1e-13 * abs(total_electron_capture)

    for geom in (
        build_geometry(INTEGER_ENDPOINT_V1, x_shared, interfaces),
        build_geometry(HALF_ENDPOINT_V1, x_shared, interfaces),
        build_geometry(PHYSICAL_HALF_ENDPOINT_V2, x_shared, interfaces),
        build_geometry(PHYSICAL_BOUNDARY_TWO_SIDED_V1, x_two, interfaces),
        build_geometry(PHYSICAL_HALF_ENDPOINT_V2, x_two, interfaces),
    ):
        src_nt = areal_interface_to_cell_source(geom, 0, 2.0e15, partition=(0.35, 0.65))
        assert abs(float(np.sum(geom.cell_measures_m * src_nt)) - 2.0e15) / 2.0e15 <= 1e-15

        src_cn = geom.areal_to_cell_source(1, side_quantities=(cn_left, cn_right))
        src_cp = geom.areal_to_cell_source(1, side_quantities=(cp_left, cp_right))
        int_cn = float(np.sum(geom.cell_measures_m * src_cn))
        int_cp = float(np.sum(geom.cell_measures_m * src_cp))
        assert abs(int_cn - total_electron_capture) / abs(total_electron_capture) <= 1e-14
        assert abs(int_cp - total_hole_capture) / abs(total_hole_capture) <= 1e-14


def test_p08_02c_default_legacy_summed_potential_is_0_ulp_bit_identical() -> None:
    """Default band_step_mode preserves 0-ULP bit-identity across M-class and N-class operators."""
    assert BAND_STEP_MODES == (LEGACY_SUMMED_POTENTIAL_V1, HOLE_SEPARATE_INCREMENT_V1)
    vt = 0.02585199102458495
    layers = (
        LayerBandInput(50.0e-9, 4.0, 3.2, 2.2e24, 1.8e25, role="etl"),
        LayerBandInput(350.0e-9, 3.9, 1.55, 2.2e24, 1.8e25, role="absorber"),
        LayerBandInput(150.0e-9, 2.15, 3.45, 2.0e25, 2.0e25, role="htl"),
    )
    x = multilayer_grid([(layer.thickness_m, 6) for layer in layers], alpha=2.0)
    bands = layered_band_structure(x, layers, thermal_voltage_V=vt, fold_dos=True)
    rng = np.random.default_rng(8023)
    phi = np.linspace(0.0, 0.9, x.size) + 1e-4 * rng.normal(size=x.size)
    n = np.exp(rng.uniform(25.0, 52.0, size=x.size))
    p = np.exp(rng.uniform(25.0, 52.0, size=x.size))
    dn = rng.uniform(1e-6, 5e-5, size=x.size - 1)
    dp = rng.uniform(1e-6, 5e-5, size=x.size - 1)

    jn_def, jp_def = legacy_carrier_face_currents(x, phi, n, p, dn, dp, vt, bands)
    jn_leg, jp_leg = legacy_carrier_face_currents(
        x, phi, n, p, dn, dp, vt, bands, band_step_mode=LEGACY_SUMMED_POTENTIAL_V1
    )
    assert _same_bits(jn_def, jn_leg)
    assert _same_bits(jp_def, jp_leg)

    jac_n_def, jac_p_def = legacy_carrier_face_jacobians(x, phi, n, p, dn, dp, vt, bands)
    jac_n_leg, jac_p_leg = legacy_carrier_face_jacobians(
        x, phi, n, p, dn, dp, vt, bands, band_step_mode=LEGACY_SUMMED_POTENTIAL_V1
    )
    assert _same_bits(jac_n_def.flux, jac_n_leg.flux)
    assert _same_bits(jac_p_def.flux, jac_p_leg.flux)
    assert _same_bits(jac_p_def.potential_left_derivative, jac_p_leg.potential_left_derivative)

    fl_def = carrier_particle_fluxes(x, phi, n, p, dn, dp, vt, bands)
    fl_leg = carrier_particle_fluxes(
        x, phi, n, p, dn, dp, vt, bands, band_step_mode=LEGACY_SUMMED_POTENTIAL_V1
    )
    assert _same_bits(fl_def.electron.flux, fl_leg.electron.flux)
    assert _same_bits(fl_def.hole.flux, fl_leg.hole.flux)
    assert _same_bits(fl_def.hole.energy_step, fl_leg.hole.energy_step)
    assert _same_bits(
        fl_def.hole.left_potential_derivative, fl_leg.hole.left_potential_derivative
    )

    # Graph-pair mode matches 1D consecutive-node mode identically on (0..N-2, 1..N-1).
    left_idx = np.arange(x.size - 1, dtype=np.intp)
    right_idx = np.arange(1, x.size, dtype=np.intp)
    for mode in BAND_STEP_MODES:
        dn_1d, dp_1d = carrier_transport_potential_steps(phi, bands, band_step_mode=mode)
        dn_g, dp_g = carrier_transport_potential_steps(
            phi, bands, band_step_mode=mode, left_nodes=left_idx, right_nodes=right_idx
        )
        assert _same_bits(dn_1d, dn_g)
        assert _same_bits(dp_1d, dp_g)

    with pytest.raises(ValueError, match="band_step_mode"):
        carrier_transport_potential_steps(phi, bands, band_step_mode="unknown_v9")


def test_p08_02c_hole_separate_increment_v1_jacobian_and_sub_uv_precision() -> None:
    """HOLE_SEPARATE_INCREMENT_V1 preserves exact d(step)/d(phi) = +/-1/V_T and sub-nV accuracy."""
    vt = 0.02585199102458495
    layers = (
        LayerBandInput(40.0e-9, 4.0, 3.2, 2.2e24, 1.8e25, role="etl"),
        LayerBandInput(160.0e-9, 3.9, 1.55, 2.2e24, 1.8e25, role="absorber"),
        LayerBandInput(80.0e-9, 2.2, 3.4, 2.0e25, 2.0e25, role="htl"),
    )
    x = multilayer_grid([(layer.thickness_m, 3) for layer in layers], alpha=1.8)
    bands = layered_band_structure(x, layers, thermal_voltage_V=vt, fold_dos=True)
    rng = np.random.default_rng(12103)
    phi = np.linspace(-0.15, 0.65, x.size) + 0.015 * rng.normal(size=x.size)
    n = np.exp(rng.uniform(35.0, 48.0, size=x.size))
    p = np.exp(rng.uniform(35.0, 48.0, size=x.size))
    dn = rng.uniform(5e-6, 3e-5, size=x.size - 1)
    dp = rng.uniform(5e-6, 3e-5, size=x.size - 1)

    fluxes = carrier_particle_fluxes(
        x, phi, n, p, dn, dp, vt, bands, band_step_mode=HOLE_SEPARATE_INCREMENT_V1
    )
    jac_n, jac_p = legacy_carrier_face_jacobians(
        x, phi, n, p, dn, dp, vt, bands, band_step_mode=HOLE_SEPARATE_INCREMENT_V1
    )

    h_phi = 1.0e-7
    for k in range(x.size):
        phi_plus = phi.copy()
        phi_minus = phi.copy()
        phi_plus[k] += h_phi
        phi_minus[k] -= h_phi
        fp = carrier_particle_fluxes(
            x, phi_plus, n, p, dn, dp, vt, bands, band_step_mode=HOLE_SEPARATE_INCREMENT_V1
        )
        fm = carrier_particle_fluxes(
            x, phi_minus, n, p, dn, dp, vt, bands, band_step_mode=HOLE_SEPARATE_INCREMENT_V1
        )
        jn_p, jp_p = legacy_carrier_face_currents(
            x, phi_plus, n, p, dn, dp, vt, bands, band_step_mode=HOLE_SEPARATE_INCREMENT_V1
        )
        jn_m, jp_m = legacy_carrier_face_currents(
            x, phi_minus, n, p, dn, dp, vt, bands, band_step_mode=HOLE_SEPARATE_INCREMENT_V1
        )
        dstep_n_fd = (fp.electron.energy_step - fm.electron.energy_step) / (2.0 * h_phi)
        dstep_p_fd = (fp.hole.energy_step - fm.hole.energy_step) / (2.0 * h_phi)
        dfn_fd = (fp.electron.flux - fm.electron.flux) / (2.0 * h_phi)
        dfp_fd = (fp.hole.flux - fm.hole.flux) / (2.0 * h_phi)
        djn_fd = (jn_p - jn_m) / (2.0 * h_phi)
        djp_fd = (jp_p - jp_m) / (2.0 * h_phi)

        if k < x.size - 1:
            # Node k is the LEFT node of face k: d(D_n)/d(phi_L) = +1/V_T, d(D_p)/d(phi_L) = -1/V_T.
            assert abs(dstep_n_fd[k] - (1.0 / vt)) / (1.0 / vt) <= 2e-9
            assert abs(dstep_p_fd[k] - (-1.0 / vt)) / (1.0 / vt) <= 2e-9
            assert abs(dfn_fd[k] - fluxes.electron.left_potential_derivative[k]) <= 1e-8 * max(
                abs(fluxes.electron.left_potential_derivative[k]), 1.0
            )
            assert abs(dfp_fd[k] - fluxes.hole.left_potential_derivative[k]) <= 1e-8 * max(
                abs(fluxes.hole.left_potential_derivative[k]), 1.0
            )
            assert abs(djn_fd[k] - jac_n.potential_left_derivative[k]) <= 1e-8 * max(
                abs(jac_n.potential_left_derivative[k]), 1.0
            )
            assert abs(djp_fd[k] - jac_p.potential_left_derivative[k]) <= 1e-8 * max(
                abs(jac_p.potential_left_derivative[k]), 1.0
            )
        if k > 0:
            # Node k is the RIGHT node of face k - 1: d(D_n)/d(phi_R) = -1/V_T, d(D_p)/d(phi_R) = +1/V_T.
            f = k - 1
            assert abs(dstep_n_fd[f] - (-1.0 / vt)) / (1.0 / vt) <= 2e-9
            assert abs(dstep_p_fd[f] - (1.0 / vt)) / (1.0 / vt) <= 2e-9
            assert abs(dfn_fd[f] - fluxes.electron.right_potential_derivative[f]) <= 1e-8 * max(
                abs(fluxes.electron.right_potential_derivative[f]), 1.0
            )
            assert abs(dfp_fd[f] - fluxes.hole.right_potential_derivative[f]) <= 1e-8 * max(
                abs(fluxes.hole.right_potential_derivative[f]), 1.0
            )
            assert abs(djn_fd[f] - jac_n.potential_right_derivative[f]) <= 1e-8 * max(
                abs(jac_n.potential_right_derivative[f]), 1.0
            )
            assert abs(djp_fd[f] - jac_p.potential_right_derivative[f]) <= 1e-8 * max(
                abs(jac_p.potential_right_derivative[f]), 1.0
            )

    # Sub-fV step inside chi_t + Eg_t = 5.6 eV HTL: N-class stable_sg_particle_flux + BandStepsV1!
    x_htl = np.array([0.0, 1.0e-9])
    bands_htl = BandStructure.without_dos_fold(np.array([2.2, 2.2]), np.array([3.4, 3.4]))
    dphi_tiny = -3.0e-16  # < 0.5 * ulp(5.6 V) = 4.44e-16 V!
    phi_htl = np.array([1.0e-4, 1.0e-4 + dphi_tiny])
    na = 2.0e24
    mu_p = 2.0e-4
    dp_htl = np.array([mu_p * vt])
    jp_exact = -Q * mu_p * na * ((phi_htl[1] - phi_htl[0]) / 1.0e-9)

    fl_leg_tiny = carrier_particle_fluxes(
        x_htl, phi_htl, np.ones(2), np.full(2, na), dp_htl, dp_htl, vt, bands_htl
    )
    fl_sep_tiny = carrier_particle_fluxes(
        x_htl,
        phi_htl,
        np.ones(2),
        np.full(2, na),
        dp_htl,
        dp_htl,
        vt,
        bands_htl,
        band_step_mode=HOLE_SEPARATE_INCREMENT_V1,
    )
    # Legacy quantizes dpsi_p to an integer multiple of ulp(5.6 V) = 8.88e-16 V (>150% error),
    # whereas N-class + HOLE_SEPARATE_INCREMENT_V1 recovers jp_exact to machine precision!
    assert abs(fl_leg_tiny.hole_current()[0] - jp_exact) / abs(jp_exact) > 1.5
    assert abs(fl_sep_tiny.hole_current()[0] - jp_exact) / abs(jp_exact) <= 1e-14


def test_p08_02c_final121_class_quantization_eliminated_and_historical_verdict_preserved() -> None:
    """BandStepsV1 drops Final121-class >1/3000 A/m^2 rounding to <=2e-5 A/m^2 while preserving default."""
    # Historical citation (AGENTS.md & ArchitectureRefactorExecutionV1.md §19.35):
    # Saved SCAPS fixed-state native residual screens recorded 8.79892088904392e-5 A/m^2 on Final61
    # (passed <= 1/3000 A/m^2) and 4.010734989424907e-4 A/m^2 on Final121 (FAILED > 1/3000 A/m^2)
    # under default LEGACY_SUMMED_POTENTIAL_V1. That historical failed criterion remains recorded
    # as failed; default calls MUST keep LEGACY_SUMMED_POTENTIAL_V1 untouched!
    screen_limit_a_m2 = 1.0 / 3000.0
    vt = 0.02585199102458495
    layers = (
        LayerBandInput(50.0e-9, 4.0, 3.2, 2.2e24, 1.8e25, role="etl"),
        LayerBandInput(400.0e-9, 3.9, 1.55, 2.2e24, 1.8e25, role="absorber"),
        LayerBandInput(200.0e-9, 2.15, 3.45, 2.0e25, 2.0e25, role="htl"),
    )
    na = 1.0e24
    mu_p = 2.0e-4
    j0 = 240.0

    def _htl_balances(intervals_per_layer: int) -> tuple[float, float, float, float]:
        x = multilayer_grid([(layer.thickness_m, intervals_per_layer) for layer in layers], alpha=2.2)
        bands = layered_band_structure(x, layers, thermal_voltage_V=vt, fold_dos=True)
        htl_start = 2 * intervals_per_layer
        # Exact steady drift solution in uniform quasi-neutral HTL (B(xi) - B(-xi) === -xi):
        # true continuum and exact-arithmetic discrete SG hole balance is identically 0!
        phi = 0.21 - (j0 / (Q * mu_p * na)) * (x - x[htl_start])
        n = np.full_like(x, 1.0e10)
        p = np.full_like(x, na)
        d_face = np.full(x.size - 1, mu_p * vt)

        # Default (no band_step_mode) MUST preserve historical LEGACY_SUMMED_POTENTIAL_V1 behavior.
        _, jp_default = legacy_carrier_face_currents(x, phi, n, p, d_face, d_face, vt, bands)
        _, jp_sep = legacy_carrier_face_currents(
            x, phi, n, p, d_face, d_face, vt, bands, band_step_mode=HOLE_SEPARATE_INCREMENT_V1
        )
        fl_default = carrier_particle_fluxes(x, phi, n, p, d_face, d_face, vt, bands)
        fl_sep = carrier_particle_fluxes(
            x, phi, n, p, d_face, d_face, vt, bands, band_step_mode=HOLE_SEPARATE_INCREMENT_V1
        )
        m_def = float(np.sum(np.abs(jp_default[htl_start:-1] - jp_default[htl_start + 1 :])))
        m_sep = float(np.sum(np.abs(jp_sep[htl_start:-1] - jp_sep[htl_start + 1 :])))
        n_def = float(
            np.sum(
                np.abs(
                    fl_default.hole_current()[htl_start:-1]
                    - fl_default.hole_current()[htl_start + 1 :]
                )
            )
        )
        n_sep = float(
            np.sum(
                np.abs(
                    fl_sep.hole_current()[htl_start:-1] - fl_sep.hole_current()[htl_start + 1 :]
                )
            )
        )
        return m_def, m_sep, n_def, n_sep

    m61_def, m61_sep, n61_def, n61_sep = _htl_balances(20)  # N = 61
    m121_def, m121_sep, n121_def, n121_sep = _htl_balances(40)  # N = 121

    # N = 61 passes <= 1/3000 A/m^2 under default legacy summed potentials, while N = 121 FAILS it:
    assert m61_def < screen_limit_a_m2 and n61_def < screen_limit_a_m2
    assert m121_def > screen_limit_a_m2 and n121_def > screen_limit_a_m2

    # Opt-in HOLE_SEPARATE_INCREMENT_V1 eliminates the 5.6 eV offset quantization (> 25x reduction):
    assert m61_sep <= 2.0e-5 and n61_sep <= 2.0e-5
    assert m121_sep <= 2.0e-5 and n121_sep <= 2.0e-5
    assert m121_def / m121_sep > 25.0
    assert n121_def / n121_sep > 25.0


def test_p08_02c_internal_coordinate_orientation_invariants() -> None:
    """Internal +x particle/current/displacement operators obey strict coordinate signs."""
    vt = 0.02585199102458495
    x = np.array([0.0, 2.0e-8, 5.0e-8])
    bands = BandStructure.without_dos_fold(np.array([3.9, 3.9, 3.9]), np.array([1.55, 1.55, 1.55]))
    phi_old = np.array([0.4, 0.15, 0.0])
    phi_new = np.array([0.5, 0.20, 0.0])
    n = np.array([5.0e21, 1.0e21, 2.0e20])
    p = np.array([8.0e21, 2.0e21, 3.0e20])
    d_face = np.array([1.0e-5, 1.0e-5])

    for mode in BAND_STEP_MODES:
        fl = carrier_particle_fluxes(
            x, phi_old, n, p, d_face, d_face, vt, bands, band_step_mode=mode
        )
        jn_m, jp_m = legacy_carrier_face_currents(
            x, phi_old, n, p, d_face, d_face, vt, bands, band_step_mode=mode
        )
        # Positive +x hole drift+diffusion => F_p > 0 and J_p = +q F_p > 0.
        assert np.all(fl.hole.flux > 0.0)
        assert _same_bits(fl.hole_current(), hole_current_from_particle_flux(fl.hole.flux))
        assert np.all(fl.hole_current() > 0.0)
        # Electron charge -q => J_n = -q F_n has opposite sign to F_n!
        assert _same_bits(
            fl.electron_current(), electron_current_from_particle_flux(fl.electron.flux)
        )
        assert np.all(np.sign(fl.electron_current()) == -np.sign(fl.electron.flux))
        np.testing.assert_allclose(fl.electron_current(), jn_m, rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(fl.hole_current(), jp_m, rtol=1e-13, atol=0.0)

    # Displacement D = -C (phi_R - phi_L) > 0 when phi decreases along +x; J_disp = (D_new - D_old)/dt.
    geom = physical_half_endpoint_geometry(x, ())
    op = PoissonOperator1D.for_geometry(geom, np.full(3, 10.0), layer_eps_r=(10.0,))
    d_old = op.face_displacement(phi_old)
    d_new = op.face_displacement(phi_new)
    assert np.all(d_old > 0.0) and np.all(d_new > d_old)
    j_disp = (d_new - d_old) / 1.0e-6
    assert np.all(j_disp > 0.0)

