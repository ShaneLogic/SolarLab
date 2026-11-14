"""Spatial refinement and 1D/2D parity integration tests for P08-02."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from examples.research_1d.steady import load_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_2d import compile_device_2d
from solarlab.discretization import (
    LAYER_SERIES_CAPACITANCE_V2,
    NODAL_HARMONIC_DIELECTRIC_V1,
    VACUUM_PERMITTIVITY_F_M,
)
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
from solarlab.equations.dc_2d import DC2DProblem
from solarlab.experiments.jv.steady import run_steady_jv
from solarlab.experiments.transient import (
    TransientOptions,
    TransientProtocol,
    TransientResult1D,
    solve_transient,
)
from solarlab.experiments.two_dimensional.inputs import JV2DInput
from solarlab.mesh import (
    INTEGER_ENDPOINT_V1,
    PHYSICAL_HALF_ENDPOINT_V2,
    RIGHT_LAYER_WINS,
    build_geometry,
    multilayer_grid,
    node_owner,
    physical_half_endpoint_geometry,
)
from solarlab.mesh.geometry import half_endpoint_geometry
from solarlab.mesh.geometry_2d import RectangularGeometry2D
from solarlab.physics.electrostatics import (
    PoissonFactorization,
    PoissonOperator1D,
    contact_displacement,
    rectangular_2d_face_capacitances,
)
from solarlab.physics.generation.beer_lambert import BeerLambertBinding
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state
from solarlab.solvers.nonlinear.steady_2d import Steady2DOptions, solve_steady_state_2d
from solarlab.solvers.time.backward_euler import inventory_m2

EPS_0 = VACUUM_PERMITTIVITY_F_M


def _mms_1d_heterojunction_error(scale: int, dielectric_rule: str) -> float:
    """1D 2-layer space-charge-bearing heterojunction MMS on shared-node grids."""
    l1, l2 = 1.8e-7, 1.2e-7
    eps1, eps2 = 4.0, 24.0
    length = l1 + l2
    d0, d1 = 1.2e-3, 6.0e-4

    def displacement_exact(x_arr: np.ndarray) -> np.ndarray:
        return d0 + d1 * np.sin(np.pi * x_arr / length)

    def phi_exact(x_arr: np.ndarray) -> np.ndarray:
        def antideriv(a: float | np.ndarray, b: float | np.ndarray, eps: float) -> np.ndarray:
            return np.asarray(
                (
                    d0 * (b - a)
                    - d1 * length / np.pi * (np.cos(np.pi * b / length) - np.cos(np.pi * a / length))
                )
                / (EPS_0 * eps),
                dtype=float,
            )

        return 0.35 - np.where(
            x_arr <= l1,
            antideriv(0.0, np.minimum(x_arr, l1), eps1),
            antideriv(0.0, l1, eps1) + antideriv(l1, x_arr, eps2),
        )

    x = multilayer_grid([(l1, 3 * scale), (l2, 2 * scale)], alpha=1.2)
    geom = build_geometry(INTEGER_ENDPOINT_V1, x, [l1])
    eps = np.array([eps1, eps2])[node_owner(x, (l1, l2), RIGHT_LAYER_WINS)]
    op = PoissonOperator1D.for_geometry(geom, eps, dielectric_rule=dielectric_rule)
    faces = geom.cell_faces_m
    cell_charge = displacement_exact(faces[1:]) - displacement_exact(faces[:-1])
    rho = np.zeros(x.size, dtype=float)
    rho[1:-1] = cell_charge[1:-1] / op.interior_widths_m
    phi_num = PoissonFactorization.factor(op).solve(
        rho, float(phi_exact(np.array([x[0]]))[0]), float(phi_exact(np.array([x[-1]]))[0])
    )
    return float(np.max(np.abs(phi_num - phi_exact(x))))


def _mms_2d_heterojunction_error(scale: int, dielectric_rule: str) -> float:
    """2D (x, y) laterally modulated space-charge heterojunction MMS on shared-node grids.

    Combines a vertical heterojunction potential (continuous D_y across y = l1)
    with a lateral mode A * cos(pi * x / lx) that has nonzero tangential
    curvature d^2 phi / dx^2 right on the shared interface row y_k = l1, so
    second-order convergence requires BOTH vertical sub-face series capacitance
    AND horizontal cell-parallel permittivity at y_k = l1.
    """
    lx = 4.0e-7
    l1, l2 = 1.5e-7, 1.0e-7
    ly = l1 + l2
    eps1, eps2 = 3.5, 24.5
    d0, d1 = 9.0e-4, 4.5e-4
    amp_x = 0.08
    kx = np.pi / lx

    def disp_y_1d(y_arr: np.ndarray) -> np.ndarray:
        return d0 + d1 * np.sin(np.pi * y_arr / ly)

    def phi_y_1d(y_arr: np.ndarray) -> np.ndarray:
        def antideriv(a: float | np.ndarray, b: float | np.ndarray, eps: float) -> np.ndarray:
            return np.asarray(
                (
                    d0 * (b - a)
                    - d1 * ly / np.pi * (np.cos(np.pi * b / ly) - np.cos(np.pi * a / ly))
                )
                / (EPS_0 * eps),
                dtype=float,
            )

        return 0.25 - np.where(
            y_arr <= l1,
            antideriv(0.0, np.minimum(y_arr, l1), eps1),
            antideriv(0.0, l1, eps1) + antideriv(l1, y_arr, eps2),
        )

    x = np.linspace(0.0, lx, 3 * scale + 1)
    y = multilayer_grid([(l1, 3 * scale), (l2, 2 * scale)], alpha=0.8)
    y_geom = half_endpoint_geometry(y, [l1])
    rect = RectangularGeometry2D(x, y_geom, lateral_bc="neumann")
    eps_y = np.array([eps1, eps2])[node_owner(y, (l1, l2), RIGHT_LAYER_WINS)]
    caps = rectangular_2d_face_capacitances(rect, eps_y, dielectric_rule=dielectric_rule)

    # Exact integral of -div(eps_0 eps(y) grad phi) over each 2D control volume [x_a, x_b] x [y_a, y_b].
    x_faces = np.r_[x[0], 0.5 * (x[:-1] + x[1:]), x[-1]]
    y_faces = y_geom.cell_faces_m
    int_cos_x = (np.sin(kx * x_faces[1:]) - np.sin(kx * x_faces[:-1])) / kx
    wx = np.diff(x_faces)
    int_dy_part = disp_y_1d(y_faces[1:]) - disp_y_1d(y_faces[:-1])

    eps_int_y: np.ndarray = np.empty(rect.ny, dtype=np.float64)
    for j in range(rect.ny):
        ya, yb = float(y_faces[j]), float(y_faces[j + 1])
        part1 = max(0.0, min(yb, l1) - max(ya, 0.0)) * eps1
        part2 = max(0.0, min(yb, ly) - max(ya, l1)) * eps2
        eps_int_y[j] = EPS_0 * (part1 + part2)

    cell_charge_2d = (
        int_dy_part[:, None] * wx[None, :]
        + (kx**2 * amp_x) * eps_int_y[:, None] * int_cos_x[None, :]
    ).ravel()

    n = rect.node_count
    cg = rect.contract_geometry
    left, right = cg.pairs.T
    mat = lil_matrix((n, n), dtype=float)
    for a, b, cap in zip(left, right, caps * cg.face_measures):
        mat[a, a] -= cap
        mat[a, b] += cap
        mat[b, a] += cap
        mat[b, b] -= cap
    rhs = -cell_charge_2d

    phi_exact_2d = phi_y_1d(y)[:, None] + amp_x * np.cos(kx * x)[None, :]
    phi_exact_flat = phi_exact_2d.ravel()
    for idx in rect.contact_nodes:
        node = int(idx)
        mat.rows[node] = [node]
        mat.data[node] = [1.0]
        rhs[node] = phi_exact_flat[node]

    phi_num = np.asarray(spsolve(mat.tocsc(), rhs), dtype=float)
    return float(np.max(np.abs(phi_num - phi_exact_flat)))


def test_p08_02a_1d_and_2d_heterojunction_spatial_refinement_restores_second_order() -> None:
    """LAYER_SERIES_CAPACITANCE_V2 restores O(h^2) in 1D and 2D heterojunctions."""
    scales_1d = (4, 8, 16, 32)
    err_1d_v1 = np.array(
        [_mms_1d_heterojunction_error(s, NODAL_HARMONIC_DIELECTRIC_V1) for s in scales_1d]
    )
    err_1d_v2 = np.array(
        [_mms_1d_heterojunction_error(s, LAYER_SERIES_CAPACITANCE_V2) for s in scales_1d]
    )
    order_1d_v1 = np.log2(err_1d_v1[:-1] / err_1d_v1[1:])
    order_1d_v2 = np.log2(err_1d_v2[:-1] / err_1d_v2[1:])

    assert np.all(order_1d_v1 < 1.2), order_1d_v1
    assert np.all(order_1d_v2 > 1.95), order_1d_v2
    assert err_1d_v2[-1] < 0.03 * err_1d_v1[-1]

    scales_2d = (2, 4, 8, 16)
    err_2d_v1 = np.array(
        [_mms_2d_heterojunction_error(s, NODAL_HARMONIC_DIELECTRIC_V1) for s in scales_2d]
    )
    err_2d_v2 = np.array(
        [_mms_2d_heterojunction_error(s, LAYER_SERIES_CAPACITANCE_V2) for s in scales_2d]
    )
    order_2d_v1 = np.log2(err_2d_v1[:-1] / err_2d_v1[1:])
    order_2d_v2 = np.log2(err_2d_v2[:-1] / err_2d_v2[1:])

    assert np.all(order_2d_v1 < 1.25), order_2d_v1
    assert np.all(order_2d_v2 > 1.95), order_2d_v2
    assert err_2d_v2[-1] < 0.05 * err_2d_v1[-1]


def test_p08_02a_compile_device_2d_heterojunction_refinement_improves_interface_accuracy() -> None:
    """Compiled 2D p-i-n heterojunction solves converge faster under layer_series_capacitance_v2."""
    prepared = load_device()
    options = Steady2DOptions()

    def solve_at(ny_per_layer: int, rule: str) -> tuple[np.ndarray, float]:
        spatial = JV2DInput(
            kind="jv_2d",
            lateral_length="0.4 um",
            Nx=2,
            Ny_per_layer=ny_per_layer,
            lateral_bc="neumann",
        )
        compiled = compile_device_2d(prepared, spatial, dielectric_rule=rule)
        res = solve_steady_state_2d(compiled, voltage_V=0.4, generation_scale=0.0, options=options)
        assert res.accepted and res.state is not None
        phi_col = res.state.potential_V[:, 0]
        # Integrated electrostatic energy proxy sum(C_f * dphi^2 * face_measure) per unit width.
        left, right = compiled.geometry.contract_geometry.pairs.T
        dphi = res.state.u[right] - res.state.u[left]
        energy = float(
            0.5
            * np.sum(
                compiled.face_capacitance_F_m2
                * compiled.geometry.contract_geometry.face_measures
                * dphi**2
            )
            / compiled.geometry.lateral_length_m
        )
        return phi_col, energy

    _, energy_fine = solve_at(24, LAYER_SERIES_CAPACITANCE_V2)
    _, energy_coarse_v1 = solve_at(4, NODAL_HARMONIC_DIELECTRIC_V1)
    _, energy_mid_v1 = solve_at(8, NODAL_HARMONIC_DIELECTRIC_V1)
    _, energy_coarse_v2 = solve_at(4, LAYER_SERIES_CAPACITANCE_V2)
    _, energy_mid_v2 = solve_at(8, LAYER_SERIES_CAPACITANCE_V2)

    err_coarse_v1 = abs(energy_coarse_v1 - energy_fine)
    err_mid_v1 = abs(energy_mid_v1 - energy_fine)
    err_coarse_v2 = abs(energy_coarse_v2 - energy_fine)
    err_mid_v2 = abs(energy_mid_v2 - energy_fine)

    assert err_coarse_v2 < err_coarse_v1
    assert err_mid_v2 < err_mid_v1
    assert err_mid_v2 < 0.6 * err_coarse_v2


def test_p08_02b_global_conservation_and_1d_2d_parity_on_physical_half_endpoint_v2() -> None:
    """PHYSICAL_HALF_ENDPOINT_V2 closes conservation <= 1e-13 and unifies 1D and 2D columns."""
    prepared = load_device()
    ny_per_layer = 4
    intervals = (ny_per_layer,) * len(prepared.layers)
    dev_int_1d = compile_device_1d(prepared, intervals_per_layer=intervals)

    # Rebind the 1D compiled column onto PHYSICAL_HALF_ENDPOINT_V2 + coupled Poisson + generation.
    iface_positions = [plane.position_m for plane in dev_int_1d.geometry.interfaces]
    geom_v2 = physical_half_endpoint_geometry(dev_int_1d.nodes_m, iface_positions)
    data = prepared.to_mapping()
    alpha_node: np.ndarray = np.empty(dev_int_1d.node_count, dtype=np.float64)
    eps_node: np.ndarray = np.empty(dev_int_1d.node_count, dtype=np.float64)
    layer_eps: list[float] = []
    for layer, mask in zip(prepared.layers, dev_int_1d.parameters.layer_masks):
        params = dict(layer.parameters)
        alpha_raw = params["alpha"]
        eps_raw = params["eps_r"]
        assert isinstance(alpha_raw, (int, float)) and not isinstance(alpha_raw, bool)
        assert isinstance(eps_raw, (int, float)) and not isinstance(eps_raw, bool)
        alpha_node[mask] = float(alpha_raw)
        eps_node[mask] = float(eps_raw)
        layer_eps.append(float(eps_raw))

    phi_inc = data["settings"]["Phi"]
    assert isinstance(phi_inc, (int, float)) and not isinstance(phi_inc, bool)
    poisson_v2 = PoissonOperator1D.for_geometry(geom_v2, eps_node, layer_eps_r=layer_eps)
    gen_v2 = BeerLambertBinding(dev_int_1d.nodes_m, alpha_node, float(phi_inc)).generation_profile(
        geom_v2
    )
    sha_v2 = hashlib.sha256(
        json.dumps(
            (dev_int_1d.sha256, PHYSICAL_HALF_ENDPOINT_V2, LAYER_SERIES_CAPACITANCE_V2),
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    dev_v2_1d = replace(
        dev_int_1d,
        sha256=sha_v2,
        geometry=geom_v2,
        poisson=poisson_v2,
        generation=gen_v2,
    )

    opts_1d = SteadyOptions(continuity_tolerance_A_m2=1e-9)
    opts_2d = Steady2DOptions(electronic=opts_1d)

    # Solve 1D (both legacy INTEGER_ENDPOINT_V1 and new PHYSICAL_HALF_ENDPOINT_V2) at two operating points.
    s1_v2_0 = solve_steady_state(dev_v2_1d, voltage_V=0.0, generation_scale=1.0, options=opts_1d)
    s1_v2_f = solve_steady_state(
        dev_v2_1d,
        voltage_V=0.08,
        generation_scale=1.0,
        initial_state=s1_v2_0.state,
        options=opts_1d,
    )
    s1_int_0 = solve_steady_state(dev_int_1d, voltage_V=0.0, generation_scale=1.0, options=opts_1d)
    s1_int_f = solve_steady_state(
        dev_int_1d,
        voltage_V=0.08,
        generation_scale=1.0,
        initial_state=s1_int_0.state,
        options=opts_1d,
    )

    # 1) BackwardEuler1DStep + step_accounting via solve_transient on AssembledDC1DProblem:
    #    global particle and Maxwell total-current conservation <= 1e-13.
    dt_s = 5.0e-6
    protocol = TransientProtocol(
        times_s=np.array([0.0, dt_s, 2.0 * dt_s]),
        voltage_V=np.array([0.0, 0.02, 0.04]),
        generation_scale=np.array([1.0, 1.0, 1.0]),
    )
    tr_options = TransientOptions(
        poisson_tolerance=opts_1d.poisson_tolerance,
        continuity_tolerance_A_m2=opts_1d.continuity_tolerance_A_m2,
        contact_log_tolerance=opts_1d.contact_log_tolerance,
        potential_step_V=opts_1d.potential_step_V,
        quasi_fermi_step=opts_1d.quasi_fermi_step,
        newton_budget=opts_1d.budget(),
        max_steps=4,
        wall_s=20.0,
        predictor="previous",
        convergence_policy=opts_1d.convergence_policy,
    )
    tr_result = solve_transient(
        dev_v2_1d,
        s1_v2_0,
        protocol,
        options=tr_options,
        initial_options=opts_1d,
        method="backward_euler",
    )
    assert isinstance(tr_result, TransientResult1D)
    assert len(tr_result.attempts) == 2
    for attempt in tr_result.attempts:
        assert attempt.accepted and attempt.accounting is not None
        acct = attempt.accounting
        particles = np.asarray(acct["particle_fluxes_m2_s"], dtype=np.float64)
        rate = np.asarray(acct["inventory_rate_m2_s"], dtype=np.float64)
        pair_src = np.asarray(acct["net_pair_source_m2_s"], dtype=np.float64)
        global_part = np.asarray(acct["global_particle_balance_m2_s"], dtype=np.float64)
        global_charge = float(acct["global_charge_balance_A_m2"])  # type: ignore[arg-type]
        maxwell_err = np.asarray(acct["maxwell_identity_error_A_m2"], dtype=np.float64)
        maxwell_spread = float(acct["maxwell_spread_A_m2"])  # type: ignore[arg-type]
        term_total = np.asarray(acct["terminal_total_current_density_A_m2"], dtype=np.float64)

        part_scale = max(
            float(np.max(np.abs(particles))),
            float(np.max(np.abs(rate))),
            float(np.max(np.abs(pair_src))),
            1.0,
        )
        j_scale = max(float(np.max(np.abs(term_total))), 1.0)
        assert np.max(np.abs(global_part)) / part_scale <= 1e-13
        assert abs(global_charge) / j_scale <= 1e-13
        assert np.max(np.abs(maxwell_err)) / j_scale <= 1e-13
        assert maxwell_spread / j_scale <= 1e-13

    # 2) 1D (PHYSICAL_HALF_ENDPOINT_V2) vs uniform-column 2D (Nx = 2 and Nx = 3):
    #    exact J(V), carrier inventory, and contact displacement-current agreement.
    prob_1d_0 = AssembledDC1DProblem(dev_v2_1d, voltage_V=0.0, generation_scale=1.0)
    prob_1d_f = AssembledDC1DProblem(dev_v2_1d, voltage_V=0.08, generation_scale=1.0)
    j1_v2_0 = prob_1d_0.contact_current_density(s1_v2_0.state.u)
    j1_v2_f = prob_1d_f.contact_current_density(s1_v2_f.state.u)
    j1_int_0 = AssembledDC1DProblem(
        dev_int_1d, voltage_V=0.0, generation_scale=1.0
    ).contact_current_density(s1_int_0.state.u)

    inv1_0 = inventory_m2(dev_v2_1d, s1_v2_0.state)
    rho1_0 = dev_v2_1d.charge_ledger(
        s1_v2_0.state.electron_density_m3, s1_v2_0.state.hole_density_m3
    ).space_charge_C_m3()
    rho1_f = dev_v2_1d.charge_ledger(
        s1_v2_f.state.electron_density_m3, s1_v2_f.state.hole_density_m3
    ).space_charge_C_m3()
    d1_contacts_0 = contact_displacement(
        dev_v2_1d.poisson.face_displacement(s1_v2_0.state.potential_V),
        rho1_0,
        dev_v2_1d.geometry.cell_measures_m,
    )
    d1_contacts_f = contact_displacement(
        dev_v2_1d.poisson.face_displacement(s1_v2_f.state.potential_V),
        rho1_f,
        dev_v2_1d.geometry.cell_measures_m,
    )
    j_disp_1d = (d1_contacts_f - d1_contacts_0) / dt_s

    for nx in (2, 3):
        spatial = JV2DInput(
            kind="jv_2d",
            lateral_length="0.6 um",
            Nx=nx,
            Ny_per_layer=ny_per_layer,
            lateral_bc="neumann",
        )
        dev_2d = compile_device_2d(
            prepared, spatial, stack_geometry=PHYSICAL_HALF_ENDPOINT_V2
        )
        # Seed 2D from the legacy INTEGER_ENDPOINT_V1 1D column PLUS a non-uniform lateral
        # cos/sin perturbation so 2D Newton genuinely relaxes both laterally and geometrically!
        def _perturbed_seed(col_u: np.ndarray, bias_v: float) -> np.ndarray:
            y_norm = (dev_2d.column.nodes_m - dev_2d.column.nodes_m[0]) / np.ptp(dev_2d.column.nodes_m)
            x_norm = (dev_2d.geometry.x_m - dev_2d.geometry.x_m[0]) / dev_2d.geometry.lateral_length_m
            wave = np.sin(np.pi * y_norm)
            wave[[0, -1]] = 0.0
            u_init = dev_2d.extrude_coordinates(col_u)
            n_tot = dev_2d.node_count
            u_init[:n_tot] += (0.0007 * wave[:, None] * np.cos(np.pi * x_norm)[None, :]).ravel()
            u_init[n_tot : 2 * n_tot] += (0.013 * wave[:, None] * np.sin(np.pi * x_norm)[None, :]).ravel()
            u_init[2 * n_tot :] += (-0.011 * wave[:, None] * np.cos(2.0 * np.pi * x_norm)[None, :]).ravel()
            return u_init

        res_2d_0 = solve_steady_state_2d(
            dev_2d,
            voltage_V=0.0,
            generation_scale=1.0,
            initial_state=dev_2d.state(_perturbed_seed(s1_int_0.state.u, 0.0), voltage_V=0.0),
            options=opts_2d,
        )
        res_2d_f = solve_steady_state_2d(
            dev_2d,
            voltage_V=0.08,
            generation_scale=1.0,
            initial_state=dev_2d.state(_perturbed_seed(s1_int_f.state.u, 0.08), voltage_V=0.08),
            options=opts_2d,
        )
        assert res_2d_0.accepted and res_2d_0.state is not None and res_2d_0.diagnostics is not None
        assert res_2d_f.accepted and res_2d_f.state is not None and res_2d_f.diagnostics is not None

        j2_0 = np.asarray(res_2d_0.diagnostics["terminal_mean_current_density_A_m2"], dtype=np.float64)
        j2_f = np.asarray(res_2d_f.diagnostics["terminal_mean_current_density_A_m2"], dtype=np.float64)
        assert np.max(np.abs(j2_0 - j1_v2_0)) / np.max(np.abs(j1_v2_0)) <= 1e-13
        assert np.max(np.abs(j2_f - j1_v2_f)) / np.max(np.abs(j1_v2_f)) <= 1e-13

        # Legacy INTEGER_ENDPOINT_V1 disagrees with uniform-column 2D on the coarse grid!
        inv1_int_0 = inventory_m2(dev_int_1d, s1_int_0.state)
        assert np.max(np.abs(j2_0 - j1_int_0)) / np.max(np.abs(j2_0)) > 5e-7

        vols_2d = dev_2d.geometry.contract_geometry.volumes
        lx = dev_2d.geometry.lateral_length_m
        inv2_n = float(np.sum(res_2d_0.state.electron_density_m3.ravel() * vols_2d) / lx)
        inv2_p = float(np.sum(res_2d_0.state.hole_density_m3.ravel() * vols_2d) / lx)
        assert abs(inv2_n - float(np.sum(inv1_0[0]))) / inv2_n <= 1e-13
        assert abs(inv2_p - float(np.sum(inv1_0[1]))) / inv2_p <= 1e-13
        assert abs(inv2_n - float(np.sum(inv1_int_0[0]))) / inv2_n > 0.01
        assert abs(inv2_p - float(np.sum(inv1_int_0[1]))) / inv2_p > 0.01

        # Contact displacement and displacement current parity between 1D and 2D columns.
        g2 = dev_2d.geometry
        disp2_0 = np.asarray(res_2d_0.diagnostics["face_displacement_C_m2"], dtype=np.float64)
        disp2_f = np.asarray(res_2d_f.diagnostics["face_displacement_C_m2"], dtype=np.float64)
        dy_0 = disp2_0[g2.x_face_count :].reshape(g2.ny - 1, g2.nx)
        dy_f = disp2_f[g2.x_face_count :].reshape(g2.ny - 1, g2.nx)
        rho2_0 = dev_2d.charge_ledger(
            res_2d_0.state.electron_density_m3.ravel(), res_2d_0.state.hole_density_m3.ravel()
        ).space_charge_C_m3().reshape(g2.shape)
        rho2_f = dev_2d.charge_ledger(
            res_2d_f.state.electron_density_m3.ravel(), res_2d_f.state.hole_density_m3.ravel()
        ).space_charge_C_m3().reshape(g2.shape)
        wy = g2.y_geometry.cell_measures_m
        d2_contacts_0 = np.array(
            [
                float(np.sum((dy_0[0] - rho2_0[0] * wy[0]) * g2.x_widths_m) / lx),
                float(np.sum((dy_0[-1] + rho2_0[-1] * wy[-1]) * g2.x_widths_m) / lx),
            ]
        )
        d2_contacts_f = np.array(
            [
                float(np.sum((dy_f[0] - rho2_f[0] * wy[0]) * g2.x_widths_m) / lx),
                float(np.sum((dy_f[-1] + rho2_f[-1] * wy[-1]) * g2.x_widths_m) / lx),
            ]
        )
        j_disp_2d = (d2_contacts_f - d2_contacts_0) / dt_s
        d_scale = float(np.max(np.abs(disp2_0)))
        jd_scale = max(float(np.max(np.abs(j_disp_1d))), float(np.max(np.abs(disp2_f - disp2_0)) / dt_s))
        assert np.max(np.abs(d2_contacts_0 - d1_contacts_0)) / d_scale <= 1e-13
        assert np.max(np.abs(j_disp_2d - j_disp_1d)) / jd_scale <= 1e-13


def test_p08_02c_internal_coordinate_signs_vs_pv_readout_polarity_1d_and_2d() -> None:
    """Internal +x/+y coordinate signs never flip; PV power-quadrant sign lives only in readout."""
    prepared = load_device()
    dev_1d = compile_device_1d(prepared, intervals_per_layer=(4, 4, 4))
    jv = run_steady_jv(dev_1d, [0.0, 0.08], generation_scale=1.0)

    # On p-i-n (p at x=0, n at x=L), voltage_polarity == +1.0; under illumination at V=0,
    # electrons flow toward +x (F_n > 0 => J_n < 0) and holes toward -x (F_p < 0 => J_p < 0),
    # so internal +x conduction current is strictly NEGATIVE while PV output readout is POSITIVE.
    assert dev_1d.voltage_polarity == 1.0
    assert np.all(jv.current_density_A_m2[0] < 0.0)
    assert jv.output_current_A_m2[0] > 0.0
    assert jv.output_current_A_m2[0] == -dev_1d.voltage_polarity * jv.current_density_A_m2[0, 0]
    assert jv.metrics.short_circuit_current_A_m2 == jv.output_current_A_m2[0]

    # 2D uniform column preserves exact same +y coordinate current sign on faces and terminals.
    spatial = JV2DInput(
        kind="jv_2d", lateral_length="0.6 um", Nx=2, Ny_per_layer=4, lateral_bc="neumann"
    )
    dev_2d = compile_device_2d(prepared, spatial)
    prob_2d = DC2DProblem(dev_2d, 0.0, 1.0)
    u2 = dev_2d.extrude_coordinates(jv.solutions[0].state.u)
    j_term_2d = prob_2d.terminal_current_density_A_m2(u2)
    fluxes_2d = prob_2d.fluxes(u2)
    yf = dev_2d.geometry.x_face_count
    j_cond_y = fluxes_2d.hole_current()[yf:] + fluxes_2d.electron_current()[yf:]
    assert np.all(j_term_2d < 0.0)
    assert np.all(j_cond_y < 0.0)
    np.testing.assert_allclose(j_term_2d[0], jv.current_density_A_m2[0, 0], rtol=2e-4)

