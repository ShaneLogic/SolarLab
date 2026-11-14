"""Spatial refinement and 1D/2D parity integration tests for P08-02."""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from examples.research_1d.steady import load_device
from solarlab.device.compile_2d import compile_device_2d
from solarlab.discretization import (
    LAYER_SERIES_CAPACITANCE_V2,
    NODAL_HARMONIC_DIELECTRIC_V1,
    VACUUM_PERMITTIVITY_F_M,
)
from solarlab.experiments.two_dimensional.inputs import JV2DInput
from solarlab.mesh import (
    INTEGER_ENDPOINT_V1,
    RIGHT_LAYER_WINS,
    build_geometry,
    multilayer_grid,
    node_owner,
)
from solarlab.mesh.geometry import half_endpoint_geometry
from solarlab.mesh.geometry_2d import RectangularGeometry2D
from solarlab.physics.electrostatics import (
    PoissonFactorization,
    PoissonOperator1D,
    rectangular_2d_face_capacitances,
)
from solarlab.solvers.nonlinear.steady_2d import Steady2DOptions, solve_steady_state_2d

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
