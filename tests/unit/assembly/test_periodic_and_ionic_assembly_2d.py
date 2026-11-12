"""Unit and sparse-assembly contracts for P06-01 (periodic 2D) and P06-02 (2D mobile-ion & extensions)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_multilayer import prepare_confined_dual_ion_device, prepare_confined_ion_device
from examples.research_1d.steady import load_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_2d import compile_device_2d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.compile_ionic_2d import (
    build_two_sided_interface_srh_2d,
    compile_ionic_device_2d,
)
from solarlab.device.inputs import DeviceInput, GrainBoundaryInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.assembly.ionic_2d import (
    AssembledIonic2DProblem,
    AssembledIonicDC2DProblem,
    AssembledIonicTransient2DProblem,
    AssembledStorage2D,
)
from solarlab.equations.assembly.two_dimensional import AssembledDC2DProblem
from solarlab.equations.dc_1d import CapabilityError, DC1DProblem
from solarlab.equations.dc_2d import DC2DProblem
from solarlab.equations.ionic_2d import (
    Ionic2DProblem,
    finite_site_secant_2d_m3,
)
from solarlab.experiments.two_dimensional.inputs import JV2DInput, MicrostructureInput
from solarlab.experiments.two_dimensional.jv_sweep_2d import SpatialSnapshot2D
from solarlab.mesh.geometry import half_endpoint_geometry
from solarlab.mesh.geometry_2d import (
    RECTANGULAR_HALF_ENDPOINT_2D_V1,
    RECTANGULAR_PERIODIC_2D_V1,
    RectangularGeometry2D,
)
from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions, solve_ionic_steady_state
from solarlab.solvers.nonlinear.ionic_2d import IonicSteady2DOptions, solve_ionic_steady_state_2d
from solarlab.solvers.nonlinear.problem import SolveBudget
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state
from solarlab.solvers.nonlinear.steady_2d import solve_steady_state_2d
from solarlab.solvers.time.backward_euler_2d import compute_carrier_only_terminal_current_2d


@pytest.fixture(scope="module", autouse=True)
def single_thread_blas() -> Any:
    with threadpool_limits(limits=1):
        yield


def _edit_device(prepared: PreparedDevice, mutate: Any) -> PreparedDevice:
    data = prepared.to_input().editing_data()
    mutate(data)
    return resolve_device(
        DeviceInput.model_validate(data),
        prepared.defaults,
        prepared.resources,
        sources=prepared.sources,
    )


def test_periodic_and_neumann_geometry_measures_wrap_faces_and_single_axis_decoupling() -> None:
    x = np.array([0.0, 0.15, 0.45, 1.0]) * 1e-6
    y = np.array([0.0, 0.1, 0.3, 0.6, 1.0]) * 1e-6
    y_geom = half_endpoint_geometry(y)

    g_neu = RectangularGeometry2D(x, y_geom, lateral_bc="neumann")
    g_per = RectangularGeometry2D(x, y_geom, lateral_bc="periodic")

    assert g_neu.name == RECTANGULAR_HALF_ENDPOINT_2D_V1
    assert g_per.name == RECTANGULAR_PERIODIC_2D_V1
    assert g_per.shape == (5, 4)
    assert g_per.x_faces_per_row == 4
    assert g_neu.x_faces_per_row == 3
    assert g_per.x_face_count == 5 * 4
    assert g_neu.x_face_count == 5 * 3
    assert g_per.wrap_face_indices.shape == (5,)
    assert g_neu.wrap_face_indices.size == 0

    # Domain total area equals lateral_length_m * stack_length_m under both lateral BCs
    dx = np.diff(x)
    wrap_spacing = 0.5 * (dx[0] + dx[-1])
    expected_area_neu = (x[-1] - x[0]) * (y[-1] - y[0])
    expected_area_per = ((x[-1] - x[0]) + wrap_spacing) * (y[-1] - y[0])
    np.testing.assert_allclose(g_neu.contract_geometry.volumes.sum(), expected_area_neu, rtol=1e-15)
    np.testing.assert_allclose(g_per.contract_geometry.volumes.sum(), expected_area_per, rtol=1e-15)
    np.testing.assert_allclose(g_per.lateral_length_m, (x[-1] - x[0]) + wrap_spacing, rtol=1e-15)

    # Periodic wrap faces connect (row, Nx-1) -> (row, 0) with spacing 0.5 * (dx[0] + dx[-1])
    for row, f_idx in enumerate(g_per.wrap_face_indices):
        left, right = g_per.contract_geometry.pairs[f_idx]
        assert left == row * g_per.nx + (g_per.nx - 1)
        assert right == row * g_per.nx
        np.testing.assert_allclose(g_per.face_spacing_m[f_idx], wrap_spacing, rtol=1e-15)

    # Opposite-face scatter conservation: sum of divergence over closed x-periodic ring is 0
    rng = np.random.default_rng(42)
    face_flux = rng.standard_normal(g_per.face_spacing_m.size)
    weighted = face_flux * g_per.contract_geometry.face_measures
    div = np.zeros(g_per.node_count)
    np.add.at(div, g_per.contract_geometry.pairs[:, 0], weighted)
    np.add.at(div, g_per.contract_geometry.pairs[:, 1], -weighted)
    assert abs(math.fsum(div.tolist())) <= 1e-14 * np.max(np.abs(weighted))

    # Single-axis perturbations decouple orthogonal faces
    phi_x_only = np.tile(np.sin(2.0 * np.pi * x / (x[-1] - x[0])), (y.size, 1)).ravel()
    grad_x_only = (phi_x_only[g_per.contract_geometry.pairs[:, 1]] - phi_x_only[g_per.contract_geometry.pairs[:, 0]])
    np.testing.assert_allclose(grad_x_only[g_per.x_face_count:], 0.0, atol=1e-16)

    phi_y_only = np.repeat(np.cos(np.pi * y / (y[-1] - y[0])), x.size)
    grad_y_only = (phi_y_only[g_per.contract_geometry.pairs[:, 1]] - phi_y_only[g_per.contract_geometry.pairs[:, 0]])
    np.testing.assert_allclose(grad_y_only[:g_per.x_face_count], 0.0, atol=1e-16)


@pytest.mark.parametrize("lateral_bc", ["periodic", "neumann"])
def test_mms_2d_poisson_second_order_convergence(lateral_bc: str) -> None:
    """Manufactured-solution 2D Poisson verification on periodic-x and Neumann-x operators."""
    lx, ly = 1.2e-6, 0.8e-6
    errors: list[float] = []
    for n_int in (8, 16, 32):
        x = np.linspace(0.0, lx, n_int + 1)
        y = np.linspace(0.0, ly, n_int + 1)
        g = RectangularGeometry2D(x, half_endpoint_geometry(y), lateral_bc=lateral_bc)  # type: ignore[arg-type]
        cg = g.contract_geometry
        xx, yy = np.meshgrid(x, y)
        if lateral_bc == "periodic":
            # Exact power-of-two ring of n_int nodes on period lx -> spacing dx = lx / n_int
            dx = lx / n_int
            x_ring = np.arange(n_int, dtype=float) * dx
            g = RectangularGeometry2D(x_ring, half_endpoint_geometry(y), lateral_bc="periodic")
            cg = g.contract_geometry
            xx, yy = np.meshgrid(x_ring, y)
            kx = 2.0 * np.pi / lx
            phi_exact = (np.sin(kx * xx) * np.sin(np.pi * yy / ly)).ravel()
            rhs_vol = ((kx**2 + (np.pi / ly) ** 2) * np.sin(kx * xx) * np.sin(np.pi * yy / ly)).ravel()
        else:
            kx = np.pi / lx
            phi_exact = (np.cos(kx * xx) * np.sin(np.pi * yy / ly)).ravel()
            rhs_vol = ((kx**2 + (np.pi / ly) ** 2) * np.cos(kx * xx) * np.sin(np.pi * yy / ly)).ravel()

        # Assemble finite-volume Laplacian L_ij: sum_{f=(i,j)} (w_f / d_f) * (phi_i - phi_j) = V_i * rhs_i
        n_nodes = g.node_count
        left, right = cg.pairs[:, 0], cg.pairs[:, 1]
        cond = cg.face_measures / g.face_spacing_m
        rows = np.concatenate([left, right, left, right])
        cols = np.concatenate([left, right, right, left])
        vals = np.concatenate([cond, cond, -cond, -cond])
        mat = csc_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)).tolil()
        b = cg.volumes * rhs_vol
        # Dirichlet y=0 and y=Ly contact rows
        for c_node in g.contact_nodes:
            mat.rows[c_node] = [int(c_node)]
            mat.data[c_node] = [1.0]
            b[c_node] = phi_exact[c_node]
        phi_num = spsolve(mat.tocsc(), b)
        l2_err = math.sqrt(float(np.sum(cg.volumes * (phi_num - phi_exact) ** 2) / np.sum(cg.volumes)))
        errors.append(l2_err)

    rate_1 = math.log2(errors[0] / errors[1])
    rate_2 = math.log2(errors[1] / errors[2])
    assert rate_1 > 1.95 and rate_2 > 1.97, f"expected O(h^2) convergence, got rates {(rate_1, rate_2)}"


@pytest.mark.parametrize("lateral_bc", ["neumann", "periodic"])
def test_laterally_uniform_periodic_and_neumann_reduce_to_1d_steady_within_1e12(lateral_bc: str) -> None:
    prepared = load_device()
    spatial = JV2DInput(kind="jv_2d", lateral_length="0.8 um", Nx=4, Ny_per_layer=4, lateral_bc=lateral_bc)  # type: ignore[arg-type]
    dev_2d = compile_device_2d(prepared, spatial)
    dev_1d = compile_device_1d(prepared, intervals_per_layer=(4, 4, 4))

    s_1d = solve_steady_state(dev_1d, voltage_V=0.45, generation_scale=1.0)
    assert s_1d.state is not None
    seed_2d = dev_2d.state(dev_2d.extrude_coordinates(s_1d.state.u), voltage_V=0.45)

    # Check both direct DC2DProblem and AssembledDC2DProblem residual & terminal current parity
    p1 = DC1DProblem(dev_1d, 0.45, 1.0)
    p2_raw = DC2DProblem(dev_2d, 0.45, 1.0)
    p2_asm = AssembledDC2DProblem(dev_2d, 0.45, 1.0)
    np.testing.assert_allclose(p2_asm.residual(seed_2d.u), p2_raw.residual(seed_2d.u), rtol=1e-13, atol=1e-22)
    assert isinstance(p2_asm.jacobian(seed_2d.u), csc_matrix)
    np.testing.assert_allclose(p2_asm.jacobian(seed_2d.u).toarray(), p2_raw.jacobian(seed_2d.u).toarray(), rtol=1e-13, atol=1e-22)

    s_2d = solve_steady_state_2d(dev_2d, voltage_V=0.45, generation_scale=1.0, initial_state=seed_2d)
    assert s_2d.accepted and s_2d.state is not None and s_2d.diagnostics is not None
    j_1d = float(p1.contact_current_density(s_1d.state.u)[1])
    j_2d = float(s_2d.diagnostics["terminal_mean_current_density_A_m2"][1])
    assert abs(j_2d - j_1d) <= 1e-12 * max(abs(j_1d), 1.0)


def test_assembled_ionic_2d_sparse_jacobian_mass_matrix_fd_and_1d_parity() -> None:
    prepared = prepare_confined_ion_device(diffusion_only=True)
    # Enable finite B_rad on absorber so radiative_reabsorption is active when requested
    prepared_rad = _edit_device(
        prepared,
        lambda d: next(lyr for lyr in d["layers"] if lyr["role"] == "absorber")["parameters"].update(
            B_rad=2.0e-16
        ),
    )
    spatial = JV2DInput(
        kind="jv_2d",
        lateral_length="0.6 um",
        Nx=3,
        Ny_per_layer=4,
        lateral_bc="neumann",
        ion_dynamics="single_mobile",
    )
    ion_2d = compile_ionic_device_2d(
        prepared_rad,
        spatial,
        ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"),
        enable_radiative_reabsorption=True,
    )
    assert len(ion_2d.radiative_reabsorption) == 1

    # Compare 1D ionic steady vs laterally uniform 2D ionic steady (with reabsorption off for exact 1D match)
    ion_2d_plain = compile_ionic_device_2d(
        prepared,
        spatial,
        ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"),
        enable_radiative_reabsorption=False,
    )
    ion_1d = compile_ionic_device_1d(
        prepared,
        intervals_per_layer=(4, 4, 4),
        ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"),
    )
    opts_1d = IonicSteadyOptions(
        SteadyOptions(), 1e-12, 1e6, 1e6, 1.0, ion_1d.ion_node_count - 1,
        SolveBudget(60, 600, 60, 24), 30.0,
    )
    res_1d = solve_ionic_steady_state(ion_1d, voltage_V=0.2, generation_scale=0.0, options=opts_1d)
    assert res_1d.accepted and res_1d.state is not None
    res_2d = solve_ionic_steady_state_2d(
        ion_2d_plain, voltage_V=0.2, generation_scale=0.0, options=IonicSteady2DOptions()
    )
    assert res_2d.accepted and res_2d.state is not None
    np.testing.assert_allclose(
        res_2d.state.positive_ion_density_m3[:, 0],
        res_1d.state.positive_ion_density_m3,
        rtol=1e-12,
        atol=1e10,
    )
    np.testing.assert_allclose(
        res_2d.state.positive_ion_density_m3[:, -1],
        res_1d.state.positive_ion_density_m3,
        rtol=1e-12,
        atol=1e10,
    )

    # Perturb state laterally and verify preallocated sparse csc_matrix, M = -S_u, 4-side ion conservation, and FD
    s0 = res_2d.state
    rng = np.random.default_rng(123)
    u = s0.u.copy()
    n_elec = ion_2d.node_count
    interior = np.setdiff1d(np.arange(n_elec), ion_2d.electronic_component.geometry.contact_nodes)
    u[interior] += 0.015 * rng.standard_normal(interior.size)
    u[n_elec + interior] += 0.1 * rng.standard_normal(interior.size)
    u[2 * n_elec + interior] += 0.1 * rng.standard_normal(interior.size)
    u[3 * n_elec:] += 0.08 * rng.standard_normal(ion_2d.ion_node_count)

    raw_prob = Ionic2DProblem(ion_2d, voltage_V=0.25, generation_scale=1.0)
    dc_asm = AssembledIonicDC2DProblem(ion_2d, voltage_V=0.25, generation_scale=1.0)
    tr_asm = AssembledIonicTransient2DProblem(ion_2d, voltage_V=0.25, generation_scale=1.0)
    assert isinstance(tr_asm, AssembledIonic2DProblem)

    f_raw, j_raw = raw_prob.residual(u), raw_prob.jacobian(u)
    f_dc, j_dc = dc_asm.residual(u), dc_asm.jacobian(u)
    f_tr, j_tr = tr_asm.residual(u), tr_asm.jacobian(u)
    assert issparse(j_dc) and isinstance(j_dc, csc_matrix)
    assert issparse(j_tr) and isinstance(j_tr, csc_matrix)
    assert j_tr.nnz < 0.15 * (ion_2d.state_size ** 2)

    np.testing.assert_allclose(f_dc, f_raw, rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(f_tr, f_raw, rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(j_tr.toarray(), j_raw.toarray(), rtol=1e-13, atol=1e-10)

    # Four-side blocking ion conservation: sum of all ion residual rows is zero
    ion_rows = f_tr[3 * n_elec:]
    assert abs(math.fsum(ion_rows.tolist())) <= 1e-12 * np.max(np.abs(ion_rows))

    # M = -S_u convention on shared Assembler
    s_vec, s_u = tr_asm.binding.assembler.storage(u, tr_asm.inputs, derivatives=True)
    assert s_u is not None
    mass = tr_asm.mass_matrix(u)
    assert isinstance(mass, csc_matrix)
    np.testing.assert_allclose(mass.toarray(), (-s_u).toarray(), rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(mass.toarray(), raw_prob.storage_jacobian(u).toarray(), rtol=1e-14, atol=0.0)

    # Directional central finite-difference verification of full coupled Jacobian (with radiative reabsorption!)
    direction = np.zeros_like(u)
    direction[interior] = 1e-3 * rng.standard_normal(interior.size)
    direction[n_elec + interior] = 1e-2 * rng.standard_normal(interior.size)
    direction[2 * n_elec + interior] = 1e-2 * rng.standard_normal(interior.size)
    direction[3 * n_elec:] = 1e-2 * rng.standard_normal(ion_2d.ion_node_count)
    eps = 1.0e-6
    fd_jv = (tr_asm.residual(u + eps * direction) - tr_asm.residual(u - eps * direction)) / (2.0 * eps)
    exact_jv = j_tr @ direction
    for sl in (
        slice(0, n_elec),
        slice(n_elec, 2 * n_elec),
        slice(2 * n_elec, 3 * n_elec),
        slice(3 * n_elec, None),
    ):
        assert float(np.linalg.norm(fd_jv[sl] - exact_jv[sl]) / np.linalg.norm(exact_jv[sl])) < 1.0e-7


def test_two_sided_interface_srh_assembly_jacobian_and_zero_overhead_when_off() -> None:
    base = load_device()

    def _enable_two_sided_srh(data: dict[str, Any]) -> None:
        defect_spec = {
            "id": "if_trap",
            "trap_depth_eV": 0.5,
            "energy_reference": "below_conduction_band",
            "total_density_m2": 1.0e14,
            "kinetics": {
                "sigma_n_m2": 1.0e-19,
                "sigma_p_m2": 1.0e-19,
                "thermal_velocity_n_m_s": 1.0e5,
                "thermal_velocity_p_m_s": 1.0e5,
            },
            "calibration_factor": 1.0,
            "iface_state_calibration_factor": 1.0,
        }
        data["settings"]["interface_two_sided"] = True
        data["interfaces"] = [
            {"id": "if_htl_abs", "left": "htl", "right": "absorber", "v_n": 150.0, "v_p": 200.0, "defect": defect_spec},
            {"id": "if_abs_etl", "left": "absorber", "right": "etl", "v_n": 150.0, "v_p": 200.0, "defect": defect_spec},
        ]

    prep_if = _edit_device(base, _enable_two_sided_srh)
    prep_clean = _edit_device(base, lambda _d: None)
    spatial = JV2DInput(kind="jv_2d", lateral_length="0.6 um", Nx=3, Ny_per_layer=3, lateral_bc="neumann")
    dev = compile_device_2d(prep_clean, spatial)

    couplings = build_two_sided_interface_srh_2d(prep_if, dev, interface_srh_mode="two_sided_cross_node")
    assert len(couplings) == 2

    st_off = AssembledStorage2D(dev, 0.7, 1.0)
    st_on = AssembledStorage2D(dev, 0.7, 1.0, interface_srh_couplings=couplings)
    dc_plain = AssembledDC2DProblem(dev, 0.7, 1.0)

    s0 = solve_steady_state_2d(dev, voltage_V=0.7, generation_scale=1.0).state
    assert s0 is not None
    rng = np.random.default_rng(7)
    u_test = s0.u.copy()
    interior = np.setdiff1d(np.arange(dev.node_count), dev.geometry.contact_nodes)
    u_test[interior] += 0.005 * rng.standard_normal(interior.size)
    u_test[dev.node_count + interior] += 0.05 * rng.standard_normal(interior.size)
    u_test[2 * dev.node_count + interior] += 0.05 * rng.standard_normal(interior.size)

    # Exact bitwise identity with AssembledDC2DProblem when optional extensions are off
    np.testing.assert_array_equal(st_off.residual(u_test), dc_plain.residual(u_test))
    np.testing.assert_array_equal(st_off.jacobian(u_test).toarray(), dc_plain.jacobian(u_test).toarray())

    # Active cross-node interface SRH alters residual & matches central FD
    assert np.max(np.abs(st_on.residual(s0.u) - st_off.residual(s0.u))) > 0.0
    direction = np.zeros_like(s0.u)
    direction[interior] = 1e-3 * rng.standard_normal(interior.size)
    direction[dev.node_count + interior] = 1e-2 * rng.standard_normal(interior.size)
    direction[2 * dev.node_count + interior] = 1e-2 * rng.standard_normal(interior.size)
    eps = 1.0e-6
    fd = (st_on.residual(s0.u + eps * direction) - st_on.residual(s0.u - eps * direction)) / (2.0 * eps)
    ana = st_on.jacobian(s0.u) @ direction
    n_el = dev.node_count
    for sl in (slice(0, n_el), slice(n_el, 2 * n_el), slice(2 * n_el, 3 * n_el)):
        assert float(np.linalg.norm(fd[sl] - ana[sl]) / np.linalg.norm(ana[sl])) < 1.0e-7


def test_finite_site_secant_2d_is_cancellation_free_at_tiny_increments() -> None:
    cap = np.array([1.0e27, 2.5e26, 5.0e26])
    a0 = np.array([-6.9, -2.5, 0.4])
    y_old = np.array([0.3, -1.1, 2.0])
    dy_req = np.array([1.0e-13, -2.5e-13, 5.0e-14])
    y_new = y_old + dy_req
    dy_exact = y_new - y_old
    secant = finite_site_secant_2d_m3(cap, a0, y_old, y_new)
    from scipy.special import expit
    sig = expit(a0 + y_old)
    expected = cap * sig * (1.0 - sig) * dy_exact
    np.testing.assert_allclose(secant, expected, rtol=1.0e-11, atol=0.0)


def test_gb_geometric_coverage_mixes_rates_rather_than_averaging_lifetimes() -> None:
    prepared = load_device()
    abs_id = next(lyr.id for lyr in prepared.layers if lyr.role == "absorber")
    # Choose x grid [0, 0.5, 1.0] um -> cell 1 spans [0.25, 0.75] um (width 0.5 um).
    # GB centered at 0.5 um with width 0.25 um covers f = 0.25 / 0.5 = 0.5 of cell 1!
    gb = GrainBoundaryInput.model_validate(
        {
            "id": "gb_half",
            "layer_ids": [abs_id],
            "x_position": "0.5 um",
            "width": "0.25 um",
            "tau_n": "1 ns",
            "tau_p": "1 ns",
        }
    )
    sp = JV2DInput(
        kind="jv_2d",
        lateral_length="1.0 um",
        Nx=2,
        Ny_per_layer=4,
        lateral_bc="neumann",
        microstructure=MicrostructureInput(grain_boundaries=(gb,)),
    )
    dev_gb = compile_device_2d(prepared, sp)
    dev_bulk = compile_device_2d(
        prepared,
        JV2DInput(kind="jv_2d", lateral_length="1.0 um", Nx=2, Ny_per_layer=4, lateral_bc="neumann"),
    )
    assert dev_gb.grain_recombination is not None
    # Pick an interior absorber row and col=1 (center column, f=0.5)
    abs_row = 6
    node_center = abs_row * dev_gb.geometry.nx + 1
    frac = float(dev_gb.grain_recombination.regions[0].fractions[node_center])
    assert math.isclose(frac, 0.5, rel_tol=1e-12)

    n = np.full(dev_gb.node_count, 1.0e21)
    p = np.full(dev_gb.node_count, 2.0e21)
    p_bulk = DC2DProblem(dev_bulk, 0.0, 0.0)
    p_gb = DC2DProblem(dev_gb, 0.0, 0.0)
    r_bulk = float(p_bulk.recombination_rate(n, p, derivatives=False).rate[node_center])
    r_actual = float(p_gb.recombination_rate(n, p, derivatives=False).rate[node_center])

    # Full-coverage GB (width = 1.0 um -> f = 1.0) gives pure R_gb
    gb_full = GrainBoundaryInput.model_validate(
        {
            "id": "gb_full",
            "layer_ids": [abs_id],
            "x_position": "0.5 um",
            "width": "1.0 um",
            "tau_n": "1 ns",
            "tau_p": "1 ns",
        }
    )
    dev_full = compile_device_2d(
        prepared,
        JV2DInput(
            kind="jv_2d",
            lateral_length="1.0 um",
            Nx=2,
            Ny_per_layer=4,
            lateral_bc="neumann",
            microstructure=MicrostructureInput(grain_boundaries=(gb_full,)),
        ),
    )
    r_pure_gb = float(DC2DProblem(dev_full, 0.0, 0.0).recombination_rate(n, p, derivatives=False).rate[node_center])

    expected_rate_mix = r_bulk + 0.5 * (r_pure_gb - r_bulk)
    np.testing.assert_allclose(r_actual, expected_rate_mix, rtol=1e-14)
    # Averaging tau_eff = 0.5 * tau_bulk + 0.5 * tau_gb would give ~2x tau_bulk -> drastically smaller rate!
    tau_b = float(dev_bulk.column.recombination.coefficients.tau_n_s[abs_row])
    ni_sq = float(dev_bulk.column.recombination.coefficients.ni_sq_m6[abs_row])
    n1 = float(dev_bulk.column.recombination.coefficients.n1_m3[abs_row])
    p1 = float(dev_bulk.column.recombination.coefficients.p1_m3[abs_row])
    tau_avg = 0.5 * tau_b + 0.5 * 1.0e-9
    r_wrong_tau_avg = (n[node_center] * p[node_center] - ni_sq) / (tau_avg * (n[node_center] + n1 + p[node_center] + p1))
    assert abs(r_actual - r_wrong_tau_avg) > 0.4 * abs(r_actual)


def test_fail_closed_2d_capability_error_refusal_matrix() -> None:
    base = load_device()
    abs_id = next(lyr.id for lyr in base.layers if lyr.role == "absorber")
    gb = GrainBoundaryInput.model_validate(
        {
            "id": "gb0",
            "layer_ids": [abs_id],
            "x_position": "0.3 um",
            "width": "20 nm",
            "tau_n": "1 ns",
            "tau_p": "1 ns",
        }
    )

    # 1. Periodic + grain boundaries -> CapabilityError
    sp_per_gb = JV2DInput(
        kind="jv_2d",
        lateral_length="0.6 um",
        Nx=3,
        Ny_per_layer=3,
        lateral_bc="periodic",
        microstructure=MicrostructureInput(grain_boundaries=(gb,)),
    )
    with pytest.raises(CapabilityError, match="Neumann"):
        compile_device_2d(base, sp_per_gb)

    # 2. Periodic + mobile-ion -> CapabilityError
    ion_prep = prepare_confined_ion_device()
    sp_per_ion = JV2DInput(
        kind="jv_2d", lateral_length="0.6 um", Nx=3, Ny_per_layer=3, lateral_bc="periodic"
    )
    with pytest.raises(CapabilityError, match="periodic|Neumann"):
        compile_ionic_device_2d(
            ion_prep, sp_per_ion, ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking")
        )

    # 3. Dual-ion 2D -> CapabilityError
    dual_prep = prepare_confined_dual_ion_device()
    sp_neu = JV2DInput(kind="jv_2d", lateral_length="0.6 um", Nx=3, Ny_per_layer=3, lateral_bc="neumann")
    with pytest.raises(CapabilityError, match="dual_ion_2d_unsupported"):
        compile_ionic_device_2d(
            dual_prep, sp_neu, ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking")
        )

    # 4. Shared-site 2D -> CapabilityError
    shared_prep = _edit_device(ion_prep, lambda d: d["settings"].update(ion_steric_shared_site=True))
    with pytest.raises(CapabilityError, match="dual_ion_2d_unsupported"):
        compile_ionic_device_2d(
            shared_prep, sp_neu, ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking")
        )

    # 5. Selective Robin contacts + mobile-ion 2D -> CapabilityError
    robin_ion_prep = _edit_device(
        ion_prep,
        lambda d: d.update(
            contacts=[{"id": "front", "side": "left", "layer": "htl", "S_n": 1e5, "S_p": 1e-2}]
        ),
    )
    with pytest.raises(CapabilityError, match="selective_robin_ionic_2d_unsupported"):
        compile_ionic_device_2d(
            robin_ion_prep, sp_neu, ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking")
        )

    # 6. Dynamic interface flags -> CapabilityError
    for updates in (
        {"interface_plane_projection": True},
        {"interface_shared_occupancy": True},
        {"interface_charge_closure": "equilibrium_referenced", "interface_charge_rebaseline_acknowledged": True},
    ):
        dyn_prep = _edit_device(ion_prep, lambda d, u=updates: d["settings"].update(u))
        with pytest.raises(CapabilityError, match="dynamic_interface_2d_unsupported"):
            compile_ionic_device_2d(
                dyn_prep, sp_neu, ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking")
            )

    # 7. Active electrical interface SRH silently ignored when interface_srh="off" -> CapabilityError
    active_if_prep = _edit_device(
        ion_prep,
        lambda d: d.update(
            interfaces=[
                {"id": "if_htl_abs", "left": "htl", "right": "absorber", "v_n": 100.0, "v_p": 100.0},
                {"id": "if_abs_etl", "left": "absorber", "right": "etl"},
            ]
        ),
    )
    with pytest.raises(CapabilityError, match="interface_srh_2d_not_enabled|interface recombination is not bound"):
        compile_ionic_device_2d(
            active_if_prep, sp_neu, ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking")
        )

    # 8. Zero-diffusivity D_ion == 0 with ion_dynamics="single_mobile" -> CapabilityError
    zero_d_prep = _edit_device(
        ion_prep,
        lambda d: next(lyr for lyr in d["layers"] if lyr["role"] == "absorber")["parameters"].update(D_ion=0.0),
    )
    sp_single = JV2DInput(
        kind="jv_2d",
        lateral_length="0.6 um",
        Nx=3,
        Ny_per_layer=3,
        lateral_bc="neumann",
        ion_dynamics="single_mobile",
    )
    with pytest.raises(CapabilityError, match="mobile_ion_2d_requires_positive_diffusivity"):
        compile_ionic_device_2d(
            zero_d_prep, sp_single, ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking")
        )

    # 9. Carrier-only terminal current helper refuses IonicState2D and P_ion-bearing snapshots
    ion_dev = compile_ionic_device_2d(
        ion_prep, sp_single, ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking")
    )
    elec_u0 = ion_dev.electronic_component.extrude_coordinates(
        np.r_[
            ion_dev.electronic_component.column.reference_potential_V,
            np.zeros(2 * ion_dev.electronic_component.geometry.ny),
        ]
    )
    ion_state = ion_dev.state(np.r_[elec_u0, np.zeros(ion_dev.ion_node_count)], voltage_V=0.0)
    with pytest.raises(CapabilityError, match="carrier_only_current_refuses_mobile_ion_2d"):
        compute_carrier_only_terminal_current_2d(ion_dev.electronic_component, ion_state)
    elec_state = ion_dev.electronic_component.state(ion_state.u[:3 * ion_dev.node_count], voltage_V=0.0)
    fake_snap = SpatialSnapshot2D(
        V_app=0.0,
        x=ion_dev.electronic_component.geometry.x_m,
        y=ion_dev.electronic_component.geometry.y_geometry.nodes_m,
        phi=elec_state.potential_V,
        n=elec_state.electron_density_m3,
        p=elec_state.hole_density_m3,
        Jx_n=np.zeros((10, 3)),
        Jy_n=np.zeros((9, 4)),
        Jx_p=np.zeros((10, 3)),
        Jy_p=np.zeros((9, 4)),
        R_total=np.zeros((10, 4)),
        P_ion=np.ones((10, 4)),
    )
    with pytest.raises(CapabilityError, match="carrier_only_current_refuses_mobile_ion_2d"):
        compute_carrier_only_terminal_current_2d(ion_dev.electronic_component, fake_snap)
