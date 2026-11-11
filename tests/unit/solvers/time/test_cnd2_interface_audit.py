"""Interface-trust and unified-assembler heterointerface conservation audit tests.

Exercises the owned ``solarlab.solvers.time`` BDF1/BDF2 kernels and unified
``Assembler`` storage adapters (``AssembledStorage1D``, ``AssembledIonicStorage1D``)
across multilayer heterointerfaces, plus the Backward Euler public default guard
from ``docs/DevelopmentGuide.md``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from examples.research_1d.ionic_residual import prepare_single_ion_device
from examples.research_1d.ionic_steady import controls as steady_ionic_controls
from examples.research_1d.ionic_transient import controls as time_ionic_controls
from examples.research_1d.multiple_ionic_domains import (
    DOMAINS,
    INTERVALS,
    prepare_multiple_ionic_device,
)
from examples.research_1d.steady import load_device
from examples.research_1d.transient import controls, json_value
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.compile_multiple_ionic_1d import compile_multiple_ionic_device_1d
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
from solarlab.equations.assembly.ionic_1d import AssembledMultipleIonic1DProblem
from solarlab.experiments.ionic_transient import (
    IonicTransientResult,
    load_ionic_steady_initial,
    solve_ionic_transient,
)
from solarlab.experiments.transient import (
    TransientProtocol,
    TransientResult1D,
    solve_transient,
)
from solarlab.solvers.nonlinear.ionic_1d import solve_ionic_steady_state
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state
from solarlab.solvers.time.assembled_ionic_bdf import (
    assembled_ionic_bdf_accounting,
    assembled_ionic_bdf_seed,
)
from solarlab.solvers.time.assembled_storage import (
    AssembledIonicStorage1D,
    AssembledStorage1D,
)
from solarlab.solvers.time.bdf import BDFOptions, BDFResult, BDFStep, HistoryPoint
from solarlab.solvers.time.bdf_1d import ElectronicStorage1D, bdf_accounting


def test_public_transient_entry_points_keep_backward_euler_default(
    tmp_path: Path,
) -> None:
    """Backward Euler remains the public default; BDF requires explicit opt-in."""
    electronic_sig = inspect.signature(solve_transient)
    ionic_sig = inspect.signature(solve_ionic_transient)
    assert electronic_sig.parameters["method"].default == "backward_euler"
    assert electronic_sig.parameters["bdf_options"].default is None
    assert ionic_sig.parameters["method"].default == "backward_euler"
    assert ionic_sig.parameters["bdf_options"].default is None

    device = compile_device_1d(load_device(), intervals_per_layer=8)
    steady_opts = SteadyOptions()
    initial = solve_steady_state(
        device, voltage_V=0.0, generation_scale=1.0, options=steady_opts
    )
    assert initial.converged
    protocol = TransientProtocol(
        np.array([0.0, 1e-7, 2e-7]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.002, 1.005]),
    )

    be_res = solve_transient(
        device, initial, protocol, options=controls(), initial_options=steady_opts
    )
    assert isinstance(be_res, TransientResult1D)
    assert len(be_res.accepted_times_s) == 3

    bdf_res = solve_transient(
        device,
        initial,
        protocol,
        options=controls(),
        initial_options=steady_opts,
        method="bdf",
        bdf_options=BDFOptions(1e-7, adaptive=False, max_order=2),
    )
    assert isinstance(bdf_res, BDFResult)
    assert bdf_res.completed

    ionic_device = compile_ionic_device_1d(
        prepare_single_ion_device(diffusion_only=True), intervals_per_layer=8
    )
    ionic_steady = steady_ionic_controls(ionic_device.ion_node_count)
    point = solve_ionic_steady_state(
        ionic_device, voltage_V=0.002, generation_scale=0.0, options=ionic_steady
    )
    assert point.accepted
    payload = json.dumps(
        json_value(
            {
                "summary": {
                    "accepted": True,
                    "compiled_identity": ionic_device.identity,
                    "voltage_V": 0.002,
                    "generation_scale": 0.0,
                },
                "result": point,
            }
        ),
        allow_nan=False,
    ).encode()
    evidence_path = tmp_path / "initial_ionic.json"
    evidence_path.write_bytes(payload)
    ionic_initial = load_ionic_steady_initial(
        ionic_device,
        evidence_path,
        hashlib.sha256(payload).hexdigest(),
        controls=ionic_steady,
    )
    ionic_protocol = TransientProtocol(
        np.array([0.0, 0.25]),
        np.array([0.002, 0.0025]),
        np.array([0.0, 0.0]),
    )
    ionic_be = solve_ionic_transient(
        ionic_device, ionic_initial, ionic_protocol, options=time_ionic_controls()
    )
    assert isinstance(ionic_be, IonicTransientResult)


def test_assembled_storage_1d_heterointerface_secant_bdf2_and_gauss_maxwell_closure() -> (
    None
):
    """Verify expm1 storage secants, BDF2 weights, equilibrium SG flux, and Gauss-Maxwell closure."""
    device = compile_device_1d(load_device(), intervals_per_layer=8)
    n_nodes = device.node_count
    # Multilayer stack must expose distinct layer dielectric / band properties across interfaces.
    assert len(device.geometry.interfaces) >= 2

    # 1. Exact zero Scharfetter-Gummel carrier flux across heterointerfaces in dark equilibrium.
    eq_problem = AssembledDC1DProblem(device, voltage_V=0.0, generation_scale=0.0)
    eq_steady = solve_steady_state(
        device, voltage_V=0.0, generation_scale=0.0, options=SteadyOptions()
    )
    assert eq_steady.converged
    eq_fluxes = eq_problem.fluxes(eq_steady.state.u)
    assert np.max(np.abs(eq_fluxes.electron.flux)) < 1e-3
    assert np.max(np.abs(eq_fluxes.hole.flux)) < 1e-3

    # 2. Cancellation-free expm1 secant vs tangent action under tiny heterointerface perturbations.
    a0 = AssembledStorage1D(
        AssembledDC1DProblem(device, voltage_V=0.0, generation_scale=1.0)
    )
    a1 = AssembledStorage1D(
        AssembledDC1DProblem(device, voltage_V=0.0, generation_scale=1.0)
    )
    rng = np.random.default_rng(20261102)
    u0 = eq_steady.state.u.copy()
    delta = 1e-11 * rng.normal(size=u0.shape)
    u1 = u0 + delta
    actual_delta = u1 - u0
    secant = a1.storage_increment(u1, a0, u0)
    midpoint_tangent = (
        0.5 * (a1.storage_tangent(u1) + a0.storage_tangent(u0)) @ actual_delta
    )
    row_roundoff = (
        8.0 * np.finfo(float).eps * (abs(a0.storage_tangent(u0)) @ np.abs(actual_delta))
    )
    scale = np.maximum(np.abs(midpoint_tangent), 1e-30) + row_roundoff
    mask = np.abs(midpoint_tangent) > 0.0
    assert np.max(np.abs(secant[mask] - midpoint_tangent[mask]) / scale[mask]) < 1e-8
    # Poisson rows are purely algebraic.
    np.testing.assert_array_equal(secant[:n_nodes], np.zeros(n_nodes))

    # 3. Variable-step BDF2 (step ratio r = 1.5 <= 2) across voltage & generation drive.
    e0 = ElectronicStorage1D(
        AssembledDC1DProblem(device, voltage_V=0.0, generation_scale=1.0)
    )
    e1 = ElectronicStorage1D(
        AssembledDC1DProblem(device, voltage_V=0.02, generation_scale=1.05)
    )
    e2 = ElectronicStorage1D(
        AssembledDC1DProblem(device, voltage_V=0.05, generation_scale=1.10)
    )
    u_a = eq_steady.state.u.copy()
    u_b = u_a + 1e-3 * rng.normal(size=u_a.shape)
    u_c = u_b + 1.5e-3 * rng.normal(size=u_a.shape)

    h0 = HistoryPoint(0.0, u_a, e0.storage(u_a), e0)
    h1 = HistoryPoint(1e-7, u_b, e1.storage(u_b), e1)
    end2 = 2.5e-7  # dt1 = 1e-7, dt2 = 1.5e-7 -> step ratio r = 1.5
    step2 = BDFStep(e2, (h1, h0), end2, 2)

    r = 1.5
    expected_alpha = ((1.0 + 2.0 * r) / (1.0 + r), -(1.0 + r), (r * r) / (1.0 + r))
    np.testing.assert_allclose(step2.coefficients, expected_alpha, rtol=0.0, atol=1e-15)
    assert abs(sum(step2.coefficients)) <= 2.0 * np.finfo(float).eps

    acc = bdf_accounting(step2, u_c)
    disp = acc["bdf_displacement_rate_A_m2"]
    q_rate = acc["charge_inventory_rate_A_m2"]
    # Discrete Gauss closure at contact ports and interior heterointerface cells:
    np.testing.assert_allclose(
        disp[:-1] - disp[1:] + q_rate,
        acc["stable_gauss_defect_rate_A_m2"],
        rtol=0.0,
        atol=1e-12,
    )
    current_scale = max(
        float(np.max(np.abs(acc["bdf_conduction_current_A_m2"]))),
        float(np.max(np.abs(disp))),
        1.0,
    )
    assert float(np.max(np.abs(acc["maxwell_identity_error_A_m2"]))) < (
        64.0 * np.finfo(float).eps * current_scale
    )


def test_assembled_ionic_storage_1d_heterointerface_shared_node_charge_and_blocking_flux() -> (
    None
):
    """Verify W_P / W_e shared heterointerface node charge projection and blocking ion endpoints."""
    device = compile_multiple_ionic_device_1d(
        prepare_multiple_ionic_device(kind="shared"),
        intervals_per_layer=INTERVALS,
        ion_domains=DOMAINS,
    )
    n_nodes = device.node_count
    a0 = AssembledIonicStorage1D(AssembledMultipleIonic1DProblem(device, 0.002, 0.0))
    a1 = AssembledIonicStorage1D(AssembledMultipleIonic1DProblem(device, 0.004, 0.0))
    a2 = AssembledIonicStorage1D(AssembledMultipleIonic1DProblem(device, 0.007, 0.0))

    u0 = np.zeros(device.state_size)
    h0 = HistoryPoint(0.0, u0, a0.storage(u0), a0)
    first = BDFStep(a1, (h0,), 0.2, 1)
    u1 = assembled_ionic_bdf_seed(first)
    u1[3 * n_nodes :] += 0.015
    h1 = HistoryPoint(0.2, u1, a1.storage(u1), a1)
    step2 = BDFStep(a2, (h1, h0), 0.35, 2)
    u2 = assembled_ionic_bdf_seed(step2)
    u2[3 * n_nodes :] -= 0.008

    # Check W_e * rho_ion == q * sum_domains W_P * delta_P at every heterointerface shared node.
    state2 = device.state(u2, voltage_V=0.007)
    _phi, n2, p2, populations = device.fields_by_domain(u2, voltage_V=0.007)
    w_elec = device.electronic_component.geometry.cell_measures_m
    base_rho = device.electronic_component.charge_ledger(n2, p2).space_charge_C_m3()
    full_rho = device.charge_ledger(n2, p2, populations).space_charge_C_m3()
    integrated_ion_charge_C_m2 = w_elec * (full_rho - base_rho)

    expected_ion_charge_C_m2 = np.zeros(n_nodes)
    for idx, domain in enumerate(device.domains):
        dom_state = state2.domain_component_states[idx]
        w_ion = domain.ion_geometry.cell_measures_m
        # Heterointerface boundary nodes use half-endpoint domain measures W_P < W_e.
        elec_indices = domain.ion_node_indices
        if elec_indices[0] > 0:
            assert w_ion[0] < w_elec[elec_indices[0]]
        if elec_indices[-1] < n_nodes - 1:
            assert w_ion[-1] < w_elec[elec_indices[-1]]
        delta_p = np.zeros_like(w_ion)
        if "positive" in domain.ion_species_names:
            delta_p += (
                dom_state.positive_ion_density_m3 - domain.positive_ion_reference_m3
            )
        if "negative" in domain.ion_species_names:
            assert domain.negative_ion_reference_m3 is not None
            assert dom_state.negative_ion_density_m3 is not None
            delta_p -= (
                dom_state.negative_ion_density_m3 - domain.negative_ion_reference_m3
            )
        expected_ion_charge_C_m2[elec_indices] += Q * w_ion * delta_p

    charge_scale = max(float(np.max(np.abs(w_elec * full_rho))), 1e-12)
    np.testing.assert_allclose(
        integrated_ion_charge_C_m2,
        expected_ion_charge_C_m2,
        rtol=1e-12,
        atol=float(16.0 * np.finfo(float).eps * charge_scale),
    )

    # Check blocking heterointerface fluxes and BDF2 Maxwell identity across domains.
    acc = assembled_ionic_bdf_accounting(step2, u2)
    for domain in device.domains:
        dom_acc = acc["domains"][domain.domain_layer_id]
        for species_record in dom_acc["species"].values():
            flux = species_record["domain_flux_m2_s"]
            assert flux[0] == 0.0
            assert flux[-1] == 0.0

    scale = max(
        float(np.max(np.abs(acc["conduction_current_density_A_m2"]))),
        float(np.max(np.abs(acc["bdf_displacement_rate_A_m2"]))),
        1.0,
    )
    assert float(np.max(np.abs(acc["maxwell_identity_error_A_m2"]))) < (
        64.0 * np.finfo(float).eps * scale
    )
