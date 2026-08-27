"""DEF-3 charged explicit-defect closure on the certified QF/DC path."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    QuasiFermiSteadyStateError,
    _QuasiFermiSystem,
    solve_quasi_fermi_jv_sweep,
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.defects import (
    ACCEPTOR,
    DONOR,
    EXPLICIT_DEFECT_SCHEMA_VERSION,
    EXPLICIT_QUASI_STEADY,
    INTEGRATED_TOTAL,
    NEUTRAL_WHEN_EMPTY,
    NEUTRAL_WHEN_FILLED,
    SINGLE_LEVEL,
    BulkDefectDistribution,
    BulkDefectKinetics,
    BulkDefectSpecies,
    ExplicitDefectCapabilityError,
)
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.physics.defect_closure import (
    evaluate_monovalent_bulk_defects,
)
from perovskite_sim.physics.recombination import total_recombination
from perovskite_sim.physics.temperature import thermal_voltage
from perovskite_sim.solver.mol import (
    EXPLICIT_DEFECT_CHARGE_QF_DC,
    StateVec,
    assemble_rhs,
    build_material_arrays,
)


TEMPERATURE_K = 300.0
GAP_EV = 0.80
NC_M3 = 1.0e24
NV_M3 = 8.0e23


def _species(transition: str) -> BulkDefectSpecies:
    return BulkDefectSpecies(
        name=f"bulk_{transition}",
        distribution=BulkDefectDistribution(
            kind=SINGLE_LEVEL,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=2.0e21,
            center_eV_above_vb=0.39,
        ),
        charge_transition=transition,
        neutral_reference=(
            NEUTRAL_WHEN_EMPTY
            if transition == ACCEPTOR
            else NEUTRAL_WHEN_FILLED
        ),
        kinetics=BulkDefectKinetics(
            sigma_n_m2=2.0e-19,
            sigma_p_m2=7.0e-20,
            thermal_velocity_n_m_s=1.0e5,
            thermal_velocity_p_m_s=8.0e4,
        ),
        degeneracy=1.0,
    )


def _stack(
    transition: str = ACCEPTOR,
    *,
    photon_flux_m2_s: float = 0.0,
    contact_mode: str = "semiconductor_work_function",
) -> DeviceStack:
    intrinsic = math.sqrt(
        NC_M3
        * NV_M3
        * math.exp(-GAP_EV / thermal_voltage(TEMPERATURE_K))
    )
    params = MaterialParams(
        eps_r=20.0,
        mu_n=2.0e-3,
        mu_p=2.0e-3,
        D_ion=0.0,
        P_lim=1.0e30,
        P0=0.0,
        ni=intrinsic,
        tau_n=1.0e-6,
        tau_p=1.0e-6,
        n1=intrinsic,
        p1=intrinsic,
        B_rad=0.0,
        C_n=0.0,
        C_p=0.0,
        alpha=4.0e5,
        N_A=0.0,
        N_D=0.0,
        chi=4.0,
        Eg=GAP_EV,
        Nc300=NC_M3,
        Nv300=NV_M3,
        defect_schema_version=EXPLICIT_DEFECT_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=(_species(transition),),
    )
    return DeviceStack(
        layers=(LayerSpec("defective", 300.0e-9, params, "absorber"),),
        V_bi=0.0,
        Phi=photon_flux_m2_s,
        interfaces=(),
        mode="legacy",
        built_in_potential_mode=contact_mode,
    )


def _grid(stack: DeviceStack, intervals: int = 12) -> np.ndarray:
    return multilayer_grid([Layer(stack.layers[0].thickness, intervals)])


def _heterojunction_stack(*, photon_flux_m2_s: float = 0.0) -> DeviceStack:
    left = _stack(photon_flux_m2_s=photon_flux_m2_s).layers[0].params
    assert left is not None
    right_gap_eV = 0.90
    right_intrinsic = math.sqrt(
        NC_M3
        * NV_M3
        * math.exp(-right_gap_eV / thermal_voltage(TEMPERATURE_K))
    )
    right = replace(
        left,
        alpha=0.0,
        chi=4.05,
        Eg=right_gap_eV,
        ni=right_intrinsic,
        n1=right_intrinsic,
        p1=right_intrinsic,
        defect_schema_version=None,
        defect_model="effective_lifetime",
        bulk_defects=(),
    )
    return DeviceStack(
        layers=(
            LayerSpec("defective_absorber", 200.0e-9, left, "absorber"),
            LayerSpec("transport", 100.0e-9, right, "ETL"),
        ),
        V_bi=0.0,
        Phi=photon_flux_m2_s,
        interfaces=((0.0, 0.0),),
        mode="full",
        built_in_potential_mode="semiconductor_work_function",
    )


def _heterojunction_grid(stack: DeviceStack, intervals: int = 6) -> np.ndarray:
    return multilayer_grid(
        [Layer(layer.thickness, intervals) for layer in stack.layers],
        alpha=(2.0, 2.0),
    )


def test_default_material_and_wrong_contact_mode_fail_closed():
    stack = _stack()
    grid = _grid(stack, 6)

    with pytest.raises(ExplicitDefectCapabilityError, match="neutral species only"):
        build_material_arrays(grid, stack)
    with pytest.raises(
        ExplicitDefectCapabilityError,
        match="semiconductor_work_function",
    ):
        build_material_arrays(
            grid,
            _stack(contact_mode="legacy_manual"),
            explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
        )


def test_qf_material_compiles_one_shared_charge_and_recombination_model():
    stack = _stack()
    grid = _grid(stack, 6)
    material = build_material_arrays(
        grid,
        stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
    )
    model = material.monovalent_bulk_defects
    assert model is not None
    assert material.neutral_bulk_defects is None
    assert model.charge_transitions == (ACCEPTOR,)
    n = np.geomspace(material.n_L, 2.0 * material.n_R, grid.size)
    p = material.ni_sq / n
    evaluation = evaluate_monovalent_bulk_defects(n, p, model)
    recombination = total_recombination(
        n,
        p,
        material.ni_sq,
        material.tau_n,
        material.tau_p,
        material.n1,
        material.p1,
        material.B_rad,
        material.C_n,
        material.C_p,
        monovalent_bulk_defects=model,
    )

    np.testing.assert_array_equal(
        recombination,
        evaluation.total_recombination_rate_m3_s,
    )
    assert np.all(evaluation.total_charge_density_C_m3 < 0.0)
    assert evaluation.minimum_kinetic_denominator_s1 > 0.0
    state = StateVec.pack(n, p, material.P_ion0.copy())
    with pytest.raises(ExplicitDefectCapabilityError, match="guarded QF/DC"):
        assemble_rhs(
            0.0,
            state,
            grid,
            stack,
            material,
            illuminated=False,
        )


@pytest.mark.parametrize("transition", [ACCEPTOR, DONOR])
def test_bulk_charge_fixed_qf_tangent_matches_centered_difference(transition):
    stack = _stack(transition)
    grid = _grid(stack, 6)
    material = build_material_arrays(
        grid,
        stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
    )
    system = _QuasiFermiSystem(
        grid,
        stack,
        material,
        0.0,
        poisson_tolerance_V=1.0e-13,
        poisson_max_iterations=100,
    )
    n = system.base[: grid.size]
    p = system.base[grid.size : 2 * grid.size]
    _rho, tangent = system._bulk_space_charge_and_tangent(n, p)
    step = 1.0e-7
    factor = np.exp(step / material.V_T_device)
    rho_plus, _ = system._bulk_space_charge_and_tangent(
        n * factor,
        p / factor,
    )
    rho_minus, _ = system._bulk_space_charge_and_tangent(
        n / factor,
        p * factor,
    )

    np.testing.assert_allclose(
        tangent,
        (rho_plus - rho_minus) / (2.0 * step),
        rtol=3.0e-8,
        atol=1.0e-10,
    )


@pytest.mark.parametrize("transition", [ACCEPTOR, DONOR])
def test_charged_dark_equilibrium_is_qf_residual_certified(transition):
    stack = _stack(transition)
    grid = _grid(stack)

    result = solve_quasi_fermi_steady_state(
        grid,
        stack,
        V_app=0.0,
        illuminated=False,
    )

    assert result.certified
    assert result.contact_thermodynamic_status == "certified"
    assert result.contact_fermi_level_span_eV is not None
    assert result.contact_fermi_level_span_eV < 1.0e-12
    assert result.max_normalized_cell_residual < 1.0e-10
    assert result.poisson_residual < 1.0e-10
    assert result.face_current_spread_A_m2 < 1.0e-9
    diagnostics = result.bulk_defect_diagnostics
    assert diagnostics is not None
    assert diagnostics.charge_transitions == (transition,)
    assert 0.0 < diagnostics.minimum_occupancy
    assert diagnostics.maximum_occupancy < 1.0
    assert diagnostics.minimum_kinetic_denominator_s1 > 0.0
    assert np.max(np.abs(diagnostics.total_recombination_rate_m3_s)) < 1.0e12


def test_charged_illuminated_biased_state_is_qf_residual_certified():
    stack = _stack(photon_flux_m2_s=2.0e16)
    grid = _grid(stack)

    result = solve_quasi_fermi_steady_state(
        grid,
        stack,
        V_app=0.01,
        illuminated=True,
        illumination_steps=(0.0, 1.0e-4, 1.0e-2, 1.0),
        continuity_tolerance_A_m2=2.0e-4,
        current_spread_tolerance_A_m2=2.0e-4,
    )

    assert result.certified
    assert result.bulk_defect_diagnostics is not None
    assert result.contact_thermodynamic_status == "certified"
    assert abs(result.current_A_m2) > 0.0
    assert result.max_normalized_cell_residual < 1.0e-10
    assert result.poisson_residual < 1.0e-8


def test_charged_defect_and_heterojunction_close_in_one_qf_dc_residual():
    dark_stack = _heterojunction_stack()
    grid = _heterojunction_grid(dark_stack)

    dark = solve_quasi_fermi_steady_state(
        grid,
        dark_stack,
        V_app=0.0,
        illuminated=False,
        interface_boundary=True,
    )

    assert dark.certified
    assert dark.interface_boundary
    assert dark.interface_faces
    assert dark.bulk_defect_diagnostics is not None
    assert dark.contact_thermodynamic_status == "certified"
    assert dark.contact_fermi_level_span_eV is not None
    assert dark.contact_fermi_level_span_eV < 1.0e-12
    assert dark.max_normalized_cell_residual < dark.numerical_residual_limit
    assert dark.poisson_residual < 1.0e-8

    illuminated_stack = _heterojunction_stack(photon_flux_m2_s=2.0e16)
    illuminated = solve_quasi_fermi_steady_state(
        grid,
        illuminated_stack,
        V_app=0.005,
        illuminated=True,
        interface_boundary=True,
        illumination_steps=(0.0, 1.0e-4, 1.0e-2, 1.0),
        continuity_tolerance_A_m2=2.0e-4,
        current_spread_tolerance_A_m2=2.0e-4,
    )

    assert illuminated.certified
    assert illuminated.bulk_defect_diagnostics is not None
    assert illuminated.interface_basin_initializations == 0
    assert (
        illuminated.max_normalized_cell_residual
        < illuminated.numerical_residual_limit
    )
    assert illuminated.poisson_residual < 1.0e-8


def test_public_qf_jv_sweep_retains_charged_defect_certificates():
    stack = _stack(photon_flux_m2_s=2.0e16)
    grid = _grid(stack)
    sweep = solve_quasi_fermi_jv_sweep(
        grid,
        stack,
        np.array([0.0, 0.005, 0.01]),
        illumination_steps=(0.0, 1.0e-4, 1.0e-2, 1.0),
        continuity_tolerance_A_m2=2.0e-4,
        current_spread_tolerance_A_m2=2.0e-4,
    )

    assert sweep.certified
    assert sweep.voltages_V.tolist() == [0.0, 0.005, 0.01]
    assert len(sweep.points) == 3
    assert all(point.bulk_defect_diagnostics is not None for point in sweep.points)
    assert all(
        point.contact_thermodynamic_status == "certified"
        for point in sweep.points
    )


def test_qf_wraps_contact_certificate_failure_in_public_error():
    stack = _stack()
    grid = _grid(stack, 6)
    material = build_material_arrays(
        grid,
        stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
    )
    inconsistent = replace(material, n_L=10.0 * material.n_L)

    with pytest.raises(
        QuasiFermiSteadyStateError,
        match="certified contact thermodynamic reference",
    ):
        solve_quasi_fermi_steady_state(
            grid,
            stack,
            illuminated=False,
            mat=inconsistent,
        )


def test_qf_rejects_supplied_material_from_another_defect_document():
    stack = _stack(ACCEPTOR)
    donor_stack = _stack(DONOR)
    grid = _grid(stack, 6)
    donor_material = build_material_arrays(
        grid,
        donor_stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
    )

    with pytest.raises(QuasiFermiSteadyStateError, match="does not match"):
        solve_quasi_fermi_steady_state(
            grid,
            stack,
            illuminated=False,
            mat=donor_material,
        )


def test_intrinsic_product_mismatch_is_rejected_before_qf_newton():
    stack = _stack()
    inconsistent = replace(
        stack,
        layers=(
            replace(
                stack.layers[0],
                params=replace(stack.layers[0].params, ni=2.0e16),
            ),
        ),
    )

    with pytest.raises(ExplicitDefectCapabilityError, match=r"ni\^2=Nc\*Nv"):
        solve_quasi_fermi_steady_state(
            _grid(inconsistent, 6),
            inconsistent,
            illuminated=False,
        )
