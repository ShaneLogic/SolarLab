"""DEF-3 charged explicit-defect closure on the certified QF/DC path."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.defect_aware_impedance import (
    BULK_DEFECT_DEVICE_AC_SCOPE,
    BulkDefectDeviceACCertificationError,
    BulkDefectDeviceACError,
    run_bulk_defect_device_impedance,
)
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    QuasiFermiSteadyStateError,
    _QuasiFermiSystem,
    solve_quasi_fermi_jv_sweep,
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.defects import (
    ACCEPTOR,
    CONDUCTION_BAND_TAIL,
    DONOR,
    ENERGY_ABOVE_VALENCE_BAND,
    EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
    EXPLICIT_DEFECT_SCHEMA_VERSION,
    EXPLICIT_QUASI_STEADY,
    GAUSSIAN,
    INTEGRATED_TOTAL,
    NEUTRAL,
    NEUTRAL_ALL_OCCUPANCIES,
    NEUTRAL_WHEN_EMPTY,
    NEUTRAL_WHEN_FILLED,
    SINGLE_LEVEL,
    UNIFORM,
    VALENCE_BAND_TAIL,
    WIDTH_GAUSSIAN_SIGMA,
    WIDTH_SCAPS_CHARACTERISTIC,
    WIDTH_UNIFORM_FULL,
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
from perovskite_sim.physics.recombination import (
    total_recombination,
    total_recombination_at_node,
)
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
            NEUTRAL_WHEN_EMPTY if transition == ACCEPTOR else NEUTRAL_WHEN_FILLED
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
        NC_M3 * NV_M3 * math.exp(-GAP_EV / thermal_voltage(TEMPERATURE_K))
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


def _distributed_species(
    kind: str,
    transition: str = ACCEPTOR,
    *,
    density_m3: float = 2.0e21,
) -> BulkDefectSpecies:
    base = _species(ACCEPTOR if transition == NEUTRAL else transition)
    values: dict[str, object] = {
        "kind": kind,
        "normalization": INTEGRATED_TOTAL,
        "total_density_m3": density_m3,
        "center_eV_above_vb": 0.39,
        "energy_reference": ENERGY_ABOVE_VALENCE_BAND,
    }
    if kind == GAUSSIAN:
        values |= {
            "width_eV": 0.05,
            "width_convention": WIDTH_GAUSSIAN_SIGMA,
            "support_width_multiplier": 6.0,
        }
    elif kind == UNIFORM:
        values |= {
            "width_eV": 0.30,
            "width_convention": WIDTH_UNIFORM_FULL,
        }
    elif kind == CONDUCTION_BAND_TAIL:
        values |= {
            "center_eV_above_vb": 0.75,
            "width_eV": 0.05,
            "width_convention": WIDTH_SCAPS_CHARACTERISTIC,
            "support_width_multiplier": 6.0,
        }
    elif kind == VALENCE_BAND_TAIL:
        values |= {
            "center_eV_above_vb": 0.05,
            "width_eV": 0.05,
            "width_convention": WIDTH_SCAPS_CHARACTERISTIC,
            "support_width_multiplier": 6.0,
        }
    neutral_reference = {
        ACCEPTOR: NEUTRAL_WHEN_EMPTY,
        DONOR: NEUTRAL_WHEN_FILLED,
        NEUTRAL: NEUTRAL_ALL_OCCUPANCIES,
    }[transition]
    return replace(
        base,
        name=f"{kind}_{transition}",
        distribution=BulkDefectDistribution(**values),
        charge_transition=transition,
        neutral_reference=neutral_reference,
    )


def _distributed_stack(
    kind: str,
    transition: str = ACCEPTOR,
    *,
    photon_flux_m2_s: float = 0.0,
) -> DeviceStack:
    base = _stack(
        transition=(ACCEPTOR if transition == NEUTRAL else transition),
        photon_flux_m2_s=photon_flux_m2_s,
    )
    params = base.layers[0].params
    assert params is not None
    return replace(
        base,
        layers=(
            replace(
                base.layers[0],
                params=replace(
                    params,
                    defect_schema_version=(EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION),
                    bulk_defects=(_distributed_species(kind, transition),),
                ),
            ),
        ),
    )


def _grid(stack: DeviceStack, intervals: int = 12) -> np.ndarray:
    return multilayer_grid([Layer(stack.layers[0].thickness, intervals)])


def _heterojunction_stack(*, photon_flux_m2_s: float = 0.0) -> DeviceStack:
    left = _stack(photon_flux_m2_s=photon_flux_m2_s).layers[0].params
    assert left is not None
    right_gap_eV = 0.90
    right_intrinsic = math.sqrt(
        NC_M3 * NV_M3 * math.exp(-right_gap_eV / thermal_voltage(TEMPERATURE_K))
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


def test_v2_single_level_material_is_exact_to_v1_qf_dc_physics():
    v1_stack = _stack()
    v1_params = v1_stack.layers[0].params
    v1_species = v1_params.bulk_defects[0]
    v2_species = replace(
        v1_species,
        distribution=replace(
            v1_species.distribution,
            energy_reference=ENERGY_ABOVE_VALENCE_BAND,
        ),
    )
    v2_stack = replace(
        v1_stack,
        layers=(
            replace(
                v1_stack.layers[0],
                params=replace(
                    v1_params,
                    defect_schema_version=(EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION),
                    bulk_defects=(v2_species,),
                ),
            ),
        ),
    )
    grid = _grid(v1_stack, 8)

    v1_material = build_material_arrays(
        grid,
        v1_stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
    )
    v2_material = build_material_arrays(
        grid,
        v2_stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
    )
    for name in ("n_L", "p_L", "n_R", "p_R"):
        assert getattr(v1_material, name) == getattr(v2_material, name)
    n = np.geomspace(v1_material.n_L, 3.0 * v1_material.n_R, grid.size)
    p = v1_material.ni_sq / n
    v1 = evaluate_monovalent_bulk_defects(
        n,
        p,
        v1_material.monovalent_bulk_defects,
    )
    v2 = evaluate_monovalent_bulk_defects(
        n,
        p,
        v2_material.monovalent_bulk_defects,
    )

    assert v1.model_identity_sha256 != v2.model_identity_sha256
    for name in (
        "kinetic_denominator_s1",
        "occupancy",
        "occupied_density_m3",
        "charge_density_C_m3",
        "recombination_rate_m3_s",
        "recombination_derivative_n_s1",
        "recombination_derivative_p_s1",
        "charge_derivative_fixed_qf_C_m3_V",
        "total_charge_density_C_m3",
        "total_recombination_rate_m3_s",
        "total_recombination_derivative_n_s1",
        "total_recombination_derivative_p_s1",
        "total_charge_derivative_fixed_qf_C_m3_V",
    ):
        np.testing.assert_array_equal(getattr(v1, name), getattr(v2, name))


def test_v2_distributed_material_opens_only_on_the_guarded_qf_dc_lane():
    stack = _distributed_stack(GAUSSIAN)
    grid = _grid(stack, 6)

    with pytest.raises(ExplicitDefectCapabilityError, match="guarded QF/DC"):
        build_material_arrays(grid, stack)

    material = build_material_arrays(
        grid,
        stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
        defect_energy_quadrature_order=16,
    )
    model = material.monovalent_bulk_defects
    assert model is not None
    assert model.has_distributed_species
    assert material.explicit_defect_energy_quadrature_order == 16
    assert model.source_energy_orders == (16,)
    n = np.geomspace(material.n_L, 2.0 * material.n_R, grid.size)
    p = material.ni_sq / n
    evaluation = evaluate_monovalent_bulk_defects(n, p, model)
    node = grid.size // 2
    scalar = total_recombination_at_node(
        float(n[node]),
        float(p[node]),
        float(material.ni_sq[node]),
        float(material.tau_n[node]),
        float(material.tau_p[node]),
        float(material.n1[node]),
        float(material.p1[node]),
        float(material.B_rad[node]),
        float(material.C_n[node]),
        float(material.C_p[node]),
        node=node,
        monovalent_bulk_defects=model,
    )
    assert scalar == pytest.approx(
        evaluation.total_recombination_rate_m3_s[node],
        rel=2.0e-15,
    )


@pytest.mark.parametrize(
    "kind",
    (GAUSSIAN, UNIFORM, CONDUCTION_BAND_TAIL, VALENCE_BAND_TAIL),
)
def test_distributed_dark_equilibrium_is_qf_residual_certified(kind):
    stack = _distributed_stack(kind)
    result = solve_quasi_fermi_steady_state(
        _grid(stack, 8),
        stack,
        V_app=0.0,
        illuminated=False,
        defect_energy_quadrature_order=20,
    )

    assert result.certified
    assert result.contact_thermodynamic_status == "certified"
    assert result.contact_fermi_level_span_eV is not None
    assert result.contact_fermi_level_span_eV < 1.0e-12
    assert result.defect_energy_quadrature_order == 20
    assert result.defect_distribution_kinds == (kind,)
    diagnostics = result.bulk_defect_diagnostics
    assert diagnostics is not None
    assert diagnostics.distribution_kinds == (kind,)
    assert diagnostics.source_energy_orders == (20,)
    assert len(diagnostics.source_node_identifiers[0]) == 20
    assert diagnostics.minimum_kinetic_denominator_s1 > 0.0
    assert np.max(np.abs(diagnostics.total_recombination_rate_m3_s)) < 1.0e12


def test_neutral_distributed_source_uses_the_same_guarded_qf_dc_contract():
    stack = _distributed_stack(GAUSSIAN, NEUTRAL)
    result = solve_quasi_fermi_steady_state(
        _grid(stack, 8),
        stack,
        illuminated=False,
        defect_energy_quadrature_order=16,
    )

    assert result.certified
    assert result.defect_energy_quadrature_order == 16
    diagnostics = result.bulk_defect_diagnostics
    assert diagnostics is not None
    assert diagnostics.charge_transitions == (NEUTRAL,)
    np.testing.assert_array_equal(
        diagnostics.total_charge_density_C_m3,
        np.zeros_like(diagnostics.total_charge_density_C_m3),
    )


def test_distributed_energy_order_is_bound_into_material_and_public_result():
    stack = _distributed_stack(GAUSSIAN)
    grid = _grid(stack, 6)
    material_12 = build_material_arrays(
        grid,
        stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
        defect_energy_quadrature_order=12,
    )
    material_24 = build_material_arrays(
        grid,
        stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
        defect_energy_quadrature_order=24,
    )

    assert material_12.monovalent_bulk_defects is not None
    assert material_24.monovalent_bulk_defects is not None
    assert (
        material_12.monovalent_bulk_defects.identity_sha256
        != material_24.monovalent_bulk_defects.identity_sha256
    )
    with pytest.raises(QuasiFermiSteadyStateError, match="quadrature order"):
        solve_quasi_fermi_steady_state(
            grid,
            stack,
            illuminated=False,
            mat=material_12,
            defect_energy_quadrature_order=24,
        )


def test_distributed_bulk_charge_fixed_qf_tangent_matches_centered_difference():
    stack = _distributed_stack(CONDUCTION_BAND_TAIL)
    grid = _grid(stack, 6)
    material = build_material_arrays(
        grid,
        stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
        defect_energy_quadrature_order=24,
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
        illuminated.max_normalized_cell_residual < illuminated.numerical_residual_limit
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
        point.contact_thermodynamic_status == "certified" for point in sweep.points
    )


def test_mixed_distributed_species_close_in_illuminated_qf_jv():
    base = _distributed_stack(GAUSSIAN, photon_flux_m2_s=2.0e16)
    params = base.layers[0].params
    assert params is not None
    species = (
        _distributed_species(
            GAUSSIAN,
            ACCEPTOR,
            density_m3=8.0e20,
        ),
        _distributed_species(
            CONDUCTION_BAND_TAIL,
            DONOR,
            density_m3=4.0e20,
        ),
    )
    stack = replace(
        base,
        layers=(
            replace(
                base.layers[0],
                params=replace(params, bulk_defects=species),
            ),
        ),
    )
    sweep = solve_quasi_fermi_jv_sweep(
        _grid(stack, 10),
        stack,
        np.asarray([0.0, 0.005, 0.01]),
        illumination_steps=(0.0, 1.0e-4, 1.0e-2, 1.0),
        continuity_tolerance_A_m2=2.0e-4,
        current_spread_tolerance_A_m2=2.0e-4,
        defect_energy_quadrature_order=16,
    )

    assert sweep.certified
    assert sweep.defect_energy_quadrature_order == 16
    assert sweep.defect_distribution_kinds == (
        GAUSSIAN,
        CONDUCTION_BAND_TAIL,
    )
    assert all(point.defect_energy_quadrature_order == 16 for point in sweep.points)
    assert all(
        point.defect_distribution_kinds == (GAUSSIAN, CONDUCTION_BAND_TAIL)
        for point in sweep.points
    )
    assert all(point.bulk_defect_diagnostics is not None for point in sweep.points)


def test_distributed_qf_scope_rejects_mobile_ions_fd_and_spatial_grading():
    base = _distributed_stack(GAUSSIAN)
    params = base.layers[0].params
    assert params is not None
    mobile = replace(
        base,
        layers=(
            replace(
                base.layers[0],
                params=replace(
                    params,
                    D_ion=1.0e-14,
                    P0=1.0e22,
                    P_lim=2.0e22,
                ),
            ),
        ),
    )
    with pytest.raises(QuasiFermiSteadyStateError, match="mobile ions"):
        solve_quasi_fermi_steady_state(
            _grid(mobile, 6),
            mobile,
            illuminated=False,
        )

    with pytest.raises(
        ValueError,
        match="explicit_quasi_steady requires.*maxwell_boltzmann",
    ):
        replace(params, carrier_statistics="fermi_dirac")

    graded = replace(
        base,
        mode="full",
        band_grading=True,
        layers=(
            replace(
                base.layers[0],
                params=replace(
                    params,
                    Eg_back=0.82,
                    chi_back=4.02,
                ),
            ),
        ),
    )
    with pytest.raises(ExplicitDefectCapabilityError, match="spatial grading"):
        solve_quasi_fermi_steady_state(
            _grid(graded, 6),
            graded,
            illuminated=False,
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


def test_dynamic_bulk_defect_device_ac_certifies_both_limits_and_currents():
    stack = _stack()
    grid = _grid(stack, 4)
    result = run_bulk_defect_device_impedance(
        grid,
        stack,
        np.geomspace(1.0e-4, 1.0e12, 33),
        illuminated=False,
    )

    certificate = result.certificate
    assert certificate.certified
    assert certificate.reasons == ()
    assert certificate.scope == BULK_DEFECT_DEVICE_AC_SCOPE
    assert certificate.frequency_window.certified
    assert certificate.dc_maximum_normalized_residual < 1.0e-10
    assert certificate.dc_electron_continuity_bound_A_m2 < 1.0e-4
    assert certificate.dc_hole_continuity_bound_A_m2 < 1.0e-4
    assert certificate.dc_face_current_spread_A_m2 < 1.0e-4
    assert certificate.dc_poisson_residual < 1.0e-8
    assert certificate.qss_embedding_normalized_error < 1.0e-18
    assert certificate.maximum_local_trap_balance_relative_error < 1.0e-4
    assert certificate.maximum_all_face_admittance_spread < 1.0e-8
    assert certificate.maximum_refinement_relative_change < 1.0e-6
    assert certificate.low_frequency_qss_relative_error < 1.0e-6
    assert certificate.high_frequency_frozen_relative_error < 1.0e-6
    assert result.layout.size == grid.size - 2
    np.testing.assert_allclose(
        result.admittance_faces_S_m2,
        result.electron_conduction_admittance_faces_S_m2
        + result.hole_conduction_admittance_faces_S_m2
        + result.displacement_admittance_faces_S_m2,
        rtol=2.0e-15,
        atol=1.0e-12,
    )
    assert abs(result.trap_charge_storage_response_F_m2[0]) > 1.0e6 * abs(
        result.trap_charge_storage_response_F_m2[-1]
    )
    assert not result.admittance_S_m2.flags.writeable


def test_dynamic_bulk_defect_device_ac_requires_an_explicit_defect_model():
    stack = _stack()
    params = stack.layers[0].params
    assert params is not None
    lifetime_stack = replace(
        stack,
        layers=(
            replace(
                stack.layers[0],
                params=replace(
                    params,
                    defect_schema_version=None,
                    defect_model="effective_lifetime",
                    bulk_defects=(),
                ),
            ),
        ),
    )

    with pytest.raises(BulkDefectDeviceACError, match="explicit-defect model"):
        run_bulk_defect_device_impedance(
            _grid(lifetime_stack, 4),
            lifetime_stack,
            np.geomspace(1.0e-4, 1.0e4, 17),
            illuminated=False,
        )


def test_dynamic_bulk_defect_device_ac_rejects_dc_state_from_another_model():
    acceptor_stack = _stack(ACCEPTOR)
    donor_stack = _stack(DONOR)
    grid = _grid(acceptor_stack, 4)
    donor_state = solve_quasi_fermi_steady_state(
        grid,
        donor_stack,
        illuminated=False,
    )

    with pytest.raises(BulkDefectDeviceACError, match="model identity"):
        run_bulk_defect_device_impedance(
            grid,
            acceptor_stack,
            np.geomspace(1.0e-4, 1.0e12, 33),
            illuminated=False,
            dc_state=donor_state,
        )


def test_dynamic_bulk_defect_device_ac_recertifies_supplied_dc_state():
    stack = _stack()
    grid = _grid(stack, 4)
    state = solve_quasi_fermi_steady_state(grid, stack, illuminated=False)
    tampered_increment = np.array(
        state.electron_quasi_fermi_increment_V,
        dtype=float,
        copy=True,
    )
    tampered_increment[1:-1] += 0.05
    tampered = replace(
        state,
        electron_quasi_fermi_increment_V=tampered_increment,
    )

    with pytest.raises(BulkDefectDeviceACError, match="not certified"):
        run_bulk_defect_device_impedance(
            grid,
            stack,
            np.geomspace(1.0e-4, 1.0e12, 33),
            illuminated=False,
            dc_state=tampered,
        )


def test_dynamic_bulk_defect_device_ac_incomplete_window_is_partial_or_raises():
    stack = _stack()
    grid = _grid(stack, 4)
    frequencies = np.geomspace(1.0e4, 1.0e6, 5)
    partial = run_bulk_defect_device_impedance(
        grid,
        stack,
        frequencies,
        illuminated=False,
        require_certificate=False,
    )
    assert not partial.certificate.certified
    assert "trap_frequency_window_incomplete" in partial.certificate.reasons
    with pytest.raises(BulkDefectDeviceACCertificationError) as exc_info:
        run_bulk_defect_device_impedance(
            grid,
            stack,
            frequencies,
            illuminated=False,
        )
    assert exc_info.value.result.certificate.reasons == partial.certificate.reasons


def test_distributed_dynamic_device_ac_uses_every_energy_node():
    stack = _distributed_stack(GAUSSIAN)
    grid = _grid(stack, 3)
    order = 6
    result = run_bulk_defect_device_impedance(
        grid,
        stack,
        np.geomspace(1.0e-5, 1.0e12, 35),
        illuminated=False,
        defect_energy_quadrature_order=order,
    )

    assert result.certificate.certified
    assert result.layout.size == (grid.size - 2) * order
    assert set(result.layout.energy_indices) == set(range(order))
    assert result.trap_occupancy_response_per_V.shape == (
        result.frequencies_Hz.size,
        result.layout.size,
    )
