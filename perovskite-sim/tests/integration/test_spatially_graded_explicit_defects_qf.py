"""D3-E4b production QF/DC integration for spatial explicit defects."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    solve_quasi_fermi_jv_sweep,
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.defects import (
    ACCEPTOR,
    ENERGY_ABOVE_VALENCE_BAND,
    EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
    EXPLICIT_DEFECT_SPATIAL_SCHEMA_VERSION,
    EXPLICIT_QUASI_STEADY,
    GAUSSIAN,
    INTEGRATED_TOTAL,
    LAYER_AVERAGE_UNITY,
    NEUTRAL_WHEN_EMPTY,
    NORMALIZED_LAYER_COORDINATE,
    PIECEWISE_LINEAR,
    WIDTH_GAUSSIAN_SIGMA,
    BulkDefectDistribution,
    BulkDefectKinetics,
    BulkDefectSpatialKnot,
    BulkDefectSpatialProfile,
    BulkDefectSpecies,
    ExplicitDefectSchemaError,
)
from perovskite_sim.models.device import DeviceStack, LayerSpec, _edge_params
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.physics.contacts import build_semiconductor_contact_state
from perovskite_sim.physics.defect_closure import (
    evaluate_monovalent_bulk_defects,
)
from perovskite_sim.physics.temperature import thermal_voltage
from perovskite_sim.solver.mol import (
    EXPLICIT_DEFECT_CHARGE_QF_DC,
    build_material_arrays,
)


TEMPERATURE_K = 300.0
NC_M3 = 1.0e24
NV_M3 = 8.0e23
FRONT_GAP_EV = 0.8
BACK_GAP_EV = 0.9


def _stack(*, photon_flux_m2_s: float = 0.0) -> DeviceStack:
    profile = BulkDefectSpatialProfile(
        coordinate=NORMALIZED_LAYER_COORDINATE,
        interpolation=PIECEWISE_LINEAR,
        density_normalization=LAYER_AVERAGE_UNITY,
        knots=(
            BulkDefectSpatialKnot(0.0, 0.5),
            BulkDefectSpatialKnot(1.0, 1.5),
        ),
    )
    species = BulkDefectSpecies(
        name="graded_gaussian_acceptor",
        distribution=BulkDefectDistribution(
            kind=GAUSSIAN,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=8.0e20,
            center_eV_above_vb=0.4,
            energy_reference=ENERGY_ABOVE_VALENCE_BAND,
            width_eV=0.05,
            width_convention=WIDTH_GAUSSIAN_SIGMA,
            support_width_multiplier=6.0,
        ),
        charge_transition=ACCEPTOR,
        neutral_reference=NEUTRAL_WHEN_EMPTY,
        kinetics=BulkDefectKinetics(
            sigma_n_m2=2.0e-19,
            sigma_p_m2=7.0e-20,
            thermal_velocity_n_m_s=1.0e5,
            thermal_velocity_p_m_s=8.0e4,
        ),
        degeneracy=1.0,
        spatial_profile=profile,
    )
    intrinsic = math.sqrt(
        NC_M3
        * NV_M3
        * math.exp(
            -FRONT_GAP_EV / thermal_voltage(TEMPERATURE_K)
        )
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
        N_A=5.0e21,
        N_D=0.0,
        chi=4.0,
        Eg=FRONT_GAP_EV,
        Eg_back=BACK_GAP_EV,
        chi_back=4.05,
        Nc300=NC_M3,
        Nv300=NV_M3,
        defect_schema_version=EXPLICIT_DEFECT_SPATIAL_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=(species,),
    )
    return DeviceStack(
        layers=(LayerSpec("graded_absorber", 300.0e-9, params, "absorber"),),
        V_bi=0.0,
        Phi=photon_flux_m2_s,
        interfaces=(),
        mode="full",
        band_grading=True,
        built_in_potential_mode="semiconductor_work_function",
    )


def _grid(stack: DeviceStack, intervals: int = 8) -> np.ndarray:
    return multilayer_grid([Layer(stack.layers[0].thickness, intervals)])


def test_material_compiler_binds_local_bands_profile_and_contact_endpoints():
    stack = _stack()
    grid = _grid(stack)
    material = build_material_arrays(
        grid,
        stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
        defect_energy_quadrature_order=12,
    )
    model = material.monovalent_bulk_defects
    assert model is not None
    assert model.has_spatial_profiles
    assert model.has_distributed_species
    region = model.regions[0]
    np.testing.assert_allclose(
        region.local_band_gap_eV,
        material.Eg_phys,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        region.source_density_multipliers[0],
        0.5 + region.normalized_layer_coordinates,
        rtol=0.0,
        atol=2.0e-16,
    )
    diagnostics = evaluate_monovalent_bulk_defects(
        np.full(grid.size, 2.0e20),
        np.full(grid.size, 3.0e20),
        model,
    )
    assert diagnostics.minimum_density_multipliers == (0.5,)
    assert diagnostics.maximum_density_multipliers == (1.5,)

    front = _edge_params(stack.layers[0], "front", True)
    back = _edge_params(stack.layers[0], "back", True)
    assert front.defect_schema_version == (
        EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION
    )
    assert back.defect_schema_version == (
        EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION
    )
    assert front.bulk_defects[0].distribution.total_density_m3 == 4.0e20
    assert back.bulk_defects[0].distribution.total_density_m3 == 1.2e21
    front_state = build_semiconductor_contact_state(
        front,
        temperature_K=TEMPERATURE_K,
        use_temperature_scaling=True,
        defect_energy_quadrature_order=12,
    )
    back_state = build_semiconductor_contact_state(
        back,
        temperature_K=TEMPERATURE_K,
        use_temperature_scaling=True,
        defect_energy_quadrature_order=12,
    )
    assert material.n_L == front_state.electron_density_m3
    assert material.p_L == front_state.hole_density_m3
    assert material.n_R == back_state.electron_density_m3
    assert material.p_R == back_state.hole_density_m3


def test_compiled_profile_preserves_layer_average_density_on_the_grid():
    stack = _stack()
    grid = _grid(stack, intervals=16)
    material = build_material_arrays(
        grid,
        stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
        defect_energy_quadrature_order=12,
    )
    model = material.monovalent_bulk_defects
    assert model is not None
    region = model.regions[0]
    source_density = float(
        region.species[0].distribution.total_density_m3
    )
    compiled_density = source_density * region.source_density_multipliers[0]

    assert np.trapezoid(compiled_density, grid) == pytest.approx(
        source_density * stack.layers[0].thickness,
        rel=2.0e-15,
        abs=0.0,
    )


def test_local_band_gap_rejects_distribution_support_before_qf_solve():
    stack = _stack()
    layer = stack.layers[0]
    # Gaussian support is [0.25, 0.55] eV above the local valence band.
    narrow_back = replace(layer.params, Eg_back=0.50)
    unsupported = replace(stack, layers=(replace(layer, params=narrow_back),))

    with pytest.raises(
        ExplicitDefectSchemaError,
        match="support must lie completely inside the local material band gap",
    ):
        build_material_arrays(
            _grid(unsupported),
            unsupported,
            explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
            defect_energy_quadrature_order=12,
        )


def test_spatially_graded_dark_qf_state_and_small_jv_are_certified():
    stack = _stack(photon_flux_m2_s=2.0e15)
    grid = _grid(stack, intervals=6)
    dark = solve_quasi_fermi_steady_state(
        grid,
        stack,
        illuminated=False,
        defect_energy_quadrature_order=12,
    )

    assert dark.certified
    assert dark.bulk_defect_diagnostics is not None
    assert dark.bulk_defect_diagnostics.spatial_profile_sha256s[0] is not None
    assert dark.contact_thermodynamic_status == "certified"

    sweep = solve_quasi_fermi_jv_sweep(
        grid,
        stack,
        np.asarray([0.0, 0.002]),
        illumination_steps=(0.0, 1.0e-4, 1.0e-2, 1.0),
        continuity_tolerance_A_m2=2.0e-4,
        current_spread_tolerance_A_m2=2.0e-4,
        defect_energy_quadrature_order=12,
    )
    assert sweep.certified
    assert sweep.defect_energy_quadrature_order == 12
    assert sweep.defect_spatial_profile_sha256s == (
        stack.layers[0].params.bulk_defects[0].spatial_profile.sha256,
    )
    assert sweep.defect_density_multiplier_bounds == ((0.5, 1.5),)
    assert all(
        point.bulk_defect_diagnostics is not None
        and point.bulk_defect_diagnostics.spatial_profile_sha256s[0] is not None
        for point in sweep.points
    )
