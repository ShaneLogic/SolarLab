"""D7-E3 stationary metastable preparation and frozen-measurement contract."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.metastable_preparation import (
    FrozenMetastableConfiguration,
    MetastablePreparationError,
    prepare_metastable_configuration,
    solve_frozen_metastable_jv_sweep,
    solve_frozen_metastable_measurement,
)
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    QuasiFermiSteadyStateError,
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.defects import (
    BulkDefectKinetics,
    ExplicitDefectCapabilityError,
)
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.multivalent_defects import (
    MetastableConversionKinetics,
    MetastableDefectDefinition,
    MetastablePreparationNumerics,
    MetastablePreparationProtocol,
    MultivalentDefectConfiguration,
    MultivalentEnergyLevels,
)
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.physics.metastable_defect_closure import (
    evaluate_metastable_configuration_closure,
)
from perovskite_sim.physics.temperature import thermal_voltage
from perovskite_sim.solver.mol import StateVec, assemble_rhs, build_material_arrays


TEMPERATURE_K = 300.0
GAP_EV = 0.80
NC_M3 = 1.0e24
NV_M3 = 8.0e23
TRANSITION_EV = 0.35


def _kinetics() -> BulkDefectKinetics:
    return BulkDefectKinetics(
        sigma_n_m2=2.0e-19,
        sigma_p_m2=7.0e-20,
        thermal_velocity_n_m_s=1.0e5,
        thermal_velocity_p_m_s=8.0e4,
    )


def _configuration(family: str, charges: tuple[int, ...]):
    return MultivalentDefectConfiguration(
        family=family,
        charge_states_e=charges,
        degeneracy_convention="unity",
        state_degeneracies=(1.0,) * len(charges),
        energy_levels=MultivalentEnergyLevels(
            first_transition_eV_above_vb=0.30,
            correlation_energies_eV=(0.15,),
        ),
        transition_kinetics=(_kinetics(), _kinetics()),
    )


def _definition(*, density_m3: float = 2.0e21) -> MetastableDefectDefinition:
    electron_capture = 0.20
    hole_capture = 0.25
    return MetastableDefectDefinition(
        name="metastable_center",
        total_density_m3=density_m3,
        donor_configuration=_configuration("double_donor", (2, 1, 0)),
        acceptor_configuration=_configuration("double_acceptor", (0, -1, -2)),
        donor_conversion_state_index=1,
        acceptor_conversion_state_index=1,
        conversion_kinetics=MetastableConversionKinetics(
            transition_energy_eV_above_vb=TRANSITION_EV,
            electron_capture_activation_eV=electron_capture,
            electron_emission_activation_eV=(
                electron_capture + 2.0 * (GAP_EV - TRANSITION_EV)
            ),
            hole_capture_activation_eV=hole_capture,
            hole_emission_activation_eV=hole_capture + 2.0 * TRANSITION_EV,
            electron_capture_path="double_electron_capture",
            hole_capture_path="double_hole_capture",
            capture_n_m3_s=1.0e-15,
            capture_p_m3_s=1.0e-15,
            phonon_frequency_Hz=1.0e15,
        ),
    )


def _protocol(
    *,
    preparation_voltage_V: float = 0.0,
    preparation_temperature_K: float = 330.0,
    measurement_temperature_K: float = 300.0,
    tag: bytes = b"d7e3",
) -> MetastablePreparationProtocol:
    return MetastablePreparationProtocol(
        schema_version="solarlab-metastable-preparation-v1",
        preparation_limit="stationary_infinite_time",
        preparation_temperature_K=preparation_temperature_K,
        preparation_voltage_V=preparation_voltage_V,
        preparation_illumination_suns=0.0,
        voltage_continuation_steps=0,
        illumination_continuation_steps=0,
        measurement_temperature_K=measurement_temperature_K,
        configuration_freeze_stage=("after_stationary_preparation_before_measurement"),
        freeze_configuration_during_measurement=True,
        measurement_protocol_sha256=hashlib.sha256(tag).hexdigest(),
        numerics=MetastablePreparationNumerics(
            initial_donor_fraction_guess=0.5,
            max_iterations=80,
            relative_tolerance=1.0e-10,
            clamping_factor=0.5,
            final_unclamped_refinement=True,
        ),
    )


def _params(**overrides) -> MaterialParams:
    intrinsic = math.sqrt(
        NC_M3 * NV_M3 * math.exp(-GAP_EV / thermal_voltage(TEMPERATURE_K))
    )
    base = dict(
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
    )
    base.update(overrides)
    return MaterialParams(**base)


def _stack(**overrides) -> DeviceStack:
    params = _params(**overrides.pop("layer", {}))
    base = dict(
        layers=(LayerSpec("meta_absorber", 300.0e-9, params, "absorber"),),
        V_bi=0.0,
        Phi=0.0,
        interfaces=(),
        mode="legacy",
        built_in_potential_mode="semiconductor_work_function",
    )
    base.update(overrides)
    return DeviceStack(**base)


def _grid(intervals: int = 12) -> np.ndarray:
    return multilayer_grid([Layer(300.0e-9, intervals)])


def _prepare(**overrides) -> FrozenMetastableConfiguration:
    stack = overrides.pop("stack", None) or _stack()
    grid = overrides.pop("grid", None)
    grid = _grid() if grid is None else grid
    return prepare_metastable_configuration(
        grid,
        stack,
        overrides.pop("definition", None) or _definition(),
        overrides.pop("protocol", None) or _protocol(),
        layer_name="meta_absorber",
        **overrides,
    )


def test_preparation_reaches_the_stationary_configuration_fixed_point():
    frozen = _prepare()

    assert frozen.preparation_state.certified
    assert 0.0 < frozen.donor_fraction.min()
    assert frozen.donor_fraction.max() < 1.0
    assert frozen.final_relative_change <= frozen.protocol.numerics.relative_tolerance
    # The protocol forbids accepting a clamped iterate, so the final unclamped
    # step must itself be inside tolerance.
    assert (
        frozen.unclamped_refinement_change
        <= frozen.protocol.numerics.relative_tolerance
    )
    assert frozen.clamped_iterations > 0

    # Fixed point: re-evaluating the conversion closure at the prepared
    # carriers must return the frozen fraction.
    count = frozen.grid_m.size
    mask = frozen.active_nodes
    closure = evaluate_metastable_configuration_closure(
        np.asarray(frozen.preparation_state.y[:count], dtype=float)[mask],
        np.asarray(frozen.preparation_state.y[count : 2 * count], dtype=float)[mask],
        frozen.definition,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=frozen.protocol.preparation_temperature_K,
    )
    np.testing.assert_allclose(
        closure.donor_fraction,
        frozen.donor_fraction,
        rtol=1.0e-9,
        atol=0.0,
    )


def test_prepared_region_never_owns_a_contact_node():
    """Contact neutrality is not re-certified for the metastable inventory."""
    frozen = _prepare()

    assert not frozen.active_nodes[0]
    assert not frozen.active_nodes[-1]
    assert np.any(frozen.active_nodes)


@pytest.mark.slow
def test_preparation_is_deterministic_and_replayable():
    first = _prepare()
    second = _prepare()

    assert first.protocol_sha256 == second.protocol_sha256
    assert first.state_sha256 == second.state_sha256
    assert first.stack_sha256 == second.stack_sha256
    assert first.grid_sha256 == second.grid_sha256
    assert first.model.identity_sha256 == second.model.identity_sha256
    np.testing.assert_array_equal(first.donor_fraction, second.donor_fraction)


@pytest.mark.slow
@pytest.mark.parametrize(
    "protocol_change",
    [
        {"preparation_voltage_V": 0.01},
        {"preparation_temperature_K": 340.0},
        {"tag": b"other-measurement-protocol"},
    ],
)
def test_any_protocol_change_changes_the_frozen_identity(protocol_change):
    baseline = _prepare()
    changed = _prepare(protocol=_protocol(**protocol_change))

    assert changed.protocol_sha256 != baseline.protocol_sha256
    assert changed.model.identity_sha256 != baseline.model.identity_sha256
    if "tag" not in protocol_change:
        # A different working point must also move the physics, not only the
        # provenance hash.
        assert not np.allclose(
            changed.donor_fraction,
            baseline.donor_fraction,
            rtol=1.0e-6,
            atol=0.0,
        )


def test_frozen_configuration_does_not_update_during_a_bias_sweep():
    """The measurement must not silently re-prepare the configuration."""
    frozen = _prepare()
    grid = frozen.grid_m
    stack = _stack()

    states = [
        solve_frozen_metastable_measurement(
            grid,
            stack,
            frozen,
            V_app=voltage,
            illuminated=False,
        )
        for voltage in (0.0, 0.01, 0.02)
    ]

    for state in states:
        assert state.certified
        diagnostics = state.frozen_metastable_diagnostics
        assert diagnostics is not None
        owned = np.asarray(diagnostics.active_nodes, dtype=bool).any(axis=0)
        np.testing.assert_array_equal(
            np.asarray(diagnostics.donor_fraction, dtype=float)[owned],
            frozen.donor_fraction,
        )
        assert diagnostics.model_identity_sha256 == frozen.model.identity_sha256
    # The carriers do move with bias, so the invariance above is a real
    # freeze rather than an unchanged device.
    assert not np.allclose(states[0].y, states[-1].y, rtol=1.0e-9, atol=0.0)


def test_frozen_jv_sweep_keeps_one_configuration_across_every_point():
    frozen = _prepare()
    sweep = solve_frozen_metastable_jv_sweep(
        frozen.grid_m,
        _stack(),
        frozen,
        np.array([0.0, 0.005, 0.01]),
        illumination_steps=(0.0, 1.0e-4, 1.0e-2, 1.0),
        continuity_tolerance_A_m2=2.0e-4,
        current_spread_tolerance_A_m2=2.0e-4,
    )

    assert sweep.certified
    identities = {
        point.frozen_metastable_diagnostics.model_identity_sha256
        for point in sweep.points
    }
    assert identities == {frozen.model.identity_sha256}


def test_measurement_rejects_a_mismatched_grid():
    frozen = _prepare()

    with pytest.raises(MetastablePreparationError, match="grid does not match"):
        solve_frozen_metastable_measurement(
            _grid(16),
            _stack(),
            frozen,
            V_app=0.0,
        )


def test_frozen_metastable_fails_closed_outside_the_prepared_measurement():
    frozen = _prepare()
    grid = frozen.grid_m
    stack = _stack()
    material = build_material_arrays(grid, stack)
    prepared = replace(material, frozen_metastable_defects=frozen.model)

    n = np.geomspace(prepared.n_L, 2.0 * prepared.n_R, grid.size)
    p = prepared.ni_sq / n
    state = StateVec.pack(n, p, prepared.P_ion0.copy())
    with pytest.raises(ExplicitDefectCapabilityError, match="guarded QF/DC"):
        assemble_rhs(0.0, state, grid, stack, prepared, illuminated=False)

    with pytest.raises(
        QuasiFermiSteadyStateError,
        match="interface-plane boundary",
    ):
        solve_quasi_fermi_steady_state(
            grid,
            stack,
            illuminated=False,
            mat=prepared,
            interface_boundary=True,
        )


def test_frozen_metastable_is_exclusive_with_other_explicit_inventories():
    from perovskite_sim.physics.recombination import total_recombination

    frozen = _prepare()
    grid = frozen.grid_m
    material = build_material_arrays(grid, _stack())
    n = np.geomspace(material.n_L, 2.0 * material.n_R, grid.size)
    p = material.ni_sq / n

    from perovskite_sim.models.defects import (
        EXPLICIT_DEFECT_SCHEMA_VERSION,
        EXPLICIT_QUASI_STEADY,
        INTEGRATED_TOTAL,
        NEUTRAL,
        NEUTRAL_ALL_OCCUPANCIES,
        SINGLE_LEVEL,
        BulkDefectDistribution,
        BulkDefectSpecies,
    )

    neutral_stack = _stack(
        layer={
            "defect_schema_version": EXPLICIT_DEFECT_SCHEMA_VERSION,
            "defect_model": EXPLICIT_QUASI_STEADY,
            "bulk_defects": (
                BulkDefectSpecies(
                    name="neutral_level",
                    distribution=BulkDefectDistribution(
                        kind=SINGLE_LEVEL,
                        normalization=INTEGRATED_TOTAL,
                        total_density_m3=1.0e21,
                        center_eV_above_vb=0.39,
                    ),
                    charge_transition=NEUTRAL,
                    neutral_reference=NEUTRAL_ALL_OCCUPANCIES,
                    kinetics=_kinetics(),
                    degeneracy=1.0,
                ),
            ),
        }
    )
    neutral_material = build_material_arrays(grid, neutral_stack)

    with pytest.raises(ValueError, match="exclusive"):
        total_recombination(
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
            neutral_bulk_defects=neutral_material.neutral_bulk_defects,
            frozen_metastable_defects=frozen.model,
        )


def test_frozen_charge_is_the_configuration_weighted_sum():
    """Charge must come from both configurations at the frozen weights."""
    from perovskite_sim.physics.metastable_defect_device import (
        configuration_species,
        evaluate_frozen_metastable_bulk_defects,
    )
    from perovskite_sim.physics.multivalent_defect_closure import (
        evaluate_multivalent_defect_closure,
    )

    frozen = _prepare()
    grid = frozen.grid_m
    material = build_material_arrays(grid, _stack())
    n = np.geomspace(material.n_L, 2.0 * material.n_R, grid.size)
    p = material.ni_sq / n
    evaluation = evaluate_frozen_metastable_bulk_defects(n, p, frozen.model)

    mask = frozen.active_nodes
    donor_species, acceptor_species = configuration_species(frozen.definition)
    weights = (frozen.donor_fraction, 1.0 - frozen.donor_fraction)
    expected = np.zeros(int(np.count_nonzero(mask)), dtype=float)
    for species, weight in zip((donor_species, acceptor_species), weights):
        closure = evaluate_multivalent_defect_closure(
            n[mask],
            p[mask],
            species,
            band_gap_eV=GAP_EV,
            effective_conduction_dos_m3=NC_M3,
            effective_valence_dos_m3=NV_M3,
            temperature_K=frozen.protocol.preparation_temperature_K,
        )
        expected += weight * closure.charge_density_C_m3

    np.testing.assert_array_equal(
        evaluation.total_charge_density_C_m3[mask],
        expected,
    )
    np.testing.assert_array_equal(
        evaluation.total_charge_density_C_m3[~mask],
        np.zeros(int(np.count_nonzero(~mask))),
    )
