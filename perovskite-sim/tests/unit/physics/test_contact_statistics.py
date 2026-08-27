"""Contact thermodynamics for opt-in bulk carrier statistics."""

from __future__ import annotations

from dataclasses import replace

import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.defects import (
    ACCEPTOR,
    DONOR,
    EXPLICIT_DEFECT_SCHEMA_VERSION,
    EXPLICIT_QUASI_STEADY,
    INTEGRATED_TOTAL,
    NEUTRAL,
    NEUTRAL_ALL_OCCUPANCIES,
    NEUTRAL_WHEN_EMPTY,
    NEUTRAL_WHEN_FILLED,
    SINGLE_LEVEL,
    BulkDefectDistribution,
    BulkDefectKinetics,
    BulkDefectSpecies,
)
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.physics.contacts import (
    build_semiconductor_contact_state,
    require_contact_thermodynamic_certificate,
)
from perovskite_sim.physics.statistics import (
    DISCRETE_LEVEL,
    FERMI_DIRAC,
    MAXWELL_BOLTZMANN,
)
from perovskite_sim.solver.mol import (
    DEGENERATE_TRANSPORT_RESEARCH_RECOMBINATION_OFF,
    BulkCarrierStatisticsCapabilityError,
    build_material_arrays,
)


def _silicon(
    *,
    acceptors: float = 0.0,
    donors: float = 0.0,
    statistics: str = MAXWELL_BOLTZMANN,
) -> MaterialParams:
    return MaterialParams(
        eps_r=11.7,
        mu_n=0.135,
        mu_p=0.048,
        D_ion=0.0,
        P_lim=1.0e30,
        P0=0.0,
        ni=1.0e16,
        tau_n=1.0e-6,
        tau_p=1.0e-6,
        n1=1.0e16,
        p1=1.0e16,
        B_rad=1.0e-20,
        C_n=2.8e-43,
        C_p=9.9e-44,
        alpha=0.0,
        N_A=acceptors,
        N_D=donors,
        chi=4.05,
        Eg=1.124,
        Nc300=2.8e25,
        Nv300=1.04e25,
        carrier_statistics=statistics,
    )


def _explicit_species(transition: str) -> BulkDefectSpecies:
    reference = {
        ACCEPTOR: NEUTRAL_WHEN_EMPTY,
        DONOR: NEUTRAL_WHEN_FILLED,
        NEUTRAL: NEUTRAL_ALL_OCCUPANCIES,
    }[transition]
    return BulkDefectSpecies(
        name=f"contact_{transition}",
        distribution=BulkDefectDistribution(
            kind=SINGLE_LEVEL,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=8.0e23,
            center_eV_above_vb=0.55,
        ),
        charge_transition=transition,
        neutral_reference=reference,
        kinetics=BulkDefectKinetics(
            sigma_n_m2=2.0e-19,
            sigma_p_m2=7.0e-20,
            thermal_velocity_n_m_s=1.0e5,
            thermal_velocity_p_m_s=1.0e5,
        ),
        degeneracy=1.0,
    )


def _with_explicit_defect(
    params: MaterialParams,
    transition: str,
) -> MaterialParams:
    return replace(
        params,
        defect_schema_version=EXPLICIT_DEFECT_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=(_explicit_species(transition),),
    )


@pytest.mark.parametrize(
    ("acceptors", "donors"),
    ((0.0, 1.0e27), (8.0e26, 0.0)),
)
def test_degenerate_contact_state_closes_neutrality_and_changes_work_function(
    acceptors,
    donors,
):
    fd_params = _silicon(
        acceptors=acceptors,
        donors=donors,
        statistics=FERMI_DIRAC,
    )
    fd = build_semiconductor_contact_state(
        fd_params,
        temperature_K=300.0,
        use_temperature_scaling=True,
    )
    mb = build_semiconductor_contact_state(
        replace(fd_params, carrier_statistics=MAXWELL_BOLTZMANN),
        temperature_K=300.0,
        use_temperature_scaling=True,
    )

    assert fd.statistics == FERMI_DIRAC
    assert fd.neutrality.normalized_charge_residual < 2.0e-13
    assert abs(fd.work_function_eV - mb.work_function_eV) > 0.05
    if donors > acceptors:
        assert fd.electron_density_m3 == pytest.approx(donors, rel=1.0e-8)
        assert fd.reduced_electron_fermi_level > 1.0
    else:
        assert fd.hole_density_m3 == pytest.approx(acceptors, rel=1.0e-8)
        assert fd.reduced_hole_fermi_level > 1.0


def test_fd_device_work_function_uses_the_same_contact_neutrality_states():
    left = _silicon(acceptors=8.0e26, statistics=FERMI_DIRAC)
    right = _silicon(donors=1.0e27, statistics=FERMI_DIRAC)
    stack = DeviceStack(
        layers=(
            LayerSpec("p_plus", 1.0e-6, left, "HTL"),
            LayerSpec("n_plus", 1.0e-6, right, "ETL"),
        ),
        built_in_potential_mode="semiconductor_work_function",
    )
    left_state = build_semiconductor_contact_state(
        left,
        temperature_K=300.0,
        use_temperature_scaling=True,
    )
    right_state = build_semiconductor_contact_state(
        right,
        temperature_K=300.0,
        use_temperature_scaling=True,
    )

    assert stack.compute_semiconductor_V_bi() == pytest.approx(
        left_state.work_function_eV - right_state.work_function_eV,
        abs=2.0e-14,
    )


@pytest.mark.parametrize("transition", [ACCEPTOR, DONOR])
def test_charged_explicit_defect_contact_state_uses_shared_local_closure(
    transition,
):
    baseline_params = (
        _silicon(donors=2.0e22)
        if transition == ACCEPTOR
        else _silicon(acceptors=2.0e22)
    )
    params = _with_explicit_defect(baseline_params, transition)

    contact = build_semiconductor_contact_state(
        params,
        temperature_K=300.0,
        use_temperature_scaling=True,
    )
    baseline = build_semiconductor_contact_state(
        baseline_params,
        temperature_K=300.0,
        use_temperature_scaling=True,
    )
    closure = contact.explicit_defect_closure

    assert closure is not None
    assert contact.neutrality.normalized_charge_residual < 1.0e-12
    assert closure.charge_transitions == (transition,)
    assert 0.0 < closure.occupancy.item() < 1.0
    assert contact.work_function_eV != pytest.approx(
        baseline.work_function_eV,
        abs=1.0e-6,
    )


def test_neutral_explicit_defect_is_inert_for_contact_thermodynamics():
    baseline_params = _silicon(donors=2.0e22)
    params = _with_explicit_defect(baseline_params, NEUTRAL)

    contact = build_semiconductor_contact_state(
        params,
        temperature_K=300.0,
        use_temperature_scaling=True,
    )
    baseline = build_semiconductor_contact_state(
        baseline_params,
        temperature_K=300.0,
        use_temperature_scaling=True,
    )

    assert contact.explicit_defect_closure is None
    assert contact.work_function_eV == baseline.work_function_eV
    assert contact.electron_density_m3 == baseline.electron_density_m3
    assert contact.hole_density_m3 == baseline.hole_density_m3


def test_charged_defect_device_work_function_uses_same_contact_closures():
    left = _with_explicit_defect(_silicon(acceptors=2.0e22), ACCEPTOR)
    right = _with_explicit_defect(_silicon(donors=2.0e22), DONOR)
    stack = DeviceStack(
        layers=(
            LayerSpec("p", 100.0e-9, left, "HTL"),
            LayerSpec("n", 100.0e-9, right, "ETL"),
        ),
        built_in_potential_mode="semiconductor_work_function",
    )
    left_state = build_semiconductor_contact_state(
        left,
        temperature_K=300.0,
        use_temperature_scaling=True,
    )
    right_state = build_semiconductor_contact_state(
        right,
        temperature_K=300.0,
        use_temperature_scaling=True,
    )

    assert stack.compute_semiconductor_V_bi() == pytest.approx(
        left_state.work_function_eV - right_state.work_function_eV,
        abs=2.0e-14,
    )


def test_discrete_donor_contact_state_resolves_temperature_freeze_out():
    params = replace(
        _silicon(donors=1.0e23, statistics=FERMI_DIRAC),
        dopant_ionization_model=DISCRETE_LEVEL,
        donor_binding_energy_eV=0.045,
    )
    cold = build_semiconductor_contact_state(
        params,
        temperature_K=100.0,
        use_temperature_scaling=True,
    )
    warm = build_semiconductor_contact_state(
        params,
        temperature_K=300.0,
        use_temperature_scaling=True,
    )

    assert cold.neutrality.dopant_ionization_model == DISCRETE_LEVEL
    assert cold.neutrality.normalized_charge_residual < 2.0e-13
    assert warm.neutrality.normalized_charge_residual < 2.0e-13
    assert (
        0.0
        < cold.neutrality.donor_ionized_fraction
        < warm.neutrality.donor_ionized_fraction
        < 1.0
    )
    assert cold.work_function_eV < warm.work_function_eV


def test_incomplete_ionization_device_work_function_uses_contact_closure():
    left = replace(
        _silicon(acceptors=1.0e23, statistics=FERMI_DIRAC),
        dopant_ionization_model=DISCRETE_LEVEL,
        acceptor_binding_energy_eV=0.045,
    )
    right = replace(
        _silicon(donors=1.0e23, statistics=FERMI_DIRAC),
        dopant_ionization_model=DISCRETE_LEVEL,
        donor_binding_energy_eV=0.045,
    )
    stack = DeviceStack(
        layers=(
            LayerSpec("p", 1.0e-6, left, "HTL"),
            LayerSpec("n", 1.0e-6, right, "ETL"),
        ),
        T=150.0,
        mode="full",
        built_in_potential_mode="semiconductor_work_function",
    )
    left_state = build_semiconductor_contact_state(
        left,
        temperature_K=150.0,
        use_temperature_scaling=True,
    )
    right_state = build_semiconductor_contact_state(
        right,
        temperature_K=150.0,
        use_temperature_scaling=True,
    )

    assert stack.compute_semiconductor_V_bi() == pytest.approx(
        left_state.work_function_eV - right_state.work_function_eV,
        abs=2.0e-14,
    )


def test_slotboom_contact_state_uses_narrowed_band_edges():
    baseline = _silicon(donors=3.0e25, statistics=FERMI_DIRAC)
    narrowed_params = replace(
        baseline,
        band_gap_narrowing_model="slotboom",
        bgn_conduction_band_fraction=0.4,
    )
    narrowed = build_semiconductor_contact_state(
        narrowed_params,
        temperature_K=300.0,
        use_temperature_scaling=True,
    )
    reference = build_semiconductor_contact_state(
        baseline,
        temperature_K=300.0,
        use_temperature_scaling=True,
    )

    assert narrowed.band_gap_narrowing_eV > 0.0
    assert narrowed.band_gap_eV == pytest.approx(
        baseline.Eg - narrowed.band_gap_narrowing_eV
    )
    assert narrowed.electron_affinity_eV == pytest.approx(
        baseline.chi + 0.4 * narrowed.band_gap_narrowing_eV
    )
    assert narrowed.conduction_band_shift_eV == pytest.approx(
        0.4 * narrowed.band_gap_narrowing_eV
    )
    assert narrowed.valence_band_shift_eV == pytest.approx(
        0.6 * narrowed.band_gap_narrowing_eV
    )
    assert narrowed.neutrality.normalized_charge_residual < 2.0e-13
    assert narrowed.work_function_eV != pytest.approx(reference.work_function_eV)


def test_slotboom_device_work_function_uses_the_same_contact_states():
    left = replace(
        _silicon(acceptors=3.0e25, statistics=FERMI_DIRAC),
        band_gap_narrowing_model="slotboom",
    )
    right = replace(
        _silicon(donors=3.0e25, statistics=FERMI_DIRAC),
        band_gap_narrowing_model="slotboom",
    )
    stack = DeviceStack(
        layers=(
            LayerSpec("p_plus", 100.0e-9, left, "HTL"),
            LayerSpec("n_plus", 100.0e-9, right, "ETL"),
        ),
        built_in_potential_mode="semiconductor_work_function",
    )
    left_state = build_semiconductor_contact_state(
        left,
        temperature_K=300.0,
        use_temperature_scaling=True,
    )
    right_state = build_semiconductor_contact_state(
        right,
        temperature_K=300.0,
        use_temperature_scaling=True,
    )

    assert stack.compute_semiconductor_V_bi() == pytest.approx(
        left_state.work_function_eV - right_state.work_function_eV,
        abs=2.0e-14,
    )


def test_slotboom_requires_explicit_thermodynamic_contact_mode():
    params = replace(
        _silicon(donors=3.0e23),
        band_gap_narrowing_model="slotboom",
    )

    with pytest.raises(
        ValueError,
        match="band-gap narrowing requires explicit",
    ):
        DeviceStack(
            layers=(LayerSpec("narrowed", 100.0e-9, params, "ETL"),),
        )


@pytest.mark.parametrize(
    "mode_fields",
    (
        {},
        {"built_in_potential_mode": "legacy_manual"},
        {
            "built_in_potential_mode": "metal_work_function",
            "work_function_left_eV": 5.0,
            "work_function_right_eV": 4.0,
        },
    ),
)
def test_fd_stack_rejects_nonphysical_contact_potential_modes(mode_fields):
    params = _silicon(donors=1.0e27, statistics=FERMI_DIRAC)
    with pytest.raises(ValueError, match="explicit built_in_potential_mode"):
        DeviceStack(
            layers=(LayerSpec("n_plus", 1.0e-6, params, "ETL"),),
            **mode_fields,
        )


def test_bulk_solver_fails_closed_until_fd_transport_is_enabled():
    params = _silicon(donors=1.0e27, statistics=FERMI_DIRAC)
    stack = DeviceStack(
        layers=(LayerSpec("n_plus", 1.0e-6, params, "ETL"),),
        built_in_potential_mode="semiconductor_work_function",
    )
    grid = multilayer_grid((Layer(1.0e-6, N=8),))

    with pytest.raises(
        BulkCarrierStatisticsCapabilityError,
        match="bulk Fermi-Dirac transport closure is not enabled",
    ):
        build_material_arrays(grid, stack)


def test_fd_contact_certificate_uses_the_fd_inverse_density_law():
    left = _silicon(acceptors=3.0e25, statistics=FERMI_DIRAC)
    right = _silicon(donors=3.0e25, statistics=FERMI_DIRAC)
    stack = DeviceStack(
        layers=(
            LayerSpec("p_plus", 100.0e-9, left, "HTL"),
            LayerSpec("n_plus", 100.0e-9, right, "ETL"),
        ),
        Phi=0.0,
        interfaces=((0.0, 0.0),),
        built_in_potential_mode="semiconductor_work_function",
    )
    grid = multilayer_grid(
        (Layer(100.0e-9, N=20), Layer(100.0e-9, N=20)),
        alpha=3.0,
    )
    material = build_material_arrays(
        grid,
        stack,
        carrier_statistics_transport=(
            DEGENERATE_TRANSPORT_RESEARCH_RECOMBINATION_OFF
        ),
    )

    certificate = require_contact_thermodynamic_certificate(stack, material)

    assert certificate.certified
    assert certificate.fermi_level_span_eV < 1.0e-12


@pytest.mark.parametrize(
    "updates",
    (
        {"Nc300": None},
        {"Nv300": 0.0},
        {"Eg": -1.0},
        {"N_D": -1.0},
    ),
)
def test_fd_material_requires_complete_physical_inputs(updates):
    base = _silicon(donors=1.0e27)
    with pytest.raises(ValueError, match="fermi_dirac carrier statistics"):
        replace(base, carrier_statistics=FERMI_DIRAC, **updates)
