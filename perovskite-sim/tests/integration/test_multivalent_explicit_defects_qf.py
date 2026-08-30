"""D7-E1 multivalent explicit-defect closure on the certified QF/DC path."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.bulk_defect_transient import (
    BulkDefectTransientError,
    run_bulk_defect_device_transient,
)
from perovskite_sim.experiments.defect_aware_impedance import (
    BulkDefectDeviceACError,
    run_bulk_defect_device_impedance,
)
from perovskite_sim.experiments.quasi_fermi_impedance import (
    run_quasi_fermi_impedance,
)
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
    NEUTRAL,
    NEUTRAL_ALL_OCCUPANCIES,
    NEUTRAL_WHEN_EMPTY,
    NEUTRAL_WHEN_FILLED,
    SINGLE_LEVEL,
    BulkDefectDistribution,
    BulkDefectKinetics,
    BulkDefectSpecies,
    ExplicitDefectCapabilityError,
)
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.multivalent_defects import (
    MULTIVALENT_DEFECT_SCHEMA_VERSION,
    MultivalentBulkDefectSpecies,
    MultivalentDefectConfiguration,
    MultivalentEnergyLevels,
)
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.physics.multivalent_defect_device import (
    evaluate_multivalent_bulk_defects,
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
_FAMILY_CHARGES = {
    "single_donor": (1, 0),
    "single_acceptor": (0, -1),
    "double_donor": (2, 1, 0),
    "double_acceptor": (0, -1, -2),
    "amphoteric": (1, 0, -1),
}


def _kinetics(scale: float = 1.0) -> BulkDefectKinetics:
    return BulkDefectKinetics(
        sigma_n_m2=2.0e-19 * scale,
        sigma_p_m2=7.0e-20 * scale,
        thermal_velocity_n_m_s=1.0e5,
        thermal_velocity_p_m_s=8.0e4,
    )


def _multivalent_species(
    family: str = "double_donor",
    *,
    density_m3: float = 2.0e21,
    degeneracy_convention: str = "unity",
    name: str | None = None,
) -> MultivalentBulkDefectSpecies:
    charges = _FAMILY_CHARGES[family]
    transition_count = len(charges) - 1
    degeneracies = (
        tuple(
            float(math.comb(len(charges) - 1, index)) for index in range(len(charges))
        )
        if degeneracy_convention == "scaps_binomial"
        else (1.0,) * len(charges)
    )
    return MultivalentBulkDefectSpecies(
        name=name or f"{family}_bulk",
        total_density_m3=density_m3,
        configuration=MultivalentDefectConfiguration(
            family=family,
            charge_states_e=charges,
            degeneracy_convention=degeneracy_convention,
            state_degeneracies=degeneracies,
            energy_levels=MultivalentEnergyLevels(
                first_transition_eV_above_vb=(0.39 if transition_count == 1 else 0.30),
                correlation_energies_eV=(0.15,) * (transition_count - 1),
            ),
            transition_kinetics=tuple(
                _kinetics(1.0 if index == 0 else 0.5)
                for index in range(transition_count)
            ),
        ),
    )


def _monovalent_species(transition: str) -> BulkDefectSpecies:
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
        kinetics=_kinetics(),
        degeneracy=1.0,
    )


def _params(
    *,
    alpha: float = 4.0e5,
    N_A: float = 0.0,
    N_D: float = 0.0,
    **defect,
) -> MaterialParams:
    intrinsic = math.sqrt(
        NC_M3 * NV_M3 * math.exp(-GAP_EV / thermal_voltage(TEMPERATURE_K))
    )
    return MaterialParams(
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
        alpha=alpha,
        N_A=N_A,
        N_D=N_D,
        chi=4.0,
        Eg=GAP_EV,
        Nc300=NC_M3,
        Nv300=NV_M3,
        **defect,
    )


def _stack(
    family: str = "double_donor",
    *,
    photon_flux_m2_s: float = 0.0,
    contact_mode: str = "semiconductor_work_function",
    species: tuple[MultivalentBulkDefectSpecies, ...] | None = None,
    mode: str = "legacy",
    stack_fields: dict | None = None,
    **layer_overrides,
) -> DeviceStack:
    params = _params(
        defect_schema_version=MULTIVALENT_DEFECT_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=((_multivalent_species(family),) if species is None else species),
        **layer_overrides,
    )
    return DeviceStack(
        layers=(LayerSpec("defective", 300.0e-9, params, "absorber"),),
        V_bi=0.0,
        Phi=photon_flux_m2_s,
        interfaces=(),
        mode=mode,
        built_in_potential_mode=contact_mode,
        **(stack_fields or {}),
    )


def _mixed_stack(*, photon_flux_m2_s: float = 0.0) -> DeviceStack:
    multivalent = _params(
        defect_schema_version=MULTIVALENT_DEFECT_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=(_multivalent_species(),),
    )
    monovalent = _params(
        alpha=0.0,
        defect_schema_version=EXPLICIT_DEFECT_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=(_monovalent_species(ACCEPTOR),),
    )
    return DeviceStack(
        layers=(
            LayerSpec("multivalent_absorber", 200.0e-9, multivalent, "absorber"),
            LayerSpec("monovalent_transport", 100.0e-9, monovalent, "ETL"),
        ),
        V_bi=0.0,
        Phi=photon_flux_m2_s,
        interfaces=((0.0, 0.0),),
        mode="legacy",
        built_in_potential_mode="semiconductor_work_function",
    )


def _grid(stack: DeviceStack, intervals: int = 12) -> np.ndarray:
    return multilayer_grid(
        [Layer(layer.thickness, intervals) for layer in stack.layers]
    )


def test_default_material_and_wrong_contact_mode_fail_closed():
    stack = _stack()
    grid = _grid(stack, 6)

    with pytest.raises(ExplicitDefectCapabilityError, match="multivalent"):
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


def test_qf_material_compiles_one_shared_multivalent_model():
    stack = _stack()
    grid = _grid(stack, 6)
    material = build_material_arrays(
        grid,
        stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
    )
    model = material.multivalent_bulk_defects
    assert model is not None
    assert material.monovalent_bulk_defects is None
    assert material.neutral_bulk_defects is None
    assert model.species_identifiers == ("layer[0]/defective/double_donor_bulk",)
    assert model.state_counts == (3,)

    n = np.geomspace(material.n_L, 2.0 * material.n_R, grid.size)
    p = material.ni_sq / n
    evaluation = evaluate_multivalent_bulk_defects(n, p, model)
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
        multivalent_bulk_defects=model,
    )
    np.testing.assert_array_equal(
        recombination,
        evaluation.total_recombination_rate_m3_s,
    )
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
        multivalent_bulk_defects=model,
    )
    assert scalar == pytest.approx(
        evaluation.total_recombination_rate_m3_s[node],
        rel=2.0e-15,
    )
    # A double donor in near-intrinsic material carries positive charge.
    assert np.all(evaluation.total_charge_density_C_m3 > 0.0)

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


def test_multivalent_bulk_charge_fixed_qf_tangent_matches_centered_difference():
    stack = _stack()
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
    rho_plus, _ = system._bulk_space_charge_and_tangent(n * factor, p / factor)
    rho_minus, _ = system._bulk_space_charge_and_tangent(n / factor, p * factor)

    np.testing.assert_allclose(
        tangent,
        (rho_plus - rho_minus) / (2.0 * step),
        rtol=3.0e-8,
        atol=1.0e-10,
    )


@pytest.mark.parametrize(
    "family",
    ["double_donor", "double_acceptor", "amphoteric"],
)
def test_multivalent_dark_equilibrium_is_qf_residual_certified(family):
    stack = _stack(family)
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
    diagnostics = result.multivalent_bulk_defect_diagnostics
    assert diagnostics is not None
    assert result.bulk_defect_diagnostics is None
    assert diagnostics.species_identifiers == (f"layer[0]/defective/{family}_bulk",)
    assert diagnostics.state_counts == (3,)
    assert diagnostics.minimum_state_probability >= 0.0
    assert diagnostics.maximum_state_probability <= 1.0
    assert diagnostics.maximum_probability_sum_error <= 1.0e-12
    assert diagnostics.minimum_transition_rate_s1 > 0.0
    charge = diagnostics.total_charge_density_C_m3[diagnostics.active_nodes.any(axis=0)]
    if family == "double_donor":
        assert np.all(charge > 0.0)
    elif family == "double_acceptor":
        assert np.all(charge < 0.0)


@pytest.mark.parametrize(
    ("family", "transition"),
    [("single_donor", DONOR), ("single_acceptor", ACCEPTOR)],
)
def test_single_transition_v4_device_recovers_the_monovalent_qf_dc_route(
    family,
    transition,
):
    multivalent_stack = _stack(family, photon_flux_m2_s=2.0e16)
    monovalent_params = _params(
        defect_schema_version=EXPLICIT_DEFECT_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=(_monovalent_species(transition),),
    )
    monovalent_stack = replace(
        multivalent_stack,
        layers=(replace(multivalent_stack.layers[0], params=monovalent_params),),
    )
    grid = _grid(multivalent_stack)
    controls = dict(
        V_app=0.01,
        illuminated=True,
        illumination_steps=(0.0, 1.0e-4, 1.0e-2, 1.0),
        continuity_tolerance_A_m2=2.0e-4,
        current_spread_tolerance_A_m2=2.0e-4,
    )

    multivalent = solve_quasi_fermi_steady_state(
        grid,
        multivalent_stack,
        **controls,
    )
    monovalent = solve_quasi_fermi_steady_state(
        grid,
        monovalent_stack,
        **controls,
    )

    # These are two INDEPENDENT Newton solves, each certified only to a ~1e-10
    # normalized residual, so the equivalence pin is deliberately looser than
    # the measured agreement (2.3e-14 relative on this stack). It is still
    # four orders tighter than any genuine closure divergence: a v4 lane that
    # dropped the shared master equation for independent SRH centres, or that
    # lost the charge/tangent in one consumer, moves these by O(1).
    assert multivalent.certified and monovalent.certified
    assert multivalent.current_A_m2 == pytest.approx(
        monovalent.current_A_m2,
        rel=1.0e-10,
    )
    np.testing.assert_allclose(
        multivalent.y,
        monovalent.y,
        rtol=1.0e-10,
        atol=0.0,
    )
    np.testing.assert_allclose(
        multivalent.phi,
        monovalent.phi,
        rtol=0.0,
        atol=1.0e-13,
    )


def test_graded_v4_layer_fails_closed_before_compilation():
    """Contract rule: only uniform v4 layers are certified at D7-E1."""
    graded = _stack(
        mode="full",
        stack_fields={"band_grading": True},
        Eg_back=0.86,
        chi_back=4.02,
    )

    with pytest.raises(
        ExplicitDefectCapabilityError,
        match="requires uniform finite positive",
    ):
        build_material_arrays(
            _grid(graded, 6),
            graded,
            explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
        )
    with pytest.raises(ExplicitDefectCapabilityError):
        solve_quasi_fermi_steady_state(
            _grid(graded, 6),
            graded,
            illuminated=False,
        )


def test_scaps_binomial_degeneracy_propagates_through_the_device_lane():
    """State degeneracies must reach contact neutrality and the QF solve.

    A degeneracy-blind reimplementation of the closure would leave the
    binomial and unity solves identical; the master equation carries the
    adjacent-state degeneracy ratio into every emission rate, so they must
    differ materially while both stay certified.
    """
    unity = _stack("amphoteric")
    binomial = _stack(
        "amphoteric",
        species=(
            _multivalent_species(
                "amphoteric",
                degeneracy_convention="scaps_binomial",
            ),
        ),
    )
    grid = _grid(unity, 8)

    unity_result = solve_quasi_fermi_steady_state(grid, unity, illuminated=False)
    binomial_result = solve_quasi_fermi_steady_state(
        grid,
        binomial,
        illuminated=False,
    )

    assert unity_result.certified and binomial_result.certified
    assert binomial_result.contact_thermodynamic_status == "certified"
    binomial_diagnostics = binomial_result.multivalent_bulk_defect_diagnostics
    assert binomial_diagnostics is not None
    assert binomial_diagnostics.maximum_probability_sum_error <= 1.0e-12
    assert binomial_diagnostics.minimum_transition_rate_s1 > 0.0
    unity_charge = np.asarray(
        unity_result.multivalent_bulk_defect_diagnostics.total_charge_density_C_m3
    )
    binomial_charge = np.asarray(binomial_diagnostics.total_charge_density_C_m3)
    interior = slice(1, -1)
    assert np.max(np.abs(binomial_charge[interior] - unity_charge[interior])) > 0.0
    # The contact reservoirs are solved from the same closure, so a changed
    # degeneracy convention must move them too.
    assert not np.isclose(
        binomial_result.y[0],
        unity_result.y[0],
        rtol=1.0e-9,
        atol=0.0,
    )


def test_multi_species_v4_layer_is_certified_through_the_qf_lane():
    """Two physical multivalent defects in one layer share the device lane."""
    stack = _stack(
        photon_flux_m2_s=2.0e16,
        species=(
            _multivalent_species("double_donor", name="donor_defect"),
            _multivalent_species(
                "double_acceptor",
                density_m3=1.5e21,
                name="acceptor_defect",
            ),
        ),
    )
    grid = _grid(stack)
    material = build_material_arrays(
        grid,
        stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
    )
    assert material.multivalent_bulk_defects is not None
    assert material.multivalent_bulk_defects.state_counts == (3, 3)

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
    rho_plus, _ = system._bulk_space_charge_and_tangent(n * factor, p / factor)
    rho_minus, _ = system._bulk_space_charge_and_tangent(n / factor, p * factor)
    np.testing.assert_allclose(
        tangent,
        (rho_plus - rho_minus) / (2.0 * step),
        rtol=3.0e-8,
        atol=1.0e-10,
    )

    sweep = solve_quasi_fermi_jv_sweep(
        grid,
        stack,
        np.array([0.0, 0.005]),
        illumination_steps=(0.0, 1.0e-4, 1.0e-2, 1.0),
        continuity_tolerance_A_m2=2.0e-4,
        current_spread_tolerance_A_m2=2.0e-4,
    )
    assert sweep.certified
    assert sweep.multivalent_species_identifiers == (
        "layer[0]/defective/donor_defect",
        "layer[0]/defective/acceptor_defect",
    )
    assert sweep.multivalent_state_counts == (3, 3)
    diagnostics = sweep.points[0].multivalent_bulk_defect_diagnostics
    assert diagnostics is not None
    assert diagnostics.charge_density_C_m3.shape == (2, grid.size)
    np.testing.assert_allclose(
        diagnostics.total_charge_density_C_m3,
        np.sum(diagnostics.charge_density_C_m3, axis=0),
        rtol=0.0,
        atol=0.0,
    )


def test_doped_and_doping_profiled_v4_layers_are_certified():
    """Dopant compensation and the per-doping-pair neutrality seed loop."""
    uniform = _stack(N_A=3.0e21)
    grid = _grid(uniform, 8)
    uniform_result = solve_quasi_fermi_steady_state(
        grid,
        uniform,
        illuminated=False,
    )
    assert uniform_result.certified
    assert uniform_result.contact_thermodynamic_status == "certified"

    profiled = _stack(
        N_A=1.0e21,
        N_A_bulk=1.0e20,
        doping_profile_shape="gaussian",
        doping_decay_length=60.0e-9,
        doping_edge="front",
    )
    profiled_material = build_material_arrays(
        grid,
        profiled,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
    )
    # The seed groups nodes by unique (N_A, N_D); a graded dopant profile is
    # what makes that loop run more than once inside a v4 region.
    assert np.unique(profiled_material.N_A).size > 1
    profiled_result = solve_quasi_fermi_steady_state(
        grid,
        profiled,
        illuminated=False,
        mat=profiled_material,
    )
    assert profiled_result.certified
    assert profiled_result.multivalent_bulk_defect_diagnostics is not None


def test_neutral_v1_and_multivalent_v4_layers_share_one_certified_solve():
    """A v4 layer beside a neutral v1 layer compiles both inventories.

    This partition cannot occur before D7-E1: under the old qf_dc closure a
    neutral layer was folded into the monovalent model, so the neutral branch
    of the mixed multivalent recombination dispatch is only reachable here.
    """
    multivalent = _params(
        defect_schema_version=MULTIVALENT_DEFECT_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=(_multivalent_species(),),
    )
    neutral = _params(
        alpha=0.0,
        defect_schema_version=EXPLICIT_DEFECT_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=(
            replace(
                _monovalent_species(ACCEPTOR),
                name="neutral_level",
                charge_transition=NEUTRAL,
                neutral_reference=NEUTRAL_ALL_OCCUPANCIES,
            ),
        ),
    )
    stack = DeviceStack(
        layers=(
            LayerSpec("multivalent_absorber", 200.0e-9, multivalent, "absorber"),
            LayerSpec("neutral_transport", 100.0e-9, neutral, "ETL"),
        ),
        V_bi=0.0,
        Phi=0.0,
        interfaces=((0.0, 0.0),),
        mode="legacy",
        built_in_potential_mode="semiconductor_work_function",
    )
    grid = _grid(stack, 6)
    material = build_material_arrays(
        grid,
        stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
    )
    assert material.neutral_bulk_defects is not None
    assert material.monovalent_bulk_defects is None
    assert material.multivalent_bulk_defects is not None
    assert not np.any(
        np.logical_and(
            material.neutral_bulk_defects.explicit_node_mask,
            material.multivalent_bulk_defects.explicit_node_mask,
        )
    )

    n = np.geomspace(material.n_L, 2.0 * material.n_R, grid.size)
    p = material.ni_sq / n
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
        neutral_bulk_defects=material.neutral_bulk_defects,
        multivalent_bulk_defects=material.multivalent_bulk_defects,
    )
    multivalent_only = evaluate_multivalent_bulk_defects(
        n,
        p,
        material.multivalent_bulk_defects,
    )
    multivalent_mask = material.multivalent_bulk_defects.explicit_node_mask
    np.testing.assert_array_equal(
        recombination[multivalent_mask],
        multivalent_only.total_recombination_rate_m3_s[multivalent_mask],
    )
    assert np.all(np.isfinite(recombination))

    result = solve_quasi_fermi_steady_state(grid, stack, illuminated=False)
    assert result.certified
    assert result.multivalent_bulk_defect_diagnostics is not None
    assert result.bulk_defect_diagnostics is None


def test_monovalent_qf_dc_material_carries_no_multivalent_model():
    """The converse guard: a v1-v3 qf_dc build must leave the new cache None."""
    monovalent_params = _params(
        defect_schema_version=EXPLICIT_DEFECT_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=(_monovalent_species(ACCEPTOR),),
    )
    stack = replace(
        _stack(),
        layers=(replace(_stack().layers[0], params=monovalent_params),),
    )
    grid = _grid(stack, 6)
    material = build_material_arrays(
        grid,
        stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
    )

    assert material.monovalent_bulk_defects is not None
    assert material.multivalent_bulk_defects is None
    result = solve_quasi_fermi_steady_state(grid, stack, illuminated=False)
    assert result.certified
    assert result.multivalent_bulk_defect_diagnostics is None
    assert result.bulk_defect_diagnostics is not None


def test_neutral_and_monovalent_models_stay_exclusive_under_multivalent():
    """The exclusivity invariant must not be dropped by the v4 dispatch."""
    grid = _grid(_stack(), 6)
    multivalent_material = build_material_arrays(
        grid,
        _stack(),
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
    )
    monovalent_stack = replace(
        _stack(),
        layers=(
            replace(
                _stack().layers[0],
                params=_params(
                    defect_schema_version=EXPLICIT_DEFECT_SCHEMA_VERSION,
                    defect_model=EXPLICIT_QUASI_STEADY,
                    bulk_defects=(_monovalent_species(ACCEPTOR),),
                ),
            ),
        ),
    )
    monovalent_material = build_material_arrays(
        grid,
        monovalent_stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
    )
    neutral_stack = replace(
        _stack(),
        layers=(
            replace(
                _stack().layers[0],
                params=_params(
                    defect_schema_version=EXPLICIT_DEFECT_SCHEMA_VERSION,
                    defect_model=EXPLICIT_QUASI_STEADY,
                    bulk_defects=(
                        replace(
                            _monovalent_species(ACCEPTOR),
                            name="neutral_level",
                            charge_transition=NEUTRAL,
                            neutral_reference=NEUTRAL_ALL_OCCUPANCIES,
                        ),
                    ),
                ),
            ),
        ),
    )
    neutral_material = build_material_arrays(grid, neutral_stack)
    n = np.geomspace(
        multivalent_material.n_L,
        2.0 * multivalent_material.n_R,
        grid.size,
    )
    p = multivalent_material.ni_sq / n

    with pytest.raises(ValueError, match="exclusive"):
        total_recombination(
            n,
            p,
            multivalent_material.ni_sq,
            multivalent_material.tau_n,
            multivalent_material.tau_p,
            multivalent_material.n1,
            multivalent_material.p1,
            multivalent_material.B_rad,
            multivalent_material.C_n,
            multivalent_material.C_p,
            neutral_bulk_defects=neutral_material.neutral_bulk_defects,
            monovalent_bulk_defects=(monovalent_material.monovalent_bulk_defects),
            multivalent_bulk_defects=(multivalent_material.multivalent_bulk_defects),
        )


def test_contact_reservoir_overrides_and_blended_recombination_fail_closed():
    """Rule (1): every consumer must use the one master-equation closure.

    Each of these composes a pre-existing feature that would give the same
    physical defect two different carrier states — a contact reservoir not
    solved from the closure, a recombination density different from the
    Poisson one, or a discarded recombination with a retained charge.
    """
    flat_band = _stack(
        mode="full",
        stack_fields={"flat_band_metal_contacts": True},
        N_A=1.0e21,
    )
    with pytest.raises(
        ExplicitDefectCapabilityError,
        match="flat_band_metal_contacts",
    ):
        build_material_arrays(
            _grid(flat_band, 6),
            flat_band,
            explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
        )

    despiked = _stack(stack_fields={"het_recomb_despike": 0.5})
    with pytest.raises(
        ExplicitDefectCapabilityError,
        match="het_recomb_despike",
    ):
        build_material_arrays(
            _grid(despiked, 6),
            despiked,
            explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
        )

    stack = _stack()
    with pytest.raises(
        ExplicitDefectCapabilityError,
        match="research_recombination_off",
    ):
        build_material_arrays(
            _grid(stack, 6),
            stack,
            explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
            carrier_statistics_transport="research_recombination_off",
        )


def test_affinity_graded_v4_layer_fails_closed():
    """The uniform-layer contract covers chi grading, not only Eg grading."""
    graded = _stack(
        mode="full",
        stack_fields={"band_grading": True},
        chi_back=4.05,
    )

    with pytest.raises(
        ExplicitDefectCapabilityError,
        match="uniform finite electron affinity",
    ):
        build_material_arrays(
            _grid(graded, 6),
            graded,
            explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
        )


def test_structured_dae_and_analytic_reaction_lanes_fail_closed():
    """Rule (2): lanes that forward only the neutral inventory must refuse v4.

    These builders substitute the effective-lifetime SRH law for anything they
    do not forward, so silently accepting a compiled v4 model would run
    different physics rather than the declared closure.
    """
    from perovskite_sim.solver.dae_jacobian import (
        DAEStructuredJacobianCapabilityError,
        require_neutral_only_defect_inventory,
    )

    stack = _stack()
    grid = _grid(stack, 6)
    material = build_material_arrays(
        grid,
        stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
    )

    with pytest.raises(
        DAEStructuredJacobianCapabilityError,
        match="multivalent bulk defects",
    ):
        require_neutral_only_defect_inventory(material)

    from perovskite_sim.experiments.ion_aware_analytic_reaction import (
        IonAwareAnalyticReactionCapabilityError,
        _node_recombination_rate,
    )

    with pytest.raises(
        IonAwareAnalyticReactionCapabilityError,
        match="multivalent bulk defects",
    ):
        _node_recombination_rate(material, 3, 1.0e18, 1.0e14)


def test_multivalent_state_probabilities_are_undefined_off_region():
    """A normalized distribution has no zero; off-region columns are NaN."""
    stack = _mixed_stack()
    grid = _grid(stack, 6)
    result = solve_quasi_fermi_steady_state(grid, stack, illuminated=False)

    diagnostics = result.multivalent_bulk_defect_diagnostics
    assert diagnostics is not None
    assert result.certified
    for index, probabilities in enumerate(diagnostics.state_probability):
        owned = diagnostics.active_nodes[index]
        assert np.any(owned) and not np.all(owned)
        np.testing.assert_allclose(
            np.sum(probabilities[:, owned], axis=0),
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        )
        assert np.all(np.isnan(probabilities[:, ~owned]))


def test_quasi_fermi_impedance_fails_closed_for_multivalent():
    stack = _stack()
    grid = _grid(stack, 6)
    material = build_material_arrays(
        grid,
        stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
    )

    with pytest.raises(
        (QuasiFermiSteadyStateError, ExplicitDefectCapabilityError),
        match="multivalent",
    ):
        run_quasi_fermi_impedance(
            grid,
            stack,
            np.geomspace(1.0e1, 1.0e5, 5),
            mat=material,
        )


def test_multivalent_illuminated_biased_state_is_qf_residual_certified():
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
    assert result.multivalent_bulk_defect_diagnostics is not None
    assert result.contact_thermodynamic_status == "certified"
    assert abs(result.current_A_m2) > 0.0
    assert result.max_normalized_cell_residual < 1.0e-10
    assert result.poisson_residual < 1.0e-8


def test_public_qf_jv_sweep_retains_multivalent_certificates():
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
    assert sweep.multivalent_species_identifiers == (
        "layer[0]/defective/double_donor_bulk",
    )
    assert sweep.multivalent_state_counts == (3,)
    assert all(
        point.multivalent_bulk_defect_diagnostics is not None for point in sweep.points
    )
    assert all(
        point.contact_thermodynamic_status == "certified" for point in sweep.points
    )


def test_mixed_monovalent_and_multivalent_layers_share_one_qf_dc_residual():
    stack = _mixed_stack()
    grid = _grid(stack, 6)
    material = build_material_arrays(
        grid,
        stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
    )
    assert material.monovalent_bulk_defects is not None
    assert material.multivalent_bulk_defects is not None
    overlap = np.logical_and(
        material.monovalent_bulk_defects.explicit_node_mask,
        material.multivalent_bulk_defects.explicit_node_mask,
    )
    assert not np.any(overlap)

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
    rho_plus, _ = system._bulk_space_charge_and_tangent(n * factor, p / factor)
    rho_minus, _ = system._bulk_space_charge_and_tangent(n / factor, p * factor)
    np.testing.assert_allclose(
        tangent,
        (rho_plus - rho_minus) / (2.0 * step),
        rtol=3.0e-8,
        atol=1.0e-10,
    )

    result = solve_quasi_fermi_steady_state(
        grid,
        stack,
        V_app=0.0,
        illuminated=False,
    )
    assert result.certified
    assert result.bulk_defect_diagnostics is not None
    assert result.multivalent_bulk_defect_diagnostics is not None


def test_multivalent_qf_scope_rejects_interface_boundary_and_mobile_ions():
    stack = _stack()
    with pytest.raises(
        QuasiFermiSteadyStateError,
        match="interface-plane boundary",
    ):
        solve_quasi_fermi_steady_state(
            _grid(stack, 6),
            stack,
            illuminated=False,
            interface_boundary=True,
        )

    params = stack.layers[0].params
    assert params is not None
    mobile = replace(
        stack,
        layers=(
            replace(
                stack.layers[0],
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


def test_multivalent_qf_scope_rejects_dynamic_and_ion_device_states():
    stack = _stack()
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
    zeros = np.zeros(grid.size, dtype=float)
    with pytest.raises(QuasiFermiSteadyStateError, match="not certified"):
        system.evaluate_quasi_fermi_increments_defect_ion_combined(
            zeros,
            zeros,
            0.0,
            positive_ion_density_m3=np.full(grid.size, 1.0e20),
        )


def test_multivalent_material_contract_rejects_mismatched_documents():
    stack = _stack()
    denser = _stack()
    denser_params = replace(
        denser.layers[0].params,
        bulk_defects=(_multivalent_species(density_m3=4.0e21),),
    )
    denser = replace(
        denser,
        layers=(replace(denser.layers[0], params=denser_params),),
    )
    grid = _grid(stack, 6)
    denser_material = build_material_arrays(
        grid,
        denser,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
    )

    with pytest.raises(QuasiFermiSteadyStateError, match="does not match"):
        solve_quasi_fermi_steady_state(
            grid,
            stack,
            illuminated=False,
            mat=denser_material,
        )


def test_multivalent_intrinsic_product_mismatch_is_rejected_before_newton():
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


def test_dynamic_ac_and_transient_lanes_fail_closed_for_multivalent():
    stack = _stack()
    grid = _grid(stack, 4)

    with pytest.raises(BulkDefectDeviceACError, match="multivalent"):
        run_bulk_defect_device_impedance(
            grid,
            stack,
            np.geomspace(1.0e-4, 1.0e4, 9),
            illuminated=False,
        )

    times = np.linspace(0.0, 1.0e-6, 5)
    with pytest.raises(BulkDefectTransientError, match="multivalent"):
        run_bulk_defect_device_transient(
            grid,
            stack,
            times,
            np.zeros_like(times),
            illuminated=False,
        )
