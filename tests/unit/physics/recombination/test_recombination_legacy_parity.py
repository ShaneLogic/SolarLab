"""M-class parity: the migrated recombination layer against the legacy engine.

Same inputs must give bit-identical payloads (``tobytes`` equality, so even
the sign of zero is compared). The legacy package is imported here only,
as the frozen reference; the new package never imports it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.physics import bulk_traps as legacy_traps
from perovskite_sim.physics import recombination as legacy
from perovskite_sim.physics import traps as legacy_profile
from solarlab.physics import recombination as new

LEGACY_ROOT = Path(__file__).resolve().parents[4] / "perovskite-sim"
DIGEST_A = "a1" * 32
DIGEST_B = "b2" * 32


def _same(a, b) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _random_state(n_nodes, seed):
    rng = np.random.default_rng(seed)
    n = 10.0 ** rng.uniform(4.0, 25.0, n_nodes)
    p = 10.0 ** rng.uniform(4.0, 25.0, n_nodes)
    coefficients = dict(
        ni_sq=10.0 ** rng.uniform(8.0, 22.0, n_nodes),
        tau_n=10.0 ** rng.uniform(-10.0, -4.0, n_nodes),
        tau_p=10.0 ** rng.uniform(-10.0, -4.0, n_nodes),
        n1=10.0 ** rng.uniform(0.0, 20.0, n_nodes),
        p1=10.0 ** rng.uniform(0.0, 20.0, n_nodes),
        B_rad=10.0 ** rng.uniform(-18.0, -15.0, n_nodes),
        C_n=10.0 ** rng.uniform(-43.0, -39.0, n_nodes),
        C_p=10.0 ** rng.uniform(-43.0, -39.0, n_nodes),
    )
    return n, p, coefficients


def _bound(n_nodes, c, model=None):
    coefficients = new.BulkRecombinationCoefficients(
        n_nodes, c["ni_sq"], c["tau_n"], c["tau_p"], c["n1"], c["p1"], c["B_rad"], c["C_n"], c["C_p"]
    )
    return new.BulkRecombinationReaction(coefficients, model)


def _legacy_args(c):
    return (c["ni_sq"], c["tau_n"], c["tau_p"], c["n1"], c["p1"])


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_lifetime_srh_radiative_auger_and_total_are_bitwise_legacy(seed):
    n, p, c = _random_state(257, seed)
    evaluation = _bound(257, c).evaluate(n, p)
    srh = legacy.srh_recombination_derivatives(n, p, *_legacy_args(c))
    rad = legacy.radiative_recombination_derivatives(n, p, c["ni_sq"], c["B_rad"])
    aug = legacy.auger_recombination_derivatives(n, p, c["ni_sq"], c["C_n"], c["C_p"])
    tot = legacy.total_recombination_derivatives(n, p, *_legacy_args(c), c["B_rad"], c["C_n"], c["C_p"])
    for new_term, old_term in ((evaluation.srh, srh), (evaluation.radiative, rad), (evaluation.auger, aug),
                               (evaluation.total, tot)):
        assert _same(new_term.rate, old_term.rate)
        assert _same(new_term.d_rate_dn, old_term.electron_density_derivative)
        assert _same(new_term.d_rate_dp, old_term.hole_density_derivative)
    assert _same(evaluation.srh.rate, legacy.srh_recombination(n, p, *_legacy_args(c)))
    assert _same(evaluation.total.rate,
                 legacy.total_recombination(n, p, *_legacy_args(c), c["B_rad"], c["C_n"], c["C_p"]))
    assert _same(evaluation.srh_denominators_s_m3,
                 legacy.bulk_recombination_denominators(n, p, c["tau_n"], c["tau_p"], c["n1"], c["p1"]))


def test_scalar_coefficients_broadcast_bitwise():
    n, p, _ = _random_state(64, 9)
    scalars = dict(ni_sq=2.1e18, tau_n=3e-7, tau_p=8e-9, n1=4e7, p1=6e11, B_rad=3e-17, C_n=2e-41, C_p=7e-42)
    evaluation = _bound(64, scalars).evaluate(n, p)
    old = legacy.total_recombination_derivatives(n, p, 2.1e18, 3e-7, 8e-9, 4e7, 6e11, 3e-17, 2e-41, 7e-42)
    assert _same(evaluation.total.rate, old.rate)
    assert _same(evaluation.total.d_rate_dn, old.electron_density_derivative)
    assert _same(evaluation.total.d_rate_dp, old.hole_density_derivative)


def _species_pair(mask_a, mask_b, n_nodes, *, tau_b=(4e-9, 9e-10)):
    rng = np.random.default_rng(5)
    fields = []
    for mask, digest, taus, ident in ((mask_a, DIGEST_A, (2e-8, 3e-9), "layer[1]/a"),
                                      (mask_b, DIGEST_B, tau_b, "layer[2]/b")):
        n1 = np.where(mask, 10.0 ** rng.uniform(2, 16, n_nodes), 0.0)
        p1 = np.where(mask, 10.0 ** rng.uniform(2, 16, n_nodes), 0.0)
        fields.append((ident, digest, mask, taus[0], taus[1], n1, p1))
    old_species = tuple(legacy.CompiledNeutralDefectSpecies(*f) for f in fields)
    new_species = tuple(new.NeutralDefectSpecies(*f) for f in fields)
    return old_species, new_species


@pytest.mark.parametrize("case", ["one_species", "two_disjoint", "two_overlapping", "one_blocked"])
def test_explicit_neutral_partition_is_bitwise_legacy(case):
    n_nodes = 41
    n, p, c = _random_state(n_nodes, 4)
    mask_a = np.zeros(n_nodes, dtype=bool)
    mask_a[5:20] = True
    mask_b = np.zeros(n_nodes, dtype=bool)
    mask_b[12:30] = True if case == "two_overlapping" else False
    if case in ("two_disjoint", "one_blocked", "one_species"):
        mask_b[25:35] = True
    tau_b = (np.inf, 9e-10) if case == "one_blocked" else (4e-9, 9e-10)
    old_species, new_species = _species_pair(mask_a, mask_b, n_nodes, tau_b=tau_b)
    if case == "one_species":
        old_species, new_species = old_species[:1], new_species[:1]
    union = np.logical_or.reduce([s.active_nodes for s in old_species])
    documents = tuple(sorted({(s.identifier.split("/")[0], s.document_sha256) for s in old_species}))
    old_model = legacy.NeutralBulkDefectModel(old_species, union, documents)
    new_model = new.NeutralExplicitDefectModel(new_species, union, documents)
    evaluation = _bound(n_nodes, c, new_model).evaluate(n, p)

    old_value = legacy.srh_recombination(n, p, *_legacy_args(c), neutral_bulk_defects=old_model)
    old_srh = legacy.srh_recombination_derivatives(n, p, *_legacy_args(c), neutral_bulk_defects=old_model)
    old_total = legacy.total_recombination_derivatives(
        n, p, *_legacy_args(c), c["B_rad"], c["C_n"], c["C_p"], neutral_bulk_defects=old_model
    )
    old_total_value = legacy.total_recombination(
        n, p, *_legacy_args(c), c["B_rad"], c["C_n"], c["C_p"], neutral_bulk_defects=old_model
    )
    assert _same(evaluation.srh.rate, old_value)
    assert _same(evaluation.srh.rate, old_srh.rate)
    assert _same(evaluation.srh.d_rate_dn, old_srh.electron_density_derivative)
    assert _same(evaluation.srh.d_rate_dp, old_srh.hole_density_derivative)
    assert _same(evaluation.total.rate, old_total.rate)
    assert _same(evaluation.total.rate, old_total_value)
    assert _same(evaluation.total.d_rate_dn, old_total.electron_density_derivative)
    assert _same(evaluation.total.d_rate_dp, old_total.hole_density_derivative)
    assert _same(
        evaluation.srh_denominators_s_m3,
        legacy.bulk_recombination_denominators(
            n, p, c["tau_n"], c["tau_p"], c["n1"], c["p1"], neutral_bulk_defects=old_model
        ),
    )

    old_eval = legacy.evaluate_neutral_bulk_defects(n, p, c["ni_sq"], old_model)
    new_eval = evaluation.explicit
    assert new_eval.model_identity_sha256 == old_eval.model_identity_sha256 == old_model.identity_sha256
    assert _same(new_eval.per_species_rate_m3_s, old_eval.per_species_rate_m3_s)
    assert _same(new_eval.per_species_d_rate_dn_s1, old_eval.per_species_electron_derivative_s1)
    assert _same(new_eval.per_species_d_rate_dp_s1, old_eval.per_species_hole_derivative_s1)
    assert new_eval.minimum_denominator_s_m3 == old_eval.minimum_denominator_s_m3
    assert new_eval.to_dict() == old_eval.to_dict()


def test_def1_species_compile_is_bitwise_the_legacy_material_build():
    from perovskite_sim.discretization.grid import Layer, multilayer_grid
    from perovskite_sim.experiments.jv_sweep import _layer_node_counts
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.models.device import electrical_layers
    from perovskite_sim.physics.temperature import thermal_voltage
    from perovskite_sim.solver.mol import build_material_arrays

    stack = load_device_from_yaml(str(LEGACY_ROOT / "tests/fixtures/configs/scaps_defect_s0_neutral.yaml"))
    elec = electrical_layers(stack)
    x = multilayer_grid([Layer(layer.thickness, count) for layer, count in zip(elec, _layer_node_counts(stack, 40))])
    mat = build_material_arrays(x, stack)
    old_model = mat.neutral_bulk_defects
    assert old_model is not None
    compiled = []
    documents = []
    for index, layer in enumerate(elec):
        document = layer.params.defect_document
        if document is None:
            continue
        layer_id = f"layer[{index}]/{layer.name}"
        documents.append((layer_id, document.sha256))
        for species, old in zip(document.bulk_defects, (s for s in old_model.species if s.identifier.startswith(layer_id))):
            compiled.append(
                new.compile_neutral_defect_species(
                    identifier=f"{layer_id}/{species.name}",
                    document_sha256=document.sha256,
                    active_nodes=old.active_nodes,
                    ni_sq_m6=mat.ni_sq,
                    band_gap_eV=mat.Eg_phys,
                    thermal_voltage_V=thermal_voltage(stack.T),
                    total_density_m3=species.distribution.total_density_m3,
                    center_eV_above_vb=species.distribution.center_eV_above_vb,
                    sigma_n_m2=species.kinetics.sigma_n_m2,
                    sigma_p_m2=species.kinetics.sigma_p_m2,
                    thermal_velocity_n_m_s=species.kinetics.thermal_velocity_n_m_s,
                    thermal_velocity_p_m_s=species.kinetics.thermal_velocity_p_m_s,
                    charge_transition=species.charge_transition,
                    distribution=species.distribution.kind,
                    degeneracy=species.degeneracy,
                )
            )
    assert len(compiled) == len(old_model.species)
    for mine, old in zip(compiled, old_model.species):
        assert mine.identifier == old.identifier
        assert mine.tau_n_s == old.tau_n_s and mine.tau_p_s == old.tau_p_s
        assert _same(mine.n1_m3, old.n1_m3) and _same(mine.p1_m3, old.p1_m3)
        assert _same(mine.active_nodes, old.active_nodes)
    model = new.NeutralExplicitDefectModel(tuple(compiled), old_model.explicit_node_mask, tuple(documents))
    assert model.identity_sha256 == old_model.identity_sha256


@pytest.mark.parametrize("edge", ["both", "left", "right"])
@pytest.mark.parametrize("length", [0.0, 2e-9, 25e-9, 400e-9, 5e-6])
@pytest.mark.parametrize("shape", ["exponential", "gaussian"])
def test_trap_profiles_and_lifetime_rule_are_bitwise_legacy(edge, length, shape):
    thickness = 300e-9
    x_local = np.linspace(0.0, thickness, 73) ** 1.0
    x_local = np.sort(np.concatenate([x_local, [1e-12, thickness - 1e-12, 0.37 * thickness]]))
    args = (x_local, thickness, 1e23, 1e20, length)
    if shape == "exponential":
        old_density = legacy_profile.exponential_edge_profile(*args, edge=edge)
        new_density = new.exponential_edge_profile(*args, edge=edge)
    else:
        old_density = legacy_profile.gaussian_edge_profile(*args, edge=edge)
        new_density = new.gaussian_edge_profile(*args, edge=edge)
    assert _same(new_density, old_density)
    tau = np.full(x_local.shape, 1e-6)
    assert _same(new.tau_from_trap_density(tau, new_density, 1e20),
                 legacy_profile.tau_from_trap_density(tau, old_density, 1e20))
    binding = new.TrapProfile(1e23, 1e20, length, shape=shape, edge=edge).bind(x_local, thickness, tau, 2 * tau)
    assert _same(binding.trap_density_m3, old_density)
    assert _same(binding.tau_n_s, legacy_profile.tau_from_trap_density(tau, old_density, 1e20))
    assert _same(binding.tau_p_s, legacy_profile.tau_from_trap_density(2 * tau, old_density, 1e20))


@pytest.mark.parametrize("shape", ["exponential", "gaussian"])
def test_trap_profile_binding_matches_the_legacy_material_build(shape):
    from perovskite_sim.discretization.grid import Layer, multilayer_grid
    from perovskite_sim.experiments.jv_sweep import _layer_node_counts
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.models.device import electrical_layers
    from perovskite_sim.solver.mol import _layer_node_masks, build_material_arrays

    stack = load_device_from_yaml(str(LEGACY_ROOT / "tests/fixtures/configs/nip_MAPbI3.yaml"))
    layers = list(stack.layers)
    index = next(i for i, layer in enumerate(layers) if layer.role == "absorber")
    params = dataclasses.replace(
        layers[index].params,
        trap_N_t_interface=5e22,
        trap_N_t_bulk=1e20,
        trap_decay_length=20e-9,
        trap_profile_shape=shape,
        trap_edge="right",
    )
    layers[index] = dataclasses.replace(layers[index], params=params)
    stack = dataclasses.replace(stack, layers=tuple(layers))
    elec = electrical_layers(stack)
    x = multilayer_grid([Layer(layer.thickness, count) for layer, count in zip(elec, _layer_node_counts(stack, 60))])
    mat = build_material_arrays(x, stack)
    absorber = next(k for k, layer in enumerate(elec) if layer.role == "absorber")
    mask = _layer_node_masks(x, elec)[absorber]
    offset = 0.0
    for layer in elec[:absorber]:
        offset += layer.thickness  # the material build's own accumulation order
    binding = new.TrapProfile(5e22, 1e20, 20e-9, shape=shape, edge="right").bind(
        x[mask] - offset, elec[absorber].thickness, params.tau_n, params.tau_p
    )
    assert mat.has_trap_profile
    assert _same(binding.trap_density_m3, mat.N_t_node[mask])
    assert _same(binding.tau_n_s, mat.tau_n[mask])
    assert _same(binding.tau_p_s, mat.tau_p[mask])


_DISTRIBUTIONS = [
    dict(distribution="single_level", total_density_m3=1e21, center_eV_above_vb=0.8, sigma_n_m2=1e-19,
         sigma_p_m2=3e-20, thermal_velocity_m_s=1e5, charge_transition="acceptor"),
    dict(distribution="single_level", total_density_m3=4e20, center_eV_above_vb=0.3, sigma_n_m2=2e-20,
         sigma_p_m2=7e-19, thermal_velocity_m_s=2e5, charge_transition="donor"),
    dict(distribution="gaussian", total_density_m3=2e22, center_eV_above_vb=0.75, sigma_n_m2=1e-19,
         sigma_p_m2=1e-19, thermal_velocity_m_s=1e5, charge_transition="acceptor", energy_sigma_eV=0.1),
    dict(distribution="gaussian", total_density_m3=1e21, center_eV_above_vb=0.05, sigma_n_m2=5e-20,
         sigma_p_m2=1e-21, thermal_velocity_m_s=1e5, charge_transition="donor", energy_sigma_eV=0.03),
]


@pytest.mark.parametrize("spec", _DISTRIBUTIONS)
def test_energy_resolved_trap_state_is_bitwise_legacy(spec):
    old_distribution = legacy_traps.bulk_trap_distribution_from_mapping(spec)
    new_distribution = new.bulk_trap_distribution_from_mapping(spec)
    old_q = legacy_traps.build_bulk_trap_quadrature(old_distribution, band_gap_eV=1.6, order=48)
    new_q = new.build_bulk_trap_quadrature(new_distribution, band_gap_eV=1.6, order=48)
    assert new_q.energy_levels_eV == old_q.energy_levels_eV
    assert new_q.density_weights_m3 == old_q.density_weights_m3
    rng = np.random.default_rng(3)
    for n, p in ((1e15, 3e12), (10.0 ** rng.uniform(5, 24, 33), 10.0 ** rng.uniform(5, 24, 33))):
        kwargs = dict(band_gap_eV=1.6, effective_conduction_dos_m3=2.2e24, effective_valence_dos_m3=1.8e25,
                      temperature_K=310.0, quadrature_order=48)
        old = legacy_traps.evaluate_bulk_trap_state(n, p, old_distribution, **kwargs)
        mine = new.evaluate_bulk_trap_state(n, p, new_distribution, **kwargs)
        for field in dataclasses.fields(old):
            assert _same(getattr(mine, field.name), getattr(old, field.name)), field.name


@pytest.mark.parametrize("spec", _DISTRIBUTIONS)
@pytest.mark.parametrize("doping", [(0.0, 0.0), (1e22, 0.0), (0.0, 3e23), (5e21, 2e21)])
def test_trap_aware_neutrality_is_bitwise_legacy(spec, doping):
    kwargs = dict(temperature_K=300.0, band_gap_eV=1.6, effective_conduction_dos_m3=2.2e24,
                  effective_valence_dos_m3=1.8e25, acceptor_density_m3=doping[0], donor_density_m3=doping[1])
    old = legacy_traps.solve_bulk_trap_charge_neutrality(
        distribution=legacy_traps.bulk_trap_distribution_from_mapping(spec), **kwargs
    )
    mine = new.solve_bulk_trap_charge_neutrality(distribution=new.bulk_trap_distribution_from_mapping(spec), **kwargs)
    state = old.neutrality
    for name in ("temperature_K", "thermal_voltage_V", "band_gap_eV", "acceptor_density_m3", "donor_density_m3",
                 "reduced_electron_fermi_level", "reduced_hole_fermi_level", "electron_density_m3",
                 "hole_density_m3", "normalized_charge_residual"):
        assert getattr(mine, name) == getattr(state, name), name
    for field in dataclasses.fields(old.trap_state):
        assert _same(getattr(mine.trap_state, field.name), getattr(old.trap_state, field.name)), field.name


@pytest.mark.parametrize(
    "spec",
    [
        {},
        {**_DISTRIBUTIONS[0], "unexpected": 1},
        {**_DISTRIBUTIONS[0], "energy_sigma_eV": 0.1},
        {**_DISTRIBUTIONS[2], "energy_sigma_eV": None},
        {**_DISTRIBUTIONS[0], "charge_transition": "neutral"},
        {**_DISTRIBUTIONS[0], "total_density_m3": True},
    ],
)
def test_bulk_trap_schema_rejections_match_legacy(spec):
    with pytest.raises(ValueError) as old_error:
        legacy_traps.bulk_trap_distribution_from_mapping(spec)
    with pytest.raises(ValueError) as new_error:
        new.bulk_trap_distribution_from_mapping(spec)
    assert str(new_error.value) == str(old_error.value)
