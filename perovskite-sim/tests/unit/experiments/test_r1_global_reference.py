"""Four-node global-QF reference and representation-boundary tests.

These fixtures run no device preparation or time integration. Decimal checks
rebuild the absolute physical laws independently of both DD implementations.
"""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
import json
from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.experiments import one_dimensional_mechanism_r1_global_reference as reference
from perovskite_sim.experiments import one_dimensional_mechanism_r1_input_lift_reference as rebased
from perovskite_sim.experiments import one_dimensional_mechanism_r1_local_carrier as local
from perovskite_sim.experiments import one_dimensional_mechanism_r1_precision as precision
from perovskite_sim.experiments.interface_defect_ion_transient import _InterfaceIonTransientSystem
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import ControlledPhysicalInterfaceIonSystem
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import canonical, digest
from perovskite_sim.physics.compensated import DD


def _decimal(value):
    return Decimal.from_float(float(value))


def _represented(value):
    return _decimal(value.hi)+_decimal(value.lo)


def _words(value):
    return {"hi": value.hi.tolist(), "lo": value.lo.tolist()}


@pytest.fixture
def global_case(monkeypatch):
    mat = SimpleNamespace(has_dual_ions=False, ion_steric_diffusion_only=True,
        ion_steric_shared_site=False, P_lim_face=np.full(3, 100.),
        physical_cell_faces_m=np.array([0., .5, 1.5, 2.5, 3.]),
        dx_cell=np.array([.5, 1., 1., .5]),
        P_lim_node=np.full(4, 100.), D_ion_face=np.ones(3), V_T_device=.25,
        P_ion0=np.ones(4), N_D=np.zeros(4), N_A=np.zeros(4),
        poisson_factor=SimpleNamespace(C=np.full(3, Q), h_cell=np.ones(2)))
    system = SimpleNamespace(material=mat, stack=object(), controls=SimpleNamespace(nu_I=1),
        system=SimpleNamespace(phi0=np.zeros(4), log_n0=np.zeros(4), log_p0=np.zeros(4),
                               source_mat=SimpleNamespace(carrier_params={})),
        node_count=4, interface_count=1, negative_nodes=np.empty(0, dtype=int),
        positive_nodes=np.array([1, 2]), grid=np.arange(4.), widths=np.array([.5, 1., 1., .5]),
        thermal_voltage=.25, trap_density=np.array([3.]), equilibrium_occupancy=np.array([.3]),
        reference_n=np.ones(4), reference_p=np.ones(4), reference_phi=np.zeros(4),
        left_nodes=(1,), right_nodes=(2,), site_occupancy_ceiling=.999,
        dark_reference=SimpleNamespace(interface_transmission=0.), _sheet_weights=lambda k: (.5, .5))
    geometry = SimpleNamespace(fixed_sheet_charge_C_m2=0., potential_jump_right_minus_left_V=0.)
    monkeypatch.setattr(local, "_material_two_sided_interface_problem", lambda *a, **k: (geometry, None, None))
    monkeypatch.setattr(reference, "poisson_right_boundary", lambda *a: 0.)
    epsilon = np.array([0., 2.**-60, -2.**-61, 0.])
    fixed = {"dqfn_V": DD(np.zeros(4)), "dqfp_V": DD(np.zeros(4)),
             "positive_m3": DD(np.ones(4), epsilon), "occupancy": DD(system.equilibrium_occupancy)}
    return system, fixed, np.zeros(4)


def _decimal_physics(system, fixed, fields):
    mat, charge = system.material, _decimal(Q)
    vt = _decimal(system.thermal_voltage)
    phi = [_represented(value) for value in fields["phi_V"]]
    positive = [_represented(value) for value in fixed["positive_m3"]]
    n, p = [], []
    for k in range(4):
        change = phi[k]-_decimal(system.system.phi0[k])
        n.append((_decimal(system.system.log_n0[k])+(_represented(fixed["dqfn_V"][k])+change)/vt).exp())
        p.append((_decimal(system.system.log_p0[k])+(_represented(fixed["dqfp_V"][k])-change)/vt).exp())
    for index in (0, 3):
        n[index], p[index] = _decimal(system.reference_n[index]), _decimal(system.reference_p[index])
    poisson = []
    for k in (1, 2):
        rho = p[k]-n[k]+_decimal(mat.N_D[k])-_decimal(mat.N_A[k])+positive[k]-_decimal(mat.P_ion0[k])
        poisson.append(_decimal(mat.poisson_factor.C[k])*(phi[k+1]-phi[k])
                       - _decimal(mat.poisson_factor.C[k-1])*(phi[k]-phi[k-1])
                       + charge*rho*_decimal(mat.poisson_factor.h_cell[k-1]))
    sheet = charge*_decimal(system.trap_density[0])*(_decimal(system.equilibrium_occupancy[0])
                                                   - _represented(fixed["occupancy"][0]))
    for index, weight in enumerate(system._sheet_weights(0)):
        poisson[index] += _decimal(weight)*sheet
    chemical = [-(1-value/_decimal(capacity)).ln() for value, capacity in zip(positive, mat.P_lim_node)]
    bernoulli = lambda x: Decimal(1) if not x else x/(x.exp()-1)
    flux = []
    for k, diffusion in enumerate(mat.D_ion_face):
        xi = (phi[k+1]-phi[k])/_decimal(mat.V_T_device)+chemical[k+1]-chemical[k]
        flux.append(_decimal(diffusion)/_decimal(system.grid[k+1]-system.grid[k])*(
            bernoulli(xi)*positive[k]-bernoulli(-xi)*positive[k+1]))
    bounded = [Decimal(0), *flux, Decimal(0)]
    rate = [-(bounded[k+1]-bounded[k])/_decimal(width) for k, width in enumerate(system.widths)]
    return {"n_m3": n, "p_m3": p, "poisson_residual_C_m2": poisson,
            "positive_flux_m2_s": flux, "positive_rate_m3_s": rate}


def test_global_poisson_and_near_cancelling_sg_match_90_digit_laws(global_case, monkeypatch):
    system, fixed, seed = global_case
    monkeypatch.setattr(precision, "ion_flux_pair", lambda *a, **k: pytest.fail("read production flux"))
    inputs = reference.make_global_ion_reference_inputs(system, fixed, seed, 0.)
    fields, receipt = reference.solve_global_ion_reference(inputs)
    assert receipt["poisson"]["converged"]
    assert receipt["carrier_mapping"] == "original_global_phi0_log_n0_log_p0"
    with localcontext() as context:
        context.prec = 90
        truth = _decimal_physics(system, fixed, fields)
        assert max(map(abs, truth["positive_flux_m2_s"])) > Decimal("1e-19")
        assert max(map(abs, truth["poisson_residual_C_m2"])) < Decimal("1e-48")
        for name, expected in truth.items():
            limit = Decimal("1e-48") if name == "poisson_residual_C_m2" else (
                max(map(abs, expected))*Decimal("1e-12") if "flux" in name or "rate" in name else Decimal("1e-30"))
            for actual, wanted in zip(fields[name], expected):
                assert abs(_represented(actual)-wanted) < limit


@pytest.mark.parametrize("name", reference.FIXED_FIELDS)
def test_each_actual_fixed_low_word_reaches_global_reference(global_case, name):
    system, fixed, seed = global_case
    delta = np.zeros(fixed[name].shape)
    delta[0 if name == "occupancy" else 1] = 2.**-60
    changed = {**fixed, name: fixed[name]+DD(delta)}
    # Population and occupancy perturbations sit below binary64 high-word ULP.
    if name in ("positive_m3", "occupancy"):
        np.testing.assert_array_equal(changed[name].hi, fixed[name].hi)
    before, _ = reference.solve_global_ion_reference(reference.make_global_ion_reference_inputs(system, fixed, seed, 0.))
    after, _ = reference.solve_global_ion_reference(reference.make_global_ion_reference_inputs(system, changed, seed, 0.))
    assert np.any((before["phi_V"]-after["phi_V"]).to_float() != 0)


def test_current_outputs_rebased_origin_and_direct_sheet_cannot_enter_inputs(global_case):
    system, fixed, seed = global_case
    inputs = reference.make_global_ion_reference_inputs(system, fixed, seed, 0.)
    poison = {key: DD(np.full(4, 999.)) for key in ("phi_V", "n_m3", "p_m3", "positive_flux_m2_s", "positive_rate_m3_s")}
    system._fine_work = system._fine_evaluation_cache = poison
    system._fine_reference = system._input_lift_reference = poison
    system._step_reference = SimpleNamespace(**poison)
    system.reference_phi = np.full(4, 999.)  # Rebase-owned potential is not a global anchor.
    again = reference.make_global_ion_reference_inputs(system, fixed, seed, 0.)
    assert again == inputs
    for name in (*poison, "sheet_charge_C_m2"):
        with pytest.raises(ValueError, match="rejects direct outputs"):
            reference.make_global_ion_reference_inputs(system, {**fixed, name: DD(99.)}, seed, 0.)
    malformed = json.loads(inputs.poisson_inputs_json)
    malformed["fixed_inputs"]["sheet_charge_C_m2"]["hi"][0] = 99.
    with pytest.raises(ValueError, match="rebuilt fixed charge"):
        replace(inputs, poisson_inputs_json=json.dumps(malformed))


def test_global_dto_owns_fixed_anchor_coefficient_and_seed_bytes(global_case):
    system, fixed, seed = global_case
    inputs = reference.make_global_ion_reference_inputs(system, fixed, seed, 0.)
    frozen = inputs.to_dict()
    with pytest.raises(FrozenInstanceError):
        inputs.fixed_inputs_json = "{}"
    seed[1] = 99.
    system.system.phi0[:] = 99.
    system.system.log_n0[:] = 99.
    system.material.poisson_factor.C[:] = 99.
    system.material.D_ion_face[:] = 99.
    fixed["positive_m3"] = DD(np.full(4, 99.))
    exported = inputs.to_dict()
    exported["fixed_inputs"]["positive_m3"]["lo"][1] = 99.
    assert inputs.to_dict() == frozen


def _rebased_inputs_from_global(inputs, *, inconsistent_origin=False):
    """Derive an origin through the global law; never copy the solved state."""
    payload = inputs.to_dict()
    poisson, a = payload["poisson_inputs"], payload["coefficients"]
    prep = poisson["preparation"]
    origin_phi = DD(np.array([0., .03125, -.015625, 0.]))
    origin_qn = DD(np.array([0., .0625, -.03125, 0.]))
    origin_qp = DD(np.array([0., -.03125, .0625, 0.]))
    change = origin_phi-DD(prep["phi0_V"])
    n = (DD(prep["log_n0"])+(origin_qn+change)/DD(prep["thermal_voltage_V"])).exp()
    p = (DD(prep["log_p0"])+(origin_qp-change)/DD(prep["thermal_voltage_V"])).exp()
    if inconsistent_origin:
        n = precision.put(n, 1, n[1]*DD(1.001))
    origin = {"phi_V": origin_phi, "n_m3": n, "p_m3": p, "dqfn_V": origin_qn, "dqfp_V": origin_qp}
    coefficients = {key: a[key] for key in (
        "trap_density_m2", "equilibrium_occupancy", "static_sheet_charge_C_m2", "trace_jump_V",
        "grid_spacing_m", "control_volume_width_m", "ion_capacity_m3", "ion_diffusion_m2_s",
        "positive_nodes", "site_occupancy_ceiling", "ion_flux_enabled", "model")}
    coefficients.update({key: prep[key] for key in (
        "thermal_voltage_V", "poisson_capacitance_F_m2", "poisson_width_m", "N_D_m3", "N_A_m3",
        "ion_background_m3", "interface_nodes", "sheet_weights", "boundary_phi_V")})
    coefficients.update(charge_C=Q, ion_thermal_voltage_V=a["ion_thermal_voltage_V"])
    return rebased.RebasedIonReferenceInputs(canonical(payload["fixed_inputs"]),
        canonical({key: _words(value) for key, value in origin.items()}),
        canonical(coefficients), canonical(poisson["seed_phi_V"]))


def test_rebased_and_global_maps_agree_only_for_a_global_consistent_origin(global_case):
    system, fixed, seed = global_case
    inputs = reference.make_global_ion_reference_inputs(system, fixed, seed, 0.)
    global_fields, _ = reference.solve_global_ion_reference(inputs)
    consistent, _ = rebased.solve_rebased_ion_reference(_rebased_inputs_from_global(inputs))
    inconsistent, _ = rebased.solve_rebased_ion_reference(_rebased_inputs_from_global(inputs, inconsistent_origin=True))
    for key in ("phi_V", "n_m3", "p_m3"):
        assert float(np.max(np.abs((global_fields[key]-consistent[key]).to_float()))) < 1e-29
    assert float(np.max(np.abs((global_fields["phi_V"]-inconsistent["phi_V"]).to_float()))) > 1e-6


def _legacy_values(state):
    values = {"potential": {"eliminated": np.zeros(4)}, "unrelated": {"must_remain": True}}
    for name, direct, floor in (("positive_ion_flux", state.positive_flux, 1.),
                                ("positive_ion_rate", state.positive_rate, 7.)):
        values[name] = {"direct": direct.copy(), "eliminated": np.full_like(direct, .25),
                        "normalization_floor": floor, "unit": "preserved", "relative_error": .75}
    return values


def test_baseline_adapter_keeps_actual_zero_lows_high_subtraction_and_other_channels(global_case):
    system, fixed, seed = global_case
    state = SimpleNamespace(dqfn=fixed["dqfn_V"].hi.copy(), dqfp=fixed["dqfp_V"].hi.copy(),
        positive=fixed["positive_m3"].hi.copy(), occupancy=fixed["occupancy"].hi.copy(),
        phi=np.zeros(4), negative=None,
        positive_flux=np.array([2., 3., 4.]), positive_rate=np.array([1., 2., 3., 4.]))
    legacy = _legacy_values(state)
    result = reference.baseline_eliminated_operator_diagnostics(system, state, 0., legacy)
    assert result["unrelated"] is legacy["unrelated"]
    assert result["potential"] is legacy["potential"]
    assert not hasattr(state, "fine") and not hasattr(state, "input_lift")
    for name in ("positive_ion_flux", "positive_ion_rate"):
        row = result[name]
        assert row["legacy_binary64"] is not legacy[name]
        assert row["normalization_floor"] == legacy[name]["normalization_floor"]
        np.testing.assert_array_equal(row["difference"], row["direct"]-row["eliminated"])
        assert row["relative_error"] == np.max(np.abs(row["difference"]))/row["normalization_scale"]
    assert result["positive_ion_rate"]["relative_error"] > 1e-6  # A bad direct state still fails.
    saved_fixed = result["positive_ion_rate"]["independent_reference"]["inputs"]["fixed_inputs"]
    for name in reference.FIXED_FIELDS:
        np.testing.assert_array_equal(saved_fixed[name]["lo"], 0.)
    state.fine = fixed
    with pytest.raises(TypeError, match="binary64"):
        reference.baseline_eliminated_operator_diagnostics(system, state, 0., legacy)


def _saved_high_transport(system, phi, positive, **kwargs):
    return reference.evaluate_independent_saved_high_ion_transport(
        phi_V=phi, positive_m3=positive, capacity_m3=system.material.P_lim_node,
        diffusion_m2_s=system.material.D_ion_face, grid_spacing_m=np.diff(system.grid),
        control_volume_width_m=system.widths, thermal_voltage_V=system.material.V_T_device,
        **kwargs)


def test_baseline_rounds_constraint_before_sg_and_retains_failed_full_root_diagnostic(global_case, monkeypatch):
    system, fixed, seed = global_case
    # This prescribed root isolates the representation boundary, not Poisson
    # convergence: all high potentials are identical, while root lows cause a
    # resolvable drift in an otherwise uniform, large ion population.
    system.material.P_lim_node = np.full(4, 1e23)
    system.material.P_lim_face = np.full(3, 1e23)
    positive = np.full(4, 1e20)
    constraint = DD(np.ones(4), np.array([0., 2.**-56, -2.**-57, 0.]))
    phi = constraint.to_float()
    direct_flux, direct_rate = _saved_high_transport(system, phi, positive)
    state = SimpleNamespace(phi=phi, negative=None, positive=positive,
        dqfn=fixed["dqfn_V"].hi, dqfp=fixed["dqfp_V"].hi,
        occupancy=fixed["occupancy"].hi,
        positive_flux=direct_flux.to_float(), positive_rate=direct_rate.to_float())
    observed_inputs = []

    def prescribed_root(inputs):
        observed_inputs.append(inputs)
        flux, rate = reference.evaluate_independent_ion_transport(inputs, constraint)
        return {"phi_V": constraint, "positive_flux_m2_s": flux,
                "positive_rate_m3_s": rate}, {"prescribed_test_root": True}

    monkeypatch.setattr(reference, "solve_global_ion_reference", prescribed_root)
    values = _legacy_values(state)
    result = reference.baseline_eliminated_operator_diagnostics(system, state, 0., values)
    assert set(result) == set(values)  # No new qualification channels.
    for name in ("positive_ion_flux", "positive_ion_rate"):
        assert result[name]["relative_error"] == 0.
        assert result[name]["normalization_floor"] == values[name]["normalization_floor"]
        np.testing.assert_array_equal(result[name]["difference"], 0.)
    boundary = result["positive_ion_rate"]["independent_reference"]["representation_boundary"]
    assert boundary["constraint_phi_V"] == _words(constraint)
    assert boundary["constitutive_phi_V"] == _words(DD(phi))
    assert boundary["constitutive_minus_constraint_phi_V"] == _words(DD(phi)-constraint)
    assert np.any(np.asarray(boundary["constitutive_minus_constraint_phi_V"]["lo"]) != 0.) or np.any(
        np.asarray(boundary["constitutive_minus_constraint_phi_V"]["hi"]) != 0.)
    for name in ("positive_ion_flux", "positive_ion_rate"):
        diagnostic = boundary["cross_representation_comparisons"][name]["direct_vs_full_DD"]
        assert diagnostic["relative_error"] > diagnostic["diagnostic_original_limit"] == 1e-6
        assert diagnostic["within_original_limit"] is False
    # Reference input and quantization do not adapt to bad direct outputs.
    state.positive_flux = np.ones(3)
    state.positive_rate = np.ones(4)
    changed = reference.baseline_eliminated_operator_diagnostics(system, state, 0., _legacy_values(state))
    assert observed_inputs[0] == observed_inputs[1]
    changed_boundary = changed["positive_ion_rate"]["independent_reference"]["representation_boundary"]
    assert changed_boundary["constitutive_phi_V"] == boundary["constitutive_phi_V"]
    assert changed_boundary["same_word_transport"] == boundary["same_word_transport"]
    assert changed["positive_ion_flux"]["relative_error"] > 1e-6


def test_baseline_legacy_label_evaluates_old_law_at_both_actual_high_potentials(global_case, monkeypatch):
    system, fixed, seed = global_case
    state = SimpleNamespace(phi=np.array([0., .02, -.01, 0.]), negative=None,
        positive=fixed["positive_m3"].hi, occupancy=fixed["occupancy"].hi,
        dqfn=fixed["dqfn_V"].hi, dqfp=fixed["dqfp_V"].hi,
        positive_flux=np.full(3, 777.), positive_rate=np.full(4, 888.))
    values = _legacy_values(state)
    values["potential"]["eliminated"] = np.array([0., .03, -.01, 0.])
    calls = []

    def old_law(owner, phi, positive, negative):
        assert owner is system and positive is state.positive and negative is None
        calls.append(phi.copy())
        return np.full(4, phi[1]*100), None, np.full(3, phi[1]*10), None

    monkeypatch.setattr(_InterfaceIonTransientSystem, "_ion_fields", old_law)
    result = reference.baseline_eliminated_operator_diagnostics(system, state, 0., values)
    np.testing.assert_array_equal(calls, [state.phi, values["potential"]["eliminated"]])
    rate = result["positive_ion_rate"]["legacy_binary64"]
    np.testing.assert_array_equal(rate["direct"], 2.)
    np.testing.assert_array_equal(rate["eliminated"], 3.)
    assert rate["normalization_floor"] == 2.
    assert rate["relative_error"] == 1/3
    assert result["positive_ion_rate"]["direct"].tolist() == [888.]*4
    system.controls.nu_I = 0
    result = reference.baseline_eliminated_operator_diagnostics(system, state, 0., values)
    assert len(calls) == 2
    for name in ("positive_ion_flux", "positive_ion_rate"):
        legacy = result[name]["legacy_binary64"]
        np.testing.assert_array_equal(legacy["direct"], 0.)
        np.testing.assert_array_equal(legacy["eliminated"], 0.)
        assert legacy["relative_error"] == 0.


@pytest.mark.parametrize("fault", ["none", "thermal_voltage", "drift_sign", "diffusion", "omit_ion", "single_face_sign"])
def test_saved_high_helper_matches_zero_low_reference_and_preserves_faults(global_case, monkeypatch, fault):
    system, fixed, seed = global_case
    fixed = {key: DD(value.to_float()) for key, value in fixed.items()}
    inputs = reference.make_global_ion_reference_inputs(system, fixed, seed, 0.)
    phi = np.array([0., .02, -.01, 0.])
    monkeypatch.setattr(precision, "ion_flux_pair", lambda *a, **k: pytest.fail("used production ion flux"))
    saved = _saved_high_transport(system, phi, fixed["positive_m3"].hi, fault=fault)
    isolated = reference.evaluate_independent_ion_transport(inputs, DD(phi), fault=fault)
    for left, right in zip(saved, isolated):
        assert _words(left) == _words(right)
    with pytest.raises(TypeError, match="binary64"):
        _saved_high_transport(system, DD(phi), fixed["positive_m3"].hi, fault=fault)


def test_saved_high_disabled_flux_preserves_explicit_boundary_divergence(global_case):
    system, fixed, seed = global_case
    flux, rate = _saved_high_transport(system, np.zeros(4), fixed["positive_m3"].hi,
                                     enabled=False, boundary_flux_m2_s=np.array([2., -3.]))
    np.testing.assert_array_equal(flux.to_float(), 0.)
    np.testing.assert_array_equal(rate.to_float(), [4., 0., 0., 6.])


def _pair_case(system, fixed, seed):
    owner = object.__new__(precision.CompensatedR1System)
    owner.__dict__.update(system.__dict__)
    # controls is a property on the actual owner; its backing field is explicit.
    owner._controls = system.controls
    owner._r1_backend = SimpleNamespace(is_pair=True)
    owner.precision_fault = "none"
    inputs = reference.make_global_ion_reference_inputs(owner, fixed, seed, 0.)
    fields, _ = reference.solve_global_ion_reference(inputs)
    state = object.__new__(precision.PrecisionState)
    state.fine = {**fixed, **fields, "sheet_charge_C_m2": DD([0.]),
                  "trace_potential_V": DD(np.zeros((1, 2))), "trace_state_m3": DD(np.ones((1, 4)))}
    state.positive_flux, state.positive_rate = fields["positive_flux_m2_s"].hi, fields["positive_rate_m3_s"].hi
    return owner, state


def test_pair_adapter_keeps_low_difference_v2_bindings_and_ignores_direct_sheet(global_case, monkeypatch):
    system, fixed, seed = global_case
    owner, state = _pair_case(system, fixed, seed)
    legacy = _legacy_values(state)
    monkeypatch.setattr(ControlledPhysicalInterfaceIonSystem, "eliminated_operator_diagnostics", lambda *a: deepcopy(legacy))
    monkeypatch.setattr(precision, "ion_flux_pair", lambda *a, **k: pytest.fail("reference read production flux"))
    healthy = owner.eliminated_operator_diagnostics(state, 0.)
    evidence = healthy.precision_evidence
    assert evidence["schema"] == "R1EliminatedPrecisionV2"
    shared = evidence["shared_inputs"]["fields"]
    assert evidence["solve"]["fixed_input_digest"] == digest(shared)
    assert evidence["solve_inputs"]["fixed_inputs"] == shared
    assert evidence["solve"]["solve_inputs_digest"] == digest(evidence["solve_inputs"])
    assert evidence["solve"]["ion_input_digest"] == digest(evidence["ion_inputs"])
    # Deliberately incorrect derived sheet cannot steer the independent solve.
    state.fine["sheet_charge_C_m2"] = DD([999.])
    poisoned = owner.eliminated_operator_diagnostics(state, 0.)
    assert poisoned.precision_evidence["solve_inputs"] == evidence["solve_inputs"]
    other = DD(evidence["fields"]["positive_flux_m2_s"]["hi"], evidence["fields"]["positive_flux_m2_s"]["lo"])
    state.fine["positive_flux_m2_s"] = other+DD(np.full(3, 2.**-125))
    changed = owner.eliminated_operator_diagnostics(state, 0.)
    np.testing.assert_array_equal(state.fine["positive_flux_m2_s"].to_float(), other.to_float())
    np.testing.assert_array_equal(changed["positive_ion_flux"]["difference"],
                                  (state.fine["positive_flux_m2_s"]-other).to_float())
    assert np.all(changed["positive_ion_flux"]["difference"] != 0.)
    state.fine["positive_flux_m2_s"] = DD(np.ones(3))
    assert owner.eliminated_operator_diagnostics(state, 0.)["positive_ion_flux"]["relative_error"] > 1e-6


def test_pair_independent_potential_and_boundary_hooks_remain_effective(global_case, monkeypatch):
    system, fixed, seed = global_case
    owner, state = _pair_case(system, fixed, seed)
    legacy = _legacy_values(state)
    monkeypatch.setattr(ControlledPhysicalInterfaceIonSystem, "eliminated_operator_diagnostics", lambda *a: deepcopy(legacy))
    before = owner.eliminated_operator_diagnostics(state, 0.)
    owner._eliminated_flux_potential = lambda phi: precision.put(phi, 1, phi[1]+DD(.001))
    shifted = owner.eliminated_operator_diagnostics(state, 0.)
    assert shifted.precision_evidence["fields"]["constraint_phi_V"] == before.precision_evidence["fields"]["constraint_phi_V"]
    assert shifted.precision_evidence["fields"]["phi_V"] != before.precision_evidence["fields"]["phi_V"]
    assert shifted["positive_ion_flux"]["relative_error"] > 1e-6
    owner._eliminated_flux_potential = lambda phi: phi
    owner._ion_boundary_flux = lambda: DD([2., -3.])
    leaking = owner.eliminated_operator_diagnostics(state, 0.)
    assert leaking.precision_evidence["ion_inputs"]["boundary_flux_m2_s"] == _words(DD([2., -3.]))
    np.testing.assert_allclose(leaking["positive_ion_rate"]["eliminated"]
                               - before["positive_ion_rate"]["eliminated"], [4., 0., 0., 6.])


@pytest.mark.parametrize("fault", ["none", "thermal_voltage", "drift_sign", "diffusion", "omit_ion", "single_face_sign"])
def test_independent_fault_paths_preserve_original_constitutive_meaning(global_case, fault):
    system, fixed, seed = global_case
    inputs = reference.make_global_ion_reference_inputs(system, fixed, seed, 0.)
    phi = DD(np.array([0., .02, -.01, 0.]))
    flux, _ = reference.evaluate_independent_ion_transport(inputs, phi, fault=fault)
    original = precision.ion_flux_pair(phi, fixed["positive_m3"], system.material, np.diff(system.grid), fault=fault)
    assert float(np.max(np.abs((flux-original).to_float()))) < 1e-29


def test_disabled_ions_keep_zero_faces_and_explicit_boundary_contract(global_case):
    system, fixed, seed = global_case
    system.controls.nu_I = 0
    inputs = reference.make_global_ion_reference_inputs(system, fixed, seed, 0., boundary_flux=DD([2., -3.]))
    flux, rate = reference.evaluate_independent_ion_transport(inputs, DD(np.zeros(4)))
    np.testing.assert_array_equal(flux.hi, 0.)
    np.testing.assert_array_equal(rate.hi, [4., 0., 0., 6.])
