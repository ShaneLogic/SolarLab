"""Independent fixed-input references on four nodes; no DC or time step."""

from dataclasses import replace
from decimal import Decimal, localcontext
import json
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.experiments import one_dimensional_mechanism_r1_input_lift as lift
from perovskite_sim.experiments import one_dimensional_mechanism_r1_input_lift_reference as reference
from perovskite_sim.experiments import one_dimensional_mechanism_r1_precision as precision
from perovskite_sim.physics.compensated import DD
from tests.unit.experiments.test_r1_input_lift import synthetic


FIXED_FIELDS = ("dqfn_V", "dqfp_V", "positive_m3", "occupancy")


def _decimal(value):
    return Decimal.from_float(float(value))


def _represented(value):
    return _decimal(value.hi) + _decimal(value.lo)


def _words(fields):
    return {name: (value.hi.tobytes(), value.lo.tobytes())
            for name, value in fields.items()}


@pytest.fixture
def manufactured(synthetic, monkeypatch):
    """Neutral at phi=0, with an ion gradient only in represented low words.

    n=2+epsilon, p=2, P=1+epsilon and P_background=1 cancel exactly in
    absolute Poisson. The nonuniform epsilon leaves a nonzero SG flux even
    though every population's rounded high view is spatially constant.
    """
    _, _, system, saved, _ = synthetic
    epsilon = np.array([0., 2.**-60, -2.**-61, 0.])
    origin = dict(system._input_lift_reference)
    origin.update(phi_V=DD(np.zeros(4)), n_m3=DD(np.full(4, 2.), epsilon),
                  p_m3=DD(np.full(4, 2.)), positive_m3=DD(np.ones(4), epsilon),
                  occupancy=DD(system.equilibrium_occupancy),
                  trace_potential_V=DD(np.zeros((1, 2))),
                  trace_state_m3=DD(np.full((1, 4), 2.)))
    system._input_lift_reference = MappingProxyType(origin)
    system._input_lift_reference_identity = lift._identity(origin)
    system._step_reference = replace(saved,
        phi=origin["phi_V"].hi.copy(), n=origin["n_m3"].hi.copy(),
        p=origin["p_m3"].hi.copy(), positive=origin["positive_m3"].hi.copy(),
        occupancy=origin["occupancy"].hi.copy(), input_lift=MappingProxyType(origin),
        coordinate_reference_identity=system._input_lift_reference_identity)
    # Scale capacitance so density lows produce resolvable potential changes.
    # No production operator is used to manufacture this exact root.
    system.material.poisson_factor.C = np.full(3, Q)
    monkeypatch.setattr(reference, "poisson_right_boundary", lambda *args: 0.)
    fixed = {name: origin[name] for name in FIXED_FIELDS}
    seed = np.zeros(4)
    return system, fixed, seed


def _decimal_sg_fv(system, potential, population):
    """90-digit original Bernoulli law, independent of DD SG helpers."""
    phi = [_represented(item) for item in potential]
    positive = [_represented(item) for item in population]
    vt = _decimal(system.material.V_T_device)
    chemical = [-(1 - p / _decimal(limit)).ln()
                for p, limit in zip(positive, system.material.P_lim_node)]

    def bernoulli(argument):
        return Decimal(1) if not argument else argument / (argument.exp() - 1)

    flux = []
    for k, diffusion in enumerate(system.material.D_ion_face):
        drive = (phi[k + 1] - phi[k]) / vt + chemical[k + 1] - chemical[k]
        flux.append(_decimal(diffusion) / _decimal(system.grid[k + 1] - system.grid[k])
                    * (bernoulli(drive) * positive[k]
                       - bernoulli(-drive) * positive[k + 1]))
    boundary_flux = [Decimal(0), *flux, Decimal(0)]
    rate = [-(boundary_flux[k + 1] - boundary_flux[k]) / _decimal(width)
            for k, width in enumerate(system.widths)]
    return flux, rate


def _decimal_poisson(system, fields, fixed):
    phi = [_represented(item) for item in fields["phi_V"]]
    n = [_represented(item) for item in fields["n_m3"]]
    p = [_represented(item) for item in fields["p_m3"]]
    positive = [_represented(item) for item in fixed["positive_m3"]]
    mat, charge = system.material, _decimal(Q)
    result = []
    for k in (1, 2):
        density = (p[k] - n[k] + _decimal(mat.N_D[k]) - _decimal(mat.N_A[k])
                   + positive[k] - _decimal(mat.P_ion0[k]))
        result.append(_decimal(mat.poisson_factor.C[k]) * (phi[k + 1] - phi[k])
                      - _decimal(mat.poisson_factor.C[k - 1]) * (phi[k] - phi[k - 1])
                      + charge * density * _decimal(mat.poisson_factor.h_cell[k - 1]))
    for k, (left, right) in enumerate(zip(system.left_nodes, system.right_nodes)):
        sheet = -charge * _decimal(system.trap_density[k]) * (
            _represented(fixed["occupancy"][k]) - _decimal(system.equilibrium_occupancy[k]))
        for node, weight in zip((left, right), system._sheet_weights(k)):
            result[node - 1] += _decimal(weight) * sheet
    return result


def test_manufactured_absolute_root_and_near_cancelling_sg_match_decimal(manufactured, monkeypatch):
    system, fixed, seed = manufactured

    def forbidden(*args, **kwargs):
        pytest.fail("independent reference called the production ion flux")

    # Catch a future module-level alias as well as an in-call import.
    for name, value in tuple(vars(reference).items()):
        if value is precision.ion_flux_pair:
            monkeypatch.setattr(reference, name, forbidden)
    monkeypatch.setattr(lift, "ion_flux_pair", forbidden)
    monkeypatch.setattr(precision, "ion_flux_pair", forbidden)
    inputs = reference.make_inputs(system, fixed, seed, 0.)
    fields, receipt = reference.solve_rebased_ion_reference(inputs)
    assert isinstance(inputs, reference.RebasedIonReferenceInputs)
    assert receipt["converged"] is True
    with localcontext() as context:
        context.prec = 90
        for value in fields["phi_V"]:
            assert abs(_represented(value)) < Decimal("1e-30")
        for name in ("n_m3", "p_m3"):
            for actual, expected in zip(fields[name], system._input_lift_reference[name]):
                assert abs(_represented(actual) - _represented(expected)) < Decimal("1e-30")
        poisson = _decimal_poisson(system, fields, fixed)
        for actual, expected in zip(fields["poisson_residual_C_m2"], poisson):
            assert abs(expected) < Decimal("1e-48")
            assert abs(_represented(actual) - expected) < Decimal("1e-48")
        flux, rate = _decimal_sg_fv(system, fields["phi_V"], fixed["positive_m3"])
        assert max(map(abs, flux)) > Decimal("1e-19")
        assert max(map(abs, rate)) > Decimal("1e-19")
        for name, expected in (("positive_flux_m2_s", flux), ("positive_rate_m3_s", rate)):
            scale = max(map(abs, expected))
            for actual, truth in zip(fields[name], expected):
                assert abs(_represented(actual) - truth) < scale * Decimal("1e-12")


@pytest.mark.parametrize("name", FIXED_FIELDS)
def test_each_fixed_low_word_changes_the_independent_solution(manufactured, name):
    system, fixed, seed = manufactured
    changed = dict(fixed)
    if name == "positive_m3":
        changed[name] = DD(fixed[name].hi)
    else:
        low = np.zeros(fixed[name].shape)
        low[0 if name == "occupancy" else 1] = 2.**-60
        changed[name] = fixed[name] + DD(low)
    np.testing.assert_array_equal(changed[name].hi, fixed[name].hi)
    original, _ = reference.solve_rebased_ion_reference(reference.make_inputs(system, fixed, seed, 0.))
    perturbed, _ = reference.solve_rebased_ion_reference(reference.make_inputs(system, changed, seed, 0.))
    with localcontext() as context:
        context.prec = 90
        difference = [_represented(a) - _represented(b)
                      for a, b in zip(original["phi_V"], perturbed["phi_V"])]
        assert max(map(abs, difference)) > Decimal("1e-22")
        for actual, expected in zip(perturbed["poisson_residual_C_m2"],
                                    _decimal_poisson(system, perturbed, changed)):
            assert abs(expected) < Decimal("1e-46")
            assert abs(_represented(actual) - expected) < Decimal("1e-46")


def test_trial_outputs_and_work_cache_cannot_change_reference(manufactured):
    system, fixed, seed = manufactured
    inputs = reference.make_inputs(system, fixed, seed, 0.)
    correct, _ = reference.solve_rebased_ion_reference(inputs)
    poisoned = {name: DD(np.full(size, 999.)) for name, size in
                (("phi_V", 4), ("n_m3", 4), ("p_m3", 4),
                 ("positive_flux_m2_s", 3), ("positive_rate_m3_s", 4),
                 ("poisson_residual_C_m2", 2), ("local_residual", 6))}
    system._input_lift_work = poisoned
    system._step_reference = SimpleNamespace(
        phi=np.full(4, 999.), n=np.full(4, 999.), p=np.full(4, 999.),
        positive_flux=np.full(3, 999.), positive_rate=np.full(4, 999.),
        poisson_residual=np.full(2, 999.), input_lift=poisoned)
    again = reference.make_inputs(system, fixed, seed, 0.)
    assert again.to_dict() == inputs.to_dict()
    actual, _ = reference.solve_rebased_ion_reference(again)
    assert _words(actual) == _words(correct)


@pytest.mark.parametrize("name", ["phi_V", "n_m3", "p_m3", "positive_flux_m2_s",
                                  "positive_rate_m3_s", "poisson_residual_C_m2"])
def test_fixed_input_allowlist_rejects_direct_outputs(manufactured, name):
    system, fixed, seed = manufactured
    extra = {**fixed, name: DD(np.zeros(4))}
    with pytest.raises((TypeError, ValueError)):
        reference.make_inputs(system, extra, seed, 0.)
    inputs = reference.make_inputs(system, fixed, seed, 0.)
    raw = json.loads(inputs.fixed_inputs_json)
    raw[name] = {"hi": [0.] * 4, "lo": [0.] * 4}
    with pytest.raises((TypeError, ValueError)):
        replace(inputs, fixed_inputs_json=json.dumps(raw))


def test_inputs_own_coefficient_origin_seed_and_fixed_copies(manufactured):
    system, fixed, seed = manufactured
    inputs = reference.make_inputs(system, fixed, seed, 0.)
    frozen = inputs.to_dict()
    expected, _ = reference.solve_rebased_ion_reference(inputs)
    seed[1] = 10.
    fixed["positive_m3"] = DD(np.full(4, 20.))
    system.material.poisson_factor.C[:] *= 5.
    system.material.N_D[:] = 10.
    system.material.D_ion_face[:] = 8.
    system.material.P_lim_node[:] = 90.
    system.grid[:] *= 2.
    system.widths[:] *= 3.
    system.trap_density[:] = 90.
    system._input_lift_reference = MappingProxyType({"phi_V": DD(np.ones(4))})
    exported = inputs.to_dict()
    exported["fixed_inputs"]["positive_m3"]["lo"][1] = 1.
    assert inputs.to_dict() == frozen
    actual, _ = reference.solve_rebased_ion_reference(inputs)
    assert _words(actual) == _words(expected)


def test_stale_coordinate_origin_identity_is_rejected(manufactured):
    system, fixed, seed = manufactured
    changed = dict(system._input_lift_reference)
    changed["n_m3"] = changed["n_m3"] + DD(np.array([0., 2.**-60, 0., 0.]))
    system._input_lift_reference = MappingProxyType(changed)
    with pytest.raises(ValueError):
        reference.make_inputs(system, fixed, seed, 0.)


@pytest.mark.parametrize("name,value", [
    ("dqfn_V", DD(np.zeros(3))), ("dqfp_V", DD(np.zeros((2, 2)))),
    ("positive_m3", DD(np.array([1., 0., 1., 1.]))),
    ("positive_m3", DD(np.array([1., -1., 1., 1.]))),
    ("positive_m3", DD(np.array([1., 100., 1., 1.]))),
    ("occupancy", DD(np.zeros(2))), ("occupancy", DD([-0.1])),
    ("occupancy", DD([1.1])),
])
def test_invalid_fixed_shapes_and_physical_domains_are_rejected(manufactured, name, value):
    system, fixed, seed = manufactured
    with pytest.raises((TypeError, ValueError)):
        reference.make_inputs(system, {**fixed, name: value}, seed, 0.)


@pytest.mark.parametrize("seed", [np.zeros(3), np.full(4, np.nan), np.full(4, np.inf)])
def test_bad_seed_is_rejected(manufactured, seed):
    system, fixed, _ = manufactured
    with pytest.raises((ArithmeticError, TypeError, ValueError)):
        reference.make_inputs(system, fixed, seed, 0.)


@pytest.mark.parametrize("name,value", [
    ("D_ion_face", np.ones(2)), ("D_ion_face", np.array([1., -1., 1.])),
    ("D_ion_face", np.array([1., np.nan, 1.])),
    ("P_lim_node", np.array([100., 0., 100., 100.])),
    ("V_T_device", 0.), ("V_T_device", np.inf),
])
def test_invalid_coefficients_are_rejected(manufactured, name, value):
    system, fixed, seed = manufactured
    setattr(system.material, name, value)
    with pytest.raises((ArithmeticError, TypeError, ValueError)):
        reference.make_inputs(system, fixed, seed, 0.)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_dto_constructor_rejects_nonfinite_words(manufactured, value):
    system, fixed, seed = manufactured
    inputs = reference.make_inputs(system, fixed, seed, 0.)
    raw = json.loads(inputs.fixed_inputs_json)
    raw["positive_m3"]["lo"][1] = value
    with pytest.raises((ArithmeticError, TypeError, ValueError)):
        replace(inputs, fixed_inputs_json=json.dumps(raw))


def test_dto_rejects_a_direct_output_seed_provenance(manufactured):
    system, fixed, seed = manufactured
    inputs = reference.make_inputs(system, fixed, seed, 0.)
    with pytest.raises(ValueError):
        replace(inputs, seed_provenance="current_trial_phi")


def test_solver_requires_the_validated_dto(manufactured):
    system, fixed, seed = manufactured
    inputs = reference.make_inputs(system, fixed, seed, 0.)
    with pytest.raises(TypeError):
        reference.solve_rebased_ion_reference(inputs.to_dict())


def test_integrated_comparison_preserves_floor_and_rejects_bad_direct_rate(manufactured, monkeypatch):
    from perovskite_sim.experiments.interface_defect_ion_transient import _InterfaceIonTransientSystem

    system, fixed, seed = manufactured
    solved, _ = reference.solve_rebased_ion_reference(reference.make_inputs(system, fixed, seed, 0.))
    primary = dict(system._input_lift_reference)
    primary.update(positive_flux_m2_s=solved["positive_flux_m2_s"],
                   positive_rate_m3_s=solved["positive_rate_m3_s"])
    state = replace(system._step_reference, input_lift=MappingProxyType(primary))
    # Supply the legacy diagnostic boundary; the independent production solve
    # and the real public-hi comparison remain active under test.
    unchanged = {"direct": seed.copy(), "eliminated": seed.copy(), "relative_error": 0.}
    legacy_channels = {}

    def legacy(self, current, voltage):
        values = {"potential": unchanged}
        for name, key in (("positive_ion_flux", "positive_flux_m2_s"),
                          ("positive_ion_rate", "positive_rate_m3_s")):
            old = {"direct": current.input_lift[key].hi.copy(), "eliminated": np.zeros_like(current.input_lift[key].hi),
                   "normalization_floor": 1., "unit": "original-unit", "relative_error": 2e-6}
            legacy_channels[name] = old
            values[name] = old
        return values

    monkeypatch.setattr(_InterfaceIonTransientSystem, "eliminated_operator_diagnostics", legacy)
    correct = system.eliminated_operator_diagnostics(state, 0.)
    assert correct["potential"] is unchanged
    assert correct["positive_ion_rate"]["relative_error"] < 1e-6
    assert correct["positive_ion_rate"]["legacy_binary64"] is legacy_channels["positive_ion_rate"]
    primary["positive_rate_m3_s"] = primary["positive_rate_m3_s"] + DD([0., 2e-6, 0., 0.])
    corrupted = replace(state, input_lift=MappingProxyType(primary))
    failed = system.eliminated_operator_diagnostics(corrupted, 0.)["positive_ion_rate"]
    assert failed["normalization_floor"] == failed["normalization_scale"] == 1.
    assert failed["relative_error"] > 1e-6
    assert failed["independent_reference"]["fields"] == correct["positive_ion_rate"]["independent_reference"]["fields"]
