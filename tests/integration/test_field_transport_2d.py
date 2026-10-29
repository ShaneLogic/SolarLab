"""Normal-face CT values, coefficient chains and explicit supported scope."""

from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from solarlab.equations.field_transport_2d import bind_field_transport_2d
from solarlab.mesh.geometry import half_endpoint_geometry
from solarlab.mesh.geometry_2d import RectangularGeometry2D
from solarlab.physics.transport.electrochemical_flux import stable_sg_particle_flux
from solarlab.physics.transport.field_mobility import apply_field_mobility
from solarlab.physics.transport.field_mobility_tangents import FACE_NORMAL_COMPONENT_2D, FieldMobilityCapabilityError


def case():
    x = np.array([0.0, .13, .41, 1.0]) * 1e-6
    y = np.array([0.0, .09, .37, .55]) * 1e-6
    geometry = RectangularGeometry2D(x, half_endpoint_geometry(y))
    ids = np.arange(16).reshape(4, 4)
    dn, dp = .002 * (1 + ids / 24), .0009 * (1 + ids / 30)
    left, right = geometry.contract_geometry.pairs.T
    def harmonic(d):
        return 2 * d.ravel()[left] * d.ravel()[right] / (d.ravel()[left] + d.ravel()[right] + 1e-300)
    nodes = {"v_sat_n": np.linspace(1200, 2500, 16), "v_sat_p": np.linspace(1600, 2800, 16),
             "ct_beta_n": np.linspace(1.4, 2.2, 16), "ct_beta_p": np.linspace(1.6, 2.4, 16),
             "pf_gamma_n": np.zeros(16), "pf_gamma_p": np.zeros(16)}
    model = bind_field_transport_2d(geometry, harmonic(dn), harmonic(dp), .02585, **nodes)
    return geometry, dn, dp, nodes, model


def legacy_module():
    """Read the immutable legacy helper alone, never invoke a legacy solver."""
    root = Path(__file__).resolve().parents[2]
    path = root / "perovskite-sim/perovskite_sim/twod/field_mobility_2d.py"
    source = path.read_text().replace("from perovskite_sim.physics.field_mobility import apply_field_mobility",
                                    "from solarlab.physics.transport.field_mobility import apply_field_mobility")
    spec = importlib.util.spec_from_loader("legacy_normal_face_arithmetic", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def test_values_and_parameter_face_means_match_legacy_x_y():
    g, dn, dp, nodes, model = case()
    xx, yy = np.meshgrid(g.x_m, g.y_geometry.nodes_m)
    phi = (1.8e5 * xx + 5.2e5 * yy + 8e10 * xx * yy).ravel()
    actual = model.evaluate(phi)
    old = legacy_module()
    kwargs = {}
    for name, values in nodes.items():
        stem = name.replace("pf_gamma", "pf_gamma")
        kwargs[stem + "_x_face"] = old.arith_mean_face_x(values.reshape(g.shape))
        kwargs[stem + "_y_face"] = old.arith_mean_face_y(values.reshape(g.shape))
    expected = old.recompute_d_eff_2d(phi=phi.reshape(g.shape), x=g.x_m, y=g.y_geometry.nodes_m,
        D_n=dn, D_p=dp, V_T=model.thermal_voltage_V, **kwargs)
    np.testing.assert_array_equal(actual.electron_diffusivity_m2_s, np.r_[expected.D_n_x.ravel(), expected.D_n_y.ravel()])
    np.testing.assert_array_equal(actual.hole_diffusivity_m2_s, np.r_[expected.D_p_x.ravel(), expected.D_p_y.ravel()])
    np.testing.assert_array_equal(model.electron.saturation_velocity_m_s,
        np.r_[old.arith_mean_face_x(nodes["v_sat_n"].reshape(g.shape)).ravel(), old.arith_mean_face_y(nodes["v_sat_n"].reshape(g.shape)).ravel()])
    assert model.field_measure == FACE_NORMAL_COMPONENT_2D
    assert np.all(actual.electron_diffusivity_m2_s < model.base_electron_diffusivity_m2_s)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_normal_component_only_and_zero_field_smoothness(axis):
    g, _, _, _, model = case()
    xx, yy = np.meshgrid(g.x_m, g.y_geometry.nodes_m)
    phi = 9e5 * (xx if axis == "x" else yy)
    actual = model.evaluate(phi.ravel())
    quiet = slice(g.x_face_count, None) if axis == "x" else slice(None, g.x_face_count)
    np.testing.assert_array_equal(actual.field_V_m[quiet], 0.0)
    np.testing.assert_array_equal(actual.electron_d_diffusivity_d_field[quiet], 0.0)
    np.testing.assert_array_equal(actual.electron_diffusivity_m2_s[quiet],
        model.base_electron_diffusivity_m2_s[quiet] / model.thermal_voltage_V * model.thermal_voltage_V)
    opposite = model.evaluate(-phi.ravel())
    np.testing.assert_array_equal(actual.electron_diffusivity_m2_s, opposite.electron_diffusivity_m2_s)
    np.testing.assert_array_equal(actual.electron_d_diffusivity_d_field, -opposite.electron_d_diffusivity_d_field)


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("potential_only", [False, True])
def test_sg_density_and_potential_directional_partials_require_ct_chain(axis, potential_only):
    g, _, _, _, model = case()
    xx, yy = np.meshgrid(g.x_m, g.y_geometry.nodes_m)
    phi = (.25e6 * xx + .48e6 * yy).ravel()
    n, p = np.linspace(1e20, 2e20, 16), np.linspace(4e19, 8e19, 16)
    chi, eg = np.linspace(-.04, .03, 16), np.linspace(.02, -.02, 16)
    rng = np.random.default_rng(471)
    dphi = .02 * rng.normal(size=16)
    dn, dp = n * .05 * rng.normal(size=16), p * .05 * rng.normal(size=16)
    if potential_only:
        dn[:] = dp[:] = 0.0
    left, right = g.contract_geometry.pairs.T
    selected = slice(None, g.x_face_count) if axis == "x" else slice(g.x_face_count, None)
    def flux(h):
        v = phi + h * dphi
        return model.fluxes(v, n + h * dn, p + h * dp, v + chi, v + chi + eg)
    value = flux(0.0)
    step = 2e-4
    trials = [flux(h * step) for h in (-2, -1, 1, 2)]
    evaluation = model.evaluate(phi)
    for name, density, direction, diffusivity_slope in (("electron", n, dn, evaluation.electron_d_diffusivity_d_field),
                                                        ("hole", p, dp, evaluation.hole_d_diffusivity_d_field)):
        f = getattr(value, name)
        numeric = (getattr(trials[0], name).flux - 8 * getattr(trials[1], name).flux
                   + 8 * getattr(trials[2], name).flux - getattr(trials[3], name).flux) / (12 * step)
        analytic = (f.left_density_derivative * direction[left] + f.right_density_derivative * direction[right]
                    + f.left_potential_derivative * dphi[left] + f.right_potential_derivative * dphi[right])
        scale = np.maximum(abs(f.left_density_derivative * direction[left]) + abs(f.right_density_derivative * direction[right])
                           + abs(f.left_potential_derivative * dphi[left]) + abs(f.right_potential_derivative * dphi[right]), 1e-20)
        assert np.max(abs(numeric[selected] - analytic[selected]) / scale[selected]) < 2e-8
        raw = stable_sg_particle_flux(density[left], density[right], f.energy_step, f.coefficient)
        chain = raw.coefficient_derivative * diffusivity_slope / g.face_spacing_m**2
        omitted = analytic - chain * (dphi[left] - dphi[right])
        assert np.max(abs(numeric[selected] - omitted[selected]) / scale[selected]) > 1e-4
        integrated = np.zeros(g.node_count)
        np.add.at(integrated, left, -f.flux * g.contract_geometry.face_measures)
        np.add.at(integrated, right, f.flux * g.contract_geometry.face_measures)
        assert abs(np.sum(integrated)) <= 5e-15 * np.sum(abs(integrated))


def test_disabled_binding_returns_none_without_einstein_or_averaging_roundtrip():
    g, _, _, nodes, model = case()
    nodes["v_sat_n"][:] = nodes["v_sat_p"][:] = 0.0
    before_n, before_p = model.base_electron_diffusivity_m2_s.copy(), model.base_hole_diffusivity_m2_s.copy()
    assert bind_field_transport_2d(g, before_n, before_p, model.thermal_voltage_V, **nodes) is None
    np.testing.assert_array_equal(before_n, model.base_electron_diffusivity_m2_s)
    np.testing.assert_array_equal(before_p, model.base_hole_diffusivity_m2_s)


@pytest.mark.parametrize("name,value", [("pf_gamma_n", 1e-4), ("pf_gamma_p", 1e-4), ("ct_beta_n", 1.0),
                                        ("ct_beta_p", .5), ("v_sat_n", -1.0)])
def test_unsupported_native_laws_refused(name, value):
    g, _, _, nodes, model = case()
    nodes[name][:] = value
    with pytest.raises(FieldMobilityCapabilityError):
        bind_field_transport_2d(g, model.base_electron_diffusivity_m2_s, model.base_hole_diffusivity_m2_s,
            model.thermal_voltage_V, **nodes)


def test_field_measure_geometry_density_and_identity_are_explicit():
    g, _, _, _, model = case()
    with pytest.raises(FieldMobilityCapabilityError):
        replace(model, field_measure="total_field_magnitude_2d_v1")
    with pytest.raises(FieldMobilityCapabilityError):
        replace(model, field_measure="face_field_1d_v1")
    with pytest.raises(ValueError):
        model.evaluate(np.zeros(g.shape))
    with pytest.raises(ValueError):
        model.evaluate(np.zeros(g.node_count, dtype=complex))
    with pytest.raises(ValueError):
        model.fluxes(np.zeros(16), np.zeros(16), np.ones(16), np.zeros(16), np.zeros(16))
    assert model.identity_sha256 != replace(model, thermal_voltage_V=1.01 * model.thermal_voltage_V).identity_sha256
    for value in (model.base_electron_diffusivity_m2_s, model.electron.ct_exponent):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    field = model.evaluate(np.arange(16) * .01)
    expected = apply_field_mobility(model.base_electron_diffusivity_m2_s / model.thermal_voltage_V,
        field.field_V_m, model.electron.saturation_velocity_m_s, model.electron.ct_exponent, np.zeros(24)) * model.thermal_voltage_V
    np.testing.assert_array_equal(field.electron_diffusivity_m2_s, expected)


def integrated_case(*, grain=False, contact=False, active=True):
    """Used only against the isolated committed+contact+CT candidate."""
    from examples.research_1d.steady import load_device
    from examples.research_2d.contacts import contact_device
    from examples.research_2d.grains import spatial_input as grain_input
    from examples.research_2d.dc_assembly import spatial_input
    from solarlab.config.resolve_device import resolve_device
    from solarlab.device.compile_2d import compile_device_2d
    from solarlab.device.inputs import DeviceInput
    from solarlab.equations.dc_2d import CompiledDevice2D
    if "field_transport" not in CompiledDevice2D.__dataclass_fields__:
        pytest.skip("shared CT hooks pending; run this against the isolated composed candidate")
    prepared = load_device()
    if active:
        data = prepared.to_input().editing_data()
        data["layers"][1]["parameters"].update(v_sat_n=1e4, v_sat_p=1.5e4, ct_beta_n=2.0, ct_beta_p=2.0)
        prepared = resolve_device(DeviceInput.model_validate(data), prepared.defaults, prepared.resources, sources=prepared.sources)
    if contact:
        prepared = contact_device(prepared=prepared)
    return compile_device_2d(prepared, grain_input() if grain else spatial_input())


@pytest.mark.parametrize("grain,contact", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("potential_only", [False, True])
def test_composed_full195_rows_and_grain_contact_preservation(grain, contact, potential_only):
    from examples.research_2d.dc_assembly import coordinates
    from solarlab.equations.dc_2d import DC2DProblem
    from solarlab.solvers.nonlinear.steady_2d import _scope
    d = integrated_case(grain=grain, contact=contact)
    _scope(d)
    assert d.field_transport is not None and d.node_count == 65
    _, u = coordinates(d, lateral=True)
    u[:65] += np.tile(np.linspace(0.0, .06, 5), 13)
    problem = DC2DProblem(d, .002, 0.0)
    matrix = problem.jacobian(u)
    direction = np.r_[.02 * np.random.default_rng(492).normal(size=65),
                      np.random.default_rng(494).normal(size=130)]
    if potential_only:
        direction[65:] = 0.0
    h = 2e-4
    numeric = (problem.residual(u-2*h*direction) - 8*problem.residual(u-h*direction)
               + 8*problem.residual(u+h*direction) - problem.residual(u+2*h*direction)) / (12*h)
    scale = np.maximum(abs(matrix) @ abs(direction), 1e-20)
    assert np.max(abs(numeric - matrix @ direction) / scale) < 2e-8
    np.testing.assert_array_equal(matrix.toarray(), DC2DProblem(d, .002, 1.0).jacobian(u).toarray())
    off = integrated_case(grain=grain, contact=contact, active=False)
    constant = DC2DProblem(off, .002, 0.0)
    np.testing.assert_array_equal(problem.residual(u)[:65], constant.residual(u)[:65])
    fixed = np.r_[d.geometry.contact_nodes, 65 + np.flatnonzero(d.fixed_carrier_mask[0]),
                  130 + np.flatnonzero(d.fixed_carrier_mask[1])]
    np.testing.assert_array_equal(problem.jacobian(u).toarray()[fixed], constant.jacobian(u).toarray()[fixed])
    assert [c.kind for c in d.column.channels] == [c.kind for c in off.column.channels]
    if grain:
        assert d.grain_recombination.model_id == off.grain_recombination.model_id
    actual = d.field_transport.evaluate(u[:65])
    from types import SimpleNamespace
    held = SimpleNamespace(**{name: getattr(d.field_transport, name) for name in (
        "geometry", "thermal_voltage_V", "base_electron_diffusivity_m2_s", "base_hole_diffusivity_m2_s")})
    held.evaluate = lambda phi: replace(actual, electron_d_diffusivity_d_field=np.zeros(actual.field_V_m.size),
        hole_d_diffusivity_d_field=np.zeros(actual.field_V_m.size))
    held.fluxes = lambda *args, **kwargs: d.field_transport.fluxes(*args, **kwargs)
    omitted = DC2DProblem(replace(d, field_transport=held), .002, 0.0).jacobian(u)
    assert np.max(abs(numeric - omitted @ direction) / scale) > 1e-4


def test_composed_active_source_guard_has_actual_law_hashes_and_off_stays_old(monkeypatch):
    import hashlib
    import sys
    from solarlab.solvers.nonlinear.steady_2d import _sources
    active, off = integrated_case(), integrated_case(active=False)
    names = ("solarlab.equations.field_transport_2d", "solarlab.physics.transport.field_mobility",
             "solarlab.physics.transport.field_mobility_tangents")
    capture = _sources(active)
    for name in names:
        assert capture[name] == hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest()
        assert name not in _sources(off)
    assert _sources(off) == _sources()
    old = capture[names[0]]
    module = sys.modules[names[0]]
    monkeypatch.setattr(module, "__file__", sys.modules[names[1]].__file__)
    assert _sources(active)[names[0]] != old


def test_example_configuration_is_explicit_and_preparation_performs_no_solve(tmp_path, monkeypatch):
    import hashlib
    from examples.research_2d import field_transport as example
    config = tmp_path / "device.json"
    config.write_bytes((Path(__file__).resolve().parents[2] / "examples/research_1d/device.json").read_bytes())
    monkeypatch.setattr(example, "solve_steady_state_2d", lambda *args, **kwargs: pytest.fail("preparation called Newton"))
    prepared = example.prepare_case(config)
    assert [(source.id, source.sha256) for source in prepared.sources] == [
        (str(config), hashlib.sha256(config.read_bytes()).hexdigest())]
    parameters = dict(prepared.layers[1].parameters)
    assert (parameters["v_sat_n"], parameters["v_sat_p"], parameters["ct_beta_n"], parameters["ct_beta_p"]) == (
        1e4, 1.5e4, 2.0, 2.0)
    assert parameters["pf_gamma_n"] == parameters["pf_gamma_p"] == 0.0
