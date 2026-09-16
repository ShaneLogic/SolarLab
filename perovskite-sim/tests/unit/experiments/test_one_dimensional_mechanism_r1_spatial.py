"""Analytic SG limits, conservative cell moments and fixed-position adapters."""

import copy

import numpy as np
import pytest

from perovskite_sim.constants import K_B, Q
from perovskite_sim.discretization.fe_operators import sg_fluxes_n, sg_fluxes_p
from perovskite_sim.experiments.one_dimensional_mechanism_r1_spatial import (
    ConservativeIonProfile, compare_spatial_responses, sample_r1_state,
    sample_sg_density, spatial_responses_from_step,
)
from perovskite_sim.physics.physical_control_volume import physical_cell_faces


@pytest.mark.parametrize("carrier,sign", [("electron", 1.), ("hole", -1.)])
def test_sg_reproduces_constant_field_zero_current_boltzmann_profile(carrier, sign):
    thermal = K_B*300/Q
    x = np.array([0., .1, .5, 1.])*1e-7
    potential = .09*x/x[-1]
    density = .1*np.exp(sign*potential/thermal)
    sample = np.linspace(0., x[-1], 101)
    result = sample_sg_density(x, density, potential, sample, thermal, carrier=carrier)
    expected = .1*np.exp(sign*.09*sample/x[-1]/thermal)
    np.testing.assert_allclose(result, expected, rtol=5e-15)


@pytest.mark.parametrize("carrier", ["electron", "hole"])
def test_sg_zero_field_is_linear_density_not_log_density(carrier):
    positions = np.array([0., .1, .5, 1.])
    result = sample_sg_density([0., 1.], [1., 1e12], [0., 0.], positions, .025, carrier=carrier)
    np.testing.assert_allclose(result, 1.+(1e12-1)*positions, rtol=1e-15)
    assert result[2] > 1e6


@pytest.mark.parametrize("carrier,sign,flux", [("electron", 1., sg_fluxes_n), ("hole", -1., sg_fluxes_p)])
def test_sg_manufactured_constant_nonzero_flux_survives_subdivision(carrier, sign, flux):
    thermal, drop, length, diffusion = .025, .03, 2e-7, 1e-6
    nodes = np.array([0., length])
    potential = np.array([0., drop])
    density = np.array([1e14, 7e14])
    positions = np.linspace(0., length, 51)
    sampled = sample_sg_density(nodes, density, potential, positions, thermal, carrier=carrier)
    # Independent solution of n(x)=A+B exp(sign*phi/Vt), with endpoint data.
    exponential = np.exp(sign*drop/thermal)
    b = (density[1]-density[0])/(exponential-1)
    a = density[0]-b
    np.testing.assert_allclose(sampled, a+b*np.exp(sign*drop*positions/length/thermal), rtol=3e-15)
    before = flux(potential, density, np.diff(nodes), diffusion, thermal)[0]
    after = flux(drop*positions/length, sampled, np.diff(positions), diffusion, thermal)
    assert before != 0
    np.testing.assert_allclose(after, before, rtol=1e-12)


def test_conservative_linear_reconstruction_matches_linear_cell_averages_and_moments():
    faces = np.array([0., .1, .4, .7, 1.])
    centers = (faces[:-1]+faces[1:])/2
    profile = ConservativeIonProfile(faces, 2.+3*centers)
    positions = np.linspace(0, 1, 101)
    np.testing.assert_allclose(profile.sample(positions), 2.+3*positions, rtol=1e-15)
    assert profile.moments() == pytest.approx((3.5, 2.))
    assert profile.moments(.2, .8) == pytest.approx((2*.6+1.5*(.8**2-.2**2),
                                                  .8**2-.2**2+(.8**3-.2**3)))
    for i, value in enumerate(profile.density_m3):
        mass, _ = profile.moments(faces[i], faces[i+1])
        assert mass == pytest.approx(value*(faces[i+1]-faces[i]), rel=1e-15)


def test_ion_limiters_keep_integral_without_clipping_density():
    faces = np.array([0., .01, .4, .45, 1.])
    average = np.array([0., 5., .001, 10.])
    profile = ConservativeIonProfile(faces, average)
    assert np.all(profile.sample(np.linspace(0., 1., 1001)) >= 0)
    np.testing.assert_array_equal(profile.density_m3, average)
    for i in range(len(average)):
        assert profile.moments(faces[i], faces[i+1])[0] == pytest.approx(average[i]*np.diff(faces)[i], abs=1e-15)
    assert profile.moments()[0] == pytest.approx(np.dot(average, np.diff(faces)))


def test_interface_break_preserves_distinct_limits_and_separate_inventory():
    profile = ConservativeIonProfile([0., .5, 1., 1.5, 2.], [2., 2., 200., 200.], (2,))
    assert profile.sample([1.], side="left")[0] == 2.
    assert profile.sample([1.], side="right")[0] == 200.
    assert profile.moments(0., 1.) == pytest.approx((2., 1.))
    assert profile.moments(1., 2.) == pytest.approx((200., 300.))
    np.testing.assert_array_equal(profile.slopes_m4, 0.)


def test_single_cell_and_physical_endpoints_are_defined():
    profile = ConservativeIonProfile([2., 4.], [3.])
    for side in ("left", "right"):
        np.testing.assert_array_equal(profile.sample([2., 3., 4.], side=side), 3.)
    assert profile.moments() == (6., 18.)


def test_strongly_nonuniform_limited_endpoints_remain_nonnegative_without_clipping():
    faces = [0., .4534195958885818, .45343896664184363, .46639534230533214,
             .4817572753632952, .4851573848462428, .4851638199959387]
    average = [320.373681135411, .12292808716414336, 420874.87121979526,
               2998490.8565391577, 60.17121601128989, 9.928300766344898e-6]
    profile = ConservativeIonProfile(faces, average)
    for side in ("left", "right"):
        assert np.all(profile.sample(faces, side=side) >= 0)
    assert profile.moments()[0] == pytest.approx(np.dot(average, np.diff(faces)), rel=3e-15)


def manufactured_prepared():
    x = np.array([0., .2, .65, 1.2, 1.65, 2.])
    faces = physical_cell_faces(x, [1.])
    phi = np.where(x < 1., .02*x, .1+.005*(x-1.))
    thermal = K_B*300/Q
    n = np.where(x < 1., 1e10, 1e12)*(1+np.exp(phi/thermal))
    p = np.where(x < 1., 3e11, 2e10)*(2+np.exp(-phi/thermal))
    trace_phi = np.array([[.02, .1]])
    trace = [[1e10*(1+np.exp(.02/thermal)), 3e11*(2+np.exp(-.02/thermal)),
              1e12*(1+np.exp(.1/thermal)), 2e10*(2+np.exp(-.1/thermal))]]
    centers = (faces[:-1]+faces[1:])/2
    ions = np.where(x < 1, 1.+centers, 10.)
    return {
        "schema": "R1CommonStateV1", "sha256": "prepared", "reference_sha256": "reference",
        "stack_sha256": "stack", "intervals": 6,
        "physical_stack": {"band_grading": False, "T": 300., "layers": [
            {"thickness": 1., "role": "absorber", "params": {"carrier_statistics": "maxwell_boltzmann"}},
            {"thickness": 1., "role": "ETL", "params": {"carrier_statistics": "maxwell_boltzmann"}},
        ]},
        "grid": {"coordinates_m": x.tolist(), "faces_m": faces.tolist(), "widths_m": np.diff(faces).tolist(),
                 "interface_positions_m": [1.], "positive_components": [[0, 1, 2]], "positive_nodes": [0, 1, 2],
                 "background_positive_m3": [2., 2., 2., 10., 10., 10.]},
        "state": {"n_m3": n.tolist(), "p_m3": p.tolist(), "phi_V": phi.tolist(), "positive_m3": ions.tolist(),
                  "trace_state_m3": trace, "trace_potential_V": trace_phi.tolist(), "occupancy": [.25]},
    }


def manufactured_step(prepared, *, times=(0., 1.)):
    base = prepared["state"]
    rows = []
    for level in (1, 2, 4):
        for time in times:
            state = copy.deepcopy(base)
            factor = 1. if time == 0 else 2.
            state["n_m3"] = (factor*np.array(base["n_m3"])).tolist()
            state["trace_state_m3"] = np.array(base["trace_state_m3"])
            state["trace_state_m3"][:, [0, 2]] *= factor
            state["trace_state_m3"] = state["trace_state_m3"].tolist()
            state["positive_m3"] = (np.array(base["positive_m3"])+np.array([time]*3+[0.]*3)).tolist()
            state["occupancy"] = [.25+.01*time]
            rows.append({"substeps": level, "time_s": time, "phase": "0+" if time == 0 else "accepted_regular_step",
                         "state": state})
    return {"schema": "R1ControlledStepV1", "prepared_sha256": prepared["sha256"],
            "reference_sha256": prepared["reference_sha256"], "intervals": prepared["intervals"],
            "times_s": list(times), "policy": {"refinement_substeps": [1, 2, 4]}, "accepted_steps": rows,
            "amplitude_V": .005, "control_label": "D", "certificate": {"certified": True}}


def test_fixed_positions_and_saved_carrier_interface_sides_match_manufactured_profiles():
    prepared = manufactured_prepared()
    samples = sample_r1_state(prepared)
    expected_position = np.r_[np.arange(1, 18)/18, 1.+np.arange(1, 18)/18]
    np.testing.assert_array_equal(samples["position_m"], expected_position)
    potential = np.where(expected_position < 1., .02*expected_position, .1+.005*(expected_position-1.))
    thermal = K_B*300/Q
    expected_n = np.where(expected_position < 1., 1e10, 1e12)*(1+np.exp(potential/thermal))
    expected_p = np.where(expected_position < 1., 3e11, 2e10)*(2+np.exp(-potential/thermal))
    np.testing.assert_allclose(samples["n_m3"], expected_n, rtol=5e-15)
    np.testing.assert_allclose(samples["p_m3"], expected_p, rtol=5e-15)
    np.testing.assert_allclose(samples["phi_V"], potential, rtol=1e-15)
    np.testing.assert_array_equal(samples["interface_carriers_m3"], prepared["state"]["trace_state_m3"])
    np.testing.assert_array_equal(samples["interface_phi_V"], [[.02, .1]])
    np.testing.assert_allclose(samples["interface_positive_m3"], [[2., 10.]], rtol=1e-15)
    assert samples["ion_inventory_m2"][0] == pytest.approx(1.5)
    assert samples["ion_centroid_m"][0] == pytest.approx((.5+1/3)/1.5)


def test_trace_changes_do_not_bleed_across_the_physical_interface():
    prepared = manufactured_prepared()
    before = sample_r1_state(prepared)
    altered = copy.deepcopy(prepared["state"])
    altered["trace_state_m3"][0][2] *= 1000
    after = sample_r1_state(prepared, altered)
    np.testing.assert_array_equal(before["n_m3"][:17], after["n_m3"][:17])
    assert after["n_m3"][17] > before["n_m3"][17]


def test_response_adapter_uses_own_dc_log_changes_and_p0_not_voltage_normalization():
    prepared = manufactured_prepared()
    result = spatial_responses_from_step(prepared, manufactured_step(prepared))
    np.testing.assert_array_equal(result.responses["carrier_log_density"].values[0], 0.)
    np.testing.assert_allclose(result.responses["carrier_log_density"].values[1, :, 0], np.log(2.), rtol=1e-14)
    np.testing.assert_array_equal(result.responses["carrier_log_density"].values[1, :, 1], 0.)
    np.testing.assert_allclose(result.responses["ion_density_over_p0"].values[1, :17], .5)
    np.testing.assert_array_equal(result.responses["ion_density_over_p0"].values[1, 17:], 0.)
    assert result.responses["trap_occupancy_change"].values[1, 0] == pytest.approx(.01)
    np.testing.assert_allclose(result.inventory_m2[:, 0], [1.5, 2.5])
    assert result.interface_responses["carrier_log_density"].components == ("n_left", "p_left", "n_right", "p_right")


def test_zero_plus_is_distinct_from_zero_minus_baseline():
    prepared = manufactured_prepared()
    step = manufactured_step(prepared)
    for row in step["accepted_steps"]:
        if row["time_s"] == 0.:
            row["state"]["phi_V"] = (np.array(row["state"]["phi_V"])+.001).tolist()
            row["state"]["trace_potential_V"] = [[.021, .101]]
    result = spatial_responses_from_step(prepared, step)
    np.testing.assert_allclose(result.responses["response_potential"].values[0], .001, rtol=1e-13)


def test_comparison_keeps_interface_side_failures_separate_and_uses_existing_spec_budget():
    prepared = manufactured_prepared()
    step = manufactured_step(prepared)
    left = spatial_responses_from_step(prepared, step)
    changed = copy.deepcopy(step)
    for row in changed["accepted_steps"]:
        if row["time_s"] > 0:
            row["state"]["trace_state_m3"][0][0] *= np.exp(.02)
    right = spatial_responses_from_step(prepared, changed)
    report = compare_spatial_responses(left, right)
    assert not report["within_compared_budgets"]
    side = report["interface_sides"]["carrier_log_density"]
    assert not side["passed"]
    assert side["absolute_tolerance"] == .01
    assert side["failures"][0]["component"] == "n_left"
    assert report["dc_potential"]["passed"]


def test_identical_reconstruction_is_only_a_scoped_comparison():
    prepared = manufactured_prepared()
    result = spatial_responses_from_step(prepared, manufactured_step(prepared))
    report = compare_spatial_responses(result, result)
    assert report["within_compared_budgets"]
    assert "certificate" not in report
    assert "undetermined" in result.metadata["carrier_resolution"]


@pytest.mark.parametrize("key,value", [("stack_sha256", "other"), ("reference_sha256", "other")])
def test_comparison_rejects_different_physical_identity(key, value):
    prepared = manufactured_prepared()
    left = spatial_responses_from_step(prepared, manufactured_step(prepared))
    changed = copy.deepcopy(prepared)
    changed[key] = value
    right = spatial_responses_from_step(changed, manufactured_step(changed))
    with pytest.raises(ValueError, match="identity differs"):
        compare_spatial_responses(left, right)


def test_comparison_never_interpolates_different_times():
    prepared = manufactured_prepared()
    left = spatial_responses_from_step(prepared, manufactured_step(prepared))
    right = spatial_responses_from_step(prepared, manufactured_step(prepared, times=(0., 2.)))
    with pytest.raises(ValueError, match="coordinate values differ"):
        compare_spatial_responses(left, right)


def test_comparison_rejects_source_changes_between_otherwise_identical_cases():
    prepared = manufactured_prepared()
    prepared["source"] = {"sha256": "first_source"}
    step = manufactured_step(prepared)
    step["source"] = copy.deepcopy(prepared["source"])
    left = spatial_responses_from_step(prepared, step)
    prepared["source"] = {"sha256": "second_source"}
    with pytest.raises(ValueError, match="identities disagree"):
        spatial_responses_from_step(prepared, step)
    step["source"] = copy.deepcopy(prepared["source"])
    right = spatial_responses_from_step(prepared, step)
    with pytest.raises(ValueError, match="source_sha256"):
        compare_spatial_responses(left, right)


@pytest.mark.parametrize("mutation", [
    lambda p: p["grid"]["widths_m"].__setitem__(0, 123.),
    lambda p: p["grid"]["interface_positions_m"].__setitem__(0, .9),
    lambda p: p["grid"].__setitem__("positive_nodes", [0, 1]),
    lambda p: p["grid"].__setitem__("positive_components", [[0, 2]]),
    lambda p: p["state"]["n_m3"].__setitem__(1, 0.),
    lambda p: p["state"]["positive_m3"].__setitem__(1, -1.),
    lambda p: p["state"]["phi_V"].__setitem__(1, np.nan),
    lambda p: p["state"].__setitem__("trace_state_m3", [[1., 2.]]),
    lambda p: p["state"].__setitem__("occupancy", [1.01]),
    lambda p: p["physical_stack"].__setitem__("band_grading", True),
    lambda p: p["physical_stack"]["layers"][0]["params"].__setitem__("carrier_statistics", "fermi_dirac"),
])
def test_adapter_rejects_invalid_geometry_and_states(mutation):
    prepared = manufactured_prepared()
    mutation(prepared)
    with pytest.raises(ValueError):
        sample_r1_state(prepared)


@pytest.mark.parametrize("mutation", [
    lambda s: s.__setitem__("prepared_sha256", "other"),
    lambda s: s.__setitem__("times_s", [0., 1.1]),
    lambda s: s["accepted_steps"].append(copy.deepcopy(s["accepted_steps"][-1])),
    lambda s: s["accepted_steps"][-1].__setitem__("phase", "0+"),
])
def test_step_adapter_rejects_unmatched_state_times_and_identity(mutation):
    prepared = manufactured_prepared()
    step = manufactured_step(prepared)
    mutation(step)
    with pytest.raises(ValueError):
        spatial_responses_from_step(prepared, step)


@pytest.mark.parametrize("faces,density,breaks", [
    ([0., 0.], [1.], ()), ([0., 1.], [-1.], ()), ([0., 1.], [np.nan], ()),
    ([0., 1.], [1., 2.], ()), ([0., 1.], [True], ()),
    ([0., .5, 1.], [1., 2.], (0,)), ([0., .5, 1.], [1., 2.], (1, 1)),
])
def test_conservative_profile_rejects_invalid_inputs(faces, density, breaks):
    with pytest.raises(ValueError):
        ConservativeIonProfile(faces, density, breaks)


def test_profile_and_samples_do_not_mutate_inputs():
    faces, density = np.array([0., .5, 1.]), np.array([1., 2.])
    profile = ConservativeIonProfile(faces, density)
    faces[:] = 5.
    density[:] = 10.
    np.testing.assert_array_equal(profile.faces_m, [0., .5, 1.])
    np.testing.assert_array_equal(profile.density_m3, [1., 2.])
    with pytest.raises(ValueError):
        profile.density_m3[0] = 0.
    samples = sample_r1_state(manufactured_prepared())
    with pytest.raises(ValueError):
        samples["n_m3"][0] = 0.


@pytest.mark.slow
def test_real_prepared_and_short_step_adapter():
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import prepare_common_state
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import run_r1_step, r1_policy
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from tests.fixtures.r1_reference import approved_r1_binding
    from tests.integration.test_one_dimensional_mechanism_r1 import FIXTURE

    stack, binding = load_device_from_yaml(FIXTURE), approved_r1_binding()
    prepared = prepare_common_state(stack, 4, binding)
    step = run_r1_step(stack, 4, binding, prepared, times_s=[0., 1e-9], policy=r1_policy(nonlinear_factor=.1))
    result = spatial_responses_from_step(prepared, step)
    assert result.responses["carrier_log_density"].values.shape == (2, 34, 2)
    np.testing.assert_allclose(result.inventory_m2, 1e15, rtol=1e-10)
    assert result.metadata["recorded_certificate_passed"]
