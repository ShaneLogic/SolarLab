"""Normative response budgets, independent axes and explicit unresolved signals."""

import json

import numpy as np
import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import (
    AMPLITUDES_V, R1ConvergenceCase, R1Response, base_convergence_cases, compare_amplitude_halving,
    compare_responses, convergence_cases, observation_times,
)


# Independently transcribed from StudySpecV1 sections 6 and 10.2. Do not import
# the implementation's rule table or derive these limits from its reports:
# these values are the external numerical policy against which it is tested.
SPEC_AMPLITUDES_V = (.01, .005, .0025, .00125, .000625, .0003125, .00015625)
SPEC_RESPONSE_RULES = [
    ("dc_potential", 1e-3, 0., "V"),
    ("response_potential", 5e-5, .01, "V"),
    ("carrier_log_density", .01, 0., "1"),
    ("ion_density_over_p0", .01, 0., "1"),
    ("trap_occupancy_change", 1e-7, .01, "1"),
    ("ion_centroid_change", 5e-11, .01, "m"),
    ("regular_current_response", 1e-7, .005, "A/m2"),
    ("integrated_charge_response", 1e-10, .005, "C/m2"),
    ("admittance", 1e-8, .01, "S/m2"),
]


def response(values, *, times=None, components=()):
    values = np.asarray(values)
    times = np.arange(values.shape[0], dtype=float) if times is None else times
    return R1Response(values, {"time_s": times}, components)


def test_observation_grid_contains_zero_and_twelve_intervals_per_decade():
    times = observation_times()
    assert times.shape == (134,)
    assert times[0] == 0. and times[1] == 1e-9 and times[-1] == 100.
    np.testing.assert_array_equal(times[1::12],
                                  [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, .001, .01, .1, 1., 10., 100.])
    assert np.all(np.diff(times) > 0)
    extended = observation_times(first_time_s=1e-12, last_time_s=1e5)
    assert len(extended) == 206
    np.testing.assert_array_equal(extended[37:170], times[1:])


@pytest.mark.parametrize("kwargs", [
    {"first_time_s": 0.}, {"first_time_s": 1e-13}, {"first_time_s": 2e-9},
    {"last_time_s": 1e6}, {"last_time_s": 200.}, {"last_time_s": np.inf},
    {"first_time_s": True},
])
def test_observation_grid_rejects_undeclared_endpoints(kwargs):
    with pytest.raises(ValueError):
        observation_times(**kwargs)


def test_base_d_matrix_has_twenty_seven_distinct_independent_settings():
    cases = base_convergence_cases()
    assert len(cases) == len(set(cases)) == 27
    assert {case.intervals for case in cases} == {16, 32, 64}
    assert {case.time_substeps for case in cases} == {(1, 2, 4), (2, 4, 8), (4, 8, 16)}
    assert {case.nonlinear_factor for case in cases} == {1., .1, .01}
    assert {(case.control, case.amplitude_V) for case in cases} == {("D", .005)}
    assert sum(case.time_substeps == (1, 2, 4) for case in cases) == 9


@pytest.mark.parametrize("control", list("ABC"))
def test_other_base_controls_have_seven_cases_crossing_the_finest_setting(control):
    cases = base_convergence_cases(control)
    assert len(cases) == len(set(cases)) == 7
    assert R1ConvergenceCase(64, (4, 8, 16), .01, control) in cases
    for n in (16, 32, 64):
        assert R1ConvergenceCase(n, (4, 8, 16), .01, control) in cases
    for time in ((1, 2, 4), (2, 4, 8), (4, 8, 16)):
        assert R1ConvergenceCase(64, time, .01, control) in cases
    for factor in (1., .1, .01):
        assert R1ConvergenceCase(64, (4, 8, 16), factor, control) in cases


def test_extensions_are_explicit_and_bounded():
    cases = convergence_cases(intervals=(128, 256), time_substeps=((8, 16, 32), (16, 32, 64)),
                              nonlinear_factors=(.001,))
    assert len(cases) == 4
    assert R1ConvergenceCase(256, (16, 32, 64), .001) in cases
    for args in ((512, (1, 2, 4), .1), (64, (32, 64, 128), .1), (64, (1, 2, 4), .0001)):
        with pytest.raises(ValueError):
            R1ConvergenceCase(*args)
    with pytest.raises(TypeError):
        R1ConvergenceCase(64., (1, 2, 4), .1)
    with pytest.raises(ValueError):
        convergence_cases(intervals=(16, 16))


@pytest.mark.parametrize("quantity,absolute,relative,units", SPEC_RESPONSE_RULES)
def test_each_spec_rule_preserves_its_absolute_relative_and_unit_budget(quantity, absolute, relative, units):
    passed = compare_responses(quantity, response([0.]), response([absolute]))
    failed = compare_responses(quantity, response([0.]), response([2*absolute]))
    assert passed["passed"] and not failed["passed"]
    assert passed["absolute_tolerance"] == absolute
    assert passed["relative_tolerance"] == relative
    assert passed["units"] == units


@pytest.mark.parametrize("quantity,absolute,relative,units", SPEC_RESPONSE_RULES)
@pytest.mark.parametrize("scale_in_absolute_units", [2., 1e6], ids=["absolute_dominated", "large_signal"])
def test_spec_acceptance_boundary_and_report_use_the_same_effective_budget(
        quantity, absolute, relative, units, scale_in_absolute_units):
    # Keep the larger operand fixed. The section-10.2 boundary then has a
    # known value independent of the comparator. The two probes are 1 ppm
    # inside/outside it, well clear of float64 subtraction roundoff, including
    # the zero-relative rules at a large nonzero baseline.
    scale = absolute*scale_in_absolute_units
    limit = absolute + relative*scale
    parts = [(1., None)] if quantity != "admittance" else [(1., "real"), (1j, "imaginary")]
    for multiplier, expected in ((1-1e-6, True), (1+1e-6, False)):
        for sign in (-1., 1.):
            for factor, part in parts:
                larger = sign*scale*factor
                smaller = sign*(scale-multiplier*limit)*factor
                for left, right in ((larger, smaller), (smaller, larger)):
                    result = compare_responses(quantity, response([left]), response([right]))
                    assert result["passed"] is expected
                    assert result["absolute_tolerance"] == absolute
                    assert result["relative_tolerance"] == relative
                    assert result["units"] == units
                    assert result["maximum_budget_ratio"] == pytest.approx(multiplier, rel=1e-9)
                    if expected:
                        assert result["failures"] == []
                    else:
                        assert result["failure_count"] == 1
                        failure = result["failures"][0]
                        assert failure["reasons"] == ["response_difference_exceeds_budget"]
                        if part is not None:
                            assert failure["part"] == part
                        assert failure["response_scale"] == scale
                        assert failure["allowed_difference"] == limit
                        # This is the budget used by the comparison arithmetic,
                        # not just the static labels in its top-level report.
                        assert failure["allowed_difference"] == (
                            result["absolute_tolerance"]
                            + result["relative_tolerance"]*failure["response_scale"])


@pytest.mark.parametrize("quantity,absolute,relative,units", SPEC_RESPONSE_RULES)
def test_twenty_eight_percent_disagreement_fails_every_spec_rule(quantity, absolute, relative, units):
    # Review counterexample: an actual .499 relative tolerance can accept this
    # pair while a report still prints the correct 0/0.005/0.01 policy.
    result = compare_responses(quantity, response([1e6*absolute]), response([5e6*absolute/7]))
    assert not result["passed"]
    assert result["failure_count"] == 1
    assert result["absolute_tolerance"] == absolute
    assert result["relative_tolerance"] == relative
    assert result["units"] == units
    assert result["failures"][0]["allowed_difference"] == absolute + relative*(1e6*absolute)


@pytest.mark.parametrize("side,passed", [(-1, True), (0, True), (1, False)])
def test_carrier_log_density_limit_is_inclusive(side, passed):
    value = .01 if side == 0 else np.nextafter(.01, -np.inf if side < 0 else np.inf)
    result = compare_responses("carrier_log_density", response([0.]), response([value]))
    assert result["passed"] is passed


def test_early_current_peak_cannot_hide_late_response_failure():
    times = [1e-9, 100.]
    coarse = response([1e5, 1e-9], times=times)
    fine = response([1e5+100., 5e-7], times=times)
    result = compare_responses("regular_current_response", coarse, fine)
    assert not result["passed"] and result["failure_count"] == 1
    failure = result["failures"][0]
    assert failure["index"] == [1]
    assert failure["coordinates"] == {"time_s": 100.}
    assert failure["allowed_difference"] == pytest.approx(1.025e-7)


def test_component_and_position_are_preserved_in_a_failure():
    coordinates = {"time_s": [0., 1.], "position_m": [1e-8, 2e-8]}
    left = R1Response(np.zeros((2, 2, 2)), coordinates, ("electron", "hole"))
    values = np.zeros((2, 2, 2)); values[1, 0, 1] = .02
    result = compare_responses("carrier_log_density", left,
                               R1Response(values, coordinates, ("electron", "hole")))
    assert result["failures"][0]["component"] == "hole"
    assert result["failures"][0]["coordinates"] == {"time_s": 1., "position_m": 1e-8}


def test_admittance_real_and_imaginary_parts_cannot_mask_one_another():
    coarse = response([1j*1e9, 1e9+0j])
    fine = response([2e-8+1j*1e9, 1e9+2e-8j])
    result = compare_responses("admittance", coarse, fine)
    assert {(f["index"][0], f["part"]) for f in result["failures"]} == {(0, "real"), (1, "imaginary")}
    assert result["scalar_comparison_count"] == 4


def test_nonfinite_admittance_part_has_its_own_failure_location():
    result = compare_responses("admittance", response([1+2j]), response([complex(1., np.inf)]))
    assert result["failure_count"] == 1
    assert result["failures"][0]["part"] == "imaginary"
    assert result["failures"][0]["reasons"] == ["nonfinite_response"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_responses_fail_at_their_point_and_remain_json_safe(value):
    result = compare_responses("response_potential", response([0., 0.]), response([0., value]))
    assert result["failure_count"] == 1
    assert result["failures"][0]["index"] == [1]
    assert result["failures"][0]["reasons"] == ["nonfinite_response"]
    assert "nonfinite" in result["failures"][0]["right"]
    json.dumps(result, allow_nan=False)


def test_finite_inputs_that_overflow_comparison_are_rejected():
    result = compare_responses("response_potential", response([1e308]), response([-1e308]))
    assert not result["passed"]
    assert result["failures"][0]["reasons"] == ["nonfinite_comparison_arithmetic"]
    json.dumps(result, allow_nan=False)


def test_coordinates_components_and_shapes_require_exact_identity():
    left = response([0., 0.], times=[0., 1.])
    with pytest.raises(ValueError, match="coordinate values differ"):
        compare_responses("response_potential", left, response([0., 0.], times=[0., np.nextafter(1., 2.)]))
    with pytest.raises(ValueError, match="shapes differ"):
        compare_responses("response_potential", left, response([0.]))
    with pytest.raises(ValueError, match="shape"):
        R1Response(np.zeros((2, 1)), {"time_s": [0., 1.]})
    with pytest.raises(ValueError, match="component identities"):
        compare_responses("carrier_log_density", response([[0., 0.]], components=("n", "p")),
                           response([[0., 0.]], components=("p", "n")))
    with pytest.raises(ValueError, match="coordinate"):
        response([0., 0.], times=[0., np.nan])


def test_response_keeps_a_readonly_copy_of_supplied_data():
    values = np.array([1.]); times = np.array([0.])
    data = response(values, times=times)
    values[0] = 2.; times[0] = 1.
    assert data.values[0] == 1. and data.coordinates["time_s"][0] == 0.
    with pytest.raises(ValueError):
        data.values[0] = 3.


def amplitude_result(coarse, fine, error_coarse, error_fine, **kwargs):
    return compare_amplitude_halving(response(coarse), response(fine),
        coarse_amplitude_V=kwargs.get("coarse_amplitude_V", .005),
        fine_amplitude_V=kwargs.get("fine_amplitude_V", .0025),
        coarse_current_error=response(error_coarse), fine_current_error=response(error_fine))


def test_section_six_amplitude_ladder_has_exactly_the_seven_declared_values():
    assert AMPLITUDES_V == SPEC_AMPLITUDES_V


@pytest.mark.parametrize("amplitude", SPEC_AMPLITUDES_V)
def test_every_declared_amplitude_is_usable_for_a_convergence_case(amplitude):
    case = R1ConvergenceCase(16, (1, 2, 4), .1, amplitude_V=amplitude)
    assert case.amplitude_V == amplitude


@pytest.mark.parametrize("amplitude", [
    .02, .000078125, 0., -.005,
    *(np.nextafter(level, direction) for level in SPEC_AMPLITUDES_V
      for direction in (-np.inf, np.inf)),
])
def test_no_extra_or_approximately_declared_amplitude_is_accepted(amplitude):
    with pytest.raises(ValueError, match="halving level"):
        R1ConvergenceCase(16, (1, 2, 4), .1, amplitude_V=amplitude)


@pytest.mark.parametrize("coarse,fine", list(zip(SPEC_AMPLITUDES_V[:-1], SPEC_AMPLITUDES_V[1:])))
def test_all_six_adjacent_amplitude_pairs_use_the_reported_one_percent_budget(coarse, fine):
    # A manufactured g=+/-1 S/m2 response fixes the scale. The measured current
    # errors separately contribute .002 and .003 S/m2, so the linearity limit
    # is .005 + .01 = .015 S/m2 at every declared adjacent amplitude pair.
    for multiplier, expected in ((1-1e-6, True), (1+1e-6, False)):
        for sign in (-1., 1.):
            result = amplitude_result(
                [sign*coarse], [sign*fine*(1-multiplier*.015)],
                [coarse*.002], [fine*.003],
                coarse_amplitude_V=coarse, fine_amplitude_V=fine)
            assert result["passed"] is expected
            assert result["status"] == ("within_linearity_budget" if expected else "response_not_linear")
            assert result["coarse_amplitude_V"] == coarse
            assert result["fine_amplitude_V"] == fine
            assert result["relative_tolerance"] == .01
            assert result["maximum_error_budget_signal_fraction"] == .01
            assert result["absolute_numerical_error_budget_S_m2"] == pytest.approx([.005], rel=1e-14)
            assert result["normalized_response_scale_S_m2"] == [1.]
            assert result["maximum_budget_ratio"] == pytest.approx(multiplier, rel=1e-12)
            if not expected:
                failure = result["failures"][0]
                assert failure["allowed_difference"] == pytest.approx(.015, rel=1e-14)
                assert failure["allowed_difference"] == (
                    result["absolute_numerical_error_budget_S_m2"][0]
                    + result["relative_tolerance"]*failure["response_scale"])


@pytest.mark.parametrize("coarse,fine", [
    (coarse, fine) for i, coarse in enumerate(SPEC_AMPLITUDES_V)
    for j, fine in enumerate(SPEC_AMPLITUDES_V) if j != i+1
])
def test_every_reverse_equal_or_nonadjacent_declared_amplitude_pair_is_rejected(coarse, fine):
    with pytest.raises(ValueError, match="adjacent declared positive halving levels"):
        amplitude_result([coarse], [fine], [0.], [0.],
                         coarse_amplitude_V=coarse, fine_amplitude_V=fine)


@pytest.mark.parametrize("coarse,fine", [(.02, .01), (.00015625, .000078125), (-.005, -.0025)])
def test_adjacent_but_out_of_ladder_amplitudes_are_rejected(coarse, fine):
    with pytest.raises(ValueError, match="adjacent declared positive halving levels"):
        amplitude_result([coarse], [fine], [0.], [0.],
                         coarse_amplitude_V=coarse, fine_amplitude_V=fine)


def test_amplitude_numerical_error_budget_is_sum_of_separately_normalized_errors():
    result = amplitude_result([.01], [.005], [5e-6], [2.5e-6])
    assert result["passed"] and result["status"] == "within_linearity_budget"
    assert result["units"] == "S/m2"
    assert result["absolute_numerical_error_budget_S_m2"] == pytest.approx([.002])
    assert result["normalized_response_scale_S_m2"] == [2.]


@pytest.mark.parametrize("side,passed", [(-1, True), (0, True), (1, False)])
def test_amplitude_one_percent_error_budget_limit_is_inclusive(side, passed):
    # g=1 S/m2; an A/m2 error of 5e-5 at 5 mV consumes exactly 1%.
    bound = 5e-5 if side == 0 else np.nextafter(5e-5, -np.inf if side < 0 else np.inf)
    result = amplitude_result([.005], [.0025], [bound], [0.])
    assert result["passed"] is passed
    assert result["status"] == ("within_linearity_budget" if passed else "linearity_undetermined")


def test_amplitude_comparison_rejects_resolved_non_linearity():
    result = amplitude_result([.005], [.00255], [0.], [0.])
    assert not result["passed"] and result["status"] == "response_not_linear"
    assert result["failures"][0]["reasons"] == ["response_difference_exceeds_budget"]


def test_large_error_budget_cannot_certify_a_weak_or_non_linear_signal():
    result = amplitude_result([.005], [.00275], [.01], [.01])
    assert not result["passed"] and result["status"] == "linearity_undetermined"
    assert result["unresolved_scalar_count"] == 1
    assert result["failures"][0]["reasons"] == ["linearity_signal_not_resolved"]


def test_zero_signal_does_not_establish_a_linear_range():
    result = amplitude_result([0.], [0.], [0.], [0.])
    assert not result["passed"] and result["status"] == "linearity_undetermined"
    assert result["failures"][0]["reasons"] == ["linearity_signal_not_resolved"]
    json.dumps(result, allow_nan=False)


def test_amplitude_comparison_does_not_inherit_the_current_absolute_tolerance():
    # A 5e-8 A/m2 difference would pass the current absolute floor, but the
    # voltage-normalized responses differ by far more than their 1% budget.
    result = amplitude_result([1e-7], [0.], [0.], [0.])
    assert result["status"] == "response_not_linear"
    assert result["failures"][0]["allowed_difference"] == pytest.approx(2e-7)


@pytest.mark.parametrize("coarse,fine", [(.01, .0025), (.0025, .005), (.02, .01), (.005, 0.)])
def test_amplitude_comparison_requires_declared_adjacent_halving_levels(coarse, fine):
    with pytest.raises(ValueError, match="halving levels"):
        amplitude_result([1.], [1.], [0.], [0.], coarse_amplitude_V=coarse, fine_amplitude_V=fine)


def test_amplitude_errors_are_finite_nonnegative_and_aligned():
    with pytest.raises(ValueError, match="nonnegative"):
        amplitude_result([1.], [1.], [-1.], [0.])
    invalid = amplitude_result([1.], [1.], [np.nan], [0.])
    assert not invalid["passed"] and invalid["status"] == "invalid_comparison"
    assert "nonfinite_numerical_error_bound" in invalid["failures"][0]["reasons"]
    json.dumps(invalid, allow_nan=False)
    with pytest.raises(ValueError, match="coordinate values differ"):
        compare_amplitude_halving(response([1.]), response([1.]), coarse_amplitude_V=.005,
            fine_amplitude_V=.0025, coarse_current_error=response([0.], times=[1.]),
            fine_current_error=response([0.]))
