"""Local arithmetic/derivative contracts; no device preparation or trajectory."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
import hashlib
import json

import numpy as np
import pytest

from perovskite_sim.physics.compensated import DD
from perovskite_sim.physics import two_sided_interface_compensated as local


def coefficients(*, geometry=None, physics=None, table=None):
    geometry = {"left_distance_m": 1.0, "right_distance_m": 1.0, **(geometry or {})}
    values = {"thermal_voltage_V": 1.0, "temperature_K": 1.0,
              "transmission": 1.0, "richardson_n_A_m2_K2": 1.0,
              "richardson_p_A_m2_K2": 1.0,
              "conduction_band_step_eV": 0.0, "hole_transport_step_eV": 0.0,
              "surface_recombination_velocity_n_m_s": 1.0,
              "surface_recombination_velocity_p_m_s": 1.0}
    for side in ("left", "right"):
        values.update({"D_n_" + side + "_m2_s": 1.0, "D_p_" + side + "_m2_s": 1.0,
                       "N_C_" + side + "_m3": 1024.0, "N_V_" + side + "_m3": 1024.0,
                       "n1_" + side + "_m3": 1.0, "p1_" + side + "_m3": 1.0})
    values.update(physics or {})
    table = table or {"eta": [-40.0, 20.0], "log_half": [-40.0, 20.0]}
    digest = hashlib.sha256(json.dumps(table, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return local.freeze_coefficients(geometry, values, fd_table=table,
                                    fd_table_sha256=digest, q_C=1.0)


def inputs(**changes):
    values = {"state_density": DD(np.ones(4)), "trace_potential": DD(np.zeros(2)),
              "bulk_density": DD(np.ones(4)), "bulk_potential": DD(np.zeros(2)),
              "occupancy": DD(0.5), "capture_multiplier": 1.0}
    values.update(changes)
    return values


def evaluate(c=None, **changes):
    return local.evaluate_local_carrier_pair(coefficients=c or coefficients(), **inputs(**changes))


def decimals(value):
    return [Decimal.from_float(float(hi)) + Decimal.from_float(float(lo))
            for hi, lo in zip(value.hi.flat, value.lo.flat, strict=True)]


def replace_component(value, index, new):
    high, low = value.hi.copy(), value.lo.copy()
    high[index], low[index] = new.hi, new.lo
    return DD(high, low)


def test_symmetric_state_has_exact_zero_and_low_only_perturbation_is_not_zero():
    c = coefficients()
    zero = evaluate(c)
    np.testing.assert_array_equal(zero["balance"]["residual_m2_s"].hi, np.zeros(4))
    np.testing.assert_array_equal(zero["balance"]["residual_m2_s"].lo, np.zeros(4))
    delta = np.ldexp(1.0, -70)
    perturbed = evaluate(c, state_density=DD(np.ones(4), [delta, 0, 0, 0]))
    residual = perturbed["balance"]["residual_m2_s"]
    np.testing.assert_array_equal(perturbed["balance"]["state_m3"].hi, np.ones(4))
    assert bool(residual[0] < 0) and bool(residual[2] > 0)
    assert bool(residual[1] == 0) and bool(residual[3] == 0)
    with localcontext() as ctx:
        ctx.prec = 90
        expected = -Decimal("1.5") * Decimal.from_float(delta)
        assert abs(sum(decimals(residual)) - expected) < Decimal("3e-31")


def test_bulk_signs_and_each_receiving_distance():
    c = coefficients(geometry={"left_distance_m": 2.0, "right_distance_m": 4.0},
                     physics={"transmission": 0.0,
                              "surface_recombination_velocity_n_m_s": 0.0,
                              "surface_recombination_velocity_p_m_s": 0.0})
    value = evaluate(c, state_density=DD([1, 2, 3, 4]), bulk_density=DD([2, 1, 5, 1]))
    np.testing.assert_array_equal(value["balance"]["bulk_flux_m2_s"].to_float(),
                                  [0.5, -0.5, 0.5, -0.75])
    np.testing.assert_array_equal(value["balance"]["residual_m2_s"].to_float(),
                                  [0.5, -0.5, 0.5, -0.75])


def test_capture_and_occupancy_partials_use_the_same_four_components():
    c = coefficients(physics={"transmission": 0.0,
        "D_n_left_m2_s": 0.0, "D_p_left_m2_s": 0.0,
        "D_n_right_m2_s": 0.0, "D_p_right_m2_s": 0.0,
        "surface_recombination_velocity_n_m_s": 2.0,
        "surface_recombination_velocity_p_m_s": 3.0,
        "n1_left_m3": 1.0, "p1_left_m3": 2.0,
        "n1_right_m3": 3.0, "p1_right_m3": 4.0})
    value = evaluate(c, state_density=DD([4, 6, 8, 10]), occupancy=DD(0.25))
    np.testing.assert_array_equal(value["balance"]["capture_flux_m2_s"].to_float(),
                                  [5.5, 0.0, 10.5, -1.5])
    np.testing.assert_array_equal(value["balance"]["residual_m2_s"].to_float(),
                                  [-5.5, 0.0, -10.5, 1.5])
    np.testing.assert_array_equal(value["tangent"]["capture_flux_occupancy_derivative_m2_s"].to_float(),
                                  [-10.0, 24.0, -22.0, 42.0])
    np.testing.assert_array_equal(np.diag(value["tangent"]["capture_flux_jacobian_log_state_m2_s"].to_float()),
                                  [6.0, 4.5, 12.0, 7.5])


def test_cross_rows_cancel_in_pairs_and_keep_particle_direction():
    c = coefficients(physics={"D_n_left_m2_s": 0.0, "D_p_left_m2_s": 0.0,
        "D_n_right_m2_s": 0.0, "D_p_right_m2_s": 0.0,
        "surface_recombination_velocity_n_m_s": 0.0,
        "surface_recombination_velocity_p_m_s": 0.0})
    value = evaluate(c, state_density=DD([2, 1, 1, 1]))
    flux = value["balance"]["cross_flux_m2_s"]
    assert bool(flux[0] < 0) and bool(flux[2] > 0)
    assert bool(flux[0] + flux[2] == 0) and bool(flux[1] + flux[3] == 0)
    assert value["metadata"]["supply_derivative_policy"] == "derivative_of_the_same_frozen_twelve_term_polynomial"
    assert np.all(value["metadata"]["analytic_old_F0_tail_bound"].hi > 0)


def test_full_words_select_the_correct_side_of_an_inverse_table_knot():
    c = coefficients(table={"eta": [-40.0, -10.0, 20.0],
                            "log_half": [-40.0, -20.0, 20.0]})
    tiny = np.ldexp(1.0, -70)
    log_ratio = DD([-20.0, -20.0, -20.0, -41.0], [-tiny, 0.0, tiny, 0.0])
    eta, slope, intervals = local._inverse_half(log_ratio, c)
    np.testing.assert_array_equal(intervals, [0, 1, 1, -1])
    np.testing.assert_array_equal(slope.to_float(), [1.5, 0.75, 0.75, 1.0])
    assert bool(eta[0] < -10) and bool(eta[1] == -10) and bool(eta[2] > -10)
    assert bool(eta[3] == -41)


@pytest.mark.parametrize("kind", ["fermi_one", "inverse_half", "bernoulli"])
def test_unimplemented_domain_is_not_a_silently_backtracked_value_error(kind):
    with pytest.raises(local.UnsupportedCompensatedInterfaceDomain) as error:
        if kind == "fermi_one":
            evaluate(state_density=DD(np.full(4, 1024.0)))
        elif kind == "inverse_half":
            local._inverse_half(DD([21.0]), coefficients())
        else:
            local._bernoulli_values_and_derivatives(DD([673.0]))
    assert isinstance(error.value, ArithmeticError)
    assert not isinstance(error.value, (ValueError, RuntimeError))


def test_immutable_evaluation_coefficients_and_explicit_float_adapter():
    c = coefficients()
    result = evaluate(c)
    with pytest.raises(FrozenInstanceError):
        result.balance = {}
    with pytest.raises(TypeError):
        result["balance"]["residual_m2_s"] = DD(np.ones(4))
    with pytest.raises(TypeError):
        result["metadata"]["fallback_used"] = True
    with pytest.raises(ValueError):
        result["balance"]["residual_m2_s"].hi[0] = 1.0
    with pytest.raises(TypeError):
        c.physics["thermal_voltage_V"] = DD(2)
    with pytest.raises(ValueError):
        c.log_half_search.setflags(write=True)
    temporary_search = c.log_half_search
    temporary_search.shape = (1, 2)
    assert c.log_half_search.shape == (2,)
    assert deepcopy(result) is result and deepcopy(c) is c
    adapter = local.to_production_tangent(result)
    np.testing.assert_array_equal(adapter.balance.residual_m2_s, np.zeros(4))
    assert adapter.balance.jacobian_log_state_m2_s.shape == (4, 4)
    assert adapter.balance.jacobian_bulk_coordinates.shape == (4, 6)
    assert result["metadata"]["inverse_intervals"] == (0, 0, 0, 0)


def test_default_table_is_original_cached_and_immutable(monkeypatch):
    from perovskite_sim.physics import fermi_dirac
    calls = []
    def table():
        calls.append(True)
        return np.array([-40., 20.]), np.array([1., 2.]), np.array([-40., 20.])
    local.default_fd_table.cache_clear()
    monkeypatch.setattr(fermi_dirac, "_half_table", table)
    try:
        frozen, digest = local.default_fd_table()
        assert local.default_fd_table()[0] is frozen and len(calls) == 1
        with pytest.raises(TypeError):
            frozen["eta"] = (1.0, 2.0)
        assert isinstance(frozen["eta"], tuple)
        expected = hashlib.sha256(json.dumps(dict(frozen), sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        assert digest == expected
    finally:
        local.default_fd_table.cache_clear()


def test_table_content_must_match_explicit_identity():
    c = coefficients()
    with pytest.raises(ValueError, match="identity"):
        local.freeze_coefficients({"left_distance_m": 1., "right_distance_m": 1.},
            {name: float(value.to_float()) for name, value in c.physics.items()},
            fd_table={"eta": [-40., 20.], "log_half": [-40., 20.]},
            fd_table_sha256="0" * 64, q_C=1.)


@pytest.mark.parametrize("coordinate,index", [
    *[("trace_log", i) for i in range(4)], *[("trace_phi", i) for i in range(2)],
    *[("bulk_log", i) for i in range(4)], *[("bulk_phi", i) for i in range(2)],
    ("occupancy", 0),
])
def test_all_thirteen_direct_partials_match_centered_same_law_difference(coordinate, index):
    c = coefficients(physics={"conduction_band_step_eV": 0.3,
                              "hole_transport_step_eV": -0.4})
    base = inputs(state_density=DD([1.1, 1.7, 2.2, 2.6]), bulk_density=DD([1.2, 1.9, 2.0, 2.8]),
                  trace_potential=DD([0.02, 0.06]), bulk_potential=DD([-0.01, 0.08]),
                  occupancy=DD(0.37))
    result = local.evaluate_local_carrier_pair(coefficients=c, **base)
    if coordinate == "trace_log":
        key, logarithmic = "state_density", True
        expected = result["balance"]["jacobian_log_state_m2_s"][:, index]
    elif coordinate == "trace_phi":
        key, logarithmic = "trace_potential", False
        expected = result["balance"]["jacobian_trace_potential_m2_s_V"][:, index]
    elif coordinate == "bulk_log":
        key, logarithmic = "bulk_density", True
        expected = result["balance"]["jacobian_bulk_coordinates"][:, index + 2]
    elif coordinate == "bulk_phi":
        key, logarithmic = "bulk_potential", False
        expected = result["balance"]["jacobian_bulk_coordinates"][:, index]
    else:
        key, logarithmic = "occupancy", False
        expected = result["tangent"]["residual_occupancy_derivative_m2_s"]
    h = DD(np.ldexp(1.0, -24))
    observations = []
    for sign in (-1, 1):
        trial = dict(base)
        if key == "occupancy":
            trial[key] = base[key] + sign * h
        else:
            changed = base[key][index] * (sign * h).exp() if logarithmic else base[key][index] + sign * h
            trial[key] = replace_component(base[key], index, changed)
        observations.append(local.evaluate_local_carrier_pair(coefficients=c, **trial)["balance"]["residual_m2_s"])
    observed = (observations[1] - observations[0]) / (2 * h)
    np.testing.assert_allclose(observed.to_float(), expected.to_float(), rtol=5e-9, atol=3e-12)


def test_nonzero_full_pair_fluxes_against_independent_decimal_formulas():
    c = coefficients()
    base = inputs(state_density=DD([1.1, 1.4, 1.8, 2.1], [1e-18, -2e-18, 3e-18, -4e-18]),
                  bulk_density=DD([1.2, 1.3, 1.9, 2.0], [-1e-18, 2e-18, -3e-18, 4e-18]),
                  trace_potential=DD([0.02, 0.06], [1e-20, -2e-20]),
                  bulk_potential=DD([-0.01, 0.08], [-1e-20, 2e-20]),
                  occupancy=DD(0.37, 1e-18))
    result = local.evaluate_local_carrier_pair(coefficients=c, **base)
    with localcontext() as ctx:
        ctx.prec = 90
        n, bulk = decimals(base["state_density"]), decimals(base["bulk_density"])
        trace, phi = decimals(base["trace_potential"]), decimals(base["bulk_potential"])
        f = decimals(base["occupancy"])[0]
        xl, xr = trace[0] - phi[0], phi[1] - trace[1]
        def bernoulli(x):
            return x / (x.exp() - 1)
        b, bm, r, rm = [bernoulli(x) for x in (xl, -xl, xr, -xr)]
        flux = [bm * bulk[0] - b * n[0], b * bulk[1] - bm * n[1],
                r * bulk[2] - rm * n[2], rm * bulk[3] - r * n[3]]
        jump = trace[1] - trace[0]
        en, ep = -jump, jump
        barriers = [max(en, 0), max(ep, 0), max(-en, 0), max(-ep, 0)]
        def supply(density, barrier):
            activity = ((density / 1024).ln() - barrier).exp()
            return sum(Decimal((-1) ** (k + 1)) * activity ** k / Decimal(k * k)
                       for k in range(1, 13))
        supplies = [supply(a, b) for a, b in zip(n, barriers, strict=True)]
        xn, xp = supplies[0] - supplies[2], supplies[1] - supplies[3]
        cross = [-xn, -xp, xn, xp]
        capture = [n[0] * (1-f) - f, n[1] * f - (1-f),
                   n[2] * (1-f) - f, n[3] * f - (1-f)]
        residual = [a + b - d for a, b, d in zip(flux, cross, capture, strict=True)]
        for name, expected in (("bulk_flux_m2_s", flux), ("cross_flux_m2_s", cross),
                               ("capture_flux_m2_s", capture), ("residual_m2_s", residual)):
            for actual, want in zip(decimals(result["balance"][name]), expected, strict=True):
                assert abs(actual - want) <= Decimal("2e-27"), (name, actual, want)
