"""Bounded selective AC, sparse anchor derivatives and closed-carrier scope."""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.steady import load_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.experiments.impedance import (
    _reference_anchor_system, _uniform_carrier_partial, solve_impedance,
)
from solarlab.physics.contacts.boundary_flux import boundary_particle_flux
from solarlab.solvers.nonlinear.steady_1d import solve_equilibrium, solve_steady_state

Q = 1.602176634e-19


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


def device(kind="selective", reverse=False, closed=None):
    prepared = load_device()
    data = prepared.to_input().editing_data()
    if reverse:
        data["layers"] = list(reversed(data["layers"]))
    if kind == "mixed":
        velocities = ((None, 0.0), (None, 200.0)) if reverse else ((0.0, None), (200.0, None))
    else:
        velocities = ((200.0, 0.0), (0.0, 200.0)) if reverse else ((0.0, 200.0), (200.0, 0.0))
    if closed is not None:
        velocities = ((0.0, 200.0), (0.0, 200.0)) if closed == "n" else ((200.0, 0.0), (200.0, 0.0))
    data["contacts"] = [
        dict(id=side, side=side, layer=layer["id"], S_n=values[0], S_p=values[1])
        for side, layer, values in zip(("left", "right"), (data["layers"][0], data["layers"][-1]), velocities)
    ]
    prepared = resolve_device(DeviceInput.model_validate(data), prepared.defaults, prepared.resources, sources=prepared.sources)
    return compile_device_1d(prepared, intervals_per_layer=12)


@pytest.fixture(scope="module")
def selective():
    compiled = device()
    point = solve_steady_state(compiled, voltage_V=1.0)
    response = solve_impedance(compiled, point, [0.0, 1.0, 1000.0, 1e6])
    return compiled, point, response


def _d(value):
    return Decimal.from_float(float(value))


def _bernoulli(value):
    return Decimal(1) if value == 0 else value / (value.exp() - 1)


def _scalar_rate(n, p, coefficients, node):
    net = n * p - _d(coefficients.ni_sq_m6[node])
    tn, tp = _d(coefficients.tau_n_s[node]), _d(coefficients.tau_p_s[node])
    srh = Decimal(0)
    if tn.is_finite() and tp.is_finite():
        denominator = tp * (n + _d(coefficients.n1_m3[node])) + tn * (p + _d(coefficients.p1_m3[node]))
        srh = net / denominator
    radiative = _d(coefficients.radiative_m3_s[node]) * net
    auger = (_d(coefficients.auger_n_m6_s[node]) * n + _d(coefficients.auger_p_m6_s[node]) * p) * net
    return srh + radiative + auger


def _rate_condition(n, p, c, node, carrier):
    product = n * p
    net = product - _d(c.ni_sq_m6[node])
    tn, tp = _d(c.tau_n_s[node]), _d(c.tau_p_s[node])
    bound = _d(c.radiative_m3_s[node]) * product
    if tn.is_finite() and tp.is_finite():
        denominator = tp * (n + _d(c.n1_m3[node])) + tn * (p + _d(c.p1_m3[node]))
        capture = tp * n if carrier == "n" else tn * p
        bound += product / denominator + abs(net) * capture / denominator**2
    cn, cp = _d(c.auger_n_m6_s[node]), _d(c.auger_p_m6_s[node])
    bound += (cn * n if carrier == "n" else cp * p) * abs(net) + (cn * n + cp * p) * product
    return bound


def _decimal_directional_column(problem, u, carrier):
    """90-digit scalar differentiation of the declared production rate/SG laws.

    Captured binary64 densities, coefficients and drives are embedded exactly;
    the uniform shift is differentiated in Decimal, not through real wrappers.
    Conditioning includes the SRH quotient's cancelling numerator terms.
    """
    compiled = problem.device
    count = compiled.node_count
    assert compiled.additional_recombination is None
    _, n_float, p_float = compiled.fields(u, voltage_V=problem.voltage_V)
    fluxes = problem.fluxes(u)
    coefficients = compiled.recombination.coefficients
    with localcontext() as context:
        context.prec = 90
        step = Decimal("1e-25")
        n = [_d(value) for value in n_float]
        p = [_d(value) for value in p_float]
        rates = []
        densities = []
        for shift in (step, -step):
            nt = [value * shift.exp() for value in n] if carrier == "n" else n
            pt = [value * (-shift).exp() for value in p] if carrier == "p" else p
            densities.append((nt, pt))
            rates.append([_scalar_rate(a, b, coefficients, node) for node, (a, b) in enumerate(zip(nt, pt))])
        rate_derivative = [(a - b) / (2 * step) for a, b in zip(*rates)]
        dn = [(a - b) / (2 * step) for a, b in zip(densities[0][0], densities[1][0])]
        dp = [(a - b) / (2 * step) for a, b in zip(densities[0][1], densities[1][1])]
        result = [Decimal(0)] * (3 * count)
        scale = [Decimal(0)] * (3 * count)
        charge_widths = compiled.poisson.charge_jacobian()
        for node in range(count):
            result[node] = _d(Q) * _d(charge_widths[node]) * (dp[node] - dn[node])
            scale[node] = abs(result[node])
        for species, offset, flux, density, density_derivative, sign in (
            ("n", count, fluxes.electron, n, dn, 1), ("p", 2 * count, fluxes.hole, p, dp, -1),
        ):
            faces = [Decimal(0)] * (count - 1)
            if species == carrier:
                for face in range(count - 1):
                    drive = _d(sign * (u[offset + face + 1] - u[offset + face]))
                    energy = _d(flux.energy_step[face])
                    upwind = _bernoulli(-energy) * density[face + 1] if drive >= 0 else -_bernoulli(energy) * density[face]
                    base = _d(flux.coefficient[face]) * ((-abs(drive)).exp() - 1) * upwind
                    faces[face] = (base * (sign * step).exp() - base * (-sign * step).exp()) / (2 * step)
            padded = [Decimal(0), *faces, Decimal(0)]
            for node in range(count):
                width = _d(compiled.geometry.cell_measures_m[node])
                result[offset + node] = padded[node] - padded[node + 1] - width * rate_derivative[node]
                scale[offset + node] = abs(padded[node]) + abs(padded[node + 1]) + width * _rate_condition(n[node], p[node], coefficients, node, carrier)
            for side, node in (("left", 0), ("right", count - 1)):
                channel = compiled.channel(species, side)
                if channel.kind == "fixed_reservoir":
                    result[offset + node] = Decimal(sign if species == carrier else 0)
                    scale[offset + node] = abs(result[offset + node])
                else:
                    slope = _d(boundary_particle_flux(channel, channel.reservoir_density_m3).density_derivative)
                    value = (1 if side == "left" else -1) * slope * density_derivative[node]
                    result[offset + node] += value
                    scale[offset + node] += abs(value)
        return np.asarray([float(value) for value in result]), np.asarray([float(value) for value in scale])


@pytest.mark.parametrize("carrier", ["n", "p"])
def test_uniform_anchor_columns_have_independent_90_digit_directional_check(selective, carrier):
    compiled, point, _ = selective
    problem = DC1DProblem(compiled, point.voltage_V, point.generation_scale)
    actual = _uniform_carrier_partial(problem, point.state.u, carrier)
    expected, conditioning = _decimal_directional_column(problem, point.state.u, carrier)
    error = float(np.max(abs(actual - expected) / np.maximum(conditioning, np.finfo(float).tiny)))
    assert error < 1e-12


def test_anchor_basis_is_invertible_and_records_rounding_difference(selective):
    compiled, point, _ = selective
    problem = DC1DProblem(compiled, point.voltage_V, point.generation_scale)
    J, K = problem.jacobian(point.state.u), problem.storage_jacobian(point.state.u)
    shifted, mass, P, anchors = _reference_anchor_system(problem, point.state.u, J, K)
    vector = np.linspace(-1.0, 1.0, 3 * compiled.node_count)
    transformed = P @ vector
    inverse = transformed.copy()
    for offset, anchor in zip((compiled.node_count, 2 * compiled.node_count), anchors):
        rows = np.arange(offset, offset + compiled.node_count)
        inverse[rows] -= transformed[anchor]
        inverse[anchor] = transformed[anchor]
    np.testing.assert_allclose(inverse, vector, rtol=0.0, atol=3e-16)
    np.testing.assert_array_equal(mass.toarray(), (K @ P).toarray())
    assert np.max(abs((shifted - J @ P).data)) > 0.0


def _check_response(result):
    faces = np.c_[result.terminal_current_density_A_m2[:, :1], result.total_current_density_A_m2,
                  result.terminal_current_density_A_m2[:, 1:]]
    for index, solve in enumerate(result.linear_solutions):
        assert solve.componentwise_backward_error <= 1e-11
        assert result.maxwell_spread_A_m2_V[index] <= 1e-8 + 1e-10 * float(np.max(abs(faces[index])))


def test_selective_actual_frequencies_and_both_terminal_dc_difference(selective):
    compiled, point, result = selective
    _check_response(result)
    step = 5e-5
    plus = solve_steady_state(compiled, initial_state=point.state, voltage_V=point.voltage_V + step)
    minus = solve_steady_state(compiled, initial_state=point.state, voltage_V=point.voltage_V - step)
    expected = compiled.voltage_polarity * (plus.contact_current_density_A_m2 - minus.contact_current_density_A_m2) / (2 * step)
    actual = compiled.voltage_polarity * result.terminal_current_density_A_m2[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-5)


@pytest.mark.parametrize("kind, reverse", [("mixed", True)])
def test_mixed_channels_and_supported_opposite_polarity(kind, reverse):
    compiled = device(kind, reverse)
    point = solve_steady_state(compiled, voltage_V=1.0)
    result = solve_impedance(compiled, point, [0.0, 1000.0])
    _check_response(result)
    assert result.voltage_polarity == (-1.0 if reverse else 1.0)
    if kind == "mixed":
        assert {channel.kind for channel in compiled.channels} == {"fixed_reservoir", "robin", "blocking"}
    step = 5e-5
    plus = solve_steady_state(compiled, initial_state=point.state, voltage_V=1.0 + step)
    minus = solve_steady_state(compiled, initial_state=point.state, voltage_V=1.0 - step)
    expected = compiled.voltage_polarity * (plus.contact_current_density_A_m2 - minus.contact_current_density_A_m2) / (2 * step)
    np.testing.assert_allclose(compiled.voltage_polarity * result.terminal_current_density_A_m2[0], expected, rtol=2e-6, atol=1e-5)


@pytest.mark.parametrize("kind, reverse", [("selective", True), ("mixed", False)])
def test_remaining_mixed_polarity_counterexamples_refuse_unchanged_maxwell_limits(kind, reverse):
    compiled = device(kind, reverse)
    point = solve_steady_state(compiled, voltage_V=1.0)
    with pytest.raises(ArithmeticError, match="Maxwell-current closure failed"):
        solve_impedance(compiled, point, [0.0, 1000.0])
    assert point.converged and all(gate.passed for gate in point.physical_gates)


@pytest.mark.parametrize("carrier", ["n", "p"])
def test_both_ends_blocked_zero_frequency_remains_unvalidated(carrier):
    compiled = device(closed=carrier)
    point = solve_equilibrium(compiled)
    with pytest.raises(ValueError, match=f"zero-frequency {carrier} blocked at both ends is not validated"):
        solve_impedance(compiled, point, [0.0, 1000.0])
