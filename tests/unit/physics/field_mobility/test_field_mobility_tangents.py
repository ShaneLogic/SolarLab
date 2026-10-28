"""N class: one-sided directional derivatives and field-measure capabilities.

Oracle: one-sided difference quotients ``[mu(E + h d) - mu(E)] / h`` of the
M-class composition for shrinking ``h``; infinite slopes are identified by
their growth rate, never by trusting the analytic sign alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from solarlab.physics.transport import field_mobility as fm
from solarlab.physics.transport import field_mobility_tangents as tangents


def _quotient(mu0, E, d, v_sat, beta, gamma, h, width=0.0):
    mu0, E, d, v_sat, beta, gamma = (np.atleast_1d(np.asarray(value, dtype=float)) for value in (mu0, E, d, v_sat, beta, gamma))
    base = fm.apply_field_mobility(mu0, E, v_sat, beta, gamma, pf_field_regularization_width_V_m=width)
    step = fm.apply_field_mobility(mu0, E + h * d, v_sat, beta, gamma, pf_field_regularization_width_V_m=width)
    return (step - base) / h


def _derivative(mu0, E, d, v_sat, beta, gamma, width=0.0):
    return tangents.field_mobility_directional_derivative(
        np.array([mu0]), np.array([E]), np.array([d]), np.array([v_sat]), np.array([beta]), np.array([gamma]),
        pf_field_regularization_width_V_m=width,
    )


@pytest.mark.parametrize("E", [-3e7, -2e5, 4e3, 6e6])
@pytest.mark.parametrize("d", [1.0, -2.5])
@pytest.mark.parametrize("v_sat, beta, gamma", [(1e4, 2.0, 3e-4), (0.0, 2.0, -2e-4), (3e3, 1.0, 0.0), (1e4, 0.7, 1e-4)])
def test_smooth_faces_give_the_linear_directional_derivative(E, d, v_sat, beta, gamma):
    result = _derivative(2e-4, E, d, v_sat, beta, gamma)
    assert result.smooth[0] and result.finite[0]
    h = 1e-7 * max(abs(E), 1.0)
    fd = _quotient(2e-4, E, d, v_sat, beta, gamma, h)[0]
    assert result.derivative_m3_V2_s[0] == pytest.approx(fd, rel=1e-5, abs=1e-30)
    linear = fm.linearize_field_mobility(2e-4, E, v_sat, beta, gamma).field_derivative_m3_V2_s
    assert result.derivative_m3_V2_s[0] == pytest.approx(float(linear) * d, rel=1e-15)


@pytest.mark.parametrize("gamma", [3e-4, -3e-4])
@pytest.mark.parametrize("d", [1.0, -1.0])
def test_zero_field_poole_frenkel_cusp_has_a_signed_infinite_slope(gamma, d):
    result = _derivative(2e-4, 0.0, d, 1e4, 2.0, gamma)
    assert not result.smooth[0] and result.derivative_m3_V2_s[0] == np.sign(gamma) * np.inf
    quotients = [_quotient(2e-4, 0.0, d, 1e4, 2.0, gamma, h)[0] for h in (1e-2, 1e-4, 1e-6)]
    assert all(np.sign(q) == np.sign(gamma) for q in quotients)
    assert abs(quotients[2]) / abs(quotients[0]) == pytest.approx(100.0, rel=0.05)  # ~ h^(-1/2)
    with pytest.raises(tangents.FieldMobilityCapabilityError, match="no finite one-sided derivative"):
        tangents.require_finite_directional_derivative(result)


def test_regularized_poole_frenkel_is_smooth_at_zero_field():
    result = _derivative(2e-4, 0.0, 1.0, 0.0, 2.0, 3e-4, width=1e4)
    assert result.smooth[0] and result.derivative_m3_V2_s[0] == 0.0


@pytest.mark.parametrize("d", [1.0, -3.0])
def test_zero_field_caughey_thomas_beta_one_has_a_finite_kink(d):
    mu0, v_sat = 2e-4, 3e3
    result = _derivative(mu0, 0.0, d, v_sat, 1.0, 0.0)
    expected = -mu0 * mu0 * abs(d) / v_sat
    assert not result.smooth[0] and result.derivative_m3_V2_s[0] == pytest.approx(expected, rel=1e-15)
    for h in (1e-3, 1e-6):
        assert _quotient(mu0, 0.0, d, v_sat, 1.0, 0.0, h)[0] == pytest.approx(expected, rel=10 * h * mu0 / v_sat + 1e-9)
    assert tangents.require_finite_directional_derivative(result)[0] == pytest.approx(expected, rel=1e-15)


def test_zero_field_caughey_thomas_beta_below_one_is_infinite_and_above_one_is_flat():
    below = _derivative(2e-4, 0.0, 1.0, 3e3, 0.6, 0.0)
    assert below.derivative_m3_V2_s[0] == -np.inf
    quotients = [abs(_quotient(2e-4, 0.0, 1.0, 3e3, 0.6, 0.0, h)[0]) for h in (1e-2, 1e-6)]
    assert quotients[1] > 10.0 * quotients[0]
    above = _derivative(2e-4, 0.0, 1.0, 3e3, 2.0, 0.0)
    assert above.smooth[0] and above.derivative_m3_V2_s[0] == 0.0


@pytest.mark.parametrize("d", [0.0, 1.0, -3.0])
@pytest.mark.parametrize("mu0, v_sat, gamma", [(1.0, 1.0, 2.0), (4.0, 1.0, 4.0)])
def test_cancelling_pf_ct_cusps_have_finite_directional_derivative(d, mu0, v_sat, gamma):
    from decimal import Decimal, localcontext

    result = _derivative(mu0, 0.0, d, v_sat, 0.5, gamma)
    expected = -mu0 * mu0 * abs(d) / v_sat
    assert tangents.require_finite_directional_derivative(result)[0] == expected
    # Independent high-precision quotient of the value law, avoiding binary64 cancellation.
    with localcontext() as context:
        context.prec = 70
        h = Decimal("1e-20")
        field = h * abs(Decimal(d))
        pf = Decimal(mu0) * (Decimal(gamma) * field.sqrt()).exp()
        value = pf / (1 + (pf * field / Decimal(v_sat)).sqrt()) ** 2
        quotient = float((value - Decimal(mu0)) / h)
    assert quotient == pytest.approx(expected, rel=2e-9, abs=0.0)


def test_rounded_square_root_is_not_an_exact_cusp_cancellation():
    gamma = float(2.0 * np.sqrt(2.0))
    above = _derivative(2.0, 0.0, 1.0, 1.0, 0.5, gamma)
    below = _derivative(2.0, 0.0, 1.0, 1.0, 0.5, np.nextafter(gamma, 0.0))
    assert above.derivative_m3_V2_s[0] == np.inf
    assert below.derivative_m3_V2_s[0] == -np.inf
    with pytest.raises(tangents.FieldMobilityCapabilityError):
        tangents.require_finite_directional_derivative(above)


@pytest.mark.parametrize("beta", [1.0, 2.0])
def test_exp_clip_surface_takes_the_side_of_the_direction(beta):
    gamma = 1e-2
    E = (80.0 / gamma) ** 2  # gamma * sqrt|E| == 80 exactly
    assert gamma * np.sqrt(E) == 80.0
    mu0, v_sat = 1e-40, 1e4  # keeps mu finite and CT barely active
    into = _derivative(mu0, E, 1.0, v_sat, beta, gamma)  # |E| grows: enters the clipped region
    out = _derivative(mu0, E, -1.0, v_sat, beta, gamma)
    assert not into.smooth[0] and not out.smooth[0]
    h = 1e-6 * E
    assert into.derivative_m3_V2_s[0] == pytest.approx(_quotient(mu0, E, 1.0, v_sat, beta, gamma, h)[0], rel=1e-4)
    assert out.derivative_m3_V2_s[0] == pytest.approx(_quotient(mu0, E, -1.0, v_sat, beta, gamma, h)[0], rel=1e-4)
    assert abs(out.derivative_m3_V2_s[0]) > abs(into.derivative_m3_V2_s[0])


def test_directional_derivative_is_positively_homogeneous_and_zero_along_zero():
    result = tangents.field_mobility_directional_derivative(
        np.full(3, 2e-4), np.array([0.0, 0.0, 5e5]), np.array([0.0, 0.0, 0.0]), 3e3, np.array([1.0, 0.5, 2.0]), 3e-4
    )
    assert np.array_equal(result.derivative_m3_V2_s, np.zeros(3))
    single = _derivative(2e-4, 0.0, 1.0, 3e3, 1.0, 0.0).derivative_m3_V2_s[0]
    double = _derivative(2e-4, 0.0, 2.0, 3e3, 1.0, 0.0).derivative_m3_V2_s[0]
    assert double == pytest.approx(2.0 * single, rel=1e-15)


def test_field_measure_capabilities():
    one_d = tangents.require_field_measure(tangents.FACE_FIELD_1D)
    two_d = tangents.require_field_measure(tangents.FACE_NORMAL_COMPONENT_2D)
    assert one_d.exact and one_d.dimension == 1
    assert not two_d.exact and two_d.dimension == 2 and "|E_x|" in two_d.description
    for label in ("total_field_magnitude", "face_normal_component_2d_v2", ""):
        with pytest.raises(tangents.FieldMobilityCapabilityError, match="model change"):
            tangents.require_field_measure(label)
