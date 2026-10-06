"""Nonnative observation checks; no toy path certifies a device trajectory."""

from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import json
import os

import numpy as np
import pytest

from scripts.benchmarks.contract_prototype import ContractError
from scripts.benchmarks.coupled_device_prototype import AffineCoupledSlab, ProtocolSegment, SlabDefinition
from scripts.benchmarks.interval_observation import (
    AbsoluteIntegralBound, AcceptedClock, BallIntegrator, ChargePrefix, Enclosure, Polynomial, PolynomialPath,
    SlabPathObserver, affine_interval_actions, endpoint_charge_mismatch, identity,
    native_observation_readiness, newton_polynomials, public_action_enclosures,
    rational, segment_input_roundoff,
)
from perovskite_sim.constants import Q


REPO = Path(__file__).resolve().parents[2]
PLAN = Path(os.environ["REAL_DEVICE_PLAN"])
CASES = ("S0NeutralPublicDeviceV1", "DynamicAcceptorIonPublicDeviceV1")
# These are numerical oracle work goals, not replacements for a physical gate.
ORACLE_GOAL = Fraction(1, 2**160)


def report(request, **values):
    def encode(value):
        if isinstance(value, Fraction):
            return [value.numerator, value.denominator]
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, (Enclosure, AbsoluteIntegralBound)):
            return value.payload()
        raise TypeError(type(value).__name__)
    if path := os.environ.get("INTERVAL_CASE_LOG"):
        with Path(path).open("a") as output:
            output.write(json.dumps({"test": request.node.nodeid, **values}, default=encode, allow_nan=False)+"\n")


def clock(**changes):
    result = dict(predecessor=0, tn=1, hused=1, generation=1, nsteps=1, kused=2,
                  segment_start=0, segment_end=1, segment_id="synthetic")
    return AcceptedClock(**dict(result, **changes))


@lru_cache(maxsize=4)
def model(case_id, area):
    return AffineCoupledSlab(SlabDefinition.from_plan(PLAN, REPO, case_id, area=area), 8)


def path_for(device, *, moving=True, weak=False, sign=1, selected_clock=None):
    fields = {}
    for variable in device.layout.variables:
        value = device.field(device.reference, variable.id)
        fields[variable.id] = [Polynomial((rational(a)+rational(b),))
                               for a, b in zip(value.high, value.low, strict=True)]
    N = device.count
    for i in range(N):
        x = Fraction(i, N-1)
        # A constant spatial field makes SG a polynomial in the varying
        # densities. Independent Simpson integration is then exact, while
        # still exercising nonlinear Bernoulli and vacancy logarithms.
        fields["phi_V"][i] = Polynomial((sign*x/Fraction(1000),))
        for name, direction in (("n_m3", 1), ("p_m3", -1)):
            initial = fields[name][i].coefficients[0]
            amplitude = Fraction(1, 2**120) if weak else initial*x/Fraction(1000)
            fields[name][i] = Polynomial((initial+amplitude*direction,
                                          amplitude*direction if moving else 0))
        if device.definition.dynamic:
            fields["c_m3"][i] = fields["c_m3"][i]*(1+x/Fraction(1000))
            f = fields["f"][i].coefficients[0]
            fields["f"][i] = Polynomial((f+x/Fraction(100000), x/Fraction(100000) if moving else 0))
    return PolynomialPath(selected_clock or clock(), fields, (Polynomial(), Polynomial()),
                          device.source_identity, device.layout.identity,
                          identity({"test": "polynomial_input", "case": device.definition.id,
                                    "area": device.definition.area, "moving": moving, "weak": weak, "sign": sign}),
                          "synthetic_component")


def decimal(value):
    value = rational(value)
    return Decimal(value.numerator)/Decimal(value.denominator)


def decimal_currents(device, path, u):
    """Independent full-SG FV reference and direct tridiagonal tangent solve."""
    with localcontext() as context:
        context.prec = 120
        m, N = device.definition, device.count
        vals = {k: [decimal(p.at(u)) for p in row] for k, row in path.fields.items()}
        rates = {k: [decimal(p.derivative().at(u)/path.clock.hused) for p in row] for k, row in path.fields.items()}
        q, area, eps, vt = map(decimal, (Q, m.area, m.epsilon, m.vt))
        dx, volume = list(map(decimal, device.dx)), list(map(decimal, device.geometry.volumes))
        n, p, phi = vals["n_m3"], vals["p_m3"], vals["phi_V"]
        cn, cp, nt, n1, p1 = map(decimal, (m.capture_n, m.capture_p, m.trap_density, m.n1, m.p1))
        B = lambda z: Decimal(1) if z == 0 else z/(z.exp()-1)
        jn, jp, fi = [], [], []
        for i in range(N-1):
            xi = (phi[i+1]-phi[i])/vt
            jn.append(q*decimal(m.mu_n)*vt/dx[i]*(B(xi)*n[i+1]-B(-xi)*n[i]))
            jp.append(q*decimal(m.mu_p)*vt/dx[i]*(B(xi)*p[i]-B(-xi)*p[i+1]))
            if m.dynamic:
                c, capacity = vals["c_m3"], decimal(m.ion_capacity)
                eta = xi+(capacity-c[i]).ln()-(capacity-c[i+1]).ln()
                fi.append(decimal(m.diffusion_ion)/dx[i]*(B(eta)*c[i]-B(-eta)*c[i+1]))
            else:
                fi.append(Decimal(0))
        rn, rp = [], []
        for i in range(N):
            if m.dynamic:
                f = vals["f"][i]
                rn.append(cn*(n[i]*(1-f)-n1*f)); rp.append(cp*(p[i]*f-p1*(1-f)))
            else:
                shared = cn*cp*(n[i]*p[i]-n1*p1)/(cn*(n[i]+n1)+cp*(p[i]+p1))
                rn.append(shared); rp.append(shared)
        dot = {name: [Decimal(0)]*N for name in vals}
        for i in range(1, N-1):
            dot["n_m3"][i] = area*(-jn[i-1]+jn[i])/q/volume[i]-nt*rn[i]
            dot["p_m3"][i] = area*(jp[i-1]-jp[i])/q/volume[i]-nt*rp[i]
        if m.dynamic:
            flux = [Decimal(0), *fi, Decimal(0)]
            dot["c_m3"] = [area*(flux[i]-flux[i+1])/volume[i] for i in range(N)]
            dot["f"] = [a-b for a, b in zip(rn, rp, strict=True)]

        def rho_rate(rate):
            return [q*(rate["p_m3"][i]-rate["n_m3"][i]
                       +(rate["c_m3"][i]-nt*rate["f"][i] if m.dynamic else 0)) for i in range(N)]

        # Independent Thomas solve with the actual stored input slope,
        # rather than the implementation's closed-form Poisson weights.
        dot["phi_V"][-1] = -decimal(path.inputs[0].derivative().at(u)/path.clock.hused)
        rho = rho_rate(dot)
        lower = [Decimal(0)]+[-area*eps/dx[i-1] for i in range(2, N-1)]
        diagonal = [area*eps*(1/dx[i-1]+1/dx[i]) for i in range(1, N-1)]
        upper = [-area*eps/dx[i] for i in range(1, N-2)]+[Decimal(0)]
        rhs = [volume[i]*rho[i] for i in range(1, N-1)]
        rhs[-1] += area*eps/dx[-1]*dot["phi_V"][-1]
        for i in range(1, len(rhs)):
            multiple = lower[i]/diagonal[i-1]
            diagonal[i] -= multiple*upper[i-1]; rhs[i] -= multiple*rhs[i-1]
        solution = [Decimal(0)]*len(rhs)
        for i in reversed(range(len(rhs))):
            solution[i] = (rhs[i]-(upper[i]*solution[i+1] if i+1 < len(rhs) else 0))/diagonal[i]
        dot["phi_V"][1:-1] = solution

        def readings(rate):
            charge = rho_rate(rate)
            d = [-eps*(rate["phi_V"][i+1]-rate["phi_V"][i])/dx[i] for i in range(N-1)]
            metal = [area*d[0]-volume[0]*charge[0], -area*d[-1]-volume[-1]*charge[-1]]
            conduction = [q*volume[i]*(rate["p_m3"][i]-rate["n_m3"][i])
                          +sign*area*(jn[face]+jp[face])-q*volume[i]*nt*(rn[i]-rp[i])
                          for i, face, sign in ((0, 0, 1), (N-1, N-2, -1))]
            return [sum(conduction), *metal], [a+b for a, b in zip(conduction, metal, strict=True)]
        return readings(rates), readings(dot)


@pytest.mark.parametrize("sign", [-1, 0, 1])
def test_exact_polynomial_signed_weak_and_zero(sign, request):
    weak = sign*Fraction(1, 2**180)
    p = Polynomial((Fraction(7, 13), weak, -weak))
    expected = weak*Fraction(1, 6)+Fraction(7, 13)
    assert p.integral(0, 1) == expected
    assert p.derivative().integral(0, 1) == 0
    assert Polynomial((0, 1, -1)).range(0, 1) == (0, Fraction(1, 2))
    report(request, integral=expected, weak_increment=weak, exactly_zero=sign == 0)


def test_newton_basis_fraction_and_quotient(request):
    result = newton_polynomials(((3,), (7,), (11,)), (2, 5), 2)[0]
    assert result == Polynomial((3, Fraction(57, 5), Fraction(22, 5)))
    assert result.derivative().derivative().at(0)/4 == Fraction(11, 5)
    with pytest.raises(ContractError, match="basis_shape_or_order"):
        newton_polynomials(((0,), (1,)), (0,), 1)
    report(request, coefficients=result.payload(), native_provenance=False)


def test_actual_saved_predecessor_strip_rejected(request):
    source = Path(os.environ["IDA_SAVED_READBACK_AUDIT"])
    evidence = json.loads(source.read_text())["native_interval_vs_previous_time_discrepancies"][0]
    current = evidence["current_step"]
    c = clock(predecessor=float.fromhex(evidence["previous_output_time_hex"]), tn=current["tn"],
              hused=current["hused"], generation=current["generation"], nsteps=current["nsteps"],
              kused=current["kused"])
    assert c.strip == Fraction(1, 2**55)
    with pytest.raises(ContractError, match="uncovered_predecessor_strip"):
        c.require_covered()
    report(request, actual_clock=c.payload(), new_native_calls=0)


def test_event_generation_and_missing_interval_guards():
    c = clock()
    c.require_successor(clock(predecessor=1, tn=2, generation=2, nsteps=1,
                              segment_start=1, segment_end=2, segment_id="next"))
    with pytest.raises(ContractError, match="event_transition"):
        c.require_successor(clock(predecessor=1, tn=2, generation=1, nsteps=1,
                                  segment_start=1, segment_end=2, segment_id="next"))
    with pytest.raises(ContractError, match="gap_or_overlap"):
        c.require_successor(clock(predecessor=Fraction(3, 2), tn=2, segment_end=2))


def test_absolute_prefix_no_cancellation_and_zero():
    prefix = ChargePrefix.start((1, 0))
    prefix, passed = prefix.append((Enclosure(Fraction(3, 5)), Enclosure(0)),
                                   (Enclosure(0), Enclosure(0)), (0, 0))
    assert passed == (True, True)
    prefix, passed = prefix.append((Enclosure(-Fraction(3, 5)), Enclosure(0)),
                                   (Enclosure(0), Enclosure(0)), (0, 0))
    assert passed == (False, True) and prefix.absolute_defects[0] == Fraction(6, 5)
    with pytest.raises(ContractError, match="missing_charge_error"):
        prefix.append((Enclosure(0),)*2, (Enclosure(0),)*2, (None, 0))


@pytest.mark.parametrize("case_id", CASES)
@pytest.mark.parametrize("area", [1.0, 0.37])
def test_actual_affine_forms_and_half_cells(case_id, area, request):
    device = model(case_id, area)
    path = path_for(device, weak=True)
    actual = affine_interval_actions(device, path)
    q, m = rational(Q), device.definition
    for name, row in actual.items():
        assert row["rate_integral"] == tuple(b-a for a, b in zip(row["left"], row["right"], strict=True))
    expected = Fraction(0)
    for i, volume in enumerate(device.geometry.volumes):
        derivative = path.fields["p_m3"][i].derivative()-path.fields["n_m3"][i].derivative()
        if m.dynamic:
            derivative += path.fields["c_m3"][i].derivative()-rational(m.trap_density)*path.fields["f"][i].derivative()
        expected += q*rational(volume)*derivative.integral(-1, 0)
    assert actual["body_charge"]["rate_integral"] == (expected,)
    assert rational(device.geometry.volumes[0]) > 0 and rational(device.geometry.volumes[-1]) > 0
    assert path.identity != replace(path, coefficient_identity="other").identity
    report(request, area_m2=area, body_delta=expected, half_volumes=list(map(rational, device.geometry.volumes[[0,-1]])),
           form_identities={key: value["form_identity"] for key, value in actual.items()})


def test_public_full_word_endpoints_are_retained(request):
    device = model(CASES[0], 0.37)
    from scripts.benchmarks.precision_prototype import DoubleArray
    high, low = np.zeros(device.layout.size), np.zeros(device.layout.size)
    i = device.layout.offsets["n_m3"].start+3
    high[i], low[i] = 1., 2.**-90
    right, increment = device.trial(DoubleArray(high, low), 1., (0., 0.))
    value = public_action_enclosures(device.linear_action("body_charge", right, increment=increment, left=device.reference))[0]
    expected = -rational(Q)*rational(device.geometry.volumes[3])*(1+Fraction(1, 2**90))
    assert abs(value.center-expected) <= value.radius
    fields = {}
    for variable in device.layout.variables:
        root = device.field(device.reference, variable.id)
        fields[variable.id] = [Polynomial((rational(a)+rational(b),)) for a, b in zip(root.high, root.low, strict=True)]
    fields["n_m3"][3] += Polynomial((1+Fraction(1, 2**90), 1+Fraction(1, 2**90)))
    path = PolynomialPath(clock(), fields, (Polynomial(), Polynomial()), device.source_identity,
                          device.layout.identity, "full-word-endpoint-test", "synthetic_component")
    mismatch = endpoint_charge_mismatch(device, path, device.reference, right)
    assert all(v >= 0 for v in mismatch)
    assert mismatch[0] <= Fraction(1, 10**48)
    report(request, public_value=value, exact_delta=expected, endpoint_mismatch=mismatch)


@pytest.mark.parametrize("value", [Fraction(-1, 10), Fraction(0), Fraction(1, 10), Fraction(1, 2**90)])
def test_bernoulli_decimal120_including_removable_zero(value, request):
    a = BallIntegrator()
    result = a.point(lambda x, _: a.bernoulli(x), value)
    with localcontext() as ctx:
        ctx.prec = 120
        x = decimal(value)
        expected = Decimal(1) if value == 0 else x/(x.exp()-1)
    if value == 0:
        assert result == Enclosure(1)
    else:
        assert abs(result.center-Fraction(expected)) <= result.radius+Fraction(1, 10**85)
    report(request, argument=value, ball=result, decimal120=expected, oracle_roundoff_allowance=Fraction(1, 10**85))


def test_integral_known_exact_value_and_absolute_kink(request):
    a = BallIntegrator()
    polynomial = a.integrate(lambda z, _: 3*z*z+2*z+1, 0, 1, ORACLE_GOAL)
    absolute = a.absolute_bound(lambda z, _: z, -1, 1, ORACLE_GOAL)
    assert abs(polynomial.center-3) <= polynomial.radius
    assert absolute.upper >= 1
    assert absolute.upper**2 >= Fraction(4, 3)
    assert (absolute.upper-ORACLE_GOAL)**2 <= Fraction(4, 3)
    report(request, exact_polynomial=polynomial, absolute_kink=absolute, calls=a.calls)


@pytest.mark.parametrize("sign", [-1, 0, 1])
def test_absolute_bound_signed_weak_and_strict_zero(sign, request):
    a = BallIntegrator()
    value = sign*Fraction(1, 2**120)
    bound = a.absolute_bound(lambda z, _: a.number(value), -1, 1, ORACLE_GOAL)
    assert 2*abs(value) <= bound.upper <= 2*abs(value)+ORACLE_GOAL
    if sign == 0:
        assert bound.upper == 0 and bound.squared_integral == Enclosure(0)
    else:
        assert bound.upper > 0
    report(request, signed_value=value, L1_bound=bound, calls=a.calls)


def test_varying_bernoulli_integral_independent_decimal_series(request):
    a = BallIntegrator()
    actual = a.integrate(lambda z, _: a.bernoulli(z), 1, 2, ORACLE_GOAL)
    with localcontext() as context:
        context.prec = 140
        reference = Decimal(0)
        for k in range(1, 401):
            k = Decimal(k)
            reference += (1/k+1/k**2)*(-k).exp()-(2/k+1/k**2)*(-2*k).exp()
        tail = (Decimal(1)/401+Decimal(1)/401**2)*Decimal(-401).exp()/(1-Decimal(-1).exp())
    difference = abs(actual.center-Fraction(reference))
    assert difference <= actual.radius+Fraction(tail)+Fraction(1, 10**120)
    report(request, result=actual, decimal140=reference, positive_series_tail=tail,
           difference=difference, oracle_roundoff_allowance=Fraction(1, 10**120), calls=a.calls)


def test_varying_vacancy_log_integral(request):
    a = BallIntegrator()
    actual = a.integrate(lambda z, analytic: (1-z).log(analytic=analytic), 0, Fraction(1, 2), ORACLE_GOAL)
    with localcontext() as context:
        context.prec = 120
        half = Decimal('0.5')
        reference = -half-half*half.ln()
    assert abs(actual.center-Fraction(reference)) <= actual.radius+Fraction(1, 10**110)
    report(request, result=actual, decimal120=reference, analytic_flag_propagated=True, calls=a.calls)


def test_negative_ball_analytic_and_budget_cases(request):
    a = BallIntegrator(evaluations=1)
    with pytest.raises(ContractError) as failure:
        a.integrate(lambda z, _: z.exp(), 0, 1, ORACLE_GOAL)
    # The mature library may stop at its own soft work limit before asking
    # for a second callback. Both paths must refuse an unqualified result.
    assert str(failure.value) in {"observation_integral_evaluation_cap",
                                  "observation_integral_radius_exceeds_budget"}
    if str(failure.value) == "observation_integral_radius_exceeds_budget":
        assert a.calls[-1]["evaluations"] <= 1
    with pytest.raises(ContractError, match="nonfinite_or_nonreal"):
        a.point(lambda z, analytic: z.log(analytic=analytic), -1)
    ordinary = BallIntegrator()
    with pytest.raises(ContractError, match="absolute_requires_real_enclosure"):
        ordinary.absolute_bound(lambda z, _: ordinary.flint.acb(0, 1), 0, 1, ORACLE_GOAL)
    report(request, bounded_rejection=str(failure.value), invalid_log_domain_rejected=True,
           nonreal_squared_callback_rejected=True,
           optional_dependency={"python_flint": ordinary.flint.__version__,
                                "FLINT": ordinary.flint.__FLINT_VERSION__,
                                "module": ordinary.flint.__file__})


def test_complex_callback_with_real_squared_integral_rejected(request):
    a = BallIntegrator()
    # Integral (1+i*u)^2 is real4/3, but sqrt(8/3) does not bound
    # integral |1+i*u|. Reject the range before using that false L1 proof.
    with pytest.raises(ContractError, match="absolute_requires_real_enclosure"):
        a.absolute_bound(lambda z, _: 1+a.flint.acb(0, 1)*z, -1, 1, ORACLE_GOAL)
    with pytest.raises(ContractError, match="absolute_nonfinite_real_range"):
        a.absolute_bound(lambda z, _: 1/z, -1, 1, ORACLE_GOAL)
    report(request, complex_counterexample_rejected=True, nonfinite_full_range_rejected=True)


@pytest.mark.parametrize("case_id", CASES)
@pytest.mark.parametrize("area", [1.0, 0.37])
def test_physical_integrals_independent_decimal_fv(case_id, area, request):
    device = model(case_id, area)
    path = path_for(device, sign=-1 if area == 0.37 else 1)
    observer, arithmetic = SlabPathObserver(device, path), BallIntegrator()
    result = observer.integrate(arithmetic, ORACLE_GOAL)
    errors = []
    with localcontext() as context:
        context.prec = 120
        samples = [decimal_currents(device, path, u) for u in (-1, Fraction(-1, 2), 0)]
        for variant, key in enumerate(("raw_polynomial", "same_state_affine_tangent")):
            for j, ball in enumerate(result[key]):
                reference = (samples[0][variant][0][j]+4*samples[1][variant][0][j]+samples[2][variant][0][j])/6
                error = abs(ball.center-Fraction(reference))
                # Decimal120 is an independent numerical oracle. It is not
                # the source of the Arb certificate or a DAE error theorem.
                allowance = Fraction(1, 10**85)
                errors.append({"row": key+str(j), "difference": error, "ball": ball, "oracle_allowance": allowance})
                assert error <= ball.radius+allowance
    assert all(bound.upper >= 0 for bound in result["raw_tangent_L1_upper_bounds"])
    assert result["DAE_time_accuracy_certified"] is False
    report(request, path_identity=path.identity, case=case_id, area=area, errors=errors,
           absolute_gap_bounds=result["raw_tangent_L1_upper_bounds"], calls=arithmetic.calls,
           native_steps=0, full_protocol_qualified=False)


def test_uncovered_strip_explicit_extension_debit(request):
    device = model(CASES[0], 0.37)
    c = clock(predecessor=Fraction(0), tn=Fraction(1), hused=1-Fraction(1, 2**55))
    path = path_for(device, moving=False, selected_clock=c)
    with pytest.raises(ContractError, match="uncovered_predecessor_strip"):
        SlabPathObserver(device, path)
    extended = replace(path, origin="declared_reconstruction", clock_policy="declared_polynomial_extension")
    debit = SlabPathObserver(device, extended).strip_current_debit(BallIntegrator())
    assert c.strip == Fraction(1, 2**55)
    assert all(x >= 0 for rows in debit.values() for x in rows)
    assert extended.identity != path.identity
    assert any(x > 0 for x in debit["raw_polynomial"])
    report(request, strip_width=c.strip, separate_current_debit=debit, origin=extended.origin,
           native_getter_success_proven=False)


def test_active_carrier_and_vacancy_domains_reject():
    device = model(CASES[1], 1.0)
    path = path_for(device)
    for name, value, message in (("n_m3", 0, "nonpositive_active_carrier"),
                                  ("c_m3", device.definition.ion_capacity, "vacancy_log_domain"),
                                  ("f", 2, "domain_upper")):
        fields = dict(path.fields); fields[name] = (Polynomial((value,)), *fields[name][1:])
        with pytest.raises(ContractError, match=message):
            SlabPathObserver(device, replace(path, fields=fields))


def test_input_rounding_endpoint_snap_and_fail_closed(request):
    segment = ProtocolSegment("synthetic", 0., 1., (0., 0.1), (0., 2e16))
    bounds = segment_input_roundoff(segment, clock())
    for t in (0., 0.1, 0.7, 1.):
        inputs, _ = segment.inputs(t)
        for j, record in enumerate(bounds):
            error = abs(rational(inputs[j])-record["stored_slope"]*rational(t))
            assert error <= record["interior_arithmetic_bound"]+record["end_snap_mismatch"]
    ready = native_observation_readiness(native_basis=None, input_map_error=None, endpoint_error=(),
                                         clock_error=None, nonlinear_error=None, tangent_error=None,
                                         independent_review=None)
    assert not ready["native_admitted"] and "native_basis" in ready["missing"]
    assert ready["full_protocols_pending"] == {"S0_s": 1.2e-6, "dynamic_ion_trap_s": 9.2}
    report(request, input_error_bounds=bounds, readiness=ready)
