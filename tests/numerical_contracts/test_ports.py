"""Independent NC06 electrostatic, RC and native-interval charge references."""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from hashlib import sha256
from math import fsum
from pathlib import Path
import json
import os

import numpy as np
import pytest
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss

from scripts.benchmarks.contract_prototype import (
    BalanceTerms, ContractError, ImplicitSystem, LinearStorage, StoragePartials,
)
from scripts.benchmarks.port_prototype import (
    Capacitor, NativeInterval, PortReading, RCModel,
    consistent_physical_rate, rc_be, rc_ida,
)


GATE_PATH = Path(__file__).with_name("AnalyticGatesV1.json")
GATE_SHA = "43c9ba4ab1b6a95b3e278efcd8eed9a67dc4d5e258295c8e6961c53f7ee57201"
assert sha256(GATE_PATH.read_bytes()).hexdigest() == GATE_SHA
GATE = next(x for x in json.loads(GATE_PATH.read_text())["gates"] if x["id"] == "NC06")
CONTROL_PLAN_SHA = "77244fde160546b41aaf60a086ec231636219c3948711da9aeeccc870512f571"
DERIVATIVE_PLAN_SHA = "dea9706bc57325bb90c9119b9480aab5ba4f871f02a0c171bc1913c7f37c0b4f"


def effective_ida_controls():
    if path := os.environ.get("PORT_CONTROL_PLAN"):
        payload = Path(path).read_bytes()
        digest = sha256(payload).hexdigest()
        assert digest in {CONTROL_PLAN_SHA, DERIVATIVE_PLAN_SHA}
        plan = json.loads(payload)
        assert plan["original_gate_sha256"] == GATE_SHA
        candidate = plan["controls"] if digest == DERIVATIVE_PLAN_SHA else plan["proposed_additional_candidate"]
        controls = {key: candidate[key] for key in ("rtol", "atol", "max_step", "max_order")}
        if digest == DERIVATIVE_PLAN_SHA:
            controls["derivative_policy"] = plan["derivative_policy"]
        emit({"kind": "effective_controls", "controls": controls, "plan_sha256": digest})
        return controls
    controls = GATE["fixed_inputs"]["rc"]["ida"]
    emit({"kind": "effective_controls", "controls": controls, "plan_sha256": None})
    return controls


def emit(record):
    if path := os.environ.get("PORT_CASE_LOG"):
        def encode(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value).__name__)
        with Path(path).open("a") as stream:
            stream.write(json.dumps(record, default=encode, allow_nan=False) + "\n")


def compare(values, references, atol, rtol):
    values = np.asarray(values).ravel()
    references = list(references)
    assert values.size == len(references)
    with localcontext() as ctx:
        ctx.prec = 80
        errors = [abs(Decimal.from_float(float(v)) - r) for v, r in zip(values, references)]
        budgets = [Decimal(str(atol)) + Decimal(str(rtol)) * abs(r) for r in references]
    return {"reference_decimal80": [str(r) for r in references],
            "errors": [float(e) for e in errors], "budgets": [float(b) for b in budgets],
            "passed": all(e <= b for e, b in zip(errors, budgets))}


def rc_reference(time, C=2, R=3, initial=1):
    with localcontext() as ctx:
        ctx.prec = 80
        return Decimal(str(initial)) * (-Decimal.from_float(float(time)) /
                                       (Decimal(str(C)) * Decimal(str(R)))).exp()


def assert_record(request, record, checks):
    record.update(test=request.node.nodeid, checks={k: bool(v) for k, v in checks.items()},
                  passed=all(checks.values()))
    emit(record)
    assert all(checks.values()), record


@pytest.mark.parametrize("slope", GATE["fixed_inputs"]["capacitor"]["slopes"])
def test_capacitor_ramps_and_input_feedthrough(request, slope):
    p = GATE["fixed_inputs"]["capacitor"]
    capacitor = Capacitor(p["epsilon"], p["area"], p["length"])
    checks, records = {}, []
    for time in p["times"]:
        sample = capacitor.sample(time, p["V0"] + slope * time, slope)
        with localcontext() as ctx:
            ctx.prec = 80
            v = Decimal(str(p["V0"])) + Decimal(str(slope)) * Decimal(str(time))
            q, current = Decimal(str(p["C"])) * v, Decimal(str(p["C"])) * Decimal(str(slope))
        record = compare(np.r_[sample.charge, sample.current, sample.conduction],
                         [q, q.copy_negate(), current, current.copy_negate(), Decimal(0), Decimal(0)],
                         GATE["algebraic_atol"], GATE["algebraic_rtol"])
        checks[str(time)] = record["passed"]
        records.append({"time": time, **record})
    steps = [capacitor.ramp_step(a, b, p["V0"] + slope * a, slope)
             for a, b in zip(p["times"][:-1], p["times"][1:])]
    for step in steps:
        checks[f"empty-storage-{step.right.time}"] = (step.left.y.size == 0 and step.storage_delta.size == 0)
        checks[f"charge-identity-{step.right.time}"] = bool(np.max(np.abs(step.residual)) < GATE["algebraic_atol"])
        accepted = step.accept((("charge_identity_defect", float(np.max(np.abs(step.residual)))),),
                               checks[f"charge-identity-{step.right.time}"])
        assert accepted.input_rate[0] == slope
    assert_record(request, {"records": records}, checks)


@pytest.mark.parametrize("frequency", GATE["fixed_inputs"]["capacitor"]["frequencies_Hz"])
def test_direct_capacitance_admittance(request, frequency):
    p = GATE["fixed_inputs"]["capacitor"]
    capacitor = Capacitor(p["epsilon"], p["area"], p["length"])
    # Decimal pi is independent of NumPy's binary64 trigonometric constant.
    with localcontext() as ctx:
        ctx.prec = 80
        pi = Decimal("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899")
        imaginary = 2 * pi * Decimal(str(frequency)) * Decimal(str(p["C"]))
    values = capacitor.admittance(frequency)
    comparison = compare(values.imag, [imaginary, imaginary.copy_negate()],
                         GATE["algebraic_atol"], GATE["algebraic_rtol"])
    assert_record(request, comparison,
                  {"imaginary": comparison["passed"], "real_zero": bool(np.all(values.real == 0))})


def test_voltage_event_and_immutable_history(request):
    p = GATE["fixed_inputs"]["event"]
    capacitor = Capacitor(p["C"], 1, 1)
    history = capacitor.ramp_step(0, p["time"], 0, 0)
    event = capacitor.event(p["time"], p["V_before"], p["V_after"])
    comparison = compare(event.charge,
                         [Decimal(str(p["expected_left_impulse_charge"])),
                          Decimal(str(p["expected_right_impulse_charge"]))],
                         GATE["algebraic_atol"], GATE["algebraic_rtol"])
    assert event.before.voltage == 0 and event.after.voltage == .1
    assert event.before.side == "left" and event.after.side == "right"
    assert event.before.time == event.after.time == history.right.time
    event_limits = []
    for sample, voltage in ((event.before, p["V_before"]), (event.after, p["V_after"])):
        with localcontext() as ctx:
            ctx.prec = 80
            charge = Decimal(str(p["C"])) * Decimal(str(voltage))
        event_limits.append(compare(
            np.r_[sample.charge, sample.current, sample.conduction],
            [charge, charge.copy_negate(), Decimal(0), Decimal(0), Decimal(0), Decimal(0)],
            GATE["algebraic_atol"], GATE["algebraic_rtol"]))
    for array in (event.charge, event.after.charge, history.right.inputs):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    leaked = event.charge
    leaked.shape = (1, 2)
    assert event.charge.shape == (2,)
    with pytest.raises(FrozenInstanceError):
        event.after.voltage = 2
    with pytest.raises(ContractError, match="physical_step_rejected"):
        history.accept((("test", 0.0),), False)
    with pytest.raises(ContractError, match="nonpositive_step"):
        capacitor.ramp_step(.5, .5, 0, 0)
    assert_record(request, {"impulse": comparison, "event_limits": event_limits},
                  {"charge_impulse": comparison["passed"],
                   "both_event_limits": all(x["passed"] for x in event_limits),
                   "old_history_preserved": history.right.inputs[0] == 0})


def test_rc_be_all_grids_and_ports(request):
    p = GATE["fixed_inputs"]["rc"]
    records, errors, checks = [], [], {}
    for h in p["be_step_sizes"]:
        samples, steps = rc_be(RCModel(p["C"], p["R"]), p["V0"], p["end_time"], h)
        with localcontext() as ctx:
            ctx.prec = 80
            ratio = 1 + Decimal(str(h)) / (Decimal(str(p["R"])) * Decimal(str(p["C"])))
            reference = [Decimal(str(p["V0"])) / ratio**i for i in range(len(samples))]
        discrete = compare([s.voltage for s in samples], reference,
                           p["be_discrete_state_atol"], p["be_discrete_state_rtol"])
        continuous_errors = [abs(s.voltage - float(rc_reference(s.time))) for s in samples]
        maximum = max(continuous_errors)
        errors.append(maximum)
        defects = [float(np.max(np.abs(s.charge - old.charge - h * (s.current - s.conduction))))
                   for old, s in zip(samples[:-1], samples[1:])]
        charge_ok = all(d <= GATE["integrated_charge_atol"] for d in defects)
        cumulative_defects = [fsum(defects[:i + 1]) for i in range(len(defects))]
        interval_integrals = [h * (s.current[0] - s.conduction[0]) for s in samples[1:]]
        cumulative_budgets = [GATE["integrated_charge_atol"] + GATE["integrated_charge_rtol"] *
                              abs(fsum(interval_integrals[:i + 1])) for i in range(len(defects))]
        cumulative_ok = all(d <= b for d, b in zip(cumulative_defects, cumulative_budgets))
        current_ok = all(np.max(np.abs(s.current)) <= GATE["trajectory_atol"] for s in samples)
        for step, defect in zip(steps, defects):
            step.accept((("interval_charge_defect", defect),), defect <= GATE["integrated_charge_atol"])
        records.append({"h": h, "times": [s.time for s in samples], "voltages": [s.voltage for s in samples],
                        "discrete": discrete, "continuous_max_error": maximum,
                        "interval_charge_defects": defects,
                        "cumulative_absolute_charge_defects": cumulative_defects,
                        "cumulative_budgets": cumulative_budgets})
        checks[str(h)] = discrete["passed"] and charge_ok and current_ok and cumulative_ok
    ratios = [a / b for a, b in zip(errors[:-1], errors[1:])]
    checks["continuous_finest"] = errors[-1] <= p["be_continuous_finest_error_limit"]
    checks["first_order"] = all(p["be_adjacent_error_ratio"][0] <= r <= p["be_adjacent_error_ratio"][1] for r in ratios)
    assert_record(request, {"grids": records, "adjacent_error_ratios": ratios}, checks)


def test_rc_ida_trajectory_and_every_native_interval(request):
    p = GATE["fixed_inputs"]["rc"]
    model = RCModel(p["C"], p["R"])
    intervals, leases, accepted = [], [], []
    nodes, weights = leggauss(16)
    cumulative_defects = np.zeros(2)
    cumulative_reference_error = np.zeros(2)
    cumulative_integral = np.zeros(2)
    cumulative_raw_reference_error = np.zeros(2)
    cumulative_raw_integral = np.zeros(2)
    controls = effective_ida_controls()
    reconstructed = controls.get("derivative_policy") == "physical-tangent-v1"

    def observe(interval):
        step = interval.step
        a, b = step.left.time, step.right.time
        right_reading = interval.reading(b)
        if reconstructed:
            interval.physical_reading(right_reading)
        integrals, error_estimates, discrepancies = [], [], []
        raw_integrals, raw_errors, raw_discrepancies = [], [], []
        for port in range(2):
            def current(time):
                sample = interval.sample(time)
                return sample.current[port] - sample.conduction[port]
            integral, estimate = quad(current, a, b, epsabs=1e-12, epsrel=1e-12)
            independent = (b - a) / 2 * fsum(
                float(weight) * current((a + b) / 2 + float(node) * (b - a) / 2)
                for node, weight in zip(nodes, weights))
            integrals.append(integral)
            error_estimates.append(estimate)
            discrepancies.append(abs(integral - independent))
            if reconstructed:
                def raw_current(time):
                    sample = interval.sample(time, derivative_policy="interpolant")
                    return sample.current[port] - sample.conduction[port]
                raw_integral, raw_estimate = quad(raw_current, a, b, epsabs=1e-12, epsrel=1e-12)
                raw_independent = (b - a) / 2 * fsum(
                    float(weight) * raw_current((a + b) / 2 + float(node) * (b - a) / 2)
                    for node, weight in zip(nodes, weights))
                raw_integrals.append(raw_integral)
                raw_errors.append(raw_estimate)
                raw_discrepancies.append(abs(raw_integral - raw_independent))
        charge_delta = np.array([p["C"], -p["C"]]) * (step.right.y[0] - step.left.y[0])
        defect = np.abs(charge_delta - integrals)
        budget = GATE["integrated_charge_atol"] + GATE["integrated_charge_rtol"] * np.abs(integrals)
        reference_error = np.asarray(error_estimates) + discrepancies + GATE["reference_error_bound"]
        cumulative_defects[:] += defect
        cumulative_reference_error[:] += reference_error
        cumulative_integral[:] += integrals
        cumulative_budget = GATE["integrated_charge_atol"] + GATE["integrated_charge_rtol"] * np.abs(cumulative_integral)
        with localcontext() as ctx:
            ctx.prec = 80
            analytic_charge = Decimal(str(p["C"])) * (rc_reference(b) - rc_reference(a))
        analytic_interval = compare(integrals, [analytic_charge, analytic_charge.copy_negate()],
                                    GATE["integrated_charge_atol"], GATE["integrated_charge_rtol"])
        checks = {"interval_charge": bool(np.all(defect + reference_error <= budget)),
                  "interval_reference_budget": bool(np.all(reference_error <= budget / 3)),
                  "cumulative_charge": bool(np.all(cumulative_defects + cumulative_reference_error <= cumulative_budget)),
                  "cumulative_reference_budget": bool(np.all(cumulative_reference_error <= cumulative_budget / 3))}
        analytic_prefix = None
        raw_reference_error = raw_budget = raw_prefix_budget = None
        if reconstructed:
            raw_reference_error = np.asarray(raw_errors) + raw_discrepancies + GATE["reference_error_bound"]
            raw_budget = GATE["integrated_charge_atol"] + GATE["integrated_charge_rtol"] * np.abs(raw_integrals)
            cumulative_raw_reference_error[:] += raw_reference_error
            cumulative_raw_integral[:] += raw_integrals
            raw_prefix_budget = GATE["integrated_charge_atol"] + GATE["integrated_charge_rtol"] * np.abs(cumulative_raw_integral)
            checks["raw_interval_reference_budget"] = bool(np.all(raw_reference_error <= raw_budget / 3))
            checks["raw_cumulative_reference_budget"] = bool(np.all(cumulative_raw_reference_error <= raw_prefix_budget / 3))
            with localcontext() as ctx:
                ctx.prec = 80
                exact_prefix = Decimal(str(p["C"])) * (rc_reference(b) - Decimal(str(p["V0"])))
            analytic_prefix = compare(cumulative_integral, [exact_prefix, exact_prefix.copy_negate()],
                                      GATE["integrated_charge_atol"], GATE["integrated_charge_rtol"])
            checks["analytical_interval_charge"] = all(
                error + bound <= allowed for error, bound, allowed in zip(
                    analytic_interval["errors"], reference_error, analytic_interval["budgets"]))
            checks["analytical_prefix_charge"] = all(
                error + bound <= allowed for error, bound, allowed in zip(
                    analytic_prefix["errors"], cumulative_reference_error, analytic_prefix["budgets"]))
        record = {"kind": "native_interval", "a": a, "b": b,
                  "output_method": step.method,
                  "left": step.left.y, "right": step.right.y, "derivative": step.derivative,
                  "residual": step.residual, "charge_delta": charge_delta,
                  "integral": integrals, "quadrature_error_estimate": error_estimates,
                  "independent_rule_discrepancy": discrepancies,
                  "reference_error_bound": reference_error, "absolute_defect": defect,
                  "budget": budget, "cumulative_absolute_defect": cumulative_defects.copy(),
                  "cumulative_budget": cumulative_budget, "checks": checks,
                  "analytic_interval_diagnostic": analytic_interval,
                  "analytic_prefix": analytic_prefix,
                  "raw_integrals": raw_integrals,
                  "raw_quadrature_error_estimates": raw_errors,
                  "raw_independent_rule_discrepancies": raw_discrepancies,
                  "raw_reference_error_bound": raw_reference_error,
                  "raw_budget": raw_budget,
                  "raw_cumulative_reference_error_bound": cumulative_raw_reference_error.copy(),
                  "raw_prefix_budget": raw_prefix_budget}
        emit(record)
        intervals.append(record)
        leases.append(interval)
        if all(checks.values()):
            accepted.append(step.accept((("charge_defect", float(np.max(defect))),), True))
        with pytest.raises(ContractError, match="outside_native_interval"):
            interval.sample(np.nextafter(a, -np.inf))

    result = rc_ida(model, p["V0"], p["end_time"], p["observation_times"],
                    **controls, observer=observe)
    observations, checks = [], {}
    for sample in result.observations:
        v = rc_reference(sample.time)
        with localcontext() as ctx:
            ctx.prec = 80
            q, current = Decimal(str(p["C"])) * v, v / Decimal(str(p["R"]))
        record = compare(np.r_[sample.voltage, sample.charge, sample.conduction, sample.current],
                         [v, q, q.copy_negate(), current, current.copy_negate(), Decimal(0), Decimal(0)],
                         GATE["trajectory_atol"], GATE["trajectory_rtol"])
        observations.append({"time": sample.time, **record})
        checks[f"observation_{sample.time}"] = record["passed"]
    for interval in leases:
        with pytest.raises(ContractError, match="interpolation_interval_expired"):
            interval.sample(interval.step.right.time)
    checks["every_interval"] = len(accepted) == len(result.native_steps) == len(intervals)
    checks["all_observations"] = [s.time for s in result.observations] == p["observation_times"]
    # The native derivative is retained independently of each endpoint quotient.
    gap = max(abs(step.derivative[0] - (step.right.y[0] - step.left.y[0]) /
                  (step.right.time - step.left.time)) for step in result.native_steps)
    checks["derivative_not_quotient"] = gap > 1e-7
    endpoint_differences = [out.voltage_rate - step.derivative[0]
                            for out, step in zip(result.endpoint_outputs, result.native_steps)]
    endpoint_state_differences = [out.voltage - step.right.y[0]
                                  for out, step in zip(result.endpoint_outputs, result.native_steps)]
    endpoint_comparison = compare(
        [out.voltage for out in result.endpoint_outputs],
        [rc_reference(out.time) for out in result.endpoint_outputs],
        GATE["trajectory_atol"], GATE["trajectory_rtol"])
    checks["endpoint_interpolation_states"] = endpoint_comparison["passed"]
    endpoint_current_diagnostics = [{
        "time": step.right.time,
        "onestep_return_current": model.sample(step.right, step.derivative).current,
        "normal_return_current": out.current,
    } for out, step in zip(result.endpoint_outputs, result.native_steps)]
    physical_rate_errors, physical_current_errors = [], []
    for reading in result.raw_readings + result.physical_readings:
        record = {"kind": "retained_reading", "origin": reading.origin,
                  "time": reading.point.time, "point_identity": reading.point.identity,
                  "state": reading.point.y, "inputs": reading.point.inputs,
                  "derivative": reading.derivative, "charge": reading.sample.charge,
                  "conduction": reading.sample.conduction, "current": reading.sample.current}
        if reading.origin.startswith("physical-tangent-v1"):
            v = rc_reference(reading.point.time)
            with localcontext() as ctx:
                ctx.prec = 80
                rate_ref = -v / (Decimal(str(p["R"])) * Decimal(str(p["C"])))
                displacement_ref = -v / Decimal(str(p["R"]))
            rate_check = compare([reading.derivative[0]], [rate_ref], 5e-11, 0)
            displacement = reading.sample.current[0] - reading.sample.conduction[0]
            displacement_check = compare([displacement], [displacement_ref], 1e-10, 0)
            physical_rate_errors.extend(rate_check["errors"])
            physical_current_errors.extend(displacement_check["errors"])
            record.update(analytic_rate=rate_check, analytic_displacement=displacement_check)
        emit(record)
    if reconstructed:
        checks["physical_rate_at_all_retained_points"] = bool(physical_rate_errors) and max(physical_rate_errors) <= 5e-11
        checks["physical_displacement_at_all_retained_points"] = bool(physical_current_errors) and max(physical_current_errors) <= 1e-10
    assert_record(request, {"native_intervals": len(intervals), "accepted_intervals": len(accepted),
                           "observations": observations, "derivative_quotient_max_gap": gap,
                           "normal_minus_native_endpoint_derivative": endpoint_differences,
                           "normal_minus_native_endpoint_state": endpoint_state_differences,
                           "endpoint_comparison": endpoint_comparison,
                           "endpoint_current_diagnostics": endpoint_current_diagnostics,
                           "derivative_policy": controls.get("derivative_policy", "interpolant"),
                           "raw_reading_count": len(result.raw_readings),
                           "physical_reading_count": len(result.physical_readings),
                           "max_physical_rate_error": max(physical_rate_errors, default=None),
                           "max_physical_displacement_error": max(physical_current_errors, default=None),
                           "analytical_interval_failures": sum(not x["analytic_interval_diagnostic"]["passed"]
                                                               for x in intervals)}, checks)


def driven_system(affine=False, missing_gauge=False):
    coordinates, _ = RCModel(2, 3).system()

    class AffineCharge:
        def value(self, point):
            return np.array([2 * point.y[0] + point.inputs[0] / 4 + point.time / 16])

        def first(self, point):
            return StoragePartials(np.array([[2., 0.]]), np.array([[.25]]), np.array([.0625]))

        def delta(self, left, right, increment):
            increment.validate(left, right)
            return self.value(right) - self.value(left)

        def rate_jvp(self, point, ydot, adot, dy, da, dt):
            return np.zeros(1)

    def terms(point):
        v, current = point.y
        return BalanceTerms(
            np.array([current - v / 3]), np.array([current - point.inputs[0] - point.time / 8]),
            np.array([[-1 / 3, 1]]), np.zeros((1, 1)), np.zeros(1),
            np.zeros((1, 2)) if missing_gauge else np.array([[0., 1.]]),
            np.array([[-1.]]), np.array([-.125]),
        )
    return coordinates, ImplicitSystem(AffineCharge() if affine else LinearStorage([[2., 0.]]), terms)


@pytest.mark.parametrize("affine", [False, True])
def test_physical_tangent_nonzero_drive_and_full_input_time_chain(request, affine):
    coordinates, system = driven_system(affine)
    point = coordinates.point([.75, .4375], .5, [.375])
    history = (coordinates.point([.5, .375], 0, [.375]),)
    identities = point.identity, tuple(p.identity for p in history)
    derivative = consistent_physical_rate(system, coordinates, point, [-.0625], history=history)
    expected_rate = Decimal("0.0703125") if affine else Decimal("0.09375")
    rates = compare(derivative, [expected_rate, Decimal("0.0625")],
                    GATE["algebraic_atol"], GATE["algebraic_rtol"])
    first = system.storage.first(point)
    charge_rate = (first.y @ derivative + first.inputs @ np.array([-.0625]) + first.time)[0]
    current = point.y[0] / 3 + charge_rate
    currents = compare([current, -current], [Decimal("0.4375"), Decimal("-0.4375")],
                       GATE["algebraic_atol"], GATE["algebraic_rtol"])
    eps = 2.0 ** -10
    plus = coordinates.point(point.y + eps * derivative, point.time + eps, point.inputs - eps * .0625)
    minus = coordinates.point(point.y - eps * derivative, point.time - eps, point.inputs + eps * .0625)
    fd = (system.storage.value(plus) - system.storage.value(minus)) / (2 * eps)
    charge_check = compare(fd, [Decimal("0.1875")], GATE["algebraic_atol"], GATE["algebraic_rtol"])
    assert identities == (point.identity, tuple(p.identity for p in history))
    if affine:
        omitted = point.y[0] / 3 + 2 * derivative[0]
        assert omitted == .390625 and omitted != current
    assert_record(request, {"rates": rates, "currents": currents, "charge_rate": charge_check},
                  {"rate": rates["passed"], "nonzero_current": currents["passed"], "full_charge_chain": charge_check["passed"]})


@pytest.mark.parametrize("bad_current", [1e-15, 1e-6])
def test_physical_tangent_rejects_nonzero_algebraic_state(bad_current):
    coordinates, system = RCModel(2, 3).system()
    point = coordinates.point([1, bad_current])
    with pytest.raises(ContractError, match="nonzero_algebraic_state"):
        consistent_physical_rate(system, coordinates, point, [])


def test_physical_tangent_rejects_missing_gauge():
    coordinates, system = driven_system(missing_gauge=True)
    point = coordinates.point([.75, .4375], .5, [.375])
    with pytest.raises(ContractError, match="missing_gauge"):
        consistent_physical_rate(system, coordinates, point, [-.0625], gauge_nullspace=True)


def test_physical_reading_requires_its_live_interval():
    model = RCModel(2, 3)
    coordinates, _ = model.system()
    _, steps = rc_be(model, 1, .1, .1)
    interval = NativeInterval(None, model, coordinates, steps[0], .1)
    reading = interval.reading(.1)
    assert np.all(interval.physical_reading(reading).sample.current == 0)
    with pytest.raises(ContractError, match="foreign_interval_reading"):
        interval.physical_reading(replace(reading))
    for time in (-.1, .2):
        point = coordinates.point(reading.point.y, time)
        outside = PortReading(point, reading.derivative,
                              model.sample(point, reading.derivative), reading.origin)
        with pytest.raises(ContractError, match="outside_native_interval"):
            interval.physical_reading(outside)
    interval.close()
    with pytest.raises(ContractError, match="interpolation_interval_expired"):
        interval.physical_reading(reading)


def test_zero_reconstructed_current_does_not_certify_wrong_state_or_curve(request):
    model = RCModel(2, 3)
    coordinates, system = model.system()
    point = coordinates.point([1.01, 0], .03)
    rate = consistent_physical_rate(system, coordinates, point, [])
    assert np.all(model.sample(point, rate).current == 0)
    bad_state = compare([point.y[0]], [rc_reference(.03)], GATE["trajectory_atol"], GATE["trajectory_rtol"])
    with localcontext() as ctx:
        ctx.prec = 80
        # Offset curve V=exp(-t/6)+1e-4. A physical derivative estimate has a
        # genuine integral defect, even while its reconstructed total I is0.
        h, offset = Decimal("0.1"), Decimal("0.0001")
        exact_delta = 2 * ((-h / 6).exp() - 1)
        integral = exact_delta - offset * h / 3
        defect = abs(exact_delta - integral)
        budget = Decimal(str(GATE["integrated_charge_atol"])) + Decimal(str(GATE["integrated_charge_rtol"])) * abs(integral)
    assert_record(request, {"wrong_state": bad_state, "offset_curve_defect": float(defect), "budget": float(budget)},
                  {"state_rejected": not bad_state["passed"], "curve_rejected": defect > budget})
