"""Independent NC01--NC05 references for the isolated storage candidate.

Numerical expectations use Decimal80 scalar equations, matrix elimination or
analytic differentiation.  No old device engine or candidate solver supplies an
oracle.  When STORAGE_CASE_LOG is set, every case (including failures) appends its
actual evidence before the assertion; the run receipt freezes that log afterward.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from hashlib import sha256
from itertools import product
from math import fsum
from pathlib import Path
import json
import os
import resource
import time

import numpy as np
import pytest

from scripts.benchmarks.contract_prototype import ContractError, ImplicitSystem
from scripts.benchmarks.storage_prototype import (
    CarrierPotentialStorage, ExponentialStorage, SharedCapacityStorage,
    conservative_be_step, consistent_initial_state, diffusion_probe,
    ida_diffusion_probe, no_constraints, rank_aware_solve, rank_evidence,
    trap_exchange_probe,
)


GATE_PATH = Path(__file__).with_name("AnalyticGatesV1.json")
GATE_SHA256 = "43c9ba4ab1b6a95b3e278efcd8eed9a67dc4d5e258295c8e6961c53f7ee57201"
assert sha256(GATE_PATH.read_bytes()).hexdigest() == GATE_SHA256
GATES = {item["id"]: item for item in json.loads(GATE_PATH.read_text())["gates"]}

# Supplemental diagnostic inputs, frozen with source before execution. They
# extend chain-rule/rank coverage without changing any frozen gate or constant.
SUPPLEMENTAL_INPUTS = {
    "full_ic_chain": {"alpha": 0.3, "beta": 0.7, "gamma": 0.2,
                      "a": 0.1, "time": 0.4, "adot": -0.3, "n": 2},
    "shared_contraction": {"capacity_rate": 0.25, "time": 0.2, "input": 0.95,
                           "ydot": [0.4, -0.2], "adot": 0.3,
                           "dy": [0.2, -0.7], "da": 0.6, "dt": -0.4},
    "shared_finite": {"theta": [0.2, 0.3], "capacity_rate": 0.25,
                      "dy": [0.02, -0.03], "da": 0.01, "dt": 0.005},
    "partial_inactive": [0.0, 0.3],
    "saturated": [0.5, 0.5],
    "ordinary_log_be": {"maxfev": 100},
    "ida_accepted_step_budget": 100000,
}


def d(value) -> Decimal:
    """Exact decimal statement of frozen JSON input."""
    return Decimal(str(value))


def binary(value) -> Decimal:
    """Exact binary64 input, not its shortened display decimal."""
    return Decimal.from_float(float(value))


def emit(record: dict) -> None:
    def json_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"unsupported evidence value: {type(value).__name__}")

    path = os.environ.get("STORAGE_CASE_LOG")
    if path:
        with Path(path).open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False, separators=(",", ":"),
                                    default=json_scalar) + "\n")


def conclude(request, record: dict, checks: dict[str, bool]) -> None:
    record.update(test=request.node.nodeid, checks={k: bool(v) for k, v in checks.items()},
                  passed=all(checks.values()))
    emit(record)
    assert all(checks.values()), {k: v for k, v in record["checks"].items() if not v}


def comparison(values, references, atol, rtol) -> dict:
    values = np.asarray(values).ravel()
    references = list(references)
    assert len(values) == len(references)
    with localcontext() as ctx:
        ctx.prec = 80
        errors = [abs(binary(x) - ref) for x, ref in zip(values, references, strict=True)]
        budgets = [d(atol) + d(rtol) * abs(ref) for ref in references]
    return {"reference_decimal80": [str(x) for x in references],
            "absolute_errors": [float(x) for x in errors],
            "budgets": [float(x) for x in budgets],
            "passed": all(error <= budget for error, budget in zip(errors, budgets, strict=True))}


def decimal_linear_step(case: dict, initial: list[Decimal], h: Decimal, precision=80
                        ) -> list[Decimal]:
    """Independent partial-pivot Decimal elimination of (M-hL)x=M*xold."""
    with localcontext() as ctx:
        ctx.prec = precision
        n = len(initial)
        volumes = [d(v) for v in case["volumes"]]
        matrix = [[Decimal(0) for _ in range(n)] for _ in range(n)]
        rhs = [volumes[i] * initial[i] for i in range(n)]
        for i, volume in enumerate(volumes):
            matrix[i][i] = volume
        for (i, j), k0 in zip(case["faces"], case["conductance"], strict=True):
            k = h * d(k0)
            matrix[i][i] += k
            matrix[j][j] += k
            matrix[i][j] -= k
            matrix[j][i] -= k
        for col in range(n):
            pivot = max(range(col, n), key=lambda row: abs(matrix[row][col]))
            assert matrix[pivot][col] != 0
            matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
            rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
            for row in range(col + 1, n):
                factor = matrix[row][col] / matrix[col][col]
                for j in range(col, n):
                    matrix[row][j] -= factor * matrix[col][j]
                rhs[row] -= factor * rhs[col]
        result = [Decimal(0)] * n
        for row in range(n - 1, -1, -1):
            result[row] = (rhs[row] - sum(matrix[row][j] * result[j]
                                         for j in range(row + 1, n))) / matrix[row][row]
        return result


def inventory_record(delta, volumes, domains, initial, cumulative, gate) -> tuple[dict, bool]:
    defects, scales, limits = [], [], []
    for domain in domains:
        defects.append(abs(fsum(float(volumes[i]) * float(delta[i]) for i in domain)))
        scale = fsum(float(volumes[i]) * float(initial[i]) for i in domain)
        scales.append(scale if scale != 0 else 1.0)
        limits.append(gate["inventory_relative_limit"] * scale if scale != 0
                      else gate["zero_inventory_absolute_limit"])
    for i, defect in enumerate(defects):
        cumulative[i] += defect
    cumulative_limits = [gate["cumulative_absolute_inventory_relative_limit"] * scale
                         if any(initial[i] != 0 for i in domain)
                         else gate["zero_inventory_absolute_limit"]
                         for scale, domain in zip(scales, domains, strict=True)]
    passed = all(x <= lim for x, lim in zip(defects, limits, strict=True)) and all(
        x <= lim for x, lim in zip(cumulative, cumulative_limits, strict=True))
    return {"absolute_interval_defect": defects, "cumulative_absolute_defect": cumulative.copy(),
            "fixed_scales": scales, "interval_limits": limits,
            "cumulative_limits": cumulative_limits}, passed


BE_CASES = [case for case in GATES["NC01"]["fixed_inputs"] if case["id"].endswith("_be")]


@pytest.mark.parametrize("case", BE_CASES, ids=lambda case: case["id"])
def test_nc01_conservative_be(case, request):
    gate = GATES["NC01"]
    probe = diffusion_probe(case["volumes"], case["faces"], case["conductance"])
    point = probe.coordinates.point(case["initial"])
    ref80 = [d(x) for x in case["initial"]]
    ref100 = ref80.copy()
    cumulative = [0.0] * len(case["domains"])
    records, checks = [], {}
    for index, h in enumerate(case["step_sizes"]):
        step = conservative_be_step(probe.system, probe.coordinates, point, h,
                                    feasible=lambda p: bool(np.all(p.y >= 0)))
        ref80 = decimal_linear_step(case, ref80, d(h), 80)
        ref100 = decimal_linear_step(case, ref100, d(h), 100)
        reference_error = max(abs(a - b) for a, b in zip(ref80, ref100, strict=True))
        state = comparison(step.right.y, ref80, gate["state_atol"], gate["state_rtol"])
        delta = step.increment.field("density").values
        inventory, inventory_pass = inventory_record(delta, case["volumes"], case["domains"],
                                                      case["initial"], cumulative, gate)
        records.append({"time": step.right.time, "h": h, "state": step.right.y.tolist(),
                        "physical_increment": delta.tolist(), "state_check": state,
                        "inventory": inventory, "residual": step.residual.tolist(),
                        "reference_80_vs_100_error": str(reference_error),
                        "newton_iterations": step.newton_iterations})
        checks[f"state_{index}"] = state["passed"]
        checks[f"inventory_{index}"] = inventory_pass
        checks[f"reference_{index}"] = reference_error <= d(gate["reference_error_bound"]) / 3
        point = step.right
    if case["id"] == "zero_be":
        checks["exact_zero_identity"] = np.array_equal(point.y.view(np.uint64), np.zeros_like(point.y).view(np.uint64))
    conclude(request, {"gate": "NC01", "case": case["id"], "route": "physical_conservative_be",
                       "steps": records, "all_intervals_recorded": True}, checks)


def test_nc01_ordinary_log_be_counterexample(request):
    """One wrong-coordinate BE equation is an explicit rejected-route control."""
    from scipy.optimize import root
    case = BE_CASES[0]
    initial = np.asarray(case["initial"], float)
    h = case["step_sizes"][0]

    def wrong_equation(q):
        n = np.exp(q)
        rate = np.array([n[1] - n[0], n[0] - n[1]])
        return (q - np.log(initial)) / h - rate / n

    result = root(wrong_equation, np.log(initial), options=SUPPLEMENTAL_INPUTS["ordinary_log_be"])
    n = np.exp(result.x)
    defect = abs(fsum(n) - fsum(initial))
    conclude(request, {"gate": "NC01", "case": "ordinary_transformed_coordinate_BE_counterexample",
                       "candidate": n.tolist(), "residual": wrong_equation(result.x).tolist(),
                       "inventory_defect": defect, "solver_calls": result.nfev,
                       "route_accepted": False},
             {"algebraic_solve_completed": bool(result.success), "positive": bool(np.all(n > 0)),
              "nonconservation_exposed": defect > GATES["NC01"]["inventory_relative_limit"] * fsum(initial)})


def decimal_trap_reference(case, precision=80):
    with localcontext() as ctx:
        ctx.prec = precision
        n0, f0, Nt, k, e, h = (d(case[key]) for key in ("n", "f", "Nt", "k", "e", "h"))
        S = n0 + Nt * f0
        lo, hi = Decimal(0), min(Decimal(1), S / Nt)
        for _ in range(300):
            midpoint = (lo + hi) / 2
            r = k * (S - Nt * midpoint) * (1 - midpoint) - e * Nt * midpoint
            residual = midpoint - f0 - h / Nt * r
            if residual > 0:
                hi = midpoint
            else:
                lo = midpoint
        f = (lo + hi) / 2
        return [S - Nt * f, f], hi - lo


@pytest.mark.parametrize("case", GATES["NC02"]["fixed_inputs"], ids=lambda case: case["id"])
def test_nc02_trap_exchange(case, request):
    gate = GATES["NC02"]
    coordinates, system = trap_exchange_probe(case["Nt"], case["k"], case["e"])
    point = coordinates.point([case["n"], case["f"]])
    feasible = lambda p: bool(p.y[0] >= 0 and 0 <= p.y[1] <= 1)
    step = conservative_be_step(system, coordinates, point, case["h"], feasible=feasible)
    ref, bracket = decimal_trap_reference(case)
    ref100, _ = decimal_trap_reference(case, 100)
    reference_error = max(abs(a - b) for a, b in zip(ref, ref100, strict=True)) + bracket
    state = comparison(step.right.y, ref, gate["state_atol"], gate["state_rtol"])
    delta = step.right.y - point.y
    inventory, inv_pass = inventory_record(delta, [1, case["Nt"]], [[0, 1]], point.y, [0.0], gate)
    kind = case["required_direction"]
    if kind.startswith("f increases") or kind == "capture":
        direction = delta[1] > 0 and delta[0] < 0
    elif kind.startswith("f decreases") or kind == "emission":
        direction = delta[1] < 0 and delta[0] > 0
    else:
        direction = np.array_equal(step.right.y.view(np.uint64), point.y.view(np.uint64))
    conclude(request, {"gate": "NC02", "case": case["id"], "state": step.right.y.tolist(),
                       "state_check": state, "inventory": inventory, "delta": delta.tolist(),
                       "rate_before": system.terms(point).rate.tolist(),
                       "rate_after": system.terms(step.right).rate.tolist(),
                       "required_direction": kind, "feasible": feasible(step.right),
                       "reference_root_iterations": 300, "reference_error": str(reference_error),
                       "BE_residual": step.residual.tolist()},
             {"state": state["passed"], "inventory": inv_pass, "direction_or_exact_identity": direction,
              "feasible": feasible(step.right), "reference": reference_error <= d(gate["reference_error_bound"]) / 3})


IC = GATES["NC03"]["fixed_inputs"]


@pytest.mark.parametrize("voltage,slope", list(product([IC["V0"], IC["V1"]], IC["slopes"])))
def test_nc03_physical_ic_and_history(voltage, slope, request):
    gate = GATES["NC03"]
    model = CarrierPotentialStorage()
    with localcontext() as ctx:
        ctx.prec = 80
        logn = d(IC["n0"]).ln()
        phi = d(voltage) - d(IC["n0"])
        q = logn - phi
    left = model.point([float(logn + d(IC["n0"])), -IC["n0"]], inputs=[IC["V0"]])
    history = (left,)
    seed = model.point(left.y, inputs=[voltage])
    result = consistent_initial_state(model.system(), model, seed, [IC["n0"]], [slope], history=history)
    state = comparison([model.value(result.point)[0], *result.point.y], [d(IC["n0"]), q, phi],
                       gate["physical_state_atol"], gate["physical_state_rtol"])
    tangent = comparison(result.derivative, [-d(slope), d(slope)], gate["tangent_absolute_limit"], 0)
    first, terms = model.system().evaluate(result.point)
    matrix = np.vstack((first.y, terms.gy))
    conclude(request, {"gate": "NC03", "case": "physical_ic", "voltage": voltage, "slope": slope,
                       "physical_n": model.value(result.point).tolist(), "coordinates": result.point.y.tolist(),
                       "derivative": result.derivative.tolist(), "state_check": state,
                       "tangent_check": tangent, "residual": result.residual.tolist(),
                       "constraint_tangent": result.tangent.tolist(), "rank": result.rank.rank,
                       "matrix": matrix.tolist(), "determinant": float(np.linalg.det(matrix)),
                       "history_identity": [p.identity for p in history], "iterations": result.iterations},
             {"state": state["passed"], "tangent": tangent["passed"],
              "full_residual": np.max(np.abs(result.residual)) <= gate["residual_absolute_limit"],
              "constraint_tangent": np.max(np.abs(result.tangent)) <= gate["tangent_absolute_limit"],
              "rank": result.rank.rank == 2, "history_objects_preserved": result.history is history,
              "history_bytes_preserved": result.history[0].identity == left.identity})


def test_nc03_holding_coordinate_is_wrong(request):
    from scipy.optimize import brentq
    model = CarrierPotentialStorage()
    qfixed = float(np.log(IC["n0"]) + IC["n0"])
    phi = brentq(lambda phi: phi + np.exp(qfixed + phi) - IC["V1"], -3, 0, xtol=1e-14)
    wrong = model.point([qfixed, phi], inputs=[IC["V1"]])
    change = model.value(wrong)[0] - IC["n0"]
    conclude(request, {"gate": "NC03", "case": "holding_q_counterexample",
                       "physical_density_change": change, "coordinates": wrong.y.tolist()},
             {"wrong_projection_exposed": abs(change) > GATES["NC03"]["physical_state_atol"]})


@pytest.mark.parametrize("name", ["dependent_constraint_matrix", "floating_poisson_matrix", "incompatible_capacitor"])
def test_nc03_rank_rejection_reasons(name, request):
    if name == "incompatible_capacitor":
        case = IC[name]
        matrix, rhs = [[case["C"], 0], [1, 0]], [case["fixed_charge"], case["imposed_voltage"]]
    else:
        matrix, rhs = IC[name], [0, 0]
    with pytest.raises(ContractError) as caught:
        rank_aware_solve(matrix, rhs, gauge_nullspace=name == "floating_poisson_matrix")
    evidence = rank_evidence(matrix)
    expected = GATES["NC03"]["negative_case_results"][name]
    conclude(request, {"gate": "NC03", "case": name, "reason": caught.value.reason,
                       "expected_reason": expected, "rank": evidence.rank,
                       "singular_values": evidence.singular_values.tolist(), "threshold": evidence.threshold},
             {"exact_reason": caught.value.reason == expected})


def test_nc03_nonzero_input_time_chain(request):
    case = SUPPLEMENTAL_INPUTS["full_ic_chain"]
    model = CarrierPotentialStorage(case["alpha"], case["beta"], case["gamma"])
    original = model.point([np.log(case["n"]) + case["n"], -case["n"]], inputs=[0.0])
    seed = model.point(original.y, case["time"], [case["a"]])
    history = (original,)
    result = consistent_initial_state(model.system(), model, seed, [case["n"]], [case["adot"]], history=history)
    with localcontext() as ctx:
        ctx.prec = 80
        phi = d(case["a"]) + d(case["gamma"]) * d(case["time"]) - d(case["n"])
        q = d(case["n"]).ln() - phi - d(case["alpha"]) * d(case["a"]) - d(case["beta"]) * d(case["time"])
        phidot = d(case["adot"]) + d(case["gamma"])
        qdot = -phidot - d(case["alpha"]) * d(case["adot"]) - d(case["beta"])
    first, terms = model.system().evaluate(result.point)
    gate = GATES["NC03"]
    state = comparison(result.point.y, [q, phi], gate["physical_state_atol"], gate["physical_state_rtol"])
    tangent = comparison(result.derivative, [qdot, phidot], gate["tangent_absolute_limit"], 0)
    conclude(request, {"gate": "NC03", "case": "supplemental_Qa_Qt_ga_gt", "state_check": state,
                       "tangent_check": tangent, "Qa": first.inputs.tolist(), "Qt": first.time.tolist(),
                       "ga": terms.ga.tolist(), "gt": terms.gt.tolist(), "storage_error": result.storage_error.tolist(),
                       "residual": result.residual.tolist(), "tangent": result.tangent.tolist()},
             {"state": state["passed"], "tangent": tangent["passed"], "history": result.history is history,
              "storage": np.max(np.abs(result.storage_error)) <= gate["physical_state_atol"],
              "residual": np.max(np.abs(result.residual)) <= gate["residual_absolute_limit"],
              "all_four_nonzero": all(np.any(a != 0) for a in (first.inputs, first.time, terms.ga, terms.gt))})


def shared_ic_fixture():
    model = SharedCapacityStorage()
    case = GATES["NC05"]["fixed_inputs"]["capacity_change"]
    point = model.from_occupancies(case["theta"], inputs=[case["left"]])
    system = ImplicitSystem(model, lambda p: no_constraints(np.zeros(2), np.zeros((2, 2)), 1))
    return model, system, point


@pytest.mark.parametrize("shape", [(), (1,), (1, 2), (2, 1)])
def test_nc03_ic_rejects_broadcast_preserved_storage(shape, request):
    model, system, seed = shared_ic_fixture()
    invalid_target = np.full(shape, model.value(seed)[0])
    with pytest.raises(ContractError) as caught:
        consistent_initial_state(system, model, seed, invalid_target, [0.0])
    conclude(request, {"gate": "NC03", "case": "preserved_storage_shape_guard",
                       "supplied_shape": list(shape), "required_shape": [2],
                       "reason": caught.value.reason},
             {"exact_shape_required": caught.value.reason == "preserved_storage_shape_mismatch"})


@pytest.mark.parametrize("shape", [(), (0,), (2,), (1, 1)])
def test_nc03_ic_rejects_mismatched_input_rate(shape, request):
    model, system, seed = shared_ic_fixture()
    invalid_rate = np.zeros(shape)
    with pytest.raises(ContractError) as caught:
        consistent_initial_state(system, model, seed, model.value(seed), invalid_rate)
    conclude(request, {"gate": "NC03", "case": "input_rate_shape_guard",
                       "supplied_shape": list(shape), "required_shape": list(seed.inputs.shape),
                       "reason": caught.value.reason},
             {"exact_shape_required": caught.value.reason == "input_rate_shape_mismatch"})


@pytest.mark.parametrize("field", ["preserved_storage", "input_rate"])
@pytest.mark.parametrize("invalid", ["nan", "inf"])
def test_nc03_ic_rejects_nonfinite_fixed_data(field, invalid, request):
    model, system, seed = shared_ic_fixture()
    target, rate = model.value(seed).copy(), np.zeros_like(seed.inputs)
    value = float(invalid)
    if field == "preserved_storage":
        target[0] = value
    else:
        rate[0] = value
    with pytest.raises(ContractError) as caught:
        consistent_initial_state(system, model, seed, target, rate)
    conclude(request, {"gate": "NC03", "case": "finite_fixed_data_guard", "field": field,
                       "invalid_value_class": invalid, "reason": caught.value.reason},
             {"nonfinite_rejected": caught.value.reason == "nonfinite_array"})


def exponential_reference(y, a, t, ydot, adot, precision=80):
    with localcontext() as ctx:
        ctx.prec = precision
        Q = (2 * y + 3 * a + 5 * t).exp()
        F = Q * (2 * ydot + 3 * adot + 5)
        return {"Q": Q, "F": F, "Fy": 2 * F, "Fa": 3 * F, "Ft": 5 * F,
                "Fydot": 2 * Q, "Fadot": 3 * Q}


NC04 = GATES["NC04"]
POINT_RATE = list(product(NC04["fixed_inputs"]["points"], NC04["fixed_inputs"]["rates"]))


@pytest.mark.parametrize("point,rate", POINT_RATE)
def test_nc04_complete_analytic_partials(point, rate, request):
    model = ExponentialStorage()
    p = model.point([point["y"]], point["t"], [point["a"]])
    ydot, adot = np.array([rate["ydot"]]), np.array([rate["adot"]])
    system = model.system()
    linear = system.linearize(p, ydot, adot)
    reference_inputs = tuple(binary(x) for x in (p.y[0], p.inputs[0], p.time, ydot[0], adot[0]))
    ref = exponential_reference(*reference_inputs)
    ref100 = exponential_reference(*reference_inputs, precision=100)
    values = {"F": system.residual(p, ydot, adot)[0], "Fy": linear.y[0, 0],
              "Fydot": linear.ydot[0, 0], "Fa": linear.inputs[0, 0],
              "Fadot": linear.input_rate[0, 0], "Ft": linear.time[0]}
    checks = {key: comparison([value], [ref[key]], NC04["analytic_atol"], NC04["analytic_rtol"])
              for key, value in values.items()}
    ref_error = max(abs(ref[key] - ref100[key]) for key in ref)
    matrices = []
    for cj in NC04["fixed_inputs"]["cj"]:
        with localcontext() as ctx:
            ctx.prec = 80
            expected = ref["Fy"] + d(cj) * ref["Fydot"]
        check = comparison(linear.ida_matrix(cj), [expected], NC04["analytic_atol"], NC04["analytic_rtol"])
        matrices.append({"cj": cj, "matrix": linear.ida_matrix(cj).tolist(), "check": check})
    conclude(request, {"gate": "NC04", "case": "analytic_partials", "point": point, "rate": rate,
                       "values": values, "comparisons": checks, "IDA_matrices": matrices,
                       "reference_80_vs_100_error": str(ref_error)},
             {**{key: value["passed"] for key, value in checks.items()},
              "IDA_matrices": all(item["check"]["passed"] for item in matrices),
              "reference": ref_error <= d(NC04["reference_error_bound"]) / 3})


@pytest.mark.parametrize("point,rate,cj,direction", [(*pair, cj, direction) for pair in POINT_RATE
                          for cj in NC04["fixed_inputs"]["cj"] for direction in NC04["fixed_inputs"]["directions"]])
def test_nc04_joint_directions(point, rate, cj, direction, request):
    model = ExponentialStorage()
    system = model.system()
    p = model.point([point["y"]], point["t"], [point["a"]])
    ydot, adot = np.array([rate["ydot"]]), np.array([rate["adot"]])
    vec = np.asarray(direction, float) / np.linalg.norm(direction)
    dy, da, dt, dadot = vec
    linear = system.linearize(p, ydot, adot)
    candidate = float(linear.ida_matrix(cj)[0, 0] * dy + linear.inputs[0, 0] * da
                      + linear.time[0] * dt + linear.input_rate[0, 0] * dadot)
    args = tuple(binary(x) for x in (p.y[0], p.inputs[0], p.time, ydot[0], adot[0]))
    with localcontext() as ctx:
        ctx.prec = 80
        ref = exponential_reference(*args)
        expected = (ref["Fy"] * binary(dy) + ref["Fa"] * binary(da) + ref["Ft"] * binary(dt)
                    + ref["Fydot"] * binary(cj) * binary(dy) + ref["Fadot"] * binary(dadot))
    analytic = comparison([candidate], [expected], NC04["analytic_atol"], NC04["analytic_rtol"])
    errors = []
    for eps in NC04["joint_direction_steps"]:
        plus = model.point(p.y + eps * dy, p.time + eps * dt, p.inputs + eps * da)
        minus = model.point(p.y - eps * dy, p.time - eps * dt, p.inputs - eps * da)
        fd = (system.residual(plus, ydot + eps * cj * dy, adot + eps * dadot)[0]
              - system.residual(minus, ydot - eps * cj * dy, adot - eps * dadot)[0]) / (2 * eps)
        with localcontext() as ctx:
            ctx.prec = 80
            delta = [binary(eps) * binary(x) for x in (dy, da, dt, cj * dy, dadot)]
            rp = exponential_reference(*(x + dx for x, dx in zip(args, delta, strict=True)))["F"]
            rm = exponential_reference(*(x - dx for x, dx in zip(args, delta, strict=True)))["F"]
            decimal_fd = (rp - rm) / (2 * binary(eps))
            denominator = max(Decimal(1), abs(expected))
            relative_error = abs(binary(fd) - expected) / denominator
            reference_truncation = abs(decimal_fd - expected) / denominator
        errors.append({"eps": eps, "finite_difference": float(fd),
                       "normalized_error": float(relative_error),
                       "independent_decimal80_stencil": str(decimal_fd),
                       "reference_truncation": float(reference_truncation),
                       "passed": relative_error <= d(NC04["finite_difference_relative_limit"])})
    adjacent = any(errors[i]["passed"] and errors[i + 1]["passed"] for i in range(len(errors) - 1))
    conclude(request, {"gate": "NC04", "case": "joint_direction", "point": point, "rate": rate,
                       "cj": cj, "direction": direction, "normalized_direction": vec.tolist(),
                       "analytic_jvp": candidate, "analytic_check": analytic, "eps_errors": errors},
             {"analytic": analytic["passed"], "adjacent_plateau": adjacent, "all_eps_recorded": len(errors) == 4})


def test_nc04_finite_storage_and_be_jacobian(request):
    model = ExponentialStorage()
    system = model.system()
    case = NC04["fixed_inputs"]["storage_delta"]
    l, inc = case["left"], case["increment"]
    left = model.point([l["y"]], l["t"], [l["a"]])
    right, increment = model.advance(left, [inc["dy"]], left.time + inc["dt"], left.inputs + inc["da"])
    delta = model.delta(left, right, increment)
    with localcontext() as ctx:
        ctx.prec = 80
        ql = (2 * binary(left.y[0]) + 3 * binary(left.inputs[0]) + 5 * binary(left.time)).exp()
        qr = (2 * binary(right.y[0]) + 3 * binary(right.inputs[0]) + 5 * binary(right.time)).exp()
        expected = qr - ql
    finite = comparison(delta, [expected], NC04["analytic_atol"], NC04["analytic_rtol"])
    jac = system.conservative_jacobian(left, right)[0, 0]
    jaccheck = comparison([jac], [2 * qr], NC04["analytic_atol"], NC04["analytic_rtol"])
    tangent_substitute = model.first(left).y[0, 0] * inc["dy"]
    errors = []
    for eps in NC04["joint_direction_steps"]:
        rp, ip = model.advance(left, right.y - left.y + eps, right.time, right.inputs)
        rm, im = model.advance(left, right.y - left.y - eps, right.time, right.inputs)
        fd = (system.conservative_residual(left, rp, ip)[0] - system.conservative_residual(left, rm, im)[0]) / (2 * eps)
        err = abs(fd - jac) / max(1.0, abs(jac))
        errors.append({"eps": eps, "normalized_error": err, "passed": err <= NC04["finite_difference_relative_limit"]})
    conclude(request, {"gate": "NC04", "case": "finite_storage_and_fixed_left_BE",
                       "left_identity": left.identity, "right_identity": right.identity,
                       "delta": delta.tolist(), "delta_check": finite, "BE_jacobian": jac,
                       "jacobian_check": jaccheck, "eps_errors": errors,
                       "incorrect_Qy_dy_only": tangent_substitute},
             {"finite_delta": finite["passed"], "BE_jacobian": jaccheck["passed"],
              "input_time_not_omitted": abs(delta[0] - tangent_substitute) > NC04["analytic_atol"],
              "BE_plateau": any(errors[i]["passed"] and errors[i + 1]["passed"] for i in range(3))})


@pytest.mark.parametrize("change_input", [False, True])
def test_nc04_nonlinear_be_retains_or_updates_inputs(change_input, request):
    """Exercise the actual BE adapter with a nonempty input vector and Qt."""
    model = ExponentialStorage()
    c = NC04["fixed_inputs"]["storage_delta"]
    left = model.point([c["left"]["y"]], c["left"]["t"], [c["left"]["a"]])
    inputs = left.inputs + c["increment"]["da"] if change_input else None
    step = conservative_be_step(model.system(), model, left, c["increment"]["dt"], inputs=inputs)
    with localcontext() as ctx:
        ctx.prec = 80
        expected_y = binary(left.y[0]) - (
            3 * (binary(step.right.inputs[0]) - binary(left.inputs[0]))
            + 5 * (binary(step.right.time) - binary(left.time))) / 2
    state = comparison(step.right.y, [expected_y], NC04["analytic_atol"], NC04["analytic_rtol"])
    delta = model.delta(left, step.right, step.increment)
    conclude(request, {"gate": "NC04", "case": "nonlinear_BE_input_carry", "change_input": change_input,
                       "inputs_left": left.inputs.tolist(), "inputs_right": step.right.inputs.tolist(),
                       "coordinates": step.right.y.tolist(), "state_check": state,
                       "finite_storage_delta": delta.tolist(), "BE_residual": step.residual.tolist(),
                       "newton_iterations": step.newton_iterations},
             {"coordinates": state["passed"], "finite_Q_preserved": np.max(np.abs(delta)) <= NC04["analytic_atol"],
              "left_history_untouched": step.left is left,
              "input_contract": np.array_equal(step.right.inputs, left.inputs if inputs is None else inputs)})


NC05 = GATES["NC05"]


def shared_decimal_matrix(theta, C=Decimal(1)):
    with localcontext() as ctx:
        ctx.prec = 80
        return [[C * ((theta[i] if i == j else Decimal(0)) - theta[i] * theta[j])
                 for j in range(len(theta))] for i in range(len(theta))]


@pytest.mark.parametrize("theta", NC05["fixed_inputs"]["occupancies"])
def test_nc05_cross_derivatives_and_rank(theta, request):
    model = SharedCapacityStorage()
    p = model.from_occupancies(theta, inputs=[NC05["fixed_inputs"]["capacity"]])
    first = model.first(p)
    ref = shared_decimal_matrix([d(x) for x in theta], d(NC05["fixed_inputs"]["capacity"]))
    check = comparison(first.y, [x for row in ref for x in row], NC05["atol"], NC05["rtol"])
    evidence = rank_evidence(first.y)
    reference_matrix = np.array([[float(x) for x in row] for row in ref])
    matrix_inf_error = float(np.linalg.norm(first.y - reference_matrix, ord=np.inf))
    matrix_budget = NC05["atol"] + NC05["rtol"] * float(np.linalg.norm(reference_matrix, ord=np.inf))
    conclude(request, {"gate": "NC05", "case": "cross_derivatives", "theta": theta,
                       "physical_population": model.value(p).tolist(), "matrix": first.y.tolist(),
                       "component_check": check, "matrix_inf_error": matrix_inf_error,
                       "matrix_budget": matrix_budget, "active_rank": evidence.rank,
                       "singular_values": evidence.singular_values.tolist(), "rank_threshold": evidence.threshold,
                       "vacancy": 1 - fsum(theta)},
             {"derivatives": check["passed"], "matrix_norm": matrix_inf_error <= matrix_budget,
              "off_diagonals_present": first.y[0, 1] < 0 and first.y[1, 0] < 0,
              "full_active_rank": evidence.rank == 2})


def test_nc05_exact_inactive_and_joint_feasibility(request):
    model = SharedCapacityStorage(active=(False, False))
    theta = NC05["fixed_inputs"]["zero_case"]["occupancies"]
    left = model.from_occupancies(theta)
    right, increment = model.advance(left, [], 1.0, [2.0])
    values = [model.value(left), model.value(right), model.delta(left, right, increment),
              model.first(right).y, model.first(right).inputs, model.first(right).time,
              model.rate_jvp(right, np.empty(0), np.array([0.3]), np.empty(0), np.array([0.2]), 0.5)]
    inactive_rank = rank_evidence(model.first(right).y)
    partial = SharedCapacityStorage(active=(False, True))
    pp = partial.from_occupancies(SUPPLEMENTAL_INPUTS["partial_inactive"])
    partial_rank = rank_evidence(partial.first(pp).y)
    with pytest.raises(ContractError) as invalid:
        SharedCapacityStorage().from_occupancies(NC05["fixed_inputs"]["invalid_occupancy"])
    with pytest.raises(ContractError) as boundary:
        SharedCapacityStorage().from_occupancies(SUPPLEMENTAL_INPUTS["saturated"])
    with pytest.raises(ContractError) as wrong_inactive:
        partial.from_occupancies([0.1, 0.3])
    boundary_matrix = np.array([[float(x) for x in row]
                                for row in shared_decimal_matrix([d(0.5), d(0.5)])])
    boundary_rank = rank_evidence(boundary_matrix)
    conclude(request, {"gate": "NC05", "case": "inactive_and_joint_feasibility",
                       "zero_coordinates": left.y.tolist(), "zero_populations": model.value(left).tolist(),
                       "inactive_rank": inactive_rank.rank, "partial_rank": partial_rank.rank,
                       "saturated_matrix": boundary_matrix.tolist(), "saturated_rank": boundary_rank.rank,
                       "invalid_positive_components_reason": invalid.value.reason,
                       "boundary_reason": boundary.value.reason, "inactive_mismatch_reason": wrong_inactive.value.reason},
             {"strict_zero_every_storage_path": all(np.array_equal(x, np.zeros_like(x)) for x in values),
              "no_finite_log_coordinate": left.y.size == 0, "inactive_rank": inactive_rank.rank == 0,
              "partial_active_rank": partial_rank.rank == 1, "partial_zero": model.value(left)[0] == 0 and partial.value(pp)[0] == 0,
              "saturated_rank_loss": boundary_rank.rank == 1,
              "positive_signs_not_sufficient": invalid.value.reason == "joint_capacity_infeasible",
              "singular_init_rejected": boundary.value.reason == "unsupported_singular_capacity_boundary",
              "nonzero_inactive_rejected": wrong_inactive.value.reason == "inactive_identity_mismatch"})


@pytest.mark.parametrize("theta", NC05["fixed_inputs"]["occupancies"])
def test_nc05_full_capacity_hessian_contraction(theta, request):
    c = SUPPLEMENTAL_INPUTS["shared_contraction"]
    model = SharedCapacityStorage(capacity_rate=c["capacity_rate"])
    p = model.from_occupancies(theta, c["time"], [c["input"]])
    ydot, adot, dy, da = np.array(c["ydot"]), np.array([c["adot"]]), np.array(c["dy"]), np.array([c["da"]])
    candidate = model.rate_jvp(p, ydot, adot, dy, da, c["dt"])
    with localcontext() as ctx:
        ctx.prec = 80
        th = [d(x) for x in theta]
        J = shared_decimal_matrix(th)
        dth = [sum(J[i][j] * d(c["dy"][j]) for j in range(2)) for i in range(2)]
        dJ = [[(dth[i] if i == j else Decimal(0)) - dth[i] * th[j] - th[i] * dth[j]
               for j in range(2)] for i in range(2)]
        C = d(c["input"]) + d(c["capacity_rate"]) * d(c["time"])
        dC = d(c["da"]) + d(c["capacity_rate"]) * d(c["dt"])
        Cdot = d(c["adot"]) + d(c["capacity_rate"])
        expected = [dC * sum(J[i][j] * d(c["ydot"][j]) for j in range(2))
                    + C * sum(dJ[i][j] * d(c["ydot"][j]) for j in range(2)) + Cdot * dth[i]
                    for i in range(2)]
    check = comparison(candidate, expected, NC05["atol"], NC05["rtol"])
    system = ImplicitSystem(model, lambda p: no_constraints(np.zeros(2), np.zeros((2, 2)), 1))
    linear = system.linearize(p, ydot, adot)
    assembled = linear.y @ dy + linear.inputs @ da + linear.time * c["dt"]
    assembled_check = comparison(assembled, expected, NC05["atol"], NC05["rtol"])
    errors = []
    for eps in NC04["joint_direction_steps"]:
        plus = model.point(p.y + eps * dy, p.time + eps * c["dt"], p.inputs + eps * da)
        minus = model.point(p.y - eps * dy, p.time - eps * c["dt"], p.inputs - eps * da)
        fd = (system.residual(plus, ydot, adot) - system.residual(minus, ydot, adot)) / (2 * eps)
        err = float(np.max(np.abs(fd - candidate) / np.maximum(1, np.abs(candidate))))
        errors.append({"eps": eps, "normalized_error": err, "passed": err <= NC04["finite_difference_relative_limit"]})
    conclude(request, {"gate": "NC05", "case": "full_capacity_contraction", "theta": theta,
                       "contraction": candidate.tolist(), "check": check, "assembled_check": assembled_check,
                       "Fy": linear.y.tolist(), "Fydot": linear.ydot.tolist(),
                       "Fa": linear.inputs.tolist(), "Fadot": linear.input_rate.tolist(),
                       "Ft": linear.time.tolist(), "supplemental_fd_gate": "NC04", "eps_errors": errors},
             {"independent_contraction": check["passed"], "public_assembly": assembled_check["passed"],
              "adjacent_plateau": any(errors[i]["passed"] and errors[i + 1]["passed"] for i in range(3))})


def test_nc05_capacity_change_and_fixed_populations(request):
    case = NC05["fixed_inputs"]["capacity_change"]
    model = SharedCapacityStorage()
    left = model.from_occupancies(case["theta"], inputs=[case["left"]])
    right, increment = model.advance(left, [0, 0], 1, [case["right"]])
    delta = model.delta(left, right, increment)
    delta_check = comparison(delta, [d(x) for x in case["fixed_coordinate_delta_Q"]], NC05["atol"], NC05["rtol"])
    system = ImplicitSystem(model, lambda p: no_constraints(np.zeros(2), np.zeros((2, 2)), 1))
    history = (left,)
    ic = consistent_initial_state(system, model, right, model.value(left), [0.0], history=history)
    theta_check = comparison(model.theta(ic.point), [d(x) for x in case["fixed_population_new_theta"]], NC05["atol"], NC05["rtol"])
    conclude(request, {"gate": "NC05", "case": "capacity_change", "fixed_coordinate_delta": delta.tolist(),
                       "delta_check": delta_check, "fixed_population_theta": model.theta(ic.point).tolist(),
                       "theta_check": theta_check, "storage_error": ic.storage_error.tolist(),
                       "history_identity": left.identity, "rank": ic.rank.rank},
             {"finite_capacity_delta": delta_check["passed"], "fixed_population": theta_check["passed"],
              "history": ic.history is history, "storage_error": np.max(np.abs(ic.storage_error)) <= NC05["atol"]})


def test_nc05_combined_finite_state_input_time_change(request):
    c = SUPPLEMENTAL_INPUTS["shared_finite"]
    model = SharedCapacityStorage(capacity_rate=c["capacity_rate"])
    left = model.from_occupancies(c["theta"], inputs=[1.0])
    right, inc = model.advance(left, c["dy"], c["dt"], [1 + c["da"]])
    with localcontext() as ctx:
        ctx.prec = 80
        def populations(point):
            weights = [binary(x).exp() for x in point.y]
            denominator = 1 + sum(weights)
            C = binary(point.inputs[0]) + binary(c["capacity_rate"]) * binary(point.time)
            return [C * w / denominator for w in weights]
        qr, ql = populations(right), populations(left)
        expected = [a - b for a, b in zip(qr, ql, strict=True)]
    delta = model.delta(left, right, inc)
    check = comparison(delta, expected, NC05["atol"], NC05["rtol"])
    conclude(request, {"gate": "NC05", "case": "combined_finite_change", "delta": delta.tolist(),
                       "check": check, "capacity_left": model.capacity(left), "capacity_right": model.capacity(right)},
             {"independent_finite_delta": check["passed"]})


IDA_CASE = next(case for case in GATES["NC01"]["fixed_inputs"] if case["id"] == "equal_ida")
IDA_SETTINGS = list(product(IDA_CASE["max_steps"], IDA_CASE["max_orders"], IDA_CASE["solver_rtols"]))


@pytest.mark.parametrize("max_step,max_order,rtol", IDA_SETTINGS)
def test_nc01_ida_all_native_intervals(max_step, max_order, rtol, request):
    pytest.importorskip("sksundae")
    gate = GATES["NC01"]
    tg = next(row for row in IDA_CASE["trajectory_gates"] if row["solver_rtol"] == rtol)
    probe = diffusion_probe(IDA_CASE["volumes"], IDA_CASE["faces"], IDA_CASE["conductance"])
    start = time.monotonic()
    result = ida_diffusion_probe(probe, IDA_CASE["initial"], IDA_CASE["end_time"],
                                 max_step=max_step, max_order=max_order, rtol=rtol,
                                 atol=IDA_CASE["solver_atol"],
                                 accepted_step_budget=SUPPLEMENTAL_INPUTS["ida_accepted_step_budget"])
    elapsed = time.monotonic() - start
    records, cumulative = [], [0.0]
    state_pass, inventory_pass = True, True
    max_state_budget_ratio, max_residual = 0.0, 0.0
    previous = None
    max_step_observed = 0.0
    for observation in result.observations:
        point = observation.point
        with localcontext() as ctx:
            ctx.prec = 80
            relaxation = (-2 * binary(point.time)).exp() / 2
            ref = [d(1.5) - relaxation, d(1.5) + relaxation]
        check = comparison(point.y, ref, tg["atol"], tg["rtol"])
        state_pass = state_pass and check["passed"]
        ratio = max(error / budget for error, budget in zip(check["absolute_errors"], check["budgets"], strict=True))
        max_state_budget_ratio = max(max_state_budget_ratio, ratio)
        inventory = None
        if previous is not None:
            increment = point.y - previous.y
            inventory, passed = inventory_record(increment, IDA_CASE["volumes"], [[0, 1]],
                                                  IDA_CASE["initial"], cumulative, gate)
            inventory_pass = inventory_pass and passed
            max_step_observed = max(max_step_observed, point.time - previous.time)
        max_residual = max(max_residual, float(np.max(np.abs(observation.residual))))
        # Every native output has time, state, native derivative, residual,
        # independent state errors and every domain's interval/cumulative defect.
        records.append({"t": point.time, "y": point.y.tolist(), "yp": observation.derivative.tolist(),
                        "residual": observation.residual.tolist(), "state_errors": check["absolute_errors"],
                        "state_budgets": check["budgets"], "inventory": inventory})
        previous = point
    conclude(request, {"gate": "NC01", "case": "equal_ida", "options": result.options,
                       "native_steps": len(records) - 1, "native_observations": records,
                       "completed": result.completed, "failure_reason": result.failure_reason,
                       "jacobian_calls": result.jacobian_calls, "residual_calls": result.residual_calls,
                       "native_run_elapsed_s": elapsed, "process_peak_rss_bytes_so_far": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       "trajectory_gate": tg, "max_state_budget_ratio": max_state_budget_ratio,
                       "max_residual_observed": max_residual, "max_step_observed": max_step_observed,
                       "cumulative_absolute_inventory_defect": cumulative,
                       "native_order_observation": "max_order configured; actual accepted order not exposed by this result"},
             {"native_completed": result.completed, "all_states": state_pass,
              "all_interval_and_cumulative_inventory": inventory_pass,
              "all_published_endpoints_recorded": len(records) == len(result.observations),
              "step_cap": max_step_observed <= max_step * (1 + 32 * np.finfo(float).eps),
              "jacobian_callback_used": result.jacobian_calls > 0})
