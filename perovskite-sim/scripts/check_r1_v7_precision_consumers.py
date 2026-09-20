"""Development checks of actual DD R1 consumers and implementation faults.

This prepares one real N16 common state and performs controlled operator
evaluations.  Perturbed states are deliberately NOT claimed as accepted
trajectories, and these checks do not establish P1, long-window or D3 status.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from perovskite_sim.physics.compensated import DD
from perovskite_sim.experiments import one_dimensional_mechanism_r1_precision as production
from scripts.verify_r1_v7_precision import (
    STATE_FIELDS, compare_vectors, dec, evaluate_state, freeze_healthy_material, verify_pair, verify_side,
)


PROJECT = Path(__file__).resolve().parents[1]
FAULTS = ("diffusion", "thermal_voltage", "drift_sign", "omit_ion", "single_face_sign")

# A priori arithmetic bound, not an acceptance threshold fitted to a probe.
# For normalized (hi, lo), |lo| <= u*|hi|/(1-u). TwoSum/Dekker recover
# high-word rounding exactly in the normal finite range. In the current DD
# addition, only the low+tail additions round; their combined absolute error
# is below 4*u**2*(|a|+|b|). For DD multiplication, the two cross-products,
# their addition, error+cross, and final low*low accumulation contribute less
# than 12*u**2*|a*b|, including normalization factors (1-u)**-k. We use the
# same conservative 32*u**2 bound for either primitive and propagate input
# errors explicitly. This applies only to these add/multiply residuals: it
# makes no claim about exp/log, complete trajectories, or subnormal arithmetic.
with localcontext() as _bound_context:
    _bound_context.prec = 120
    DD_BINARY64_U = Decimal(2) ** -53
    DD_PRIMITIVE_GUARD = Decimal(32) * DD_BINARY64_U ** 2


@dataclass(frozen=True)
class _BoundedDecimal:
    value: Decimal
    error: Decimal = Decimal(0)

    def __add__(self, other):
        inherited = self.error + other.error
        rounding = DD_PRIMITIVE_GUARD * (abs(self.value) + abs(other.value) + inherited)
        return _BoundedDecimal(self.value + other.value, inherited + rounding)

    def __sub__(self, other):
        return self + _BoundedDecimal(-other.value, other.error)

    def __mul__(self, other):
        inherited = (abs(self.value) * other.error + abs(other.value) * self.error
                     + self.error * other.error)
        rounding = DD_PRIMITIVE_GUARD * (abs(self.value) + self.error) * (abs(other.value) + other.error)
        return _BoundedDecimal(self.value * other.value, inherited + rounding)


def _poisson_roundoff_bounds(frozen, state):
    """Independent forward error propagation for the declared DD residual DAG."""
    # The primitive bound assumes normalized finite words without subnormal
    # inputs. Fail closed if a future state leaves that domain.
    for name in STATE_FIELDS:
        pair = state.fine[name]
        for high, low in zip(pair.hi.ravel(), pair.lo.ravel()):
            h, l = dec(high), dec(low)
            if h == 0 and l != 0:
                raise ValueError("roundoff bound requires normalized DD inputs")
            if h != 0 and abs(l) > DD_BINARY64_U / (1 - DD_BINARY64_U) * abs(h):
                raise ValueError("roundoff bound requires normalized DD inputs")
            if any(Decimal(0) < abs(x) < Decimal(2) ** -1022 for x in (h, l)):
                raise ValueError("roundoff bound does not cover subnormal input words")

    def exact(value):
        value = dec(value)
        return _BoundedDecimal(value)

    phi, n, p, ions, sheet = ([exact(value) for value in decimals(state.fine[name])]
                             for name in STATE_FIELDS)
    raw, coefficients = frozen["raw"], frozen["coefficients"]
    q = exact(frozen["constants"]["q_C"])
    na, nd, background = ([exact(value) for value in raw[name]]
                          for name in ("N_A_m3", "N_D_m3", "P_ion0_m3"))
    widths = [exact(value) for value in coefficients["physical_width_m"]]
    capacitance = [exact(value) for value in coefficients["poisson_capacitance_F_m2"]]
    rho = [q * (p[j] - n[j] + nd[j] - na[j] + ions[j] - background[j]) for j in range(len(phi))]
    field = [capacitance[j] * (phi[j + 1] - phi[j]) for j in range(len(phi) - 1)]
    result = [field[j] - field[j - 1] + rho[j] * widths[j] for j in range(1, len(phi) - 1)]
    for nodes, weights, charge in zip(coefficients["interface_nodes"], coefficients["sheet_weights"], sheet):
        for node, weight in zip(nodes, weights):
            result[node - 1] = result[node - 1] + exact(weight) * charge
    return result


def absolute_response_report(actual, expected, absolute_bounds):
    """Require a valid absolute error bound and a response interval excluding 0."""
    with localcontext() as context:
        context.prec = 100
        a = decimals(actual) if isinstance(actual, DD) else [dec(x) for x in actual]
        e, bounds = [dec(x) for x in expected], [dec(x) for x in absolute_bounds]
        if len(a) != len(e) or len(a) != len(bounds) or any(x < 0 for x in bounds):
            raise ValueError("absolute response bounds must align and be nonnegative")
        errors = [abs(x - y) for x, y in zip(a, e)]
        nonzero = [j for j, value in enumerate(e) if value != 0]
        resolved = bool(nonzero) and all(abs(e[j]) > bounds[j] for j in nonzero)
        direction = bool(nonzero) and all(a[j] * e[j] > 0 for j in nonzero)
        within_bound = all(error <= bound for error, bound in zip(errors, bounds))
        scale = max(map(abs, e), default=Decimal(0))
        return {"actual": [str(x) for x in a], "expected": [str(x) for x in e],
                "absolute_error_bounds": [str(x) for x in bounds],
                "absolute_errors": [str(x) for x in errors],
                "absolute_error": str(max(errors, default=Decimal(0))),
                "maximum_absolute_error_bound": str(max(bounds, default=Decimal(0))),
                "expected_scale": str(scale),
                "relative_error": None if scale == 0 else str(max(errors, default=Decimal(0)) / scale),
                "nonzero_response_indices": nonzero, "response_interval_excludes_zero": resolved,
                "actual_nonzero_correct_direction": direction,
                "minimum_signal_to_bound": (None if not nonzero else str(min(
                    abs(e[j]) / bounds[j] if bounds[j] else Decimal("Infinity") for j in nonzero))),
                "criterion": "absolute_DD_roundoff_bound_and_resolvable_nonzero_response",
                "units": "C m^-2", "passed": within_bound and resolved and direction}


def poisson_low_response_report(frozen, state, previous, actual):
    with localcontext() as context:
        context.prec = 100
        new = _poisson_roundoff_bounds(frozen, state)
        old = _poisson_roundoff_bounds(frozen, previous)
        final = [a - b for a, b in zip(new, old)]
        # Oracle values independently evaluate the unchanged discrete equation;
        # differences refer to the actual represented inputs, not to the ideal
        # requested perturbation before its own DD normalization.
        a = evaluate_state(frozen, state_record(state), precision=100)["poisson"]["residual_C_m2"]
        b = evaluate_state(frozen, state_record(previous), precision=100)["poisson"]["residual_C_m2"]
        expected = [dec(x) - dec(y) for x, y in zip(a, b)]
        report = absolute_response_report(actual, expected, [value.error for value in final])
        report["bound_derivation"] = {
            "binary64_unit_roundoff": str(DD_BINARY64_U),
            "primitive_guard": str(DD_PRIMITIVE_GUARD),
            "formula": "32*u64^2; add: eA+eB+guard*(|a|+|b|+eA+eB); "
                       "multiply: |a|eB+|b|eA+eAeB+guard*(|a|+eA)*(|b|+eB)",
            "scope": "both_complete_DD_Poisson_residuals_then_final_DD_subtraction",
            "not_a_physical_acceptance_limit": True,
        }
        return report


def words(value):
    return {"hi": value.hi.tolist(), "lo": value.lo.tolist()}


def decimals(value):
    with localcontext() as context:
        context.prec = 100
        return [dec(a) + dec(b) for a, b in zip(value.hi.ravel(), value.lo.ravel())]


def state_record(state):
    return {"precision": {"representation": "float64-pair-v1",
                          "fields": {name: words(state.fine[name]) for name in STATE_FIELDS}}}


def observed(state, boundaries=(0., 0.)):
    return {"positive_flux_m2_s": words(state.fine["positive_flux_m2_s"]),
            "positive_rate_m3_s": words(state.fine["positive_rate_m3_s"]),
            "boundary_flux_m2_s": list(boundaries)}


def magnitude(value):
    return max(map(abs, decimals(value)), default=Decimal(0))


def delta_report(actual, expected, *, relative_limit="1e-9"):
    """Use expected response scale here, not floor=1 masking tiny consumers."""
    with localcontext() as context:
        context.prec = 100
        a = decimals(actual) if isinstance(actual, DD) else [dec(x) for x in np.asarray(actual).ravel()]
        e = [dec(x) for x in expected]
        if len(a) != len(e):
            raise ValueError("consumer response shapes disagree")
        size = max(map(abs, e), default=Decimal(0))
        error = max((abs(x - y) for x, y in zip(a, e)), default=Decimal(0))
        passed = size > 0 and error <= size * Decimal(relative_limit)
        return {"actual": [str(x) for x in a], "expected": [str(x) for x in e],
                "absolute_error": str(error), "expected_scale": str(size),
                "relative_error": None if size == 0 else str(error / size),
                "relative_limit": relative_limit, "passed": passed}


def changed_reference(system, state, field_name, node, change):
    fine = {name: value.copy() for name, value in state.fine.items()}
    fine[field_name] = production.put(fine[field_name], node, fine[field_name][node] + DD(change))
    changed = replace(state, fine=fine)
    working, _ = system.rebase(changed)
    evaluated = working.evaluate(np.zeros(working.dimension), 0.)
    return working, evaluated


def prepare_real_system():
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.experiments.one_dimensional_mechanism_r1 import build_r1_material
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as state_api
    fixture = PROJECT / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"
    binding_path = PROJECT / "tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json"
    stack = load_device_from_yaml(fixture)
    binding = json.loads(binding_path.read_text())
    grid, material = build_r1_material(stack, 16)
    frozen = freeze_healthy_material(grid, material, source_identity="development-current-source-before-faults")
    start = time.monotonic()
    with threadpool_limits(1), production.precision_context():
        prepared = state_api.prepare_common_state(stack, 16, binding, policy=r1_policy())
        system, state = state_api.verify_prepared_physics(prepared, stack, binding, policy=r1_policy())
    return system, state, frozen, {
        "wall_seconds": time.monotonic() - start, "prepared_sha256": prepared.sha256,
        "prepared": prepared.to_dict(), "fixture_sha256": sha(fixture), "binding_sha256": sha(binding_path),
        "scope": "fresh_common_state_from_raw_baseline_no_accepted_transient_steps",
    }


@contextmanager
def capture_jacobian_arguments():
    calls = []
    original = production.bernoulli_derivative_pair

    def captured(value):
        result = original(value)
        calls.append((value.copy(), result.copy()))
        return result

    production.bernoulli_derivative_pair = captured
    try:
        yield calls
    finally:
        production.bernoulli_derivative_pair = original


def jacobian_checks(system, state, low_system, low_state, frozen, node):
    with capture_jacobian_arguments() as base_calls:
        system._fine_work = {name: value.copy() for name, value in state.fine.items()}
        base = system._ion_jacobians(state.phi, state.positive, None)[2]
    with capture_jacobian_arguments() as low_calls:
        low_system._fine_work = {name: value.copy() for name, value in low_state.fine.items()}
        changed = low_system._ion_jacobians(low_state.phi, low_state.positive, None)[2]
    argument_delta = low_calls[0][0] - base_calls[0][0]
    derivative_delta = low_calls[0][1] - base_calls[0][1]
    with localcontext() as context:
        context.prec = 100

        def bprime(x):
            if x == 0:
                return Decimal("-0.5")
            ex = x.exp()
            return (ex - Decimal(1) - x * ex) / (ex - Decimal(1)) ** 2

        expected = [bprime(x) - bprime(y) for x, y in zip(decimals(low_calls[0][0]), decimals(base_calls[0][0]))]
        response = delta_report(derivative_delta, expected, relative_limit="1e-7")
        # Independent Decimal derivative of the rate at the ACTUAL fine state.
        # The production sparse column is a derivative in dimensionless phi/VT.
        h = Decimal("1e-24")
        plain = {name: [str(x) for x in decimals(state.fine[name])] for name in STATE_FIELDS}
        plus = {name: list(values) for name, values in plain.items()}
        minus = {name: list(values) for name, values in plain.items()}
        plus["phi_V"][node] = str(dec(plus["phi_V"][node]) + h)
        minus["phi_V"][node] = str(dec(minus["phi_V"][node]) - h)
        a, b = evaluate_state(frozen, plus), evaluate_state(frozen, minus)
        vt = dec(frozen["constants"]["thermal_voltage_V"])
        expected_column = [(dec(x) - dec(y)) / (2 * h) * vt
                           for x, y in zip(a["positive_rate_m3_s"], b["positive_rate_m3_s"])]
        column = base[:, system.potential_slice.start + node - 1].toarray().ravel()
        numerical = delta_report(column, expected_column, relative_limit="1e-10")
    return {"xi_low_response_nonzero": magnitude(argument_delta) > 0,
            "dd_derivative_low_response": response,
            "production_sparse_column_vs_decimal": numerical,
            "rounded_sparse_column_changed": bool((base != changed).nnz),
            "passed": magnitude(argument_delta) > 0 and response["passed"] and numerical["passed"],
            "scope": "transparent_instrumentation_of_real_DD_stage_before_explicit_sparse_float_rounding"}


def run_checks(system, initial, frozen):
    node = int(system.left_nodes[0])
    low_system, low = changed_reference(system, initial, "phi_V", node, 1e-23)
    driven_system, driven = changed_reference(system, initial, "phi_V", node, 1e-7)
    cases = (("fresh_zero_minus", system, initial), ("phi_low_1e-23V", low_system, low),
             ("resolved_phi_1e-7V", driven_system, driven))
    case_records, fault_records = [], []
    for name, owner, state in cases:
        check = verify_side(frozen, state_record(state), observed(state))
        case_records.append({"name": name, "state": state_record(state), "observed": observed(state),
                             "oracle": check, "scope": "operator_state_not_an_accepted_transient"})
        for fault in FAULTS:
            # Actual DD implementation faults are installed before evaluate;
            # both paths traverse the same faulty constitutive implementation.
            faulty_system, _ = owner.rebase(state)
            faulty_system.precision_fault = fault
            direct = faulty_system.evaluate(np.zeros(faulty_system.dimension), 0.)
            eliminated = faulty_system.evaluate(np.zeros(faulty_system.dimension), 0.)
            checked = verify_pair(frozen, state_record(direct), observed(direct),
                                  state_record(eliminated), observed(eliminated))
            detected = (check["qualified"] and not checked["direct"]["qualified"]
                        and not checked["eliminated"]["qualified"])
            fault_records.append({"case": name, "fault": fault,
                                  "injection": "CompensatedR1System.precision_fault -> ion_flux_pair -> actual evaluate",
                                  "healthy_qualified": check["qualified"], "detected": detected,
                                  "original_shared_difference_passed": checked["original_gate"]["passed"],
                                  "result": checked})

    consumers = {}
    with localcontext() as context:
        context.prec = 100
        actual_phi = low.fine["phi_V"] - initial.fine["phi_V"]
        expected_phi = [Decimal(0)] * system.node_count
        expected_phi[node] = dec(1e-23)
        consumers["rebase_potential"] = delta_report(actual_phi, expected_phi)
        expected_increment = [float(x) for x in expected_phi]
        consumers["potential_increment"] = delta_report(
            low_system.potential_increment(low, initial), expected_increment)
        q = dec(frozen["constants"]["q_C"])
        widths = [dec(x) for x in frozen["coefficients"]["physical_width_m"]]
        base_oracle, low_oracle = evaluate_state(frozen, state_record(initial)), evaluate_state(frozen, state_record(low))
        consumers["ion_flux_phi_low"] = delta_report(
            low.fine["positive_flux_m2_s"] - initial.fine["positive_flux_m2_s"],
            [dec(a) - dec(b) for a, b in zip(low_oracle["positive_flux_m2_s"], base_oracle["positive_flux_m2_s"])],
            relative_limit="1e-6")
        # Nonzero trace/bulk difference produces a small directly represented
        # displacement increment even if rounded absolute potentials coincide.
        _, displacement, _ = low_system.interface_current_sides(low, initial, 1.)
        cl = (production.EPS_0 * system.material.eps_r[node]
              / system.material.iface_qss_left_distances_m[0])
        expected_current = [dec(system.polarity) * dec(cl) * dec(1e-23), Decimal(0)]
        consumers["interface_displacement_phi_low"] = delta_report(displacement[0], expected_current)

        for field_name, change in (("p_m3", 1e-5), ("positive_m3", 1000.)):
            owner, state = changed_reference(system, initial, field_name, node, change)
            label = "hole" if field_name == "p_m3" else "ion"
            consumers[f"rebase_{label}"] = delta_report(
                state.fine[field_name] - initial.fine[field_name],
                [dec(change) if j == node else Decimal(0) for j in range(system.node_count)])
            storage = owner.storage_increment(state, initial)
            slot = system.interior_count + node - 1 if label == "hole" else (
                2 * system.interior_count + system.interface_count
                + list(system.positive_nodes).index(node))
            expected_storage = [Decimal(0)] * len(storage)
            expected_storage[slot] = dec(change)
            consumers[f"storage_{label}_low"] = delta_report(storage, expected_storage)
            expected_charge = [q * dec(change) * widths[node]]
            consumers[f"charge_{label}_low"] = delta_report(
                [owner.integrated_charge_increment(state, initial)], expected_charge)
            calculated = owner._poisson_pair(state.fine["phi_V"], state.fine["n_m3"], state.fine["p_m3"],
                                             state.fine["positive_m3"], state.fine["sheet_charge_C_m2"])
            previous = system._poisson_pair(initial.fine["phi_V"], initial.fine["n_m3"], initial.fine["p_m3"],
                                           initial.fine["positive_m3"], initial.fine["sheet_charge_C_m2"])
            consumers[f"poisson_{label}_low"] = poisson_low_response_report(
                frozen, state, initial, calculated - previous)
            if label == "ion":
                changed_oracle = evaluate_state(frozen, state_record(state))
                consumers["ion_flux_population_low"] = delta_report(
                    state.fine["positive_flux_m2_s"] - initial.fine["positive_flux_m2_s"],
                    [dec(a) - dec(b) for a, b in zip(changed_oracle["positive_flux_m2_s"],
                                                    base_oracle["positive_flux_m2_s"])], relative_limit="1e-8")
    consumers["jacobian"] = jacobian_checks(system, initial, low_system, low, frozen, node)

    # A real continuity-assembly fault: the helper receives a nonzero contact
    # flux before taking the divergence. The reported boundary is the actual
    # injected value captured at that call, not an inferred zero.
    original_cat, calls = production.cat, []

    def leaky_cat(*items):
        if len(items) == 3 and isinstance(items[1], DD) and items[1].shape == (system.node_count - 1,):
            calls.append([1e-6, 0.])
            return original_cat(1e-6, items[1], 0.)
        return original_cat(*items)

    try:
        production.cat = leaky_cat
        owner, _ = driven_system.rebase(driven)
        leaked = owner.evaluate(np.zeros(owner.dimension), 0.)
    finally:
        production.cat = original_cat
    boundary_check = verify_side(frozen, state_record(leaked), observed(leaked, calls[-1]))
    boundary = {"injection": "actual DD continuity boundary assembly before divergence",
                "actual_calls": calls, "result": boundary_check,
                "detected": bool(calls) and not boundary_check["checks"]["boundary_flux_m2_s"]["passed"]}
    qualified = (all(item["oracle"]["qualified"] for item in case_records)
                 and all(item["detected"] for item in fault_records)
                 and all(item["passed"] for item in consumers.values()) and boundary["detected"])
    return {"schema": "R1V7PrecisionConsumerChecksV1", "run_class": "development",
            "operator_consumer_checks_passed": qualified, "P1_qualified": False,
            "scope": "actual_N16_fine_initialization_and_controlled_operator_states_no_transient_qualification",
            "cases": case_records, "faults": fault_records, "consumers": consumers,
            "boundary_fault": boundary, "actual_accepted_transient_steps": 0,
            "fault_boundary": "shared_operator_implementation_not_independent_eliminated_Poisson_trajectory",
            "newton_jacobian_rounding": "explicit float sparse boundary; low parts need not change final float columns"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity():
    paths = sorted((PROJECT / "perovskite_sim").rglob("*.py"))
    paths += [Path(__file__).resolve(), PROJECT / "scripts/verify_r1_v7_precision.py"]
    return {"head_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
            "working_tree_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT, text=True),
            "source_files": {str(p.relative_to(PROJECT)): sha(p) for p in paths},
            "run_class": "development", "final_source_claim": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before, start = source_identity(), time.monotonic()
    try:
        with threadpool_limits(1):
            system, state, frozen, preparation = prepare_real_system()
            result = run_checks(system, state, frozen)
            for name, value in (("FrozenHealthyMaterialV1.json", frozen), ("PreparationV1.json", preparation)):
                (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
            result["threadpools"] = threadpool_info()
    except Exception as exc:
        result = {"schema": "R1V7PrecisionConsumerChecksV1", "run_class": "development",
                  "operator_consumer_checks_passed": False, "P1_qualified": False,
                  "error": {"type": type(exc).__name__, "message": str(exc)}}
    after = source_identity()
    result.update(source_before=before, source_after=after,
                  source_changed_during_run=before["source_files"] != after["source_files"],
                  wall_seconds=time.monotonic() - start)
    if result["source_changed_during_run"]:
        result["operator_consumer_checks_passed"] = False
        result["source_note"] = "source changed during development checks; a frozen rerun is required"
    (args.output / "ResultV1.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result.get(key) for key in (
        "operator_consumer_checks_passed", "error", "source_changed_during_run", "wall_seconds")}, indent=2))
    return 0 if result["operator_consumer_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
