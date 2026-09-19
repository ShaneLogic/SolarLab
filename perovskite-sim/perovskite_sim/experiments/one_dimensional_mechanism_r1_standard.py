"""Machine-readable R1 limits, checked against a caller-selected implementation.

The JSON is part of the frozen source and optional external standard approval.
These checks compare actual runtime constants, not arbitrary Python semantics.
Analytic/physical tests separately constrain the formulas recorded in the JSON.
"""
from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from types import MappingProxyType

from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import (
    PHYSICAL_STANDARD_RELATIVE_PATH, EXECUTION_STANDARD_RELATIVE_PATH,
    R1_DECLARATIONS, R1CheckoutError, current_execution_context, require_r1_checkout,
)


def load_physical_standard(context=None):
    context = context or require_r1_checkout()
    raw = context.read_bytes(PHYSICAL_STANDARD_RELATIVE_PATH)
    expected = R1_DECLARATIONS[PHYSICAL_STANDARD_RELATIVE_PATH][2]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise R1CheckoutError("R1 physical standard differs from its pinned digest")
    standard = json.loads(raw)
    if standard.get("schema") != "R1PhysicalStandardV1" or standard.get("version") != 1:
        raise R1CheckoutError("unsupported R1 physical standard")
    for name, metric in standard["metrics"].items():
        if (not isinstance(metric.get("limit"), (float, int))
                or isinstance(metric["limit"], bool) or not math.isfinite(metric["limit"])
                or metric["limit"] <= 0 or metric.get("comparison") != "finite_and_less_equal"):
            raise R1CheckoutError("invalid physical metric contract: " + name)
    return standard


def runtime_standard_contract():
    """Read the exact constants consumed by the installed R1 checks."""
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_independent_physics as independent
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol

    policy = protocol.r1_policy()
    return {
        "metric_limits": dict(independent.METRIC_LIMITS),
        "metric_units": dict(independent.METRIC_UNITS),
        "metric_applicability": dict(independent.METRIC_APPLICABILITY),
        "normalization": {"spread_floor_A_m2": independent.SPREAD_FLOOR_A_M2,
                          "charge_scale_floor_A_m2": independent.CHARGE_SCALE_FLOOR_A_M2},
        "ion_site_occupancy_ceiling": independent.ION_SITE_OCCUPANCY_CEILING,
        "protocol_row_limits": dict(protocol._PHYSICAL_LIMITS),
        "policy_limits": {name: getattr(policy, name) for name in (
            "maximum_charge_balance_relative_error", "maximum_all_face_current_spread_relative",
            "maximum_two_sided_interface_total_current_relative_error", "maximum_ion_inventory_relative_drift",
            "maximum_newton_iterations", "maximum_line_search_steps", "maximum_near_acceptance_nonmonotone_steps")},
    }


def verify_runtime_standard(standard, *, runtime=None):
    """Reject a changed runtime threshold under an unchanged physical standard.

    ``runtime`` supports direct conformance tests; production callers omit it
    so the installed implementation, not a result's own labels, is checked.
    """
    runtime = runtime_standard_contract() if runtime is None else runtime
    expected = {
        "metric_limits": {name: metric["limit"] for name, metric in standard["metrics"].items()},
        "metric_units": {name: metric["unit"] for name, metric in standard["metrics"].items()},
        "metric_applicability": {name: metric["applicability"] for name, metric in standard["metrics"].items()},
        "normalization": standard["normalization"],
        "ion_site_occupancy_ceiling": standard["populations"]["ion_site_occupancy_ceiling"],
        "protocol_row_limits": standard["protocol_row_limits"],
        "policy_limits": standard["policy_limits"],
    }
    mismatches = sorted(key for key in set(expected) | set(runtime) if runtime.get(key) != expected.get(key))
    if mismatches:
        raise R1CheckoutError("executed R1 criteria differ from the physical standard: " + ", ".join(mismatches))
    return {"schema": "R1RuntimeStandardCheckV1", "constants_match": True,
            "checked": sorted(expected),
            "scope": "installed_runtime_limits_units_normalization_applicability; formulas_require_separate_tests"}


def physical_standard_binding(context=None):
    context = context or require_r1_checkout()
    standard = load_physical_standard(context)
    execution = load_execution_standard(context)
    return standard, {"schema": "R1RuntimeStandardCheckV2", "constants_match": True,
        "physical_standard": verify_runtime_standard(standard),
        "execution_standard": verify_runtime_execution_standard(execution),
        "scope": "installed_effective_limits; per_row_used_limits_checked_at_execution; formulas_require_separate_tests"}


def load_execution_standard(context=None):
    """Load the separately versioned execution contract; physical V1 is unchanged."""
    context = context or require_r1_checkout()
    raw = context.read_bytes(EXECUTION_STANDARD_RELATIVE_PATH)
    _checked_execution_contract(raw)
    return json.loads(raw)


@lru_cache(maxsize=2)
def _checked_execution_contract(raw):
    expected = R1_DECLARATIONS[EXECUTION_STANDARD_RELATIVE_PATH][2]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise R1CheckoutError("R1 execution standard differs from its pinned digest")
    standard = json.loads(raw)
    if standard.get("schema") != "R1ExecutionStandardV1" or standard.get("version") != 1:
        raise R1CheckoutError("unsupported R1 execution standard")
    for group, limits in standard["effective_limits"].items():
        for key, value in limits.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise R1CheckoutError("invalid R1 execution limit: " + group + "." + key)
    return MappingProxyType({group: MappingProxyType(limits)
                             for group, limits in standard["effective_limits"].items()})


@lru_cache(maxsize=1)
def _development_execution_bytes():
    # Development rows reuse one captured dependency instead of rescanning Git
    # at every time step. Formal execution reads its own frozen context below.
    return require_r1_checkout().read_bytes(EXECUTION_STANDARD_RELATIVE_PATH)


def _execution_limits():
    context = current_execution_context()
    raw = (context.read_bytes(EXECUTION_STANDARD_RELATIVE_PATH) if context is not None
           else _development_execution_bytes())
    return _checked_execution_contract(raw)


def verify_used_execution_limits(group, actual):
    """Compare the exact limits just consumed/published, including all row keys."""
    expected = _execution_limits().get(group)
    if expected is None or actual != expected:
        raise R1CheckoutError("executed R1 limits differ from the execution standard: " + group)


def _effective_iteration_limits(policy):
    return {name: getattr(policy, name) for name in (
        "maximum_newton_iterations", "maximum_line_search_steps", "maximum_near_acceptance_nonmonotone_steps")}


def verify_execution_policy(policy):
    """Check the settings consumed by Newton for this particular execution."""
    from perovskite_sim.experiments.interface_defect_transient import effective_newton_acceptance_settings
    verify_used_execution_limits("newton_acceptance", effective_newton_acceptance_settings(policy))
    verify_used_execution_limits("iteration_caps", _effective_iteration_limits(policy))


def runtime_execution_contract(policy=None):
    """Read helpers consumed by production, not a parallel mirror of defaults."""
    from perovskite_sim.experiments.interface_defect_transient import effective_newton_acceptance_settings
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_step as step
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_independent_regular as regular
    policy = policy or protocol.r1_policy()
    return {
        "newton_acceptance": effective_newton_acceptance_settings(policy),
        "iteration_caps": _effective_iteration_limits(policy),
        "protocol_finite_row": protocol.effective_physical_row_limits(policy, finite_step=True),
        "protocol_initial_row": protocol.effective_physical_row_limits(policy, finite_step=False),
        "published_regular_relative": step.effective_regular_limits(policy),
        "published_regular_zero_excitation": step.effective_regular_limits(policy, require_relative_closure=False),
        "independent_regular_relative": regular.effective_regular_limits(policy),
        "independent_regular_zero_excitation": regular.effective_regular_limits(policy, require_relative_closure=False),
    }


def verify_runtime_execution_standard(standard, *, runtime=None):
    runtime = runtime_execution_contract() if runtime is None else runtime
    expected = standard["effective_limits"]
    mismatches = sorted(key for key in set(expected) | set(runtime) if runtime.get(key) != expected.get(key))
    if mismatches:
        raise R1CheckoutError("executed R1 criteria differ from the execution standard: " + ", ".join(mismatches))
    return {"schema": "R1RuntimeExecutionStandardCheckV1", "constants_match": True,
            "checked": sorted(expected),
            "scope": "effective_newton_and_regular_limits_and_published_row_limits; veto_and_formulas_require_separate_tests"}
