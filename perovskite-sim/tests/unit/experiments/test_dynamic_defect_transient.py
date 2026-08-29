from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.experiments.dynamic_defect_transient import (
    ALLOWED_TIME_STEP_REFINEMENT_FACTORS,
    DYNAMIC_DEFECT_TRANSIENT_REFERENCE_CERTIFICATE_SHA256,
    DynamicDefectTransientCapabilityError,
    DynamicDefectTransientCertificationError,
    DynamicDefectTransientProtocol,
    DynamicDefectTransientProtocolError,
    _maximum_positive_ion_relative_motion,
    _state_sha256,
    build_dynamic_defect_transient_protocol,
    classify_dynamic_defect_transient_capability,
    default_dynamic_defect_transient_policy,
    resolve_dynamic_defect_transient_protocol,
)
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    build_two_sided_trace_grid,
)
from perovskite_sim.models.config_loader import load_device_from_yaml


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"
TIMES_S = (0.0, 1.0e-8, 1.0e-6, 1.0e-4)
VOLTAGE_V = (0.0, 0.05, 0.05, 0.05)


def _stack():
    return load_device_from_yaml(CONFIG)


def _grid(stack):
    return build_two_sided_trace_grid(build_electrical_grid(stack, 4), stack)


def _protocol() -> DynamicDefectTransientProtocol:
    stack = _stack()
    return build_dynamic_defect_transient_protocol(
        stack,
        _grid(stack),
        TIMES_S,
        VOLTAGE_V,
        requested_grid_intervals=4,
    )


def test_protocol_round_trip_is_strict_canonical_and_content_addressed():
    protocol = _protocol()

    rebuilt = DynamicDefectTransientProtocol.from_json(protocol.canonical_json())

    assert rebuilt == protocol
    assert rebuilt.protocol_hash == protocol.protocol_hash
    assert rebuilt.to_dict() == json.loads(protocol.canonical_json())
    assert (
        rebuilt.reference_certificate_sha256
        == DYNAMIC_DEFECT_TRANSIENT_REFERENCE_CERTIFICATE_SHA256
    )
    assert rebuilt.solver_policy.maximum_newton_iterations == 100
    assert rebuilt.solver_policy.maximum_line_search_steps == 40
    assert rebuilt.solver_policy.maximum_near_acceptance_nonmonotone_steps == 2
    assert rebuilt.time_step_refinement_factor == 1.0
    assert rebuilt.solver_policy.refinement_substeps == (1, 2, 4)


@pytest.mark.parametrize("mutation", ["unknown", "missing", "nested_unknown"])
def test_protocol_schema_rejects_unknown_and_missing_fields(mutation):
    payload = _protocol().to_dict()
    if mutation == "unknown":
        payload["claim"] = "externally_validated"
    elif mutation == "missing":
        del payload["grid_sha256"]
    else:
        payload["solver_policy"]["unregistered_gate"] = 1.0

    with pytest.raises(DynamicDefectTransientProtocolError, match="keys"):
        DynamicDefectTransientProtocol.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("times_s", (0.0, 2.0e-8, 1.0e-6, 1.0e-4)),
        ("voltage_V", (0.0, 0.04, 0.04, 0.04)),
        ("requested_grid_intervals", 6),
        ("grid_sha256", "0" * 64),
        ("stack_sha256", "1" * 64),
    ],
)
def test_protocol_mismatch_fails_closed(field, value):
    expected = _protocol()
    supplied = replace(expected, **{field: value})

    with pytest.raises(DynamicDefectTransientProtocolError, match="does not match"):
        resolve_dynamic_defect_transient_protocol(supplied, expected)


def test_solver_policy_mismatch_fails_closed():
    expected = _protocol()

    with pytest.raises(ValueError, match="frozen time-step refinement factor"):
        replace(
            expected,
            solver_policy=replace(
                expected.solver_policy,
                maximum_newton_iterations=99,
            ),
        )


@pytest.mark.parametrize(
    ("factor", "substeps"),
    [(1.0, (1, 2, 4)), (0.5, (2, 4, 8)), (0.25, (4, 8, 16))],
)
def test_time_step_refinement_factor_binds_policy_and_protocol_hash(factor, substeps):
    stack = _stack()
    protocol = build_dynamic_defect_transient_protocol(
        stack,
        _grid(stack),
        TIMES_S,
        VOLTAGE_V,
        requested_grid_intervals=4,
        time_step_refinement_factor=factor,
    )

    assert ALLOWED_TIME_STEP_REFINEMENT_FACTORS == (1.0, 0.5, 0.25)
    assert protocol.time_step_refinement_factor == factor
    assert protocol.solver_policy.refinement_substeps == substeps
    assert protocol.solver_policy == default_dynamic_defect_transient_policy(factor)
    if factor != 1.0:
        assert protocol.protocol_hash != _protocol().protocol_hash
        with pytest.raises(DynamicDefectTransientProtocolError, match="does not match"):
            resolve_dynamic_defect_transient_protocol(protocol, _protocol())


def test_time_step_refinement_factor_rejects_unregistered_values():
    stack = _stack()

    with pytest.raises(ValueError, match="must be one of"):
        build_dynamic_defect_transient_protocol(
            stack,
            _grid(stack),
            TIMES_S,
            VOLTAGE_V,
            requested_grid_intervals=4,
            time_step_refinement_factor=0.75,
        )


def test_classifier_accepts_only_absorber_positive_ion_interface_slice():
    stack = _stack()

    assert classify_dynamic_defect_transient_capability(stack) == (
        "interface_defect_plus_positive_ions"
    )

    negative = replace(
        stack,
        layers=(
            replace(
                stack.layers[0],
                params=replace(
                    stack.layers[0].params,
                    D_ion_neg=1.0e-14,
                    P0_neg=1.0e22,
                ),
            ),
            stack.layers[1],
        ),
    )
    with pytest.raises(DynamicDefectTransientCapabilityError, match="negative ions"):
        classify_dynamic_defect_transient_capability(negative)

    two_positive = replace(
        stack,
        layers=(
            stack.layers[0],
            replace(
                stack.layers[1],
                params=replace(stack.layers[1].params, D_ion=1.0e-14),
            ),
        ),
    )
    with pytest.raises(DynamicDefectTransientCapabilityError, match="exactly one"):
        classify_dynamic_defect_transient_capability(two_positive)


def test_illuminated_public_transient_fails_before_execution():
    stack = _stack()

    with pytest.raises(DynamicDefectTransientCapabilityError, match="dark-only"):
        build_dynamic_defect_transient_protocol(
            stack,
            _grid(stack),
            TIMES_S,
            VOLTAGE_V,
            requested_grid_intervals=4,
            illuminated=True,
        )


def test_history_requires_t0_and_strictly_increasing_times():
    stack = _stack()
    grid = _grid(stack)

    with pytest.raises(ValueError, match="start at t=0"):
        build_dynamic_defect_transient_protocol(
            stack,
            grid,
            (1.0e-9, 1.0e-8),
            (0.0, 0.01),
            requested_grid_intervals=4,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        build_dynamic_defect_transient_protocol(
            stack,
            grid,
            (0.0, 1.0e-8, 1.0e-8),
            (0.0, 0.01, 0.01),
            requested_grid_intervals=4,
        )


def test_state_hash_accepts_readonly_solver_arrays_without_mutating_them():
    values = np.array([0.0, -0.0, 1.0])
    values.setflags(write=False)

    digest = _state_sha256("readonly-regression", values)

    assert len(digest) == 64
    assert not values.flags.writeable
    np.testing.assert_array_equal(values, np.array([0.0, -0.0, 1.0]))


def test_positive_ion_motion_ignores_inactive_structural_zero_nodes():
    density = np.array(
        [
            [0.0, 10.0, 20.0, 0.0],
            [0.0, 11.0, 18.0, 0.0],
        ]
    )

    assert _maximum_positive_ion_relative_motion(density, (1, 2)) == pytest.approx(0.1)
    with pytest.raises(
        DynamicDefectTransientCertificationError,
        match="positive reference",
    ):
        _maximum_positive_ion_relative_motion(density, (0,))
