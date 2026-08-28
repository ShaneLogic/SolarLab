from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from perovskite_sim.experiments.dynamic_defect_impedance import (
    DynamicDefectImpedanceCapabilityError,
    DynamicDefectImpedanceProtocol,
    DynamicDefectImpedanceProtocolError,
    build_dynamic_defect_impedance_protocol,
    classify_dynamic_defect_capability,
    resolve_dynamic_defect_impedance_protocol,
)
from perovskite_sim.models.defects import EFFECTIVE_LIFETIME
from tests.integration.test_charged_explicit_defects_qf import (
    _grid as _bulk_grid,
)
from tests.integration.test_charged_explicit_defects_qf import (
    _stack as _bulk_stack,
)
from tests.integration.test_defect_ion_combined_impedance import (
    _bulk_interface_ion_stack,
    _contact_consistent_interface_stack,
    _with_mobile_ions,
)
from tests.integration.test_interface_defect_aware_impedance import (
    _grid as _interface_grid,
)


def _bulk_protocol() -> DynamicDefectImpedanceProtocol:
    stack = _bulk_stack()
    return build_dynamic_defect_impedance_protocol(
        stack,
        _bulk_grid(stack, 4),
        np.logspace(-4.0, 12.0, 33),
        requested_grid_intervals=4,
        V_dc=0.0,
        delta_V=0.01,
        illuminated=False,
    )


def test_protocol_round_trip_is_canonical_and_content_addressed():
    protocol = _bulk_protocol()

    rebuilt = DynamicDefectImpedanceProtocol.from_json(protocol.canonical_json())

    assert rebuilt == protocol
    assert rebuilt.protocol_hash == protocol.protocol_hash
    assert rebuilt.to_dict() == json.loads(protocol.canonical_json())


@pytest.mark.parametrize("mutation", ["unknown", "missing", "nested_unknown"])
def test_protocol_schema_rejects_unknown_and_missing_fields(mutation):
    payload = _bulk_protocol().to_dict()
    if mutation == "unknown":
        payload["claim"] = "externally_validated"
    elif mutation == "missing":
        del payload["grid_sha256"]
    else:
        payload["gates"]["unregistered_gate"] = 1.0

    with pytest.raises(DynamicDefectImpedanceProtocolError, match="keys"):
        DynamicDefectImpedanceProtocol.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("V_dc_V", 0.1),
        ("frequencies_Hz", (1.0e-4, 1.0, 1.0e10)),
        ("requested_grid_intervals", 8),
        ("grid_sha256", "0" * 64),
        ("stack_sha256", "1" * 64),
        ("defect_energy_quadrature_order", 16),
        ("state_step", 2.0e-5),
        ("voltage_step", 2.0e-5),
        ("refinement_factors", (1.0, 0.25)),
    ],
)
def test_protocol_mismatch_fails_closed(field, value):
    expected = _bulk_protocol()
    supplied = replace(expected, **{field: value})

    with pytest.raises(DynamicDefectImpedanceProtocolError, match="does not match"):
        resolve_dynamic_defect_impedance_protocol(supplied, expected)


def test_capability_classifier_routes_every_certified_combination():
    bulk = _bulk_stack()
    interface = _contact_consistent_interface_stack()

    assert classify_dynamic_defect_capability(bulk) == "bulk_dynamic_defect"
    assert classify_dynamic_defect_capability(interface) == (
        "interface_dynamic_defect"
    )
    assert classify_dynamic_defect_capability(_with_mobile_ions(bulk)) == (
        "bulk_defect_plus_ions"
    )
    assert classify_dynamic_defect_capability(_with_mobile_ions(interface)) == (
        "interface_defect_plus_ions"
    )
    assert classify_dynamic_defect_capability(_bulk_interface_ion_stack()) == (
        "bulk_interface_defect_plus_ions"
    )


def test_all_null_interface_slots_do_not_create_a_false_interface_capability():
    stack = _bulk_stack()
    stack = replace(stack, interface_defects=(None, None))

    assert classify_dynamic_defect_capability(stack) == "bulk_dynamic_defect"


def test_bulk_and_interface_without_ions_fail_closed():
    bulk = _bulk_stack()
    interface = _contact_consistent_interface_stack()
    combined = replace(
        interface,
        layers=(
            replace(interface.layers[0], params=bulk.layers[0].params),
            interface.layers[1],
        ),
    )

    with pytest.raises(
        DynamicDefectImpedanceCapabilityError,
        match=r"bulk \+ interface",
    ):
        classify_dynamic_defect_capability(combined)


def test_effective_lifetime_only_stack_has_no_dynamic_defect_capability():
    stack = _bulk_stack()
    params = stack.layers[0].params
    assert params is not None
    stack = replace(
        stack,
        layers=(
            replace(
                stack.layers[0],
                params=replace(
                    params,
                    defect_schema_version=None,
                    defect_model=EFFECTIVE_LIFETIME,
                    bulk_defects=(),
                ),
            ),
        ),
    )

    with pytest.raises(
        DynamicDefectImpedanceCapabilityError,
        match="requires an explicit bulk or canonical interface defect",
    ):
        classify_dynamic_defect_capability(stack)


def test_interface_protocol_binds_two_sided_trace_grid_identity():
    stack = _contact_consistent_interface_stack()
    grid = _interface_grid(stack)

    protocol = build_dynamic_defect_impedance_protocol(
        stack,
        grid,
        np.logspace(-8.0, 14.0, 45),
        requested_grid_intervals=8,
        V_dc=0.0,
        delta_V=0.01,
        illuminated=False,
    )

    assert protocol.capability == "interface_dynamic_defect"
    assert protocol.actual_grid_nodes == grid.size
    assert protocol.interface_current_observation == (
        "symmetric_adjacent_physical_faces"
    )
    assert len(protocol.interface_defect_document_sha256) == 1
