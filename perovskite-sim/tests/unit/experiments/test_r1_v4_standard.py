"""An unchanged physical standard refuses changed runtime criteria."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import PHYSICAL_STANDARD_RELATIVE_PATH
from perovskite_sim.experiments.one_dimensional_mechanism_r1_standard import load_physical_standard, verify_runtime_standard

PROJECT = Path(__file__).resolve().parents[3]


@pytest.fixture
def standard():
    return load_physical_standard(SimpleNamespace(read_bytes=lambda name: (PROJECT / name).read_bytes()))


@pytest.fixture
def reference_runtime():
    current = ("internal_face_current_spread_relative", "contact_internal_current_spread_relative",
               "interface_current_spread_relative")
    metrics = {**dict.fromkeys(current, 2e-6), "charge_balance_normalized": 1e-10, "inventory_relative_drift": 1e-10}
    return {"metric_limits": metrics, "metric_units": dict.fromkeys(metrics, "1"),
            "metric_applicability": {**dict.fromkeys((*current, "charge_balance_normalized"), "finite_step"),
                                     "inventory_relative_drift": "all_saved_rows"},
            "normalization": {"spread_floor_A_m2": 1e-20, "charge_scale_floor_A_m2": 1.},
            "ion_site_occupancy_ceiling": .999,
            "protocol_row_limits": {"gauss_normalized": 1e-10, "charge_balance_normalized": 1e-10,
                "inventory_relative_drift": 1e-10, "contact_internal_current_spread_relative": 2e-6},
            "policy_limits": {"maximum_charge_balance_relative_error": 1e-10,
                "maximum_all_face_current_spread_relative": 2e-6,
                "maximum_two_sided_interface_total_current_relative_error": 2e-6,
                "maximum_ion_inventory_relative_drift": 1e-10, "maximum_newton_iterations": 100,
                "maximum_line_search_steps": 40, "maximum_near_acceptance_nonmonotone_steps": 2}}


def test_original_standard_retains_explicit_normalization_and_units(standard, reference_runtime):
    assert verify_runtime_standard(standard, runtime=reference_runtime)["constants_match"]
    assert standard["populations"]["ion_site_comparison"] == "strict_less"


def test_installed_runtime_matches_the_frozen_physical_standard(standard):
    assert verify_runtime_standard(standard)["constants_match"]


def test_installed_limit_change_is_rejected_without_changing_the_standard(standard, monkeypatch):
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_independent_physics as independent
    changed = dict(independent.METRIC_LIMITS)
    changed["charge_balance_normalized"] = 1e-7
    monkeypatch.setattr(independent, "METRIC_LIMITS", changed)
    with pytest.raises(ValueError, match="executed R1 criteria differ.*metric_limits"):
        verify_runtime_standard(standard)


@pytest.mark.parametrize("metric", ["internal_face_current_spread_relative",
    "contact_internal_current_spread_relative", "interface_current_spread_relative",
    "charge_balance_normalized", "inventory_relative_drift"])
def test_changed_runtime_limit_does_not_inherit_standard_approval(standard, reference_runtime, metric):
    reference_runtime["metric_limits"][metric] *= 1000
    with pytest.raises(ValueError, match="executed R1 criteria differ"):
        verify_runtime_standard(standard, runtime=reference_runtime)


@pytest.mark.parametrize("group,key,value", [
    ("normalization", "spread_floor_A_m2", 1e-10),
    ("normalization", "charge_scale_floor_A_m2", 2.),
    ("metric_units", "charge_balance_normalized", "A/m2"),
    ("metric_applicability", "charge_balance_normalized", "all_saved_rows"),
    ("policy_limits", "maximum_newton_iterations", 1000),
    ("protocol_row_limits", "contact_internal_current_spread_relative", .1),
])
def test_normalization_unit_applicability_and_solver_bounds_are_bound(standard, reference_runtime, group, key, value):
    reference_runtime[group][key] = value
    with pytest.raises(ValueError, match="executed R1 criteria differ"):
        verify_runtime_standard(standard, runtime=reference_runtime)


def test_machine_standard_byte_change_requires_a_new_identity():
    original = (PROJECT / PHYSICAL_STANDARD_RELATIVE_PATH).read_bytes()
    raw = original.replace(b"2e-06", b"2e-03")
    assert raw != original
    with pytest.raises(ValueError, match="pinned digest"):
        load_physical_standard(SimpleNamespace(read_bytes=lambda name: raw))
