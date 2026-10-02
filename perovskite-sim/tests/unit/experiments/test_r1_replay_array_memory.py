"""Exact replay-array equivalence and temporary lifetime without physical solves."""

from copy import deepcopy
from dataclasses import dataclass
import pickle
import weakref

import numpy as np
import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_physics_validation as physics


def _legacy_arrays(rows, fields):
    return {key: np.asarray([row["state"][key] for row in rows]) for key in fields}


def _legacy_compare(actual, rows, fields):
    physics._same(actual, _legacy_arrays(rows, fields), "accepted state arrays")


def _check_both(actual, rows, fields, error=None):
    before = pickle.dumps((actual, rows, fields), protocol=5)
    for compare in (_legacy_compare, physics._same_accepted_state_arrays):
        if error is None:
            compare(actual, rows, fields)
        else:
            with pytest.raises(error):
                compare(actual, rows, fields)
        # Preserve representations, dtype, signed zero, ordering and aliases.
        assert pickle.dumps((actual, rows, fields), protocol=5) == before


@pytest.mark.parametrize("row_values", [
    [[1, 2], [3, 4]],
    [[1, 2], [3.0, 4.0]],
    [[[1, 2], [3, 4]], [[5.0, 6.0], [7.0, 8.0]]],
    [[0.0, -0.0, 5e-324], [-5e-324, 2.0**-80, 1.2345678901234567]],
], ids=["integers", "mixed-dtype", "nested-shape", "exact-floats"])
@pytest.mark.parametrize("row_wrap", [list, tuple, np.asarray], ids=["list", "tuple", "ndarray"])
@pytest.mark.parametrize("saved_wrap", [list, tuple, np.asarray], ids=["list", "tuple", "ndarray"])
def test_field_comparison_matches_full_block_for_supported_representations(
    row_values, row_wrap, saved_wrap,
):
    rows = [{"state": {"field": row_wrap(value)}} for value in row_values]
    expected = _legacy_arrays(rows, ("field",))["field"]
    actual = {"field": saved_wrap(expected.tolist())}
    _check_both(actual, rows, ("field",))


def test_dtype_is_inferred_across_rows_before_exact_comparison():
    rows = [{"state": {"field": [1, 2]}}, {"state": {"field": [3.0, 4.0]}}]
    _check_both({"field": [[1.0, 2.0], [3.0, 4.0]]}, rows, ("field",))
    # Individually comparing rows would wrongly accept the integer spelling.
    _check_both({"field": [[1, 2], [3.0, 4.0]]}, rows, ("field",),
                physics.R1PhysicsValidationError)


def _fixture():
    rows = [{"state": {
        "density": [[1.0 + i, 2.0 + i], [3.0 + i, 4.0 + i]],
        "potential": [float(i), -0.0, 5e-324],
        "precision_lo": [2.0**-80 * (i + 1), -2.0**-90 * (i + 1)],
    }} for i in range(3)]
    fields = tuple(rows[0]["state"])
    return _legacy_arrays(rows, fields), rows, fields


@pytest.mark.parametrize("mutation", [
    "missing-field", "extra-field", "missing-row", "no-rows", "extra-row", "swapped-rows",
    "last-element", "precision-low-word", "signed-zero", "subnormal", "shape",
])
def test_full_coverage_and_exact_value_mutations_are_rejected(mutation):
    actual, rows, fields = _fixture()
    if mutation == "missing-field":
        actual.pop(fields[-1])
    elif mutation == "extra-field":
        actual["extra"] = np.zeros(len(rows))
    elif mutation == "missing-row":
        rows.pop()
    elif mutation == "no-rows":
        rows.clear()
    elif mutation == "extra-row":
        rows.append(deepcopy(rows[-1]))
    elif mutation == "swapped-rows":
        rows[0], rows[-1] = rows[-1], rows[0]
    elif mutation == "last-element":
        actual[fields[-1]][-1, -1] += 1.0
    elif mutation == "precision-low-word":
        value = actual["precision_lo"][-1, -1]
        actual["precision_lo"][-1, -1] = np.nextafter(value, np.inf)
    elif mutation == "signed-zero":
        actual["potential"][-1, 1] = 0.0
    elif mutation == "subnormal":
        actual["potential"][-1, -1] = 0.0
    elif mutation == "shape":
        actual["density"] = actual["density"].reshape(len(rows), -1)
    _check_both(actual, rows, fields, physics.R1PhysicsValidationError)


def test_missing_field_within_recomputed_row_keeps_original_failure():
    actual, rows, fields = _fixture()
    del rows[-1]["state"][fields[-1]]
    _check_both(actual, rows, fields, KeyError)


@dataclass
class _ArrayRecord:
    field: object


@pytest.mark.parametrize("actual,states,fields,error", [
    (_ArrayRecord([[1.0]]), [{"field": [1.0]}], ("field",), None),
    (None, [{"field": [1.0]}], ("field",), physics.R1PhysicsValidationError),
    ([[1.0]], [{"field": [1.0]}], ("field",), physics.R1PhysicsValidationError),
    ({1: [[1.0]]}, [{"1": [1.0]}], ("1",), None),
    ({"1": [[2.0]]}, [{1: [1.0], "1": [2.0]}], (1, "1"), None),
    ({"1": [[1.0]]}, [{1: [1.0], "1": [2.0]}], ("1", 1), None),
    ({1: [[9.0]], "1": [[2.0]]}, [{"1": [2.0]}], ("1",), None),
    ({"1": [[9.0]], 1: [[2.0]]}, [{"1": [2.0]}], ("1",), None),
    ({1: [[2.0]], "1": [[9.0]]}, [{"1": [2.0]}], ("1",), physics.R1PhysicsValidationError),
    ({}, [], (), None),
], ids=["dataclass", "none", "list", "coerced-actual-key", "expected-collision",
        "reversed-expected-collision", "actual-collision", "reversed-actual-collision",
        "collision-mismatch", "empty-fields"])
def test_fallback_preserves_original_normalization(actual, states, fields, error):
    _check_both(actual, [{"state": state} for state in states], fields, error)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_field_coverage_is_checked_before_constructing_arrays(monkeypatch, mutation):
    actual, rows, fields = _fixture()
    if mutation == "missing":
        actual.pop(fields[-1])
    else:
        actual["extra"] = []

    def forbidden(*args, **kwargs):
        raise AssertionError("array allocation preceded complete field coverage")

    monkeypatch.setattr(physics.np, "asarray", forbidden)
    with pytest.raises(physics.R1PhysicsValidationError, match="accepted state arrays"):
        physics._same_accepted_state_arrays(actual, rows, fields)


def test_previous_temporary_is_released_before_next_field_is_built(monkeypatch):
    actual, rows, fields = _fixture()
    real_asarray = np.asarray
    references = []

    def observed_asarray(values):
        assert all(reference() is None for reference in references)
        result = real_asarray(values)
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(physics.np, "asarray", observed_asarray)
    physics._same_accepted_state_arrays(actual, rows, fields)
    assert len(references) == len(fields)
    assert all(reference() is None for reference in references)
