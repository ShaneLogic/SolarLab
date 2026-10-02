"""Schedule-only coverage checks; these tests perform no physical integration."""
from copy import deepcopy
import math

import pytest

from scripts.r1_v10_extent import expected_schedule, extent, schedule_index_map


@pytest.fixture
def case():
    return {"time_substeps": [1, 2, 4], "times_s": [0.0, 0.1, 0.3]}


@pytest.mark.parametrize("levels,point_count,row_count", [
    ([8, 16, 32], 5, 227), ([1, 2, 4], 114, 794), ([1, 2, 4], 134, 934),
])
def test_counts_derive_from_supplied_case(levels, point_count, row_count):
    # Generic unit-test coordinates check the count formula, not a physical grid.
    case = {"time_substeps": levels, "times_s": [float(i) for i in range(point_count)]}
    rows = expected_schedule(case)
    report = extent(rows, case)
    assert len(rows) == report["expected_rows"] == row_count
    assert report["complete"] and report["prefix_valid"]
    assert report["first_mismatch"] is None and report["unassigned_rows"] == 0
    assert [level["expected_rows"] for level in report["levels"]] == [
        1 + (point_count - 1) * level for level in levels]


def test_schedule_uses_original_interval_duration_not_rounded_time_differences():
    case = {"time_substeps": [4], "times_s": [0.0, 0.1, 0.3]}
    rows = expected_schedule(case)
    assert rows[0] == {"substeps": 4, "time_s": 0.0, "dt_s": 0.0}
    assert rows[5] == {"substeps": 4, "time_s": 0.15, "dt_s": 0.049999999999999996}
    assert rows[6] == {"substeps": 4, "time_s": 0.2, "dt_s": 0.049999999999999996}
    assert rows[6]["time_s"] - rows[5]["time_s"] != rows[6]["dt_s"]
    rows[6]["dt_s"] = rows[6]["time_s"] - rows[5]["time_s"]
    report = extent(rows, case)
    assert not report["complete"] and not report["prefix_valid"]
    assert report["first_mismatch"]["row_index"] == 6


@pytest.mark.parametrize("field", ["time_s", "dt_s"])
def test_one_ulp_change_is_not_a_schedule_match(case, field):
    rows = expected_schedule(case)
    rows[1][field] = math.nextafter(rows[1][field], math.inf)
    report = extent(rows, case)
    assert not report["complete"] and not report["prefix_valid"]
    assert report["first_mismatch"]["row_index"] == 1
    assert report["first_mismatch"]["reason"] == "schedule_mismatch"


def test_tier_reordering_cannot_hide_behind_complete_individual_levels(case):
    rows = expected_schedule(case)
    rows = rows[3:8] + rows[:3] + rows[8:]
    report = extent(rows, case)
    assert all(item["row_count_complete"] and item["strictly_increasing"]
               for item in report["levels"])
    assert not report["prefix_valid"] and not report["complete"]
    assert report["first_mismatch"]["row_index"] == 0


def test_duplicate_row_is_reported_even_when_count_is_complete(case):
    rows = expected_schedule(case)
    rows[2] = deepcopy(rows[1])
    report = extent(rows, case)
    assert report["accepted_rows"] == report["expected_rows"]
    assert not report["complete"] and not report["prefix_valid"]
    assert report["first_mismatch"]["reason"] == "duplicate_row"


@pytest.mark.parametrize("length", [0, 1, 3, 4, 8, 16])
def test_empty_and_truncated_true_prefix_is_valid_but_incomplete(case, length):
    rows = expected_schedule(case)[:length]
    report = extent(rows, case)
    assert report["prefix_valid"] and report["first_mismatch"] is None
    assert not report["complete"]


def test_skipped_row_is_not_a_true_prefix(case):
    rows = expected_schedule(case)
    del rows[2]
    report = extent(rows, case)
    assert not report["prefix_valid"] and not report["complete"]
    assert report["first_mismatch"]["row_index"] == 2


@pytest.mark.parametrize("field,value", [
    ("time_s", math.nan), ("time_s", math.inf), ("dt_s", -math.inf),
    ("dt_s", None), ("time_s", True), ("substeps", True), ("substeps", 99),
])
def test_invalid_or_unassigned_row_identity_fails_closed(case, field, value):
    rows = expected_schedule(case)
    rows[1][field] = value
    report = extent(rows, case)
    assert not report["complete"] and not report["prefix_valid"]
    assert report["first_mismatch"]["row_index"] == 1
    assert report["unassigned_rows"] == (1 if field == "substeps" else 0)


def test_non_mapping_extra_row_and_signed_zero_fail_closed(case):
    rows = expected_schedule(case)
    report = extent(rows + [None], case)
    assert not report["complete"] and report["unassigned_rows"] == 1
    assert report["first_mismatch"]["reason"] == "invalid_row"
    rows[0]["dt_s"] = -0.0
    assert extent(rows, case)["first_mismatch"]["reason"] == "schedule_mismatch"


def test_extra_distinct_row_is_not_a_prefix(case):
    rows = expected_schedule(case)
    report = extent(rows + [{"substeps": 4, "time_s": 1.0, "dt_s": 0.1}], case)
    assert not report["complete"] and not report["prefix_valid"]
    assert report["first_mismatch"]["reason"] == "unexpected_row"


def test_short_to_long_mapping_preserves_tier_identity_and_exposes_tail(case):
    longer = {**case, "times_s": case["times_s"] + [1.0, 2.0]}
    short_rows, long_rows = expected_schedule(case), expected_schedule(longer)
    mapping = schedule_index_map(case, longer)
    assert mapping == [0, 1, 2, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    assert [long_rows[index] for index in mapping] == short_rows
    tail = [row for index, row in enumerate(long_rows) if index not in mapping]
    assert len(tail) == 14 and all(row["time_s"] > case["times_s"][-1] for row in tail)
    assert not extent(short_rows, longer)["prefix_valid"]


def test_mapping_refuses_changed_intervals_even_at_shared_output_times(case):
    changed = {**case, "times_s": [0.0, 0.05, 0.1, 0.3]}
    with pytest.raises(ValueError, match="not contained"):
        schedule_index_map(case, changed)


@pytest.mark.parametrize("case", [
    {"time_substeps": [], "times_s": [0.0, 1.0]},
    {"time_substeps": [1, 1], "times_s": [0.0, 1.0]},
    {"time_substeps": [True], "times_s": [0.0, 1.0]},
    {"time_substeps": [0], "times_s": [0.0, 1.0]},
    {"time_substeps": [1], "times_s": [0.0]},
    {"time_substeps": [1], "times_s": [1.0, 2.0]},
    {"time_substeps": [1], "times_s": [0.0, 0.0]},
    {"time_substeps": [1], "times_s": [0.0, math.inf]},
    {"time_substeps": [1], "times_s": [0.0, 1e-13]},
    {"time_substeps": [4], "times_s": [0.0, 1.0, math.nextafter(1.0, math.inf)]},
])
def test_invalid_case_or_collapsed_float_grid_is_rejected(case):
    with pytest.raises(ValueError):
        expected_schedule(case)
