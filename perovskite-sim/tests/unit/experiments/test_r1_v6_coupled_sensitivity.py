"""The local diagnostic retains algebraic coupling and physical rate units."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

PATH = Path(__file__).resolve().parents[3] / "scripts/diagnose_r1_v6_coupled_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("coupled_sensitivity_v6", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ion_forcing_propagates_through_carrier_and_algebraic_blocks():
    # Independent analytic solution: x0=x1/2, x2=x1, x1=2.
    jacobian = csr_matrix([[2., -1., 0.], [-1., 3., -1.], [0., -1., 1.]])
    direction, rhs, backward = MODULE.solve_conditional_tangent(
        jacobian, [6.], [1., 1.], .5, [1])
    np.testing.assert_allclose(direction, [1., 2., 2.], rtol=1e-14)
    np.testing.assert_array_equal(rhs, [0., 3., 0.])
    assert backward < 1e-14


def test_storage_scaling_does_not_change_physical_tangent():
    a = csr_matrix([[2., -1., 0.], [-1., 3., -1.], [0., -1., 1.]])
    # Divide the ion equation and its forcing by two: same physical problem.
    b = csr_matrix([[2., -1., 0.], [-.5, 1.5, -.5], [0., -1., 1.]])
    x, _, _ = MODULE.solve_conditional_tangent(a, [6.], [1., 1.], .5, [1])
    y, _, _ = MODULE.solve_conditional_tangent(b, [6.], [1., 2.], .5, [1])
    np.testing.assert_allclose(x, y, rtol=1e-14)


@pytest.mark.parametrize("rate,scale,dt,rows", [
    ([6.], [1., 1.], 0., [1]),
    ([float("nan")], [1., 1.], .5, [1]),
    ([6.], [1., 0.], .5, [1]),
    ([6., 6.], [1., 1.], .5, [1, 1]),
    ([6.], [1., 1.], .5, [2]),
    ([6.], [1., 1.], .5, [1.5]),
])
def test_unavailable_or_ambiguous_forcing_is_rejected(rate, scale, dt, rows):
    with pytest.raises(ValueError):
        MODULE.solve_conditional_tangent(csr_matrix(np.eye(3)), rate, scale, dt, rows)


def test_clean_but_different_equations_cannot_relabel_an_old_trajectory():
    recorded = {"files": {"operator.py": "original"}, "study_input": {"sha": "fixed"}}
    changed = {"files": {"operator.py": "different"}, "study_input": {"sha": "fixed"}}
    with pytest.raises(ValueError, match="equations differ"):
        MODULE.validate_recorded_source(recorded, changed)
    MODULE.validate_recorded_source(recorded, dict(recorded, source_commit="new_wrapper_commit"))


def test_removing_a_required_file_from_the_manifest_is_rejected():
    manifest = {name: "hash" for name in (
        "FailureV1.json", "AcceptedStepsV1.jsonl", "CompletionV1.json")}
    MODULE.validate_manifest_coverage(manifest)
    del manifest["FailureV1.json"]
    with pytest.raises(ValueError, match="omits"):
        MODULE.validate_manifest_coverage(manifest)
