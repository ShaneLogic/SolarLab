"""Independent adversarial checks of the M-class parity decision."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix

from support.assembly_parity import Block, Extraction, Tolerance, compare


def reference() -> Extraction:
    vector = np.array([1.0, 2.0, 4.0, 8.0])
    matrix = np.arange(1.0, 17.0).reshape(4, 4)
    return Extraction(vector, csc_matrix(matrix), {"phi": (0, 2), "n": (2, 4)},
        S=vector / 2, dS_du=csc_matrix(matrix / 2), M=csc_matrix(-matrix / 2),
        boundary_masks={"fixed": np.array([True, False, False, True])})


@pytest.mark.parametrize("quantity", ("F", "J", "S", "dS_du", "M"))
def test_comparator_detects_one_ulp_in_each_canonical_quantity(quantity: str) -> None:
    old = reference()
    values = getattr(old, quantity)
    changed = values.toarray() if hasattr(values, "toarray") else values.copy()
    changed.flat[-1] = np.nextafter(changed.flat[-1], np.inf)
    result = compare(old, replace(old, **{quantity: changed}))
    assert not result.passed
    failures = [item for item in result.differences if not item.passed]
    assert len(failures) == 1 and failures[0].quantity == quantity
    assert failures[0].row_block == "n" and failures[0].changed_entries == 1
    with pytest.raises(AssertionError, match="assembly parity failed"):
        result.assert_equal()


def test_block_permutation_is_reported_by_semantic_block() -> None:
    old = reference()
    permutation = np.array([2, 3, 0, 1])
    candidate = replace(old, F=old.F[permutation], J=old.J.toarray()[np.ix_(permutation, permutation)])
    result = compare(old, candidate)
    assert not result.passed
    assert {item.row_block for item in result.differences if not item.passed} == {"phi", "n"}
    assert {item.quantity for item in result.differences if not item.passed} == {"F", "J"}


def test_layout_permutation_fails_even_with_identical_values() -> None:
    old = reference()
    changed = replace(old, blocks=(Block("n", 0, 2), Block("phi", 2, 4)))
    result = compare(old, changed)
    assert not result.passed and result.issues == ("named block layout differs",)
    assert all(item.bitwise_equal for item in result.differences)


def test_dense_sparse_and_column_vector_inputs_have_the_same_entries() -> None:
    old = reference()
    candidate = replace(old, F=csr_matrix(old.F[:, None]), J=old.J.toarray(),
                        M=csr_matrix(old.M), dS_du=old.dS_du.toarray())
    compare(old, candidate).assert_equal()


def test_tolerance_exception_must_be_explicit_and_quantity_specific() -> None:
    with pytest.raises(ValueError, match="justification"):
        Tolerance(rtol=1e-12)
    old = reference()
    changed = old.F.copy()
    changed[-1] = np.nextafter(changed[-1], np.inf)
    policy = Tolerance(rtol=float(2 * np.finfo(float).eps),
                       justification="Synthetic comparator test only; no migration tolerance is authorized")
    result = compare(old, replace(old, F=changed), tolerances={"F": policy})
    assert result.passed
    delta = next(item for item in result.differences if item.quantity == "F" and item.row_block == "n")
    assert not delta.bitwise_equal and delta.tolerance == policy
    with pytest.raises(ValueError, match="reference quantity"):
        compare(old, old, tolerances={"unknown": policy})


def test_relative_difference_has_no_hidden_epsilon() -> None:
    old = reference()
    changed = old.F.copy()
    changed[1] = 3.0
    result = compare(old, replace(old, F=changed))
    delta = next(item for item in result.differences if item.quantity == "F" and item.row_block == "phi")
    assert delta.max_absolute == 1.0 and delta.max_relative == 1 / 3


def test_masks_missing_storage_and_dtype_changes_cannot_hide() -> None:
    old = reference()
    assert not compare(old, replace(old, S=None)).passed
    assert not compare(old, replace(old, F=old.F.astype(np.float32))).passed
    masks = {"fixed": ~old.boundary_masks["fixed"]}
    result = compare(old, replace(old, boundary_masks=masks))
    assert not result.passed
    assert {item.quantity for item in result.differences if not item.passed} == {"mask:fixed"}


def test_signed_zero_complex_values_and_input_ownership() -> None:
    vector: np.ndarray = np.zeros(4, dtype=complex)
    matrix = np.eye(4)
    captured = Extraction(vector, matrix, {"all": (0, 4)})
    vector[0] = 1.0
    matrix[0, 0] = 9.0
    assert captured.F[0] == 0.0 and captured.J[0, 0] == 1.0
    negative_zero = captured.F.copy()
    negative_zero.real[0] = -0.0
    assert not compare(captured, replace(captured, F=negative_zero)).passed
    changed = captured.F.copy()
    changed.imag[0] = np.nextafter(0.0, 1.0)
    assert not compare(captured, replace(captured, F=changed)).passed


def test_sparse_stored_signed_zero_is_not_erased_by_dense_conversion() -> None:
    matrix = csc_matrix((np.array([1.0, 0.0]), ([0, 1], [0, 1])), shape=(2, 2))
    old = Extraction(np.zeros(2), matrix, {"all": (0, 2)})
    changed = matrix.copy()
    changed.data[1] = -0.0
    result = compare(old, replace(old, J=changed))
    assert not result.passed
    delta = next(item for item in result.differences if item.quantity == "J")
    assert delta.max_absolute == delta.max_relative == 0.0 and delta.changed_entries == 1


@pytest.mark.parametrize("blocks", ({"phi": (0, 2), "n": (3, 4)}, {"phi": (0, 3), "n": (2, 4)}))
def test_missing_or_overlapping_block_ranges_are_rejected(blocks: dict[str, tuple[int, int]]) -> None:
    with pytest.raises(ValueError, match="contiguous"):
        replace(reference(), blocks=blocks)


def test_nonfinite_outputs_are_not_parity_evidence() -> None:
    old = reference()
    bad = old.F.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        replace(old, F=bad)
