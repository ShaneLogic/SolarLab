"""Manufactured band-drive checks; no device, SG, or reference evaluation."""

from fractions import Fraction

import numpy as np
import pytest

from scripts.benchmarks.band_steps import band_steps


def exact_steps(*fields):
    """Independent rational differences of the supplied binary64 inputs."""
    return np.array([
        float(sum((Fraction(float(field[i + 1])) - Fraction(float(field[i]))
                   for field in fields), Fraction(0)))
        for i in range(len(fields[0]) - 1)
    ])


def test_common_band_offset_does_not_erase_small_electric_drive():
    phi = np.array([0.0, 1e-16, 2e-16])
    affinity = np.full(3, 4.0)
    gap = np.full(3, 1.6)
    electron, hole = band_steps(phi, affinity, gap)
    np.testing.assert_array_equal(electron, exact_steps(phi, affinity))
    np.testing.assert_array_equal(hole, exact_steps(phi, affinity, gap))
    assert np.all(hole > 0)
    np.testing.assert_array_equal(np.diff(phi + affinity + gap), [0.0, 0.0])


def test_distinct_conduction_and_valence_offsets_keep_their_signs():
    phi = np.array([0.125, 0.25, 0.5, 1.0])
    affinity = np.array([4.0, 4.5, 4.5, 5.0])
    gap = np.array([2.0, 1.5, 1.5, 0.5])
    before = tuple(value.copy() for value in (phi, affinity, gap))
    electron, hole = band_steps(phi, affinity, gap)
    np.testing.assert_array_equal(electron, exact_steps(phi, affinity))
    np.testing.assert_array_equal(hole, exact_steps(phi, affinity, gap))
    assert electron[-1] == 1.0 and hole[-1] == 0.0
    for original, value in zip(before, (phi, affinity, gap), strict=True):
        np.testing.assert_array_equal(original, value)


def test_reversing_nodes_reverses_face_drives():
    fields = ([0.0, 0.125, 0.25], [4.0, 4.5, 5.0], [2.0, 1.0, 0.5])
    forward = band_steps(*fields)
    reverse = band_steps(*(field[::-1] for field in fields))
    for left, right in zip(forward, reverse, strict=True):
        np.testing.assert_array_equal(right, -left[::-1])


@pytest.mark.parametrize("fields", [
    ([0.0], [4.0], [1.6]),
    ([0.0, 1.0], [4.0, 4.0, 4.0], [1.6, 1.6]),
    ([[0.0, 1.0]], [[4.0, 4.0]], [[1.6, 1.6]]),
    ([0.0, np.nan], [4.0, 4.0], [1.6, 1.6]),
    ([0.0, 1.0], [4.0, np.inf], [1.6, 1.6]),
    ([False, True], [4.0, 4.0], [1.6, 1.6]),
])
def test_invalid_component_arrays_are_rejected(fields):
    with pytest.raises(ValueError):
        band_steps(*fields)
