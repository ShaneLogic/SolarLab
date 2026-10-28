"""M parity for extracted finite-site mappings, without running legacy DAE."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from perovskite_sim.solver.dae_dual_ions import DualIonDAE, _shifted_logistic_density
from perovskite_sim.solver.dae_ions import SinglePositiveIonDAE
from solarlab.physics.ions.finite_site import (
    shared_site_densities, shared_site_fractions,
    shifted_logistic_density, single_site_density,
)


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


def same(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def single_reference(reference, capacity, logit_reference, coordinate):
    count = coordinate.size
    layout = SimpleNamespace(
        size=4 * count, electron_reference_m3=np.full(count, 1e20),
        hole_reference_m3=np.full(count, 2e20),
        electron_slice=slice(0, count), hole_slice=slice(count, 2 * count),
        positive_ion_slice=slice(2 * count, 3 * count),
        potential_slice=slice(3 * count, 4 * count),
        positive_ion_reference_m3=reference,
        positive_ion_site_limit_m3=capacity,
        positive_ion_logit_reference=logit_reference,
    )
    packed = np.r_[np.zeros(2 * count), coordinate, np.zeros(count)]
    return SinglePositiveIonDAE.physical_fields(SimpleNamespace(layout=layout), packed)[2]


def shared_reference(capacity, positive_reference, negative_reference, positive, negative):
    count = positive.size
    layout = SimpleNamespace(
        size=5 * count, node_count=count, shared_site=True,
        electron_reference_m3=np.full(count, 1e20), hole_reference_m3=np.full(count, 2e20),
        electron_slice=slice(0, count), hole_slice=slice(count, 2 * count),
        positive_ion_slice=slice(2 * count, 3 * count),
        negative_ion_slice=slice(3 * count, 4 * count),
        potential_slice=slice(4 * count, 5 * count),
        positive_ion_coordinate_reference=positive_reference,
        negative_ion_coordinate_reference=negative_reference,
        positive_ion_site_limit_m3=capacity,
    )
    packed = np.r_[np.zeros(2 * count), positive, negative, np.zeros(count)]
    source = SimpleNamespace(layout=layout)
    source._shared_site_fractions = lambda value: DualIonDAE._shared_site_fractions(source, value)
    fractions = DualIonDAE._shared_site_fractions(source, packed)
    densities = DualIonDAE.physical_fields(source, packed)[2:4]
    return fractions, densities


@pytest.mark.parametrize("count", [1, 5, 37])
@pytest.mark.parametrize("coordinate_kind", ["zero", "tiny", "mixed", "near_saturation"])
def test_single_and_distinct_site_maps_are_bitwise(count, coordinate_kind):
    capacity = np.geomspace(1e23, 1e28, count)
    occupancy = np.linspace(0.001, 0.45, count)
    reference = capacity * occupancy
    logit_reference = np.log(occupancy) - np.log1p(-occupancy)
    coordinate = {
        "zero": np.zeros(count),
        "tiny": np.linspace(-1e-16, 1e-16, count),
        "mixed": np.linspace(-10.0, 10.0, count),
        "near_saturation": np.linspace(-35.0, 30.0, count),
    }[coordinate_kind]
    actual = single_site_density(reference, capacity, logit_reference, coordinate)
    same(actual, single_reference(reference, capacity, logit_reference, coordinate))
    for label in ("positive-ion", "negative-ion", "custom-species"):
        same(
            shifted_logistic_density(reference, capacity, logit_reference, coordinate, label=label),
            _shifted_logistic_density(reference, capacity, logit_reference, coordinate, label=label),
        )
    if coordinate_kind == "zero":
        same(actual, reference)


@pytest.mark.parametrize("count", [1, 5, 37])
@pytest.mark.parametrize("coordinate_kind", ["zero", "tiny", "mixed", "near_saturation"])
def test_shared_site_fractions_and_common_capacity_are_bitwise(count, coordinate_kind):
    capacity = np.geomspace(1e23, 1e28, count)
    positive_fraction = np.linspace(1e-5, 0.3, count)
    negative_fraction = np.linspace(0.03, 0.2, count)
    vacancy_fraction = 1.0 - positive_fraction - negative_fraction
    positive_reference = np.log(positive_fraction / vacancy_fraction)
    negative_reference = np.log(negative_fraction / vacancy_fraction)
    positive = {
        "zero": np.zeros(count), "tiny": np.linspace(-1e-16, 1e-16, count),
        "mixed": np.linspace(-20.0, 20.0, count),
        "near_saturation": np.linspace(-700.0, 700.0, count),
    }[coordinate_kind]
    negative = -positive / 4.0
    # Keep all three fractions representable at the extreme shared-site input.
    if coordinate_kind == "near_saturation":
        negative = positive / 2.0
    args = (positive_reference, negative_reference, positive, negative)
    expected_fractions, expected_densities = shared_reference(capacity, *args)
    for actual, expected in zip(shared_site_fractions(*args), expected_fractions):
        same(actual, expected)
    for actual, expected in zip(shared_site_densities(capacity, *args), expected_densities):
        same(actual, expected)


def test_distinct_sublattices_keep_unequal_capacities():
    positive_limit = np.array([1e24, 3e26, 1e28])
    negative_limit = np.array([9e27, 2e25, 5e23])
    coordinate = np.array([-3.0, 0.0, 2.0])
    for capacity, occupancy, label in (
        (positive_limit, 0.1, "positive-ion"),
        (negative_limit, 0.2, "negative-ion"),
    ):
        reference = capacity * occupancy
        logit_reference = np.full(3, np.log(occupancy) - np.log1p(-occupancy))
        same(
            shifted_logistic_density(reference, capacity, logit_reference, coordinate, label=label),
            _shifted_logistic_density(reference, capacity, logit_reference, coordinate, label=label),
        )


@pytest.mark.parametrize("coordinate_value", [-1000.0, 1000.0])
def test_existing_single_distinct_and_shared_saturation_refusals_are_retained(coordinate_value):
    capacity = np.full(2, 1e27)
    reference = capacity * 0.1
    logit_reference = np.full(2, np.log(0.1) - np.log1p(-0.1))
    coordinate = np.full(2, coordinate_value)
    for actual, expected in (
        (lambda: single_site_density(reference, capacity, logit_reference, coordinate),
         lambda: single_reference(reference, capacity, logit_reference, coordinate)),
        (lambda: shifted_logistic_density(reference, capacity, logit_reference, coordinate, label="negative-ion"),
         lambda: _shifted_logistic_density(reference, capacity, logit_reference, coordinate, label="negative-ion")),
        (lambda: shared_site_fractions(logit_reference, logit_reference, coordinate, coordinate),
         lambda: shared_reference(capacity, logit_reference, logit_reference, coordinate, coordinate)),
    ):
        with pytest.raises(ValueError) as source_error:
            expected()
        with pytest.raises(type(source_error.value)) as new_error:
            actual()
        assert str(new_error.value) == str(source_error.value)
