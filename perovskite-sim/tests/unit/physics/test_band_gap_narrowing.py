"""Constitutive tests for the opt-in Slotboom BGN closure."""

from __future__ import annotations

import math

import pytest

from perovskite_sim.physics.band_gap_narrowing import (
    apply_band_gap_narrowing,
    normalize_band_gap_narrowing_model,
    slotboom_band_gap_narrowing,
)


def test_slotboom_zero_and_reference_density_values_are_exactly_defined():
    assert slotboom_band_gap_narrowing(0.0) == 0.0
    assert slotboom_band_gap_narrowing(1.0e23) == pytest.approx(
        0.009 * math.sqrt(0.5),
        rel=2.0e-15,
    )


def test_slotboom_is_monotone_and_stable_below_ratio_underflow():
    densities = (5.0e-324, 1.0e-200, 1.0, 1.0e23, 1.0e25)
    narrowing = [slotboom_band_gap_narrowing(value) for value in densities]

    assert all(math.isfinite(value) and value > 0.0 for value in narrowing)
    assert narrowing == sorted(narrowing)
    logarithm = math.log(1.0e25) - math.log(1.0e23)
    expected = 0.009 * (logarithm + math.sqrt(logarithm**2 + 0.5))
    assert narrowing[-1] == pytest.approx(expected, rel=2.0e-15)


def test_band_edge_partition_preserves_requested_gap_reduction():
    state = apply_band_gap_narrowing(
        electron_affinity_eV=4.05,
        band_gap_eV=1.124,
        acceptor_density_m3=2.0e25,
        donor_density_m3=0.0,
        model="slotboom",
        conduction_band_fraction=0.3,
    )

    assert state.narrowing_eV > 0.0
    assert state.conduction_band_shift_eV == pytest.approx(0.3 * state.narrowing_eV)
    assert state.valence_band_shift_eV == pytest.approx(0.7 * state.narrowing_eV)
    assert state.effective_electron_affinity_eV == pytest.approx(
        4.05 + state.conduction_band_shift_eV
    )
    assert state.effective_band_gap_eV == pytest.approx(1.124 - state.narrowing_eV)


def test_bgn_identifiers_and_inert_companions_fail_closed():
    assert normalize_band_gap_narrowing_model(" Slotboom ") == "slotboom"
    with pytest.raises(ValueError, match="band-gap narrowing model"):
        normalize_band_gap_narrowing_model("jain_roulston")
    with pytest.raises(ValueError, match="BGN parameters require"):
        apply_band_gap_narrowing(
            electron_affinity_eV=4.05,
            band_gap_eV=1.124,
            acceptor_density_m3=0.0,
            donor_density_m3=0.0,
            model="off",
            reference_energy_eV=0.01,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"total_dopant_density_m3": -1.0}, "non-negative"),
        ({"reference_energy_eV": 0.0}, "positive"),
        ({"reference_density_m3": math.inf}, "positive"),
        ({"log_shape": 0.0}, "positive"),
    ),
)
def test_slotboom_parameters_are_strict(updates, message):
    arguments = {"total_dopant_density_m3": 1.0e23, **updates}
    with pytest.raises(ValueError, match=message):
        slotboom_band_gap_narrowing(**arguments)


def test_band_edge_partition_rejects_nonphysical_closure():
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        apply_band_gap_narrowing(
            electron_affinity_eV=4.05,
            band_gap_eV=1.124,
            acceptor_density_m3=1.0e23,
            donor_density_m3=0.0,
            model="slotboom",
            conduction_band_fraction=1.1,
        )
    with pytest.raises(ValueError, match="smaller than the base gap"):
        apply_band_gap_narrowing(
            electron_affinity_eV=4.05,
            band_gap_eV=0.01,
            acceptor_density_m3=1.0e30,
            donor_density_m3=0.0,
            model="slotboom",
        )
