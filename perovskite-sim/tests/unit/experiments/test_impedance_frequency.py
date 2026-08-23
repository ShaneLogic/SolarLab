from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.experiments.impedance_frequency import (
    assess_impedance_frequency_window,
)


def _material(
    positive_diffusion: np.ndarray,
    positive_density: np.ndarray,
    *,
    negative_diffusion: np.ndarray | None = None,
    negative_density: np.ndarray | None = None,
):
    size = len(positive_diffusion)
    return SimpleNamespace(
        D_ion_node=np.asarray(positive_diffusion, dtype=float),
        P_ion0=np.asarray(positive_density, dtype=float),
        has_dual_ions=negative_density is not None,
        D_ion_neg_node=(
            None
            if negative_diffusion is None
            else np.asarray(negative_diffusion, dtype=float)
        ),
        D_ion_neg_face=np.zeros(max(size - 1, 0), dtype=float),
        P_ion0_neg=(
            None
            if negative_density is None
            else np.asarray(negative_density, dtype=float)
        ),
        eps_r=np.full(size, 24.0),
        dx_cell=np.full(size, 1.0e-9),
        V_T_device=0.02585,
    )


def _grid(size: int) -> np.ndarray:
    return np.arange(size, dtype=float) * 1.0e-9


def _dense_recommended_window(seed) -> np.ndarray:
    low = seed.recommended_f_min_Hz
    high = seed.recommended_f_max_Hz
    assert low is not None and high is not None
    decades = np.log10(high / low)
    count = int(np.ceil(decades / 0.25)) + 3
    return np.logspace(
        np.log10(low) - 0.01,
        np.log10(high) + 0.01,
        count,
    )


def test_no_mobile_ion_region_returns_not_applicable_evidence():
    material = _material(np.zeros(5), np.ones(5))

    result = assess_impedance_frequency_window(
        _grid(5),
        material,
        np.array([1.0, 10.0]),
    )

    assert not result.has_mobile_ions
    assert result.characteristic_frequency_bracketed is None
    assert result.full_timescale_envelope_bracketed is None
    assert result.ionic_branch_covered is None
    assert result.recommended_f_min_Hz is None
    assert result.recommended_f_max_Hz is None
    assert not result.ionic_timescales
    assert not result.ionic_branch_assessments
    assert not result.warnings


def test_disconnected_active_masks_produce_stable_region_segmentation():
    diffusion = np.array([1.0e-16, 1.0e-16, 0.0, 2.0e-16, 2.0e-16, 0.0, 3.0e-16])
    density = np.array([1.0e24, 1.0e24, 0.0, 2.0e24, 2.0e24, 0.0, 3.0e24])
    grid = _grid(diffusion.size)

    result = assess_impedance_frequency_window(
        grid,
        _material(diffusion, density),
        np.array([1.0]),
    )

    assert len(result.ionic_timescales) == 3
    assert tuple(
        (item.region_start_m, item.region_end_m)
        for item in result.ionic_timescales
    ) == (
        (grid[0], grid[1]),
        (grid[3], grid[4]),
        (grid[6], grid[6]),
    )
    assert tuple(
        item.region_length_m for item in result.ionic_timescales
    ) == pytest.approx((2.0e-9, 2.0e-9, 1.0e-9))


def test_symmetric_dual_ions_share_scales_and_dense_coverage():
    diffusion = np.array([0.0, 1.0e-16, 1.0e-16, 1.0e-16, 0.0])
    density = np.array([0.0, 1.0e24, 1.0e24, 1.0e24, 0.0])
    material = _material(
        diffusion,
        density,
        negative_diffusion=diffusion,
        negative_density=density,
    )
    seed = assess_impedance_frequency_window(
        _grid(5),
        material,
        np.array([1.0]),
    )

    result = assess_impedance_frequency_window(
        _grid(5),
        material,
        _dense_recommended_window(seed),
    )

    assert tuple(item.species for item in result.ionic_timescales) == (
        "positive",
        "negative",
    )
    positive, negative = result.ionic_timescales
    assert positive.debye_length_m == pytest.approx(negative.debye_length_m)
    assert positive.diffusion_frequency_Hz == pytest.approx(
        negative.diffusion_frequency_Hz
    )
    assert result.full_timescale_envelope_bracketed
    assert result.ionic_branch_covered
    assert all(item.covered for item in result.ionic_branch_assessments)
    assert result.warnings == ()


def test_policy_changes_recommendations_without_changing_requested_points():
    diffusion = np.full(4, 1.0e-16)
    density = np.full(4, 1.0e24)
    frequencies = np.array([1.0e-8, 1.0, 1.0e8])
    material = _material(diffusion, density)
    baseline = assess_impedance_frequency_window(
        _grid(4),
        material,
        frequencies,
        branch_margin_decades=1.0,
    )
    expanded = assess_impedance_frequency_window(
        _grid(4),
        material,
        frequencies,
        branch_margin_decades=2.0,
    )

    assert expanded.f_min_Hz == baseline.f_min_Hz == frequencies[0]
    assert expanded.f_max_Hz == baseline.f_max_Hz == frequencies[-1]
    assert expanded.recommended_f_min_Hz == pytest.approx(
        baseline.recommended_f_min_Hz / 10.0
    )
    assert expanded.recommended_f_max_Hz == pytest.approx(
        baseline.recommended_f_max_Hz * 10.0
    )


@pytest.mark.parametrize(
    "change, message",
    [
        ({"D_ion_node": np.array([0.0, np.nan, 0.0])}, "finite"),
        ({"P_ion0": np.array([0.0, -1.0, 0.0])}, "nonnegative"),
        ({"eps_r": np.ones(2)}, "grid aligned"),
        ({"dx_cell": np.array([1.0e-9, 0.0, 1.0e-9])}, "positive"),
    ],
)
def test_invalid_material_caches_fail_closed(change, message):
    material = _material(np.zeros(3), np.zeros(3))
    for name, value in change.items():
        setattr(material, name, value)

    with pytest.raises(ValueError, match=message):
        assess_impedance_frequency_window(
            _grid(3),
            material,
            np.array([1.0]),
        )


@pytest.mark.parametrize(
    "frequencies, kwargs, error",
    [
        (np.array([[1.0]]), {}, ValueError),
        (np.array([0.0]), {}, ValueError),
        (np.array([1.0]), {"branch_margin_decades": 0.0}, ValueError),
        (
            np.array([1.0]),
            {"max_sampling_gap_decades": False},
            TypeError,
        ),
    ],
)
def test_invalid_frequency_and_policy_inputs_fail_closed(
    frequencies,
    kwargs,
    error,
):
    with pytest.raises(error):
        assess_impedance_frequency_window(
            _grid(3),
            _material(np.zeros(3), np.zeros(3)),
            frequencies,
            **kwargs,
        )
