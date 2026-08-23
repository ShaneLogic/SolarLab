"""Device-timescale frequency-window evidence for impedance experiments."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np

from perovskite_sim.constants import EPS_0, Q


@dataclass(frozen=True)
class IonicTimescale:
    """Order-of-magnitude ionic frequencies for one mobile-ion region."""

    species: Literal["positive", "negative"]
    region_start_m: float
    region_end_m: float
    region_length_m: float
    diffusion_coefficient_m2_s: float
    equilibrium_density_m3: float
    debye_length_m: float
    dielectric_frequency_Hz: float
    blocking_charge_frequency_Hz: float
    diffusion_frequency_Hz: float


@dataclass(frozen=True)
class IonicBranchCoverage:
    """Frequency evidence for one contiguous mobile-ion region."""

    species: Literal["positive", "negative"]
    region_start_m: float
    region_end_m: float
    diffusion_frequency_bracketed: bool
    blocking_charge_frequency_bracketed: bool
    dielectric_frequency_bracketed: bool
    full_timescale_envelope_bracketed: bool
    recommended_f_min_Hz: float
    recommended_f_max_Hz: float
    margin_covered: bool
    max_sampling_gap_decades: float
    covered: bool


@dataclass(frozen=True)
class FrequencyWindowAssessment:
    """Whether a requested sweep resolves the model's ionic timescales."""

    f_min_Hz: float
    f_max_Hz: float
    has_mobile_ions: bool
    characteristic_frequency_bracketed: bool | None = None
    ionic_branch_covered: bool | None = None
    ionic_timescales: tuple[IonicTimescale, ...] = ()
    warnings: tuple[str, ...] = ()
    full_timescale_envelope_bracketed: bool | None = None
    recommended_f_min_Hz: float | None = None
    recommended_f_max_Hz: float | None = None
    branch_margin_decades: float = 1.0
    max_allowed_sampling_gap_decades: float = 0.5
    max_observed_sampling_gap_decades: float | None = None
    ionic_branch_assessments: tuple[IonicBranchCoverage, ...] = ()


def _positive(value: object, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return number


def _contiguous_true_regions(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return inclusive index bounds for each contiguous true region."""
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    if indices.size == 0:
        return ()
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks], indices[-1]]
    return tuple(
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
    )


def _node_values_from_faces(values: np.ndarray, size: int) -> np.ndarray:
    """Build diagnostic node values from a face cache."""
    faces = np.asarray(values, dtype=float)
    if faces.shape != (size - 1,):
        raise ValueError("negative-ion diffusion cache is not grid aligned")
    nodes = np.zeros(size, dtype=float)
    nodes[0] = faces[0]
    nodes[-1] = faces[-1]
    if size > 2:
        nodes[1:-1] = 0.5 * (faces[:-1] + faces[1:])
    return nodes


def _species_data(material, grid_size: int) -> tuple[
    tuple[Literal["positive", "negative"], np.ndarray, np.ndarray], ...
]:
    positive = (
        "positive",
        np.asarray(material.D_ion_node, dtype=float),
        np.asarray(material.P_ion0, dtype=float),
    )
    species = [positive]
    if material.has_dual_ions and material.P_ion0_neg is not None:
        negative_diffusion = getattr(material, "D_ion_neg_node", None)
        if negative_diffusion is None:
            negative_diffusion = _node_values_from_faces(
                material.D_ion_neg_face,
                grid_size,
            )
        species.append(
            (
                "negative",
                np.asarray(negative_diffusion, dtype=float),
                np.asarray(material.P_ion0_neg, dtype=float),
            )
        )
    return tuple(species)


def assess_impedance_frequency_window(
    x: np.ndarray,
    material,
    frequencies: np.ndarray,
    *,
    branch_margin_decades: float = 1.0,
    max_sampling_gap_decades: float = 0.5,
) -> FrequencyWindowAssessment:
    """Compare a sweep with diffusion, blocking-charge, and dielectric scales.

    The returned limits are screening estimates, not fitted equivalent-circuit
    constants. The input frequencies are never changed.
    """
    grid = np.asarray(x, dtype=float)
    requested = np.asarray(frequencies, dtype=float)
    margin = _positive(branch_margin_decades, "branch_margin_decades")
    max_gap = _positive(
        max_sampling_gap_decades,
        "max_sampling_gap_decades",
    )
    if (
        grid.ndim != 1
        or grid.size < 2
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("x must be a finite, strictly increasing 1-D grid")
    if (
        requested.ndim != 1
        or requested.size == 0
        or not np.all(np.isfinite(requested))
        or np.any(requested <= 0.0)
    ):
        raise ValueError(
            "frequencies must be a finite, positive, non-empty 1-D array"
        )

    eps_r = np.asarray(material.eps_r, dtype=float)
    widths = np.asarray(material.dx_cell, dtype=float)
    thermal_voltage = _positive(material.V_T_device, "material.V_T_device")
    if (
        eps_r.shape != grid.shape
        or widths.shape != grid.shape
        or not np.all(np.isfinite(eps_r))
        or not np.all(np.isfinite(widths))
        or np.any(eps_r <= 0.0)
        or np.any(widths <= 0.0)
    ):
        raise ValueError(
            "material permittivity and dual-cell widths must be finite, "
            "positive, and grid aligned"
        )

    timescales: list[IonicTimescale] = []
    for species, diffusion, density in _species_data(material, grid.size):
        if (
            diffusion.shape != grid.shape
            or density.shape != grid.shape
            or not np.all(np.isfinite(diffusion))
            or not np.all(np.isfinite(density))
            or np.any(diffusion < 0.0)
            or np.any(density < 0.0)
        ):
            raise ValueError(
                f"{species}-ion diffusion and density caches must be finite, "
                "nonnegative, and grid aligned"
            )
        active = (diffusion > 0.0) & (density > 0.0)
        for start, end in _contiguous_true_regions(active):
            region = slice(start, end + 1)
            diffusion_eff = float(np.median(diffusion[region]))
            density_eff = float(np.median(density[region]))
            eps_eff = float(np.mean(eps_r[region]))
            region_length = float(np.sum(widths[region]))
            debye = float(
                np.sqrt(
                    EPS_0
                    * eps_eff
                    * thermal_voltage
                    / (Q * density_eff)
                )
            )
            tau_dielectric = debye * debye / diffusion_eff
            tau_charging = region_length * debye / (2.0 * diffusion_eff)
            tau_diffusion = region_length * region_length / diffusion_eff
            derived = (
                region_length,
                debye,
                tau_dielectric,
                tau_charging,
                tau_diffusion,
            )
            if any(
                not np.isfinite(value) or value <= 0.0
                for value in derived
            ):
                raise ValueError(
                    "mobile-ion timescale inputs produced a non-finite or "
                    "non-positive diagnostic"
                )
            scale = 2.0 * np.pi
            timescales.append(
                IonicTimescale(
                    species=species,
                    region_start_m=float(grid[start]),
                    region_end_m=float(grid[end]),
                    region_length_m=region_length,
                    diffusion_coefficient_m2_s=diffusion_eff,
                    equilibrium_density_m3=density_eff,
                    debye_length_m=debye,
                    dielectric_frequency_Hz=float(
                        1.0 / (scale * tau_dielectric)
                    ),
                    blocking_charge_frequency_Hz=float(
                        1.0 / (scale * tau_charging)
                    ),
                    diffusion_frequency_Hz=float(
                        1.0 / (scale * tau_diffusion)
                    ),
                )
            )

    f_min = float(np.min(requested))
    f_max = float(np.max(requested))
    if not timescales:
        return FrequencyWindowAssessment(
            f_min_Hz=f_min,
            f_max_Hz=f_max,
            has_mobile_ions=False,
            branch_margin_decades=margin,
            max_allowed_sampling_gap_decades=max_gap,
        )

    log_samples = np.unique(np.log10(requested))
    branch_assessments: list[IonicBranchCoverage] = []
    margin_factor = 10.0**margin
    for item in timescales:
        characteristic = (
            item.diffusion_frequency_Hz,
            item.blocking_charge_frequency_Hz,
            item.dielectric_frequency_Hz,
        )
        diffusion_bracketed = f_min <= characteristic[0] <= f_max
        blocking_bracketed = f_min <= characteristic[1] <= f_max
        dielectric_bracketed = f_min <= characteristic[2] <= f_max
        envelope_bracketed = (
            diffusion_bracketed
            and blocking_bracketed
            and dielectric_bracketed
        )
        recommended_low = min(characteristic) / margin_factor
        recommended_high = max(characteristic) * margin_factor
        margin_covered = (
            f_min <= recommended_low and f_max >= recommended_high
        )
        log_low = float(np.log10(recommended_low))
        log_high = float(np.log10(recommended_high))
        in_branch = log_samples[
            (log_samples >= log_low) & (log_samples <= log_high)
        ]
        sampling_nodes = np.unique(np.r_[log_low, in_branch, log_high])
        observed_gap = float(np.max(np.diff(sampling_nodes)))
        covered = margin_covered and observed_gap <= max_gap
        branch_assessments.append(
            IonicBranchCoverage(
                species=item.species,
                region_start_m=item.region_start_m,
                region_end_m=item.region_end_m,
                diffusion_frequency_bracketed=diffusion_bracketed,
                blocking_charge_frequency_bracketed=blocking_bracketed,
                dielectric_frequency_bracketed=dielectric_bracketed,
                full_timescale_envelope_bracketed=envelope_bracketed,
                recommended_f_min_Hz=recommended_low,
                recommended_f_max_Hz=recommended_high,
                margin_covered=margin_covered,
                max_sampling_gap_decades=observed_gap,
                covered=covered,
            )
        )

    blocking_bracketed = all(
        item.blocking_charge_frequency_bracketed
        for item in branch_assessments
    )
    full_envelope_bracketed = all(
        item.full_timescale_envelope_bracketed
        for item in branch_assessments
    )
    covered = all(item.covered for item in branch_assessments)
    warnings: list[str] = []
    if not blocking_bracketed:
        warnings.append(
            "ionic_blocking_charge_frequency_not_bracketed; extend the sweep "
            "before attributing or excluding a low-frequency ionic branch"
        )
    elif not full_envelope_bracketed:
        warnings.append(
            "ionic_timescale_envelope_not_bracketed; include the diffusion, "
            "blocking-charge, and dielectric characteristic frequencies"
        )
    elif not covered:
        warnings.append(
            "ionic_branch_sampling_inadequate; include the declared decade "
            "margin around the full ionic timescale envelope with no sampling "
            "gap above the protocol limit"
        )

    return FrequencyWindowAssessment(
        f_min_Hz=f_min,
        f_max_Hz=f_max,
        has_mobile_ions=True,
        characteristic_frequency_bracketed=blocking_bracketed,
        ionic_branch_covered=covered,
        ionic_timescales=tuple(timescales),
        warnings=tuple(warnings),
        full_timescale_envelope_bracketed=full_envelope_bracketed,
        recommended_f_min_Hz=min(
            item.recommended_f_min_Hz for item in branch_assessments
        ),
        recommended_f_max_Hz=max(
            item.recommended_f_max_Hz for item in branch_assessments
        ),
        branch_margin_decades=margin,
        max_allowed_sampling_gap_decades=max_gap,
        max_observed_sampling_gap_decades=max(
            item.max_sampling_gap_decades for item in branch_assessments
        ),
        ionic_branch_assessments=tuple(branch_assessments),
    )


__all__ = [
    "FrequencyWindowAssessment",
    "IonicBranchCoverage",
    "IonicTimescale",
    "assess_impedance_frequency_window",
]
