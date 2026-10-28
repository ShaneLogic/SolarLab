"""M extraction of the implemented finite-site coordinate mappings.

These array kernels preserve the legacy single/dual DAE mapping arithmetic
and saturation refusals. They do not create a DAE layout, residual or solver.
Distinct-site calls retain separate limits; shared-site densities use one
explicit common capacity, as in the original shared-site mapping.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

__all__ = [
    "single_site_density", "shifted_logistic_density",
    "shared_site_fractions", "shared_site_densities",
]


def single_site_density(
    reference_m3: np.ndarray,
    limit_m3: np.ndarray,
    logit_reference: np.ndarray,
    coordinate: np.ndarray,
) -> np.ndarray:
    """Retain the single-ion shifted logistic map and saturation error."""
    theta = expit(logit_reference + coordinate)
    theta_reference = reference_m3 / limit_m3
    positive_delta = (
        theta
        * (1.0 - theta_reference)
        * (-np.expm1(-np.maximum(coordinate, 0.0)))
    )
    negative_delta = (
        -theta_reference
        * (1.0 - theta)
        * (-np.expm1(np.minimum(coordinate, 0.0)))
    )
    density = reference_m3 + (
        limit_m3 * np.where(coordinate >= 0.0, positive_delta, negative_delta)
    )
    if np.any(theta <= 0.0) or np.any(theta >= 1.0):
        raise ValueError("single-ion DAE logit coordinate saturated")
    return density


def shifted_logistic_density(
    reference_m3: np.ndarray,
    limit_m3: np.ndarray,
    logit_reference: np.ndarray,
    coordinate: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    """Retain the distinct-site dual-ion map, including its label argument."""
    theta = expit(logit_reference + coordinate)
    theta_reference = reference_m3 / limit_m3
    positive_delta = (
        theta
        * (1.0 - theta_reference)
        * (-np.expm1(-np.maximum(coordinate, 0.0)))
    )
    negative_delta = (
        -theta_reference
        * (1.0 - theta)
        * (-np.expm1(np.minimum(coordinate, 0.0)))
    )
    density = reference_m3 + limit_m3 * np.where(
        coordinate >= 0.0,
        positive_delta,
        negative_delta,
    )
    if np.any(theta <= 0.0) or np.any(theta >= 1.0):
        raise ValueError(f"dual-ion {label} logit coordinate saturated")
    return density


def shared_site_fractions(
    positive_coordinate_reference: np.ndarray,
    negative_coordinate_reference: np.ndarray,
    positive_coordinate: np.ndarray,
    negative_coordinate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retain the shared-site (positive, negative, vacancy) softmax map."""
    positive_log_ratio = positive_coordinate_reference + positive_coordinate
    negative_log_ratio = negative_coordinate_reference + negative_coordinate
    offset = np.maximum.reduce(
        (
            np.zeros(positive_log_ratio.size),
            positive_log_ratio,
            negative_log_ratio,
        )
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        positive_weight = np.exp(positive_log_ratio - offset)
        negative_weight = np.exp(negative_log_ratio - offset)
        vacancy_weight = np.exp(-offset)
    denominator = positive_weight + negative_weight + vacancy_weight
    positive_fraction = positive_weight / denominator
    negative_fraction = negative_weight / denominator
    vacancy_fraction = vacancy_weight / denominator
    fractions = (positive_fraction, negative_fraction, vacancy_fraction)
    if any(
        not np.all(np.isfinite(value)) or np.any(value <= 0.0)
        for value in fractions
    ):
        raise ValueError("dual-ion shared-site coordinate saturated")
    return fractions


def shared_site_densities(
    site_limit_m3: np.ndarray,
    positive_coordinate_reference: np.ndarray,
    negative_coordinate_reference: np.ndarray,
    positive_coordinate: np.ndarray,
    negative_coordinate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Retain the original common-capacity multiplication for both species."""
    positive_fraction, negative_fraction, _vacancy = shared_site_fractions(
        positive_coordinate_reference, negative_coordinate_reference,
        positive_coordinate, negative_coordinate,
    )
    positive_ion = site_limit_m3 * positive_fraction
    negative_ion = site_limit_m3 * negative_fraction
    return positive_ion, negative_ion
