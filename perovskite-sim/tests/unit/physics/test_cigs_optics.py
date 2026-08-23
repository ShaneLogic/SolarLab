"""Unit contracts for composition-dependent CIGS optical constants."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest

from perovskite_sim.physics.cigs_optics import (
    CIGSGradedOptics,
    HC_EV_NM,
    _REFERENCES,
    _spectrum_at_cgi_anchor,
    _target_epsilon2_and_infinity,
    _tauc_lorentz_epsilon2,
    carron_absorption_coefficient,
    carron_band_gap_eV,
    cigs_graded_optics_from_mapping,
    ggi_profile_from_coordinate,
    minoura_critical_points,
    minoura_dielectric_function,
    minoura_nk,
)


def test_schema_is_frozen_and_strict() -> None:
    model = CIGSGradedOptics(0.2, 0.6, 0.9)
    with pytest.raises(FrozenInstanceError):
        model.cgi = 0.95
    with pytest.raises(ValueError, match="unknown"):
        cigs_graded_optics_from_mapping(
            {"ggi_front": 0.2, "ggi_back": 0.6, "cgi": 0.9, "slice": 25}
        )
    with pytest.raises(ValueError, match="missing"):
        cigs_graded_optics_from_mapping({"ggi_front": 0.2, "cgi": 0.9})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ggi_front", True),
        ("ggi_back", False),
        ("cgi", True),
        ("slices", 25.5),
        ("slices", True),
        ("kk_quadrature_order", 191.5),
        ("kk_quadrature_order", False),
    ],
)
def test_mapping_rejects_noninteger_resolution_values(
    field: str, value: object
) -> None:
    raw = {
        "ggi_front": 0.2,
        "ggi_back": 0.6,
        "cgi": 0.9,
        field: value,
    }
    with pytest.raises(ValueError, match=field):
        cigs_graded_optics_from_mapping(raw)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ggi_front": -0.1, "ggi_back": 0.5, "cgi": 0.9}, "ggi_front"),
        ({"ggi_front": 0.1, "ggi_back": 1.1, "cgi": 0.9}, "ggi_back"),
        ({"ggi_front": 0.1, "ggi_back": 0.5, "cgi": 0.7}, "cgi"),
        ({"ggi_front": 0.1, "ggi_back": 0.5, "cgi": 0.9, "slices": 0}, "slices"),
        (
            {
                "ggi_front": 0.1,
                "ggi_back": 0.5,
                "cgi": 0.9,
                "kk_quadrature_order": 16,
            },
            "kk_quadrature_order",
        ),
    ],
)
def test_schema_rejects_out_of_domain_values(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CIGSGradedOptics(**kwargs)


def test_minoura_critical_points_match_published_endpoint_equations() -> None:
    np.testing.assert_allclose(
        minoura_critical_points(0.0, 0.9),
        [1.00, 2.94, 3.71, 4.71, 5.24],
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        minoura_critical_points(1.0, 0.9),
        [1.71, 3.33, 4.20, 5.15, 5.88],
        atol=1e-14,
        rtol=0.0,
    )


@pytest.mark.parametrize("name", ["A", "B", "C", "D", "E", "F"])
def test_reference_compositions_recover_published_tl_epsilon2(name: str) -> None:
    reference = _REFERENCES[name]
    energy = np.linspace(0.7, 6.5, 181)
    if reference.cgi >= 0.75:
        actual, epsilon_infinity = _target_epsilon2_and_infinity(
            energy, reference.ggi, reference.cgi
        )
    else:
        # F is the lower spectral-average anchor.  It remains necessary to
        # interpolate public CGI>=0.75 requests, but is not itself exposed by
        # the stricter device-domain API.
        actual, epsilon_infinity = _spectrum_at_cgi_anchor(
            energy, target_ggi=reference.ggi, anchor_cgi=reference.cgi
        )
    expected = _tauc_lorentz_epsilon2(energy, reference.peaks)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
    assert epsilon_infinity == pytest.approx(reference.epsilon_infinity, abs=2e-15)


def test_minoura_nk_is_finite_nonnegative_and_causal_conversion() -> None:
    wavelengths = np.linspace(300.0, 1400.0, 101)
    n, k = minoura_nk(wavelengths, 0.25, 0.90, quadrature_order=192)
    assert np.all(np.isfinite(n))
    assert np.all(np.isfinite(k))
    assert np.all(n > 1.0)
    assert np.all(k >= 0.0)

    energy = HC_EV_NM / wavelengths
    epsilon1, epsilon2 = minoura_dielectric_function(
        energy, 0.25, 0.90, quadrature_order=192
    )
    np.testing.assert_allclose(n * n - k * k, epsilon1, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(2.0 * n * k, epsilon2, rtol=2e-9, atol=2e-12)


def test_kk_quadrature_refinement_converges_below_half_percent() -> None:
    wavelengths = np.linspace(300.0, 1400.0, 71)
    coarse = minoura_nk(wavelengths, 0.37, 0.86, quadrature_order=96)
    medium = minoura_nk(wavelengths, 0.37, 0.86, quadrature_order=192)
    fine = minoura_nk(wavelengths, 0.37, 0.86, quadrature_order=384)
    for coarse_quantity, medium_quantity, fine_quantity in zip(
        coarse, medium, fine, strict=True
    ):
        scale = np.maximum(np.abs(fine_quantity), 1e-8)
        coarse_error = np.max(np.abs(coarse_quantity - fine_quantity) / scale)
        medium_error = np.max(np.abs(medium_quantity - fine_quantity) / scale)
        assert coarse_error < 5e-3
        assert medium_error < coarse_error


@pytest.mark.parametrize("wavelength", [180.0, 1800.0])
def test_minoura_wavelength_domain_fails_closed(wavelength: float) -> None:
    with pytest.raises(ValueError, match="outside"):
        minoura_nk(np.array([wavelength]), 0.2, 0.9)


def _carron_connections(ggi: float, cgi: float) -> tuple[float, float]:
    theta = 0.2076
    q_comp = ggi * math.sin(theta) + cgi * math.cos(theta)
    amplitude = (
        80311.0 * q_comp
        + 427633.0 * (1.0 - q_comp)
        - 596825.0 * q_comp * (1.0 - q_comp)
    )
    gap = 1.004 * (1.0 - ggi) + 1.663 * ggi - 0.033 * ggi * (1.0 - ggi)
    urbach = 0.025
    connection_1 = 0.25 * (
        2.0 * gap
        - urbach
        + math.sqrt(4.0 * gap**2 + 12.0 * gap * urbach + urbach**2)
    )
    exponent = 5
    high_amplitude = 1.8e3 * cgi * (0.5 + 0.5 * ggi)
    connection_2 = gap + (
        (2.0**exponent) * (exponent**exponent) * high_amplitude / amplitude
    ) ** (-2.0 / (2.0 * exponent - 1.0))
    return connection_1, connection_2


@pytest.mark.parametrize("connection_index", [0, 1])
def test_carron_piecewise_absorption_is_c1_continuous(
    connection_index: int,
) -> None:
    ggi, cgi = 0.30, 0.90
    connection = _carron_connections(ggi, cgi)[connection_index]
    h = 1e-6
    energy = np.array([connection - 2 * h, connection - h, connection, connection + h, connection + 2 * h])
    alpha = carron_absorption_coefficient(energy, ggi, cgi)
    left_derivative = (alpha[2] - alpha[1]) / h
    right_derivative = (alpha[3] - alpha[2]) / h
    assert alpha[2] > 0.0
    assert abs(left_derivative - right_derivative) / max(
        abs(left_derivative), abs(right_derivative)
    ) < 5e-4


def test_carron_absorption_has_expected_device_scale() -> None:
    alpha = carron_absorption_coefficient(
        np.array([1.0, 1.2, 1.5, 2.0]), 0.25, 0.90
    )
    assert alpha[0] < 1e4
    assert 5e5 < alpha[1] < 2e6
    assert 2e6 < alpha[2] < 4e6
    assert 5e6 < alpha[3] < 8e6
    assert np.all(np.diff(alpha) > 0.0)


def test_carron_gap_matches_published_endpoint_law_and_rejects_bad_ggi() -> None:
    assert carron_band_gap_eV(0.0) == pytest.approx(1.004, abs=0.0)
    assert carron_band_gap_eV(1.0) == pytest.approx(1.663, abs=0.0)
    assert carron_band_gap_eV(0.6) == pytest.approx(1.39148, abs=1e-14)
    with pytest.raises(ValueError, match="finite and lie"):
        carron_band_gap_eV(float("nan"))


def test_ggi_profile_uses_shared_coordinate_and_flat_limit_is_exact() -> None:
    coordinate = np.array([0.0, 0.25, 0.75, 1.0])
    graded = CIGSGradedOptics(0.2, 0.6, 0.9)
    np.testing.assert_allclose(
        ggi_profile_from_coordinate(coordinate, graded),
        np.array([0.2, 0.3, 0.5, 0.6]),
        atol=1e-15,
        rtol=0.0,
    )
    flat = CIGSGradedOptics(0.3, 0.3, 0.9)
    np.testing.assert_array_equal(
        ggi_profile_from_coordinate(coordinate, flat),
        np.full_like(coordinate, 0.3),
    )
