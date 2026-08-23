"""Generalized Scharfetter-Gummel transport tests."""

from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.discretization.fe_operators import sg_fluxes_n, sg_fluxes_p
from perovskite_sim.physics.degenerate_transport import (
    generalized_carrier_face_statistics,
    generalized_sg_fluxes_n,
    generalized_sg_fluxes_p,
)
from perovskite_sim.physics.statistics import (
    FERMI_DIRAC,
    MAXWELL_BOLTZMANN,
    carrier_density_from_reduced_fermi_level,
)
from perovskite_sim.physics.temperature import thermal_voltage


def test_generalized_mb_flux_is_exactly_the_classical_sg_flux():
    potential = np.asarray([0.0, 0.017, -0.008, 0.025])
    density_n = np.asarray([1.0e22, 3.0e22, 8.0e21, 7.0e22])
    density_p = np.asarray([5.0e21, 9.0e22, 4.0e22, 2.0e21])
    spacing = np.asarray([2.0e-9, 3.0e-9, 4.0e-9])
    mobility_n = np.asarray([0.1, 0.2, 0.15])
    mobility_p = np.asarray([0.04, 0.03, 0.05])
    thermal = thermal_voltage(300.0)

    np.testing.assert_array_equal(
        generalized_sg_fluxes_n(
            potential,
            density_n,
            spacing,
            mobility_n,
            thermal,
            2.8e25,
            statistics=MAXWELL_BOLTZMANN,
        ),
        sg_fluxes_n(
            potential,
            density_n,
            spacing,
            mobility_n * thermal,
            thermal,
        ),
    )
    np.testing.assert_array_equal(
        generalized_sg_fluxes_p(
            potential,
            density_p,
            spacing,
            mobility_p,
            thermal,
            1.04e25,
            statistics=MAXWELL_BOLTZMANN,
        ),
        sg_fluxes_p(
            potential,
            density_p,
            spacing,
            mobility_p * thermal,
            thermal,
        ),
    )


def test_fd_face_factor_is_positive_and_satisfies_the_secant_identity():
    dos = 2.8e25
    eta = np.asarray([-8.0, -1.0, 2.0, 8.0])
    density = np.asarray(
        [
            carrier_density_from_reduced_fermi_level(
                value,
                dos,
                statistics=FERMI_DIRAC,
            )
            for value in eta
        ]
    )
    face = generalized_carrier_face_statistics(
        density,
        dos,
        statistics=FERMI_DIRAC,
    )

    assert np.all(face.diffusion_enhancement > 0.0)
    assert face.diffusion_enhancement[-1] > 2.0
    np.testing.assert_allclose(
        face.diffusion_enhancement * np.diff(face.log_occupation),
        np.diff(face.reduced_fermi_level),
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_fd_flux_vanishes_for_constant_quasi_fermi_potentials():
    thermal = thermal_voltage(300.0)
    potential = np.asarray([-0.03, -0.005, 0.017, 0.041])
    spacing = np.asarray([2.0e-9, 3.0e-9, 4.0e-9])
    electron_dos = 2.8e25
    hole_dos = 1.04e25
    eta_n = -1.7 + potential / thermal
    eta_p = 0.8 - potential / thermal
    density_n = np.asarray(
        [
            carrier_density_from_reduced_fermi_level(
                value,
                electron_dos,
                statistics=FERMI_DIRAC,
            )
            for value in eta_n
        ]
    )
    density_p = np.asarray(
        [
            carrier_density_from_reduced_fermi_level(
                value,
                hole_dos,
                statistics=FERMI_DIRAC,
            )
            for value in eta_p
        ]
    )

    current_n = generalized_sg_fluxes_n(
        potential,
        density_n,
        spacing,
        0.135,
        thermal,
        electron_dos,
        statistics=FERMI_DIRAC,
    )
    current_p = generalized_sg_fluxes_p(
        potential,
        density_p,
        spacing,
        0.048,
        thermal,
        hole_dos,
        statistics=FERMI_DIRAC,
    )
    current_scale_n = 1.602176634e-19 * 0.135 * thermal * max(density_n) / min(spacing)
    current_scale_p = 1.602176634e-19 * 0.048 * thermal * max(density_p) / min(spacing)
    assert np.max(np.abs(current_n)) / current_scale_n < 2.0e-13
    assert np.max(np.abs(current_p)) / current_scale_p < 2.0e-13


def test_equal_fd_density_uses_the_local_generalized_einstein_limit():
    density = np.full(3, 20.0 * 2.8e25)
    face = generalized_carrier_face_statistics(
        density,
        2.8e25,
        statistics=FERMI_DIRAC,
    )
    assert np.all(face.diffusion_enhancement > 1.0)
    assert face.diffusion_enhancement[0] == pytest.approx(
        face.diffusion_enhancement[1]
    )


@pytest.mark.parametrize(
    "density",
    (
        np.asarray([1.0, 0.0]),
        np.asarray([1.0, -1.0]),
        np.asarray([1.0, np.nan]),
    ),
)
def test_generalized_flux_rejects_nonpositive_or_nonfinite_density(density):
    with pytest.raises(ValueError, match="carrier density"):
        generalized_sg_fluxes_n(
            np.zeros(2),
            density,
            np.ones(1),
            1.0,
            0.025,
            1.0,
            statistics=FERMI_DIRAC,
        )
