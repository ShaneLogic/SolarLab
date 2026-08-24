from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.physics.ion_migration import ion_face_flux
from perovskite_sim.twod.ion_migration_2d import (
    assess_mobile_ion_terminal_2d,
    control_volume_areas_2d,
    ion_inventory_2d,
    positive_ion_continuity_rhs_2d,
    positive_ion_fluxes_2d,
)


def _fields() -> tuple[np.ndarray, ...]:
    x = np.array([0.0, 0.12e-6, 0.47e-6, 1.0e-6])
    y = np.array([0.0, 0.08e-6, 0.31e-6, 0.55e-6])
    yy, xx = np.meshgrid(y / y[-1], x / x[-1], indexing="ij")
    phi = 0.03 * xx - 0.05 * yy + 0.004 * xx * yy
    density = 1.0e22 * (1.0 + 0.17 * xx + 0.11 * yy)
    diffusion = 1.0e-16 * (1.0 + 0.4 * xx + 0.2 * yy)
    site_limit = 1.0e24 * (1.0 + 0.1 * yy)
    return x, y, phi, density, diffusion, site_limit


def _harmonic(values: np.ndarray) -> np.ndarray:
    return 2.0 * values[:-1] * values[1:] / (values[:-1] + values[1:])


def test_control_volume_areas_cover_nonuniform_domain_exactly():
    x, y, *_ = _fields()
    areas = control_volume_areas_2d(x, y)

    assert areas.shape == (y.size, x.size)
    assert np.sum(areas) == pytest.approx(x[-1] * y[-1], rel=2e-16)
    assert areas[0, 0] == pytest.approx(0.25 * x[1] * y[1])


@pytest.mark.parametrize("steric_diffusion_only", [False, True])
def test_vertical_flux_matches_1d_single_source_of_truth(
    steric_diffusion_only: bool,
):
    x, y, phi, density, diffusion, site_limit = _fields()
    fluxes = positive_ion_fluxes_2d(
        x,
        y,
        phi,
        density,
        diffusion,
        0.025852,
        site_limit,
        steric_diffusion_only=steric_diffusion_only,
    )

    for column in range(x.size):
        expected = ion_face_flux(
            phi[:, column],
            density[:, column],
            np.diff(y),
            _harmonic(diffusion[:, column]),
            0.025852,
            0.5 * (
                site_limit[:-1, column] + site_limit[1:, column]
            ),
            steric_diffusion_only=steric_diffusion_only,
            P_lim_node=site_limit[:, column],
        )
        np.testing.assert_allclose(
            fluxes.y[:, column],
            expected,
            rtol=2e-15,
            atol=0.0,
        )


@pytest.mark.parametrize("steric_diffusion_only", [False, True])
def test_blocking_continuity_conserves_exact_discrete_inventory(
    steric_diffusion_only: bool,
):
    x, y, phi, density, diffusion, site_limit = _fields()
    derivative = positive_ion_continuity_rhs_2d(
        x,
        y,
        phi,
        density,
        diffusion,
        0.025852,
        site_limit,
        lateral_bc="neumann",
        steric_diffusion_only=steric_diffusion_only,
    )
    weighted = derivative * control_volume_areas_2d(x, y)
    cancellation_scale = max(float(np.sum(np.abs(weighted))), 1.0)

    assert abs(float(np.sum(weighted))) / cancellation_scale < 5e-15


def test_mobile_ion_continuity_rejects_periodic_duplicate_endpoint_topology():
    x, y, phi, density, diffusion, site_limit = _fields()
    with pytest.raises(ValueError, match="not topology-certified"):
        positive_ion_continuity_rhs_2d(
            x,
            y,
            phi,
            density,
            diffusion,
            0.025852,
            site_limit,
            lateral_bc="periodic",
        )


def test_inventory_uses_half_endpoint_2d_control_volumes():
    x = np.array([0.0, 0.2, 1.0])
    y = np.array([0.0, 0.4, 1.0])
    density = np.ones((3, 3))
    assert ion_inventory_2d(x, y, density) == pytest.approx(1.0)


def test_terminal_report_accepts_physical_conservative_state():
    x, y, _phi, density, _diffusion, site_limit = _fields()
    report = assess_mobile_ion_terminal_2d(
        x,
        y,
        density,
        density.copy(),
        site_limit,
        terminal_electron_density=np.full_like(density, 2.0e15),
        terminal_hole_density=np.full_like(density, 3.0e15),
        inventory_rtol=1.0e-12,
    )

    assert report.passed is True
    assert report.violations == ()
    assert report.relative_inventory_drift == 0.0
    assert report.terminal_min_electron_density_m3 == 2.0e15
    assert report.terminal_min_hole_density_m3 == 3.0e15


@pytest.mark.parametrize(
    ("field", "violation"),
    [
        ("electron", "negative_terminal_electron_density"),
        ("hole", "negative_terminal_hole_density"),
        ("ion", "negative_terminal_density"),
        ("site", "terminal_site_limit_exceeded"),
        ("inventory", "inventory_drift_exceeded"),
    ],
)
def test_terminal_report_fails_closed_on_each_physical_gate(
    field: str,
    violation: str,
):
    x, y, _phi, density, _diffusion, site_limit = _fields()
    electrons = np.full_like(density, 2.0e15)
    holes = np.full_like(density, 3.0e15)
    terminal = density.copy()
    if field == "electron":
        electrons[1, 1] = -1.0
    elif field == "hole":
        holes[1, 1] = -1.0
    elif field == "ion":
        terminal[1, 1] = -1.0
    elif field == "site":
        terminal[1, 1] = 1.01 * site_limit[1, 1]
    else:
        terminal *= 1.01

    report = assess_mobile_ion_terminal_2d(
        x,
        y,
        density,
        terminal,
        site_limit,
        terminal_electron_density=electrons,
        terminal_hole_density=holes,
        inventory_rtol=1.0e-12,
    )

    assert report.passed is False
    assert violation in report.violations
