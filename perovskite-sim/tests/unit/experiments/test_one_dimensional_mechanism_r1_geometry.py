"""Independent GEO-01..06 oracles for the R1 physical-volume lane."""

from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.experiments.one_dimensional_mechanism_r1 import build_r1_material
from perovskite_sim.experiments.defect_ion_combined_impedance import _ion_fields
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.ion_migration import ion_face_flux_jacobian
from perovskite_sim.physics.physical_control_volume import (
    flux_divergence, physical_cell_faces, physical_contact_displacement,
)
from perovskite_sim.physics.poisson import (
    factor_poisson_from_finite_volume, solve_poisson_prefactored,
)


FIXTURE = Path(__file__).parents[2] / "fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"


@pytest.mark.parametrize("intervals", [4, 16, 32, 64])
def test_geo_01_02_geometry_inventory(intervals):
    stack = load_device_from_yaml(FIXTURE)
    x, mat = build_r1_material(stack, intervals)
    w = mat.dx_cell
    assert np.all(w > 0)
    np.testing.assert_allclose(np.cumsum(w), mat.physical_cell_faces_m[1:], atol=1e-22)
    for mask in [x < 1e-7, x > 1e-7]:
        assert abs(w[mask].sum() / 1e-7 - 1) <= 1e-12
        assert abs(np.dot(w[mask], mat.P_ion0[mask]) / 1e15 - 1) <= 1e-12
    if intervals == 4:
        np.testing.assert_allclose(w, np.array([25, 75, 75, 25]) * 1e-9)
    assert not mat.physical_cell_faces_m.flags.writeable


@pytest.mark.parametrize("intervals", [4, 16, 32, 64])
def test_geo_03_flux_telescopes_and_rate_uses_same_volumes(intervals):
    x, mat = build_r1_material(load_device_from_yaml(FIXTURE), intervals)
    rng = np.random.default_rng(104)
    flux = rng.normal(size=x.size-1)
    flux[x[:-1] < 1e-7] *= 1e5
    for left, right in [(0., 0.), (1.5, -2.)]:
        rate = flux_divergence(flux, mat.dx_cell, left_flux=left, right_flux=right)
        assert abs(np.dot(rate, mat.dx_cell) - (left-right)) < 1e-9
    positive = mat.P_ion0 * (1 + 1e-3 * rng.normal(size=x.size))
    rate, _, flux, _ = _ion_fields(x, mat, positive, None, np.zeros_like(x))
    np.testing.assert_array_equal(rate, flux_divergence(flux, mat.dx_cell))
    assert abs(np.dot(rate, mat.dx_cell)) < 1e-14 * max(abs(flux))


def test_geo_04_half_occupied_diffusion_cosine_decay():
    errors = []
    for intervals in [16, 32, 64]:
        x = np.linspace(0, 1e-7, intervals + 1)
        w = np.diff(physical_cell_faces(x))
        derivative = ion_face_flux_jacobian(
            np.zeros_like(x), np.full_like(x, 1e22), np.diff(x),
            np.full(intervals, 1e-14), .02585, 2e22,
            steric_diffusion_only=True, P_lim_node=2e22,
        )
        np.testing.assert_allclose(derivative.density_left_derivative, 2e-14/np.diff(x))
        np.testing.assert_allclose(derivative.density_right_derivative, -2e-14/np.diff(x))
        flux_matrix = np.zeros((intervals, intervals+1))
        flux_matrix[np.arange(intervals), np.arange(intervals)] = derivative.density_left_derivative
        flux_matrix[np.arange(intervals), np.arange(intervals)+1] = derivative.density_right_derivative
        operator = np.column_stack([flux_divergence(column, w) for column in flux_matrix.T])
        mode = np.cos(np.pi*x/x[-1])
        t = .01
        expected = mode*np.exp(-2e-14*(np.pi/x[-1])**2*t)
        errors.append(np.max(abs(expm(t*operator)@mode - expected)))
    assert errors[1] < errors[0]/3.9
    assert errors[2] < errors[1]/3.9
    assert errors[-1] < 4e-5


@pytest.mark.parametrize("intervals", [4, 16, 32, 64])
def test_geo_05_constant_charge_poisson(intervals):
    x, mat = build_r1_material(load_device_from_yaml(FIXTURE), intervals)
    rho = np.full(x.size, 13.)
    eps = EPS_0*10
    phi = solve_poisson_prefactored(mat.poisson_factor, rho, 0., .02)
    exact = .02*x/x[-1] + rho*x*(x[-1]-x)/(2*eps)
    np.testing.assert_allclose(phi, exact, atol=1e-15)
    contact = physical_contact_displacement(-mat.poisson_factor.C*np.diff(phi), rho, mat.dx_cell)
    expected = np.array([-eps*.02/x[-1]-13*x[-1]/2, -eps*.02/x[-1]+13*x[-1]/2])
    np.testing.assert_allclose(contact, expected, atol=1e-18)
    assert abs(np.diff(contact)[0] - np.dot(rho, mat.dx_cell)) <= Q*1e15*1e-10


@pytest.mark.parametrize("intervals", [4, 16, 32, 64])
def test_geo_06_dielectric_sheet_and_series_capacitance(intervals):
    x, mat = build_r1_material(load_device_from_yaml(FIXTURE), intervals)
    face = mat.iface_qss_interface_faces[0]
    a, L = 1e-7, 2e-7
    eps = np.where(x < a, 10., 30.)*EPS_0
    capacitance = 1/(np.diff(x)/eps[:-1])
    cl, cr = eps[face]/(a-x[face]), eps[face+1]/(x[face+1]-a)
    capacitance[face] = cl*cr/(cl+cr)
    factor = factor_poisson_from_finite_volume(capacitance, mat.dx_cell[1:-1])
    sigma, voltage = 1e-5, .02
    allocated = np.zeros(x.size)
    allocated[face] = sigma*cl/(cl+cr)/mat.dx_cell[face]
    allocated[face+1] = sigma*cr/(cl+cr)/mat.dx_cell[face+1]
    phi = solve_poisson_prefactored(factor, allocated, 0., voltage)
    trace = (cl*phi[face]+cr*phi[face+1]+sigma)/(cl+cr)
    dl, dr = -cl*(trace-phi[face]), cr*(trace-phi[face+1])
    assert abs(dr-dl-sigma) <= Q*1e15*1e-10
    c = 1/(a/eps[0]+(L-a)/eps[-1])
    expected_dl = -c*(voltage+sigma*(L-a)/eps[-1])
    assert abs(dl-expected_dl) < 1e-18
    assert abs(trace-(-dl*a/eps[0])) < 1e-15
    perturbed = solve_poisson_prefactored(factor, allocated, 0., voltage+.001)
    measured = -capacitance[0]*(np.diff(perturbed)[0]-np.diff(phi)[0])/.001
    assert abs(measured/(-c)-1) < 1e-11


def test_geo_05_spatial_convergence_nonquadratic_charge():
    errors = []
    for n in [16, 32, 64]:
        x = np.linspace(0, 2e-7, n+1)
        w = np.diff(physical_cell_faces(x))
        factor = factor_poisson_from_finite_volume(EPS_0*10/np.diff(x), w[1:-1])
        exact = .01*np.sin(np.pi*x/x[-1])
        rho = EPS_0*10*(np.pi/x[-1])**2*exact
        numerical = solve_poisson_prefactored(factor, rho, 0., 0.)
        errors.append(max(abs(numerical-exact)))
    assert errors[1] < errors[0]/3.9
    assert errors[2] < errors[1]/3.9


@pytest.mark.parametrize("x,planes", [
    ([0, 1, 1], []), ([0, 1, 2], [1]), ([0, 1, 2], [-1]),
    ([0, 1, 2], [.2, .3]), ([0, 1, 2], [float("nan")]),
])
def test_invalid_physical_geometry_fails_closed(x, planes):
    with pytest.raises(ValueError):
        physical_cell_faces(x, planes)
