"""Tests for spatial profile export (extract_spatial_snapshot)."""
import numpy as np
import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.models.spatial import SpatialSnapshot
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.experiments.jv_sweep import (
    extract_spatial_snapshot,
    run_jv_sweep,
)


@pytest.fixture
def setup():
    """Build grid, mat, and illuminated SS for nip_MAPbI3."""
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    N_grid = 40
    elec = electrical_layers(stack)
    layers_grid = [Layer(l.thickness, N_grid // len(elec)) for l in elec]
    x = multilayer_grid(layers_grid)
    mat = build_material_arrays(x, stack)
    y_ss = solve_illuminated_ss(x, stack, V_app=0.0)
    return stack, x, mat, y_ss


def test_returns_spatial_snapshot(setup):
    """extract_spatial_snapshot should return a SpatialSnapshot."""
    stack, x, mat, y_ss = setup
    snap = extract_spatial_snapshot(x, y_ss, stack, V_app=0.0, mat=mat)
    assert isinstance(snap, SpatialSnapshot)


def test_correct_shapes(setup):
    """Node arrays should have shape (N,) and face arrays shape (N-1,)."""
    stack, x, mat, y_ss = setup
    snap = extract_spatial_snapshot(x, y_ss, stack, V_app=0.0, mat=mat)
    N = len(x)
    assert snap.x.shape == (N,)
    assert snap.phi.shape == (N,)
    assert snap.n.shape == (N,)
    assert snap.p.shape == (N,)
    assert snap.P.shape == (N,)
    assert snap.rho.shape == (N,)
    assert snap.E.shape == (N - 1,)


def test_v_app_stored(setup):
    """V_app should be stored in the snapshot."""
    stack, x, mat, y_ss = setup
    snap = extract_spatial_snapshot(x, y_ss, stack, V_app=0.42, mat=mat)
    assert snap.V_app == 0.42


def test_carrier_densities_nonnegative(setup):
    """n and p should be non-negative (may be ~0 in transport layers)."""
    stack, x, mat, y_ss = setup
    snap = extract_spatial_snapshot(x, y_ss, stack, V_app=0.0, mat=mat)
    # Carrier densities can be near-zero in wide-gap transport layers;
    # allow small negative values from numerical noise (< 1e-20 m^-3).
    assert np.all(snap.n >= -1e-20), "Electron density should be non-negative"
    assert np.all(snap.p >= -1e-20), "Hole density should be non-negative"


def test_ion_density_nonnegative(setup):
    """Ion density P should be non-negative (zero in non-perovskite layers)."""
    stack, x, mat, y_ss = setup
    snap = extract_spatial_snapshot(x, y_ss, stack, V_app=0.0, mat=mat)
    # P is zero in transport layers (no ions) and > 0 in the perovskite
    assert np.all(snap.P >= 0), "Ion density should be non-negative"
    # At least some nodes should have ions (perovskite layer)
    assert np.any(snap.P > 0), "Expected non-zero ion density in perovskite"


def test_all_finite(setup):
    """All spatial fields should be finite."""
    stack, x, mat, y_ss = setup
    snap = extract_spatial_snapshot(x, y_ss, stack, V_app=0.0, mat=mat)
    assert np.all(np.isfinite(snap.x))
    assert np.all(np.isfinite(snap.phi))
    assert np.all(np.isfinite(snap.E))
    assert np.all(np.isfinite(snap.n))
    assert np.all(np.isfinite(snap.p))
    assert np.all(np.isfinite(snap.P))
    assert np.all(np.isfinite(snap.rho))


def test_phi_boundary_conditions(setup):
    """Potential should be 0 at left contact (grounded)."""
    stack, x, mat, y_ss = setup
    snap = extract_spatial_snapshot(x, y_ss, stack, V_app=0.0, mat=mat)
    assert abs(snap.phi[0]) < 1e-10, f"phi(0) = {snap.phi[0]}, expected 0"


def test_e_field_consistent_with_phi(setup):
    """E should be -(dphi/dx) at each face."""
    stack, x, mat, y_ss = setup
    snap = extract_spatial_snapshot(x, y_ss, stack, V_app=0.0, mat=mat)
    dx = np.diff(snap.x)
    E_expected = -(snap.phi[1:] - snap.phi[:-1]) / dx
    np.testing.assert_allclose(snap.E, E_expected, rtol=1e-10)


def test_snapshot_is_frozen(setup):
    """SpatialSnapshot should be immutable (frozen dataclass)."""
    stack, x, mat, y_ss = setup
    snap = extract_spatial_snapshot(x, y_ss, stack, V_app=0.0, mat=mat)
    with pytest.raises(AttributeError):
        snap.V_app = 0.5


def test_save_snapshots_in_jv_sweep():
    """run_jv_sweep with save_snapshots=True should populate snapshot fields."""
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    result = run_jv_sweep(
        stack, N_grid=40, n_points=10, v_rate=1.0, save_snapshots=True,
    )
    assert result.snapshots_fwd is not None
    assert result.snapshots_rev is not None
    assert len(result.snapshots_fwd) == 10
    assert len(result.snapshots_rev) == 10
    # Each snapshot should be a SpatialSnapshot
    assert isinstance(result.snapshots_fwd[0], SpatialSnapshot)


def test_save_snapshots_default_none():
    """By default, snapshot fields should be None."""
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    result = run_jv_sweep(
        stack, N_grid=40, n_points=10, v_rate=1.0,
    )
    assert result.snapshots_fwd is None
    assert result.snapshots_rev is None
