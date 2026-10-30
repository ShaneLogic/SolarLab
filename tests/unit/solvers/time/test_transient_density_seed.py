"""Closed-form physical mass roots for the unwired transient seed proposal."""

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.sparse import diags

from solarlab.solvers.time.carrier_seed import transient_density_basin_seed


@dataclass(frozen=True)
class DensityDevice:
    node_count: int = 2

    def fields(self, u, *, voltage_V=0.0):
        n = self.node_count
        return u[:n], np.exp(u[n:2 * n]), np.exp(-u[2 * n:])


@dataclass(frozen=True)
class PhysicalMassStep:
    device: DensityDevice
    old: np.ndarray
    older: np.ndarray
    coefficients: np.ndarray
    dt: float
    generation: np.ndarray
    phi: np.ndarray

    def residual(self, u):
        phi, n, p = self.device.fields(u)
        populations = np.stack((n, p))
        rate = (self.coefficients[0] * populations + self.coefficients[1] * self.old
                + self.coefficients[2] * self.older) / self.dt
        return np.r_[phi - self.phi, (self.generation - rate).ravel()]

    def jacobian(self, u):
        _, n, p = self.device.fields(u)
        return diags(np.r_[np.ones(self.device.node_count),
                           -self.coefficients[0] * n / self.dt,
                           self.coefficients[0] * p / self.dt], format="csc")


def problem(order, dt=1e-7, generation=1.0):
    device = DensityDevice()
    old = np.array([[1e-20, 2e-18], [4e-21, 5e-19]])
    older = .5 * old
    phi = np.array([-.1, .2])
    u = np.r_[phi, np.log(old[0]), -np.log(old[1])]
    coefficients = np.array([1.0, -1.0, 0.0]) if order == 1 else np.array([1.5, -2.0, .5])
    return PhysicalMassStep(device, old, older, coefficients, dt,
                            generation * np.array([2e7, 1e6]), phi), u


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("dt", [1e-7, 2.5e-8])
def test_seed_uses_complete_BE_or_BDF2_mass_and_preserves_phi_and_old_history(order, dt):
    target, u = problem(order, dt)
    before = u.tobytes(), target.old.tobytes(), target.older.tobytes()
    seed = transient_density_basin_seed(target, u, device=target.device, voltage_V=0.0)
    phi, n, p = target.device.fields(seed)
    expected = (dt * target.generation - target.coefficients[1] * target.old
                - target.coefficients[2] * target.older) / target.coefficients[0]
    np.testing.assert_allclose(np.stack((n, p)), expected, rtol=64 * np.finfo(float).eps, atol=0.0)
    assert phi.tobytes() == u[:target.device.node_count].tobytes()
    assert before == (u.tobytes(), target.old.tobytes(), target.older.tobytes())
    assert np.max(abs(target.residual(seed)[target.device.node_count:])) <= (
        64 * np.finfo(float).eps * np.max(abs(target.generation)))


def test_nonpositive_density_candidate_is_rejected_without_clipping_or_fallback():
    target, u = problem(1, generation=-1.0)
    before = u.tobytes()
    with pytest.raises(ValueError, match="finite positive carriers"):
        transient_density_basin_seed(target, u, device=target.device, voltage_V=0.0)
    assert u.tobytes() == before


def test_non_electronic_state_shape_is_rejected():
    target, u = problem(1)
    with pytest.raises(ValueError, match="electronic 3N"):
        transient_density_basin_seed(target, np.r_[u, 0.0], device=target.device, voltage_V=0.0)
