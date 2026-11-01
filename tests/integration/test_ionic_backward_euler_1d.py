"""Physical N/M BE secants, analytic derivatives and full interval accounting."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_multilayer import prepare_confined_ion_device
from examples.research_1d.ionic_residual import prepare_single_ion_device
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.solvers.time.ionic_backward_euler import (
    IonicBackwardEuler1DStep, finite_site_secant_m3, inventory_m2, step_accounting,
)

Q = 1.602176634e-19


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


def compile_case(confined):
    if confined:
        return compile_ionic_device_1d(
            prepare_confined_ion_device(), intervals_per_layer=(12, 12, 12),
            ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"),
        )
    return compile_ionic_device_1d(prepare_single_ion_device(diffusion_only=True), intervals_per_layer=12)


@pytest.fixture(scope="module", params=[False, True], ids=["single_layer", "confined37_13"])
def case(request):
    device = compile_case(request.param)
    N, M = device.node_count, device.ion_node_count
    old_y = device.initial_state(voltage_V=0.002).u.copy()
    old_y[N:2 * N] = 0.01 * np.sin(np.linspace(0.0, np.pi, N))
    old_y[3 * N:] = 0.15 * np.sin(np.linspace(-1.0, 1.0, M))
    old = device.state(old_y, voltage_V=0.002)
    step = IonicBackwardEuler1DStep(Ionic1DProblem(device, voltage_V=0.004, generation_scale=0.2), old, 0.05)
    y = step.initial_coordinates(voltage_lift=True)
    y[N:2 * N] += 0.015 * np.sin(np.linspace(0.0, np.pi, N))
    y[3 * N:] += 0.07 * np.cos(np.linspace(-1.0, 1.0, M))
    return device, step, y


def d(value):
    return Decimal.from_float(float(value))


@pytest.mark.parametrize("old,new", [
    (0.0, 1e-14), (0.4, np.nextafter(0.4, np.inf)), (0.4, 0.2),
    (-0.4, -0.2), (20.0, 20.01), (-20.0, -19.99),
])
def test_finite_site_secant_has_independent_100_digit_oracle(old, new):
    reference = np.log(0.001) - np.log1p(-0.001)
    actual = finite_site_secant_m3(np.array([1e27]), np.array([reference]), np.array([old]), np.array([new]))[0]
    with localcontext() as context:
        context.prec = 100
        before = Decimal(1) / (Decimal(1) + (-(d(reference) + d(old))).exp())
        after = Decimal(1) / (Decimal(1) + (-(d(reference) + d(new))).exp())
        expected = float(d(1e27) * (after - before))
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0.0)
    assert actual != 0.0


def test_all_M_balances_use_physical_secants_and_fixed_contacts_remain_algebraic(case):
    device, step, y = case
    N = device.node_count
    carrier_rate, ion_rate = step.inventory_rate_m2_s(y)
    dc, actual = step.endpoint.residual(y), step.residual(y)
    np.testing.assert_allclose(actual[3 * N:], dc[3 * N:] - ion_rate, rtol=0.0, atol=0.0)
    mask = np.ones((2, N), dtype=bool)
    mask[:, [0, -1]] = False
    expected = dc[N:3 * N].reshape(2, N).copy()
    expected[mask] -= carrier_rate[mask]
    np.testing.assert_array_equal(actual[N:3 * N].reshape(2, N), expected)
    np.testing.assert_array_equal(actual[:N], dc[:N])
    assert ion_rate.shape == (device.ion_node_count,)
    assert np.max(abs(ion_rate)) > 0.0
    assert actual.shape == (device.state_size,)


def test_full_BE_jacobian_has_independent_physical_directional_check(case):
    device, step, y = case
    rng = np.random.default_rng(40608)
    direction = rng.uniform(-0.05, 0.05, device.state_size)
    direction[:device.node_count] *= device.electronic_component.thermal_voltage_V
    J = step.jacobian(y)
    # Remove only the state-independent generation term before subtracting
    # residuals: tiny minority-carrier derivatives cannot be read from 1e19
    # source totals in binary64. The tested illuminated Jacobian is identical.
    conditioned = IonicBackwardEuler1DStep(replace(step.endpoint, generation_scale=0.0), step.previous, step.dt_s)
    np.testing.assert_array_equal(J.toarray(), conditioned.jacobian(y).toarray())
    h = 1e-5
    numerical = (conditioned.residual(y + h * direction) - conditioned.residual(y - h * direction)) / (2 * h)
    scale = abs(J) @ abs(direction)
    error = float(np.max(abs(J @ direction - numerical) / np.maximum(scale, np.finfo(float).tiny)))
    print(f"N={device.node_count},M={device.ion_node_count}: BE directional error {error:.9g}")
    assert error < 2e-7
    expected = step.endpoint.jacobian(y) + step.endpoint.storage_jacobian(y) / step.dt_s
    np.testing.assert_array_equal(J.toarray(), expected.toarray())


def test_explicit_carrier_reference_rebase_preserves_populations_without_extra_drive(case):
    device, step, _ = case
    y = step.initial_coordinates(voltage_lift=False)
    new = device.state(y, voltage_V=step.endpoint.voltage_V)
    np.testing.assert_allclose(new.electronic_state.electron_density_m3, step.previous.electronic_state.electron_density_m3, rtol=5e-15)
    np.testing.assert_allclose(new.electronic_state.hole_density_m3, step.previous.electronic_state.hole_density_m3, rtol=5e-15)
    carrier_rate, ion_rate = step.inventory_rate_m2_s(y)
    old_carrier, _ = inventory_m2(device, step.previous)
    assert np.max(abs(carrier_rate * step.dt_s) / old_carrier) < 64 * np.finfo(float).eps
    np.testing.assert_array_equal(ion_rate, np.zeros(device.ion_node_count))


def test_motion_accounting_has_full_charge_current_and_signed_readback_identity(case):
    device, step, y = case
    info = step_accounting(step, y)
    N = device.node_count
    assert np.asarray(info["total_current_density_A_m2"]).shape == (N + 1,)
    assert np.max(abs(info["ion_domain_fluxes_m2_s"])) > 0.0
    carriers, ions = step.inventory_rate_m2_s(y)
    charge_rate = Q * (carriers[1] - carriers[0])
    charge_rate[device.ion_node_indices] += Q * ions
    np.testing.assert_array_equal(info["charge_inventory_rate_A_m2"], charge_rate)
    np.testing.assert_allclose(info["conduction_current_density_A_m2"],
                               info["carrier_conduction_current_density_A_m2"] + info["ion_conduction_current_density_A_m2"], rtol=0.0, atol=0.0)
    scale = max(1.0, float(np.max(abs(info["total_current_density_A_m2"]))),
                float(np.max(abs(info["predicted_maxwell_cell_jump_A_m2"]))))
    assert np.max(abs(info["maxwell_identity_error_A_m2"])) <= 128 * np.finfo(float).eps * scale
    assert np.max(abs(info["readback_maxwell_identity_error_A_m2"])) <= 256 * np.finfo(float).eps * scale
    np.testing.assert_allclose(info["rounded_interval_displacement_C_m2"] + info["displacement_readback_error_C_m2"],
                               info["interval_displacement_C_m2"], rtol=2e-15, atol=1e-20)
    np.testing.assert_array_equal(info["backward_euler_residual"], step.residual(y))
    assert info["physical_certificate_status"] == "not_evaluated"


def test_nonzero_old_coordinate_tiny_motion_is_not_replaced_by_density_subtraction():
    device = compile_case(False)
    old_y = device.initial_state().u.copy()
    old_y[3 * device.node_count:] = 0.4
    old = device.state(old_y)
    step = IonicBackwardEuler1DStep(Ionic1DProblem(device), old, 0.25)
    y = old_y.copy()
    y[3 * device.node_count:] = np.nextafter(0.4, np.inf)
    _, rate = step.inventory_rate_m2_s(y)
    rounded = (inventory_m2(device, device.state(y))[1] - inventory_m2(device, old)[1]) / step.dt_s
    assert np.all(rate > 0.0)
    assert np.any(rate != rounded)
    info = step_accounting(step, y)
    np.testing.assert_array_equal(info["ion_inventory_readback_error_m2_s"], rate - rounded)


def test_missing_foreign_or_inconsistent_previous_states_and_wrong_shapes_refuse(case):
    device, step, y = case
    with pytest.raises(ValueError, match="full ionic"):
        IonicBackwardEuler1DStep(step.endpoint, step.previous.electronic_state, 0.1)
    with pytest.raises(ValueError, match="same complete"):
        IonicBackwardEuler1DStep(step.endpoint, replace(step.previous, device_sha256="foreign"), 0.1)
    with pytest.raises(ValueError, match="fields/inventory"):
        IonicBackwardEuler1DStep(step.endpoint, replace(step.previous, positive_ion_density_m3=step.previous.positive_ion_density_m3 * 1.01), 0.1)
    with pytest.raises(ValueError, match="positive"):
        IonicBackwardEuler1DStep(step.endpoint, step.previous, 0.0)
    with pytest.raises(ValueError, match="3.N.*M"):
        step.residual(np.r_[y, 0.0])
