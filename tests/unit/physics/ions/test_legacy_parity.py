"""Bitwise M evidence against the immutable implemented ion source."""

from __future__ import annotations

from dataclasses import fields
import inspect

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from perovskite_sim.physics import ion_migration as old
from solarlab.physics.ions import migration as new

V_T = 0.025852


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


def same(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("count", [2, 13, 73])
@pytest.mark.parametrize("diffusion_only", [False, True])
@pytest.mark.parametrize("array_parameters", [False, True])
@pytest.mark.parametrize("partner_kind", ["none", "zero", "shared"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_normal_single_dual_flux_rates_and_existing_jacobian_are_bitwise(
    count, diffusion_only, array_parameters, partner_kind, sign,
):
    rng = np.random.default_rng(20261028 + count)
    x = np.r_[0.0, np.cumsum(rng.uniform(1e-10, 2e-8, count - 1))]
    dx = np.diff(x)
    phi = np.cumsum(rng.uniform(-0.05, 0.05, count))
    density = rng.uniform(1e24, 3e26, count)
    limit = rng.uniform(1e27, 2e27, count) if array_parameters else 1.6e27
    face_limit = 0.5 * (limit[:-1] + limit[1:]) if array_parameters else limit
    diffusion = rng.uniform(1e-18, 2e-16, count - 1) if array_parameters else 1e-16
    partner = {
        "none": None,
        "zero": np.zeros(count),
        "shared": rng.uniform(1e24, 3e26, count),
    }[partner_kind]
    kwargs = dict(
        steric_diffusion_only=diffusion_only, P_lim_node=limit,
        P_other_node=partner, drift_sign=sign,
    )
    inputs = [np.array(value, copy=True) for value in (x, phi, density)]
    actual = new.ion_face_flux(phi, density, dx, diffusion, V_T, face_limit, **kwargs)
    same(actual, old.ion_face_flux(phi, density, dx, diffusion, V_T, face_limit, **kwargs))
    actual_j = new.ion_face_flux_jacobian(phi, density, dx, diffusion, V_T, face_limit, **kwargs)
    expected_j = old.ion_face_flux_jacobian(phi, density, dx, diffusion, V_T, face_limit, **kwargs)
    for field in fields(actual_j):
        same(getattr(actual_j, field.name), getattr(expected_j, field.name))
    widths = np.r_[dx[0] / 2, 0.5 * (dx[:-1] + dx[1:]), dx[-1] / 2]
    for current, original in (
        (new.ion_continuity_rhs, old.ion_continuity_rhs),
        (new.ion_continuity_rhs_neg, old.ion_continuity_rhs_neg),
    ):
        for explicit_widths in (None, widths, widths * 1.25):
            options = {key: value for key, value in kwargs.items() if key != "drift_sign"}
            options["cell_widths_m"] = explicit_widths
            same(
                current(x, phi, density, diffusion, V_T, face_limit, **options),
                original(x, phi, density, diffusion, V_T, face_limit, **options),
            )
    for value, before in zip((x, phi, density), inputs):
        same(value, before)


@pytest.mark.parametrize("occupancy", [
    -0.1, 0.0, np.nextafter(0.0, 1.0), 0.5,
    np.nextafter(0.999999, 0.0), 0.999999,
    np.nextafter(0.999999, 1.0), 1.0, 1.3,
])
@pytest.mark.parametrize("diffusion_only", [False, True])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_clips_floors_and_differentiability_masks_remain_bitwise(occupancy, diffusion_only, sign):
    density = np.full(3, occupancy)
    phi = np.array([0.0, 0.03, -0.01])
    kwargs = dict(
        steric_diffusion_only=diffusion_only, P_lim_node=1.0,
        P_other_node=np.zeros(3), drift_sign=sign,
    )
    arguments = (phi, density, np.array([1e-9, 2e-9]), np.array([0.0, 1e-16]), V_T, 1.0)
    same(new.ion_face_flux(*arguments, **kwargs), old.ion_face_flux(*arguments, **kwargs))
    actual, expected = (
        model.ion_face_flux_jacobian(*arguments, **kwargs) for model in (new, old)
    )
    for field in fields(actual):
        same(getattr(actual, field.name), getattr(expected, field.name))


@pytest.mark.parametrize("xi", [
    -800.0, -701.0, -700.0, -1e-8, -1e-10, 0.0,
    1e-10, 1e-8, 700.0, 701.0, 800.0,
])
def test_scalar_flux_and_historical_bernoulli_branches_are_bitwise(xi):
    phi = np.array([0.0, xi * V_T])
    density = np.array([1e24, 3e24])
    same(
        new.ion_flux_steric(phi, density, 1e-8, 1e-16, V_T, 1.6e27),
        old.ion_flux_steric(phi, density, 1e-8, 1e-16, V_T, 1.6e27),
    )
    for sign in (-1.0, 1.0):
        same(
            new.ion_face_flux(phi, density, np.array([1e-8]), 1e-16, V_T, 1.6e27, drift_sign=sign),
            old.ion_face_flux(phi, density, np.array([1e-8]), 1e-16, V_T, 1.6e27, drift_sign=sign),
        )


@pytest.mark.parametrize("diffusion_only", [False, True])
def test_node_limit_fallback_and_historically_ignored_arguments_are_preserved(diffusion_only):
    phi = np.array([0.0, 0.01, 0.05])
    density = np.array([1e24, 2e24, 4e24])
    args = (phi, density, np.array([1e-8, 2e-8]), 1e-16, V_T, 1.6e27)
    kwargs = dict(steric_diffusion_only=diffusion_only)
    same(new.ion_face_flux(*args, **kwargs), old.ion_face_flux(*args, **kwargs))
    if not diffusion_only:
        ignored = dict(P_lim_node=np.array([1.0]), P_other_node=np.array([9.0]))
        same(new.ion_face_flux(*args, **ignored), old.ion_face_flux(*args, **ignored))
    else:
        same(
            new.ion_face_flux(*args, P_lim_node=1.6e27, **kwargs),
            new.ion_face_flux(*args, **kwargs),
        )


@pytest.mark.parametrize("diffusion_only", [False, True])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_float32_inputs_retain_historical_promotion_and_bytes(diffusion_only, sign):
    phi = np.array([0.0, 0.01, -0.05], dtype=np.float32)
    density = np.array([1e24, 2e24, 3e24], dtype=np.float32)
    dx = np.array([1e-9, 2e-9], dtype=np.float32)
    diffusion = np.array([1e-16, 2e-16], dtype=np.float32)
    kwargs = dict(steric_diffusion_only=diffusion_only, drift_sign=sign)
    same(
        new.ion_face_flux(phi, density, dx, diffusion, V_T, 1.6e27, **kwargs),
        old.ion_face_flux(phi, density, dx, diffusion, V_T, 1.6e27, **kwargs),
    )


@pytest.mark.parametrize("diffusion_only", [False, True])
def test_infinite_capacity_flux_semantics_are_preserved(diffusion_only):
    phi = np.array([0.0, 0.01, -0.05])
    density = np.array([1e24, 2e24, 3e24])
    args = (phi, density, np.array([1e-9, 2e-9]), 1e-16, V_T, np.inf)
    same(
        new.ion_face_flux(*args, steric_diffusion_only=diffusion_only),
        old.ion_face_flux(*args, steric_diffusion_only=diffusion_only),
    )


@pytest.mark.parametrize("name", [
    "ion_face_flux", "ion_face_flux_jacobian", "ion_flux_steric",
    "ion_continuity_rhs", "ion_continuity_rhs_neg",
])
def test_original_callable_parameters_and_defaults_are_retained(name):
    actual, expected = (inspect.signature(getattr(model, name)) for model in (new, old))
    assert actual == expected


@pytest.mark.parametrize("invalid", ["spacing", "sign", "diffusion", "partner", "node_limit", "face_limit"])
def test_existing_jacobian_refusals_retain_exception_type_and_message(invalid):
    args = [np.array([0.0, 0.01]), np.array([1e24, 2e24]), np.array([1e-8]), 1e-16, V_T, 1.6e27]
    kwargs = {}
    if invalid == "spacing":
        args[2] = np.array([0.0])
    elif invalid == "sign":
        kwargs["drift_sign"] = 0.0
    elif invalid == "diffusion":
        args[3] = -1e-16
    elif invalid == "partner":
        kwargs["P_other_node"] = np.array([np.nan, 1.0])
    elif invalid == "node_limit":
        kwargs.update(steric_diffusion_only=True, P_lim_node=0.0)
    else:
        args[5] = -1.0
    with np.errstate(all="ignore"):
        with pytest.raises(ValueError) as expected:
            old.ion_face_flux_jacobian(*args, **kwargs)
        with pytest.raises(type(expected.value)) as actual:
            new.ion_face_flux_jacobian(*args, **kwargs)
    assert str(actual.value) == str(expected.value)
