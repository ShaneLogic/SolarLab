"""Independent checks for the bounded V6 ion-precision investigation."""
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg
from threadpoolctl import threadpool_limits

from perovskite_sim.physics.ion_migration import ion_face_flux_jacobian
from scripts.probe_r1_v6_ion_precision import (
    balanced_float_flux, comparison, dec, decimal_sg_flux,
    direct_split_inputs, poisson_project, retain_saved_increments,
)


@pytest.fixture(autouse=True)
def single_thread_numerics():
    with threadpool_limits(1):
        yield


@pytest.mark.parametrize("sign", [-1, 1])
def test_saved_split_representation_retains_sub_ulp_signed_drive(sign):
    vt = 0.025
    old = {"phi_V": [.4, .4, .4], "positive_m3": [1e20] * 3,
           "n_m3": [1e15] * 3, "p_m3": [1e15] * 3}
    coordinate = np.zeros(6)
    coordinate[-1] = sign * 1e-19
    row = {"state": old, "time_s": 2., "physics_reconstruction": {"coordinate": coordinate}}
    material = SimpleNamespace(V_T_device=vt, D_ion_face=np.ones(2), iface_qss_left_nodes=[])
    with localcontext() as context:
        context.prec = 90
        split = direct_split_inputs(row, {"state": old}, material)
        represented = decimal_sg_flux(old["phi_V"], old["positive_m3"], [1, 1], [1, 1], vt, [1e27] * 3)
        resolved = decimal_sg_flux(split["phi"], split["density"], [1, 1], [1, 1], vt, [1e27] * 3)
    assert represented == [Decimal(0), Decimal(0)]
    assert resolved[0] * sign < 0
    assert resolved[1] * sign > 0
    assert split["checked_saved_float_reconstruction"]
    assert not comparison(resolved, represented)["passed"]


@pytest.mark.parametrize("drop", [-1., -.003, 0., .003, 1.])
def test_flux_jacobian_matches_independent_decimal_derivative(drop):
    phi = np.asarray([0., drop * .025])
    density = np.asarray([1e20, 1.02e20])
    dx, diffusion, limit = np.asarray([1e-8]), np.asarray([1e-14]), np.asarray([1e25, 1e25])
    jacobian = ion_face_flux_jacobian(phi, density, dx, diffusion, .025, 1e25,
                                    steric_diffusion_only=True, P_lim_node=limit)
    with localcontext() as context:
        context.prec = 80
        h = Decimal("1e-30")
        positive_phi = [dec(phi[0]), dec(phi[1]) + h]
        negative_phi = [dec(phi[0]), dec(phi[1]) - h]
        a = decimal_sg_flux(positive_phi, density, dx, diffusion, .025, limit)[0]
        b = decimal_sg_flux(negative_phi, density, dx, diffusion, .025, limit)[0]
        derivative_phi = float((a - b) / (2 * h))
        h_density = Decimal("1e-10")
        a = decimal_sg_flux(phi, [dec(density[0]) + h_density, dec(density[1])],
                            dx, diffusion, .025, limit)[0]
        b = decimal_sg_flux(phi, [dec(density[0]) - h_density, dec(density[1])],
                            dx, diffusion, .025, limit)[0]
        derivative_density = float((a - b) / (2 * h_density))
    assert jacobian.potential_right_derivative[0] == pytest.approx(derivative_phi, rel=3e-13)
    assert jacobian.density_left_derivative[0] == pytest.approx(derivative_density, rel=3e-13)


@pytest.mark.parametrize("drive", [-1e-6, 0., 1e-6])
def test_candidate_retains_normal_zero_and_signed_nonzero_flux(drive):
    phi, density = np.asarray([.1, .1 + drive]), np.asarray([1e20, 1e20])
    args = (phi, density, np.asarray([1e-8]), np.asarray([1e-14]), .025, np.asarray([1e25, 1e25]))
    actual = balanced_float_flux(*args)
    exact = decimal_sg_flux(*args)
    assert actual[0] == pytest.approx(float(exact[0]), rel=2e-14, abs=0.)
    assert np.sign(actual[0]) == -np.sign(drive)


def test_original_comparison_still_rejects_declared_faults():
    original = decimal_sg_flux([0., 1e-8, 0.], [1e20] * 3, [1e-8] * 2,
                               [1e-14] * 2, .025, [1e25] * 3)
    assert comparison(original, original)["passed"]
    faults = {"sign": [-x for x in original],
              "scale": [Decimal("1.01") * x for x in original],
              "omitted_ion": [Decimal(0), Decimal(0)],
              "boundary_face": [Decimal(0), original[1]],
              "one_face_sign": [-original[0], original[1]]}
    for fault in faults.values():
        assert not comparison(original, fault)["passed"]


def test_decimal_poisson_projection_has_independent_zero_constraint_root():
    material = SimpleNamespace(
        V_T_device=.025, poisson_factor=SimpleNamespace(C=np.ones(2), h_cell=np.ones(1)),
        N_A=np.zeros(3), N_D=np.zeros(3), P_ion0=np.full(3, 1e20),
        iface_qss_left_nodes=[], iface_qss_right_nodes=[],
    )
    with localcontext() as context:
        context.prec = 90
        shift = Decimal("1e-20")
        n = [Decimal("1e15"), Decimal("1e15") * (shift / dec(.025)).exp(), Decimal("1e15")]
        p = [Decimal("1e15"), Decimal("1e15") * (-shift / dec(.025)).exp(), Decimal("1e15")]
        result = poisson_project([0, shift, 0], n, p, [1e20] * 3, [], material)
    assert abs(result["phi"][1]) < Decimal("1e-65")
    assert max(map(abs, result["final_residual"])) < Decimal("1e-65")
    assert result["initial_residual"][0] != 0


def test_persistent_shadow_accumulates_low_parts_without_claiming_replay():
    old = {"phi_V": [.4, .4, .4], "positive_m3": [1e20] * 3,
           "n_m3": [1e15] * 3, "p_m3": [1e15] * 3}
    coordinate = np.zeros(6)
    coordinate[-1] = 1e-19
    row = {"state": old, "physics_reconstruction": {"coordinate": coordinate}}
    material = SimpleNamespace(V_T_device=.025, D_ion_face=np.ones(2), iface_qss_left_nodes=[])
    with localcontext() as context:
        context.prec = 90
        retained = retain_saved_increments(row, None, material)
        for _ in range(7):
            retained = retain_saved_increments(row, retained, material)
        difference = retained["phi"][1] - dec(.4)
        expected = 7 * dec(.025) * dec(1e-19)
    assert abs(difference - expected) < Decimal("1e-88")
    assert float(retained["phi"][1]) == .4
