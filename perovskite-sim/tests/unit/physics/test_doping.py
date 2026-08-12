from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.physics.doping import (
    doping_at_position,
    layer_doping_profiles,
    validate_doping_profile_params,
)


def _params(**updates) -> MaterialParams:
    values = {
        "eps_r": 11.7,
        "mu_n": 0.135,
        "mu_p": 0.048,
        "D_ion": 0.0,
        "P_lim": 1.0e30,
        "P0": 0.0,
        "ni": 1.0e16,
        "tau_n": 1.0e-2,
        "tau_p": 1.0e-2,
        "n1": 1.0e16,
        "p1": 1.0e16,
        "B_rad": 1.0e-20,
        "C_n": 2.8e-43,
        "C_p": 9.9e-44,
        "alpha": 0.0,
        "N_A": 0.0,
        "N_D": 1.0e21,
    }
    values.update(updates)
    return MaterialParams(**values)


def test_uniform_doping_is_exactly_backward_compatible():
    params = _params(N_A=2.0e22, N_D=3.0e20)
    x = np.array([0.0, 1.0e-6, 2.0e-6])

    acceptors, donors = layer_doping_profiles(x, 2.0e-6, params)

    assert np.array_equal(acceptors, np.full_like(x, params.N_A))
    assert np.array_equal(donors, np.full_like(x, params.N_D))


def test_gaussian_profile_reproduces_published_diffused_emitter_definition():
    decay_length = 5.0e-6 / np.sqrt(np.log(10.0))
    params = _params(
        N_A=1.0e25,
        N_A_bulk=0.0,
        doping_profile_shape="gaussian",
        doping_decay_length=decay_length,
        doping_edge="front",
    )
    x = np.array([0.0, 5.0e-6, 10.0e-6, 15.0e-6])

    acceptors, donors = layer_doping_profiles(x, 15.0e-6, params)

    np.testing.assert_allclose(
        acceptors,
        [1.0e25, 1.0e24, 1.0e21, 1.0e16],
        rtol=2.0e-14,
    )
    np.testing.assert_array_equal(donors, np.full_like(x, 1.0e21))
    assert acceptors[2] == pytest.approx(donors[2], rel=2.0e-14)


def test_back_edge_profile_is_the_front_profile_reflected():
    front = _params(
        N_D=4.0e23,
        N_D_bulk=1.0e20,
        doping_profile_shape="gaussian",
        doping_decay_length=0.7e-6,
        doping_edge="front",
    )
    back = dataclasses.replace(front, doping_edge="back")
    x = np.linspace(0.0, 3.0e-6, 11)

    _acceptors, front_donors = layer_doping_profiles(x, 3.0e-6, front)
    _acceptors, back_donors = layer_doping_profiles(x, 3.0e-6, back)

    np.testing.assert_allclose(back_donors, front_donors[::-1], rtol=1.0e-14)
    assert doping_at_position(back, 3.0e-6, 3.0e-6)[1] == pytest.approx(
        back.N_D
    )


@pytest.mark.parametrize(
    "updates, match",
    [
        ({"doping_profile_shape": "gaussian"}, "requires N_A_bulk"),
        ({"N_A_bulk": 0.0}, "doping_profile_shape"),
        (
            {
                "N_A_bulk": 0.0,
                "doping_profile_shape": "gaussian",
                "doping_decay_length": 0.0,
            },
            "finite and positive",
        ),
        (
            {
                "N_A_bulk": 0.0,
                "doping_profile_shape": "gaussian",
                "doping_decay_length": 1.0e-6,
                "doping_edge": "middle",
            },
            "doping_edge",
        ),
    ],
)
def test_incomplete_or_nonphysical_profiles_fail_closed(updates, match):
    with pytest.raises(ValueError, match=match):
        validate_doping_profile_params(_params(**updates))
