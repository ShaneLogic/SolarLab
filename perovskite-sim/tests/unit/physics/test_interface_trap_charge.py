from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import InterfaceChargeClosureParkedError
from perovskite_sim.physics.interface_plane import (
    equilibrium_referenced_interface_trap_charge,
)
from perovskite_sim.solver.mol import (
    StateVec,
    assemble_rhs,
    build_material_arrays,
)


@pytest.mark.parametrize("trap_character", ["acceptor_like", "donor_like"])
def test_donor_and_acceptor_increment_have_the_same_physical_sign(
    trap_character,
):
    del trap_character  # Character changes absolute charge, not Delta charge.
    density = 2.5e16

    charge = equilibrium_referenced_interface_trap_charge(0.75, 0.25, density)

    assert float(charge) == pytest.approx(-Q * density * 0.5)


def test_reference_state_is_exactly_zero_and_direction_reverses():
    density = np.array([1.0e12, 1.0e16, 1.0e17])
    reference = np.array([0.2, 0.5, 0.8])

    zero = equilibrium_referenced_interface_trap_charge(
        reference, reference, density
    )
    positive = equilibrium_referenced_interface_trap_charge(
        reference - 0.1, reference, density
    )
    negative = equilibrium_referenced_interface_trap_charge(
        reference + 0.1, reference, density
    )

    np.testing.assert_array_equal(zero, np.zeros(3))
    assert np.all(positive > 0.0)
    assert np.all(negative < 0.0)


def test_charge_is_linear_in_density_and_bounded_by_one_electron_per_trap():
    densities = np.array([1.0e12, 1.0e15, 1.0e17])
    charge = equilibrium_referenced_interface_trap_charge(
        np.array([0.0, 0.4, 1.0]),
        np.array([1.0, 0.1, 0.0]),
        densities,
    )

    assert np.all(np.abs(charge) <= Q * densities)
    base = equilibrium_referenced_interface_trap_charge(0.8, 0.3, 1.0e16)
    scaled = equilibrium_referenced_interface_trap_charge(0.8, 0.3, 4.0e16)
    assert float(scaled) == pytest.approx(4.0 * float(base))


@pytest.mark.parametrize(
    "occupancy, reference, density, message",
    [
        (-0.1, 0.5, 1.0, "occupancy"),
        (0.5, 1.1, 1.0, "equilibrium_occupancy"),
        (0.5, 0.5, -1.0, "non-negative"),
        (np.nan, 0.5, 1.0, "finite"),
    ],
)
def test_charge_law_rejects_nonphysical_inputs(
    occupancy, reference, density, message
):
    with pytest.raises(ValueError, match=message):
        equilibrium_referenced_interface_trap_charge(
            occupancy, reference, density
        )


def test_retired_scalar_sign_path_cannot_reach_poisson():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    x = multilayer_grid([Layer(layer.thickness, 5) for layer in stack.layers])
    material = build_material_arrays(x, stack)
    retired = dataclasses.replace(material, iface_state_charge=1.0)
    state = StateVec.pack(
        np.full(x.size, 1.0e18),
        np.full(x.size, 1.0e18),
        material.P_ion0.copy(),
        P_neg=(
            None
            if material.P_ion0_neg is None
            else material.P_ion0_neg.copy()
        ),
    )

    with pytest.raises(InterfaceChargeClosureParkedError, match="retired"):
        assemble_rhs(
            0.0,
            state,
            x,
            stack,
            retired,
            illuminated=False,
        )
