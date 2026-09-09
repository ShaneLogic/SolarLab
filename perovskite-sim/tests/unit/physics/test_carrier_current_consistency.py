"""Current observations must use the same active interface flux as continuity."""

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import (
    _state_fields,
    compute_current_components,
)
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.continuity import carrier_continuity_rhs
from perovskite_sim.scaps_compat.loader import load_scaps_yaml
from perovskite_sim.solver.mol import StateVec, build_material_arrays
from perovskite_sim.twod.continuity_2d import apply_thermionic_caps_y
from perovskite_sim.twod.grid_2d import Grid2D
from perovskite_sim.twod.microstructure import Microstructure
from perovskite_sim.twod.solver_2d import build_material_arrays_2d


@pytest.mark.parametrize("temperature", [250.0, 300.0, 350.0])
def test_observed_carrier_current_closes_the_actual_continuity_operator(temperature):
    stack = replace(
        load_scaps_yaml("configs/scaps_mirror_v2.yaml"),
        T=temperature,
        te_physical_norm=True,
    )
    grid = multilayer_grid(
        [Layer(layer.thickness, 4) for layer in electrical_layers(stack)]
    )
    material = build_material_arrays(grid, stack)
    material = replace(
        material,
        A_star_n=np.full(grid.size, 1e-20),
        A_star_p=np.full(grid.size, 1e-20),
    )
    n = np.full(grid.size, 1e20)
    p = np.full(grid.size, 2e20)
    state = StateVec.pack(n, p, material.P_ion0)
    n, p, phi, _ = _state_fields(grid, state, stack, 0.1, material)
    params = material.carrier_params
    params["degenerate_recombination_model"] = "off"
    dn, dp = carrier_continuity_rhs(grid, phi, n, p, np.zeros_like(grid), params)
    no_cap = dict(params, interface_faces=())
    raw_dn, raw_dp = carrier_continuity_rhs(
        grid, phi, n, p, np.zeros_like(grid), no_cap
    )
    assert not (np.array_equal(dn, raw_dn) and np.array_equal(dp, raw_dp))
    observed = compute_current_components(grid, state, stack, 0.1, mat=material)
    electron = -material.junction_polarity * observed.J_n
    hole = -material.junction_polarity * observed.J_p
    for balance, flux, sign in ((dn, electron, 1.0), (dp, hole, -1.0)):
        expected = sign * np.diff(flux) / (Q * material.dx_cell[1:-1])
        np.testing.assert_allclose(balance[1:-1], expected, rtol=1e-12, atol=1e-6)


def test_two_dimensional_cap_never_reverses_the_supplied_current():
    current = np.ones((1, 2))
    n = np.array([[1.0, 1.0], [2.0, 2.0]])
    p = np.ones_like(n)
    chi = np.array([[0.1, 0.1], [0.0, 0.0]])
    limited, _ = apply_thermionic_caps_y(
        current,
        current,
        n,
        p,
        chi,
        np.zeros_like(n),
        0.02585,
        interface_y_faces=(0,),
        A_star_n=np.full_like(n, 1e-7),
        A_star_p=np.full_like(n, 1e-7),
        T=300.0,
    )
    assert np.all(limited > 0.0)
    assert np.all(limited < current)


@pytest.mark.parametrize("physical", [False, True])
def test_two_dimensional_material_retains_physical_bands_and_normalization(physical):
    stack = replace(
        load_scaps_yaml("configs/scaps_mirror_v2.yaml"), te_physical_norm=physical
    )
    y = multilayer_grid(
        [Layer(layer.thickness, 4) for layer in electrical_layers(stack)]
    )
    grid = Grid2D(x=np.linspace(0.0, 300e-9, 3), y=y)
    one = build_material_arrays(y, stack)
    two = build_material_arrays_2d(grid, stack, Microstructure(), lateral_bc="neumann")
    np.testing.assert_array_equal(two.chi_phys[:, 0], one.chi_phys)
    np.testing.assert_array_equal(two.Eg_phys[:, 0], one.Eg_phys)
    assert not np.array_equal(two.chi, two.chi_phys)
    assert two.te_physical_norm == one.te_physical_norm == physical
    assert two.thermionic_normalization_status == one.thermionic_normalization_status
    assert one.thermionic_normalization_status == (
        "physical_dos_normalized" if physical else "legacy_density_weighted"
    )


@pytest.mark.parametrize("temperature", [250.0, 300.0, 350.0])
@pytest.mark.parametrize("physical", [False, True])
def test_two_dimensional_cap_matches_independent_richardson_expression(
    temperature, physical
):
    from perovskite_sim.constants import K_B

    n = np.array([[1.0, 2.0], [2.0, 1.0]])
    p = np.array([[2.0, 3.0], [4.0, 1.0]])
    chi = np.array([[0.1, 0.2], [0.0, 0.0]])
    gap = np.full_like(chi, 1.0)
    nc = np.array([[2.0, 3.0], [8.0, 12.0]])
    nv = 3 * nc
    raw_n = np.array([[1.0, -2.0]])
    raw_p = -raw_n
    prefactor = np.full_like(n, 1e-7)
    thermal = K_B * temperature / Q
    result = apply_thermionic_caps_y(
        raw_n,
        raw_p,
        n,
        p,
        np.zeros_like(chi),
        gap,
        thermal,
        interface_y_faces=(0,),
        A_star_n=prefactor,
        A_star_p=prefactor,
        T=temperature,
        chi_te=chi,
        Eg_te=gap,
        te_physical_norm=physical,
        N_C_node=nc,
        N_V_node=nv,
    )
    for density, dos, supplied, actual in zip((n, p), (nc, nv), (raw_n, raw_p), result):
        barrier = chi[0] - chi[1]
        emission = (
            1e-7
            * temperature**2
            * (density[0] * np.exp(-barrier / thermal) - density[1])
        )
        if physical:
            emission /= np.sqrt(dos[0] * dos[1])
        expected = np.copysign(
            np.minimum(np.abs(supplied[0]), np.abs(emission)), supplied[0]
        )
        np.testing.assert_allclose(actual[0], expected, rtol=2e-14, atol=0.0)
        assert np.all(np.abs(actual[0]) < np.abs(supplied[0]))


def test_missing_dos_is_labelled_as_legacy_compatibility():
    from perovskite_sim.physics.thermionic_transport import (
        thermionic_normalization_status,
    )

    params = {"interface_faces": (0,), "te_physical_norm": True, "N_C_node": np.ones(2)}
    assert thermionic_normalization_status(params) == "legacy_fallback_missing_dos"
    params["N_V_node"] = np.ones(2)
    assert thermionic_normalization_status(params) == "physical_dos_normalized"
    params["N_V_node"][1] = np.nan
    assert thermionic_normalization_status(params) == "legacy_fallback_missing_dos"
