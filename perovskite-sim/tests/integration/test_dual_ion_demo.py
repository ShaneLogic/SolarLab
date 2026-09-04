"""End-to-end cover for the second mobile ionic species.

The solver has carried the neg-species path since the 2026-07 closed-loop work,
but no shipped preset used it, so nothing pinned that the path survives the
loader, the backend's inline-device parser and an actual transient. These tests
do that, and pin the two contracts a dual-ion configuration depends on:

  * the state vector grows to 4N when a second species is active, and stays 3N
    when it is not;
  * LEGACY forces ``use_dual_ions`` off, so the same file reduces to the
    single-species problem there — which is why the preset does not pin a tier.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from perovskite_sim.experiments import jv_sweep as jv
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.mode import resolve_mode

CONFIG = "configs/dual_ion_demo.yaml"


@pytest.fixture(scope="module")
def stack():
    return load_device_from_yaml(CONFIG)


def test_preset_carries_a_second_species(stack):
    absorber = stack.layers[1].params
    assert absorber.D_ion > 0 and absorber.P0 > 0, "positive species must be active"
    assert absorber.D_ion_neg == pytest.approx(1.01e-18)
    assert absorber.P0_neg == pytest.approx(1.6e25)
    # Must equal P_lim, not the 1e30 default. Shared-site crowding divides the
    # TOTAL occupancy by each species' own P_lim, so leaving this at the default
    # would give the anion theta ~ 0 — no crowding at all — while the cation
    # still felt the anion. See _steric_diffusion_only_flux.
    assert absorber.P_lim_neg == pytest.approx(absorber.P_lim)
    assert stack.ion_steric_shared_site is True
    # transport layers stay single-species
    for idx in (0, 2):
        assert stack.layers[idx].params.D_ion_neg == 0.0


def test_net_ionic_charge_is_zero_at_rest(stack):
    """P0_neg equals P0, so the equilibrium electrostatics are the benchmark's."""
    absorber = stack.layers[1].params
    assert absorber.P0_neg == pytest.approx(absorber.P0)


def test_state_vector_grows_to_four_blocks(stack):
    x = jv.build_electrical_grid(stack, 25)
    y = jv.solve_equilibrium(x, stack)
    assert y.size == 4 * x.size, "a second species must add a fourth state block"
    P_neg = y[3 * x.size:4 * x.size]
    assert np.all(np.isfinite(P_neg))
    assert P_neg.max() > 0.0


def test_single_species_stack_stays_three_blocks(stack):
    single = dataclasses.replace(
        stack,
        layers=tuple(
            dataclasses.replace(
                layer, params=dataclasses.replace(layer.params, D_ion_neg=0.0, P0_neg=0.0)
            )
            for layer in stack.layers
        ),
    )
    x = jv.build_electrical_grid(single, 25)
    assert jv.solve_equilibrium(x, single).size == 3 * x.size


def test_legacy_tier_disables_the_second_species():
    assert resolve_mode("legacy").use_dual_ions is False
    assert resolve_mode("fast").use_dual_ions is True
    assert resolve_mode("full").use_dual_ions is True


def test_backend_inline_path_carries_the_neg_fields():
    """The frontend posts an inline device dict, never a file path."""
    from backend.main import stack_from_dict

    cfg = {
        "device": {"V_bi": 1.1, "Phi": 1.4e21, "mode": "full"},
        "layers": [
            {
                "name": "PVK", "role": "absorber", "thickness": 4e-7, "eps_r": 24.1,
                "mu_n": 6.62e-3, "mu_p": 6.62e-3, "ni": 2.89e10, "N_D": 0.0, "N_A": 0.0,
                "D_ion": 1.01e-17, "P_lim": 1.6e27, "P0": 1.6e25,
                "D_ion_neg": 1.01e-18, "P0_neg": 1.6e25,
                "tau_n": 3e-9, "tau_p": 3e-7, "n1": 2.89e10, "p1": 2.89e10,
                "B_rad": 0.0, "C_n": 0.0, "C_p": 0.0, "alpha": 1.3e7,
            },
        ],
    }
    built = stack_from_dict(cfg)
    assert built.layers[0].params.D_ion_neg == pytest.approx(1.01e-18)
    assert built.layers[0].params.P0_neg == pytest.approx(1.6e25)


def test_transient_redistributes_both_species_and_conserves_each(stack):
    """A settle past the ionic charging time must move both species and lose none.

    The dwell is 10 s deliberately. At 1 ms the profiles shift by <0.1%, so a
    conservation-only assertion there passes even if the neg-species flux is
    dead — it conserves an inventory that never moved. 10 s is past the ionic
    charging time (L_D·L/D_ion, order seconds here), which makes the movement
    assertions load-bearing. Costs ~2 s, so it stays in the default suite.
    """
    x = jv.build_electrical_grid(stack, 40)
    mat = jv.build_material_arrays(x, stack)
    y0 = jv.solve_equilibrium(x, stack)
    N = x.size
    y1 = jv._integrate_step(x, y0, stack, mat, 0.0, 0.0, 10.0, 1e-4, 1e-6, illuminated=True)
    assert np.all(np.isfinite(y1))

    shifts = {}
    for block, name in ((2, "P_+"), (3, "P_-")):
        before = y0[block * N:(block + 1) * N]
        after = y1[block * N:(block + 1) * N]
        assert np.all(after >= 0.0), f"{name} went negative"
        assert np.trapezoid(after, x) == pytest.approx(
            np.trapezoid(before, x), rel=1e-6
        ), f"ionic inventory of {name} is not conserved"
        shifts[name] = np.max(np.abs(after - before)) / before.max()
        assert shifts[name] > 0.05, f"{name} did not redistribute — is its flux wired up?"


def test_each_species_moves_on_its_own_diffusivity(stack):
    """Early-time displacement must scale with each species' own D_ion.

    At 10 s both species are near their settled profiles, so P_+ outruns P_-
    there by ~1.8x whatever the diffusivities are — that gap is set by charge
    sign, not mobility, and a mutant with D_ion_neg = D_ion still shows it. In
    the diffusion-linear regime at 1 ms the ratio instead tracks D_ion/D_ion_neg
    (= 10 in this preset): measured 9.98 here, and 1.00 for that mutant. This is
    what pins the neg species onto its own diffusivity rather than the cation's.
    """
    x = jv.build_electrical_grid(stack, 40)
    mat = jv.build_material_arrays(x, stack)
    y0 = jv.solve_equilibrium(x, stack)
    N = x.size
    y1 = jv._integrate_step(x, y0, stack, mat, 0.0, 0.0, 1e-3, 1e-4, 1e-6, illuminated=True)

    def displacement(block):
        before, after = y0[block * N:(block + 1) * N], y1[block * N:(block + 1) * N]
        return np.max(np.abs(after - before)) / before.max()

    absorber = stack.layers[1].params
    expected = absorber.D_ion / absorber.D_ion_neg
    assert displacement(2) / displacement(3) == pytest.approx(expected, rel=0.3)
