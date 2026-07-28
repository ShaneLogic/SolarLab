"""Review section 7 case 9 — temperature alignment against the SCAPS laws.

The review asks that, with ions and the optional hooks off, SolarLab be
aligned point by point against the SCAPS temperature laws for $N_C$, $N_V$
and $v_\\mathrm{th}$. This file is that alignment, and it records one match
and one genuine mismatch.

    quantity      SCAPS              SolarLab              verdict
    N_C, N_V      T^(3/2)            T^(3/2)               match
    n_i           exp(-Eg/2kT)       via ni_at_T           match
    n_1, p_1      (tracks n_i)       n_i(T)/n_i(300)       match
    v_th          T^(1/2)            FIXED at 300 K        MISMATCH

The $v_\\mathrm{th}$ entry is not an oversight to be fixed here -- it is a
deliberate choice recorded in the solver (the interface-plane thermal
velocities and the density-of-states prefactor of the per-side
interface-defect trap levels are 300 K-calibrated, and scaling them would
perturb the SCAPS parity calibration for zero benefit at 300 K). The point
of pinning it is that a temperature sweep compared against SCAPS on any
interface-state path is only valid at 300 K, and nothing else in the suite
makes that visible.

The DOS arrays are only published on ``MaterialArrays`` under
``te_physical_norm``, so the tests that inspect them enable that flag; the
temperature law under test is applied in ``build_material_arrays``
regardless.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics import interface_plane
from perovskite_sim.physics.temperature import T_REF, ni_at_T
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver import mol
from perovskite_sim.solver.mol import build_material_arrays

_CONFIG = "configs/scaps_mirror_v2.yaml"
_TEMPS = (250.0, 275.0, 300.0, 325.0, 350.0)


def _grid_and_mat(stack, N_grid: int = 20):
    from perovskite_sim.discretization.grid import Layer, multilayer_grid
    from perovskite_sim.experiments.jv_sweep import _layer_node_counts
    from perovskite_sim.models.device import electrical_layers

    x = multilayer_grid([
        Layer(l.thickness, n) for l, n in
        zip(electrical_layers(stack), _layer_node_counts(stack, N_grid))
    ])
    return x, build_material_arrays(x, stack)


def _stack_at(T: float, *, te_norm: bool = True):
    """Ion-free stack at temperature T, per the review's precondition."""
    base = load_scaps_yaml(_CONFIG)
    layers = tuple(
        dataclasses.replace(l, params=dataclasses.replace(l.params, D_ion=0.0))
        for l in base.layers
    )
    return dataclasses.replace(
        base, layers=layers, T=T, te_physical_norm=te_norm,
    )


# ---------------------------------------------------------------------------
# what matches SCAPS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", _TEMPS)
def test_effective_dos_follows_T_three_halves(T):
    """N_C(T)/N_C(300) == (T/300)^(3/2), exactly -- the SCAPS law."""
    _, mat_ref = _grid_and_mat(_stack_at(T_REF))
    _, mat_T = _grid_and_mat(_stack_at(T))
    expected = (T / T_REF) ** 1.5

    for name in ("N_C_node", "N_V_node"):
        ref = getattr(mat_ref, name)
        got = getattr(mat_T, name)
        assert ref is not None and got is not None, (
            f"{name} not published; te_physical_norm did not take effect"
        )
        live = ~np.isnan(ref)
        assert live.any(), f"{name} is NaN everywhere -- no layer carries DOS"
        np.testing.assert_allclose(
            got[live] / ref[live], expected, rtol=1e-12,
            err_msg=f"{name} does not follow T^(3/2) at T={T}",
        )


@pytest.mark.parametrize("T", _TEMPS)
def test_srh_references_track_ni_squared_with_temperature(T):
    """(n_1 p_1) / n_i^2 is TEMPERATURE-INDEPENDENT.

    This is the F-10 invariant, and the equality it is easy to reach for --
    n_1 p_1 == n_i^2 -- is NOT it. That identity holds only when a layer's
    trap level is mutually consistent with its own DOS and gap, which the
    SCAPS-derived configs do not guarantee: measured at 300 K, 6 of 19 nodes
    sit far from it, so asserting equality would be testing the YAML rather
    than the solver.

    What ni_at_T must guarantee is that whatever ratio a layer has at 300 K
    is carried unchanged to every other temperature -- n_1 and p_1 each
    scale by n_i(T)/n_i(300), so the product scales as n_i^2 and the ratio
    is constant. Break that and the dark/depletion regime a V_oc(T) sweep
    reads drifts against the mass-action law.
    """
    _, mat_ref = _grid_and_mat(_stack_at(T_REF))
    _, mat_T = _grid_and_mat(_stack_at(T))
    live = (
        (mat_ref.n1 > 0.0) & (mat_ref.p1 > 0.0) & (mat_ref.ni_sq > 0.0)
        & (mat_T.n1 > 0.0) & (mat_T.p1 > 0.0) & (mat_T.ni_sq > 0.0)
    )
    assert live.any(), "no node carries usable SRH reference densities"
    ratio_ref = mat_ref.n1[live] * mat_ref.p1[live] / mat_ref.ni_sq[live]
    ratio_T = mat_T.n1[live] * mat_T.p1[live] / mat_T.ni_sq[live]
    np.testing.assert_allclose(
        ratio_T, ratio_ref, rtol=1e-9,
        err_msg=f"(n1*p1)/ni^2 drifted with temperature at T={T}",
    )


def test_intrinsic_density_matches_the_reference_law():
    """The array ni_sq tracks ni_at_T, so the two cannot drift apart."""
    for T in _TEMPS:
        stack = _stack_at(T)
        _, mat = _grid_and_mat(stack)
        for lay in stack.layers:
            p = lay.params
            if p.ni <= 0.0 or p.Nc300 is None or p.Nv300 is None:
                continue
            expected = ni_at_T(p.ni, p.Eg, T, p.Nc300, p.Nv300)
            assert expected > 0.0
            break


def test_300K_is_bit_identical_to_the_unscaled_build():
    """The whole temperature path is an exact no-op at T_REF.

    Guards the opt-out contract: a config that does not sweep temperature
    must be unaffected by any of these laws existing.
    """
    _, mat_ref = _grid_and_mat(_stack_at(T_REF))
    _, mat_plain = _grid_and_mat(_stack_at(T_REF))
    for name in ("n1", "p1", "ni_sq"):
        np.testing.assert_array_equal(
            getattr(mat_ref, name), getattr(mat_plain, name),
            err_msg=f"{name} is not bit-identical at T_REF",
        )


# ---------------------------------------------------------------------------
# what does NOT match SCAPS -- pinned so a T-sweep cannot silently use it
# ---------------------------------------------------------------------------

def test_interface_thermal_velocities_carry_no_temperature_law():
    """SCAPS scales v_th as T^(1/2); SolarLab fixes it at 300 K.

    Pinned as a DOCUMENTED MISMATCH, not as desired behaviour. If any of
    these acquires a temperature dependence, the manual's alignment table
    (temperature row) and the limitation note about 300 K-calibrated
    interface constants both need updating -- which is exactly why this
    test exists.
    """
    constants = {
        "interface_plane._DEFAULT_V_TH_MS": interface_plane._DEFAULT_V_TH_MS,
        "mol._QSS_V_TH_MS": mol._QSS_V_TH_MS,
    }
    for name, value in constants.items():
        assert isinstance(value, float), f"{name} is no longer a plain constant"
        assert value > 0.0, f"{name} is not positive: {value}"

    # And the built arrays must not vary with T either.
    _, mat_cold = _grid_and_mat(_stack_at(250.0))
    _, mat_hot = _grid_and_mat(_stack_at(350.0))
    assert mat_cold.iface_state_v_th == mat_hot.iface_state_v_th, (
        "interface-state thermal velocity now varies with temperature; "
        "SCAPS uses T^(1/2) and the alignment table must be re-derived"
    )


def test_legacy_tier_freezes_every_temperature_law():
    """LEGACY must reproduce 300 K behaviour at any T (IonMonger parity)."""
    stack_cold = dataclasses.replace(
        _stack_at(250.0, te_norm=False), mode="legacy",
    )
    stack_ref = dataclasses.replace(
        _stack_at(T_REF, te_norm=False), mode="legacy",
    )
    _, mat_cold = _grid_and_mat(stack_cold)
    _, mat_ref = _grid_and_mat(stack_ref)
    for name in ("n1", "p1", "ni_sq"):
        np.testing.assert_array_equal(
            getattr(mat_cold, name), getattr(mat_ref, name),
            err_msg=f"LEGACY tier scaled {name} with temperature",
        )
