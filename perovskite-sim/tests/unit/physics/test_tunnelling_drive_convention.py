"""D8-E2R: what the QF lane actually hands the tunnelling channels.

The D8-E1/E2 headline numbers — "the channel carries ~20 % of the terminal
current" and "equilibrium net flux is exactly zero by reciprocity" — were both
artifacts of a level-vs-potential convention error. This file freezes the
mechanism so the retraction is executable rather than a doc sentence, and adds
the negative control the original gate was missing.

The defect
----------
`quasi_fermi_steady_state.py` defines its solver variable as

    qfn0 = V_T*ln(n0) - (phi0 + chi)

which, since ``E_C = -(phi + chi)`` and ``E_Fn = E_C + V_T*ln(n/N_C)``, is

    qfn0 = E_Fn + V_T*ln(N_C)

That is correct *for the solver*: it only ever uses ``diff(qfn0)/V_T``, where
the constant offset cancels. It is the channel wiring that misreads it — the
channels feed it to a Fermi-Dirac occupation as an absolute level.

Why it cannot be defended as a convention
-----------------------------------------
Under Maxwell-Boltzmann the offset is exactly a factor ``N_C`` on the
occupation, so "DOS folded into the level" would cancel against a ``1/N_C``
supply prefactor and the whole thing would be self-consistent. The channels
use Fermi-Dirac, which saturates at 1, so the factor is level-dependent
instead of constant and nothing can cancel it. That asymmetry is the subject
of `test_the_offset_is_only_a_constant_factor_under_boltzmann`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.tunneling_channels import (
    _fermi,
    _turning_point_levels,
    local_barrier_window,
)


LANE_CONFIG = "configs/wkb_tunnelling_intraband_spike.yaml"
THERMAL_VOLTAGE_V = 0.025851999786187952


def _project_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[3]


def _solve_and_capture(V_app: float):
    """Solve the lane config and capture what the channel was actually handed."""

    import perovskite_sim.experiments.quasi_fermi_steady_state as qf

    stack = load_device_from_yaml(_project_root() / LANE_CONFIG)
    grid = multilayer_grid(
        tuple(Layer(layer.thickness, 24) for layer in electrical_layers(stack)),
        alpha=2.0,
    )
    captured: dict = {}
    original = qf.evaluate_tunnelling_channels

    def spy(compiled, **kwargs):
        captured["face"] = compiled.interface_faces[0]
        captured.update(kwargs)
        return original(compiled, **kwargs)

    qf.evaluate_tunnelling_channels = spy
    try:
        result = qf.solve_quasi_fermi_steady_state(
            grid, stack, V_app=V_app, illuminated=False
        )
    finally:
        qf.evaluate_tunnelling_channels = original
    return result, captured, grid, stack


def _true_level(captured, result, stack, grid):
    """E_Fn = -(phi + chi) + V_T ln(n / N_C), the repository's own definition."""

    potential = np.asarray(captured["potential_V"], dtype=float)
    affinity = np.asarray(captured["affinity_eV"], dtype=float)
    density = np.asarray(result.y[: grid.size], dtype=float)
    dos = electrical_layers(stack)[0].params.Nc300
    return -(potential + affinity) + THERMAL_VOLTAGE_V * np.log(
        np.maximum(density, 1.0) / dos
    )


@pytest.mark.slow
def test_the_channel_is_handed_a_potential_not_a_level():
    """Frozen measurement of the offset, to the DOS term it equals."""

    result, captured, grid, stack = _solve_and_capture(0.2)
    passed = np.asarray(captured["electron_quasi_fermi_eV"], dtype=float)
    true_level = _true_level(captured, result, stack, grid)
    offset = passed - true_level
    dos = electrical_layers(stack)[0].params.Nc300
    expected = THERMAL_VOLTAGE_V * math.log(dos)

    assert result.certified is True
    assert float(np.max(offset)) == pytest.approx(expected, rel=1.0e-9)
    # Uniform here ONLY because every layer of this config carries the same
    # N_C. At a DOS-contrast heterointerface — which is the case a tunnelling
    # channel exists for — the offset differs across the barrier and injects a
    # spurious kT*ln(N_C ratio) drive on top of the constant error.
    assert float(np.max(offset) - np.min(offset)) < 1.0e-9


@pytest.mark.slow
def test_the_offset_puts_the_level_on_the_wrong_side_of_the_band_edge():
    """Not a small error: it changes which regime the occupation is in."""

    result, captured, grid, stack = _solve_and_capture(0.2)
    face = captured["face"]
    potential = np.asarray(captured["potential_V"], dtype=float)
    affinity = np.asarray(captured["affinity_eV"], dtype=float)
    conduction = -(potential + affinity)
    passed = np.asarray(captured["electron_quasi_fermi_eV"], dtype=float)
    true_level = _true_level(captured, result, stack, grid)

    # The level handed over sits ABOVE the conduction edge; the real one sits
    # below it. The Fermi factor is exponential in exactly this difference.
    assert passed[face] - conduction[face] > 0.5
    assert true_level[face] - conduction[face] < -0.4


def test_the_offset_is_only_a_constant_factor_under_boltzmann():
    """Why the convention cannot be rescued as "DOS folded into the level".

    Under Maxwell-Boltzmann the offset multiplies the occupation by exactly
    ``N_C`` at every level, so it would cancel against a ``1/N_C`` supply
    prefactor. Under Fermi-Dirac the same offset saturates the occupation at
    1, so the factor depends on the level and nothing can cancel it. The
    channels use Fermi-Dirac.
    """

    dos = 1.0e24
    offset = THERMAL_VOLTAGE_V * math.log(dos)
    energy = -4.5
    boltzmann_ratios = []
    fermi_ratios = []
    for level in (-5.2, -4.9, -4.6):
        shifted = level + offset
        boltzmann_ratios.append(
            math.exp(-(energy - shifted) / THERMAL_VOLTAGE_V)
            / math.exp(-(energy - level) / THERMAL_VOLTAGE_V)
        )
        fermi_ratios.append(
            float(_fermi(np.array([energy]), shifted, THERMAL_VOLTAGE_V)[0])
            / float(_fermi(np.array([energy]), level, THERMAL_VOLTAGE_V)[0])
        )

    for ratio in boltzmann_ratios:
        assert ratio == pytest.approx(dos, rel=1.0e-9)
    # Fermi-Dirac: not constant, and orders below the Boltzmann factor.
    assert max(fermi_ratios) / min(fermi_ratios) > 1.0e6
    assert all(ratio < dos for ratio in fermi_ratios)


# --------------------------------------------------------------------------
# The negative control the original equilibrium gate was missing
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_the_equilibrium_zero_is_saturation_not_reciprocity():
    """The registered exact-zero gate passes for the wrong reason.

    `docs/wkb-tunnelling-family-contract.md` gated equilibrium net flux as
    exact "because a threshold there would hide a sign or bookkeeping error".
    Measured, it passes because both Fermi factors round to the SAME double
    near 1.0: the residual equilibrium quasi-Fermi gradient is ~4e-13 eV, which
    perturbs an occupation of ~1 by ~1e-20 — below the ~2e-16 ulp there. Read
    at the true level the occupation is ~3e-12, where the ulp is ~1e-27 and
    the same gradient IS resolvable.

    So the gate cannot fail, and a gate that cannot fail is not evidence.
    """

    result, captured, grid, stack = _solve_and_capture(0.0)
    face = captured["face"]
    potential = np.asarray(captured["potential_V"], dtype=float)
    affinity = np.asarray(captured["affinity_eV"], dtype=float)
    conduction = -(potential + affinity)
    passed = np.asarray(captured["electron_quasi_fermi_eV"], dtype=float)
    true_level = _true_level(captured, result, stack, grid)

    peak, base = local_barrier_window(conduction, face)
    energies = np.linspace(base, peak, 96)

    left_passed, right_passed = _turning_point_levels(
        conduction, energies, face, passed
    )
    left_true, right_true = _turning_point_levels(
        conduction, energies, face, true_level
    )
    occupation_gap_passed = float(
        np.max(
            np.abs(
                _fermi(energies, left_passed, THERMAL_VOLTAGE_V)
                - _fermi(energies, right_passed, THERMAL_VOLTAGE_V)
            )
        )
    )
    occupation_gap_true = float(
        np.max(
            np.abs(
                _fermi(energies, left_true, THERMAL_VOLTAGE_V)
                - _fermi(energies, right_true, THERMAL_VOLTAGE_V)
            )
        )
    )

    # The equilibrium quasi-Fermi profile is flat only to solver residual, not
    # exactly — so a discriminating gate must see something non-zero here.
    assert float(np.max(passed) - np.min(passed)) > 0.0

    # As wired: bitwise-equal occupations, hence the "exact" zero.
    assert occupation_gap_passed == 0.0
    # At the true level: the same physical state resolves a non-zero gap.
    assert occupation_gap_true > 0.0

    # And the reported flux is correspondingly, unfalsifiably, exactly zero.
    diagnostics = result.tunnelling_channel_diagnostics
    assert diagnostics.channel_net_flux_m2_s[0] == 0.0


def test_reciprocity_itself_is_still_exact_when_the_occupations_are_equal():
    """The D8-E0 unit-level claim survives, and is a different statement.

    That test passes ONE occupation array as both sides, so the integrand is
    identically zero by construction — a property of `reciprocal_net_flux`,
    not of any device state. The device-level gate above is the one that was
    vacuous, because there the two occupations come from different positions
    and are only equal to within the solve's residual.
    """

    from perovskite_sim.physics.wkb_tunneling import reciprocal_net_flux

    energies = np.linspace(-4.7, -4.4, 48)
    transmission = np.linspace(1.0e-12, 1.0, 48)
    occupation = _fermi(energies, -5.2, THERMAL_VOLTAGE_V)

    flux = reciprocal_net_flux(energies, transmission, occupation, occupation, 1.0e24)

    assert flux.net_flux_m2_s == 0.0
    assert flux.forward_flux_m2_s == flux.reverse_flux_m2_s
