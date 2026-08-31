"""D8-E2 WKB tunnelling-channel numerical-refinement contract tests.

These cover the lane's *contract* — its registration, its protocol hash, its
refusals, and the structural claims it certifies. The 9-cell matrix itself is
too slow for the default lane and is exercised by the registered run.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from perovskite_sim.validation.numerical_certificate import (
    MatrixPoint,
    content_sha256,
    load_refinement_registry,
)
from perovskite_sim.validation.tunnelling_channel_refinement import (
    _execution_protocol,
    _quadrature_orders,
    _with_order,
    run_tunnelling_channel_qf_dc_refinement,
)


ROOT = Path(__file__).resolve().parents[3]
LANE_ID = "wkb-tunnelling-channel-qf-dc-v1"


def _lane():
    return load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    ).lane(LANE_ID)


def _metrics(measurement, *, quality: bool = False):
    values = measurement.quality if quality else measurement.observables
    return {item.name: item for item in values}


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


def test_lane_declares_a_three_by_three_matrix_and_an_energy_ladder():
    lane = _lane()

    assert len(lane.matrix_points) == 9
    assert lane.grid_values == (24, 48, 96)
    assert lane.tolerance_factors == (1.0, 0.1, 0.01)
    # The third axis lives in options, not in MatrixPoint: the shared runner
    # is a two-axis Cartesian product and ConvergenceCheck.dimension is a
    # closed literal, so a real third axis would mean changing machinery that
    # every other lane depends on.
    assert lane.options["energy_quadrature_orders"] == [96, 192, 384]


def test_the_channel_flux_is_an_observable_not_only_the_terminal_current():
    """A terminal-current lane would certify nothing about this channel.

    Enabling the channel moves the terminal current by ~1e-5 relative while
    the channel itself carries ~20 % of it, because the tunnelling path is in
    parallel with the drift-diffusion flux on the same face.
    """
    metrics = {gate.metric for gate in _lane().observables}

    assert "intraband_electron_net_flux_m2_s" in metrics
    # The ACTION, not exp(-2S): the exponential amplifies the exponent's own
    # discretisation error by 2S (~28 on this barrier), so gating on the raw
    # transmission would report a converged action as unconverged.
    assert "intraband_electron_maximum_action" in metrics
    assert "intraband_electron_minimum_transmission" not in metrics


def test_the_lane_config_actually_enables_a_tunnelling_channel():
    from perovskite_sim.models.config_loader import load_device_from_yaml

    lane = _lane()
    stack = load_device_from_yaml(ROOT / lane.config_path)

    assert stack.tunnelling_channels is not None
    assert stack.tunnelling_channels.enabled_channels == ("intraband",)
    assert stack.tunnelling_channels.intraband.carrier == "electron"


# --------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------


def test_the_protocol_hash_moves_when_the_declared_protocol_moves():
    lane = _lane()
    base = _execution_protocol(
        lane, quadrature_orders=(96, 192, 384), bias_V=0.2, profile_points=17
    )
    shifted = _execution_protocol(
        lane, quadrature_orders=(96, 192, 384), bias_V=0.3, profile_points=17
    )

    assert base["schema_version"]
    assert content_sha256(base) != content_sha256(shifted)
    # Nothing that varies per matrix cell may enter the protocol: the
    # effective Newton tolerance and finite-difference step ARE the tolerance
    # axis, so including them would make every cell disagree with every other
    # and the certificate would report the matrix as inconsistent rather than
    # as a converging study. They belong in metadata.actual.
    assert "solve_controls" not in base
    assert "tolerance_factor" not in base


def test_the_energy_ladder_must_be_consecutive_doublings():
    """A ladder that is not a doubling makes the convergence ratio meaningless."""
    assert _quadrature_orders({"energy_quadrature_orders": [16, 32, 64]}) == (
        16,
        32,
        64,
    )
    for bad in ([16], [16, 48], [16, 32, 96], [0, 0], "16,32"):
        with pytest.raises(ValueError):
            _quadrature_orders({"energy_quadrature_orders": bad})


def test_a_non_standard_config_loader_is_refused():
    """The executor reads a standard YAML; a SCAPS-shaped one is a different
    contract and must fail closed rather than be reinterpreted."""
    base = _lane()
    lane = replace(
        base,
        options_json=json.dumps({**base.options, "config_loader": "scaps"}),
    )

    with pytest.raises(ValueError, match="config_loader"):
        run_tunnelling_channel_qf_dc_refinement(lane, MatrixPoint(24, 1.0), ROOT)


def test_a_stack_without_a_channel_is_refused_rather_than_measured_as_zero():
    from perovskite_sim.models.config_loader import load_device_from_yaml

    stack = load_device_from_yaml(ROOT / _lane().config_path)
    stripped = replace(stack, tunnelling_channels=None)

    with pytest.raises(ValueError, match="intraband tunnelling channel"):
        _with_order(stripped, 96)


def test_with_order_rewrites_only_the_quadrature_order_and_enable_flag():
    from perovskite_sim.models.config_loader import load_device_from_yaml

    stack = load_device_from_yaml(ROOT / _lane().config_path)
    rewritten = _with_order(stack, 192)
    disabled = _with_order(stack, 192, enabled=False)
    original = stack.tunnelling_channels.intraband
    updated = rewritten.tunnelling_channels.intraband

    assert updated.energy_quadrature_order == 192
    assert updated.enabled is True
    assert disabled.tunnelling_channels.intraband.enabled is False
    # Nothing else about the channel may drift when the lane re-declares it.
    assert updated.carrier == original.carrier
    assert updated.electron_effective_mass_rel == original.electron_effective_mass_rel
    assert updated.hole_effective_mass_rel == original.hole_effective_mass_rel
    # A changed order must change the document identity, or the certificate
    # could not tell two energy rungs apart.
    assert rewritten.tunnelling_channels.sha256 != stack.tunnelling_channels.sha256


# --------------------------------------------------------------------------
# One real cell — the structural claims the lane exists to certify
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_a_coarse_cell_certifies_the_structural_channel_claims():
    """The exact-zero claims are the ones worth a real solve.

    Equilibrium reciprocity and the flux-to-face-current identity are exact
    statements, so they are asserted as exact rather than with a tolerance;
    anything else here would hide a sign or bookkeeping error behind a
    threshold.
    """
    measurement = run_tunnelling_channel_qf_dc_refinement(
        _lane(), MatrixPoint(24, 1.0), ROOT
    )
    quality = _metrics(measurement, quality=True)
    observables = _metrics(measurement)

    assert quality["certified"].values[0] == 1.0
    assert quality["equilibrium_certified"].values[0] == 1.0
    # Structural, not numerical: one transmission drives both directions.
    assert quality["equilibrium_net_flux_m2_s"].values[0] == 0.0
    assert quality["equilibrium_face_current_A_m2"].values[0] == 0.0
    # The injected face current IS the reported flux, on exactly one face.
    assert quality["face_current_injection_relative_error"].values[0] == 0.0
    assert quality["injected_face_count"].values[0] == 1.0
    # A disabled family must produce no diagnostics at all.
    assert quality["disabled_family_reports_nothing"].values[0] == 1.0
    # The barrier must actually block something, or the lane measures nothing.
    assert quality["minimum_transmission_below_unity"].values[0] == 1.0
    assert observables["intraband_electron_maximum_action"].values[0] > 1.0
    # The channel must be a real contributor, not a rounding perturbation.
    assert quality["channel_flux_fraction_of_terminal_current"].values[0] > 0.01
    # The solve stays inside the solver's own accepted residual limit.
    assert quality["residual_over_solver_limit"].values[0] <= 1.0
    # Energy-order convergence over the registered ladder.
    assert quality["energy_quadrature_orders_completed"].values[0] == 3.0
    assert quality["max_energy_flux_relative_change"].values[0] < 0.05
