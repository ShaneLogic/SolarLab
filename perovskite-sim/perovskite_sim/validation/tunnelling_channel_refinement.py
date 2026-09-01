"""Registered D8 WKB tunnelling-channel refinement lane.

What this lane certifies, and what it deliberately does not
-----------------------------------------------------------
The observable is the **channel's own flux**, not the terminal current. That
distinction is load-bearing: enabling the channel on this stack moves the
terminal current by only ~1e-5 relative, because the tunnelling path sits in
parallel with the drift-diffusion flux on the same face and the rest of the
device sets the operating point. A gate on terminal current would therefore
pass whether or not the channel worked.

RETRACTED 2026-09-01 (D8-E2R): the companion claim that the channel "carries
~19 % of the terminal current" is an artifact. The QF lane hands the channel
`V_T*ln(n) - (phi+chi)` = `E_Fn + V_T*ln(N_C)`, a +1.4286 eV offset that
saturates the Fermi factors; with the true level the fraction is 6.2e-7, and
this lane's `equilibrium_net_flux_m2_s` gate passes by float64 saturation
rather than by reciprocity. Convergence orders and the injection identity are
unaffected. See docs/wkb-tunnelling-family-contract.md.

Three refinement axes, two of which are the registry's
-----------------------------------------------------
Grid and solver tolerance are the shared matrix. The energy quadrature order
is a third axis specific to this family, and it is swept **inside** each cell
and reported as quality metrics — the same shape the four existing
energy-distributed lanes use. The shared `MatrixPoint` carries exactly
`(grid, tolerance_factor)` and `ConvergenceCheck.dimension` is a closed
literal, so adding a real third axis would mean changing machinery that 43
other lanes depend on for no gain here.

Why the flux converges at all
-----------------------------
The channel is driven by the quasi-Fermi drop between the two **turning
points** of the barrier at each energy. An earlier form read it across the
anchor face's two nodes; that is a difference over one cell, so it shrank with
the mesh and the flux halved on every grid doubling. Turning points sit at
fixed physical positions, and interpolating the crossing (rather than snapping
it to a node) recovers order 1.5 — the O(h^3/2) a square-root turning point
imposes, and the same order D8-E0 measured for the bare WKB action.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from .dae_refinement import (
    _finite_option,
    _integer_option,
    _protocol_metadata,
    _string_option,
)
from .numerical_certificate import LaneDefinition, MatrixPoint
from .refinement_runner import CellMeasurement


_CHANNEL_NAME = "intraband_electron"


def _quadrature_orders(options: dict[str, Any]) -> tuple[int, ...]:
    """Read and validate the energy ladder swept inside every cell."""

    raw = options.get("energy_quadrature_orders", [96, 192, 384])
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("energy_quadrature_orders must be a list of at least two")
    orders: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError("energy_quadrature_orders must contain integers >= 2")
        orders.append(value)
    if any(right != 2 * left for left, right in zip(orders, orders[1:])):
        raise ValueError("energy_quadrature_orders must be consecutive doublings")
    return tuple(orders)


def _with_order(stack: Any, order: int, *, enabled: bool = True) -> Any:
    """The same stack with the intraband channel re-declared at one order."""

    from dataclasses import replace

    document = stack.tunnelling_channels
    if document is None or document.intraband is None:
        raise ValueError("lane config must declare an intraband tunnelling channel")
    channel = replace(
        document.intraband,
        enabled=enabled,
        energy_quadrature_order=int(order),
    )
    return replace(
        stack,
        tunnelling_channels=replace(document, intraband=channel),
    )


def _resample(x: np.ndarray, values: np.ndarray, points: int) -> np.ndarray:
    """Put a profile on a fixed number of points so shapes match across cells.

    A profile sampled on the solver grid has a different length at every grid
    refinement, and the certificate compares observables element-wise. Without
    this the lane reports a shape mismatch instead of a convergence result.
    Normalised position is the right abscissa here because the device geometry
    is identical across the matrix; only the discretisation moves.
    """

    positions = np.asarray(x, dtype=float)
    span = float(positions[-1] - positions[0])
    if span <= 0.0:
        raise ValueError("profile resampling needs a positive span")
    normalised = (positions - positions[0]) / span
    return np.interp(np.linspace(0.0, 1.0, int(points)), normalised, values)


def _state_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(np.asarray(array, dtype=float))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _execution_protocol(
    lane: LaneDefinition,
    *,
    quadrature_orders: tuple[int, ...],
    bias_V: float,
    profile_points: int,
) -> dict[str, Any]:
    """The frozen protocol every cell in the matrix claims to have executed.

    Deliberately free of anything that varies per cell. The effective Newton
    tolerance and finite-difference step ARE per-cell — they are the matrix
    axis — so putting them here would give every cell a different protocol
    hash and the certificate would report the matrix as inconsistent with
    itself rather than as a converging study. They are recorded under
    `metadata.actual` instead.
    """

    return {
        "schema_version": "solarlab-wkb-tunnelling-refinement-protocol-v1",
        "lane_id": lane.lane_id,
        "executor_version": lane.executor_version,
        "channel": _CHANNEL_NAME,
        "energy_quadrature_orders": list(quadrature_orders),
        "bias_V": bias_V,
        "illuminated": False,
        "profile_points": int(profile_points),
        "grid_values": list(lane.grid_values),
        "tolerance_factors": list(lane.tolerance_factors),
    }


def run_tunnelling_channel_qf_dc_refinement(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Execute one WKB tunnelling-channel refinement matrix cell."""

    options = lane.options
    if _string_option(options, "config_loader", "standard") != "standard":
        raise ValueError("tunnelling refinement requires config_loader='standard'")
    quadrature_orders = _quadrature_orders(options)
    grid_alpha = _finite_option(options, "grid_alpha", 2.0)
    bias_V = _finite_option(options, "bias_V", 0.2)
    profile_points = _integer_option(options, "profile_points", 17, minimum=3)
    base_newton_tolerance = _finite_option(
        options, "base_newton_residual_tolerance", 1.0e-8
    )
    base_poisson_tolerance = _finite_option(
        options, "base_poisson_tolerance_V", 1.0e-10
    )
    base_fd_step = _finite_option(options, "base_finite_difference_step", 1.0e-5)
    continuity_tolerance = _finite_option(options, "continuity_tolerance_A_m2", 2.0e-4)
    spread_tolerance = _finite_option(options, "current_spread_tolerance_A_m2", 2.0e-4)
    max_newton_iterations = _integer_option(options, "max_newton_iterations", 30)

    newton_tolerance = base_newton_tolerance * point.tolerance_factor
    poisson_tolerance = base_poisson_tolerance * point.tolerance_factor
    finite_difference_step = base_fd_step * math.sqrt(point.tolerance_factor)
    solve_controls = {
        "finite_difference_step": finite_difference_step,
        "newton_residual_tolerance": newton_tolerance,
        "poisson_tolerance_V": poisson_tolerance,
        "continuity_tolerance_A_m2": continuity_tolerance,
        "current_spread_tolerance_A_m2": spread_tolerance,
    }

    stack = load_device_from_yaml(project_root / lane.config_path)
    layers = electrical_layers(stack)
    if len(layers) != 3:
        raise ValueError("tunnelling lane config must have three electrical layers")
    grid = multilayer_grid(
        tuple(Layer(layer.thickness, point.grid) for layer in layers),
        alpha=grid_alpha,
    )

    def _solve(active_stack: Any, *, illuminated: bool, voltage: float):
        return solve_quasi_fermi_steady_state(
            grid,
            active_stack,
            V_app=voltage,
            illuminated=illuminated,
            max_newton_iterations=max_newton_iterations,
            **solve_controls,
        )

    # --- the energy ladder, swept inside this cell -------------------------
    fluxes: list[float] = []
    transmissions: list[float] = []
    results = []
    for order in quadrature_orders:
        result = _solve(_with_order(stack, order), illuminated=False, voltage=bias_V)
        diagnostics = result.tunnelling_channel_diagnostics
        if diagnostics is None:
            raise ValueError("tunnelling diagnostics missing from a certified solve")
        if diagnostics.channel_names != (_CHANNEL_NAME,):
            raise ValueError(
                f"expected exactly {_CHANNEL_NAME!r}, got {diagnostics.channel_names!r}"
            )
        fluxes.append(float(diagnostics.channel_net_flux_m2_s[0]))
        transmissions.append(float(diagnostics.channel_minimum_transmission[0]))
        results.append(result)

    finest = results[-1]
    finest_diagnostics = finest.tunnelling_channel_diagnostics
    energy_flux_changes = [
        abs(right - left) / abs(right)
        for left, right in zip(fluxes, fluxes[1:])
        if right != 0.0
    ]
    # NOTE: there is deliberately no energy-order convergence metric for the
    # transmission. The window is linspace(base, peak, order), so its first
    # point is exactly `base` at every order and the minimum transmission is
    # order-independent by construction. Reporting its "convergence" would be
    # a gate that reads 0.0 whatever the physics does. The minimum
    # transmission IS grid-dependent, so it is carried as an observable and
    # gated by the shared grid/tolerance convergence check instead.

    # --- the structural claims, checked at the finest order ---------------
    # Equilibrium reciprocity: one transmission drives both directions, so a
    # flat quasi-Fermi profile must give EXACTLY zero, not a small residual.
    equilibrium = _solve(
        _with_order(stack, quadrature_orders[-1]), illuminated=False, voltage=0.0
    )
    equilibrium_diagnostics = equilibrium.tunnelling_channel_diagnostics
    equilibrium_flux = float(equilibrium_diagnostics.channel_net_flux_m2_s[0])
    equilibrium_face_current = float(
        np.max(np.abs(equilibrium_diagnostics.electron_face_current_A_m2))
    )

    # A disabled family must be bit-identical to no family at all.
    disabled = _solve(
        _with_order(stack, quadrature_orders[-1], enabled=False),
        illuminated=False,
        voltage=bias_V,
    )
    disabled_identical = float(disabled.tunnelling_channel_diagnostics is None)

    # The injected face current must be exactly the reported flux.
    face_current = np.asarray(finest_diagnostics.electron_face_current_A_m2)
    injected = float(face_current.sum())
    expected = -Q * fluxes[-1]
    injection_error = (
        abs(injected - expected) / abs(expected) if expected != 0.0 else abs(injected)
    )
    nonzero_faces = int(np.count_nonzero(face_current))

    return CellMeasurement.from_mapping(
        {
            "observables": {
                "intraband_electron_net_flux_m2_s": fluxes[-1],
                "intraband_electron_maximum_action": (
                    -0.5 * math.log(transmissions[-1])
                    if transmissions[-1] > 0.0
                    else float("inf")
                ),
                "intraband_electron_face_current_A_m2": injected,
                "dark_terminal_current_A_m2": float(finest.current_A_m2),
                "dark_potential_V": _resample(
                    grid, np.asarray(finest.phi, dtype=float), profile_points
                ),
                "dark_electron_quasi_fermi_V": _resample(
                    grid,
                    np.asarray(finest.electron_quasi_fermi_potential_V, dtype=float),
                    profile_points,
                ),
            },
            "quality": {
                "energy_quadrature_orders_completed": float(len(quadrature_orders)),
                "max_energy_flux_relative_change": (
                    max(energy_flux_changes) if energy_flux_changes else 0.0
                ),
                "equilibrium_net_flux_m2_s": abs(equilibrium_flux),
                "equilibrium_face_current_A_m2": equilibrium_face_current,
                "face_current_injection_relative_error": injection_error,
                "injected_face_count": float(nonzero_faces),
                "disabled_family_reports_nothing": disabled_identical,
                "channel_flux_fraction_of_terminal_current": (
                    abs(Q * fluxes[-1]) / abs(finest.current_A_m2)
                    if finest.current_A_m2 != 0.0
                    else 0.0
                ),
                "max_normalized_cell_residual": float(
                    finest.max_normalized_cell_residual
                ),
                # Reported as a ratio against the solver's OWN accepted limit
                # rather than against a number chosen here, so the gate cannot
                # drift away from what `certified` actually means.
                "residual_over_solver_limit": (
                    float(finest.max_normalized_cell_residual)
                    / float(finest.numerical_residual_limit)
                    if finest.numerical_residual_limit
                    else 0.0
                ),
                "face_current_spread_A_m2": float(finest.face_current_spread_A_m2),
                "poisson_residual_C_m2": float(finest.poisson_residual_C_m2),
                "certified": float(finest.certified),
                "equilibrium_certified": float(equilibrium.certified),
                "newton_iterations": float(finest.newton_iterations),
                "minimum_transmission_below_unity": float(transmissions[-1] < 1.0),
            },
            "units": {
                "channel_flux_fraction_of_terminal_current": "1",
                "certified": "1",
                "dark_electron_quasi_fermi_V": "V",
                "dark_potential_V": "V",
                "dark_terminal_current_A_m2": "A m-2",
                "disabled_family_reports_nothing": "1",
                "energy_quadrature_orders_completed": "1",
                "equilibrium_certified": "1",
                "equilibrium_face_current_A_m2": "A m-2",
                "equilibrium_net_flux_m2_s": "m-2 s-1",
                "face_current_injection_relative_error": "1",
                "face_current_spread_A_m2": "A m-2",
                "injected_face_count": "1",
                "intraband_electron_face_current_A_m2": "A m-2",
                "intraband_electron_maximum_action": "1",
                "intraband_electron_net_flux_m2_s": "m-2 s-1",
                "max_energy_flux_relative_change": "1",
                "max_normalized_cell_residual": "1",
                "residual_over_solver_limit": "1",
                "minimum_transmission_below_unity": "1",
                "newton_iterations": "1",
                "poisson_residual_C_m2": "C m-2",
            },
            "metadata": {
                **_protocol_metadata(
                    _execution_protocol(
                        lane,
                        quadrature_orders=quadrature_orders,
                        bias_V=bias_V,
                        profile_points=profile_points,
                    )
                ),
                "actual": {
                    "channel_document_sha256": (finest_diagnostics.identity_sha256),
                    "effective_finite_difference_step": finite_difference_step,
                    "effective_newton_residual_tolerance": newton_tolerance,
                    "effective_poisson_tolerance_V": poisson_tolerance,
                    "energy_ladder_flux_m2_s": list(fluxes),
                    # T itself stays in the provenance rather than being
                    # gated: exp(-2S) amplifies the exponent's error by 2S,
                    # so a converged action reads as an unconverged T.
                    "energy_ladder_minimum_transmission": list(transmissions),
                    "finest_minimum_transmission": transmissions[-1],
                    "grid_intervals_per_layer": int(point.grid),
                    "grid_nodes": int(grid.size),
                    "state_sha256": {
                        "dark_finest": _state_sha256(finest.y, finest.phi),
                        "equilibrium": _state_sha256(equilibrium.y, equilibrium.phi),
                    },
                },
            },
        }
    )


__all__ = ["run_tunnelling_channel_qf_dc_refinement"]
