"""Frequency-domain impedance for the certified local QF device envelope.

This adapter intentionally shares the cancellation-safe Scharfetter-Gummel
currents and eliminated Poisson-Boltzmann operator used by the c-Si QF J-V
certificate. Unsupported physics fails through the same capability gate; the
module is not a silent replacement for the general ion-aware transient path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from perovskite_sim.constants import EPS_0
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    QuasiFermiSteadyStateError,
    QuasiFermiSteadyStateResult,
    _QuasiFermiSystem,
    _require_supported,
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.solver.mol import (
    MaterialArrays,
    _harmonic_face_average,
    build_material_arrays,
    poisson_right_boundary,
)
from perovskite_sim.solver.small_signal import (
    FrequencyDomainResult,
    SmallSignalEvaluation,
    SmallSignalLinearizationError,
    solve_frequency_domain,
)


MAX_LINEAR_PERTURBATION_V = 0.02
DC_CONTINUATION_STEP_V = 0.005
DEFAULT_MAX_RELATIVE_FACE_SPREAD = 5.0e-4
DEFAULT_MAX_BACKWARD_ERROR = 1.0e-10
ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class QuasiFermiImpedanceResult:
    """Frequency response plus the DC and all-face physical certificates."""

    frequencies: np.ndarray
    Z: np.ndarray
    Y: np.ndarray
    Y_faces: np.ndarray
    max_relative_face_spread: np.ndarray
    reciprocal_condition: np.ndarray
    backward_error: np.ndarray
    dc_state: QuasiFermiSteadyStateResult


class QuasiFermiImpedanceError(SmallSignalLinearizationError):
    """The QF operating point or AC response lacks a physical certificate."""


def _contact_quasi_fermi_increments(
    dqfn: np.ndarray,
    dqfp: np.ndarray,
    qfn_reference: np.ndarray,
    qfp_reference: np.ndarray,
    mat: MaterialArrays,
    V_app: float,
) -> None:
    """Apply fixed-density contacts without collapsing QF increments."""
    phi_right = poisson_right_boundary(mat, V_app)
    thermal_voltage = mat.V_T_device
    dqfn[0] = thermal_voltage * np.log(mat.n_L) - mat.chi[0] - qfn_reference[0]
    dqfp[0] = (
        thermal_voltage * np.log(mat.p_L)
        + mat.chi[0]
        + mat.Eg[0]
        - qfp_reference[0]
    )
    dqfn[-1] = thermal_voltage * np.log(mat.n_R) - (
        phi_right + mat.chi[-1]
    ) - qfn_reference[-1]
    dqfp[-1] = thermal_voltage * np.log(mat.p_R) + (
        phi_right + mat.chi[-1] + mat.Eg[-1]
    ) - qfp_reference[-1]


def run_quasi_fermi_impedance(
    x: np.ndarray,
    stack: DeviceStack,
    frequencies: np.ndarray,
    *,
    V_dc: float = 0.0,
    delta_V: float = 0.01,
    illuminated: bool = False,
    mat: MaterialArrays | None = None,
    dc_state: QuasiFermiSteadyStateResult | None = None,
    state_step: float = 1.0e-5,
    voltage_step: float = 1.0e-5,
    max_relative_face_spread: float = DEFAULT_MAX_RELATIVE_FACE_SPREAD,
    max_backward_error: float = DEFAULT_MAX_BACKWARD_ERROR,
    progress: ProgressCallback | None = None,
) -> QuasiFermiImpedanceResult:
    """Solve impedance by linearizing about a certified QF operating point.

    ``delta_V`` is the nominal experimental amplitude. The linear operator is
    solved per volt, so the returned impedance is amplitude-independent; the
    value is still required to lie strictly below 20 mV to preserve the public
    small-signal protocol. ``state_step`` and ``voltage_step`` control only the
    central finite differences used to assemble the linear operator.
    """
    grid = np.asarray(x, dtype=float)
    if grid.ndim != 1 or grid.size < 3 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("x must be a strictly increasing one-dimensional grid")
    if not np.isfinite(delta_V) or not 0.0 < delta_V < MAX_LINEAR_PERTURBATION_V:
        raise ValueError(
            "delta_V must be finite, positive, and below the 20 mV "
            "small-signal limit"
        )
    if (
        not np.isfinite(max_relative_face_spread)
        or max_relative_face_spread <= 0.0
    ):
        raise ValueError("max_relative_face_spread must be finite and positive")
    if not np.isfinite(max_backward_error) or max_backward_error <= 0.0:
        raise ValueError("max_backward_error must be finite and positive")

    material = build_material_arrays(grid, stack) if mat is None else mat
    _require_supported(material)
    operating_point = dc_state
    if operating_point is None:
        try:
            operating_point = solve_quasi_fermi_steady_state(
                grid,
                stack,
                V_app=float(V_dc),
                illuminated=illuminated,
                mat=material,
            )
        except QuasiFermiSteadyStateError as direct_error:
            # A nearby, slightly more reverse-biased state is a deterministic
            # continuation seed for high-conductivity junctions whose direct
            # Newton solve can stall at machine-scale corrections. Both the
            # seed and target still pass the full independent QF certificate;
            # no residual or current tolerance is relaxed.
            seed_voltage = float(V_dc) - DC_CONTINUATION_STEP_V
            try:
                seed = solve_quasi_fermi_steady_state(
                    grid,
                    stack,
                    V_app=seed_voltage,
                    illuminated=illuminated,
                    mat=material,
                )
                operating_point = solve_quasi_fermi_steady_state(
                    grid,
                    stack,
                    V_app=float(V_dc),
                    illuminated=illuminated,
                    mat=material,
                    initial_state=seed,
                )
            except QuasiFermiSteadyStateError as continuation_error:
                raise QuasiFermiImpedanceError(
                    "could not certify the impedance DC state directly or "
                    f"through the {DC_CONTINUATION_STEP_V:g} V reverse-bias "
                    f"continuation seed; direct: {direct_error}; "
                    f"continuation: {continuation_error}"
                ) from continuation_error
    if not operating_point.certified:
        raise QuasiFermiImpedanceError(
            "frequency-domain impedance requires a certified DC state"
        )
    if not np.isclose(operating_point.V_app, V_dc, rtol=0.0, atol=1.0e-12):
        raise QuasiFermiImpedanceError(
            "DC-state voltage does not match the requested impedance bias"
        )
    if bool(operating_point.illuminated) != bool(illuminated):
        raise QuasiFermiImpedanceError(
            "DC-state illumination does not match the impedance protocol"
        )
    if operating_point.y.shape[0] < 2 * grid.size:
        raise QuasiFermiImpedanceError("DC-state vector does not match the grid")

    system = _QuasiFermiSystem(
        grid,
        stack,
        material,
        float(V_dc),
        poisson_tolerance_V=1.0e-13,
        poisson_max_iterations=100,
    )
    interior_count = grid.size - 2
    coordinate = np.zeros(2 * interior_count, dtype=float)
    thermal_voltage = material.V_T_device
    eps_face = EPS_0 * _harmonic_face_average(material.eps_r)
    dx = np.diff(grid)
    face_weights = dx / float(grid[-1] - grid[0])
    polarity = float(material.junction_polarity)

    qfn_reference = operating_point.electron_quasi_fermi_reference_V
    qfp_reference = operating_point.hole_quasi_fermi_reference_V
    dqfn_dc = operating_point.electron_quasi_fermi_increment_V
    dqfp_dc = operating_point.hole_quasi_fermi_increment_V
    if any(
        value is None
        for value in (qfn_reference, qfp_reference, dqfn_dc, dqfp_dc)
    ):
        raise QuasiFermiImpedanceError(
            "DC state lacks the cancellation-safe QF reference/increment data"
        )
    qfn_reference = np.asarray(qfn_reference, dtype=float)
    qfp_reference = np.asarray(qfp_reference, dtype=float)
    dqfn_dc = np.asarray(dqfn_dc, dtype=float)
    dqfp_dc = np.asarray(dqfp_dc, dtype=float)
    qf_arrays = (qfn_reference, qfp_reference, dqfn_dc, dqfp_dc)
    if any(
        value.shape != grid.shape or not np.all(np.isfinite(value))
        for value in qf_arrays
    ):
        raise QuasiFermiImpedanceError(
            "DC-state QF reference/increment arrays are invalid for the grid"
        )
    if not (
        np.array_equal(qfn_reference, system.qfn0)
        and np.array_equal(qfp_reference, system.qfp0)
    ):
        raise QuasiFermiImpedanceError(
            "DC-state QF reference does not match the impedance operator"
        )

    def evaluate(state_coordinate: np.ndarray, voltage: float) -> SmallSignalEvaluation:
        dqfn = dqfn_dc.copy()
        dqfp = dqfp_dc.copy()
        dqfn[1:-1] += thermal_voltage * state_coordinate[:interior_count]
        dqfp[1:-1] += thermal_voltage * state_coordinate[interior_count:]
        _contact_quasi_fermi_increments(
            dqfn,
            dqfp,
            qfn_reference,
            qfp_reference,
            material,
            voltage,
        )
        value = system.evaluate_quasi_fermi_increments(
            dqfn,
            dqfp,
            1.0 if illuminated else 0.0,
            V_app=voltage,
        )
        storage = np.r_[
            value.y[1 : grid.size - 1],
            value.y[grid.size + 1 : 2 * grid.size - 1],
        ]
        rate = np.r_[value.rate_n[1:-1], value.rate_p[1:-1]]
        # QF currents are in the raw transport orientation. Multiplication by
        # junction polarity converts them directly to passive convention;
        # this is the negative of the solar-output convention used by J-V.
        conduction_passive = polarity * (value.current_n + value.current_p)
        electric_field = -np.diff(value.phi) / dx
        displacement_charge_passive = polarity * eps_face * electric_field
        return SmallSignalEvaluation(
            storage=storage,
            rate=rate,
            conduction_current_faces=conduction_passive,
            displacement_charge_faces=displacement_charge_passive,
        )

    response: FrequencyDomainResult = solve_frequency_domain(
        evaluate,
        coordinate,
        float(V_dc),
        np.asarray(frequencies, dtype=float),
        state_step=state_step,
        voltage_step=voltage_step,
        face_weights=face_weights,
        progress=progress,
    )
    if np.any(response.max_relative_face_spread > max_relative_face_spread):
        worst = float(np.max(response.max_relative_face_spread))
        raise QuasiFermiImpedanceError(
            "frequency-domain current-continuity certificate failed: "
            f"relative all-face admittance spread {worst:.6g} exceeds "
            f"{max_relative_face_spread:.6g}"
        )
    if np.any(response.backward_error > max_backward_error):
        worst = float(np.max(response.backward_error))
        raise QuasiFermiImpedanceError(
            "frequency-domain linear-solve certificate failed: "
            f"componentwise backward error {worst:.6g} exceeds "
            f"{max_backward_error:.6g}"
        )
    return QuasiFermiImpedanceResult(
        frequencies=response.frequencies,
        Z=response.impedance,
        Y=response.admittance,
        Y_faces=response.admittance_faces,
        max_relative_face_spread=response.max_relative_face_spread,
        reciprocal_condition=response.reciprocal_condition,
        backward_error=response.backward_error,
        dc_state=operating_point,
    )


__all__ = [
    "DC_CONTINUATION_STEP_V",
    "DEFAULT_MAX_BACKWARD_ERROR",
    "DEFAULT_MAX_RELATIVE_FACE_SPREAD",
    "MAX_LINEAR_PERTURBATION_V",
    "QuasiFermiImpedanceError",
    "QuasiFermiImpedanceResult",
    "run_quasi_fermi_impedance",
]
