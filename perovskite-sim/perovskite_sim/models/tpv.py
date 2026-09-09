"""Transient photovoltage result dataclass."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from perovskite_sim.experiments.protocol import ExperimentProtocol


@dataclass(frozen=True)
class TPVDecayFit:
    """Identifiability and residuals of a single exponential voltage response."""

    tau_s: float | None
    amplitude_V: float
    status: str
    points: int
    r_squared: float | None = None
    normalized_max_error: float | None = None


@dataclass(frozen=True)
class TPVNumericalSettings:
    rtol: float
    density_atol_m3: float
    charge_coordinate_atol_V: float
    max_step_s: float
    max_charge_voltage_error_V: float
    max_current_error_A_m2: float


@dataclass(frozen=True)
class TPVResult:
    """Result of a transient photovoltage (TPV) experiment.

    The light pulse and unpulsed control start from the same open-circuit
    state. Finite-time preparation can retain slow ionic evolution, so the
    fitted response is V - V_reference, not necessarily V - V_oc.

    All quantities in SI units.
    """

    t: np.ndarray          # time array [s]
    V: np.ndarray          # voltage transient [V]
    J: np.ndarray          # terminal current density [A/m^2] (≈ 0 at OC)
    V_oc: float            # initial zero-current voltage after declared preparation
    tau: float | None      # unavailable when a single decay is not identifiable
    delta_V0: float        # initial voltage perturbation amplitude [V]
    protocol: ExperimentProtocol | None = None
    fit: TPVDecayFit | None = None
    V_reference: np.ndarray | None = None
    delta_V: np.ndarray | None = None
    valid: np.ndarray | None = None
    max_face_current_A_m2: np.ndarray | None = None
    interval_current_residual_A_m2: np.ndarray | None = None
    charge_voltage_error_V: np.ndarray | None = None
    reference_protocol: ExperimentProtocol | None = None
    numerical_settings: TPVNumericalSettings | None = None
