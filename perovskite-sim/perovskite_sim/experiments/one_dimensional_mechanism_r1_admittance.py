"""Finite-window R1 step-to-admittance reconstruction, without certification.

The contract is ``docs/OneDimensionalMechanismR1AdmittanceV1.md``. All inputs
are SI, currents are voltage-conjugate signed currents, and the phasor
convention is exp(+i*omega*t). Missing error information stays unbounded.
This module never estimates G0 from a trajectory or fits/extrapolates a tail.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.integrate import quad


@dataclass(frozen=True)
class R1AdmittanceErrors:
    """Caller-supplied absolute uncertainties; ``None`` means unknown.

    The three integral bounds are L1 bounds on the *normalized residual*
    r = (j_reg-j_dc)/a-G0: the omitted [0,t_first), omitted (T,infinity),
    and interpolation error on [t_first,T], respectively, in F/m2.
    They require independent evidence; finite-window differences alone are
    not rigorous infinite-tail bounds. Sample errors describe node errors,
    excluding the separately supplied baseline/G0/impulse errors. Voltage,
    time and frequency coordinates are treated as exact declared inputs.
    """

    early_omission_integral_F_m2: float | None = None
    tail_omission_integral_F_m2: float | None = None
    interpolation_integral_F_m2: float | None = None
    current_A_m2: object = None
    baseline_current_A_m2: float | None = None
    dc_conductance_S_m2: float | None = None
    impulse_charge_C_m2: float | None = None


@dataclass(frozen=True)
class R1AdmittanceReconstruction:
    """Numerical estimate with explicit, conditional component uncertainties.

    Every error is an absolute complex-modulus budget in S/m2, and therefore
    also bounds either component conditionally on the supplied bounds. The
    total includes estimated quadrature and floating-point errors; it is not
    a mathematically certified bound. Arrays are defensive read-only copies.
    No result from this type establishes a valid frequency band.
    """

    frequency_Hz: np.ndarray
    omega_rad_s: np.ndarray
    admittance_S_m2: np.ndarray
    residual_integral_F_m2: np.ndarray
    impulse_capacitance_F_m2: float
    dc_conductance_S_m2: float
    first_time_s: float
    last_time_s: float
    error_components_S_m2: Mapping[str, np.ndarray]
    total_error_estimate_S_m2: np.ndarray
    unknown_error_sources: tuple[str, ...]
    quadrature_absolute_tolerance_F_m2: float
    quadrature_relative_tolerance: float
    scope: str = "finite_window_reconstruction_only"

    def __post_init__(self):
        for name in ("frequency_Hz", "omega_rad_s", "admittance_S_m2",
                     "residual_integral_F_m2", "total_error_estimate_S_m2"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(self, "error_components_S_m2", MappingProxyType({
            name: _readonly(value) for name, value in self.error_components_S_m2.items()
        }))


class R1AdmittanceIntegrationError(RuntimeError):
    """A requested oscillatory integral did not finish with finite estimates."""


def _readonly(value):
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


def _scalar(value, name, *, nonnegative=False, positive=False):
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf" or array.dtype.kind == "b":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(array)
    if not np.isfinite(result) or (nonnegative and result < 0) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite and satisfy its sign constraint")
    return result


def _vector(value, name, *, minimum_size=1):
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size < minimum_size or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real one-dimensional array")
    array = np.array(raw, dtype=float, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _times(value):
    time = _vector(value, "time_s", minimum_size=2)
    if time[0] <= 0 or np.any(np.diff(time) <= 0):
        raise ValueError("time_s must be strictly increasing positive regular-current times; exclude 0-/0+")
    return time


def _nonnegative_array(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.shape not in ((), shape):
        raise ValueError(f"{name} must be a nonnegative scalar or have shape {shape}")
    array = np.broadcast_to(np.asarray(raw, dtype=float), shape)
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must contain finite nonnegative bounds")
    return array


def linear_interpolation_integral_bound(time_s, second_derivative_bound_S_m2_s2):
    """Bound integral |r-linear(r)| dt using independently bounded curvature.

    A scalar M applies to every interval; an array must have len(time_s)-1
    entries, each bounding |r''| throughout that interval. The returned bound
    is sum(M_i*h_i**3/12), in F/m2. Curvature is never inferred from samples.
    Node errors are excluded and must be supplied separately.
    """
    time = _times(time_s)
    curvature = _nonnegative_array(second_derivative_bound_S_m2_s2,
                                   (len(time)-1,), "second_derivative_bound_S_m2_s2")
    with np.errstate(over="ignore", invalid="ignore"):
        bound = float(np.sum(curvature*np.diff(time)**3/12))
    if not np.isfinite(bound):
        raise ValueError("interpolation bound arithmetic is nonfinite")
    return bound


def _integrate_piecewise_linear(time, residual, omega, absolute_tolerance, relative_tolerance):
    """QUADPACK cos/sin weights integrate each explicitly linear interval."""
    pieces = []
    estimates = []
    widths = np.diff(time)
    for index, (left, width) in enumerate(zip(time[:-1], widths)):
        r_left, r_right = residual[index:index+2]
        # Integration over u in [0,1] avoids subtracting nearby time values.
        def linear(u):
            return (1-u)*r_left + u*r_right

        component = []
        for weight in ("cos", "sin"):
            result = quad(linear, 0., 1., weight=weight, wvar=omega*width,
                          epsabs=absolute_tolerance/(2*len(widths))/width,
                          epsrel=relative_tolerance, full_output=1)
            if len(result) != 3:
                message = result[3] if len(result) > 3 else "missing quadrature diagnostics"
                raise R1AdmittanceIntegrationError(f"interval {index}, {weight}: {message}")
            value, error = result[:2]
            if not np.isfinite(value) or not np.isfinite(error) or error < 0:
                raise R1AdmittanceIntegrationError(f"interval {index}, {weight}: nonfinite/negative estimate")
            component.append(value)
            estimates.append(width*error)
        pieces.append(width*np.exp(-1j*omega*left)*(component[0]-1j*component[1]))
    integral = complex(fsum(value.real for value in pieces), fsum(value.imag for value in pieces))
    return integral, fsum(estimates)


def reconstruct_admittance(
    frequency_Hz, *, time_s, regular_current_A_m2, step_amplitude_V,
    baseline_current_A_m2, dc_conductance_S_m2, impulse_charge_C_m2,
    errors: R1AdmittanceErrors | None = None,
    quadrature_absolute_tolerance_F_m2=1e-12, quadrature_relative_tolerance=1e-10,
):
    """Reconstruct G0 + i*omega*Qimp/a + i*omega*integral(r*exp(-iwt)).

    Only the measured [t_first,T] interval is integrated, using a piecewise
    linear interpolant and SciPy's weighted oscillatory quadrature. Qimp is
    separate from regular current; G0 must come from an independent DC study.
    Signed nonzero steps are accepted, without taking absolute current/charge.
    Frequency is in Hz (nonnegative, strictly increasing), not rad/s.

    Unknown omissions/uncertainties become infinity in the result, including
    at zero frequency, so absent evidence never silently becomes a zero bound.
    The estimate at zero frequency itself is the supplied G0 exactly.
    """
    frequency = _vector(frequency_Hz, "frequency_Hz")
    if frequency[0] < 0 or np.any(np.diff(frequency) <= 0):
        raise ValueError("frequency_Hz must be nonnegative and strictly increasing")
    time = _times(time_s)
    current = _vector(regular_current_A_m2, "regular_current_A_m2", minimum_size=2)
    if current.shape != time.shape:
        raise ValueError("regular_current_A_m2 and time_s shapes must agree; broadcasting is forbidden")
    amplitude = _scalar(step_amplitude_V, "step_amplitude_V")
    if amplitude == 0:
        raise ValueError("step_amplitude_V must be nonzero")
    baseline = _scalar(baseline_current_A_m2, "baseline_current_A_m2")
    dc = _scalar(dc_conductance_S_m2, "dc_conductance_S_m2")
    impulse = _scalar(impulse_charge_C_m2, "impulse_charge_C_m2")
    atol = _scalar(quadrature_absolute_tolerance_F_m2,
                   "quadrature_absolute_tolerance_F_m2", positive=True)
    rtol = _scalar(quadrature_relative_tolerance, "quadrature_relative_tolerance", positive=True)
    if rtol >= 1:
        raise ValueError("quadrature_relative_tolerance must be less than 1")
    if errors is None:
        errors = R1AdmittanceErrors()
    if not isinstance(errors, R1AdmittanceErrors):
        raise TypeError("errors must be R1AdmittanceErrors")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        omega = 2*np.pi*frequency
        residual = (current-baseline)/amplitude-dc
        capacitance = impulse/amplitude
        phase = omega*time[-1]
        widths = np.diff(time)
        # This scale also covers cancellation while subtracting DC/baseline.
        normalization_scale = (np.abs(current)+abs(baseline))/abs(amplitude)+abs(dc)
        integral_scale = np.sum(widths*(normalization_scale[:-1]/2+normalization_scale[1:]/2))
    if not all(np.all(np.isfinite(value)) for value in
               (omega, residual, capacitance, phase, normalization_scale, integral_scale)):
        raise ValueError("reconstruction normalization or phase arithmetic is nonfinite")

    integral = np.empty(len(frequency), dtype=complex)
    quadrature = np.empty(len(frequency))
    for index, w in enumerate(omega):
        integral[index], quadrature[index] = _integrate_piecewise_linear(time, residual, w, atol, rtol)
    with np.errstate(over="ignore", invalid="ignore"):
        admittance = dc+1j*omega*capacitance+1j*omega*integral
    if not np.all(np.isfinite(admittance)) or not np.all(np.isfinite(integral)):
        raise ValueError("reconstructed admittance arithmetic is nonfinite")

    budgets = {}
    unknown = []

    def add_budget(name, value, factor):
        if value is None:
            budgets[name] = np.full(len(frequency), np.inf)
            unknown.append(name)
        else:
            bound = _scalar(value, name, nonnegative=True)
            with np.errstate(over="ignore", invalid="ignore"):
                budgets[name] = bound*factor
            if not np.all(np.isfinite(budgets[name])):
                raise ValueError(f"{name} error propagation arithmetic is nonfinite")

    add_budget("early_omission", errors.early_omission_integral_F_m2, omega)
    add_budget("tail_omission", errors.tail_omission_integral_F_m2, omega)
    add_budget("interpolation", errors.interpolation_integral_F_m2, omega)
    if errors.current_A_m2 is None:
        add_budget("sample_current", None, omega)
    else:
        node_errors = _nonnegative_array(errors.current_A_m2, time.shape, "current_A_m2")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            sample_integral = np.sum(widths*(node_errors[:-1]/2+node_errors[1:]/2))/abs(amplitude)
        add_budget("sample_current", sample_integral, omega)
    duration = time[-1]-time[0]
    add_budget("baseline_current", errors.baseline_current_A_m2, omega*duration/abs(amplitude))
    # G0 occurs both outside the integral and inside r: preserve both errors.
    add_budget("dc_conductance", errors.dc_conductance_S_m2, 1+omega*duration)
    add_budget("impulse_charge", errors.impulse_charge_C_m2, omega/abs(amplitude))
    budgets["quadrature_estimate"] = omega*quadrature
    # Engineering estimate, not directed-rounding arithmetic: includes phase
    # argument uncertainty, normalization cancellation and complex summation.
    eps = np.finfo(float).eps
    with np.errstate(over="ignore", invalid="ignore"):
        budgets["floating_point_estimate"] = (
            omega*integral_scale*(64*eps+np.minimum(2., 16*eps*phase))
            + 64*eps*(abs(dc)+omega*abs(capacitance)+omega*np.abs(integral))
        )
        total = np.sum(tuple(budgets.values()), axis=0)
    if any(not np.all(np.isfinite(value)) for name, value in budgets.items() if name not in unknown):
        raise ValueError("reconstruction error estimate arithmetic is nonfinite")
    if not unknown and not np.all(np.isfinite(total)):
        raise ValueError("total reconstruction error estimate is nonfinite")
    return R1AdmittanceReconstruction(
        frequency, omega, admittance, integral, capacitance, dc, float(time[0]), float(time[-1]),
        budgets, total, tuple(unknown), atol, rtol,
    )


__all__ = [
    "R1AdmittanceErrors", "R1AdmittanceReconstruction", "R1AdmittanceIntegrationError",
    "linear_interpolation_integral_bound", "reconstruct_admittance",
]
