"""Dark DC J-V sweep, exponential diode window selection, and ideality extraction.

Solves genuine dark steady-state operating points (``generation_scale = 0.0``) on
the unified assembled 1D drift-diffusion (or coupled ionic-electronic) residual.
Both terminal sign conventions are reported explicitly:

- ``pv_output_current_A_m2``: photovoltaic output convention (negative under
  forward dark injection ``V > 0``, matching ``SteadyJVResult.output_current_A_m2``
  and legacy ``DarkJVResult.J``).
- ``voltage_conjugate_current_A_m2``: passive port input convention (positive
  under forward dark injection ``V > 0``).

Failed points record ``np.nan`` and an explicit ``point_reasons`` entry; ``0.0``
is never substituted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np

from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.equations.dc_1d import CompiledDevice1D
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, IonicState1D
from solarlab.experiments.suns_voc import (
    default_ionic_steady_options,
    solve_ionic_dc_operating_point,
)
from solarlab.reproducibility.job_provenance import current_provenance
from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions
from solarlab.solvers.nonlinear.steady_1d import (
    SteadyOptions,
    SteadyStateError,
    solve_steady_state,
)

__all__ = [
    "DarkJVOptions",
    "DarkJVResult",
    "extract_dark_diode_parameters",
    "run_dark_jv",
    "select_dark_diode_window",
]

BOLTZMANN_J_K = 1.380649e-23
ProgressCallback = Callable[[str, int, int, str], None]


def _ro_float(value: Any) -> np.ndarray:
    arr = np.array(value, dtype=np.float64, copy=True)
    arr.setflags(write=False)
    return arr


def _ro_bool(value: Any) -> np.ndarray:
    arr = np.array(value, dtype=np.bool_, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class DarkJVOptions:
    """Explicit numerical and diode-fitting controls for dark DC J-V."""

    steady: SteadyOptions = field(default_factory=SteadyOptions)
    ionic_steady: IonicSteadyOptions | None = None
    min_diode_window_points: int = 8
    current_floor_A_m2: float = 1e-6
    min_forward_voltage_V: float = 0.01
    require_all_converged: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.steady, SteadyOptions):
            raise ValueError("steady must be an explicit SteadyOptions instance")
        if self.ionic_steady is not None and not isinstance(
            self.ionic_steady, IonicSteadyOptions
        ):
            raise ValueError(
                "ionic_steady must be None or an explicit IonicSteadyOptions"
            )
        if (
            type(self.min_diode_window_points) is not int
            or self.min_diode_window_points < 3
        ):
            raise ValueError("min_diode_window_points must be an integer >= 3")
        if not math.isfinite(self.current_floor_A_m2) or self.current_floor_A_m2 <= 0.0:
            raise ValueError("current_floor_A_m2 must be finite and positive")
        if (
            not math.isfinite(self.min_forward_voltage_V)
            or self.min_forward_voltage_V < 0.0
        ):
            raise ValueError("min_forward_voltage_V must be finite and nonnegative")


@dataclass(frozen=True)
class DarkJVResult:
    """Genuine dark DC J-V curve and extracted Shockley diode metrics."""

    voltage_V: np.ndarray
    pv_output_current_A_m2: np.ndarray
    voltage_conjugate_current_A_m2: np.ndarray
    converged_mask: np.ndarray
    point_reasons: tuple[str | None, ...]
    local_ideality_factor: np.ndarray
    fit_mask: np.ndarray
    n_ideality: float
    saturation_current_J0_A_m2: float
    V_fit_lo_V: float
    V_fit_hi_V: float
    r_squared: float
    series_resistance_est_ohm_m2: float
    fit_failure_reason: str | None
    max_current_spread_A_m2: float
    temperature_K: float
    voltage_polarity: float
    compiled_identity: str
    solver_controls: dict[str, Any]
    job_provenance: dict[str, Any] | None = field(default_factory=current_provenance)

    def __post_init__(self) -> None:
        for name in (
            "voltage_V",
            "pv_output_current_A_m2",
            "voltage_conjugate_current_A_m2",
            "local_ideality_factor",
        ):
            object.__setattr__(self, name, _ro_float(getattr(self, name)))
        for mask_name in ("converged_mask", "fit_mask"):
            object.__setattr__(self, mask_name, _ro_bool(getattr(self, mask_name)))

    @property
    def V(self) -> np.ndarray:
        return self.voltage_V

    @property
    def J(self) -> np.ndarray:
        return self.pv_output_current_A_m2

    @property
    def J_0(self) -> float:
        return self.saturation_current_J0_A_m2

    @property
    def V_fit_lo(self) -> float:
        return self.V_fit_lo_V

    @property
    def V_fit_hi(self) -> float:
        return self.V_fit_hi_V

    @property
    def completed(self) -> bool:
        return bool(np.all(self.converged_mask))


def select_dark_diode_window(
    voltage_V: Sequence[float] | np.ndarray,
    pv_output_current_A_m2: Sequence[float] | np.ndarray,
    *,
    converged_mask: Sequence[bool] | np.ndarray | None = None,
    min_points: int = 8,
    J_floor_A_m2: float = 1e-6,
    min_forward_voltage_V: float = 0.01,
) -> tuple[np.ndarray, str | None]:
    """Identify the contiguous exponential forward-injection diode window.

    Selects ``min_points`` valid forward-injecting samples (``V > min_forward_voltage_V``,
    ``J_pv < 0``, ``|J_pv| > J_floor_A_m2``) with minimal mean ``|d^2(ln|J|)/dV^2|``
    and grows outward while ``|d^2(ln|J|)/dV^2| <= max(10 * best_mean, 1e-10)``.
    """
    v_arr: np.ndarray = np.asarray(voltage_V, dtype=float)
    j_pv: np.ndarray = np.asarray(pv_output_current_A_m2, dtype=float)
    if v_arr.ndim != 1 or j_pv.shape != v_arr.shape:
        raise ValueError("voltage_V and pv_output_current_A_m2 must be matching 1D arrays")
    conv = (
        np.isfinite(v_arr) & np.isfinite(j_pv)
        if converged_mask is None
        else np.asarray(converged_mask, dtype=bool)
        & np.isfinite(v_arr)
        & np.isfinite(j_pv)
    )
    abs_j = np.abs(j_pv)
    valid = (
        conv
        & (v_arr > min_forward_voltage_V)
        & (j_pv < 0.0)
        & (abs_j > J_floor_A_m2)
    )
    idx = np.flatnonzero(valid)
    fit_mask: np.ndarray = np.zeros(v_arr.shape, dtype=bool)
    if idx.size < min_points:
        return (
            fit_mask,
            f"insufficient_forward_injection_points:{idx.size}_need_{min_points}",
        )

    v_valid = v_arr[idx]
    if np.any(np.diff(v_valid) <= 0.0):
        raise ValueError("valid forward voltage samples must be strictly ascending")
    log_j = np.log(abs_j[idx])
    dlog = np.gradient(log_j, v_valid)
    d2log = np.gradient(dlog, v_valid)

    w_len = min_points
    best_i = 0
    best_score = float("inf")
    for i in range(0, int(idx.size) - w_len + 1):
        # Also require positive local log-slope across candidate window.
        if np.any(dlog[i : i + w_len] <= 0.0):
            continue
        score = float(np.mean(np.abs(d2log[i : i + w_len])))
        if score < best_score:
            best_score = score
            best_i = i

    if not math.isfinite(best_score):
        return fit_mask, "no_positive_slope_exponential_window"

    threshold = max(10.0 * best_score, 1e-10)
    lo = best_i
    hi = best_i + w_len - 1
    while lo > 0 and abs(d2log[lo - 1]) <= threshold and dlog[lo - 1] > 0.0:
        lo -= 1
    while (
        hi < int(idx.size) - 1
        and abs(d2log[hi + 1]) <= threshold
        and dlog[hi + 1] > 0.0
    ):
        hi += 1

    # Keep only a contiguous slice in the original voltage array if non-converged
    # gaps existed inside idx[lo:hi+1].
    fit_indices = idx[lo : hi + 1]
    fit_mask[fit_indices] = True
    return fit_mask, None


def extract_dark_diode_parameters(
    voltage_V: Sequence[float] | np.ndarray,
    pv_output_current_A_m2: Sequence[float] | np.ndarray,
    *,
    temperature_K: float,
    converged_mask: Sequence[bool] | np.ndarray | None = None,
    min_points: int = 8,
    J_floor_A_m2: float = 1e-6,
    min_forward_voltage_V: float = 0.01,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    float,
    float,
    str | None,
]:
    """Compute local ideality, exponential diode fit, and high-bias R_s estimate."""
    v_arr: np.ndarray = np.asarray(voltage_V, dtype=float)
    j_pv: np.ndarray = np.asarray(pv_output_current_A_m2, dtype=float)
    if not math.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError("temperature_K must be finite and positive")
    v_thermal = BOLTZMANN_J_K * float(temperature_K) / ELEMENTARY_CHARGE_C

    conv = (
        np.isfinite(v_arr) & np.isfinite(j_pv)
        if converged_mask is None
        else np.asarray(converged_mask, dtype=bool)
        & np.isfinite(v_arr)
        & np.isfinite(j_pv)
    )
    j_conj = -j_pv
    fwd_valid = conv & (v_arr > min_forward_voltage_V) & (j_conj > J_floor_A_m2)
    local_n: np.ndarray = np.full(v_arr.shape, np.nan, dtype=float)
    if int(np.count_nonzero(fwd_valid)) >= 2:
        v_fwd = v_arr[fwd_valid]
        if np.all(np.diff(v_fwd) > 0.0):
            dln_j_dv = np.gradient(np.log(j_conj[fwd_valid]), v_fwd)
            with np.errstate(divide="ignore", invalid="ignore"):
                n_vals = np.where(dln_j_dv > 0.0, 1.0 / (v_thermal * dln_j_dv), np.nan)
            local_n[fwd_valid] = n_vals

    fit_mask, reason = select_dark_diode_window(
        v_arr,
        j_pv,
        converged_mask=conv,
        min_points=min_points,
        J_floor_A_m2=J_floor_A_m2,
        min_forward_voltage_V=min_forward_voltage_V,
    )
    if reason is not None or not np.any(fit_mask):
        return (
            local_n,
            fit_mask,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            reason or "empty_diode_window",
        )

    v_fit = v_arr[fit_mask]
    log_j_fit = np.log(j_conj[fit_mask])
    slope, intercept = np.polyfit(v_fit, log_j_fit, 1)
    if not math.isfinite(slope) or slope <= 0.0:
        return (
            local_n,
            fit_mask,
            float("nan"),
            float("nan"),
            float(v_fit[0]),
            float(v_fit[-1]),
            float("nan"),
            float("nan"),
            "nonpositive_diode_slope",
        )

    n_id = float(1.0 / (v_thermal * slope))
    j0 = float(math.exp(intercept))
    pred = slope * v_fit + intercept
    ss_res = float(np.sum((log_j_fit - pred) ** 2))
    ss_tot = float(np.sum((log_j_fit - np.mean(log_j_fit)) ** 2))
    r_sq = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 1.0

    # Estimate R_s [Ohm * m^2] from post-window series roll-off where V > V_fit_hi.
    hi_v = float(v_fit[-1])
    post_mask = conv & (v_arr > hi_v) & (j_conj > J_floor_A_m2)
    rs_est = float("nan")
    if np.any(post_mask):
        v_post = v_arr[post_mask]
        j_post = j_conj[post_mask]
        v_ideal = (np.log(j_post) - intercept) / slope
        dv = v_post - v_ideal
        pos_dv = dv > 0.0
        if np.any(pos_dv):
            rs_est = float(np.median(dv[pos_dv] / j_post[pos_dv]))

    return (
        local_n,
        fit_mask,
        n_id,
        j0,
        float(v_fit[0]),
        hi_v,
        r_sq,
        rs_est,
        None,
    )


def _normalize_voltages(
    voltage_V: Sequence[float] | np.ndarray | None,
    *,
    V_min: float,
    V_max: float,
    n_points: int,
) -> np.ndarray:
    if voltage_V is not None:
        arr: np.ndarray = np.asarray([float(v) for v in voltage_V], dtype=float)
        if arr.ndim != 1 or arr.size < 3 or not np.all(np.isfinite(arr)):
            raise ValueError("voltage_V must be a 1D finite sequence with >= 3 points")
        if np.any(np.diff(arr) <= 0.0):
            raise ValueError("voltage_V must be strictly ascending")
        return arr
    if type(n_points) is not int or n_points < 3:
        raise ValueError("n_points must be an integer >= 3")
    if not math.isfinite(V_min) or not math.isfinite(V_max) or V_max <= V_min:
        raise ValueError("V_min and V_max must be finite with V_max > V_min")
    return np.linspace(float(V_min), float(V_max), n_points, dtype=float)


def _rebase_ionic_state(
    compiled: CompiledIonicDevice1D, state: IonicState1D, voltage_V: float
) -> IonicState1D:
    if state.voltage_V == voltage_V:
        return state
    coords = state.u.copy()
    n_elec = 3 * compiled.node_count
    coords[:n_elec] = compiled.electronic_component.rebase_coordinates(
        coords[:n_elec], state.voltage_V, voltage_V
    )
    return compiled.state(coords, voltage_V=voltage_V)


def run_dark_jv(
    compiled: CompiledDevice1D | CompiledIonicDevice1D,
    *,
    voltage_V: Sequence[float] | np.ndarray | None = None,
    V_min: float = 0.0,
    V_max: float = 1.2,
    n_points: int = 60,
    options: DarkJVOptions | None = None,
    progress: ProgressCallback | None = None,
) -> DarkJVResult:
    """Run a genuine dark DC J-V sweep and extract Shockley diode metrics."""
    opts = DarkJVOptions() if options is None else options
    voltages = _normalize_voltages(
        voltage_V, V_min=V_min, V_max=V_max, n_points=n_points
    )
    count = int(voltages.size)

    j_pv: np.ndarray = np.full(count, np.nan, dtype=float)
    j_conj: np.ndarray = np.full(count, np.nan, dtype=float)
    conv_mask: np.ndarray = np.zeros(count, dtype=bool)
    reasons: list[str | None] = [None] * count
    spread_max = 0.0

    if isinstance(compiled, CompiledDevice1D):
        polarity = float(compiled.voltage_polarity)
        temperature_K = float(
            compiled.thermal_voltage_V * ELEMENTARY_CHARGE_C / BOLTZMANN_J_K
        )
        controls_record: dict[str, Any] = {
            "mode": "electronic_dc",
            "steady": asdict(opts.steady),
        }
        prev_state = None
        for idx, v_val in enumerate(voltages):
            try:
                sol = solve_steady_state(
                    compiled,
                    voltage_V=float(v_val),
                    generation_scale=0.0,
                    initial_state=prev_state,
                    options=opts.steady,
                )
                if not sol.converged:
                    reasons[idx] = "dc_not_converged:unconverged_steady_result"
                    continue
            except SteadyStateError as exc:
                reasons[idx] = f"dc_not_converged:{exc}"
                continue

            prev_state = sol.state
            faces = np.asarray(sol.current_density_A_m2, dtype=float)
            contacts = np.asarray(sol.contact_current_density_A_m2, dtype=float)
            spread_max = max(spread_max, float(np.ptp(faces)))
            j_pv[idx] = float(-polarity * faces[0])
            j_conj[idx] = float(
                polarity * contacts[0 if polarity > 0.0 else -1]
            )
            conv_mask[idx] = True
            if progress is not None:
                progress(
                    "dark_jv",
                    idx + 1,
                    count,
                    f"V={v_val:.4f} V, J_pv={j_pv[idx]:.3e} A/m^2",
                )

    elif isinstance(compiled, CompiledIonicDevice1D):
        polarity = float(compiled.electronic_component.voltage_polarity)
        temperature_K = float(
            compiled.electronic_component.thermal_voltage_V
            * ELEMENTARY_CHARGE_C
            / BOLTZMANN_J_K
        )
        ionic_controls = (
            default_ionic_steady_options(
                compiled.ion_node_count, electronic=opts.steady
            )
            if opts.ionic_steady is None
            else opts.ionic_steady
        )
        controls_record = {
            "mode": "ionic_dc",
            "ionic_steady": asdict(ionic_controls),
        }
        prev_ionic: IonicState1D | None = None
        for idx, v_val in enumerate(voltages):
            sol_ion = solve_ionic_dc_operating_point(
                compiled,
                voltage_V=float(v_val),
                generation_scale=0.0,
                options=ionic_controls,
                base_state=prev_ionic,
                base_generation_scale=0.0,
            )
            if (
                not sol_ion.accepted
                or sol_ion.state is None
                or sol_ion.diagnostics is None
            ):
                reason = sol_ion.error or sol_ion.termination
                reasons[idx] = f"dc_not_converged:{reason}"
                continue

            prev_ionic = sol_ion.state
            all_faces = np.asarray(
                sol_ion.diagnostics["total_current_density_A_m2"], dtype=float
            )
            spread_max = max(spread_max, float(np.ptp(all_faces)))
            j_pv[idx] = float(-polarity * all_faces[0])
            j_conj[idx] = float(
                polarity * all_faces[0 if polarity > 0.0 else -1]
            )
            conv_mask[idx] = True
            if progress is not None:
                progress(
                    "dark_jv",
                    idx + 1,
                    count,
                    f"V={v_val:.4f} V, J_pv={j_pv[idx]:.3e} A/m^2",
                )
    else:
        raise TypeError(
            "compiled must be a CompiledDevice1D or CompiledIonicDevice1D"
        )

    if opts.require_all_converged and not bool(np.all(conv_mask)):
        bad = [
            f"V={voltages[i]:.4g}V ({reasons[i]})"
            for i in range(count)
            if not conv_mask[i]
        ]
        raise RuntimeError(
            "Dark J-V failed to converge at all points: " + ", ".join(bad)
        )

    (
        local_n,
        fit_mask,
        n_id,
        j0,
        v_lo,
        v_hi,
        r_sq,
        rs_est,
        fit_reason,
    ) = extract_dark_diode_parameters(
        voltages,
        j_pv,
        temperature_K=temperature_K,
        converged_mask=conv_mask,
        min_points=opts.min_diode_window_points,
        J_floor_A_m2=opts.current_floor_A_m2,
        min_forward_voltage_V=opts.min_forward_voltage_V,
    )

    return DarkJVResult(
        voltage_V=voltages,
        pv_output_current_A_m2=j_pv,
        voltage_conjugate_current_A_m2=j_conj,
        converged_mask=conv_mask,
        point_reasons=tuple(reasons),
        local_ideality_factor=local_n,
        fit_mask=fit_mask,
        n_ideality=n_id,
        saturation_current_J0_A_m2=j0,
        V_fit_lo_V=v_lo,
        V_fit_hi_V=v_hi,
        r_squared=r_sq,
        series_resistance_est_ohm_m2=rs_est,
        fit_failure_reason=fit_reason,
        max_current_spread_A_m2=float(spread_max),
        temperature_K=temperature_K,
        voltage_polarity=polarity,
        compiled_identity=compiled.identity,
        solver_controls=controls_record,
    )
