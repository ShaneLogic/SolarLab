"""Lumped areal thermal-balance ledger, steady-state root, and transient ODE.

Thermodynamic boundary (``system_boundary = 'device_plus_lumped_parasitics'``):
the semiconductor stack together with any lumped external series/shunt
resistive parasitics forms a single isothermal control volume at temperature
``T`` [K], exchanging power with ambient ``T_amb`` [K] and exported electrical
power ``P_elec_out = V_terminal * J_terminal`` [W/m^2]:

    C_th * dT/dt = P_absorbed + Q_internal - P_elec_out
                   - h * (T - T_amb) - epsilon * sigma * (T^4 - T_amb^4)

Series ``J_terminal^2 * R_s`` and shunt ``V_junction^2 / R_sh`` Joule heat stay
inside the control volume and are accounted for ONCE through the reduction of
``P_elec_out`` relative to ``P_absorbed``: they are never added a second time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

__all__ = [
    "STEFAN_BOLTZMANN_W_M2_K4",
    "THERMAL_UNIT_SCHEMA",
    "LumpedThermalProtocol",
    "ThermalBalanceError",
    "ThermalEnergySourceError",
    "ThermalIntegrationProtocol",
    "ThermalPowerLedger",
    "ThermalSteadyStateError",
    "ThermalSteadyStateResult",
    "ThermalTransientError",
    "ThermalTransientResult",
    "run_lumped_thermal_transient",
    "solve_lumped_thermal_steady_state",
    "thermal_power_ledger",
]

STEFAN_BOLTZMANN_W_M2_K4: float = 5.670374419e-8

THERMAL_UNIT_SCHEMA: dict[str, str] = {
    "absorbed_optical_power_W_m2": "W/m^2",
    "ambient_temperature_K": "K",
    "areal_heat_capacity_J_m2_K": "J/(m^2*K)",
    "heat_transfer_coefficient_W_m2_K": "W/(m^2*K)",
    "emissivity": "1",
    "maximum_temperature_K": "K",
    "constant_internal_heat_W_m2": "W/m^2",
}


class ThermalBalanceError(ValueError):
    """Invalid lumped thermal protocol or thermodynamic domain."""


class ThermalEnergySourceError(ThermalBalanceError):
    """Exported electrical power exceeds available input power without a declared source."""


class ThermalSteadyStateError(RuntimeError):
    """Lumped thermal steady-state root is not bracketed or failed to converge."""


class ThermalTransientError(RuntimeError):
    """Lumped thermal transient integration failed or violated energy conservation."""


def _is_real_number(value: object) -> bool:
    return isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(
        value, bool
    )


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_hex_list(values: np.ndarray) -> list[str]:
    flat = np.asarray(values, dtype=float).ravel()
    return [float(x).hex() for x in flat]


@dataclass(frozen=True)
class LumpedThermalProtocol:
    """Explicit areal lumped thermal boundary and heat-capacity parameters."""

    absorbed_optical_power_W_m2: float
    ambient_temperature_K: float = 300.0
    areal_heat_capacity_J_m2_K: float = 1000.0
    heat_transfer_coefficient_W_m2_K: float = 10.0
    emissivity: float = 0.0
    maximum_temperature_K: float = 450.0
    constant_internal_heat_W_m2: float = 0.0
    system_boundary: str = "device_plus_lumped_parasitics"

    def __post_init__(self) -> None:
        fields = (
            "absorbed_optical_power_W_m2",
            "ambient_temperature_K",
            "areal_heat_capacity_J_m2_K",
            "heat_transfer_coefficient_W_m2_K",
            "emissivity",
            "maximum_temperature_K",
            "constant_internal_heat_W_m2",
        )
        for name in fields:
            raw = getattr(self, name)
            if not _is_real_number(raw) or not math.isfinite(float(raw)):
                raise ThermalBalanceError(f"{name} must be a finite real number")
            object.__setattr__(self, name, float(raw))

        if self.absorbed_optical_power_W_m2 < 0.0:
            raise ThermalBalanceError("absorbed_optical_power_W_m2 must be >= 0")
        if self.ambient_temperature_K <= 0.0:
            raise ThermalBalanceError("ambient_temperature_K must be > 0 K")
        if self.areal_heat_capacity_J_m2_K <= 0.0:
            raise ThermalBalanceError("areal_heat_capacity_J_m2_K must be > 0 J/(m^2*K)")
        if self.heat_transfer_coefficient_W_m2_K < 0.0:
            raise ThermalBalanceError("heat_transfer_coefficient_W_m2_K must be >= 0")
        if not 0.0 <= self.emissivity <= 1.0:
            raise ThermalBalanceError("emissivity must lie in [0, 1]")
        if (
            self.heat_transfer_coefficient_W_m2_K == 0.0
            and self.emissivity == 0.0
        ):
            raise ThermalBalanceError(
                "at least one cooling channel (heat_transfer_coefficient_W_m2_K > 0 or emissivity > 0) is required"
            )
        if self.maximum_temperature_K <= self.ambient_temperature_K:
            raise ThermalBalanceError(
                "maximum_temperature_K must exceed ambient_temperature_K"
            )
        if self.constant_internal_heat_W_m2 < 0.0:
            raise ThermalBalanceError("constant_internal_heat_W_m2 must be >= 0")
        if self.system_boundary != "device_plus_lumped_parasitics":
            raise ThermalBalanceError(
                f"Unsupported system_boundary {self.system_boundary!r}; "
                "only 'device_plus_lumped_parasitics' is supported"
            )

    @property
    def parameter_units(self) -> dict[str, str]:
        return dict(THERMAL_UNIT_SCHEMA)


@dataclass(frozen=True)
class ThermalIntegrationProtocol:
    """Explicit ODE integration controls for lumped thermal transients."""

    method: str = "DOP853"
    rtol: float = 1e-10
    atol_temperature_K: float = 1e-12
    atol_energy_J_m2: float = 1e-10
    max_step_s: float = math.inf
    energy_closure_atol_J_m2: float = 1e-6
    energy_closure_rtol: float = 1e-8

    def __post_init__(self) -> None:
        if self.method not in {"RK45", "DOP853", "Radau", "BDF"}:
            raise ThermalBalanceError(
                f"Unsupported ODE method {self.method!r}; expected RK45, DOP853, Radau, or BDF"
            )
        for name in (
            "rtol",
            "atol_temperature_K",
            "atol_energy_J_m2",
            "energy_closure_atol_J_m2",
            "energy_closure_rtol",
        ):
            raw = getattr(self, name)
            if not _is_real_number(raw) or not math.isfinite(float(raw)) or float(raw) <= 0.0:
                raise ThermalBalanceError(f"{name} must be finite and > 0")
            object.__setattr__(self, name, float(raw))
        if not _is_real_number(self.max_step_s) or float(self.max_step_s) <= 0.0:
            raise ThermalBalanceError("max_step_s must be > 0")
        object.__setattr__(self, "max_step_s", float(self.max_step_s))


@dataclass(frozen=True)
class ThermalPowerLedger:
    """Instantaneous first-law areal power balance at temperature ``T``."""

    temperature_K: float
    absorbed_optical_power_W_m2: float
    constant_internal_heat_W_m2: float
    exported_electrical_power_W_m2: float
    net_heating_power_W_m2: float
    convective_loss_W_m2: float
    radiative_loss_W_m2: float
    total_cooling_loss_W_m2: float
    net_storage_rate_W_m2: float


@dataclass(frozen=True)
class ThermalSteadyStateResult:
    """Certified steady-state lumped thermal operating point."""

    protocol: LumpedThermalProtocol
    temperature_K: float
    ledger: ThermalPowerLedger
    root_calls: int
    root_iterations: int
    converged: bool
    power_residual_W_m2: float
    tolerance_W_m2: float
    result_sha256: str


@dataclass(frozen=True)
class ThermalTransientResult:
    """Certified lumped thermal transient trajectory and cumulative energy closure."""

    protocol: LumpedThermalProtocol
    integration: ThermalIntegrationProtocol
    t_s: np.ndarray
    temperature_K: np.ndarray
    exported_electrical_power_W_m2: np.ndarray
    net_heating_power_W_m2: np.ndarray
    convective_loss_W_m2: np.ndarray
    radiative_loss_W_m2: np.ndarray
    cumulative_absorbed_J_m2: np.ndarray
    cumulative_internal_heat_J_m2: np.ndarray
    cumulative_exported_electrical_J_m2: np.ndarray
    cumulative_convective_loss_J_m2: np.ndarray
    cumulative_radiative_loss_J_m2: np.ndarray
    stored_thermal_energy_change_J_m2: np.ndarray
    energy_closure_residual_J_m2: np.ndarray
    max_abs_energy_closure_residual_J_m2: float
    nfev: int
    certified: bool
    result_sha256: str


def thermal_power_ledger(
    temperature_K: float,
    protocol: LumpedThermalProtocol,
    *,
    exported_electrical_power_W_m2: float = 0.0,
    allow_external_electrical_drive: bool = False,
) -> ThermalPowerLedger:
    """Evaluate instantaneous areal heating, cooling, and storage rate at ``T``."""
    if not isinstance(protocol, LumpedThermalProtocol):
        raise ThermalBalanceError("protocol must be a LumpedThermalProtocol")
    if not _is_real_number(temperature_K) or not math.isfinite(float(temperature_K)):
        raise ThermalBalanceError("temperature_K must be a finite real scalar")
    t_val = float(temperature_K)
    if t_val <= 0.0:
        raise ThermalBalanceError(f"temperature_K must be > 0 K, got {t_val}")

    if not _is_real_number(exported_electrical_power_W_m2) or not math.isfinite(
        float(exported_electrical_power_W_m2)
    ):
        raise ThermalBalanceError(
            "exported_electrical_power_W_m2 must be a finite real scalar"
        )
    p_elec = float(exported_electrical_power_W_m2)

    p_abs = protocol.absorbed_optical_power_W_m2
    q_int = protocol.constant_internal_heat_W_m2
    available_input = p_abs + q_int
    tol = 1e-12 * max(1.0, abs(available_input))
    if not allow_external_electrical_drive and p_elec > available_input + tol:
        raise ThermalEnergySourceError(
            f"exported_electrical_power_W_m2 ({p_elec:.6g} W/m^2) exceeds "
            f"absorbed + internal input ({available_input:.6g} W/m^2)"
        )

    q_heat = available_input - p_elec
    t_amb = protocol.ambient_temperature_K
    q_conv = protocol.heat_transfer_coefficient_W_m2_K * (t_val - t_amb)
    q_rad = (
        protocol.emissivity
        * STEFAN_BOLTZMANN_W_M2_K4
        * (t_val**4 - t_amb**4)
    )
    q_cool = q_conv + q_rad
    q_net = q_heat - q_cool

    return ThermalPowerLedger(
        temperature_K=t_val,
        absorbed_optical_power_W_m2=p_abs,
        constant_internal_heat_W_m2=q_int,
        exported_electrical_power_W_m2=p_elec,
        net_heating_power_W_m2=q_heat,
        convective_loss_W_m2=q_conv,
        radiative_loss_W_m2=q_rad,
        total_cooling_loss_W_m2=q_cool,
        net_storage_rate_W_m2=q_net,
    )


def solve_lumped_thermal_steady_state(
    protocol: LumpedThermalProtocol,
    *,
    exported_electrical_power_fn: Callable[[float], float] | None = None,
    xtol_K: float = 1e-11,
    rtol: float = 1e-12,
    maxiter: int = 100,
    residual_atol_W_m2: float = 1e-7,
    allow_external_electrical_drive: bool = False,
    lower_temperature_K: float | None = None,
    upper_temperature_K: float | None = None,
) -> ThermalSteadyStateResult:
    """Solve ``net_storage_rate_W_m2(T*) = 0`` via bracketed Brent root finding."""
    if not isinstance(protocol, LumpedThermalProtocol):
        raise ThermalBalanceError("protocol must be a LumpedThermalProtocol")
    if type(maxiter) is not int or maxiter < 4:
        raise ThermalBalanceError("maxiter must be an integer >= 4")
    for name, val in (
        ("xtol_K", xtol_K),
        ("rtol", rtol),
        ("residual_atol_W_m2", residual_atol_W_m2),
    ):
        if not _is_real_number(val) or not math.isfinite(float(val)) or float(val) <= 0.0:
            raise ThermalBalanceError(f"{name} must be finite and > 0")

    t_lo = (
        protocol.ambient_temperature_K
        if lower_temperature_K is None
        else float(lower_temperature_K)
    )
    t_hi = (
        protocol.maximum_temperature_K
        if upper_temperature_K is None
        else float(upper_temperature_K)
    )
    if not (0.0 < t_lo < t_hi <= protocol.maximum_temperature_K):
        raise ThermalBalanceError(
            f"Invalid temperature bracket [{t_lo}, {t_hi}] within (0, {protocol.maximum_temperature_K}]"
        )

    p_fn: Callable[[float], float] = (
        (lambda _t: 0.0)
        if exported_electrical_power_fn is None
        else exported_electrical_power_fn
    )
    cache: dict[float, ThermalPowerLedger] = {}
    calls = 0

    def eval_ledger(temp_K: float) -> ThermalPowerLedger:
        nonlocal calls
        t_key = float(temp_K)
        cached = cache.get(t_key)
        if cached is not None:
            return cached
        calls += 1
        p_out = float(p_fn(t_key))
        led = thermal_power_ledger(
            t_key,
            protocol,
            exported_electrical_power_W_m2=p_out,
            allow_external_electrical_drive=allow_external_electrical_drive,
        )
        cache[t_key] = led
        return led

    f_lo = eval_ledger(t_lo).net_storage_rate_W_m2
    if abs(f_lo) <= residual_atol_W_m2:
        led_star = eval_ledger(t_lo)
        sha = _canonical_sha256(
            {
                "protocol": asdict(protocol),
                "temperature_K": float(t_lo).hex(),
                "ledger": asdict(led_star),
            }
        )
        return ThermalSteadyStateResult(
            protocol=protocol,
            temperature_K=t_lo,
            ledger=led_star,
            root_calls=calls,
            root_iterations=0,
            converged=True,
            power_residual_W_m2=led_star.net_storage_rate_W_m2,
            tolerance_W_m2=float(residual_atol_W_m2),
            result_sha256=sha,
        )

    f_hi = eval_ledger(t_hi).net_storage_rate_W_m2
    if f_lo * f_hi > 0.0:
        raise ThermalSteadyStateError(
            f"Thermal steady-state root is not bracketed on [{t_lo:.4f}, {t_hi:.4f}] K: "
            f"f(T_lo)={f_lo:.6g} W/m^2, f(T_hi)={f_hi:.6g} W/m^2"
        )

    t_root, root_info = brentq(
        lambda t_val: eval_ledger(float(t_val)).net_storage_rate_W_m2,
        t_lo,
        t_hi,
        xtol=float(xtol_K),
        rtol=float(rtol),
        maxiter=maxiter,
        full_output=True,
        disp=False,
    )
    led_star = eval_ledger(float(t_root))
    if not root_info.converged or abs(led_star.net_storage_rate_W_m2) > float(
        residual_atol_W_m2
    ):
        raise ThermalSteadyStateError(
            f"Thermal steady-state brentq failed residual gate: "
            f"|f(T*)|={abs(led_star.net_storage_rate_W_m2):.3e} W/m^2 > {residual_atol_W_m2:.3e} W/m^2"
        )

    sha = _canonical_sha256(
        {
            "protocol": asdict(protocol),
            "temperature_K": float(t_root).hex(),
            "ledger": asdict(led_star),
        }
    )
    return ThermalSteadyStateResult(
        protocol=protocol,
        temperature_K=float(t_root),
        ledger=led_star,
        root_calls=calls,
        root_iterations=int(root_info.iterations),
        converged=True,
        power_residual_W_m2=led_star.net_storage_rate_W_m2,
        tolerance_W_m2=float(residual_atol_W_m2),
        result_sha256=sha,
    )


def run_lumped_thermal_transient(
    protocol: LumpedThermalProtocol,
    t_eval_s: object,
    *,
    initial_temperature_K: float | None = None,
    exported_electrical_power_fn: Callable[[float, float], float] | None = None,
    integration: ThermalIntegrationProtocol | None = None,
    allow_external_electrical_drive: bool = False,
) -> ThermalTransientResult:
    """Integrate ``C_th * dT/dt`` alongside cumulative energy ledgers and verify closure."""
    if not isinstance(protocol, LumpedThermalProtocol):
        raise ThermalBalanceError("protocol must be a LumpedThermalProtocol")
    integ = ThermalIntegrationProtocol() if integration is None else integration
    if not isinstance(integ, ThermalIntegrationProtocol):
        raise ThermalBalanceError("integration must be a ThermalIntegrationProtocol")

    if np.iscomplexobj(t_eval_s):
        raise ThermalBalanceError("t_eval_s must be real-valued")
    times = np.array(t_eval_s, dtype=float, copy=True)
    if (
        times.ndim != 1
        or times.size < 2
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ThermalBalanceError(
            "t_eval_s must be a 1-D strictly increasing array with at least two finite times"
        )
    times.setflags(write=False)

    t0_K = (
        protocol.ambient_temperature_K
        if initial_temperature_K is None
        else float(initial_temperature_K)
    )
    if not (0.0 < t0_K <= protocol.maximum_temperature_K):
        raise ThermalBalanceError(
            f"initial_temperature_K ({t0_K}) must lie in (0, {protocol.maximum_temperature_K}]"
        )

    p_fn: Callable[[float, float], float] = (
        (lambda _time_s, _temp_K: 0.0)
        if exported_electrical_power_fn is None
        else exported_electrical_power_fn
    )
    c_th = protocol.areal_heat_capacity_J_m2_K

    def rhs(time_s: float, state: np.ndarray) -> np.ndarray:
        temp_K = float(state[0])
        if temp_K <= 0.0:
            raise ThermalTransientError(
                f"non-physical nonpositive trial temperature {temp_K} K at t={time_s} s"
            )
        p_elec = float(p_fn(float(time_s), temp_K))
        led = thermal_power_ledger(
            temp_K,
            protocol,
            exported_electrical_power_W_m2=p_elec,
            allow_external_electrical_drive=allow_external_electrical_drive,
        )
        return np.array(
            [
                led.net_storage_rate_W_m2 / c_th,
                led.absorbed_optical_power_W_m2,
                led.constant_internal_heat_W_m2,
                led.exported_electrical_power_W_m2,
                led.convective_loss_W_m2,
                led.radiative_loss_W_m2,
            ],
            dtype=float,
        )

    y0 = np.array([t0_K, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    atol_vec = np.array(
        [
            integ.atol_temperature_K,
            integ.atol_energy_J_m2,
            integ.atol_energy_J_m2,
            integ.atol_energy_J_m2,
            integ.atol_energy_J_m2,
            integ.atol_energy_J_m2,
        ],
        dtype=float,
    )
    sol = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        y0,
        method=integ.method,
        t_eval=times,
        rtol=integ.rtol,
        atol=atol_vec,
        max_step=integ.max_step_s,
    )
    if not sol.success or sol.y.shape != (6, times.size):
        raise ThermalTransientError(
            f"Thermal transient integration failed: {sol.message}"
        )

    temp_hist = np.array(sol.y[0], dtype=float, copy=True)
    if np.any(temp_hist <= 0.0) or np.any(temp_hist > protocol.maximum_temperature_K):
        raise ThermalTransientError(
            f"Thermal transient temperature left (0, {protocol.maximum_temperature_K}] K"
        )
    e_abs = np.array(sol.y[1], dtype=float, copy=True)
    e_int = np.array(sol.y[2], dtype=float, copy=True)
    e_elec = np.array(sol.y[3], dtype=float, copy=True)
    e_conv = np.array(sol.y[4], dtype=float, copy=True)
    e_rad = np.array(sol.y[5], dtype=float, copy=True)

    p_elec_arr = np.empty(times.size, dtype=float)
    q_heat_arr = np.empty(times.size, dtype=float)
    q_conv_arr = np.empty(times.size, dtype=float)
    q_rad_arr = np.empty(times.size, dtype=float)
    for k in range(times.size):
        p_val = float(p_fn(float(times[k]), float(temp_hist[k])))
        led_k = thermal_power_ledger(
            float(temp_hist[k]),
            protocol,
            exported_electrical_power_W_m2=p_val,
            allow_external_electrical_drive=allow_external_electrical_drive,
        )
        p_elec_arr[k] = led_k.exported_electrical_power_W_m2
        q_heat_arr[k] = led_k.net_heating_power_W_m2
        q_conv_arr[k] = led_k.convective_loss_W_m2
        q_rad_arr[k] = led_k.radiative_loss_W_m2

    delta_u = c_th * (temp_hist - t0_K)
    net_flow = e_abs + e_int - e_elec - e_conv - e_rad
    closure_res = delta_u - net_flow
    max_res = float(np.max(np.abs(closure_res)))
    scale = float(
        np.max(
            np.abs(e_abs)
            + np.abs(e_int)
            + np.abs(e_elec)
            + np.abs(e_conv)
            + np.abs(e_rad)
            + np.abs(delta_u)
        )
    )
    allowed = integ.energy_closure_atol_J_m2 + integ.energy_closure_rtol * max(
        1.0, scale
    )
    if max_res > allowed:
        raise ThermalTransientError(
            f"Cumulative thermal energy closure residual {max_res:.3e} J/m^2 exceeds {allowed:.3e} J/m^2"
        )

    for arr in (
        temp_hist,
        p_elec_arr,
        q_heat_arr,
        q_conv_arr,
        q_rad_arr,
        e_abs,
        e_int,
        e_elec,
        e_conv,
        e_rad,
        delta_u,
        closure_res,
    ):
        arr.setflags(write=False)

    sha = _canonical_sha256(
        {
            "protocol": asdict(protocol),
            "integration": {
                "method": integ.method,
                "rtol": integ.rtol,
                "atol_temperature_K": integ.atol_temperature_K,
                "atol_energy_J_m2": integ.atol_energy_J_m2,
                "max_step_s": (
                    "inf" if math.isinf(integ.max_step_s) else integ.max_step_s
                ),
            },
            "t_s": _array_hex_list(times),
            "temperature_K": _array_hex_list(temp_hist),
            "max_abs_energy_closure_residual_J_m2": max_res.hex(),
        }
    )
    return ThermalTransientResult(
        protocol=protocol,
        integration=integ,
        t_s=times,
        temperature_K=temp_hist,
        exported_electrical_power_W_m2=p_elec_arr,
        net_heating_power_W_m2=q_heat_arr,
        convective_loss_W_m2=q_conv_arr,
        radiative_loss_W_m2=q_rad_arr,
        cumulative_absorbed_J_m2=e_abs,
        cumulative_internal_heat_J_m2=e_int,
        cumulative_exported_electrical_J_m2=e_elec,
        cumulative_convective_loss_J_m2=e_conv,
        cumulative_radiative_loss_J_m2=e_rad,
        stored_thermal_energy_change_J_m2=delta_u,
        energy_closure_residual_J_m2=closure_res,
        max_abs_energy_closure_residual_J_m2=max_res,
        nfev=int(sol.nfev),
        certified=True,
        result_sha256=sha,
    )
