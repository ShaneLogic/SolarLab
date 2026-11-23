"""Temperature-dependent open-circuit voltage V_oc(T) and 0 K activation energy.

Each requested temperature ``T_k`` cold-rebinds ``PreparedDevice`` with
``settings.T = T_k`` and compiles a fresh 1D device with zero cross-temperature
carrier or ionic state hand-off. Every point is solved as a genuine DC
operating point via ``run_suns_voc(..., suns_levels=[generation_scale])``;
unbracketed or non-converged temperatures record ``np.nan`` and are excluded
from the linear 0 K extrapolation ``V_oc(T) = slope * T + E_A / q``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import json
import math
from typing import Any
import warnings

import numpy as np

from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.suns_voc import SunsVocOptions, run_suns_voc
from solarlab.reproducibility.job_provenance import current_provenance
from solarlab.reproducibility.sources import SourceDocument

__all__ = [
    "VocTOptions",
    "VocTResult",
    "fit_voc_t_extrapolation",
    "rebind_prepared_temperature",
    "run_voc_t",
]

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
class VocTOptions:
    """Explicit grid, mode, and DC root-finding controls for V_oc(T)."""

    intervals_per_layer: int = 20
    generation_scale: float = 1.0
    ion_domain_layer: str | None = None
    ion_domain_endpoints: tuple[str, str] = ("blocking", "blocking")
    suns_voc: SunsVocOptions = field(default_factory=SunsVocOptions)
    allow_partial: bool = False

    def __post_init__(self) -> None:
        if type(self.intervals_per_layer) is not int or self.intervals_per_layer < 2:
            raise ValueError("intervals_per_layer must be an integer >= 2")
        if not math.isfinite(self.generation_scale) or self.generation_scale <= 0.0:
            raise ValueError("generation_scale must be finite and positive")
        if not isinstance(self.suns_voc, SunsVocOptions):
            raise ValueError("suns_voc must be an explicit SunsVocOptions instance")


@dataclass(frozen=True)
class VocTResult:
    """Genuine DC V_oc(T) sweep with 0 K linear activation-energy fit."""

    temperatures_K: np.ndarray
    V_oc_V: np.ndarray
    J_sc_A_m2: np.ndarray
    converged_mask: np.ndarray
    voc_bracketed_mask: np.ndarray
    point_reasons: tuple[str | None, ...]
    slope_V_K: float
    intercept_0K_V: float
    activation_energy_eV: float
    r_squared: float
    fit_failure_reason: str | None
    generation_scale: float
    compiled_identities: tuple[str, ...]
    job_provenance: dict[str, Any] | None = field(default_factory=current_provenance)

    def __post_init__(self) -> None:
        for name in (
            "temperatures_K",
            "V_oc_V",
            "J_sc_A_m2",
        ):
            object.__setattr__(self, name, _ro_float(getattr(self, name)))
        for mask_name in ("converged_mask", "voc_bracketed_mask"):
            object.__setattr__(self, mask_name, _ro_bool(getattr(self, mask_name)))

    @property
    def T_arr(self) -> np.ndarray:
        return self.temperatures_K

    @property
    def V_oc_arr(self) -> np.ndarray:
        return self.V_oc_V

    @property
    def J_sc_arr(self) -> np.ndarray:
        return self.J_sc_A_m2

    @property
    def E_A_eV(self) -> float:
        return self.activation_energy_eV

    @property
    def V_oc_0K(self) -> float:
        return self.intercept_0K_V

    @property
    def completed(self) -> bool:
        return bool(np.all(self.converged_mask & self.voc_bracketed_mask))


def rebind_prepared_temperature(
    prepared: PreparedDevice, temperature_K: float
) -> PreparedDevice:
    """Return a fresh ``PreparedDevice`` bound at ``settings.T = temperature_K``."""
    if not isinstance(prepared, PreparedDevice):
        raise TypeError("prepared must be a PreparedDevice instance")
    t_val = float(temperature_K)
    if not math.isfinite(t_val) or t_val <= 0.0:
        raise ValueError(f"temperature_K must be finite and positive, got {temperature_K}")

    editing = prepared.to_input().editing_data()
    settings = dict(editing.get("settings") or {})
    settings["T"] = t_val
    editing["settings"] = settings

    payload = json.dumps(
        editing, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    source = SourceDocument(f"voc_t.T_{t_val:.6g}K", payload)
    return resolve_device(
        DeviceInput.model_validate(editing),
        prepared.defaults,
        prepared.resources,
        sources=(*prepared.sources, source),
    )


def fit_voc_t_extrapolation(
    temperatures_K: Sequence[float] | np.ndarray,
    V_oc_V: Sequence[float] | np.ndarray,
    *,
    converged_mask: Sequence[bool] | np.ndarray,
    voc_bracketed_mask: Sequence[bool] | np.ndarray,
    allow_partial: bool = False,
) -> tuple[float, float, float, float, str | None]:
    """Fit ``V_oc(T) = slope * T + intercept_0K`` over valid bracketed points.

    Unbracketed or non-converged points are never substituted with ``0.0``.
    When fewer than 2 valid distinct temperatures exist, raises ``ValueError``
    unless ``allow_partial=True`` (which returns ``np.nan`` fit parameters and
    an explicit ``fit_failure_reason``).
    """
    temps: np.ndarray = np.asarray(temperatures_K, dtype=float)
    voc: np.ndarray = np.asarray(V_oc_V, dtype=float)
    conv: np.ndarray = np.asarray(converged_mask, dtype=bool)
    brack: np.ndarray = np.asarray(voc_bracketed_mask, dtype=bool)
    if (
        temps.ndim != 1
        or voc.shape != temps.shape
        or conv.shape != temps.shape
        or brack.shape != temps.shape
    ):
        raise ValueError("V_oc(T) arrays and masks must have matching 1D shapes")

    valid = conv & brack & np.isfinite(temps) & (temps > 0.0) & np.isfinite(voc)
    n_valid = int(np.count_nonzero(valid))
    n_total = int(temps.size)

    if n_valid < 2 or np.unique(temps[valid]).size < 2:
        reason = f"insufficient_valid_bracketed_points:{n_valid}_of_{n_total}"
        if not allow_partial:
            raise ValueError(
                f"run_voc_t requires >= 2 valid bracketed temperatures to "
                f"extrapolate E_A, got {n_valid}/{n_total}"
            )
        return float("nan"), float("nan"), float("nan"), float("nan"), reason

    if n_valid < n_total:
        excluded = [f"{temps[i]:.1f}K" for i in range(n_total) if not valid[i]]
        warnings.warn(
            f"V_oc(T) fit excluded {n_total - n_valid}/{n_total} non-converged or "
            f"unbracketed temperatures ({', '.join(excluded)}).",
            RuntimeWarning,
            stacklevel=2,
        )

    t_fit = temps[valid]
    v_fit = voc[valid]
    slope, intercept = np.polyfit(t_fit, v_fit, 1)
    v_pred = slope * t_fit + intercept
    ss_res = float(np.sum((v_fit - v_pred) ** 2))
    ss_tot = float(np.sum((v_fit - np.mean(v_fit)) ** 2))
    r_sq = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 1.0
    # E_A [eV] numerically equals q * V_oc(0 K) / q = intercept [V].
    return (
        float(slope),
        float(intercept),
        float(intercept),
        r_sq,
        None,
    )


def _normalize_temperatures(
    temperatures_K: Sequence[float] | np.ndarray | None,
    *,
    T_min: float,
    T_max: float,
    n_points: int,
) -> np.ndarray:
    if temperatures_K is not None:
        arr: np.ndarray = np.asarray([float(t) for t in temperatures_K], dtype=float)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError("temperatures_K must contain at least 2 temperatures")
        if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError("temperatures_K must be finite and positive")
        return np.array(sorted(set(arr.tolist())), dtype=float)
    if type(n_points) is not int or n_points < 2:
        raise ValueError(f"n_points must be an integer >= 2, got {n_points}")
    if not math.isfinite(T_min) or not math.isfinite(T_max) or T_min <= 0.0:
        raise ValueError(f"T_min and T_max must be finite and > 0, got {T_min}, {T_max}")
    if T_max <= T_min:
        raise ValueError(f"T_max ({T_max}) must exceed T_min ({T_min})")
    return np.linspace(float(T_min), float(T_max), n_points, dtype=float)


def run_voc_t(
    prepared: PreparedDevice,
    *,
    temperatures_K: Sequence[float] | np.ndarray | None = None,
    T_min: float = 250.0,
    T_max: float = 350.0,
    n_points: int = 5,
    options: VocTOptions | None = None,
    progress: ProgressCallback | None = None,
) -> VocTResult:
    """Sweep temperature via cold device rebinding and extrapolate E_A at 0 K."""
    if not isinstance(prepared, PreparedDevice):
        raise TypeError("prepared must be a PreparedDevice instance")
    opts = VocTOptions() if options is None else options
    temps = _normalize_temperatures(
        temperatures_K, T_min=T_min, T_max=T_max, n_points=n_points
    )
    if temps.size < 2:
        raise ValueError("At least 2 distinct temperatures are required for V_oc(T)")

    count = int(temps.size)
    voc_arr: np.ndarray = np.full(count, np.nan, dtype=float)
    jsc_arr: np.ndarray = np.full(count, np.nan, dtype=float)
    conv_mask: np.ndarray = np.zeros(count, dtype=bool)
    brack_mask: np.ndarray = np.zeros(count, dtype=bool)
    reasons: list[str | None] = [None] * count
    identities: list[str] = []

    for idx, t_k in enumerate(temps):
        rebound = rebind_prepared_temperature(prepared, float(t_k))
        if opts.ion_domain_layer is not None:
            compiled: Any = compile_ionic_device_1d(
                rebound,
                intervals_per_layer=opts.intervals_per_layer,
                ion_domain_layer=opts.ion_domain_layer,
                ion_domain_endpoints=opts.ion_domain_endpoints,
            )
        else:
            compiled = compile_device_1d(
                rebound, intervals_per_layer=opts.intervals_per_layer
            )
        identities.append(str(compiled.identity))

        point_res = run_suns_voc(
            compiled,
            [opts.generation_scale],
            options=opts.suns_voc,
        )
        voc_arr[idx] = float(point_res.V_oc_V[0])
        jsc_arr[idx] = float(point_res.J_sc_A_m2[0])
        conv_mask[idx] = bool(point_res.converged_mask[0])
        brack_mask[idx] = bool(point_res.voc_bracketed_mask[0])
        reasons[idx] = point_res.point_reasons[0]

        if progress is not None:
            msg = (
                f"T={t_k:.1f} K, V_oc={voc_arr[idx]:.4f} V"
                if conv_mask[idx] and brack_mask[idx]
                else f"T={t_k:.1f} K ({reasons[idx]})"
            )
            progress("voc_t", idx + 1, count, msg)

    slope, intercept_0k, e_a_ev, r_sq, fit_reason = fit_voc_t_extrapolation(
        temps,
        voc_arr,
        converged_mask=conv_mask,
        voc_bracketed_mask=brack_mask,
        allow_partial=opts.allow_partial,
    )

    return VocTResult(
        temperatures_K=temps,
        V_oc_V=voc_arr,
        J_sc_A_m2=jsc_arr,
        converged_mask=conv_mask,
        voc_bracketed_mask=brack_mask,
        point_reasons=tuple(reasons),
        slope_V_K=slope,
        intercept_0K_V=intercept_0k,
        activation_energy_eV=e_a_ev,
        r_squared=r_sq,
        fit_failure_reason=fit_reason,
        generation_scale=opts.generation_scale,
        compiled_identities=tuple(identities),
    )
