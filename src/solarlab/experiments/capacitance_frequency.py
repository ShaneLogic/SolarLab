"""Defect- and ion-aware capacitance-frequency (C-f) spectroscopy.

Extracts parallel/series capacitance, conductance, loss tangent, scaled
conductance ``G_p / omega``, and the Walter trap-DOS derivative
``-omega * dC_p / domega`` from unified ``DynamicDefectSmallSignal1D`` sweeps or
admittance arrays while preserving sign-faithful susceptance masking and dark
equilibrium passivity checks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts import frozen_array
from solarlab.device.compile_dynamic_defect_1d import compile_dynamic_defect_device_1d
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.assembly.dynamic_defects_1d import (
    DynamicDefectACSweepResult,
    DynamicDefectSmallSignal1D,
    DynamicDefectSteadyOptions,
    solve_dynamic_defect_steady_1d,
)
from solarlab.equations.dynamic_defect_1d import CompiledDynamicDefectDevice1D
from solarlab.physics.defects import DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER
from solarlab.solvers.linear.frequency_domain import FrequencyOptions

Vector: TypeAlias = NDArray[np.float64]
ComplexVector: TypeAlias = NDArray[np.complex128]
BoolVector: TypeAlias = NDArray[np.bool_]

__all__ = [
    "DefectCFError",
    "DefectCFOptions",
    "DefectCFResult",
    "capacitance_frequency_from_admittance",
    "capacitance_frequency_from_sweep",
    "run_defect_capacitance_frequency",
]


class DefectCFError(RuntimeError):
    """Raised when strict capacitance-frequency extraction or passivity fails."""


@dataclass(frozen=True, slots=True)
class DefectCFOptions:
    """Extraction, masking, and passivity controls for C-f spectroscopy."""

    strict_capacitive: bool = False
    minimum_phase_angle_rad: float = 0.0
    passivity_tolerance_S_m2: float = 1.0e-12
    require_dark_passivity: bool = True
    voltage_phasor_V: complex = 1.0
    steady_options: DynamicDefectSteadyOptions | None = None
    linear_options: FrequencyOptions | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum_phase_angle_rad) or not (
            0.0 <= self.minimum_phase_angle_rad < 0.5 * math.pi
        ):
            raise ValueError("minimum_phase_angle_rad must lie in [0, pi/2)")
        if not math.isfinite(self.passivity_tolerance_S_m2) or self.passivity_tolerance_S_m2 < 0.0:
            raise ValueError("passivity_tolerance_S_m2 must be finite and non-negative")
        if not np.isfinite(self.voltage_phasor_V) or self.voltage_phasor_V == 0.0:
            raise ValueError("voltage_phasor_V must be finite and nonzero")


@dataclass(frozen=True)
class DefectCFResult:
    """Capacitance-frequency observables, Walter derivative, and masking ledger."""

    frequencies_Hz: Vector
    angular_frequencies_rad_s: Vector
    admittance_density_S_m2: ComplexVector
    impedance_area_ohm_m2: ComplexVector
    parallel_capacitance_F_m2: Vector
    parallel_conductance_S_m2: Vector
    series_capacitance_F_m2: Vector
    loss_tangent: Vector
    scaled_conductance_F_m2: Vector
    walter_derivative_F_m2: Vector
    phase_angle_rad: Vector
    capacitance_accepted: BoolVector
    mask_reasons: tuple[str | None, ...]
    operating_voltage_V: float
    generation_scale: float
    active_species: tuple[str, ...]
    frozen_species: tuple[str, ...]
    working_state_digest: str
    dark_equilibrium_passive: bool | None
    peak_walter_frequency_Hz: float | None
    peak_walter_value_F_m2: float | None
    scientific_qualification: bool = False
    physical_certificate_status: str = "not_evaluated"

    def __post_init__(self) -> None:
        for attr in ("frequencies_Hz", "angular_frequencies_rad_s"):
            object.__setattr__(self, attr, frozen_array(getattr(self, attr)))
        for attr in ("admittance_density_S_m2", "impedance_area_ohm_m2"):
            c_arr = np.asarray(getattr(self, attr), dtype=np.complex128)
            object.__setattr__(
                self,
                attr,
                np.frombuffer(c_arr.tobytes(), dtype=np.complex128).reshape(c_arr.shape),
            )
        for attr in (
            "parallel_capacitance_F_m2",
            "parallel_conductance_S_m2",
            "series_capacitance_F_m2",
            "loss_tangent",
            "scaled_conductance_F_m2",
            "walter_derivative_F_m2",
            "phase_angle_rad",
        ):
            f_arr = np.asarray(getattr(self, attr), dtype=np.float64)
            object.__setattr__(
                self,
                attr,
                np.frombuffer(f_arr.tobytes(), dtype=np.float64).reshape(f_arr.shape),
            )
        acc = np.array(self.capacitance_accepted, dtype=bool, copy=True)
        acc.setflags(write=False)
        object.__setattr__(self, "capacitance_accepted", acc)


def capacitance_frequency_from_admittance(
    frequencies_Hz: ArrayLike,
    admittance_density_S_m2: ArrayLike,
    *,
    operating_voltage_V: float = 0.0,
    generation_scale: float = 0.0,
    active_species: Sequence[str] = (),
    frozen_species: Sequence[str] = (),
    working_state_digest: str = "",
    options: DefectCFOptions | None = None,
) -> DefectCFResult:
    """Compute C-f observables and Walter derivative from ``(f, Y(f))`` arrays."""
    controls = DefectCFOptions() if options is None else options
    freqs = np.asarray(frequencies_Hz, dtype=np.float64)
    y_arr = np.asarray(admittance_density_S_m2, dtype=np.complex128)
    if freqs.ndim != 1 or freqs.size == 0 or not np.all(np.isfinite(freqs)) or np.any(freqs < 0.0):
        raise ValueError("frequencies_Hz must be a non-empty 1D array of finite non-negative values")
    if y_arr.shape != freqs.shape:
        raise ValueError("admittance_density_S_m2 must match frequencies_Hz shape")
    if not math.isfinite(operating_voltage_V) or not math.isfinite(generation_scale) or generation_scale < 0.0:
        raise ValueError("operating_voltage_V must be finite and generation_scale >= 0")

    count = freqs.size
    omega = 2.0 * math.pi * freqs
    z_arr: ComplexVector = np.full(count, np.nan + 1j * np.nan, dtype=np.complex128)
    cp: Vector = np.full(count, np.nan, dtype=np.float64)
    gp: Vector = np.full(count, np.nan, dtype=np.float64)
    cs: Vector = np.full(count, np.nan, dtype=np.float64)
    tan_d: Vector = np.full(count, np.nan, dtype=np.float64)
    g_over_w: Vector = np.full(count, np.nan, dtype=np.float64)
    phase: Vector = np.full(count, np.nan, dtype=np.float64)
    accepted: BoolVector = np.zeros(count, dtype=bool)
    reasons: list[str | None] = [None] * count

    for idx in range(count):
        f_val = float(freqs[idx])
        w_val = float(omega[idx])
        y_val = complex(y_arr[idx])
        g_val = float(y_val.real)
        b_val = float(y_val.imag)
        if not (math.isfinite(g_val) and math.isfinite(b_val)):
            if controls.strict_capacitive:
                raise DefectCFError(f"non-finite admittance at f={f_val:.4g} Hz")
            reasons[idx] = "nonfinite_admittance"
            continue

        gp[idx] = g_val
        phase[idx] = math.atan2(b_val, g_val)
        if y_val != 0.0j:
            z_arr[idx] = 1.0 / y_val

        if f_val <= 0.0:
            if controls.strict_capacitive:
                raise DefectCFError("capacitance extraction requires strictly positive frequency (f > 0)")
            reasons[idx] = "zero_frequency"
            continue

        g_over_w[idx] = g_val / w_val
        if b_val < 0.0:
            if controls.strict_capacitive:
                raise DefectCFError(
                    f"inductive susceptance Im(Y)={b_val:.3e} S/m^2 at f={f_val:.4g} Hz"
                )
            reasons[idx] = "inductive_susceptance"
            continue
        if b_val == 0.0:
            if controls.strict_capacitive:
                raise DefectCFError(f"zero susceptance Im(Y)=0 at f={f_val:.4g} Hz")
            reasons[idx] = "zero_susceptance"
            continue

        if controls.minimum_phase_angle_rad > 0.0 and math.atan2(b_val, abs(g_val)) < controls.minimum_phase_angle_rad:
            if controls.strict_capacitive:
                raise DefectCFError(
                    f"unresolved capacitive phase at f={f_val:.4g} Hz"
                )
            tan_d[idx] = g_val / b_val
            reasons[idx] = "unresolved_phase"
            continue

        cp[idx] = b_val / w_val
        cs[idx] = (g_val * g_val + b_val * b_val) / (w_val * b_val)
        tan_d[idx] = g_val / b_val
        accepted[idx] = True
        reasons[idx] = None

    walter: Vector = np.full(count, np.nan, dtype=np.float64)
    acc_indices = np.flatnonzero(accepted)
    peak_f: float | None = None
    peak_w: float | None = None
    if acc_indices.size >= 2:
        w_acc = omega[acc_indices]
        c_acc = cp[acc_indices]
        if np.all(np.diff(w_acc) > 0.0):
            ln_w = np.log(w_acc)
            dcp_dlnw = np.gradient(c_acc, ln_w)
            w_vals = -dcp_dlnw
            walter[acc_indices] = w_vals
            best_local = int(np.argmax(w_vals))
            peak_f = float(freqs[acc_indices[best_local]])
            peak_w = float(w_vals[best_local])

    dark_passive: bool | None = None
    if generation_scale == 0.0 and abs(operating_voltage_V) <= 1.0e-12:
        finite_g = gp[np.isfinite(gp)]
        dark_passive = bool(np.all(finite_g >= -controls.passivity_tolerance_S_m2))
        if controls.require_dark_passivity and not dark_passive:
            raise DefectCFError(
                f"dark 0 V passivity violated: min Re(Y)={float(np.min(finite_g)):.3e} S/m^2"
            )

    return DefectCFResult(
        frequencies_Hz=freqs,
        angular_frequencies_rad_s=omega,
        admittance_density_S_m2=y_arr,
        impedance_area_ohm_m2=z_arr,
        parallel_capacitance_F_m2=cp,
        parallel_conductance_S_m2=gp,
        series_capacitance_F_m2=cs,
        loss_tangent=tan_d,
        scaled_conductance_F_m2=g_over_w,
        walter_derivative_F_m2=walter,
        phase_angle_rad=phase,
        capacitance_accepted=accepted,
        mask_reasons=tuple(reasons),
        operating_voltage_V=float(operating_voltage_V),
        generation_scale=float(generation_scale),
        active_species=tuple(active_species),
        frozen_species=tuple(frozen_species),
        working_state_digest=str(working_state_digest),
        dark_equilibrium_passive=dark_passive,
        peak_walter_frequency_Hz=peak_f,
        peak_walter_value_F_m2=peak_w,
    )


def capacitance_frequency_from_sweep(
    sweep: DynamicDefectACSweepResult,
    *,
    options: DefectCFOptions | None = None,
) -> DefectCFResult:
    """Extract C-f observables from an existing ``DynamicDefectACSweepResult``."""
    return capacitance_frequency_from_admittance(
        sweep.frequencies_Hz,
        sweep.admittance_density_S_m2,
        operating_voltage_V=sweep.operating_voltage_V,
        generation_scale=sweep.generation_scale,
        active_species=sweep.active_species,
        frozen_species=sweep.frozen_species,
        working_state_digest=sweep.working_state_digest,
        options=options,
    )


def _layer_param_float(layer: object, key: str) -> float:
    params = dict(getattr(layer, "parameters", ()))
    val = params.get(key)
    return float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else 0.0


def _infer_active_ion_layer(prepared: PreparedDevice) -> str | None:
    elec_layers = [ly for ly in prepared.layers if ly.role != "substrate"]
    ion_layers = [
        ly.id
        for ly in elec_layers
        if (
            _layer_param_float(ly, "D_ion") > 0.0
            and _layer_param_float(ly, "P0") > 0.0
        )
        or (
            _layer_param_float(ly, "D_ion_neg") > 0.0
            and _layer_param_float(ly, "P0_neg") > 0.0
        )
    ]
    return ion_layers[0] if len(ion_layers) == 1 else None


def run_defect_capacitance_frequency(
    device_or_prepared: CompiledDynamicDefectDevice1D | PreparedDevice,
    frequencies_Hz: ArrayLike,
    *,
    voltage_V: float = 0.0,
    generation_scale: float = 0.0,
    intervals_per_layer: int | Sequence[int] = 24,
    energy_quadrature_order: int = DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    dynamic_bulk_defects: bool | None = None,
    dynamic_interface_defects: bool | None = None,
    ion_domain_layer: str | None = None,
    ion_domain_endpoints: tuple[str, str] = ("blocking", "blocking"),
    freeze_ions: bool = False,
    freeze_bulk_traps: bool = False,
    freeze_interface_traps: bool = False,
    options: DefectCFOptions | None = None,
) -> DefectCFResult:
    """Solve coupled DC + small-signal AC and extract C-f observables."""
    controls = DefectCFOptions() if options is None else options
    if isinstance(device_or_prepared, CompiledDynamicDefectDevice1D):
        compiled = device_or_prepared
    elif isinstance(device_or_prepared, PreparedDevice):
        has_bulk = any(
            ly.role != "substrate"
            and (bool(ly.defects) or ly.bulk_trap_distribution_json is not None)
            and ly.defect_schema_version != "solarlab-explicit-bulk-defects-v4"
            for ly in device_or_prepared.layers
        )
        has_iface = any(
            iface.electrical and iface.to_mapping().get("defect") is not None
            for iface in device_or_prepared.interfaces
        )
        inferred_ion = (
            ion_domain_layer
            if ion_domain_layer is not None
            else _infer_active_ion_layer(device_or_prepared)
        )
        compiled = compile_dynamic_defect_device_1d(
            device_or_prepared,
            intervals_per_layer=intervals_per_layer,
            energy_quadrature_order=energy_quadrature_order,
            dynamic_bulk_defects=has_bulk if dynamic_bulk_defects is None else dynamic_bulk_defects,
            dynamic_interface_defects=(
                has_iface if dynamic_interface_defects is None else dynamic_interface_defects
            ),
            ion_domain_layer=inferred_ion,
            ion_domain_endpoints=ion_domain_endpoints,
        )
    else:
        raise TypeError("device_or_prepared must be CompiledDynamicDefectDevice1D or PreparedDevice")

    steady = solve_dynamic_defect_steady_1d(
        compiled,
        voltage_V=voltage_V,
        generation_scale=generation_scale,
        options=controls.steady_options,
    )
    ss = DynamicDefectSmallSignal1D(
        compiled,
        steady.state,
        generation_scale=generation_scale,
    )
    sweep = ss.solve_sweep(
        frequencies_Hz,
        voltage_phasor_V=controls.voltage_phasor_V,
        options=controls.linear_options,
        freeze_ions=freeze_ions,
        freeze_bulk_traps=freeze_bulk_traps,
        freeze_interface_traps=freeze_interface_traps,
    )
    return capacitance_frequency_from_sweep(sweep, options=controls)
