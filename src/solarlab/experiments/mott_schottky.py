"""AC capacitance-frequency C(omega), capacitance-voltage C(V), and Mott-Schottky.

Built directly on the exact linearized small-signal operators around accepted DC
operating points:

- ``solve_impedance`` for ion-free ``CompiledDevice1D``
- ``solve_ionic_impedance`` for single-domain ``CompiledIonicDevice1D``
  (``ion_mode="mobile"`` or ``"frozen"``)
- ``solve_multiple_ionic_impedance`` for multi-domain
  ``CompiledMultipleIonicDevice1D``

Key scientific and numerical contracts (P05-03a):

1. **Active vs frozen species**: ``active_species`` and ``frozen_species`` are
   recorded explicitly on every CF, CV, and Mott-Schottky result.
2. **Sign-faithful capacitive susceptance**: ``C = Im(Y) / (2 * pi * f)`` is
   reported only where ``f > 0`` and ``Im(Y) > 0``. Inductive or zero-susceptance
   points are never hidden with ``abs(Im(Y))`` or ``0.0``; they record ``np.nan``,
   ``valid_capacitive_mask[i] = False``, and an explicit entry in ``point_reasons``
   (or raise ``RuntimeError`` when ``strict_capacitive=True``).
3. **Apparent built-in voltage**: The Mott-Schottky intercept is explicitly
   named ``apparent_built_in_voltage_V`` (not an unqualified ``V_bi``), alongside
   ``apparent_doping_m3``, ``V_fit_lo_V``, ``V_fit_hi_V``, ``fit_mask``, and
   ``r_squared``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Literal, TypeAlias, cast

import numpy as np

from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.device.resolved import PreparedDevice
from solarlab.discretization.face_average import VACUUM_PERMITTIVITY_F_M
from solarlab.equations.dc_1d import CompiledDevice1D
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, IonicState1D
from solarlab.equations.ionic_closure import ion_mobility_partition
from solarlab.equations.multiple_ionic_1d import (
    CompiledMultipleIonicDevice1D,
    MultipleIonicState1D,
)
from solarlab.experiments.impedance import solve_impedance
from solarlab.experiments.ionic_impedance import (
    IonicImpedanceAttempt,
    IonicImpedanceOptions,
    solve_ionic_impedance,
)
from solarlab.experiments.ionic_transient import IonicTransientInitial
from solarlab.experiments.multiple_ionic_impedance import (
    solve_multiple_ionic_impedance,
)
from solarlab.experiments.multiple_ionic_transient import (
    MultipleIonicTransientInitial,
)
from solarlab.experiments.results import encode_array
from solarlab.experiments.suns_voc import (
    default_ionic_steady_options,
    solve_ionic_dc_operating_point,
)
from solarlab.experiments.waveform_jv import package_ionic_steady_initial
from solarlab.reproducibility.job_provenance import current_provenance
from solarlab.solvers.linear import FrequencyOptions
from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions
from solarlab.solvers.nonlinear.multiple_ionic_1d import (
    MultipleIonicSteadyResult,
    solve_multiple_ionic_steady_state,
)
from solarlab.solvers.nonlinear.steady_1d import (
    SteadyOptions,
    SteadyState1D,
    SteadyStateError,
    solve_steady_state,
)

__all__ = [
    "CFOptions",
    "CFResult",
    "CVResult",
    "MottSchottkyOptions",
    "MottSchottkyResult",
    "fit_mott_schottky_curve",
    "package_multiple_ionic_steady_initial",
    "resolve_depletion_permittivity",
    "run_capacitance_frequency",
    "run_capacitance_voltage",
    "run_mott_schottky",
    "select_mott_schottky_window",
]

BOLTZMANN_J_K = 1.380649e-23
ProgressCallback = Callable[[str, int, int, str], None]
AnyCompiledACDevice: TypeAlias = (
    CompiledDevice1D | CompiledIonicDevice1D | CompiledMultipleIonicDevice1D
)


def _ro_float(value: Any) -> np.ndarray:
    arr = np.array(value, dtype=np.float64, copy=True)
    arr.setflags(write=False)
    return arr


def _ro_bool(value: Any) -> np.ndarray:
    arr = np.array(value, dtype=np.bool_, copy=True)
    arr.setflags(write=False)
    return arr


def _ro_complex(value: Any) -> np.ndarray:
    arr = np.array(value, dtype=np.complex128, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class CFOptions:
    """Explicit DC, linear AC, and species-mode controls for CF/CV sweeps."""

    steady: SteadyOptions = field(default_factory=SteadyOptions)
    ionic_steady: IonicSteadyOptions | None = None
    linear: FrequencyOptions = field(default_factory=FrequencyOptions)
    ionic_impedance: IonicImpedanceOptions = field(
        default_factory=IonicImpedanceOptions
    )
    ion_mode: Literal["mobile", "frozen"] = "mobile"
    voltage_phasor_V: complex = 1.0 + 0.0j
    strict_capacitive: bool = False
    require_all_dc_converged: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.steady, SteadyOptions):
            raise ValueError("steady must be an explicit SteadyOptions instance")
        if self.ionic_steady is not None and not isinstance(
            self.ionic_steady, IonicSteadyOptions
        ):
            raise ValueError(
                "ionic_steady must be None or an explicit IonicSteadyOptions"
            )
        if not isinstance(self.linear, FrequencyOptions):
            raise ValueError("linear must be an explicit FrequencyOptions instance")
        if not isinstance(self.ionic_impedance, IonicImpedanceOptions):
            raise ValueError(
                "ionic_impedance must be an explicit IonicImpedanceOptions"
            )
        if self.ion_mode not in ("mobile", "frozen"):
            raise ValueError("ion_mode must be 'mobile' or 'frozen'")
        phasor = complex(self.voltage_phasor_V)
        if not np.isfinite(phasor) or phasor == 0.0j:
            raise ValueError("voltage_phasor_V must be finite and nonzero")
        object.__setattr__(self, "voltage_phasor_V", phasor)


@dataclass(frozen=True)
class MottSchottkyOptions:
    """Controls for CV probing and Mott-Schottky linear window extraction."""

    ac: CFOptions = field(default_factory=CFOptions)
    min_window_points: int = 4
    max_residual_ratio: float = 0.01
    min_relative_variation: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.ac, CFOptions):
            raise ValueError("ac must be an explicit CFOptions instance")
        if type(self.min_window_points) is not int or self.min_window_points < 3:
            raise ValueError("min_window_points must be an integer >= 3")
        if (
            not math.isfinite(self.max_residual_ratio)
            or self.max_residual_ratio <= 0.0
        ):
            raise ValueError("max_residual_ratio must be finite and positive")
        if (
            not math.isfinite(self.min_relative_variation)
            or self.min_relative_variation < 0.0
        ):
            raise ValueError("min_relative_variation must be finite and >= 0")


@dataclass(frozen=True)
class CFResult:
    """Small-signal capacitance-frequency C(omega) sweep at fixed DC bias."""

    frequencies_Hz: np.ndarray
    bias_voltage_V: float
    generation_scale: float
    admittance_density_S_m2: np.ndarray
    capacitance_F_m2: np.ndarray
    conductance_S_m2: np.ndarray
    loss_tangent: np.ndarray
    ac_accepted_mask: np.ndarray
    valid_capacitive_mask: np.ndarray
    point_reasons: tuple[str | None, ...]
    active_species: tuple[str, ...]
    frozen_species: tuple[str, ...]
    compiled_identity: str
    job_provenance: dict[str, Any] | None = field(default_factory=current_provenance)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "admittance_density_S_m2",
            _ro_complex(self.admittance_density_S_m2),
        )
        for name in (
            "frequencies_Hz",
            "capacitance_F_m2",
            "conductance_S_m2",
            "loss_tangent",
        ):
            object.__setattr__(self, name, _ro_float(getattr(self, name)))
        for mask_name in ("ac_accepted_mask", "valid_capacitive_mask"):
            object.__setattr__(self, mask_name, _ro_bool(getattr(self, mask_name)))


@dataclass(frozen=True)
class CVResult:
    """Small-signal capacitance-voltage C(V) sweep at fixed AC frequency."""

    voltage_V: np.ndarray
    frequency_Hz: float
    generation_scale: float
    admittance_density_S_m2: np.ndarray
    capacitance_F_m2: np.ndarray
    conductance_S_m2: np.ndarray
    dc_converged_mask: np.ndarray
    ac_accepted_mask: np.ndarray
    valid_capacitive_mask: np.ndarray
    point_reasons: tuple[str | None, ...]
    active_species: tuple[str, ...]
    frozen_species: tuple[str, ...]
    compiled_identity: str
    job_provenance: dict[str, Any] | None = field(default_factory=current_provenance)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "admittance_density_S_m2",
            _ro_complex(self.admittance_density_S_m2),
        )
        for name in (
            "voltage_V",
            "capacitance_F_m2",
            "conductance_S_m2",
        ):
            object.__setattr__(self, name, _ro_float(getattr(self, name)))
        for mask_name in (
            "dc_converged_mask",
            "ac_accepted_mask",
            "valid_capacitive_mask",
        ):
            object.__setattr__(self, mask_name, _ro_bool(getattr(self, mask_name)))


@dataclass(frozen=True)
class MottSchottkyResult:
    """C-V sweep and Mott-Schottky 1/C^2 linear depletion extraction."""

    voltage_V: np.ndarray
    capacitance_F_m2: np.ndarray
    one_over_C2_m4_F2: np.ndarray
    dc_converged_mask: np.ndarray
    ac_accepted_mask: np.ndarray
    valid_capacitive_mask: np.ndarray
    point_reasons: tuple[str | None, ...]
    fit_mask: np.ndarray
    apparent_built_in_voltage_V: float
    apparent_doping_m3: float
    slope_m4_F2_V: float
    intercept_m4_F2: float
    V_fit_lo_V: float
    V_fit_hi_V: float
    r_squared: float
    fit_failure_reason: str | None
    frequency_Hz: float
    eps_r_used: float
    temperature_K: float
    active_species: tuple[str, ...]
    frozen_species: tuple[str, ...]
    compiled_identity: str
    job_provenance: dict[str, Any] | None = field(default_factory=current_provenance)

    def __post_init__(self) -> None:
        for name in (
            "voltage_V",
            "capacitance_F_m2",
            "one_over_C2_m4_F2",
        ):
            object.__setattr__(self, name, _ro_float(getattr(self, name)))
        for mask_name in (
            "dc_converged_mask",
            "ac_accepted_mask",
            "valid_capacitive_mask",
            "fit_mask",
        ):
            object.__setattr__(self, mask_name, _ro_bool(getattr(self, mask_name)))

    @property
    def V(self) -> np.ndarray:
        return self.voltage_V

    @property
    def C(self) -> np.ndarray:
        return self.capacitance_F_m2

    @property
    def one_over_C2(self) -> np.ndarray:
        return self.one_over_C2_m4_F2

    @property
    def V_bi_fit(self) -> float:
        return self.apparent_built_in_voltage_V

    @property
    def N_eff_fit(self) -> float:
        return self.apparent_doping_m3

    @property
    def V_fit_lo(self) -> float:
        return self.V_fit_lo_V

    @property
    def V_fit_hi(self) -> float:
        return self.V_fit_hi_V

    @property
    def frequency(self) -> float:
        return self.frequency_Hz

    @property
    def fit_window_V(self) -> tuple[float, float]:
        return (self.V_fit_lo_V, self.V_fit_hi_V)


def _species_partition(
    compiled: AnyCompiledACDevice, ion_mode: Literal["mobile", "frozen"]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if isinstance(compiled, CompiledDevice1D):
        return ("electrons", "holes"), ()
    if isinstance(compiled, CompiledIonicDevice1D):
        mobile, immobile = ion_mobility_partition(compiled)
        if ion_mode == "frozen":
            return (
                ("electrons", "holes"),
                tuple(f"ion:{s}" for s in compiled.ion_species_names),
            )
        return (
            ("electrons", "holes", *(f"ion:{s}" for s in mobile)),
            tuple(f"ion:{s}" for s in immobile),
        )
    if isinstance(compiled, CompiledMultipleIonicDevice1D):
        if ion_mode == "frozen":
            return (
                ("electrons", "holes"),
                tuple(
                    f"{d.domain_layer_id}:{s}"
                    for d in compiled.domains
                    for s in d.ion_species_names
                ),
            )
        active_ions: list[str] = []
        frozen_ions: list[str] = []
        for dom in compiled.domains:
            mob, imm = ion_mobility_partition(dom)
            active_ions.extend(f"{dom.domain_layer_id}:{s}" for s in mob)
            frozen_ions.extend(f"{dom.domain_layer_id}:{s}" for s in imm)
        return ("electrons", "holes", *active_ions), tuple(frozen_ions)
    raise TypeError(f"unsupported AC device type: {type(compiled).__name__}")


def _encode_diagnostic_tree(value: Any) -> Any:
    """Lossless public codec matching ``solarlab.multiple_ionic_steady.v1``."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _encode_diagnostic_tree(getattr(value, f.name))
            for f in fields(value)
        }
    if type(value) is np.ndarray:
        return {
            "$array": encode_array(
                value, unit="raw diagnostic; units follow the named producer field"
            )
        }
    if isinstance(value, np.generic):
        return {"$scalar": encode_array(np.asarray(value), unit="raw diagnostic")}
    if isinstance(value, Enum):
        return _encode_diagnostic_tree(value.value)
    if type(value) is tuple:
        return {"$tuple": [_encode_diagnostic_tree(item) for item in value]}
    if type(value) is list:
        return [_encode_diagnostic_tree(item) for item in value]
    if type(value) is dict:
        return {str(k): _encode_diagnostic_tree(v) for k, v in value.items()}
    if type(value) is float and not math.isfinite(value):
        return {"$float": encode_array(np.asarray(value), unit="raw diagnostic")}
    return value


def package_multiple_ionic_steady_initial(
    device: CompiledMultipleIonicDevice1D,
    point: MultipleIonicSteadyResult,
    *,
    controls: IonicSteadyOptions,
    source_label: str = "in_memory://multiple_ionic_steady.json",
) -> MultipleIonicTransientInitial:
    """Package an accepted ``MultipleIonicSteadyResult`` for multi-domain AC."""
    if not point.accepted or point.state is None:
        reason = point.error or point.termination
        raise RuntimeError(
            f"cannot initialize multi-domain AC from unaccepted DC result ({reason})"
        )
    doc = {
        "schema": "solarlab.multiple_ionic_steady.v1",
        "result": _encode_diagnostic_tree(point),
        "qualification": False,
    }
    evidence_json = json.dumps(doc, sort_keys=True, allow_nan=False) + "\n"
    digest = hashlib.sha256(evidence_json.encode("utf-8")).hexdigest()
    return MultipleIonicTransientInitial(
        state=point.state,
        generation_scale=0.0,
        source_path=source_label,
        source_sha256=digest,
        evidence_json=evidence_json,
        steady_controls=controls,
    )


def resolve_depletion_permittivity(
    compiled: AnyCompiledACDevice,
    *,
    prepared: PreparedDevice | None = None,
    eps_r: float | None = None,
) -> float:
    """Resolve the relative permittivity ``eps_r`` of the depletion region."""
    if eps_r is not None:
        val = float(eps_r)
        if not math.isfinite(val) or val <= 0.0:
            raise ValueError("eps_r must be finite and positive")
        return val

    if prepared is not None:
        for layer in prepared.layers:
            if layer.role == "absorber":
                return float(cast(float, dict(layer.parameters)["eps_r"]))
        best_eps: float | None = None
        best_doping = float("inf")
        for layer in prepared.layers:
            if layer.role == "substrate":
                continue
            params = dict(layer.parameters)
            doping = max(
                float(cast(float, params.get("N_A", 0.0))),
                float(cast(float, params.get("N_D", 0.0))),
                float(cast(float, params.get("ni", 0.0))),
            )
            if doping < best_doping:
                best_doping = doping
                best_eps = float(cast(float, params["eps_r"]))
        if best_eps is not None:
            return best_eps
        raise ValueError("prepared device contains no electrical layer with eps_r")

    elec = (
        compiled
        if isinstance(compiled, CompiledDevice1D)
        else compiled.electronic_component
    )
    dx = np.diff(elec.nodes_m)
    face_eps = elec.poisson.face_capacitance_F_m2 * dx / VACUUM_PERMITTIVITY_F_M
    n_ref = elec.reference_electron_density_m3
    p_ref = elec.reference_hole_density_m3
    face_carrier_scale = 0.5 * (
        np.maximum(n_ref[:-1], p_ref[:-1]) + np.maximum(n_ref[1:], p_ref[1:])
    )
    idx = int(np.argmin(face_carrier_scale))
    return float(face_eps[idx])


def _capacitance_from_admittance(
    freq_hz: float,
    y_val: complex,
    *,
    strict_capacitive: bool,
) -> tuple[float, float, float, bool, str | None]:
    g_val = float(y_val.real)
    b_val = float(y_val.imag)
    if freq_hz <= 0.0:
        if strict_capacitive:
            raise RuntimeError(
                "capacitance extraction requires strictly positive AC frequency (f > 0)"
            )
        return float("nan"), g_val, float("nan"), False, "zero_frequency"
    if b_val < 0.0:
        if strict_capacitive:
            raise RuntimeError(
                f"inductive susceptance Im(Y)={b_val:.3e} S/m^2 at f={freq_hz:.4g} Hz"
            )
        return float("nan"), g_val, float("nan"), False, "inductive_susceptance"
    if b_val == 0.0:
        if strict_capacitive:
            raise RuntimeError(f"zero susceptance Im(Y)=0 at f={freq_hz:.4g} Hz")
        return float("nan"), g_val, float("nan"), False, "zero_susceptance"
    cap = float(b_val / (2.0 * math.pi * freq_hz))
    tan_d = float(g_val / b_val)
    return cap, g_val, tan_d, True, None


def _unpack_ionic_attempts(
    requested_Hz: np.ndarray,
    attempts: Sequence[IonicImpedanceAttempt],
    *,
    strict_capacitive: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str | None],
]:
    count = int(requested_Hz.size)
    y_arr: np.ndarray = np.full(count, np.nan + 1j * np.nan, dtype=complex)
    c_arr: np.ndarray = np.full(count, np.nan, dtype=float)
    g_arr: np.ndarray = np.full(count, np.nan, dtype=float)
    loss_arr: np.ndarray = np.full(count, np.nan, dtype=float)
    ac_ok: np.ndarray = np.zeros(count, dtype=bool)
    cap_ok: np.ndarray = np.zeros(count, dtype=bool)
    reasons: list[str | None] = [None] * count

    for idx in range(count):
        if idx >= len(attempts):
            reasons[idx] = "ac_not_evaluated:preceding_frequency_failed"
            continue
        att = attempts[idx]
        if not att.accepted or att.observation is None:
            reason = att.error or att.termination
            reasons[idx] = f"ac_not_accepted:{reason}"
            continue
        y_val = complex(att.observation["admittance_density_S_m2"])
        y_arr[idx] = y_val
        ac_ok[idx] = True
        cap, g_val, tan_d, ok_c, r_c = _capacitance_from_admittance(
            float(requested_Hz[idx]), y_val, strict_capacitive=strict_capacitive
        )
        c_arr[idx] = cap
        g_arr[idx] = g_val
        loss_arr[idx] = tan_d
        cap_ok[idx] = ok_c
        reasons[idx] = r_c
    return y_arr, c_arr, g_arr, loss_arr, ac_ok, cap_ok, reasons


def run_capacitance_frequency(
    compiled: AnyCompiledACDevice,
    frequencies_Hz: Sequence[float] | np.ndarray,
    *,
    bias_voltage_V: float = 0.0,
    generation_scale: float = 0.0,
    initial: SteadyState1D
    | IonicTransientInitial
    | MultipleIonicTransientInitial
    | None = None,
    options: CFOptions | None = None,
) -> CFResult:
    """Evaluate C(omega) around an accepted DC operating point."""
    opts = CFOptions() if options is None else options
    freqs: np.ndarray = np.asarray([float(f) for f in frequencies_Hz], dtype=float)
    if (
        freqs.ndim != 1
        or freqs.size == 0
        or not np.all(np.isfinite(freqs))
        or np.any(freqs < 0.0)
        or np.unique(freqs).size != freqs.size
    ):
        raise ValueError(
            "frequencies_Hz must be a non-empty 1D sequence of distinct finite >= 0 Hz"
        )
    v_bias = float(bias_voltage_V)
    gen = float(generation_scale)
    if not math.isfinite(v_bias) or not math.isfinite(gen) or gen < 0.0:
        raise ValueError("bias_voltage_V must be finite and generation_scale >= 0")

    active_sp, frozen_sp = _species_partition(compiled, opts.ion_mode)
    count = int(freqs.size)

    if isinstance(compiled, CompiledDevice1D):
        if initial is None:
            dc_sol = solve_steady_state(
                compiled, voltage_V=v_bias, generation_scale=gen, options=opts.steady
            )
            if not dc_sol.converged:
                raise RuntimeError(f"CF DC solve failed to converge at {v_bias} V")
        elif isinstance(initial, SteadyState1D):
            dc_sol = initial
        else:
            raise TypeError(
                "CompiledDevice1D requires initial=None or SteadyState1D"
            )

        ss = solve_impedance(
            compiled,
            dc_sol,
            freqs,
            voltage_phasor_V=opts.voltage_phasor_V,
            steady_options=opts.steady,
            linear_options=opts.linear,
        )
        y_arr: np.ndarray = np.array(
            ss.admittance_density_S_m2, dtype=complex, copy=True
        )
        c_arr: np.ndarray = np.full(count, np.nan, dtype=float)
        g_arr: np.ndarray = np.empty(count, dtype=float)
        loss_arr: np.ndarray = np.full(count, np.nan, dtype=float)
        ac_ok: np.ndarray = np.ones(count, dtype=bool)
        cap_ok: np.ndarray = np.zeros(count, dtype=bool)
        reasons: list[str | None] = [None] * count
        for idx in range(count):
            cap, g_val, tan_d, ok_c, r_c = _capacitance_from_admittance(
                float(freqs[idx]),
                complex(y_arr[idx]),
                strict_capacitive=opts.strict_capacitive,
            )
            c_arr[idx] = cap
            g_arr[idx] = g_val
            loss_arr[idx] = tan_d
            cap_ok[idx] = ok_c
            reasons[idx] = r_c

    elif isinstance(compiled, CompiledIonicDevice1D):
        ion_controls = (
            default_ionic_steady_options(
                compiled.ion_node_count, electronic=opts.steady
            )
            if opts.ionic_steady is None
            else opts.ionic_steady
        )
        if initial is None:
            dc_ion = solve_ionic_dc_operating_point(
                compiled,
                voltage_V=v_bias,
                generation_scale=gen,
                options=ion_controls,
            )
            ion_init = package_ionic_steady_initial(
                compiled,
                dc_ion,
                voltage_V=v_bias,
                generation_scale=gen,
                controls=ion_controls,
                source_label="in_memory://cf_ionic_steady.json",
            )
        elif isinstance(initial, IonicTransientInitial):
            ion_init = initial
            v_bias = float(ion_init.state.voltage_V)
            gen = float(ion_init.generation_scale)
        else:
            raise TypeError(
                "CompiledIonicDevice1D requires initial=None or IonicTransientInitial"
            )

        ion_ac = solve_ionic_impedance(
            compiled,
            ion_init,
            freqs,
            ion_mode=opts.ion_mode,
            voltage_phasor_V=opts.voltage_phasor_V,
            options=opts.ionic_impedance,
        )
        y_arr, c_arr, g_arr, loss_arr, ac_ok, cap_ok, reasons = _unpack_ionic_attempts(
            freqs, ion_ac.attempts, strict_capacitive=opts.strict_capacitive
        )

    elif isinstance(compiled, CompiledMultipleIonicDevice1D):
        if gen != 0.0:
            raise ValueError(
                "multiple-domain ionic AC currently admits only dark DC (generation_scale=0.0)"
            )
        ion_controls = (
            default_ionic_steady_options(
                min(d.ion_node_count for d in compiled.domains),
                electronic=opts.steady,
            )
            if opts.ionic_steady is None
            else opts.ionic_steady
        )
        if initial is None:
            dc_mult = solve_multiple_ionic_steady_state(
                compiled,
                voltage_V=v_bias,
                generation_scale=0.0,
                options=ion_controls,
            )
            mult_init = package_multiple_ionic_steady_initial(
                compiled, dc_mult, controls=ion_controls
            )
        elif isinstance(initial, MultipleIonicTransientInitial):
            mult_init = initial
            v_bias = float(mult_init.state.voltage_V)
        else:
            raise TypeError(
                "CompiledMultipleIonicDevice1D requires initial=None or MultipleIonicTransientInitial"
            )

        mult_ac = solve_multiple_ionic_impedance(
            compiled,
            mult_init,
            freqs,
            voltage_phasor_V=opts.voltage_phasor_V,
            ion_mode=opts.ion_mode,
            options=opts.ionic_impedance,
        )
        y_arr, c_arr, g_arr, loss_arr, ac_ok, cap_ok, reasons = _unpack_ionic_attempts(
            freqs, mult_ac.attempts, strict_capacitive=opts.strict_capacitive
        )
    else:
        raise TypeError(f"unsupported AC device type: {type(compiled).__name__}")

    return CFResult(
        frequencies_Hz=freqs,
        bias_voltage_V=v_bias,
        generation_scale=gen,
        admittance_density_S_m2=y_arr,
        capacitance_F_m2=c_arr,
        conductance_S_m2=g_arr,
        loss_tangent=loss_arr,
        ac_accepted_mask=ac_ok,
        valid_capacitive_mask=cap_ok,
        point_reasons=tuple(reasons),
        active_species=active_sp,
        frozen_species=frozen_sp,
        compiled_identity=compiled.identity,
    )


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


def _rebase_multiple_ionic_state(
    device: CompiledMultipleIonicDevice1D,
    state: MultipleIonicState1D,
    voltage_V: float,
) -> MultipleIonicState1D:
    if state.voltage_V == voltage_V:
        return state
    coords = state.u.copy()
    n_elec = 3 * device.node_count
    coords[:n_elec] = device.electronic_component.rebase_coordinates(
        coords[:n_elec], state.voltage_V, voltage_V
    )
    return device.state(coords, voltage_V=voltage_V)


def run_capacitance_voltage(
    compiled: AnyCompiledACDevice,
    voltage_V: Sequence[float] | np.ndarray,
    *,
    frequency_Hz: float = 1e5,
    generation_scale: float = 0.0,
    options: CFOptions | None = None,
    progress: ProgressCallback | None = None,
) -> CVResult:
    """Sweep DC bias ``voltage_V`` and probe small-signal C(V) at ``frequency_Hz``."""
    opts = CFOptions() if options is None else options
    freq = float(frequency_Hz)
    if not math.isfinite(freq) or freq <= 0.0:
        raise ValueError(f"frequency_Hz must be finite and positive, got {frequency_Hz}")
    gen = float(generation_scale)
    if not math.isfinite(gen) or gen < 0.0:
        raise ValueError("generation_scale must be finite and >= 0")

    raw_v = [float(v) for v in voltage_V]
    if len(raw_v) < 2 or any(not math.isfinite(v) for v in raw_v):
        raise ValueError("voltage_V must contain at least 2 finite bias values")
    voltages = np.array(sorted(set(raw_v)), dtype=float)
    if voltages.size < 2:
        raise ValueError("voltage_V must contain at least 2 distinct bias values")

    active_sp, frozen_sp = _species_partition(compiled, opts.ion_mode)
    count = int(voltages.size)
    y_arr: np.ndarray = np.full(count, np.nan + 1j * np.nan, dtype=complex)
    c_arr: np.ndarray = np.full(count, np.nan, dtype=float)
    g_arr: np.ndarray = np.full(count, dtype=float, fill_value=np.nan)
    dc_ok: np.ndarray = np.zeros(count, dtype=bool)
    ac_ok: np.ndarray = np.zeros(count, dtype=bool)
    cap_ok: np.ndarray = np.zeros(count, dtype=bool)
    reasons: list[str | None] = [None] * count
    freq_vec = np.array([freq], dtype=float)

    if isinstance(compiled, CompiledDevice1D):
        prev_state = None
        for idx, v_val in enumerate(voltages):
            try:
                dc_sol = solve_steady_state(
                    compiled,
                    voltage_V=float(v_val),
                    generation_scale=gen,
                    initial_state=prev_state,
                    options=opts.steady,
                )
                if not dc_sol.converged:
                    reasons[idx] = "dc_not_converged:unconverged_steady_result"
                    continue
            except SteadyStateError as exc:
                reasons[idx] = f"dc_not_converged:{exc}"
                continue

            prev_state = dc_sol.state
            dc_ok[idx] = True
            cf_pt = run_capacitance_frequency(
                compiled,
                freq_vec,
                bias_voltage_V=float(v_val),
                generation_scale=gen,
                initial=dc_sol,
                options=opts,
            )
            y_arr[idx] = cf_pt.admittance_density_S_m2[0]
            c_arr[idx] = cf_pt.capacitance_F_m2[0]
            g_arr[idx] = cf_pt.conductance_S_m2[0]
            ac_ok[idx] = bool(cf_pt.ac_accepted_mask[0])
            cap_ok[idx] = bool(cf_pt.valid_capacitive_mask[0])
            reasons[idx] = cf_pt.point_reasons[0]
            if progress is not None:
                progress(
                    "cv",
                    idx + 1,
                    count,
                    f"V={v_val:.3f} V, C={c_arr[idx]:.3e} F/m^2",
                )

    elif isinstance(compiled, CompiledIonicDevice1D):
        ion_controls = (
            default_ionic_steady_options(
                compiled.ion_node_count, electronic=opts.steady
            )
            if opts.ionic_steady is None
            else opts.ionic_steady
        )
        prev_ion: IonicState1D | None = None
        for idx, v_val in enumerate(voltages):
            dc_ion = solve_ionic_dc_operating_point(
                compiled,
                voltage_V=float(v_val),
                generation_scale=gen,
                options=ion_controls,
                base_state=prev_ion,
                base_generation_scale=gen,
            )
            if not dc_ion.accepted or dc_ion.state is None:
                reason = dc_ion.error or dc_ion.termination
                reasons[idx] = f"dc_not_converged:{reason}"
                continue

            prev_ion = dc_ion.state
            dc_ok[idx] = True
            ion_init = package_ionic_steady_initial(
                compiled,
                dc_ion,
                voltage_V=float(v_val),
                generation_scale=gen,
                controls=ion_controls,
                source_label=f"in_memory://cv_ionic_steady_{idx}.json",
            )
            cf_pt = run_capacitance_frequency(
                compiled,
                freq_vec,
                bias_voltage_V=float(v_val),
                generation_scale=gen,
                initial=ion_init,
                options=opts,
            )
            y_arr[idx] = cf_pt.admittance_density_S_m2[0]
            c_arr[idx] = cf_pt.capacitance_F_m2[0]
            g_arr[idx] = cf_pt.conductance_S_m2[0]
            ac_ok[idx] = bool(cf_pt.ac_accepted_mask[0])
            cap_ok[idx] = bool(cf_pt.valid_capacitive_mask[0])
            reasons[idx] = cf_pt.point_reasons[0]
            if progress is not None:
                progress(
                    "cv",
                    idx + 1,
                    count,
                    f"V={v_val:.3f} V, C={c_arr[idx]:.3e} F/m^2",
                )

    elif isinstance(compiled, CompiledMultipleIonicDevice1D):
        if gen != 0.0:
            raise ValueError(
                "multiple-domain ionic CV currently admits only dark DC (generation_scale=0.0)"
            )
        ion_controls = (
            default_ionic_steady_options(
                min(d.ion_node_count for d in compiled.domains),
                electronic=opts.steady,
            )
            if opts.ionic_steady is None
            else opts.ionic_steady
        )
        prev_mult: MultipleIonicState1D | None = None
        for idx, v_val in enumerate(voltages):
            seed_m = (
                None
                if prev_mult is None
                else _rebase_multiple_ionic_state(compiled, prev_mult, float(v_val))
            )
            dc_mult = solve_multiple_ionic_steady_state(
                compiled,
                voltage_V=float(v_val),
                generation_scale=0.0,
                options=ion_controls,
                initial_state=seed_m,
            )
            if not dc_mult.accepted or dc_mult.state is None:
                reason = dc_mult.error or dc_mult.termination
                reasons[idx] = f"dc_not_converged:{reason}"
                continue

            prev_mult = dc_mult.state
            dc_ok[idx] = True
            mult_init = package_multiple_ionic_steady_initial(
                compiled,
                dc_mult,
                controls=ion_controls,
                source_label=f"in_memory://cv_mult_ionic_steady_{idx}.json",
            )
            cf_pt = run_capacitance_frequency(
                compiled,
                freq_vec,
                bias_voltage_V=float(v_val),
                generation_scale=0.0,
                initial=mult_init,
                options=opts,
            )
            y_arr[idx] = cf_pt.admittance_density_S_m2[0]
            c_arr[idx] = cf_pt.capacitance_F_m2[0]
            g_arr[idx] = cf_pt.conductance_S_m2[0]
            ac_ok[idx] = bool(cf_pt.ac_accepted_mask[0])
            cap_ok[idx] = bool(cf_pt.valid_capacitive_mask[0])
            reasons[idx] = cf_pt.point_reasons[0]
            if progress is not None:
                progress(
                    "cv",
                    idx + 1,
                    count,
                    f"V={v_val:.3f} V, C={c_arr[idx]:.3e} F/m^2",
                )
    else:
        raise TypeError(f"unsupported AC device type: {type(compiled).__name__}")

    if opts.require_all_dc_converged and not bool(np.all(dc_ok)):
        bad = [
            f"V={voltages[i]:.4g}V ({reasons[i]})"
            for i in range(count)
            if not dc_ok[i]
        ]
        raise RuntimeError("CV DC operating points failed: " + ", ".join(bad))

    return CVResult(
        voltage_V=voltages,
        frequency_Hz=freq,
        generation_scale=gen,
        admittance_density_S_m2=y_arr,
        capacitance_F_m2=c_arr,
        conductance_S_m2=g_arr,
        dc_converged_mask=dc_ok,
        ac_accepted_mask=ac_ok,
        valid_capacitive_mask=cap_ok,
        point_reasons=tuple(reasons),
        active_species=active_sp,
        frozen_species=frozen_sp,
        compiled_identity=compiled.identity,
    )


def select_mott_schottky_window(
    voltage_V: Sequence[float] | np.ndarray,
    one_over_C2_m4_F2: Sequence[float] | np.ndarray,
    *,
    valid_mask: Sequence[bool] | np.ndarray | None = None,
    min_points: int = 4,
    max_residual_ratio: float = 0.01,
) -> tuple[int, int] | None:
    """Select the widest contiguous valid sub-slice that is linear in ``(V, 1/C^2)``."""
    v_arr: np.ndarray = np.asarray(voltage_V, dtype=float)
    y_arr: np.ndarray = np.asarray(one_over_C2_m4_F2, dtype=float)
    if v_arr.ndim != 1 or y_arr.shape != v_arr.shape:
        raise ValueError("voltage_V and one_over_C2_m4_F2 must have matching 1D shapes")
    mask = (
        np.isfinite(v_arr) & np.isfinite(y_arr) & (y_arr > 0.0)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
        & np.isfinite(v_arr)
        & np.isfinite(y_arr)
        & (y_arr > 0.0)
    )
    n_tot = int(v_arr.size)
    if n_tot < min_points:
        return None

    best: tuple[int, int] | None = None
    best_len = 0
    for lo in range(0, n_tot - min_points + 1):
        for hi in range(lo + min_points - 1, n_tot):
            if not np.all(mask[lo : hi + 1]):
                break
            v_sub = v_arr[lo : hi + 1]
            y_sub = y_arr[lo : hi + 1]
            slope, intercept = np.polyfit(v_sub, y_sub, 1)
            resid = y_sub - (slope * v_sub + intercept)
            y_span = float(y_sub.max() - y_sub.min())
            if y_span <= 0.0:
                continue
            rms = float(np.sqrt(np.mean(resid * resid)))
            win_len = hi - lo + 1
            if rms <= max_residual_ratio * y_span and win_len > best_len:
                best_len = win_len
                best = (lo, hi)
    return best


def fit_mott_schottky_curve(
    voltage_V: Sequence[float] | np.ndarray,
    capacitance_F_m2: Sequence[float] | np.ndarray,
    *,
    eps_r: float,
    temperature_K: float,
    valid_capacitive_mask: Sequence[bool] | np.ndarray | None = None,
    min_window_points: int = 4,
    max_residual_ratio: float = 0.01,
    min_relative_variation: float = 0.05,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    str | None,
]:
    """Fit ``1/C^2 = slope * V + intercept`` and extract apparent V_bi and N_eff."""
    v_arr: np.ndarray = np.asarray(voltage_V, dtype=float)
    c_arr: np.ndarray = np.asarray(capacitance_F_m2, dtype=float)
    if not math.isfinite(eps_r) or eps_r <= 0.0:
        raise ValueError("eps_r must be finite and positive")
    if not math.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError("temperature_K must be finite and positive")

    val_mask = (
        np.isfinite(v_arr) & np.isfinite(c_arr) & (c_arr > 0.0)
        if valid_capacitive_mask is None
        else np.asarray(valid_capacitive_mask, dtype=bool)
        & np.isfinite(v_arr)
        & np.isfinite(c_arr)
        & (c_arr > 0.0)
    )
    one_over_c2: np.ndarray = np.full(v_arr.shape, np.nan, dtype=float)
    one_over_c2[val_mask] = 1.0 / (c_arr[val_mask] * c_arr[val_mask])
    fit_mask: np.ndarray = np.zeros(v_arr.shape, dtype=bool)

    window = select_mott_schottky_window(
        v_arr,
        one_over_c2,
        valid_mask=val_mask,
        min_points=min_window_points,
        max_residual_ratio=max_residual_ratio,
    )
    if window is None:
        return (
            one_over_c2,
            fit_mask,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            "no_contiguous_linear_depletion_window",
        )

    lo, hi = window
    v_fit = v_arr[lo : hi + 1]
    y_fit = one_over_c2[lo : hi + 1]
    fit_mask[lo : hi + 1] = True

    y_span = float(y_fit.max() - y_fit.min())
    y_mid = float(np.median(np.abs(y_fit)))
    if y_mid <= 0.0 or (y_span / y_mid) < min_relative_variation:
        return (
            one_over_c2,
            fit_mask,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float(v_fit[0]),
            float(v_fit[-1]),
            float("nan"),
            "flat_geometric_capacitance_insufficient_variation",
        )

    slope, intercept = np.polyfit(v_fit, y_fit, 1)
    if not math.isfinite(slope) or slope >= -1e-30:
        return (
            one_over_c2,
            fit_mask,
            float("nan"),
            float("nan"),
            float(slope),
            float(intercept),
            float(v_fit[0]),
            float(v_fit[-1]),
            float("nan"),
            "nonnegative_mott_schottky_slope",
        )

    v_thermal = BOLTZMANN_J_K * float(temperature_K) / ELEMENTARY_CHARGE_C
    apparent_vbi = float(-intercept / slope + 2.0 * v_thermal)
    apparent_n = float(
        -2.0 / (ELEMENTARY_CHARGE_C * float(eps_r) * VACUUM_PERMITTIVITY_F_M * slope)
    )
    pred = slope * v_fit + intercept
    ss_res = float(np.sum((y_fit - pred) ** 2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r_sq = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 1.0

    return (
        one_over_c2,
        fit_mask,
        apparent_vbi,
        apparent_n,
        float(slope),
        float(intercept),
        float(v_fit[0]),
        float(v_fit[-1]),
        r_sq,
        None,
    )


def run_mott_schottky(
    compiled: AnyCompiledACDevice,
    *,
    voltage_V: Sequence[float] | np.ndarray | None = None,
    V_lo: float = -0.3,
    V_hi: float = 0.5,
    n_points: int = 20,
    frequency_Hz: float = 1e5,
    generation_scale: float = 0.0,
    prepared: PreparedDevice | None = None,
    eps_r: float | None = None,
    options: MottSchottkyOptions | None = None,
    progress: ProgressCallback | None = None,
) -> MottSchottkyResult:
    """Run a dark/illuminated C-V sweep and fit apparent V_bi and N_eff."""
    opts = MottSchottkyOptions() if options is None else options
    if voltage_V is None:
        if type(n_points) is not int or n_points < opts.min_window_points:
            raise ValueError(
                f"n_points must be an integer >= {opts.min_window_points}, got {n_points}"
            )
        if not math.isfinite(V_lo) or not math.isfinite(V_hi) or V_hi <= V_lo:
            raise ValueError(f"V_hi ({V_hi}) must exceed V_lo ({V_lo})")
        bias_grid: np.ndarray = np.linspace(
            float(V_lo), float(V_hi), n_points, dtype=float
        )
    else:
        bias_grid = np.asarray([float(v) for v in voltage_V], dtype=float)
        if bias_grid.ndim != 1 or bias_grid.size < opts.min_window_points:
            raise ValueError(
                f"voltage_V must have at least {opts.min_window_points} points"
            )

    eps_used = resolve_depletion_permittivity(
        compiled, prepared=prepared, eps_r=eps_r
    )
    elec = (
        compiled
        if isinstance(compiled, CompiledDevice1D)
        else compiled.electronic_component
    )
    temperature_K = float(
        elec.thermal_voltage_V * ELEMENTARY_CHARGE_C / BOLTZMANN_J_K
    )

    cv = run_capacitance_voltage(
        compiled,
        bias_grid,
        frequency_Hz=frequency_Hz,
        generation_scale=generation_scale,
        options=opts.ac,
        progress=progress,
    )
    (
        one_over_c2,
        fit_mask,
        app_vbi,
        app_n,
        slope,
        intercept,
        v_fit_lo,
        v_fit_hi,
        r_sq,
        fit_reason,
    ) = fit_mott_schottky_curve(
        cv.voltage_V,
        cv.capacitance_F_m2,
        eps_r=eps_used,
        temperature_K=temperature_K,
        valid_capacitive_mask=cv.valid_capacitive_mask,
        min_window_points=opts.min_window_points,
        max_residual_ratio=opts.max_residual_ratio,
        min_relative_variation=opts.min_relative_variation,
    )

    return MottSchottkyResult(
        voltage_V=cv.voltage_V,
        capacitance_F_m2=cv.capacitance_F_m2,
        one_over_C2_m4_F2=one_over_c2,
        dc_converged_mask=cv.dc_converged_mask,
        ac_accepted_mask=cv.ac_accepted_mask,
        valid_capacitive_mask=cv.valid_capacitive_mask,
        point_reasons=cv.point_reasons,
        fit_mask=fit_mask,
        apparent_built_in_voltage_V=app_vbi,
        apparent_doping_m3=app_n,
        slope_m4_F2_V=slope,
        intercept_m4_F2=intercept,
        V_fit_lo_V=v_fit_lo,
        V_fit_hi_V=v_fit_hi,
        r_squared=r_sq,
        fit_failure_reason=fit_reason,
        frequency_Hz=cv.frequency_Hz,
        eps_r_used=eps_used,
        temperature_K=temperature_K,
        active_species=cv.active_species,
        frozen_species=cv.frozen_species,
        compiled_identity=cv.compiled_identity,
    )
