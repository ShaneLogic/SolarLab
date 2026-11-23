"""External lumped DC and small-signal AC circuit wrappers for 1D experiments.

System-level series resistance, shunt resistance, and external parallel
capacitance live strictly outside the 1D semiconductor drift-diffusion PDE:
device contacts and compiled parameters are never mutated.

Sign and circuit conventions (photovoltaic output-positive):
- ``J_device > 0`` delivers positive power to an external load at ``V > 0``.
- The shunt branch sits across the device junction terminals ``V_junction``:
  ``J_shunt = V_junction / R_shunt`` and ``J_terminal = J_device - J_shunt``.
- The series resistance sits between the junction/shunt node and the external
  measurement terminals: ``V_terminal = V_junction - J_terminal * R_series``.
- Small-signal AC phasors follow ``exp(+i * omega * t)`` with passive input
  admittance per unit area ``Y(omega)`` [S/m^2]:
  ``Y_junc(omega) = Y_device(omega) + G_shunt + i * omega * C_ext``,
  ``Y_terminal(omega) = Y_junc(omega) / (1 + R_series * Y_junc(omega))``.

All electrical parasitics are area-normalized (``R_s``, ``R_sh`` in ``ohm*m^2``,
``C_ext`` in ``F/m^2``) and explicitly separated from lumped thermal parameters
(``C_th`` in ``J/(m^2*K)``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from solarlab.experiments.jv.steady import (
    SteadyJVMetrics,
    SteadyJVResult,
    steady_jv_metrics,
)
from solarlab.experiments.impedance import SmallSignalResult1D
from solarlab.experiments.ionic_impedance import IonicImpedanceResult

__all__ = [
    "EXTERNAL_AC_UNIT_SCHEMA",
    "EXTERNAL_DC_UNIT_SCHEMA",
    "ExternalACCircuitProtocol",
    "ExternalACImpedanceResult",
    "ExternalCircuitBranch",
    "ExternalCircuitError",
    "ExternalCircuitJVResult",
    "ExternalCircuitProtocol",
    "ExternalCircuitSteadyResult",
    "ExternalCircuitTopologyError",
    "apply_external_circuit",
    "apply_external_circuit_steady",
    "compose_external_ac_impedance",
    "map_external_circuit_branch",
]

EXTERNAL_DC_UNIT_SCHEMA: dict[str, str] = {
    "series_resistance_ohm_m2": "ohm*m^2",
    "shunt_resistance_ohm_m2": "ohm*m^2",
}

EXTERNAL_AC_UNIT_SCHEMA: dict[str, str] = {
    "series_resistance_ohm_m2": "ohm*m^2",
    "shunt_resistance_ohm_m2": "ohm*m^2",
    "external_capacitance_F_m2": "F/m^2",
}


class ExternalCircuitError(ValueError):
    """Invalid external-circuit protocol or uncertified electrical input."""


class ExternalCircuitTopologyError(ExternalCircuitError):
    """External series drop folded or singularized the terminal coordinate."""


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


def _complex_hex_list(values: np.ndarray) -> list[list[str]]:
    flat = np.asarray(values, dtype=complex).ravel()
    return [[float(z.real).hex(), float(z.imag).hex()] for z in flat]


def _freeze_float_1d(values: object, *, name: str, min_size: int = 1) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ExternalCircuitError(f"{name} must be real-valued")
    arr = np.array(values, dtype=float, copy=True)
    if arr.ndim != 1 or arr.size < min_size or not np.all(np.isfinite(arr)):
        raise ExternalCircuitError(
            f"{name} must be a 1-D finite array with at least {min_size} sample(s)"
        )
    arr.setflags(write=False)
    return arr


def _freeze_complex_1d(values: object, *, name: str, min_size: int = 1) -> np.ndarray:
    arr = np.array(values, dtype=complex, copy=True)
    if arr.ndim != 1 or arr.size < min_size or not np.all(np.isfinite(arr)):
        raise ExternalCircuitError(
            f"{name} must be a 1-D finite complex array with at least {min_size} sample(s)"
        )
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class ExternalCircuitProtocol:
    """Area-normalized lumped external DC parasitic elements."""

    series_resistance_ohm_m2: float = 0.0
    shunt_resistance_ohm_m2: float | None = None
    shunt_voltage_reference: str = "junction"
    current_convention: str = "photovoltaic_output_positive"

    def __post_init__(self) -> None:
        if not _is_real_number(self.series_resistance_ohm_m2):
            raise ExternalCircuitError("series_resistance_ohm_m2 must be a real scalar")
        rs = float(self.series_resistance_ohm_m2)
        if not math.isfinite(rs) or rs < 0.0:
            raise ExternalCircuitError(
                f"series_resistance_ohm_m2 must be finite and >= 0, got {self.series_resistance_ohm_m2!r}"
            )
        object.__setattr__(self, "series_resistance_ohm_m2", rs)

        if self.shunt_resistance_ohm_m2 is not None:
            if not _is_real_number(self.shunt_resistance_ohm_m2):
                raise ExternalCircuitError(
                    "shunt_resistance_ohm_m2 must be None or a positive real scalar"
                )
            rsh = float(self.shunt_resistance_ohm_m2)
            if not math.isfinite(rsh) or rsh <= 0.0:
                raise ExternalCircuitError(
                    f"shunt_resistance_ohm_m2 must be finite and > 0 when set, got {self.shunt_resistance_ohm_m2!r}"
                )
            object.__setattr__(self, "shunt_resistance_ohm_m2", rsh)

        if self.shunt_voltage_reference != "junction":
            raise ExternalCircuitError(
                f"Unsupported shunt_voltage_reference {self.shunt_voltage_reference!r}; only 'junction' is supported"
            )
        if self.current_convention != "photovoltaic_output_positive":
            raise ExternalCircuitError(
                f"Unsupported current_convention {self.current_convention!r}; "
                "only 'photovoltaic_output_positive' is supported"
            )

    @property
    def parameter_units(self) -> dict[str, str]:
        return dict(EXTERNAL_DC_UNIT_SCHEMA)


@dataclass(frozen=True)
class ExternalCircuitBranch:
    """Mapped single-branch J-V curve with explicit series/shunt loss accounting."""

    direction: str
    V_junction: np.ndarray
    J_device: np.ndarray
    V_terminal: np.ndarray
    J_terminal: np.ndarray
    J_shunt: np.ndarray
    series_voltage_drop_V: np.ndarray
    power_terminal_W_m2: np.ndarray
    series_loss_W_m2: np.ndarray
    shunt_loss_W_m2: np.ndarray
    metrics: SteadyJVMetrics


@dataclass(frozen=True)
class ExternalCircuitSteadyResult:
    """Single-branch steady J-V curve mapped to external terminals."""

    protocol: ExternalCircuitProtocol
    branch: ExternalCircuitBranch
    incident_power_W_m2: float | None
    source_identity: str | None
    source_sha256: str
    result_sha256: str


@dataclass(frozen=True)
class ExternalCircuitJVResult:
    """Forward and reverse J-V branches mapped to external terminals."""

    protocol: ExternalCircuitProtocol
    V_junction_fwd: np.ndarray
    J_device_fwd: np.ndarray
    V_terminal_fwd: np.ndarray
    J_terminal_fwd: np.ndarray
    J_shunt_fwd: np.ndarray
    metrics_fwd: SteadyJVMetrics
    V_junction_rev: np.ndarray
    J_device_rev: np.ndarray
    V_terminal_rev: np.ndarray
    J_terminal_rev: np.ndarray
    J_shunt_rev: np.ndarray
    metrics_rev: SteadyJVMetrics
    hysteresis_index: float | None
    incident_power_W_m2: float | None
    source_certified: bool
    source_sha256: str
    result_sha256: str


@dataclass(frozen=True)
class ExternalACCircuitProtocol:
    """Area-normalized lumped external small-signal AC parasitic network.

    ``external_capacitance_F_m2`` has units of ``F/m^2`` (electrical areal
    capacitance across the junction port) and never aliases lumped thermal
    heat capacity ``J/(m^2*K)``.
    """

    series_resistance_ohm_m2: float = 0.0
    shunt_resistance_ohm_m2: float | None = None
    external_capacitance_F_m2: float = 0.0
    shunt_branch_reference: str = "junction"
    phasor_convention: str = "exp(+i omega t)"

    def __post_init__(self) -> None:
        if not _is_real_number(self.series_resistance_ohm_m2):
            raise ExternalCircuitError("series_resistance_ohm_m2 must be a real scalar")
        rs = float(self.series_resistance_ohm_m2)
        if not math.isfinite(rs) or rs < 0.0:
            raise ExternalCircuitError(
                f"series_resistance_ohm_m2 must be finite and >= 0, got {self.series_resistance_ohm_m2!r}"
            )
        object.__setattr__(self, "series_resistance_ohm_m2", rs)

        if self.shunt_resistance_ohm_m2 is not None:
            if not _is_real_number(self.shunt_resistance_ohm_m2):
                raise ExternalCircuitError(
                    "shunt_resistance_ohm_m2 must be None or a positive real scalar"
                )
            rsh = float(self.shunt_resistance_ohm_m2)
            if not math.isfinite(rsh) or rsh <= 0.0:
                raise ExternalCircuitError(
                    f"shunt_resistance_ohm_m2 must be finite and > 0 when set, got {self.shunt_resistance_ohm_m2!r}"
                )
            object.__setattr__(self, "shunt_resistance_ohm_m2", rsh)

        if not _is_real_number(self.external_capacitance_F_m2):
            raise ExternalCircuitError("external_capacitance_F_m2 must be a real scalar")
        cext = float(self.external_capacitance_F_m2)
        if not math.isfinite(cext) or cext < 0.0:
            raise ExternalCircuitError(
                f"external_capacitance_F_m2 must be finite and >= 0, got {self.external_capacitance_F_m2!r}"
            )
        object.__setattr__(self, "external_capacitance_F_m2", cext)

        if self.shunt_branch_reference != "junction":
            raise ExternalCircuitError(
                f"Unsupported shunt_branch_reference {self.shunt_branch_reference!r}; only 'junction' is supported"
            )
        if self.phasor_convention != "exp(+i omega t)":
            raise ExternalCircuitError(
                f"Unsupported phasor_convention {self.phasor_convention!r}; only 'exp(+i omega t)' is supported"
            )

    @property
    def parameter_units(self) -> dict[str, str]:
        return dict(EXTERNAL_AC_UNIT_SCHEMA)


@dataclass(frozen=True)
class ExternalACImpedanceResult:
    """Small-signal AC admittance/impedance after external lumped composition."""

    protocol: ExternalACCircuitProtocol
    frequency_Hz: np.ndarray
    angular_frequency_rad_s: np.ndarray
    device_admittance_S_m2: np.ndarray
    shunt_admittance_S_m2: np.ndarray
    junction_admittance_S_m2: np.ndarray
    terminal_admittance_S_m2: np.ndarray
    terminal_impedance_ohm_m2: np.ndarray | None
    source_identity: str | None
    source_sha256: str
    result_sha256: str

    @property
    def admittance_density_S_m2(self) -> np.ndarray:
        return self.terminal_admittance_S_m2

    @property
    def impedance_area_ohm_m2(self) -> np.ndarray:
        if self.terminal_impedance_ohm_m2 is not None:
            return self.terminal_impedance_ohm_m2
        if np.any(self.terminal_admittance_S_m2 == 0.0):
            raise ExternalCircuitTopologyError(
                "zero terminal admittance has no finite impedance-area product"
            )
        inv = 1.0 / self.terminal_admittance_S_m2
        if not np.all(np.isfinite(inv)):
            raise ExternalCircuitTopologyError(
                "terminal impedance-area product overflows binary64"
            )
        inv.setflags(write=False)
        return inv


def _require_strict_monotonic(v: np.ndarray, *, label: str) -> int:
    dv = np.diff(v)
    if np.all(dv > 0.0):
        return 1
    if np.all(dv < 0.0):
        return -1
    raise ExternalCircuitTopologyError(
        f"{label} voltage coordinates are not strictly monotonic"
    )


def map_external_circuit_branch(
    V_junction: object,
    J_device: object,
    protocol: ExternalCircuitProtocol,
    *,
    direction: str = "steady",
    incident_power_W_m2: float | None = None,
) -> ExternalCircuitBranch:
    """Map one strictly monotonic device J-V branch to external terminals."""
    if not isinstance(protocol, ExternalCircuitProtocol):
        raise ExternalCircuitError("protocol must be an ExternalCircuitProtocol")
    if direction not in {"forward", "reverse", "steady"}:
        raise ExternalCircuitError(
            f"direction must be 'forward', 'reverse', or 'steady', got {direction!r}"
        )
    if incident_power_W_m2 is not None and (
        not _is_real_number(incident_power_W_m2)
        or not math.isfinite(float(incident_power_W_m2))
        or float(incident_power_W_m2) <= 0.0
    ):
        raise ExternalCircuitError("incident_power_W_m2 must be None or finite positive")

    v_junc = _freeze_float_1d(V_junction, name=f"V_junction ({direction})", min_size=2)
    j_dev = _freeze_float_1d(J_device, name=f"J_device ({direction})", min_size=2)
    if v_junc.shape != j_dev.shape:
        raise ExternalCircuitError(
            f"V_junction and J_device ({direction}) must have identical shapes, "
            f"got {v_junc.shape} and {j_dev.shape}"
        )
    junc_orient = _require_strict_monotonic(v_junc, label=f"{direction} junction")
    if direction == "forward" and junc_orient != 1:
        raise ExternalCircuitError(
            "forward branch junction voltages must be strictly increasing"
        )
    if direction == "reverse" and junc_orient != -1:
        raise ExternalCircuitError(
            "reverse branch junction voltages must be strictly decreasing"
        )

    if protocol.shunt_resistance_ohm_m2 is None:
        j_shunt = np.zeros_like(v_junc)
    else:
        j_shunt = v_junc / protocol.shunt_resistance_ohm_m2

    j_term = j_dev - j_shunt
    v_drop = j_term * protocol.series_resistance_ohm_m2
    v_term = v_junc - v_drop
    if not np.all(np.isfinite(v_term)) or not np.all(np.isfinite(j_term)):
        raise ExternalCircuitError(
            f"nonfinite terminal J-V samples produced on {direction} branch"
        )

    term_orient = _require_strict_monotonic(
        v_term,
        label=(
            f"{direction} terminal (series_resistance_ohm_m2="
            f"{protocol.series_resistance_ohm_m2:g} folded the sweep)"
        ),
    )
    if term_orient != junc_orient:
        raise ExternalCircuitTopologyError(
            f"{direction} terminal voltage orientation reversed relative to junction sweep"
        )

    p_term = v_term * j_term
    p_series = (j_term * j_term) * protocol.series_resistance_ohm_m2
    p_shunt = v_junc * j_shunt

    for arr in (j_shunt, j_term, v_drop, v_term, p_term, p_series, p_shunt):
        arr.setflags(write=False)

    metrics = steady_jv_metrics(
        v_term,
        j_term,
        incident_power_W_m2=(
            None if incident_power_W_m2 is None else float(incident_power_W_m2)
        ),
    )
    return ExternalCircuitBranch(
        direction=direction,
        V_junction=v_junc,
        J_device=j_dev,
        V_terminal=v_term,
        J_terminal=j_term,
        J_shunt=j_shunt,
        series_voltage_drop_V=v_drop,
        power_terminal_W_m2=p_term,
        series_loss_W_m2=p_series,
        shunt_loss_W_m2=p_shunt,
        metrics=metrics,
    )


def apply_external_circuit_steady(
    source: SteadyJVResult | tuple[object, object],
    protocol: ExternalCircuitProtocol,
    *,
    incident_power_W_m2: float | None = None,
) -> ExternalCircuitSteadyResult:
    """Apply an external DC circuit to a certified ``SteadyJVResult`` or ``(V, J)`` pair."""
    if isinstance(source, SteadyJVResult):
        if len(source.solutions) != source.voltage_V.size or any(
            not pt.converged for pt in source.solutions
        ):
            raise ExternalCircuitError(
                "SteadyJVResult contains uncertified steady-state points"
            )
        v_in: object = source.voltage_V
        j_in: object = source.output_current_A_m2
        p_in = source.incident_power_W_m2 if incident_power_W_m2 is None else incident_power_W_m2
        source_identity: str | None = source.compiled_identity
    elif isinstance(source, tuple) and len(source) == 2:
        v_in, j_in = source
        p_in = incident_power_W_m2
        source_identity = None
    else:
        raise ExternalCircuitError(
            "source must be a SteadyJVResult or a (V_junction, J_device) tuple"
        )

    branch = map_external_circuit_branch(
        v_in, j_in, protocol, direction="steady", incident_power_W_m2=p_in
    )
    source_sha = _canonical_sha256(
        {
            "V_junction": _array_hex_list(branch.V_junction),
            "J_device": _array_hex_list(branch.J_device),
            "source_identity": source_identity,
        }
    )
    result_sha = _canonical_sha256(
        {
            "source_sha256": source_sha,
            "protocol": asdict(protocol),
            "V_terminal": _array_hex_list(branch.V_terminal),
            "J_terminal": _array_hex_list(branch.J_terminal),
            "metrics": asdict(branch.metrics),
        }
    )
    return ExternalCircuitSteadyResult(
        protocol=protocol,
        branch=branch,
        incident_power_W_m2=None if p_in is None else float(p_in),
        source_identity=source_identity,
        source_sha256=source_sha,
        result_sha256=result_sha,
    )


def _hysteresis_index(
    metrics_fwd: SteadyJVMetrics,
    metrics_rev: SteadyJVMetrics,
) -> float | None:
    p_fwd = metrics_fwd.maximum_power_W_m2
    p_rev = metrics_rev.maximum_power_W_m2
    if p_fwd is None or p_rev is None or not math.isfinite(p_rev) or p_rev <= 0.0:
        return None
    return float((p_rev - p_fwd) / p_rev)


def apply_external_circuit(
    jv: Any,
    protocol: ExternalCircuitProtocol,
    *,
    incident_power_W_m2: float | None = None,
) -> ExternalCircuitJVResult:
    """Map a dual-branch (forward + reverse) J-V sweep to external terminals.

    Accepts either a dual-branch sweep object exposing ``(V_fwd, J_fwd, V_rev, J_rev)``
    (refusing ``certified is False`` or ``completed is False``) or a 4-tuple
    ``(V_fwd, J_fwd, V_rev, J_rev)``.
    """
    if isinstance(jv, tuple) and len(jv) == 4:
        v_fwd_raw, j_fwd_raw, v_rev_raw, j_rev_raw = jv
        source_certified = True
    elif all(hasattr(jv, attr) for attr in ("V_fwd", "J_fwd", "V_rev", "J_rev")):
        if getattr(jv, "certified", True) is False or getattr(jv, "completed", True) is False:
            raise ExternalCircuitError(
                "apply_external_circuit refuses an uncertified or incomplete J-V sweep"
            )
        v_fwd_raw = jv.V_fwd
        j_fwd_raw = jv.J_fwd
        v_rev_raw = jv.V_rev
        j_rev_raw = jv.J_rev
        source_certified = bool(getattr(jv, "certified", True))
    else:
        raise ExternalCircuitError(
            "jv must provide (V_fwd, J_fwd, V_rev, J_rev) attributes or be a 4-tuple"
        )

    fwd = map_external_circuit_branch(
        v_fwd_raw,
        j_fwd_raw,
        protocol,
        direction="forward",
        incident_power_W_m2=incident_power_W_m2,
    )
    rev = map_external_circuit_branch(
        v_rev_raw,
        j_rev_raw,
        protocol,
        direction="reverse",
        incident_power_W_m2=incident_power_W_m2,
    )
    hi = _hysteresis_index(fwd.metrics, rev.metrics)

    source_sha = _canonical_sha256(
        {
            "V_junction_fwd": _array_hex_list(fwd.V_junction),
            "J_device_fwd": _array_hex_list(fwd.J_device),
            "V_junction_rev": _array_hex_list(rev.V_junction),
            "J_device_rev": _array_hex_list(rev.J_device),
            "source_certified": source_certified,
        }
    )
    result_sha = _canonical_sha256(
        {
            "source_sha256": source_sha,
            "protocol": asdict(protocol),
            "V_terminal_fwd": _array_hex_list(fwd.V_terminal),
            "J_terminal_fwd": _array_hex_list(fwd.J_terminal),
            "V_terminal_rev": _array_hex_list(rev.V_terminal),
            "J_terminal_rev": _array_hex_list(rev.J_terminal),
            "metrics_fwd": asdict(fwd.metrics),
            "metrics_rev": asdict(rev.metrics),
            "hysteresis_index": hi,
        }
    )
    return ExternalCircuitJVResult(
        protocol=protocol,
        V_junction_fwd=fwd.V_junction,
        J_device_fwd=fwd.J_device,
        V_terminal_fwd=fwd.V_terminal,
        J_terminal_fwd=fwd.J_terminal,
        J_shunt_fwd=fwd.J_shunt,
        metrics_fwd=fwd.metrics,
        V_junction_rev=rev.V_junction,
        J_device_rev=rev.J_device,
        V_terminal_rev=rev.V_terminal,
        J_terminal_rev=rev.J_terminal,
        J_shunt_rev=rev.J_shunt,
        metrics_rev=rev.metrics,
        hysteresis_index=hi,
        incident_power_W_m2=(
            None if incident_power_W_m2 is None else float(incident_power_W_m2)
        ),
        source_certified=source_certified,
        source_sha256=source_sha,
        result_sha256=result_sha,
    )


def compose_external_ac_impedance(
    source: SmallSignalResult1D | IonicImpedanceResult | tuple[object, object],
    protocol: ExternalACCircuitProtocol,
) -> ExternalACImpedanceResult:
    """Compose small-signal device admittance with external ``(R_s, R_sh, C_ext)``.

    Under the ``exp(+i * omega * t)`` phasor convention:
    ``Y_shunt(omega) = G_shunt + i * omega * C_ext``,
    ``Y_junc(omega) = Y_device(omega) + Y_shunt(omega)``,
    ``Y_terminal(omega) = Y_junc(omega) / (1 + R_series * Y_junc(omega))``.

    Neither ``source`` nor any compiled semiconductor device is mutated.
    """
    if not isinstance(protocol, ExternalACCircuitProtocol):
        raise ExternalCircuitError("protocol must be an ExternalACCircuitProtocol")

    if isinstance(source, SmallSignalResult1D):
        freq = _freeze_float_1d(source.frequency_Hz, name="frequency_Hz", min_size=1)
        y_dev = _freeze_complex_1d(
            source.admittance_density_S_m2, name="admittance_density_S_m2", min_size=1
        )
        source_identity: str | None = source.compiled_identity
    elif isinstance(source, IonicImpedanceResult):
        if not source.completed:
            raise ExternalCircuitError(
                "compose_external_ac_impedance refuses an incomplete IonicImpedanceResult"
            )
        freq = _freeze_float_1d(
            source.requested_frequencies_Hz, name="requested_frequencies_Hz", min_size=1
        )
        admittances: list[complex] = []
        for attempt in source.attempts:
            if not attempt.accepted or attempt.observation is None:
                raise ExternalCircuitError(
                    "IonicImpedanceResult contains an unaccepted frequency attempt"
                )
            raw_y = attempt.observation.get("admittance_density_S_m2")
            if raw_y is None:
                raise ExternalCircuitError(
                    "IonicImpedanceAttempt observation omits admittance_density_S_m2"
                )
            admittances.append(complex(raw_y))
        y_dev = _freeze_complex_1d(
            admittances, name="admittance_density_S_m2", min_size=1
        )
        source_identity = source.compiled_identity
    elif isinstance(source, tuple) and len(source) == 2:
        freq = _freeze_float_1d(source[0], name="frequency_Hz", min_size=1)
        y_dev = _freeze_complex_1d(
            source[1], name="admittance_density_S_m2", min_size=1
        )
        source_identity = None
    else:
        raise ExternalCircuitError(
            "source must be SmallSignalResult1D, IonicImpedanceResult, or (frequency_Hz, admittance_S_m2)"
        )

    if freq.shape != y_dev.shape:
        raise ExternalCircuitError(
            f"frequency_Hz and admittance_density_S_m2 shapes disagree: {freq.shape} vs {y_dev.shape}"
        )
    if np.any(freq < 0.0):
        raise ExternalCircuitError("frequency_Hz must be nonnegative")

    omega = 2.0 * np.pi * freq
    omega.setflags(write=False)

    g_shunt = (
        0.0
        if protocol.shunt_resistance_ohm_m2 is None
        else 1.0 / protocol.shunt_resistance_ohm_m2
    )
    y_shunt = g_shunt + 1j * omega * protocol.external_capacitance_F_m2
    y_junc = y_dev + y_shunt
    denom = 1.0 + protocol.series_resistance_ohm_m2 * y_junc
    if np.any(denom == 0.0) or not np.all(np.isfinite(denom)):
        raise ExternalCircuitTopologyError(
            "1 + R_series * Y_junction vanished or overflowed during AC circuit composition"
        )
    y_term = y_junc / denom
    if not np.all(np.isfinite(y_term)):
        raise ExternalCircuitError("nonfinite terminal admittance in AC composition")

    z_term: np.ndarray | None
    if np.all(y_junc != 0.0):
        candidate_z = protocol.series_resistance_ohm_m2 + 1.0 / y_junc
        if np.all(np.isfinite(candidate_z)):
            candidate_z.setflags(write=False)
            z_term = candidate_z
        else:
            z_term = None
    else:
        z_term = None

    for arr in (y_shunt, y_junc, y_term):
        arr.setflags(write=False)

    source_sha = _canonical_sha256(
        {
            "frequency_Hz": _array_hex_list(freq),
            "device_admittance_S_m2": _complex_hex_list(y_dev),
            "source_identity": source_identity,
        }
    )
    result_sha = _canonical_sha256(
        {
            "source_sha256": source_sha,
            "protocol": asdict(protocol),
            "terminal_admittance_S_m2": _complex_hex_list(y_term),
        }
    )
    return ExternalACImpedanceResult(
        protocol=protocol,
        frequency_Hz=freq,
        angular_frequency_rad_s=omega,
        device_admittance_S_m2=y_dev,
        shunt_admittance_S_m2=y_shunt,
        junction_admittance_S_m2=y_junc,
        terminal_admittance_S_m2=y_term,
        terminal_impedance_ohm_m2=z_term,
        source_identity=source_identity,
        source_sha256=source_sha,
        result_sha256=result_sha,
    )
