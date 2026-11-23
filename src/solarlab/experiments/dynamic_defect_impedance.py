"""Dynamic explicit-defect small-signal impedance on unified 1D assembly.

Migrates bulk dynamic-defect impedance, interface dynamic-defect impedance, and
the protocol-bound dynamic-defect capability classifier onto
``CompiledDynamicDefectDevice1D`` + ``DynamicDefectSmallSignal1D``, computing
low-frequency QSS (`0 Hz`) and high-frequency frozen-trap reference limits from
the exact same nonlinear DC working state.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts import frozen_array
from solarlab.device.compile_dynamic_defect_1d import compile_dynamic_defect_device_1d
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.assembly.dynamic_defects_1d import (
    DynamicDefectACSweepResult,
    DynamicDefectSmallSignal1D,
    DynamicDefectSteadyOptions,
    DynamicDefectSteadyResult,
    solve_dynamic_defect_steady_1d,
)
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.dynamic_defect_1d import CompiledDynamicDefectDevice1D
from solarlab.physics.defects import (
    DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    evaluate_dynamic_bulk_traps,
    validate_defect_energy_quadrature_order,
)
from solarlab.physics.interface_transport import (
    EquilibriumReferencedSheetCharge,
    TwoSidedBulkState,
    solve_fixed_occupancy_two_sided_interface,
)
from solarlab.solvers.linear.frequency_domain import FrequencyOptions

Vector: TypeAlias = NDArray[np.float64]
ComplexVector: TypeAlias = NDArray[np.complex128]

DynamicDefectCapability = Literal[
    "bulk_dynamic_defect",
    "interface_dynamic_defect",
    "bulk_defect_plus_ions",
    "interface_defect_plus_ions",
    "bulk_interface_defect_plus_ions",
]

DYNAMIC_DEFECT_IMPEDANCE_SCHEMA = "solarlab.dynamic-defect-impedance.v1"

__all__ = [
    "DYNAMIC_DEFECT_IMPEDANCE_SCHEMA",
    "DynamicDefectCapability",
    "DynamicDefectImpedanceCapabilityError",
    "DynamicDefectImpedanceError",
    "DynamicDefectImpedanceOptions",
    "DynamicDefectImpedanceProtocol",
    "DynamicDefectImpedanceResult",
    "build_dynamic_defect_impedance_protocol",
    "classify_dynamic_defect_capability",
    "solve_bulk_defect_impedance",
    "solve_dynamic_defect_impedance",
    "solve_interface_defect_impedance",
]


class DynamicDefectImpedanceError(RuntimeError):
    """Raised when dynamic-defect impedance validation or execution fails."""


class DynamicDefectImpedanceCapabilityError(DynamicDefectImpedanceError, CapabilityError):
    """Raised when a device stack falls outside qualified dynamic-defect capabilities."""


def _relative_admittance_error(left: complex, right: complex) -> float:
    scale = max(abs(left), abs(right), 1.0e-30)
    return float(abs(left - right) / scale)


def _validate_positive_increasing_frequencies(frequencies_Hz: ArrayLike) -> Vector:
    freqs = np.asarray(frequencies_Hz, dtype=np.float64)
    if (
        freqs.ndim != 1
        or freqs.size < 3
        or not np.all(np.isfinite(freqs))
        or np.any(freqs <= 0.0)
        or np.any(np.diff(freqs) <= 0.0)
    ):
        raise ValueError(
            "frequencies_Hz must contain at least three finite, positive, strictly increasing values"
        )
    return freqs


def _layer_param_float(layer: object, key: str) -> float:
    params = dict(getattr(layer, "parameters", ()))
    val = params.get(key)
    return float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else 0.0


def _active_ion_layers(prepared: PreparedDevice) -> list[str]:
    layers: list[str] = []
    for layer in prepared.layers:
        if layer.role == "substrate":
            continue
        p0 = _layer_param_float(layer, "P0")
        d_pos = _layer_param_float(layer, "D_ion")
        p0_neg = _layer_param_float(layer, "P0_neg")
        d_neg = _layer_param_float(layer, "D_ion_neg")
        if (d_pos > 0.0 and p0 > 0.0) or (d_neg > 0.0 and p0_neg > 0.0):
            layers.append(layer.id)
    return layers


def classify_dynamic_defect_capability(
    prepared: PreparedDevice,
) -> DynamicDefectCapability:
    """Classify ``prepared`` into a supported dynamic-defect impedance capability."""
    if not isinstance(prepared, PreparedDevice):
        raise TypeError("classify_dynamic_defect_capability requires PreparedDevice")

    elec_layers = [ly for ly in prepared.layers if ly.role != "substrate"]
    if not elec_layers:
        raise DynamicDefectImpedanceCapabilityError(
            "dynamic-defect impedance requires at least one electrical layer"
        )

    has_bulk = False
    for layer in elec_layers:
        if layer.metastable_document is not None:
            raise DynamicDefectImpedanceCapabilityError(
                f"layers.{layer.id}: metastable bulk defects are not dynamic-trap impedance species"
            )
        if layer.defect_schema_version == "solarlab-explicit-bulk-defects-v4":
            raise DynamicDefectImpedanceCapabilityError(
                f"layers.{layer.id}: multivalent bulk defects do not expose dynamic single-level impedance modes"
            )
        if layer.bulk_trap_distribution_json is not None:
            raw_legacy = json.loads(layer.bulk_trap_distribution_json.decode())
            if str(raw_legacy.get("charge_transition", "")) == "neutral":
                raise DynamicDefectImpedanceCapabilityError(
                    f"layers.{layer.id}.bulk_trap_distribution: neutral bulk defects are refused for dynamic-defect impedance"
                )
            has_bulk = True
        for raw_bd in layer.defects:
            bd_map = raw_bd.to_mapping()
            if str(bd_map.get("charge_transition", "")) == "neutral":
                raise DynamicDefectImpedanceCapabilityError(
                    f"layers.{layer.id}.{bd_map['id']}: neutral bulk defects are refused for dynamic-defect impedance"
                )
            has_bulk = True

    has_interface = False
    for iface in prepared.interfaces:
        if not iface.electrical:
            continue
        entry = iface.to_mapping()
        defect = entry.get("defect")
        if defect is None:
            continue
        has_interface = True
        cal = float(defect["calibration_factor"])
        state_cal = float(defect["iface_state_calibration_factor"])
        if cal != 1.0 or state_cal != 1.0:
            raise DynamicDefectImpedanceCapabilityError(
                f"interfaces.{entry['id']}: empirical calibration factors are refused"
            )
        raw_vn = entry.get("v_n")
        raw_vp = entry.get("v_p")
        micro_pair = tuple(float(v) for v in entry["microscopic_capture_velocities_m_s"])
        if raw_vn is not None and raw_vp is not None and not (
            math.isclose(float(raw_vn), micro_pair[0], rel_tol=1.0e-12, abs_tol=0.0)
            and math.isclose(float(raw_vp), micro_pair[1], rel_tol=1.0e-12, abs_tol=0.0)
        ):
            raise DynamicDefectImpedanceCapabilityError(
                f"interfaces.{entry['id']}: compatibility SRV disagrees with microscopic capture"
            )

    if has_interface:
        settings = prepared.to_mapping()["settings"]
        if str(settings.get("interface_charge_closure", "off")) != "equilibrium_referenced":
            raise DynamicDefectImpedanceCapabilityError(
                "dynamic interface-defect impedance requires interface_charge_closure='equilibrium_referenced'"
            )
        if not bool(settings.get("interface_charge_rebaseline_acknowledged", False)):
            raise DynamicDefectImpedanceCapabilityError(
                "dynamic interface-defect impedance requires interface_charge_rebaseline_acknowledged=True"
            )

    ion_layers = _active_ion_layers(prepared)
    if len(ion_layers) > 1:
        raise DynamicDefectImpedanceCapabilityError(
            "dynamic-defect impedance currently supports at most one active mobile-ion layer"
        )
    has_ions = len(ion_layers) == 1

    if not has_bulk and not has_interface:
        raise DynamicDefectImpedanceCapabilityError(
            "dynamic-defect impedance requires an explicit bulk or microscopic interface defect population"
        )

    if has_ions:
        if has_bulk and has_interface:
            return "bulk_interface_defect_plus_ions"
        if has_bulk:
            return "bulk_defect_plus_ions"
        return "interface_defect_plus_ions"

    if has_bulk and has_interface:
        raise DynamicDefectImpedanceCapabilityError(
            "bulk + interface dynamic defects without active mobile ions do not have a jointly certified production operator"
        )
    return "bulk_dynamic_defect" if has_bulk else "interface_dynamic_defect"


@dataclass(frozen=True, slots=True)
class DynamicDefectImpedanceOptions:
    """Execution controls for dynamic-defect small-signal impedance."""

    intervals_per_layer: int | tuple[int, ...] = 24
    energy_quadrature_order: int = DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER
    voltage_phasor_V: complex = 1.0
    ion_domain_endpoints: tuple[str, str] = ("blocking", "blocking")
    verify_zero_hz_fd: bool = False
    fd_voltage_step_V: float = 1.0e-5
    steady_options: DynamicDefectSteadyOptions | None = None
    linear_options: FrequencyOptions | None = None

    def __post_init__(self) -> None:
        validate_defect_energy_quadrature_order(self.energy_quadrature_order)
        if isinstance(self.intervals_per_layer, int):
            if isinstance(self.intervals_per_layer, bool) or self.intervals_per_layer < 2:
                raise ValueError("intervals_per_layer must be >= 2")
        else:
            tup = tuple(int(v) for v in self.intervals_per_layer)
            if not tup or any(v < 2 for v in tup):
                raise ValueError("intervals_per_layer entries must be >= 2")
            object.__setattr__(self, "intervals_per_layer", tup)
        if not np.isfinite(self.voltage_phasor_V) or self.voltage_phasor_V == 0.0:
            raise ValueError("voltage_phasor_V must be finite and nonzero")
        if not math.isfinite(self.fd_voltage_step_V) or self.fd_voltage_step_V <= 0.0:
            raise ValueError("fd_voltage_step_V must be finite and positive")


@dataclass(frozen=True, slots=True)
class DynamicDefectImpedanceProtocol:
    """Immutable protocol digest for a dynamic-defect impedance calculation."""

    schema_version: str
    capability: DynamicDefectCapability
    voltage_V: float
    generation_scale: float
    frequencies_Hz: tuple[float, ...]
    intervals_per_layer: int | tuple[int, ...]
    actual_grid_nodes: int
    energy_quadrature_order: int
    prepared_sha256: str
    compiled_device_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "capability": self.capability,
            "voltage_V": self.voltage_V,
            "generation_scale": self.generation_scale,
            "frequencies_Hz": list(self.frequencies_Hz),
            "intervals_per_layer": (
                list(self.intervals_per_layer)
                if isinstance(self.intervals_per_layer, tuple)
                else self.intervals_per_layer
            ),
            "actual_grid_nodes": self.actual_grid_nodes,
            "energy_quadrature_order": self.energy_quadrature_order,
            "prepared_sha256": self.prepared_sha256,
            "compiled_device_sha256": self.compiled_device_sha256,
        }

    @property
    def protocol_sha256(self) -> str:
        text = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return sha256(text.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class DynamicDefectImpedanceResult:
    """Result of dynamic-defect small-signal impedance analysis."""

    capability: DynamicDefectCapability
    protocol: DynamicDefectImpedanceProtocol
    device: CompiledDynamicDefectDevice1D
    steady: DynamicDefectSteadyResult
    ac_sweep: DynamicDefectACSweepResult
    frozen_reference_sweep: DynamicDefectACSweepResult
    frequencies_Hz: Vector
    admittance_density_S_m2: ComplexVector
    impedance_area_ohm_m2: ComplexVector
    qss_reference_admittance_S_m2: complex
    frozen_high_frequency_admittance_S_m2: complex
    low_frequency_qss_relative_error: float
    high_frequency_frozen_relative_error: float
    trap_relaxation_frequencies_Hz: Vector
    frequency_window_brackets_traps: bool
    maximum_maxwell_spread_A_m2_V: float
    maximum_ion_inventory_relative_per_V: float
    dc_differential_conductance_S_m2: float | None = None
    zero_hz_fd_relative_error: float | None = None
    scientific_qualification: bool = False
    physical_certificate_status: str = "not_evaluated"

    def __post_init__(self) -> None:
        for attr in ("frequencies_Hz", "trap_relaxation_frequencies_Hz"):
            object.__setattr__(self, attr, frozen_array(getattr(self, attr)))
        for attr in ("admittance_density_S_m2", "impedance_area_ohm_m2"):
            c_arr = np.asarray(getattr(self, attr), dtype=np.complex128)
            object.__setattr__(
                self,
                attr,
                np.frombuffer(c_arr.tobytes(), dtype=np.complex128).reshape(c_arr.shape),
            )


def _trap_relaxation_frequencies_Hz(
    device: CompiledDynamicDefectDevice1D,
    steady: DynamicDefectSteadyResult,
) -> Vector:
    rates: list[float] = []
    elec = steady.state.electronic_state
    n = elec.electron_density_m3
    p = elec.hole_density_m3
    phi = elec.potential_V
    N = device.node_count
    zn = elec.u[N : 2 * N]
    zp = elec.u[2 * N : 3 * N]

    if device.dynamic_bulk_traps is not None:
        ev = evaluate_dynamic_bulk_traps(
            n, p, steady.state.bulk_trap_occupancy, device.dynamic_bulk_traps
        )
        # Relaxation rate k_relax = -d(C_n - C_p) / (N_t df) [s^-1]:
        k_bt = -ev.d_trap_storage_rate_df_m3_s / device.dynamic_bulk_traps.population_density_m3
        rates.extend(float(val) for val in k_bt if val > 0.0)

    for idx, itrap in enumerate(device.dynamic_interface_traps):
        ln, rn = itrap.left_node, itrap.right_node
        eq_charge = EquilibriumReferencedSheetCharge(
            equilibrium_occupancy=itrap.trap_spec.equilibrium_occupancy,
            trap_density_m2=itrap.trap_density_m2 if itrap.has_sheet_charge else 0.0,
        )
        res = solve_fixed_occupancy_two_sided_interface(
            itrap.trace_electrostatics,
            itrap.physics,
            TwoSidedBulkState(
                phi_left_V=float(phi[ln]),
                phi_right_V=float(phi[rn]),
                n_left_m3=float(n[ln]),
                p_left_m3=float(p[ln]),
                n_right_m3=float(n[rn]),
                p_right_m3=float(p[rn]),
                electron_electrochemical_step=float(zn[rn] - zn[ln]),
                hole_electrochemical_step=float(-(zp[rn] - zp[ln])),
            ),
            eq_charge,
            occupancy=float(steady.state.interface_trap_occupancy[idx]),
        )
        if res.relaxation_rate_s1 > 0.0:
            rates.append(float(res.relaxation_rate_s1))

    if not rates:
        return np.empty(0, dtype=np.float64)
    return np.asarray(sorted(r / (2.0 * math.pi) for r in rates), dtype=np.float64)


def build_dynamic_defect_impedance_protocol(
    prepared: PreparedDevice,
    frequencies_Hz: ArrayLike,
    *,
    voltage_V: float = 0.0,
    generation_scale: float = 0.0,
    options: DynamicDefectImpedanceOptions | None = None,
) -> tuple[DynamicDefectImpedanceProtocol, CompiledDynamicDefectDevice1D]:
    """Classify capability, compile unified device, and return bound protocol."""
    controls = DynamicDefectImpedanceOptions() if options is None else options
    freqs = _validate_positive_increasing_frequencies(frequencies_Hz)
    if not math.isfinite(voltage_V) or not math.isfinite(generation_scale) or generation_scale < 0.0:
        raise ValueError("voltage_V must be finite and generation_scale >= 0")

    capability = classify_dynamic_defect_capability(prepared)
    ion_layers = _active_ion_layers(prepared)
    ion_layer = ion_layers[0] if ion_layers else None

    try:
        compiled = compile_dynamic_defect_device_1d(
            prepared,
            intervals_per_layer=controls.intervals_per_layer,
            dynamic_bulk_defects=("bulk" in capability),
            dynamic_interface_defects=("interface" in capability),
            energy_quadrature_order=controls.energy_quadrature_order,
            ion_domain_layer=ion_layer,
            ion_domain_endpoints=controls.ion_domain_endpoints,
        )
    except CapabilityError as error:
        raise DynamicDefectImpedanceCapabilityError(str(error)) from error

    protocol = DynamicDefectImpedanceProtocol(
        schema_version=DYNAMIC_DEFECT_IMPEDANCE_SCHEMA,
        capability=capability,
        voltage_V=float(voltage_V),
        generation_scale=float(generation_scale),
        frequencies_Hz=tuple(float(f) for f in freqs),
        intervals_per_layer=controls.intervals_per_layer,
        actual_grid_nodes=compiled.node_count,
        energy_quadrature_order=controls.energy_quadrature_order,
        prepared_sha256=prepared.content_sha256,
        compiled_device_sha256=compiled.identity,
    )
    return protocol, compiled


def solve_dynamic_defect_impedance(
    prepared: PreparedDevice,
    frequencies_Hz: ArrayLike,
    *,
    voltage_V: float = 0.0,
    generation_scale: float = 0.0,
    options: DynamicDefectImpedanceOptions | None = None,
) -> DynamicDefectImpedanceResult:
    """Solve dynamic-defect small-signal impedance across ``frequencies_Hz``."""
    controls = DynamicDefectImpedanceOptions() if options is None else options
    freqs = _validate_positive_increasing_frequencies(frequencies_Hz)
    protocol, compiled = build_dynamic_defect_impedance_protocol(
        prepared,
        freqs,
        voltage_V=voltage_V,
        generation_scale=generation_scale,
        options=controls,
    )

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

    ac_sweep = ss.solve_sweep(
        freqs,
        voltage_phasor_V=controls.voltage_phasor_V,
        options=controls.linear_options,
    )
    qss_zero = ss.solve_sweep(
        np.array([0.0]),
        voltage_phasor_V=controls.voltage_phasor_V,
        options=controls.linear_options,
    )
    frozen_sweep = ss.solve_sweep(
        freqs,
        voltage_phasor_V=controls.voltage_phasor_V,
        options=controls.linear_options,
        freeze_ions=compiled.has_ions,
        freeze_bulk_traps=compiled.has_dynamic_bulk_traps,
        freeze_interface_traps=compiled.has_dynamic_interface_traps,
    )

    y_qss = complex(qss_zero.admittance_density_S_m2[0])
    y_hf_frozen = complex(frozen_sweep.admittance_density_S_m2[-1])
    low_rel = _relative_admittance_error(complex(ac_sweep.admittance_density_S_m2[0]), y_qss)
    high_rel = _relative_admittance_error(
        complex(ac_sweep.admittance_density_S_m2[-1]), y_hf_frozen
    )

    f_traps = _trap_relaxation_frequencies_Hz(compiled, steady)
    brackets = bool(
        f_traps.size > 0
        and float(freqs[0]) <= float(f_traps[0])
        and float(freqs[-1]) >= float(f_traps[-1])
    )

    max_inv = 0.0
    if compiled.has_ions:
        for pt in (*qss_zero.points, *ac_sweep.points):
            if pt.ion_inventory_relative_per_V:
                max_inv = max(max_inv, max(pt.ion_inventory_relative_per_V.values()))

    dc_g: float | None = None
    fd_rel: float | None = None
    if controls.verify_zero_hz_fd:
        dv = controls.fd_voltage_step_V
        u_plus = compiled.rebase_coordinates(steady.state.u, voltage_V, voltage_V + dv)
        u_minus = compiled.rebase_coordinates(steady.state.u, voltage_V, voltage_V - dv)
        st_plus = solve_dynamic_defect_steady_1d(
            compiled,
            voltage_V=voltage_V + dv,
            generation_scale=generation_scale,
            initial_u=u_plus,
            options=controls.steady_options,
        )
        st_minus = solve_dynamic_defect_steady_1d(
            compiled,
            voltage_V=voltage_V - dv,
            generation_scale=generation_scale,
            initial_u=u_minus,
            options=controls.steady_options,
        )
        dc_g = float(
            (st_plus.terminal_current_density_A_m2 - st_minus.terminal_current_density_A_m2)
            / (2.0 * dv)
        )
        fd_rel = _relative_admittance_error(y_qss, complex(dc_g, 0.0))

    return DynamicDefectImpedanceResult(
        capability=protocol.capability,
        protocol=protocol,
        device=compiled,
        steady=steady,
        ac_sweep=ac_sweep,
        frozen_reference_sweep=frozen_sweep,
        frequencies_Hz=ac_sweep.frequencies_Hz,
        admittance_density_S_m2=ac_sweep.admittance_density_S_m2,
        impedance_area_ohm_m2=ac_sweep.impedance_area_ohm_m2,
        qss_reference_admittance_S_m2=y_qss,
        frozen_high_frequency_admittance_S_m2=y_hf_frozen,
        low_frequency_qss_relative_error=low_rel,
        high_frequency_frozen_relative_error=high_rel,
        trap_relaxation_frequencies_Hz=f_traps,
        frequency_window_brackets_traps=brackets,
        maximum_maxwell_spread_A_m2_V=float(np.max(ac_sweep.maxwell_spread_A_m2_V)),
        maximum_ion_inventory_relative_per_V=max_inv,
        dc_differential_conductance_S_m2=dc_g,
        zero_hz_fd_relative_error=fd_rel,
    )


def solve_bulk_defect_impedance(
    prepared: PreparedDevice,
    frequencies_Hz: ArrayLike,
    *,
    voltage_V: float = 0.0,
    generation_scale: float = 0.0,
    options: DynamicDefectImpedanceOptions | None = None,
) -> DynamicDefectImpedanceResult:
    """Solve ion-free bulk dynamic-defect impedance (requires ``bulk_dynamic_defect``)."""
    cap = classify_dynamic_defect_capability(prepared)
    if cap != "bulk_dynamic_defect":
        raise DynamicDefectImpedanceCapabilityError(
            f"solve_bulk_defect_impedance requires 'bulk_dynamic_defect', got {cap!r}"
        )
    return solve_dynamic_defect_impedance(
        prepared,
        frequencies_Hz,
        voltage_V=voltage_V,
        generation_scale=generation_scale,
        options=options,
    )


def solve_interface_defect_impedance(
    prepared: PreparedDevice,
    frequencies_Hz: ArrayLike,
    *,
    voltage_V: float = 0.0,
    generation_scale: float = 0.0,
    options: DynamicDefectImpedanceOptions | None = None,
) -> DynamicDefectImpedanceResult:
    """Solve ion-free two-sided interface dynamic-defect impedance."""
    cap = classify_dynamic_defect_capability(prepared)
    if cap != "interface_dynamic_defect":
        raise DynamicDefectImpedanceCapabilityError(
            f"solve_interface_defect_impedance requires 'interface_dynamic_defect', got {cap!r}"
        )
    return solve_dynamic_defect_impedance(
        prepared,
        frequencies_Hz,
        voltage_V=voltage_V,
        generation_scale=generation_scale,
        options=options,
    )
