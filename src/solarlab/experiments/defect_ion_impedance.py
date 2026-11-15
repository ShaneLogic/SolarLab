"""Joint dynamic-defect and mobile-ion small-signal impedance.

Linearizes coupled electronic, mobile-ion, bulk-trap, and two-sided
interface-trap balances around a single certified nonlinear DC working state.
Deliberately evaluates four diagnostic species-freezing modes (``coupled``,
``frozen_ions``, ``frozen_traps``, and ``frozen_ions_and_traps``) from the
exact same ``working_state_digest`` rather than summing independent spectra.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, TypeAlias

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
from solarlab.experiments.dynamic_defect_impedance import classify_dynamic_defect_capability
from solarlab.physics.defects import (
    DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    validate_defect_energy_quadrature_order,
)
from solarlab.solvers.linear.frequency_domain import FrequencyOptions

Vector: TypeAlias = NDArray[np.float64]
ComplexVector: TypeAlias = NDArray[np.complex128]

DefectIonCapability = Literal[
    "bulk_defect_plus_ions",
    "interface_defect_plus_ions",
    "bulk_interface_defect_plus_ions",
]

__all__ = [
    "DefectIonCapability",
    "DefectIonImpedanceCapabilityError",
    "DefectIonImpedanceError",
    "DefectIonImpedanceOptions",
    "DefectIonImpedanceResult",
    "solve_defect_ion_impedance",
]


class DefectIonImpedanceError(RuntimeError):
    """Raised when combined defect/ion impedance validation or execution fails."""


class DefectIonImpedanceCapabilityError(DefectIonImpedanceError, CapabilityError):
    """Raised when a device stack lacks joint dynamic-defect + mobile-ion support."""


@dataclass(frozen=True, slots=True)
class DefectIonImpedanceOptions:
    """Execution controls for coupled dynamic-defect + mobile-ion impedance."""

    intervals_per_layer: int | tuple[int, ...] = 24
    energy_quadrature_order: int = DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER
    voltage_phasor_V: complex = 1.0
    ion_domain_layer: str | None = None
    ion_domain_endpoints: tuple[str, str] = ("blocking", "blocking")
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


@dataclass(frozen=True)
class DefectIonImpedanceResult:
    """Coupled and species-frozen small-signal spectra about one DC working state."""

    capability: DefectIonCapability
    device: CompiledDynamicDefectDevice1D
    steady: DynamicDefectSteadyResult
    working_state_digest: str
    frequencies_Hz: Vector
    coupled: DynamicDefectACSweepResult
    frozen_ions: DynamicDefectACSweepResult
    frozen_traps: DynamicDefectACSweepResult
    frozen_ions_and_traps: DynamicDefectACSweepResult
    qss_zero_hz_admittance_S_m2: complex
    low_frequency_qss_relative_error: float
    high_frequency_frozen_relative_error: float
    maximum_maxwell_spread_A_m2_V: float
    maximum_ion_inventory_relative_per_V: float
    scientific_qualification: bool = False
    physical_certificate_status: str = "not_evaluated"

    def __post_init__(self) -> None:
        object.__setattr__(self, "frequencies_Hz", frozen_array(self.frequencies_Hz))


def _layer_param_float(layer: object, key: str) -> float:
    params = dict(getattr(layer, "parameters", ()))
    val = params.get(key)
    return float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else 0.0


def _infer_single_ion_layer(prepared: PreparedDevice) -> str:
    layers = [
        ly.id
        for ly in prepared.layers
        if ly.role != "substrate"
        and (
            (_layer_param_float(ly, "D_ion") > 0.0 and _layer_param_float(ly, "P0") > 0.0)
            or (
                _layer_param_float(ly, "D_ion_neg") > 0.0
                and _layer_param_float(ly, "P0_neg") > 0.0
            )
        )
    ]
    if len(layers) != 1:
        raise DefectIonImpedanceCapabilityError(
            "combined defect/ion impedance requires exactly one active mobile-ion layer"
        )
    return layers[0]


def solve_defect_ion_impedance(
    prepared: PreparedDevice,
    frequencies_Hz: ArrayLike,
    *,
    voltage_V: float = 0.0,
    generation_scale: float = 0.0,
    options: DefectIonImpedanceOptions | None = None,
) -> DefectIonImpedanceResult:
    """Solve coupled and frozen-species impedance sweeps about one DC working state."""
    controls = DefectIonImpedanceOptions() if options is None else options
    freqs = np.asarray(frequencies_Hz, dtype=np.float64)
    if (
        freqs.ndim != 1
        or freqs.size < 1
        or not np.all(np.isfinite(freqs))
        or np.any(freqs < 0.0)
    ):
        raise ValueError("frequencies_Hz must be a non-empty 1D array of finite non-negative values")
    if freqs.size > 1 and np.any(np.diff(freqs) <= 0.0):
        raise ValueError("frequencies_Hz must be strictly increasing")
    if not math.isfinite(voltage_V) or not math.isfinite(generation_scale) or generation_scale < 0.0:
        raise ValueError("voltage_V must be finite and generation_scale >= 0")

    try:
        raw_cap = classify_dynamic_defect_capability(prepared)
    except CapabilityError as error:
        raise DefectIonImpedanceCapabilityError(str(error)) from error

    if raw_cap not in {
        "bulk_defect_plus_ions",
        "interface_defect_plus_ions",
        "bulk_interface_defect_plus_ions",
    }:
        raise DefectIonImpedanceCapabilityError(
            f"combined defect/ion impedance requires active mobile ions plus dynamic defects, got {raw_cap!r}"
        )
    capability: DefectIonCapability = raw_cap  # type: ignore[assignment]

    ion_layer = (
        controls.ion_domain_layer
        if controls.ion_domain_layer is not None
        else _infer_single_ion_layer(prepared)
    )
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
        raise DefectIonImpedanceCapabilityError(str(error)) from error

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

    coupled = ss.solve_sweep(
        freqs,
        voltage_phasor_V=controls.voltage_phasor_V,
        options=controls.linear_options,
        freeze_ions=False,
        freeze_bulk_traps=False,
        freeze_interface_traps=False,
    )
    frozen_ions = ss.solve_sweep(
        freqs,
        voltage_phasor_V=controls.voltage_phasor_V,
        options=controls.linear_options,
        freeze_ions=True,
        freeze_bulk_traps=False,
        freeze_interface_traps=False,
    )
    frozen_traps = ss.solve_sweep(
        freqs,
        voltage_phasor_V=controls.voltage_phasor_V,
        options=controls.linear_options,
        freeze_ions=False,
        freeze_bulk_traps=True,
        freeze_interface_traps=True,
    )
    frozen_both = ss.solve_sweep(
        freqs,
        voltage_phasor_V=controls.voltage_phasor_V,
        options=controls.linear_options,
        freeze_ions=True,
        freeze_bulk_traps=True,
        freeze_interface_traps=True,
    )
    zero_hz = ss.solve_sweep(
        np.array([0.0]),
        voltage_phasor_V=controls.voltage_phasor_V,
        options=controls.linear_options,
    )

    digest = ss.working_state_digest
    if not (
        coupled.working_state_digest
        == frozen_ions.working_state_digest
        == frozen_traps.working_state_digest
        == frozen_both.working_state_digest
        == digest
    ):
        raise DefectIonImpedanceError("working_state_digest drifted across frozen-species modes")

    y0 = complex(zero_hz.admittance_density_S_m2[0])
    y_hf = complex(frozen_both.admittance_density_S_m2[-1])
    low_err = float(
        abs(complex(coupled.admittance_density_S_m2[0]) - y0)
        / max(abs(complex(coupled.admittance_density_S_m2[0])), abs(y0), 1.0e-30)
    )
    high_err = float(
        abs(complex(coupled.admittance_density_S_m2[-1]) - y_hf)
        / max(abs(complex(coupled.admittance_density_S_m2[-1])), abs(y_hf), 1.0e-30)
    )

    max_spread = float(
        max(
            np.max(coupled.maxwell_spread_A_m2_V),
            np.max(frozen_ions.maxwell_spread_A_m2_V),
            np.max(frozen_traps.maxwell_spread_A_m2_V),
            np.max(frozen_both.maxwell_spread_A_m2_V),
        )
    )
    max_inv = 0.0
    for pt in (*zero_hz.points, *coupled.points, *frozen_traps.points):
        if pt.ion_inventory_relative_per_V:
            max_inv = max(max_inv, max(pt.ion_inventory_relative_per_V.values()))

    return DefectIonImpedanceResult(
        capability=capability,
        device=compiled,
        steady=steady,
        working_state_digest=digest,
        frequencies_Hz=coupled.frequencies_Hz,
        coupled=coupled,
        frozen_ions=frozen_ions,
        frozen_traps=frozen_traps,
        frozen_ions_and_traps=frozen_both,
        qss_zero_hz_admittance_S_m2=y0,
        low_frequency_qss_relative_error=low_err,
        high_frequency_frozen_relative_error=high_err,
        maximum_maxwell_spread_A_m2_V=max_spread,
        maximum_ion_inventory_relative_per_V=max_inv,
    )
