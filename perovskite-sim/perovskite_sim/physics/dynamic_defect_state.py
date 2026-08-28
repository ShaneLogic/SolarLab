"""Dynamic occupancy states for compiled monovalent bulk defects.

The quasi-steady defect closure eliminates occupancy locally.  Frequency and
time-domain device equations instead need one explicit occupancy for every
active spatial node and energy-quadrature node.  This module compiles that
state layout from the same immutable material model and evaluates carrier
capture, trap storage, and trap charge without clipping.

No device solver is selected here.  Device adapters own Poisson, continuity,
terminal-current, and capability certification.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.models.defects import ACCEPTOR, DONOR, NEUTRAL
from perovskite_sim.physics.defect_closure import MonovalentBulkDefectModel
from perovskite_sim.physics.temperature import thermal_voltage


DYNAMIC_BULK_TRAP_LAYOUT_VERSION = "monovalent-bulk-dynamic-layout-v1"
DYNAMIC_BULK_TRAP_EVALUATION_VERSION = "monovalent-bulk-dynamic-evaluation-v1"
SUPPORTED_TRANSITIONS = frozenset({NEUTRAL, ACCEPTOR, DONOR})


class DynamicBulkTrapStateError(ValueError):
    """A dynamic bulk-trap layout or state is incomplete or non-physical."""


def _readonly(value: object, *, dtype: object) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _canonical_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _stable_expit(value: np.ndarray) -> np.ndarray:
    result = np.empty_like(value)
    positive = value >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


@dataclass(frozen=True, slots=True)
class DynamicBulkTrapLayout:
    """Canonical state ordering for a compiled bulk-defect model."""

    model_identity_sha256: str
    node_count: int
    device_node_indices: np.ndarray
    region_indices: np.ndarray
    source_indices: np.ndarray
    energy_indices: np.ndarray
    region_identifiers: tuple[str, ...]
    source_identifiers: tuple[str, ...]
    energy_node_identifiers: tuple[str, ...]
    charge_transitions: tuple[str, ...]
    population_density_m3: np.ndarray
    capture_n_m3_s: np.ndarray
    capture_p_m3_s: np.ndarray
    n1_m3: np.ndarray
    p1_m3: np.ndarray
    intrinsic_product_m6: np.ndarray
    layout_version: str = DYNAMIC_BULK_TRAP_LAYOUT_VERSION

    def __post_init__(self) -> None:
        digest = str(self.model_identity_sha256)
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise DynamicBulkTrapStateError(
                "model_identity_sha256 must be a lowercase SHA-256"
            )
        node_count = int(self.node_count)
        if node_count < 3:
            raise DynamicBulkTrapStateError("node_count must be at least three")
        object.__setattr__(self, "node_count", node_count)

        integer_names = (
            "device_node_indices",
            "region_indices",
            "source_indices",
            "energy_indices",
        )
        integer_values: dict[str, np.ndarray] = {}
        for name in integer_names:
            values = _readonly(getattr(self, name), dtype=np.int64)
            if values.ndim != 1 or values.size == 0 or np.any(values < 0):
                raise DynamicBulkTrapStateError(
                    f"{name} must be a non-empty non-negative vector"
                )
            integer_values[name] = values
            object.__setattr__(self, name, values)
        size = integer_values["device_node_indices"].size
        if any(value.size != size for value in integer_values.values()):
            raise DynamicBulkTrapStateError("layout index vectors must align")
        if np.any(integer_values["device_node_indices"] >= node_count):
            raise DynamicBulkTrapStateError("device node index is outside the grid")

        string_names = (
            "region_identifiers",
            "source_identifiers",
            "energy_node_identifiers",
            "charge_transitions",
        )
        for name in string_names:
            values = tuple(str(value).strip() for value in getattr(self, name))
            if len(values) != size or any(not value for value in values):
                raise DynamicBulkTrapStateError(
                    f"{name} must contain one non-empty value per state"
                )
            object.__setattr__(self, name, values)
        if any(value not in SUPPORTED_TRANSITIONS for value in self.charge_transitions):
            raise DynamicBulkTrapStateError("layout contains an unsupported transition")

        positive_names = (
            "population_density_m3",
            "n1_m3",
            "p1_m3",
            "intrinsic_product_m6",
        )
        nonnegative_names = ("capture_n_m3_s", "capture_p_m3_s")
        for name in (*positive_names, *nonnegative_names):
            values = _readonly(getattr(self, name), dtype=float)
            if values.shape != (size,) or not np.all(np.isfinite(values)):
                raise DynamicBulkTrapStateError(f"{name} must align and be finite")
            if name in positive_names and np.any(values <= 0.0):
                raise DynamicBulkTrapStateError(f"{name} must be positive")
            if name in nonnegative_names and np.any(values < 0.0):
                raise DynamicBulkTrapStateError(f"{name} must be non-negative")
            object.__setattr__(self, name, values)
        if not np.all((self.capture_n_m3_s > 0.0) | (self.capture_p_m3_s > 0.0)):
            raise DynamicBulkTrapStateError(
                "every dynamic state needs at least one active capture leg"
            )
        if self.layout_version != DYNAMIC_BULK_TRAP_LAYOUT_VERSION:
            raise DynamicBulkTrapStateError("unsupported dynamic trap layout version")

    @property
    def size(self) -> int:
        return int(self.device_node_indices.size)

    @property
    def charged_state_mask(self) -> np.ndarray:
        return _readonly(
            np.asarray(self.charge_transitions) != NEUTRAL,
            dtype=bool,
        )

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(
            {
                "layout_version": self.layout_version,
                "model_identity_sha256": self.model_identity_sha256,
                "node_count": self.node_count,
                "device_node_indices": self.device_node_indices.tolist(),
                "region_indices": self.region_indices.tolist(),
                "source_indices": self.source_indices.tolist(),
                "energy_indices": self.energy_indices.tolist(),
                "region_identifiers": list(self.region_identifiers),
                "source_identifiers": list(self.source_identifiers),
                "energy_node_identifiers": list(self.energy_node_identifiers),
                "charge_transitions": list(self.charge_transitions),
                "population_density_m3": self.population_density_m3.tolist(),
                "capture_n_m3_s": self.capture_n_m3_s.tolist(),
                "capture_p_m3_s": self.capture_p_m3_s.tolist(),
                "n1_m3": self.n1_m3.tolist(),
                "p1_m3": self.p1_m3.tolist(),
                "intrinsic_product_m6": self.intrinsic_product_m6.tolist(),
            }
        )


@dataclass(frozen=True, slots=True)
class DynamicBulkTrapEvaluation:
    """Carrier capture, storage, and electrostatic charge for one occupancy."""

    layout_identity_sha256: str
    occupancy: np.ndarray
    relaxation_rate_s1: np.ndarray
    occupied_storage_m3: np.ndarray
    occupancy_rate_s1: np.ndarray
    trap_storage_rate_m3_s: np.ndarray
    electron_capture_rate_m3_s: np.ndarray
    hole_capture_rate_m3_s: np.ndarray
    charge_density_C_m3: np.ndarray
    total_electron_capture_rate_m3_s: np.ndarray
    total_hole_capture_rate_m3_s: np.ndarray
    total_trap_storage_rate_m3_s: np.ndarray
    total_charge_density_C_m3: np.ndarray
    maximum_local_charge_balance_relative_error: float
    evaluation_version: str = DYNAMIC_BULK_TRAP_EVALUATION_VERSION

    def __post_init__(self) -> None:
        digest = str(self.layout_identity_sha256)
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise DynamicBulkTrapStateError(
                "layout_identity_sha256 must be a lowercase SHA-256"
            )
        state_names = (
            "occupancy",
            "relaxation_rate_s1",
            "occupied_storage_m3",
            "occupancy_rate_s1",
            "trap_storage_rate_m3_s",
            "electron_capture_rate_m3_s",
            "hole_capture_rate_m3_s",
            "charge_density_C_m3",
        )
        state_size: int | None = None
        for name in state_names:
            values = _readonly(getattr(self, name), dtype=float)
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                raise DynamicBulkTrapStateError(f"{name} must be a finite vector")
            if state_size is None:
                state_size = values.size
            elif values.size != state_size:
                raise DynamicBulkTrapStateError("dynamic state outputs must align")
            object.__setattr__(self, name, values)
        if state_size is None or state_size == 0:
            raise DynamicBulkTrapStateError("dynamic evaluation must not be empty")
        if np.any((self.occupancy < 0.0) | (self.occupancy > 1.0)):
            raise DynamicBulkTrapStateError("occupancy must lie in [0, 1]")
        if np.any(self.relaxation_rate_s1 <= 0.0):
            raise DynamicBulkTrapStateError("relaxation rates must be positive")
        node_names = (
            "total_electron_capture_rate_m3_s",
            "total_hole_capture_rate_m3_s",
            "total_trap_storage_rate_m3_s",
            "total_charge_density_C_m3",
        )
        node_size: int | None = None
        for name in node_names:
            values = _readonly(getattr(self, name), dtype=float)
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                raise DynamicBulkTrapStateError(f"{name} must be a finite vector")
            if node_size is None:
                node_size = values.size
            elif values.size != node_size:
                raise DynamicBulkTrapStateError("node aggregates must align")
            object.__setattr__(self, name, values)
        error = float(self.maximum_local_charge_balance_relative_error)
        if not math.isfinite(error) or error < 0.0:
            raise DynamicBulkTrapStateError(
                "maximum local charge-balance error must be finite and non-negative"
            )
        object.__setattr__(self, "maximum_local_charge_balance_relative_error", error)
        if self.evaluation_version != DYNAMIC_BULK_TRAP_EVALUATION_VERSION:
            raise DynamicBulkTrapStateError("unsupported dynamic evaluation version")


def compile_dynamic_bulk_trap_layout(
    model: MonovalentBulkDefectModel,
    *,
    dynamic_node_mask: object | None = None,
) -> DynamicBulkTrapLayout:
    """Expand regions, positions, sources, and energy nodes into one state order."""
    if not isinstance(model, MonovalentBulkDefectModel):
        raise TypeError("model must be a MonovalentBulkDefectModel")
    selected = (
        np.ones(model.node_count, dtype=bool)
        if dynamic_node_mask is None
        else np.asarray(dynamic_node_mask, dtype=bool)
    )
    if selected.shape != (model.node_count,):
        raise DynamicBulkTrapStateError(
            "dynamic_node_mask must match the compiled electrical grid"
        )

    device_nodes: list[int] = []
    region_indices: list[int] = []
    source_indices: list[int] = []
    energy_indices: list[int] = []
    region_identifiers: list[str] = []
    source_identifiers: list[str] = []
    energy_identifiers: list[str] = []
    transitions: list[str] = []
    populations: list[float] = []
    capture_n_values: list[float] = []
    capture_p_values: list[float] = []
    n1_values: list[float] = []
    p1_values: list[float] = []
    intrinsic_products: list[float] = []
    global_source_offset = 0

    for region_index, region in enumerate(model.regions):
        region_nodes = np.flatnonzero(region.active_nodes)
        thermal = thermal_voltage(region.temperature_K)
        for local_node_index, device_node in enumerate(region_nodes):
            if not selected[device_node]:
                continue
            if region.has_spatial_profiles:
                local_gap = float(region.local_band_gap_eV[local_node_index])
                local_nc = float(
                    region.local_effective_conduction_dos_m3[local_node_index]
                )
                local_nv = float(
                    region.local_effective_valence_dos_m3[local_node_index]
                )
            else:
                local_gap = region.band_gap_eV
                local_nc = region.effective_conduction_dos_m3
                local_nv = region.effective_valence_dos_m3
            for source_index, (source, quadrature) in enumerate(
                zip(region.species, region.source_quadratures, strict=True)
            ):
                multiplier = (
                    float(
                        region.source_density_multipliers[
                            source_index, local_node_index
                        ]
                    )
                    if region.has_spatial_profiles
                    else 1.0
                )
                capture_n = (
                    source.kinetics.sigma_n_m2 * source.kinetics.thermal_velocity_n_m_s
                )
                capture_p = (
                    source.kinetics.sigma_p_m2 * source.kinetics.thermal_velocity_p_m_s
                )
                for energy_index, (energy, density, identifier) in enumerate(
                    zip(
                        quadrature.energy_levels_eV_above_vb,
                        quadrature.density_weights_m3,
                        region.source_node_identifiers[source_index],
                        strict=True,
                    )
                ):
                    device_nodes.append(int(device_node))
                    region_indices.append(region_index)
                    source_indices.append(global_source_offset + source_index)
                    energy_indices.append(energy_index)
                    region_identifiers.append(region.identifier)
                    source_identifiers.append(str(source.name))
                    energy_identifiers.append(identifier)
                    transitions.append(source.charge_transition)
                    populations.append(float(density) * multiplier)
                    capture_n_values.append(capture_n)
                    capture_p_values.append(capture_p)
                    n1_values.append(
                        local_nc * math.exp(-(local_gap - float(energy)) / thermal)
                    )
                    p1_values.append(local_nv * math.exp(-float(energy) / thermal))
                    intrinsic_products.append(
                        local_nc * local_nv * math.exp(-local_gap / thermal)
                    )
        global_source_offset += len(region.species)

    if not device_nodes:
        raise DynamicBulkTrapStateError(
            "dynamic_node_mask selects no compiled defect state"
        )
    return DynamicBulkTrapLayout(
        model_identity_sha256=model.identity_sha256,
        node_count=model.node_count,
        device_node_indices=np.asarray(device_nodes),
        region_indices=np.asarray(region_indices),
        source_indices=np.asarray(source_indices),
        energy_indices=np.asarray(energy_indices),
        region_identifiers=tuple(region_identifiers),
        source_identifiers=tuple(source_identifiers),
        energy_node_identifiers=tuple(energy_identifiers),
        charge_transitions=tuple(transitions),
        population_density_m3=np.asarray(populations),
        capture_n_m3_s=np.asarray(capture_n_values),
        capture_p_m3_s=np.asarray(capture_p_values),
        n1_m3=np.asarray(n1_values),
        p1_m3=np.asarray(p1_values),
        intrinsic_product_m6=np.asarray(intrinsic_products),
    )


def quasi_steady_bulk_trap_occupancy(
    electron_density_m3: object,
    hole_density_m3: object,
    layout: DynamicBulkTrapLayout,
) -> np.ndarray:
    """Return the exact local QSS occupancy in dynamic state order."""
    n, p = _validate_carriers(electron_density_m3, hole_density_m3, layout)
    node = layout.device_node_indices
    filled = layout.capture_n_m3_s * n[node] + layout.capture_p_m3_s * layout.p1_m3
    empty = layout.capture_n_m3_s * layout.n1_m3 + layout.capture_p_m3_s * p[node]
    denominator = filled + empty
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise DynamicBulkTrapStateError(
            "dynamic trap kinetic denominator must be finite and positive"
        )
    occupancy = filled / denominator
    if not np.all(np.isfinite(occupancy)) or np.any(
        (occupancy <= 0.0) | (occupancy >= 1.0)
    ):
        raise DynamicBulkTrapStateError(
            "QSS occupancy must lie strictly inside (0, 1) for logit coordinates"
        )
    return _readonly(occupancy, dtype=float)


def occupancy_logit(occupancy: object, layout: DynamicBulkTrapLayout) -> np.ndarray:
    """Map a physical occupancy to its unbounded coordinate without clipping."""
    values = np.asarray(occupancy, dtype=float)
    if values.shape != (layout.size,) or not np.all(np.isfinite(values)):
        raise DynamicBulkTrapStateError("occupancy must be finite and match layout")
    if np.any((values <= 0.0) | (values >= 1.0)):
        raise DynamicBulkTrapStateError("logit occupancy must lie inside (0, 1)")
    return _readonly(np.log(values) - np.log1p(-values), dtype=float)


def occupancy_from_logit_increment(
    reference_logit: object,
    increment: object,
    layout: DynamicBulkTrapLayout,
) -> np.ndarray:
    """Apply an unbounded logit increment and return physical occupancy."""
    reference = np.asarray(reference_logit, dtype=float)
    delta = np.asarray(increment, dtype=float)
    if (
        reference.shape != (layout.size,)
        or delta.shape != (layout.size,)
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(delta))
    ):
        raise DynamicBulkTrapStateError(
            "reference_logit and increment must be finite and match layout"
        )
    occupancy = _stable_expit(reference + delta)
    if not np.all(np.isfinite(occupancy)) or np.any(
        (occupancy <= 0.0) | (occupancy >= 1.0)
    ):
        raise DynamicBulkTrapStateError(
            "logit transform saturated outside resolvable occupancy bounds"
        )
    return _readonly(occupancy, dtype=float)


def _validate_carriers(
    electron_density_m3: object,
    hole_density_m3: object,
    layout: DynamicBulkTrapLayout,
) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(electron_density_m3, dtype=float)
    p = np.asarray(hole_density_m3, dtype=float)
    if (
        n.shape != (layout.node_count,)
        or p.shape != (layout.node_count,)
        or not np.all(np.isfinite(n))
        or not np.all(np.isfinite(p))
        or np.any(n < 0.0)
        or np.any(p < 0.0)
    ):
        raise DynamicBulkTrapStateError(
            "carrier densities must be finite, non-negative, and match the grid"
        )
    return n, p


def bulk_trap_charge_density(
    occupancy: object,
    layout: DynamicBulkTrapLayout,
) -> np.ndarray:
    """Aggregate absolute defect charge onto the electrical grid."""
    if not isinstance(layout, DynamicBulkTrapLayout):
        raise TypeError("layout must be a DynamicBulkTrapLayout")
    f = np.asarray(occupancy, dtype=float)
    if f.shape != (layout.size,) or not np.all(np.isfinite(f)):
        raise DynamicBulkTrapStateError("occupancy must be finite and match layout")
    if np.any((f < 0.0) | (f > 1.0)):
        raise DynamicBulkTrapStateError("occupancy must lie in [0, 1]")
    transition = np.asarray(layout.charge_transitions)
    state_charge = np.zeros(layout.size, dtype=float)
    acceptor = transition == ACCEPTOR
    donor = transition == DONOR
    state_charge[acceptor] = -Q * layout.population_density_m3[acceptor] * f[acceptor]
    state_charge[donor] = Q * layout.population_density_m3[donor] * (1.0 - f[donor])
    total = np.zeros(layout.node_count, dtype=float)
    np.add.at(total, layout.device_node_indices, state_charge)
    return _readonly(total, dtype=float)


def _build_dynamic_evaluation(
    layout: DynamicBulkTrapLayout,
    occupancy: np.ndarray,
    relaxation_rate_s1: np.ndarray,
    occupancy_rate_s1: np.ndarray,
    electron_capture_rate_per_trap_s1: np.ndarray,
    hole_capture_rate_per_trap_s1: np.ndarray,
) -> DynamicBulkTrapEvaluation:
    population = layout.population_density_m3
    electron_capture = population * electron_capture_rate_per_trap_s1
    hole_capture = population * hole_capture_rate_per_trap_s1
    storage = population * occupancy
    storage_rate = population * occupancy_rate_s1
    transition = np.asarray(layout.charge_transitions)
    charge = np.zeros(layout.size, dtype=float)
    acceptor = transition == ACCEPTOR
    donor = transition == DONOR
    charge[acceptor] = -Q * storage[acceptor]
    charge[donor] = Q * (population[donor] - storage[donor])

    node = layout.device_node_indices
    total_electron = np.zeros(layout.node_count, dtype=float)
    total_hole = np.zeros(layout.node_count, dtype=float)
    total_storage_rate = np.zeros(layout.node_count, dtype=float)
    total_charge = np.zeros(layout.node_count, dtype=float)
    np.add.at(total_electron, node, electron_capture)
    np.add.at(total_hole, node, hole_capture)
    np.add.at(total_storage_rate, node, storage_rate)
    np.add.at(total_charge, node, charge)

    charge_rate = np.zeros(layout.size, dtype=float)
    charged = acceptor | donor
    charge_rate[charged] = -Q * storage_rate[charged]
    capture_charge_rate = np.zeros(layout.size, dtype=float)
    capture_charge_rate[charged] = -Q * (
        electron_capture[charged] - hole_capture[charged]
    )
    scale = np.abs(charge_rate) + np.abs(capture_charge_rate)
    relative = np.divide(
        np.abs(charge_rate - capture_charge_rate),
        scale,
        out=np.zeros_like(scale),
        where=scale > 0.0,
    )
    return DynamicBulkTrapEvaluation(
        layout_identity_sha256=layout.identity_sha256,
        occupancy=occupancy,
        relaxation_rate_s1=relaxation_rate_s1,
        occupied_storage_m3=storage,
        occupancy_rate_s1=occupancy_rate_s1,
        trap_storage_rate_m3_s=storage_rate,
        electron_capture_rate_m3_s=electron_capture,
        hole_capture_rate_m3_s=hole_capture,
        charge_density_C_m3=charge,
        total_electron_capture_rate_m3_s=total_electron,
        total_hole_capture_rate_m3_s=total_hole,
        total_trap_storage_rate_m3_s=total_storage_rate,
        total_charge_density_C_m3=total_charge,
        maximum_local_charge_balance_relative_error=float(np.max(relative)),
    )


def evaluate_dynamic_bulk_traps(
    electron_density_m3: object,
    hole_density_m3: object,
    occupancy: object,
    layout: DynamicBulkTrapLayout,
) -> DynamicBulkTrapEvaluation:
    """Evaluate non-equilibrium capture, occupancy rate, storage, and charge."""
    if not isinstance(layout, DynamicBulkTrapLayout):
        raise TypeError("layout must be a DynamicBulkTrapLayout")
    n, p = _validate_carriers(electron_density_m3, hole_density_m3, layout)
    f = np.asarray(occupancy, dtype=float)
    if f.shape != (layout.size,) or not np.all(np.isfinite(f)):
        raise DynamicBulkTrapStateError("occupancy must be finite and match layout")
    if np.any((f < 0.0) | (f > 1.0)):
        raise DynamicBulkTrapStateError("occupancy must lie in [0, 1]")

    node = layout.device_node_indices
    capture_n = layout.capture_n_m3_s
    capture_p = layout.capture_p_m3_s
    n_local = n[node]
    p_local = p[node]
    filled = capture_n * n_local + capture_p * layout.p1_m3
    empty = capture_n * layout.n1_m3 + capture_p * p_local
    relaxation = filled + empty
    qss_occupancy = filled / relaxation
    occupancy_offset = f - qss_occupancy
    # Algebraically identical to direct capture evaluation, without losing
    # the small net rate to cancellation near a QSS operating point.
    qss_rate_per_trap = (
        capture_n
        * capture_p
        * (n_local * p_local - layout.intrinsic_product_m6)
        / relaxation
    )
    electron_per_trap = qss_rate_per_trap - (
        capture_n * (n_local + layout.n1_m3) * occupancy_offset
    )
    hole_per_trap = qss_rate_per_trap + (
        capture_p * (p_local + layout.p1_m3) * occupancy_offset
    )
    occupancy_rate = -relaxation * occupancy_offset
    return _build_dynamic_evaluation(
        layout,
        f,
        relaxation,
        occupancy_rate,
        electron_per_trap,
        hole_per_trap,
    )


def evaluate_dynamic_bulk_traps_about_qss(
    electron_density_m3: object,
    hole_density_m3: object,
    occupancy: object,
    layout: DynamicBulkTrapLayout,
    *,
    reference_electron_density_m3: object,
    reference_hole_density_m3: object,
    reference_occupancy: object,
) -> DynamicBulkTrapEvaluation:
    """Evaluate the exact kinetics as increments about a certified QSS state.

    The expanded form avoids subtracting two nearly identical occupancies when
    the carrier-induced ``df`` is below the resolution of their absolute
    floating-point representation.  Bilinear ``dn*df`` and ``dp*df`` terms are
    retained, so this is an exact rewrite rather than a linear approximation.
    """
    if not isinstance(layout, DynamicBulkTrapLayout):
        raise TypeError("layout must be a DynamicBulkTrapLayout")
    n, p = _validate_carriers(electron_density_m3, hole_density_m3, layout)
    n0, p0 = _validate_carriers(
        reference_electron_density_m3,
        reference_hole_density_m3,
        layout,
    )
    f = np.asarray(occupancy, dtype=float)
    f0 = np.asarray(reference_occupancy, dtype=float)
    for name, values in (("occupancy", f), ("reference_occupancy", f0)):
        if values.shape != (layout.size,) or not np.all(np.isfinite(values)):
            raise DynamicBulkTrapStateError(f"{name} must be finite and match layout")
        if np.any((values < 0.0) | (values > 1.0)):
            raise DynamicBulkTrapStateError(f"{name} must lie in [0, 1]")
    expected_f0 = quasi_steady_bulk_trap_occupancy(n0, p0, layout)
    if not np.array_equal(f0, expected_f0):
        raise DynamicBulkTrapStateError(
            "reference occupancy must be the exact QSS state for its carriers"
        )

    node = layout.device_node_indices
    capture_n = layout.capture_n_m3_s
    capture_p = layout.capture_p_m3_s
    n0_local = n0[node]
    p0_local = p0[node]
    dn = n[node] - n0_local
    dp = p[node] - p0_local
    df = f - f0
    relaxation0 = capture_n * (n0_local + layout.n1_m3) + capture_p * (
        p0_local + layout.p1_m3
    )
    qss_rate0 = (
        capture_n
        * capture_p
        * (n0_local * p0_local - layout.intrinsic_product_m6)
        / relaxation0
    )
    electron_per_trap = qss_rate0 + capture_n * (
        (1.0 - f0) * dn - (n0_local + layout.n1_m3) * df - dn * df
    )
    hole_per_trap = qss_rate0 + capture_p * (
        f0 * dp + (p0_local + layout.p1_m3) * df + dp * df
    )
    occupancy_rate = (
        capture_n * (1.0 - f0) * dn
        - capture_p * f0 * dp
        - relaxation0 * df
        - capture_n * dn * df
        - capture_p * dp * df
    )
    relaxation = capture_n * (n[node] + layout.n1_m3) + capture_p * (
        p[node] + layout.p1_m3
    )
    return _build_dynamic_evaluation(
        layout,
        f,
        relaxation,
        occupancy_rate,
        electron_per_trap,
        hole_per_trap,
    )


__all__ = [
    "DYNAMIC_BULK_TRAP_EVALUATION_VERSION",
    "DYNAMIC_BULK_TRAP_LAYOUT_VERSION",
    "DynamicBulkTrapEvaluation",
    "DynamicBulkTrapLayout",
    "DynamicBulkTrapStateError",
    "bulk_trap_charge_density",
    "compile_dynamic_bulk_trap_layout",
    "evaluate_dynamic_bulk_traps",
    "evaluate_dynamic_bulk_traps_about_qss",
    "occupancy_from_logit_increment",
    "occupancy_logit",
    "quasi_steady_bulk_trap_occupancy",
]
