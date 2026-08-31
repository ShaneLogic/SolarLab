"""D7-E3 stationary metastable preparation and frozen measurement.

SCAPS establishes the donor/acceptor configuration distribution at an initial
working point and then freezes it for the measurement. This module implements
that split explicitly:

``prepare_metastable_configuration``
    Solves the outer fixed point ``y = Y(n(y), p(y))`` at the preparation
    working point declared by a :class:`MetastablePreparationProtocol`. The
    iteration is clamped by the protocol's own numerics and must finish with
    an unclamped refinement step, so a clamped iterate can never be accepted
    as the answer.

``FrozenMetastableConfiguration``
    The immutable result. It carries the protocol hash, a content hash over
    the converged configuration and the state it was solved from, and the
    compiled :class:`FrozenMetastableBulkDefectModel` the measurement uses.

``solve_frozen_metastable_measurement`` / ``solve_frozen_metastable_jv_sweep``
    Measurement entry points. They re-check the frozen provenance and then
    run the ordinary guarded QF/DC solver with the frozen inventory attached.
    The configuration fraction is *not* recomputed, so a bias sweep cannot
    silently re-prepare the device.

Capability boundary
-------------------
The prepared region never owns the two contact nodes (the same convention the
D6 dynamic-trap layout uses), so contact charge neutrality is unaffected by
the metastable inventory and needs no separate certification here. A prepared
device may not also carry a compiled neutral, monovalent or multivalent
inventory: the preparation certifies the metastable partition only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math

import numpy as np

from perovskite_sim.experiments.quasi_fermi_steady_state import (
    DEFAULT_ILLUMINATION_STEPS,
    QuasiFermiJVSweepResult,
    QuasiFermiSteadyStateError,
    QuasiFermiSteadyStateResult,
    solve_quasi_fermi_jv_sweep,
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.models.multivalent_defects import (
    MetastableDefectDefinition,
    MetastablePreparationProtocol,
)
from perovskite_sim.physics.metastable_defect_closure import (
    evaluate_metastable_configuration_closure,
)
from perovskite_sim.physics.metastable_defect_device import (
    FrozenMetastableBulkDefectModel,
    FrozenMetastableRegion,
)
from perovskite_sim.solver.mol import MaterialArrays, build_material_arrays


METASTABLE_PREPARATION_VERSION = "metastable-stationary-preparation-v1"


class MetastablePreparationError(RuntimeError):
    """The preparation solve did not converge or violated its own protocol."""


def _array_sha256(label: str, *arrays: np.ndarray) -> str:
    digest = hashlib.sha256(label.encode("ascii"))
    for value in arrays:
        contiguous = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _payload_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class FrozenMetastableConfiguration:
    """An immutable prepared configuration split plus its full provenance."""

    model: FrozenMetastableBulkDefectModel
    definition: MetastableDefectDefinition
    protocol: MetastablePreparationProtocol
    grid_m: np.ndarray
    donor_fraction: np.ndarray
    active_nodes: np.ndarray
    preparation_state: QuasiFermiSteadyStateResult
    outer_iterations: int
    final_relative_change: float
    unclamped_refinement_change: float
    clamped_iterations: int
    protocol_sha256: str
    state_sha256: str
    stack_sha256: str
    grid_sha256: str

    @property
    def frozen(self) -> bool:
        """Whether the protocol demands a frozen measurement configuration."""
        return bool(self.protocol.freeze_configuration_during_measurement)


def _prepared_stack(
    stack: DeviceStack, protocol: MetastablePreparationProtocol
) -> DeviceStack:
    """Return the stack at the declared preparation working point."""

    return replace(
        stack,
        T=float(protocol.preparation_temperature_K),
        Phi=float(stack.Phi) * float(protocol.preparation_illumination_suns),
    )


def _region_layer_index(stack: DeviceStack, layer_name: str) -> int:
    for index, layer in enumerate(electrical_layers(stack)):
        if layer.name == layer_name:
            return index
    raise MetastablePreparationError(
        f"metastable layer {layer_name!r} is not an electrical layer"
    )


def _layer_mask(
    grid_m: np.ndarray,
    stack: DeviceStack,
    layer_index: int,
) -> np.ndarray:
    layers = electrical_layers(stack)
    edges = np.concatenate(([0.0], np.cumsum([layer.thickness for layer in layers])))
    left, right = float(edges[layer_index]), float(edges[layer_index + 1])
    grid = np.asarray(grid_m, dtype=float)
    mask = (grid >= left) & (grid <= right)
    # The prepared region never owns a contact node: the preparation solve
    # certifies the bulk partition, not a contact reservoir law.
    mask[[0, -1]] = False
    if not np.any(mask):
        raise MetastablePreparationError(
            "metastable layer has no interior nodes on this grid"
        )
    return mask


def _build_model(
    *,
    identifier: str,
    definition: MetastableDefectDefinition,
    mask: np.ndarray,
    donor_fraction: np.ndarray,
    material: MaterialArrays,
    temperature_K: float,
    protocol_sha256: str,
    state_sha256: str,
) -> FrozenMetastableBulkDefectModel:
    gap = np.asarray(material.Eg_phys, dtype=float)[mask]
    conduction = np.asarray(material.N_C_physical, dtype=float)[mask]
    valence = np.asarray(material.N_V_physical, dtype=float)[mask]
    for name, values in (
        ("band gap", gap),
        ("conduction DOS", conduction),
        ("valence DOS", valence),
    ):
        if (
            values.size == 0
            or not np.all(np.isfinite(values))
            or np.any(values <= 0.0)
            or not np.all(values == values[0])
        ):
            raise MetastablePreparationError(
                "D7-E3 metastable preparation requires a uniform finite "
                f"positive {name} across the prepared region"
            )
    return FrozenMetastableBulkDefectModel(
        regions=(
            FrozenMetastableRegion(
                identifier=identifier,
                definition_sha256=_payload_sha256(definition.to_dict()),
                preparation_protocol_sha256=protocol_sha256,
                preparation_state_sha256=state_sha256,
                active_nodes=mask,
                donor_fraction=donor_fraction,
                band_gap_eV=float(gap[0]),
                effective_conduction_dos_m3=float(conduction[0]),
                effective_valence_dos_m3=float(valence[0]),
                temperature_K=float(temperature_K),
                definition=definition,
            ),
        )
    )


def prepare_metastable_configuration(
    x: np.ndarray,
    stack: DeviceStack,
    definition: MetastableDefectDefinition,
    protocol: MetastablePreparationProtocol,
    *,
    layer_name: str,
    illuminated: bool | None = None,
    illumination_steps: tuple[float, ...] = DEFAULT_ILLUMINATION_STEPS,
    **solver_controls,
) -> FrozenMetastableConfiguration:
    """Solve the stationary configuration split at the declared working point.

    The outer unknown is the configuration fraction ``y``. Each iteration
    builds the frozen inventory at the current ``y``, solves the guarded QF/DC
    state, and re-evaluates the conversion closure at the resulting carriers.
    The update is clamped by ``protocol.numerics.clamping_factor``; the
    protocol also requires a final unclamped step, whose measured change is
    reported so a clamped iterate can never masquerade as the converged
    answer.
    """

    if not isinstance(definition, MetastableDefectDefinition):
        raise TypeError("definition must be a MetastableDefectDefinition")
    if not isinstance(protocol, MetastablePreparationProtocol):
        raise TypeError("protocol must be a MetastablePreparationProtocol")
    grid = np.asarray(x, dtype=float)
    if grid.ndim != 1 or grid.size < 4 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("x must be a strictly increasing grid with >= 4 nodes")

    numerics = protocol.numerics
    prepared_stack = _prepared_stack(stack, protocol)
    lit = (
        bool(protocol.preparation_illumination_suns > 0.0)
        if illuminated is None
        else bool(illuminated)
    )
    layer_index = _region_layer_index(prepared_stack, layer_name)
    mask = _layer_mask(grid, prepared_stack, layer_index)
    identifier = f"layer[{layer_index}]/{layer_name}"
    protocol_sha256 = protocol.sha256
    base_material = build_material_arrays(grid, prepared_stack)
    definition.validate_band_gap(float(np.asarray(base_material.Eg_phys)[mask][0]))

    node_count = int(np.count_nonzero(mask))
    fraction = np.full(node_count, float(numerics.initial_donor_fraction_guess))
    if not 0.0 < fraction[0] < 1.0:
        raise MetastablePreparationError(
            "initial_donor_fraction_guess must lie strictly inside (0, 1) so "
            "both configurations are represented in the first iterate"
        )

    def solve_at(values: np.ndarray) -> tuple[QuasiFermiSteadyStateResult, np.ndarray]:
        model = _build_model(
            identifier=identifier,
            definition=definition,
            mask=mask,
            donor_fraction=values,
            material=base_material,
            temperature_K=float(prepared_stack.T),
            protocol_sha256=protocol_sha256,
            state_sha256="0" * 64,
        )
        material = replace(base_material, frozen_metastable_defects=model)
        state = solve_quasi_fermi_steady_state(
            grid,
            prepared_stack,
            V_app=float(protocol.preparation_voltage_V),
            illuminated=lit,
            mat=material,
            illumination_steps=illumination_steps if lit else (0.0,),
            **solver_controls,
        )
        count = grid.size
        closure = evaluate_metastable_configuration_closure(
            np.asarray(state.y[:count], dtype=float)[mask],
            np.asarray(state.y[count : 2 * count], dtype=float)[mask],
            definition,
            band_gap_eV=float(np.asarray(base_material.Eg_phys)[mask][0]),
            effective_conduction_dos_m3=float(
                np.asarray(base_material.N_C_physical)[mask][0]
            ),
            effective_valence_dos_m3=float(
                np.asarray(base_material.N_V_physical)[mask][0]
            ),
            temperature_K=float(prepared_stack.T),
        )
        return state, np.asarray(closure.donor_fraction, dtype=float)

    clamp = float(numerics.clamping_factor)
    tolerance = float(numerics.relative_tolerance)
    clamped_iterations = 0
    change = math.inf
    state = None
    for iteration in range(1, numerics.max_iterations + 1):
        state, target = solve_at(fraction)
        step = target - fraction
        scale = max(float(np.max(np.abs(fraction))), 1.0e-12)
        change = float(np.max(np.abs(step))) / scale
        if clamp < 1.0:
            clamped_iterations += 1
        fraction = fraction + clamp * step
        if not np.all(np.isfinite(fraction)):
            raise MetastablePreparationError(
                "metastable preparation produced a non-finite configuration"
            )
        if change <= tolerance:
            break
    else:
        raise MetastablePreparationError(
            "metastable preparation did not converge within "
            f"{numerics.max_iterations} iterations; last relative change "
            f"{change:.6g} > {tolerance:.6g}"
        )
    outer_iterations = iteration

    # The protocol forbids accepting a clamped iterate: take one full,
    # unclamped step and report how far it moved.
    state, target = solve_at(fraction)
    unclamped_step = target - fraction
    scale = max(float(np.max(np.abs(fraction))), 1.0e-12)
    unclamped_change = float(np.max(np.abs(unclamped_step))) / scale
    fraction = target
    if unclamped_change > tolerance:
        raise MetastablePreparationError(
            "final unclamped metastable refinement exceeded the protocol "
            f"tolerance: {unclamped_change:.6g} > {tolerance:.6g}"
        )
    state, _final_target = solve_at(fraction)
    if not state.certified:
        raise MetastablePreparationError(
            "prepared metastable working point is not residual-certified"
        )

    grid_sha256 = _array_sha256("metastable-preparation-grid-v1", grid)
    stack_sha256 = hashlib.sha256(repr(prepared_stack).encode("utf-8")).hexdigest()
    state_sha256 = _array_sha256(
        "metastable-preparation-state-v1",
        grid,
        fraction,
        state.y,
        state.phi,
        np.asarray([float(protocol.preparation_voltage_V)]),
        np.asarray([float(protocol.preparation_temperature_K)]),
        np.asarray([float(protocol.preparation_illumination_suns)]),
    )
    model = _build_model(
        identifier=identifier,
        definition=definition,
        mask=mask,
        donor_fraction=fraction,
        material=base_material,
        temperature_K=float(prepared_stack.T),
        protocol_sha256=protocol_sha256,
        state_sha256=state_sha256,
    )
    return FrozenMetastableConfiguration(
        model=model,
        definition=definition,
        protocol=protocol,
        grid_m=np.array(grid, copy=True),
        donor_fraction=np.array(fraction, copy=True),
        active_nodes=np.array(mask, copy=True),
        preparation_state=state,
        outer_iterations=outer_iterations,
        final_relative_change=change,
        unclamped_refinement_change=unclamped_change,
        clamped_iterations=clamped_iterations,
        protocol_sha256=protocol_sha256,
        state_sha256=state_sha256,
        stack_sha256=stack_sha256,
        grid_sha256=grid_sha256,
    )


def _measurement_material(
    grid: np.ndarray,
    stack: DeviceStack,
    frozen: FrozenMetastableConfiguration,
) -> MaterialArrays:
    if not isinstance(frozen, FrozenMetastableConfiguration):
        raise TypeError("frozen must be a FrozenMetastableConfiguration")
    if not frozen.frozen:
        raise MetastablePreparationError(
            "measurement requires a protocol that freezes the configuration"
        )
    if not np.array_equal(np.asarray(grid, dtype=float), frozen.grid_m):
        raise MetastablePreparationError(
            "measurement grid does not match the prepared grid"
        )
    material = build_material_arrays(grid, stack)
    return replace(material, frozen_metastable_defects=frozen.model)


def solve_frozen_metastable_measurement(
    x: np.ndarray,
    stack: DeviceStack,
    frozen: FrozenMetastableConfiguration,
    V_app: float = 0.0,
    *,
    illuminated: bool = False,
    **solver_controls,
) -> QuasiFermiSteadyStateResult:
    """Solve one measurement point against a frozen configuration."""

    grid = np.asarray(x, dtype=float)
    material = _measurement_material(grid, stack, frozen)
    measurement_stack = replace(
        stack, T=float(frozen.protocol.measurement_temperature_K)
    )
    try:
        return solve_quasi_fermi_steady_state(
            grid,
            measurement_stack,
            V_app=float(V_app),
            illuminated=illuminated,
            mat=material,
            **solver_controls,
        )
    except QuasiFermiSteadyStateError as exc:
        raise MetastablePreparationError(
            f"frozen metastable measurement did not certify: {exc}"
        ) from exc


def solve_frozen_metastable_jv_sweep(
    x: np.ndarray,
    stack: DeviceStack,
    frozen: FrozenMetastableConfiguration,
    voltages_V: np.ndarray,
    **solver_controls,
) -> QuasiFermiJVSweepResult:
    """Sweep bias against a frozen configuration that never re-prepares."""

    grid = np.asarray(x, dtype=float)
    material = _measurement_material(grid, stack, frozen)
    measurement_stack = replace(
        stack, T=float(frozen.protocol.measurement_temperature_K)
    )
    return solve_quasi_fermi_jv_sweep(
        grid,
        measurement_stack,
        np.asarray(voltages_V, dtype=float),
        mat=material,
        **solver_controls,
    )


__all__ = [
    "METASTABLE_PREPARATION_VERSION",
    "FrozenMetastableConfiguration",
    "MetastablePreparationError",
    "prepare_metastable_configuration",
    "solve_frozen_metastable_jv_sweep",
    "solve_frozen_metastable_measurement",
]
