"""Explicit A-D rate controls for the R1 physical-volume research operator."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from perovskite_sim.experiments.interface_defect_transient import (
    InterfaceDefectTransientError,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1 import (
    PhysicalInterfaceIonSystem,
)
from perovskite_sim.physics.dynamic_storage import logit_occupancy_increment
from perovskite_sim.physics.physical_control_volume import physical_contact_displacement


CURRENT_METRIC_SEMANTICS = "r1-v3-separate-internal-contact-interface"


@dataclass(frozen=True, slots=True)
class R1DynamicsControls:
    """Select rate equations while retaining original species and charge."""

    nu_I: int = 1
    nu_t: int = 1

    def __post_init__(self):
        for name in ("nu_I", "nu_t"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer 0 or 1")
            if value not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
            object.__setattr__(self, name, int(value))

    @property
    def label(self) -> str:
        return {(0, 0): "A", (1, 0): "B", (0, 1): "C", (1, 1): "D"}[
            self.nu_I, self.nu_t
        ]

    @classmethod
    def from_label(cls, label: str):
        if not isinstance(label, str):
            raise TypeError("control label must be A, B, C, or D")
        pairs = {"A": (0, 0), "B": (1, 0), "C": (0, 1), "D": (1, 1)}
        try:
            return cls(*pairs[label.strip().upper()])
        except KeyError as exc:
            raise ValueError("control label must be A, B, C, or D") from exc


class ControlledPhysicalInterfaceIonSystem(PhysicalInterfaceIonSystem):
    """Apply controls to fluxes and direct tangents, retaining all storage."""

    def __init__(self, *args, controls=R1DynamicsControls(), **kwargs):
        if not isinstance(controls, R1DynamicsControls):
            raise TypeError("controls must be R1DynamicsControls")
        if "capture_multiplier" in kwargs:
            raise TypeError("R1 capture dynamics are selected through controls.nu_t")
        self._controls = controls
        super().__init__(*args, capture_multiplier=controls.nu_t, **kwargs)

    @property
    def controls(self):
        return self._controls

    def _coordinates(self, coordinate, voltage):
        result = list(super()._coordinates(coordinate, voltage))
        # Pinned endpoint reservoirs have exactly fixed physical populations;
        # avoid recreating them through cancelling QF and voltage increments.
        result[3] = result[3].copy()
        result[4] = result[4].copy()
        result[3][[0, -1]] = self.reference_n[[0, -1]]
        result[4][[0, -1]] = self.reference_p[[0, -1]]
        reference = (
            self.reference_occupancy
            if self._step_reference is None
            else self._step_reference.occupancy
        )
        increment = np.asarray(coordinate, dtype=float)[self.trap_slice]
        occupancy = reference + logit_occupancy_increment(reference, increment)
        if np.any((occupancy <= 0.0) | (occupancy >= 1.0)):
            raise InterfaceDefectTransientError("R1 interface occupancy left (0, 1)")
        result[5] = occupancy
        return tuple(result)

    def _ion_fields(self, phi, positive, negative):
        if self.controls.nu_I:
            return super()._ion_fields(phi, positive, negative)
        return (
            np.zeros_like(positive),
            None if negative is None else np.zeros_like(negative),
            np.zeros(self.node_count - 1),
            None if negative is None else np.zeros(self.node_count - 1),
        )

    def _trace_density_coordinates(self, coordinate):
        if self._step_reference is None:
            return None
        values = np.asarray(coordinate, dtype=float)
        reference = np.asarray([item.state_m3 for item in self._step_reference.local])
        increments = np.asarray([values[self._local_block_slice(k)][2:]
                                 for k in range(self.interface_count)])
        return reference * np.exp(increments)

    def solver_current_metrics(self, state, previous, dt):
        """Keep contact-aware stopping separate from published internal spread.

        This is a solver consistency condition, not independent evidence. The
        separate state reconstruction supplies the physical acceptance checks.
        """
        metrics = super().transient_current_metrics(state, previous, dt)
        storage = self.storage_increment(state, previous)
        delta_rho = self._increment_charge_density(storage)
        delta_displacement = -self.material.poisson_factor.C * np.diff(
            self.potential_increment(state, previous))
        contact_displacement = physical_contact_displacement(
            delta_displacement, delta_rho, self.widths) / dt
        # The endpoint electron/hole recombination corrections cancel in the
        # physical total current. Include contacts in Newton's existing gate,
        # so a locally accepted step cannot stop before its boundary closes.
        contact_total = self.polarity * (state.current_n[[0, -1]]
                                        + state.current_p[[0, -1]] + contact_displacement)
        all_total = np.r_[metrics[1], contact_total]
        spread = float(np.ptp(all_total)) / max(float(np.max(np.abs(all_total))), 1e-20)
        return (*metrics[:4], spread, metrics[5])

    def failure_evidence(self, state, previous, voltage, dt, residual, storage_scale,
                         poisson_scale, local_scale, diagnostics):
        """Retain the actual failed Newton iterate without changing its verdict."""
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import snapshot, json_data
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_independent_physics import independent_physics_row
        return json_data({"schema": "R1NewtonFailureWitnessV1", "terminal_state_available": True,
            "scope": "failed_iterate_not_accepted_step_or_complete_newton_history",
            "voltage_V": float(voltage), "dt_s": float(dt),
            "coordinate": state.coordinate, "previous_coordinate": previous.coordinate,
            "previous_state": snapshot(self, previous), "attempted_state": snapshot(self, state),
            "scaled_residual_vector": residual, "storage_scale": storage_scale,
            "poisson_scale": poisson_scale, "local_scale": local_scale,
            "diagnostics": diagnostics,
            "independent_physics": independent_physics_row(self, state, previous, dt)})

    def _ion_jacobians(self, phi, positive, negative):
        if self.controls.nu_I:
            return super()._ion_jacobians(phi, positive, negative)
        # Disabled transport need not evaluate flux clipping or its tangent.
        # Population storage and the ion contribution to Poisson remain live.
        positive_density = self._density_jacobian(
            positive, self.positive_nodes, self.positive_slice, self.dimension,
        )
        negative_density = sparse.csr_matrix((self.node_count, self.dimension))
        if negative is not None:
            negative_density = self._density_jacobian(
                negative, self.negative_nodes, self.negative_slice, self.dimension,
            )
        return (
            positive_density, negative_density,
            sparse.csr_matrix((self.node_count, self.dimension)),
            sparse.csr_matrix((self.node_count, self.dimension)),
        )


__all__ = ["R1DynamicsControls", "ControlledPhysicalInterfaceIonSystem"]
