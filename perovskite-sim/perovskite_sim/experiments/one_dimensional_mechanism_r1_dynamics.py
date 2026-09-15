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
