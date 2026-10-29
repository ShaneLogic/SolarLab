"""Physical BE steps on the unified single/multiple-domain ionic assembly.

The existing BDFStep at order one supplies R = F - delta(S)/dt and its
Jacobian. All storage/secants come from the Assembler through the thin
storage protocol; carrier and each domain/species remain independent.
The original BE predictors and current/charge accounting are reused without
changing their equations, gates or tolerances. No adaptive ionic driver or
new physics is introduced by these constructor-compatible step adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, cast

import numpy as np
from scipy.sparse import csc_matrix

from solarlab.equations.assembly.ionic_1d import AssembledIonic1DProblem, AssembledMultipleIonic1DProblem
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.equations.ionic_1d import IonicState1D
from solarlab.equations.multiple_ionic_1d import MultipleIonicState1D
from solarlab.solvers.time.assembled_storage import AssembledIonicStorage1D, AssembledStorage1D
from solarlab.solvers.time.backward_euler import BackwardEuler1DStep
from solarlab.solvers.time.bdf import BDFStep, HistoryPoint
from solarlab.solvers.time.ionic_backward_euler import (
    IonicBackwardEuler1DStep, step_accounting as ionic_accounting,
)
from solarlab.solvers.time.multiple_ionic_backward_euler import (
    MultipleIonicBackwardEuler1DStep, multiple_ionic_step_accounting as multiple_accounting,
)

__all__ = ["AssembledIonicBackwardEuler1DStep", "AssembledMultipleIonicBackwardEuler1DStep"]


def _step(
    endpoint: AssembledIonic1DProblem | AssembledMultipleIonic1DProblem,
    previous: IonicState1D | MultipleIonicState1D, dt_s: float,
) -> BDFStep:
    old = AssembledIonicStorage1D(endpoint.at(previous.voltage_V, endpoint.generation_scale))
    history = HistoryPoint(0.0, previous.u, old.storage(previous.u), old)
    return BDFStep(AssembledIonicStorage1D(endpoint), (history,), dt_s, 1)


@dataclass(frozen=True)
class AssembledIonicBackwardEuler1DStep:
    """One single-domain step, including independent or shared-site species."""

    endpoint: AssembledIonic1DProblem
    previous: IonicState1D
    dt_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, AssembledIonic1DProblem) or not isinstance(self.previous, IonicState1D):
            raise ValueError("an assembled ionic endpoint and previous ionic state are required")
        if not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        device = self.endpoint.device
        if self.previous.device_sha256 != device.identity:
            raise ValueError("previous state must belong to the same complete ionic device")
        actual = device.state(self.previous.u, voltage_V=self.previous.voltage_V)
        if self.previous.electronic_state.device_sha256 != device.electronic_component.identity:
            raise ValueError("previous electronic component identity is inconsistent")
        for name in ("potential_V", "electron_density_m3", "hole_density_m3"):
            if not np.array_equal(getattr(actual.electronic_state, name), getattr(self.previous.electronic_state, name)):
                raise ValueError("previous electronic fields disagree with their voltage reference")
        if (any(not np.array_equal(getattr(actual, f"{species}_ion_density_m3"),
                                   getattr(self.previous, f"{species}_ion_density_m3"))
                for species in device.ion_species_names)
                or actual.domain_inventories_m2 != self.previous.domain_inventories_m2):
            raise ValueError("previous physical ion fields/inventory disagree with the bound coordinates")

    @property
    def electronic_step(self) -> BackwardEuler1DStep:
        # The unchanged BE constructor is annotated with its old concrete
        # endpoint, but consumes this same public surface at runtime.
        endpoint = cast(DC1DProblem, AssembledStorage1D(self.endpoint.electronic_problem))
        return BackwardEuler1DStep(endpoint, self.previous.electronic_state, self.dt_s)

    @property
    def bdf_step(self) -> BDFStep:
        return _step(self.endpoint, self.previous, self.dt_s)

    def physical_increment(self, y: np.ndarray) -> np.ndarray:
        step = self.bdf_step
        old = step.history[0]
        endpoint = cast(AssembledIonicStorage1D, step.endpoint)
        return endpoint.physical_increment(y, old.problem, old.u)

    def inventory_rates_m2_s(self, y: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        rate = self.physical_increment(y) / self.dt_s
        count = self.endpoint.device.node_count
        return rate[count:3 * count].reshape(2, count), tuple(rate[block] for block in self.endpoint.device.ion_species_slices)

    def inventory_rate_m2_s(self, y: np.ndarray, *, species: str = "positive") -> tuple[np.ndarray, np.ndarray]:
        if species not in self.endpoint.device.ion_species_names:
            raise ValueError("requested ion species is not assembled in this device")
        carrier, ions = self.inventory_rates_m2_s(y)
        return carrier, ions[self.endpoint.device.ion_species_names.index(species)]

    def residual(self, y: np.ndarray) -> np.ndarray:
        return self.bdf_step.residual(y)

    def jacobian(self, y: np.ndarray) -> csc_matrix:
        return csc_matrix(self.bdf_step.jacobian(y))

    def initial_coordinates(self, predictor: str = "previous", *, voltage_lift: bool = False) -> np.ndarray:
        electronic = self.electronic_step.initial_coordinates(predictor, voltage_lift=voltage_lift)
        return np.r_[electronic, self.previous.u[3 * self.endpoint.device.node_count:]]

    def accounting(self, y: np.ndarray) -> dict[str, object]:
        return ionic_accounting(cast(IonicBackwardEuler1DStep, self), y)


@dataclass(frozen=True)
class AssembledMultipleIonicBackwardEuler1DStep:
    """One full step with separate storage for every declared ionic domain."""

    endpoint: AssembledMultipleIonic1DProblem
    previous: MultipleIonicState1D
    dt_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, AssembledMultipleIonic1DProblem):
            raise TypeError("the assembled multiple-domain endpoint equations are required")
        if isinstance(self.dt_s, bool) or not isinstance(self.dt_s, Real) or not math.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("dt_s must be a finite positive real scalar")
        self.endpoint.device.validate_state(self.previous)

    @property
    def electronic_step(self) -> BackwardEuler1DStep:
        endpoint = cast(DC1DProblem, AssembledStorage1D(self.endpoint.electronic_problem))
        return BackwardEuler1DStep(endpoint, self.previous.electronic_state, self.dt_s)

    @property
    def bdf_step(self) -> BDFStep:
        return _step(self.endpoint, self.previous, self.dt_s)

    def physical_increment(self, y: np.ndarray) -> np.ndarray:
        step = self.bdf_step
        old = step.history[0]
        endpoint = cast(AssembledIonicStorage1D, step.endpoint)
        return endpoint.physical_increment(y, old.problem, old.u)

    def inventory_rates_m2_s(self, y: np.ndarray) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
        device, count = self.endpoint.device, self.endpoint.device.node_count
        rate = self.physical_increment(y) / self.dt_s
        domains = {name: {species: rate[block] for species, block in blocks.items()}
                   for name, blocks in device.ion_species_slices.items()}
        return rate[count:3 * count].reshape(2, count), domains

    def residual(self, y: np.ndarray) -> np.ndarray:
        return self.bdf_step.residual(y)

    def jacobian(self, y: np.ndarray) -> csc_matrix:
        return csc_matrix(self.bdf_step.jacobian(y))

    def initial_coordinates(self) -> np.ndarray:
        electronic = self.electronic_step.initial_coordinates("previous", voltage_lift=True)
        return np.r_[electronic, self.previous.u[3 * self.endpoint.device.node_count:]]

    def accounting(self, y: np.ndarray) -> dict[str, Any]:
        return multiple_accounting(cast(MultipleIonicBackwardEuler1DStep, self), y)
