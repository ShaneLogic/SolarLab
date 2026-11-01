"""Thin physical-storage BDF adapter for the unified classical 1D assembly.

Equation storage has zero algebraic rows. The separate physical readers keep
the boundary-cell inventories needed by conservative interval accounting.
Every value and tangent comes from the bound assembler; no DC problem is
constructed and no storage secant is replaced by coordinate differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
from solarlab.equations.dc_1d import CompiledDevice1D
from solarlab.physics.contacts import CarrierExchangeChannel
from solarlab.physics.recombination.bulk import RecombinationRate
from solarlab.physics.transport.electrochemical_flux import CarrierFaceFluxes
from solarlab.solvers.nonlinear.problem import Matrix, Vector
from solarlab.solvers.time.bdf import StorageProblem

if TYPE_CHECKING:
    # Ion-free transients bind no ionic source; the ionic adapter imports these when used.
    from solarlab.equations.assembly.ionic_1d import AssembledIonic1DProblem, AssembledMultipleIonic1DProblem
    from solarlab.equations.ionic_1d import CompiledIonicDevice1D
    from solarlab.equations.multiple_ionic_1d import CompiledMultipleIonicDevice1D


@dataclass(frozen=True)
class AssembledStorage1D:
    problem: AssembledDC1DProblem

    def __post_init__(self) -> None:
        if not isinstance(self.problem, AssembledDC1DProblem):
            raise TypeError("assembled storage requires an AssembledDC1DProblem")

    def residual(self, u: Vector) -> Vector:
        return self.problem.residual(u)

    @property
    def device(self) -> CompiledDevice1D:
        return self.problem.device

    @property
    def voltage_V(self) -> float:
        return self.problem.voltage_V

    @property
    def generation_scale(self) -> float:
        return self.problem.generation_scale

    def jacobian(self, u: Vector) -> Matrix:
        return self.problem.jacobian(u)

    def storage(self, u: Vector) -> Vector:
        return self.problem.storage(u)

    def storage_tangent(self, u: Vector) -> Matrix:
        return self.problem.storage_tangent(u)

    def mass_matrix(self, u: Vector) -> Matrix:
        """The signed DAE mass M = -S_u, without a second convention."""
        return self.problem.storage_jacobian(u)

    def storage_jacobian(self, u: Vector) -> Matrix:
        """Historical BE name for the same signed mass."""
        return self.mass_matrix(u)

    def fluxes(self, u: Vector) -> CarrierFaceFluxes:
        return self.problem.fluxes(u)

    def recombination_rate(self, n: Vector, p: Vector, *, derivatives: bool = True) -> RecombinationRate:
        """Original BE observation, provided by Architecture's public facade."""
        return self.problem.recombination_rate(n, p, derivatives=derivatives)

    def contact_exchange(self, u: Vector, channel: CarrierExchangeChannel, node: int) -> tuple[float, float]:
        return self.problem.contact_exchange(u, channel, node)

    def _contact_exchange(self, u: Vector, channel: CarrierExchangeChannel, node: int) -> tuple[float, float]:
        """Map the unchanged BE private name to the assembled public name."""
        return self.contact_exchange(u, channel, node)

    def _previous(self, old_problem: StorageProblem) -> AssembledDC1DProblem:
        if not isinstance(old_problem, AssembledStorage1D):
            raise TypeError("assembled storage history requires the same adapter")
        old = old_problem.problem
        if old.binding.identity != self.problem.binding.identity:
            raise ValueError("assembled storage history requires the same bound system")
        return old

    def storage_increment(self, u: Vector, old_problem: StorageProblem, old_u: Vector) -> Vector:
        return self.problem.storage_increment(u, self._previous(old_problem), old_u)

    def physical_storage(self, u: Vector) -> Vector:
        return self.problem.binding.assembler.physical_storage(u, self.problem.inputs)

    def physical_increment(self, u: Vector, old_problem: StorageProblem, old_u: Vector) -> Vector:
        old = self._previous(old_problem)
        return self.problem.binding.assembler.storage_increment(
            u, self.problem.inputs, old_u, old.inputs, physical=True)


@dataclass(frozen=True)
class AssembledIonicStorage1D:
    """The same storage protocol for the original single/multiple-domain rows.

    S is masked on algebraic carrier rows. Physical readers include those
    endpoint-cell inventories for current and charge accounting. Independent
    and shared-site species retain their bound cancellation-free secants and
    domain weights; no domain is collapsed into a total ion inventory.
    """

    problem: AssembledIonic1DProblem | AssembledMultipleIonic1DProblem

    def __post_init__(self) -> None:
        from solarlab.equations.assembly.ionic_1d import AssembledIonic1DProblem, AssembledMultipleIonic1DProblem

        if not isinstance(self.problem, (AssembledIonic1DProblem, AssembledMultipleIonic1DProblem)):
            raise TypeError("ionic assembled storage requires original-row ionic assembly")

    @property
    def device(self) -> CompiledIonicDevice1D | CompiledMultipleIonicDevice1D:
        return self.problem.device

    @property
    def voltage_V(self) -> float:
        return self.problem.voltage_V

    @property
    def generation_scale(self) -> float:
        return self.problem.generation_scale

    def residual(self, u: Vector) -> Vector:
        return self.problem.residual(u)

    def jacobian(self, u: Vector) -> Matrix:
        return self.problem.jacobian(u)

    def storage(self, u: Vector) -> Vector:
        return self.problem.storage(u)

    def storage_tangent(self, u: Vector) -> Matrix:
        return self.problem.storage_tangent(u)

    def mass_matrix(self, u: Vector) -> Matrix:
        """The signed DAE mass M = -S_u, as in the electronic bridge."""
        return self.problem.storage_jacobian(u)

    def storage_jacobian(self, u: Vector) -> Matrix:
        return self.mass_matrix(u)

    def _previous(self, old_problem: StorageProblem) -> AssembledIonic1DProblem | AssembledMultipleIonic1DProblem:
        if not isinstance(old_problem, AssembledIonicStorage1D):
            raise TypeError("ionic assembled storage history requires the same adapter")
        old = old_problem.problem
        if type(old) is not type(self.problem) or old.binding.identity != self.problem.binding.identity:
            raise ValueError("ionic assembled history requires the same bound domain system")
        return old

    def storage_increment(self, u: Vector, old_problem: StorageProblem, old_u: Vector) -> Vector:
        old = self._previous(old_problem)
        return self.problem.binding.assembler.storage_increment(u, self.problem.inputs, old_u, old.inputs)

    def physical_storage(self, u: Vector) -> Vector:
        return self.problem.binding.assembler.physical_storage(u, self.problem.inputs)

    def physical_increment(self, u: Vector, old_problem: StorageProblem, old_u: Vector) -> Vector:
        old = self._previous(old_problem)
        return self.problem.binding.assembler.storage_increment(
            u, self.problem.inputs, old_u, old.inputs, physical=True)
