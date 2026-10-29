"""Term protocol: typed contributions with exact local tangents.

A Term is one bound constitutive contribution. It declares the node set it
lives on (``support``), the equation blocks whose rows it feeds, the blocks
it differentiates, and the unit of its contribution per unit measure. Its
``evaluate`` returns, per equation, either

* :class:`FaceValue` (``kind="face_flux"``): an oriented flux on the faces of
  the support, from ``pairs[:, 0]`` to ``pairs[:, 1]``, with its tangents with
  respect to each dependency at the face's left and right nodes; or
* :class:`LocalValue` (every other kind): a value on support nodes with its
  tangents with respect to each dependency at the same node.

Kinds: ``face_flux`` (net inflow into cells), ``cell_source`` (volumetric
source), ``port_flux`` (flux along the support axis through one declared
port), ``constraint`` (already-integrated algebraic rows that replace the
rows declared by :class:`~solarlab.equations.assembly.layout.ConstraintRows`),
and ``storage`` (physical storage density ``S`` per unit measure).

Geometry is applied exactly once, by the Assembler. A contribution is either
an unweighted array, which the Assembler sums with the other unweighted
contributions of the same slot and then multiplies by the measure once, or a
:mod:`~solarlab.equations.assembly.weighted` expression that contains the
measure symbol exactly once (to keep an existing product order). Constraint
values never contain a measure. A Term never scatters, never reads a
measure, and never writes another Term's rows.

Tangents are with respect to the layout coordinates of the dependency
blocks. A Term may compute them through the field map's chain factors or
supply a cancellation-free coordinate tangent directly (the SG quasi-Fermi
chain does).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping, Protocol, TypeAlias, Union

import numpy as np
from numpy.typing import NDArray

from solarlab.contracts.errors import ContractError
from solarlab.contracts.units import Unit
from solarlab.equations.assembly.layout import SystemLayout
from solarlab.equations.assembly.weighted import Weighted

__all__ = [
    "CELL_SOURCE", "CONSTRAINT", "Contribution", "EvaluationState", "FACE_FLUX",
    "FaceValue", "FieldMap", "FieldValues", "LocalValue", "OperatingInputs",
    "PORT_FLUX", "STORAGE", "StorageTerm", "TERM_KINDS", "Term", "TermValue",
]

FACE_FLUX = "face_flux"
CELL_SOURCE = "cell_source"
PORT_FLUX = "port_flux"
CONSTRAINT = "constraint"
STORAGE = "storage"
TERM_KINDS = (FACE_FLUX, CELL_SOURCE, PORT_FLUX, CONSTRAINT, STORAGE)

Vector: TypeAlias = NDArray[np.float64]
#: An unweighted array (per unit measure) or a weighted expression.
Contribution: TypeAlias = Union[Vector, Weighted]


def _frozen_mapping(value: Mapping[str, Contribution]) -> Mapping[str, Contribution]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class FaceValue:
    """Oriented face flux and its two-point tangents (empty without derivatives)."""

    flux: Contribution
    left: Mapping[str, Contribution] = field(default_factory=dict)
    right: Mapping[str, Contribution] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.left) != set(self.right):
            raise ContractError("face_tangents_need_both_nodes")
        object.__setattr__(self, "left", _frozen_mapping(self.left))
        object.__setattr__(self, "right", _frozen_mapping(self.right))


@dataclass(frozen=True)
class LocalValue:
    """Node values and same-node tangents (empty without derivatives)."""

    value: Contribution
    tangents: Mapping[str, Contribution] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tangents", _frozen_mapping(self.tangents))


TermValue: TypeAlias = Union[FaceValue, LocalValue]


@dataclass(frozen=True)
class OperatingInputs:
    """Prescribed operating drive of one evaluation (not part of the state)."""

    voltage_V: float = 0.0
    generation_scale: float = 0.0

    def __post_init__(self) -> None:
        voltage, generation = float(self.voltage_V), float(self.generation_scale)
        if not math.isfinite(voltage):
            raise ValueError("applied voltage must be finite")
        if not math.isfinite(generation) or generation < 0.0:
            raise ValueError("generation_scale must be finite and non-negative")
        object.__setattr__(self, "voltage_V", voltage)
        object.__setattr__(self, "generation_scale", generation)


@dataclass(frozen=True)
class FieldValues:
    """Physical fields of one coordinate vector; ``admissible`` gates evaluation."""

    fields: Mapping[str, Vector]
    admissible: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


class FieldMap(Protocol):
    """Coordinate map: global coordinates and inputs to named physical fields."""

    @property
    def identity(self) -> str: ...

    def evaluate(self, layout: SystemLayout, u: Vector, inputs: OperatingInputs) -> FieldValues: ...


@dataclass(frozen=True)
class EvaluationState:
    """One coordinate vector, its inputs and its fields, shared by every Term."""

    layout: SystemLayout
    u: Vector
    inputs: OperatingInputs
    fields: Mapping[str, Vector]

    def block(self, block_id: str) -> Vector:
        return self.layout.view(self.u, block_id)

    def field(self, name: str) -> Vector:
        value = self.fields.get(name)
        if value is None:
            raise ContractError("unknown_field")
        return value


class Term(Protocol):
    """A bound contribution; see the module docstring for the kinds."""

    @property
    def id(self) -> str: ...

    @property
    def kind(self) -> str: ...

    @property
    def support(self) -> str: ...

    @property
    def equations(self) -> tuple[str, ...]: ...

    @property
    def dependencies(self) -> tuple[str, ...]: ...

    @property
    def unit(self) -> Unit: ...

    @property
    def port(self) -> str | None: ...

    @property
    def identity(self) -> str: ...

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]: ...


class StorageTerm(Term, Protocol):
    """A storage Term also supplies the cancellation-free physical increment.

    ``increment`` returns ``S(state) - S(previous)`` per unit measure (or as a
    weighted expression), computed without subtracting two inventories.
    """

    def increment(self, state: EvaluationState, previous: EvaluationState) -> Mapping[str, Contribution]: ...
