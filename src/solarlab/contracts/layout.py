"""Named variables, equations and their supports, independent of registration order."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping

import numpy as np

from solarlab.contracts.errors import ContractError
from solarlab.contracts.units import Unit

__all__ = ["EquationSpec", "Layout", "Support", "VariableSpec"]


@dataclass(frozen=True)
class Support:
    id: str
    location: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.id or self.location not in {"cell", "face", "global"}:
            raise ContractError("invalid_support")
        if not self.shape or any(type(n) is not int or n < 0 for n in self.shape):
            raise ContractError("invalid_shape")


@dataclass(frozen=True)
class VariableSpec:
    id: str
    owner: str
    support: str
    shape: tuple[int, ...]
    unit: Unit
    role: str = "storage"
    orientation: int = 0
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class EquationSpec:
    id: str
    owner: str
    support: str
    shape: tuple[int, ...]
    unit: Unit
    role: str = "storage"
    orientation: int = 0
    derivative_support: tuple[str, ...] = ()


@dataclass(frozen=True)
class Layout:
    """Canonical (sorted) declarations, offsets and a content identity."""

    supports: tuple[Support, ...]
    variables: tuple[VariableSpec, ...]
    equations: tuple[EquationSpec, ...]
    offsets: Mapping[str, slice] = field(init=False)
    size: int = field(init=False)
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        supports = {x.id: x for x in self.supports}
        if len(supports) != len(self.supports):
            raise ContractError("duplicate_support")
        variables = tuple(sorted(self.variables, key=lambda x: x.id))
        equations = tuple(sorted(self.equations, key=lambda x: x.id))
        specs: tuple[VariableSpec | EquationSpec, ...]
        for specs in (variables, equations):
            if len({x.id for x in specs}) != len(specs):
                raise ContractError("duplicate_owner")
            for spec in specs:
                if not spec.id or not spec.owner:
                    raise ContractError("missing_instance_owner")
                if spec.support not in supports:
                    raise ContractError("unknown_support")
                if spec.shape != supports[spec.support].shape:
                    raise ContractError("support_shape_mismatch")
                if spec.orientation not in {-1, 0, 1}:
                    raise ContractError("invalid_orientation")
                if spec.role not in {"storage", "constraint", "prescribed"}:
                    raise ContractError("invalid_role")
        ids = {x.id for x in variables}
        for equation in equations:
            if not set(equation.derivative_support) <= ids:
                raise ContractError("unknown_derivative_support")
        for variable in variables:
            if (variable.lower is not None and variable.upper is not None
                    and variable.lower > variable.upper):
                raise ContractError("invalid_feasible_domain")
        offsets: dict[str, slice] = {}
        size = 0
        for variable in variables:
            end = size + int(np.prod(variable.shape))
            offsets[variable.id] = slice(size, end)
            size = end
        object.__setattr__(self, "supports", tuple(sorted(self.supports, key=lambda x: x.id)))
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "equations", equations)
        object.__setattr__(self, "offsets", MappingProxyType(offsets))
        object.__setattr__(self, "size", size)
        identity = repr((self.supports, variables, equations)).encode()
        object.__setattr__(self, "identity", sha256(identity).hexdigest())
