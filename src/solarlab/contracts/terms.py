"""Typed unweighted contributions; geometry is applied exactly once by TermSink.

Constitutive models hand over volumetric sources and face fluxes that are not
multiplied by any measure. TermSink multiplies cell sources by the cell
measure and face fluxes by the face measure, and owns the divergence sign:
a face oriented from ``pairs[:, 0]`` to ``pairs[:, 1]`` removes its weighted
flux from the first cell and adds it to the second (net inflow).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts.arithmetic import AssemblyArithmetic, FloatArithmetic
from solarlab.contracts.arrays import (
    ImmutableArrays, PhysicalArray, Vector, frozen_array, integer_indices,
)
from solarlab.contracts.errors import ContractError
from solarlab.contracts.layout import Layout
from solarlab.contracts.units import AREA, VOLUME, Unit

__all__ = [
    "AlgebraicTerm", "CellSource", "Contribution", "FaceFlux", "Geometry",
    "SurfaceCharge", "TermSink",
]


@dataclass(frozen=True)
class Geometry(ImmutableArrays):
    """Control-volume measures and oriented face connectivity."""

    volumes: Vector
    pairs: NDArray[np.int64]
    face_measures: Vector
    volume_unit: Unit = VOLUME
    face_unit: Unit = AREA
    cell_support: str = "cells"
    face_support: str = "faces"

    def __post_init__(self) -> None:
        object.__setattr__(self, "volumes", frozen_array(self.volumes))
        object.__setattr__(self, "face_measures", frozen_array(self.face_measures))
        pairs = integer_indices(self.pairs)
        immutable_pairs = np.frombuffer(pairs.tobytes(), dtype=np.int64).reshape(pairs.shape)
        object.__setattr__(self, "pairs", immutable_pairs)
        if self.volumes.ndim != 1 or np.any(self.volumes <= 0):
            raise ContractError("invalid_volumes")
        if pairs.ndim != 2 or pairs.shape != (self.face_measures.size, 2):
            raise ContractError("invalid_face_pairs")
        if self.face_measures.ndim != 1 or np.any(self.face_measures <= 0):
            raise ContractError("invalid_face_measures")
        if (np.any(pairs < 0) or np.any(pairs >= self.volumes.size)
                or np.any(pairs[:, 0] == pairs[:, 1])):
            raise ContractError("face_outside_support")
        if not self.cell_support or not self.face_support or self.cell_support == self.face_support:
            raise ContractError("invalid_geometry_support")


@dataclass(frozen=True)
class Contribution:
    id: str
    support: str
    values: ArrayLike | PhysicalArray
    unit: Unit


@dataclass(frozen=True)
class CellSource(Contribution):
    """Volumetric source density; TermSink multiplies by the cell measure."""


@dataclass(frozen=True)
class FaceFlux(Contribution):
    """Oriented flux density on faces; TermSink multiplies by the face measure."""


@dataclass(frozen=True)
class AlgebraicTerm(Contribution):
    """Already-integrated constraint value; never weighted by geometry."""


@dataclass(frozen=True)
class SurfaceCharge(Contribution):
    """Areal density on a face-supported equation; weighted by the face measure."""


class TermSink:
    """Accumulate typed contributions for one equation of a layout."""

    def __init__(self, layout: Layout, equation_id: str, geometry: Geometry,
                 arithmetic: AssemblyArithmetic | None = None):
        equations = {x.id: x for x in layout.equations}
        if equation_id not in equations:
            raise ContractError("unknown_equation")
        self.equation = equations[equation_id]
        self.supports = {x.id: x for x in layout.supports}
        self.geometry = geometry
        self.arithmetic: AssemblyArithmetic = (
            arithmetic if arithmetic is not None else FloatArithmetic())
        for support_id, location, shape in (
            (geometry.cell_support, "cell", geometry.volumes.shape),
            (geometry.face_support, "face", geometry.face_measures.shape),
        ):
            support = self.supports.get(support_id)
            if support is None or support.location != location or support.shape != shape:
                raise ContractError("geometry_support_mismatch")
        self._seen: set[str] = set()
        self._value = self.arithmetic.zeros(self.equation.shape)

    def add(self, term: Contribution) -> None:
        if not term.id or term.id in self._seen:
            raise ContractError("duplicate_contribution")
        if term.support not in self.supports:
            raise ContractError("unknown_support")
        support = self.supports[term.support]
        target = self.supports[self.equation.support]
        arithmetic = self.arithmetic
        values = arithmetic.array(term.values)
        if values.shape != support.shape:
            raise ContractError("term_shape_mismatch")
        geometry = self.geometry
        contribution: Any
        if isinstance(term, CellSource):
            if support != target or support.location != "cell" or values.shape != geometry.volumes.shape:
                raise ContractError("wrong_source_support")
            if support.id != geometry.cell_support:
                raise ContractError("geometry_support_mismatch")
            expected = term.unit * geometry.volume_unit
            contribution = arithmetic.weighted(values, geometry.volumes)
        elif isinstance(term, FaceFlux):
            if (support.location != "face" or target.location != "cell"
                    or values.shape != geometry.face_measures.shape
                    or target.shape != geometry.volumes.shape):
                raise ContractError("wrong_flux_support")
            if support.id != geometry.face_support or target.id != geometry.cell_support:
                raise ContractError("geometry_support_mismatch")
            expected = term.unit * geometry.face_unit
            contribution = arithmetic.zeros(self.equation.shape)
            weighted = arithmetic.weighted(values, geometry.face_measures)
            contribution = arithmetic.scatter_add(
                contribution, geometry.pairs[:, 0],
                arithmetic.weighted(weighted, -np.ones(values.shape)),
            )
            contribution = arithmetic.scatter_add(contribution, geometry.pairs[:, 1], weighted)
        elif isinstance(term, AlgebraicTerm):
            if support != target or self.equation.role != "constraint":
                raise ContractError("wrong_algebraic_support")
            expected, contribution = term.unit, values
        elif isinstance(term, SurfaceCharge):
            if (support != target or support.location != "face"
                    or values.shape != geometry.face_measures.shape):
                raise ContractError("wrong_surface_charge_support")
            if support.id != geometry.face_support:
                raise ContractError("geometry_support_mismatch")
            expected = term.unit * geometry.face_unit
            contribution = arithmetic.weighted(values, geometry.face_measures)
        else:
            raise ContractError("untyped_contribution")
        if expected != self.equation.unit:
            raise ContractError("unit_mismatch")
        self._value = arithmetic.add(self._value, contribution)
        self._seen.add(term.id)

    def value(self) -> Any:
        return self.arithmetic.freeze(self._value)
