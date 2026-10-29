"""Unified layout, Term and assembly layer (milestone A, phase 1).

One :class:`~solarlab.equations.assembly.assembler.Assembler` turns declared
blocks (:mod:`~solarlab.equations.assembly.layout`) and bound Terms
(:mod:`~solarlab.equations.assembly.terms`) on a mesh into ``F``, ``J``,
``S`` and ``S_u`` for steady, storage-difference time and frequency
analyses. Bindings live in their own modules, for example
:mod:`~solarlab.equations.assembly.classical_1d`.
"""

from solarlab.equations.assembly.assembler import (
    AssembledSystem, Assembler, Mesh, Port, SupportGeometry,
)
from solarlab.equations.assembly.layout import Block, ConstraintRows, NodeSet, SystemLayout
from solarlab.equations.assembly.terms import (
    CELL_SOURCE, CONSTRAINT, FACE_FLUX, PORT_FLUX, STORAGE, EvaluationState, FaceValue,
    FieldMap, FieldValues, LocalValue, OperatingInputs, StorageTerm, Term,
)
from solarlab.equations.assembly.weighted import MEASURE, Weighted

__all__ = [
    "AssembledSystem", "Assembler", "Block", "CELL_SOURCE", "CONSTRAINT", "ConstraintRows",
    "EvaluationState", "FACE_FLUX", "FaceValue", "FieldMap", "FieldValues", "LocalValue",
    "MEASURE", "Mesh", "NodeSet", "OperatingInputs", "PORT_FLUX", "Port", "STORAGE",
    "StorageTerm", "SupportGeometry", "SystemLayout", "Term", "Weighted",
]
