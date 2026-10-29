"""Unified layout, Term and assembly layer (milestone A).

One :class:`~solarlab.equations.assembly.assembler.Assembler` turns declared
blocks (:mod:`~solarlab.equations.assembly.layout`) and bound Terms
(:mod:`~solarlab.equations.assembly.terms`) on a mesh into ``F``, ``J``,
``S`` and ``S_u`` for steady, storage-difference time and frequency
analyses. Bindings live in their own modules:
:mod:`~solarlab.equations.assembly.classical_1d` (electronic 1D) and
:mod:`~solarlab.equations.assembly.ionic_1d` (finite-site ions, single and
per-domain, with their steady closures).
"""

from solarlab.equations.assembly.assembler import (
    AssembledSystem, Assembler, Mesh, Port, SupportGeometry,
)
from solarlab.equations.assembly.layout import Block, ConstraintRows, NodeSet, SystemLayout
from solarlab.equations.assembly.terms import (
    CELL_SOURCE, CHAIN_DIVERGENCE, CONSTRAINT, FACE_FLUX, PORT_FLUX, SCATTER_DIVERGENCE, STORAGE,
    EvaluationState, FaceValue, FieldMap, FieldValues, LocalValue, OperatingInputs, ReductionValue,
    StorageTerm, Term,
)
from solarlab.equations.assembly.weighted import MEASURE, Weighted

__all__ = [
    "AssembledSystem", "Assembler", "Block", "CELL_SOURCE", "CHAIN_DIVERGENCE", "CONSTRAINT",
    "ConstraintRows", "EvaluationState", "FACE_FLUX", "FaceValue", "FieldMap", "FieldValues",
    "LocalValue", "MEASURE", "Mesh", "NodeSet", "OperatingInputs", "PORT_FLUX", "Port",
    "ReductionValue", "SCATTER_DIVERGENCE", "STORAGE", "StorageTerm", "SupportGeometry",
    "SystemLayout", "Term", "Weighted",
]
