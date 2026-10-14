"""Route-independent public numerical contracts (P02-02 subset).

Only interfaces that do not depend on IDA storage or coordinate choices are
implemented here. Interfaces waiting on the G2-R route decision are listed in
:mod:`solarlab.contracts.pending` and fail closed when used.
"""

from solarlab.contracts.arithmetic import AssemblyArithmetic, FloatArithmetic
from solarlab.contracts.arrays import (
    FloatArray, ImmutableArrays, PhysicalArray, Vector, frozen_array, integer_indices,
)
from solarlab.contracts.errors import ContractError
from solarlab.contracts.layout import EquationSpec, Layout, Support, VariableSpec
from solarlab.contracts.pending import PENDING_G2R
from solarlab.contracts.ports import TerminalPort
from solarlab.contracts.state import (
    FieldProjection, LinearCoordinates, Point, ResolvedAuthority, StateAuthority,
    StateIncrement, StateView,
)
from solarlab.contracts.terms import (
    AlgebraicTerm, CellSource, Contribution, FaceFlux, Geometry, SurfaceCharge, TermSink,
)
from solarlab.contracts.units import (
    AREA, COULOMB, ELEMENTARY_CHARGE_C, KELVIN, LENGTH, ONE, PARTICLE, SECOND, VOLT,
    VOLUME, Unit,
)

__all__ = [
    "AREA", "AlgebraicTerm", "AssemblyArithmetic", "COULOMB", "CellSource",
    "ContractError", "Contribution", "ELEMENTARY_CHARGE_C", "EquationSpec", "FaceFlux", "FieldProjection",
    "FloatArithmetic", "FloatArray", "Geometry", "ImmutableArrays", "KELVIN", "LENGTH",
    "Layout", "LinearCoordinates", "ONE", "PARTICLE", "PENDING_G2R", "PhysicalArray",
    "Point", "ResolvedAuthority", "SECOND", "StateAuthority", "StateIncrement",
    "StateView", "Support", "SurfaceCharge", "TermSink", "TerminalPort", "Unit",
    "VOLT", "VOLUME", "VariableSpec", "Vector", "frozen_array", "integer_indices",
]
