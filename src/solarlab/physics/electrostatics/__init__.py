"""Electrostatics: shared charge ledger, Poisson residual and interface traces.

The namespace exports the M-class (legacy bit-identical) electrostatics. The
TermSink assembly of the same Gauss law (N class) is
:mod:`solarlab.physics.electrostatics.gauss_assembly`.
"""

from solarlab.physics.electrostatics.charge import (
    SPACE_CHARGE_GROUP, ChargeEntry, ChargeLedger, SheetChargeEntry,
)
from solarlab.physics.electrostatics.interface_traces import (
    InterfaceTraceDeclaration, TraceElectrostatics, add_sheet_charge_to_interior_residual,
    electrostatic_trace_residual_and_jacobian, interface_trace_layout, require_carrier_trace,
    require_occupancy, sheet_charge_row_weights, solve_electrostatic_traces,
)
from solarlab.physics.electrostatics.poisson import (
    PoissonFactorization, PoissonOperator1D, contact_displacement,
)

__all__ = [
    "ChargeEntry", "ChargeLedger", "InterfaceTraceDeclaration",
    "PoissonFactorization", "PoissonOperator1D", "SPACE_CHARGE_GROUP", "SheetChargeEntry",
    "TraceElectrostatics", "add_sheet_charge_to_interior_residual", "contact_displacement",
    "electrostatic_trace_residual_and_jacobian",
    "interface_trace_layout", "require_carrier_trace", "require_occupancy",
    "sheet_charge_row_weights", "solve_electrostatic_traces",
]
