"""Terminal-port sign and area conventions (route independent)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from solarlab.contracts.arithmetic import AssemblyArithmetic, FloatArithmetic
from solarlab.contracts.arrays import PhysicalArray
from solarlab.contracts.errors import ContractError
from solarlab.contracts.units import AREA, COULOMB, SECOND, Unit

__all__ = ["TerminalPort"]


@dataclass(frozen=True)
class TerminalPort:
    """An electrode face with its outward normal (-1 left, +1 right) and area.

    The electrode charge balances the displacement on its face,
    ``Q = -area*outward_normal*D``; the external current into the device is
    the conduction plus displacement current through that face.
    """

    id: str
    outward_normal: int
    area: float

    def __post_init__(self) -> None:
        if (not self.id or self.outward_normal not in {-1, 1}
                or not np.isfinite(self.area) or self.area <= 0):
            raise ContractError("invalid_port")

    def charge(self, displacement: ArrayLike | PhysicalArray, *,
               arithmetic: AssemblyArithmetic | None = None,
               unit: Unit = COULOMB / AREA) -> Any:
        if unit != COULOMB / AREA:
            raise ContractError("port_unit_mismatch")
        a = arithmetic if arithmetic is not None else FloatArithmetic()
        value = a.array(displacement)
        return a.freeze(a.weighted(value, np.full(value.shape, -self.area * self.outward_normal)))

    def current(self, conduction: ArrayLike | PhysicalArray,
                displacement_rate: ArrayLike | PhysicalArray, *,
                arithmetic: AssemblyArithmetic | None = None,
                conduction_unit: Unit = COULOMB / AREA / SECOND,
                displacement_rate_unit: Unit = COULOMB / AREA / SECOND) -> Any:
        if (conduction_unit != COULOMB / AREA / SECOND
                or displacement_rate_unit != conduction_unit):
            raise ContractError("port_unit_mismatch")
        a = arithmetic if arithmetic is not None else FloatArithmetic()
        jc, ddot = a.array(conduction), a.array(displacement_rate)
        if jc.shape != ddot.shape:
            raise ContractError("port_shape_mismatch")
        return a.freeze(a.weighted(a.add(jc, ddot),
                                   np.full(jc.shape, -self.area * self.outward_normal)))
