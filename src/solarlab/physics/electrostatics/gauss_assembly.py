"""Gauss law assembled through the public TermSink contract (N class).

The same algebraic Poisson residual as
:meth:`~solarlab.physics.electrostatics.poisson.PoissonOperator1D.interior_residual`,
but the constitutive displacement and charge density are handed to
:class:`~solarlab.contracts.TermSink` unweighted, which applies the cell and face
measures exactly once. With unit face measures every term is the historical
product and IEEE addition is commutative, so the rows are bitwise equal to the
historical residual and contact displacements on every finite state whose row
is not an exact zero. The one difference is the sign of an exact zero: the
contract accumulator starts from ``+0.0``, so a row whose historical value is
``-0.0`` comes back ``+0.0``. That byte difference is why this stays N class.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts import (
    AREA, COULOMB, VOLUME, CellSource, EquationSpec, FaceFlux, Geometry, Layout, Support,
    TermSink,
)
from solarlab.mesh.geometry import ControlVolumeGeometry1D
from solarlab.physics.electrostatics.poisson import PoissonOperator1D

__all__ = ["GAUSS_EQUATION", "gauss_balance_with_term_sink"]

GAUSS_EQUATION = "electrostatics.gauss"


def gauss_balance_with_term_sink(operator: PoissonOperator1D,
                                 geometry: ControlVolumeGeometry1D,
                                 phi: ArrayLike, rho_C_m3: ArrayLike) -> NDArray[np.float64]:
    """Assemble ``sum(inflow D) + w*rho`` on every cell through the public TermSink.

    The constitutive displacement and charge density are handed over unweighted;
    TermSink applies the cell/face measures once. Interior rows equal
    :meth:`PoissonOperator1D.interior_residual` bitwise up to the sign of an
    exact zero; the two contact rows are ``-D(left contact)`` and
    ``+D(right contact)``.
    """
    if geometry.node_count != operator.node_count:
        raise ValueError("geometry and operator must share the nodes")
    n = geometry.node_count
    layout = Layout(
        (Support("cells", "cell", (n,)), Support("faces", "face", (n - 1,))),
        (),
        (EquationSpec(GAUSS_EQUATION, "electrostatics", "cells", (n,), COULOMB / AREA,
                      role="constraint"),),
    )
    contract_geometry: Geometry = geometry.to_contract_geometry()
    sink = TermSink(layout, GAUSS_EQUATION, contract_geometry)
    sink.add(FaceFlux("electrostatics.displacement", "faces",
                      operator.face_displacement(phi), COULOMB / AREA))
    sink.add(CellSource("electrostatics.space_charge", "cells",
                        np.asarray(rho_C_m3, dtype=float), COULOMB / VOLUME))
    balance: NDArray[np.float64] = sink.value()
    return balance
