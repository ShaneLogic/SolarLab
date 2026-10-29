"""The existing rectangular 2D face graph as one assembler support.

Nodes flatten ``(Ny, Nx)`` with x fastest; every x face precedes every y
face. The two stack contact strips have their physical x half-cell widths
as port measures. Exterior lateral fluxes remain the declared zero Neumann
condition: no exterior x faces or additional contact law are introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from solarlab.contracts.units import LENGTH
from solarlab.equations.assembly.assembler import Mesh, Port, SupportGeometry
from solarlab.mesh.geometry_2d import RectangularGeometry2D

__all__ = ["RectangularSupport2D"]

Index: TypeAlias = NDArray[np.intp]
PORTS = ("y_min", "y_max")


@dataclass(frozen=True)
class RectangularSupport2D:
    """One full connected support, with the historical ordered-port stencils."""

    geometry: RectangularGeometry2D
    node_set: str = "nodes"
    mesh: Mesh = field(init=False, repr=False)

    def __post_init__(self) -> None:
        g = self.geometry
        sides = g.contact_nodes.reshape(2, g.nx)
        ports = tuple(Port(name, tuple(int(node) for node in nodes), normal,
                           g.x_widths_m, LENGTH)
                      for name, nodes, normal in zip(PORTS, sides, (-1, 1)))
        object.__setattr__(self, "mesh", Mesh((SupportGeometry(
            self.node_set, g.contract_geometry, ports),)))

    def normal_faces(self, port_id: str) -> Index:
        """One adjacent y face per port node, in the port's x-fast node order."""
        if port_id not in PORTS:
            raise ValueError("2D contact readout requires y_min or y_max")
        g = self.geometry
        first = g.x_face_count + (0 if port_id == "y_min" else (g.ny - 2) * g.nx)
        result: Index = np.arange(first, first + g.nx, dtype=np.intp)
        result.setflags(write=False)
        return result

    @property
    def transverse_faces(self) -> Index:
        """Every x face in the original order, including zero lateral inflow."""
        result: Index = np.arange(self.geometry.x_face_count, dtype=np.intp)
        result.setflags(write=False)
        return result
