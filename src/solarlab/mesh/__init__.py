"""Mesh geometry: grids, layer alignment and named control-volume geometries."""

from solarlab.mesh.geometry import (
    GEOMETRY_NAMES, HALF_ENDPOINT_V1, INTEGER_ENDPOINT_V1, PHYSICAL_BOUNDARY_TWO_SIDED_V1,
    TWO_SIDED_TRACE_V1, ControlVolumeGeometry1D, InterfacePlane, PortFace, build_geometry,
)
from solarlab.mesh.grid import (
    LEFT_LAYER_WINS, RIGHT_LAYER_WINS, InterfaceStencil, build_interface_stencils,
    multilayer_grid, node_owner, remove_shared_interface_nodes, tanh_grid,
)
from solarlab.mesh.layers import ElectricalStack, StackLayer

__all__ = [
    "ControlVolumeGeometry1D", "ElectricalStack", "GEOMETRY_NAMES", "HALF_ENDPOINT_V1",
    "INTEGER_ENDPOINT_V1", "InterfacePlane", "InterfaceStencil", "LEFT_LAYER_WINS",
    "PHYSICAL_BOUNDARY_TWO_SIDED_V1", "PortFace", "RIGHT_LAYER_WINS", "StackLayer",
    "TWO_SIDED_TRACE_V1", "build_geometry", "build_interface_stencils", "multilayer_grid",
    "node_owner", "remove_shared_interface_nodes", "tanh_grid",
]
