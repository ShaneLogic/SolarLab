"""Independent geometry oracle for P04-01 (control volumes, measures, ports).

Three historical geometries must be *named* by the new implementation and
must not be silently unified (V1 P04-01 step 1 and its failure clause).  The
definitions below are written from first principles and from the published
behaviour of the legacy engine; they are the oracle, the legacy code is only
the historical comparison.

``legacy_1d_integer_endpoint_v1``  (1D MoL, ``solver/mol.py``)
    Nodes include every material interface.  A node on an interface belongs
    to the RIGHT layer (``_layer_node_masks``, last-layer-wins).  Carrier
    control volumes are the dual cells with FULL-width end cells::

        w_0 = h_0,   w_{N-1} = h_{N-2},   w_i = (h_{i-1} + h_i)/2

    so the total measure is  L + (h_0 + h_{N-2})/2  (not L).  Node-owned
    material data (eps_r, rho, alpha, doping) therefore change at the dual
    face x_I - h_{I-1}/2, i.e. the EFFECTIVE material interface sits half an
    interval to the left of the geometric interface node x_I.

``legacy_2d_half_endpoint_v1``  (2D Stage A, ``twod/continuity_2d.py``)
    Vertical (stack) end cells are half width, ``hy_0 = h_0/2``; the total
    vertical measure is exactly L_y.  Lateral Neumann end cells are half
    width too (total L_x); lateral periodic end cells are
    (h_0 + h_last)/2 each, so the periodic lateral measure is
    L_x + (h_0 + h_last)/2 (the historical wrap face is an extra interval).

``r1_physical_boundary_two_sided_v1``  (R1 lane, ``physical_control_volume``)
    No node lies on an interface.  Outer cells are clipped at the contacts,
    interior faces are interval midpoints, and the face containing an
    interface plane is moved to the plane itself.  Total measure is exactly
    L and every material's measure is exactly its thickness.

Port conventions used by the oracle: a 1D device has unit cross-section
(port area 1 m^2); a 2D device has unit out-of-plane depth, so a contact's
port measure is its lateral length in m.  The outward normal of the left
(x=0) contact is -x and of the right contact +x.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GEOMETRY_LEGACY_1D = "legacy_1d_integer_endpoint_v1"
GEOMETRY_LEGACY_2D = "legacy_2d_half_endpoint_v1"
GEOMETRY_R1_TWO_SIDED = "r1_physical_boundary_two_sided_v1"
#: Fourth historical geometry named by W3a (review R3-2): the quasi-Fermi
#: ``two_sided_trace`` topology without physical boundary volumes.
GEOMETRY_QF_TWO_SIDED_TRACE = "qf_two_sided_trace_v1"
ALL_GEOMETRIES = (GEOMETRY_LEGACY_1D, GEOMETRY_LEGACY_2D, GEOMETRY_R1_TWO_SIDED)
ALL_GEOMETRIES_V2 = ALL_GEOMETRIES + (GEOMETRY_QF_TWO_SIDED_TRACE,)


def _require_grid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 3 or not np.all(np.diff(x) > 0.0):
        raise ValueError("grid must be strictly increasing with >= 3 nodes")
    return x


def integer_endpoint_widths(x: np.ndarray) -> np.ndarray:
    """``legacy_1d_integer_endpoint_v1`` carrier control-volume widths."""

    x = _require_grid(x)
    h = np.diff(x)
    w = np.empty(x.size)
    w[0] = h[0]
    w[-1] = h[-1]
    w[1:-1] = 0.5 * (h[:-1] + h[1:])
    return w


def half_endpoint_widths(x: np.ndarray) -> np.ndarray:
    """``legacy_2d_half_endpoint_v1`` vertical / lateral-Neumann widths."""

    x = _require_grid(x)
    h = np.diff(x)
    w = np.empty(x.size)
    w[0] = 0.5 * h[0]
    w[-1] = 0.5 * h[-1]
    w[1:-1] = 0.5 * (h[:-1] + h[1:])
    return w


def periodic_wrap_widths(x: np.ndarray) -> np.ndarray:
    """Historical 2D lateral periodic widths (wrap face = (h_0+h_last)/2)."""

    w = half_endpoint_widths(x)
    h = np.diff(np.asarray(x, dtype=float))
    w[0] = w[-1] = 0.5 * (h[0] + h[-1])
    return w


def two_sided_faces(x: np.ndarray, interfaces: tuple[float, ...]) -> np.ndarray:
    """``r1_physical_boundary_two_sided_v1`` cell faces (N+1 values)."""

    x = _require_grid(x)
    faces = np.concatenate(([x[0]], 0.5 * (x[:-1] + x[1:]), [x[-1]]))
    for plane in interfaces:
        right = int(np.searchsorted(x, plane))
        if right in (0, x.size) or x[right] == plane:
            raise ValueError("two-sided geometry needs every plane strictly inside a face")
        faces[right] = plane
    if not np.all(np.diff(faces) > 0.0):
        raise ValueError("non-positive two-sided control volume")
    return faces


def two_sided_widths(x: np.ndarray, interfaces: tuple[float, ...]) -> np.ndarray:
    return np.diff(two_sided_faces(x, interfaces))


def two_sided_trace_widths(x: np.ndarray, interfaces: tuple[float, ...]) -> np.ndarray:
    """``qf_two_sided_trace_v1``: integer-endpoint contact cells, plane-split interiors.

    No node lies on a plane; the two cells beside a plane end at the plane
    (widths h_{l-1}/2 + d_L and d_R + h_r/2), while both contact cells keep
    the full adjacent interval as in the 1D MoL.
    """

    x = _require_grid(x)
    faces = two_sided_faces(x, interfaces)
    w = np.diff(faces)
    h = np.diff(x)
    w[0] = h[0]
    w[-1] = h[-1]
    return w


def expected_total_measure(kind: str, x: np.ndarray) -> float:
    """Closed-form total carrier measure of each named geometry."""

    x = _require_grid(x)
    h = np.diff(x)
    length = float(x[-1] - x[0])
    if kind in (GEOMETRY_LEGACY_1D, GEOMETRY_QF_TWO_SIDED_TRACE):
        return length + 0.5 * float(h[0] + h[-1])
    if kind in (GEOMETRY_LEGACY_2D, GEOMETRY_R1_TWO_SIDED):
        return length
    raise ValueError(f"unknown geometry {kind!r}")


def exact_sum(values: np.ndarray) -> float:
    """Correctly rounded sum (``math.fsum``) used for every measure check."""

    return math.fsum(float(v) for v in np.asarray(values).ravel())


def right_owner_layer_index(x: np.ndarray, thicknesses: tuple[float, ...]) -> np.ndarray:
    """Node -> layer ownership of ``legacy_1d_integer_endpoint_v1``.

    Interfaces are nodes; the node on a shared boundary goes to the right
    layer.  A relative roundoff allowance of 1e-12 of the stack thickness is
    used to recognise an interface node.
    """

    x = _require_grid(x)
    bounds = np.concatenate(([0.0], np.cumsum(thicknesses)))
    tol = 1e-12 * float(bounds[-1])
    owner = np.full(x.size, -1, dtype=int)
    for k in range(len(thicknesses)):
        owner[(x >= bounds[k] - tol) & (x <= bounds[k + 1] + tol)] = k
    if np.any(owner < 0):
        raise ValueError("grid extends outside the stack")
    return owner


def effective_interface_positions_integer_endpoint(
    x: np.ndarray, thicknesses: tuple[float, ...]
) -> tuple[float, ...]:
    """Effective material planes x_I - h_{I-1}/2 of the legacy 1D geometry."""

    x = _require_grid(x)
    planes = []
    for position in np.cumsum(thicknesses)[:-1]:
        node = int(np.argmin(np.abs(x - position)))
        if abs(x[node] - position) > 1e-12 * float(x[-1]):
            raise ValueError("legacy 1D geometry requires interface nodes")
        planes.append(0.5 * (x[node - 1] + x[node]))
    return tuple(planes)


# --------------------------------------------------------------------------
# Substrate-prefix interface alignment (V1 P04-01 acceptance)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ElectricalAlignment:
    """Map from the full layer list to the electrical layer list."""

    substrate_prefix: int
    electrical_layer_indices: tuple[int, ...]
    electrical_interface_full_indices: tuple[int, ...]
    electrical_interface_positions_m: tuple[float, ...]


def electrical_alignment(
    roles: tuple[str, ...], thicknesses: tuple[float, ...]
) -> ElectricalAlignment:
    """Reference alignment: substrates are an optical-only contiguous prefix.

    Full interface k separates full layers k and k+1.  The electrical
    interface j separates electrical layers j and j+1, i.e. full interface
    ``prefix + j``.  Positions are measured from the first electrical layer
    (the electrical grid origin), excluding the substrate thickness.
    """

    if len(roles) != len(thicknesses):
        raise ValueError("roles and thicknesses must align")
    prefix = 0
    for role in roles:
        if role != "substrate":
            break
        prefix += 1
    if any(role == "substrate" for role in roles[prefix:]):
        raise ValueError("substrate layers must form a contiguous prefix")
    electrical = tuple(range(prefix, len(roles)))
    n_iface = max(0, len(electrical) - 1)
    positions = tuple(
        math.fsum(thicknesses[prefix : prefix + j + 1]) for j in range(n_iface)
    )
    return ElectricalAlignment(
        substrate_prefix=prefix,
        electrical_layer_indices=electrical,
        electrical_interface_full_indices=tuple(prefix + j for j in range(n_iface)),
        electrical_interface_positions_m=positions,
    )


def port_outward_particle_flux(face_flux: np.ndarray, port_area_m2: float) -> tuple[float, float]:
    """Outward particle current through (left, right) ports for +x face fluxes."""

    face_flux = np.asarray(face_flux, dtype=float)
    return (-float(face_flux[0]) * port_area_m2, float(face_flux[-1]) * port_area_m2)
