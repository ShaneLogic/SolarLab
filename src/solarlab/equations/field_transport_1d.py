"""Smooth CT face transport and its electrostatic-potential chain rule."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from numpy.typing import ArrayLike

from solarlab.physics.transport.field_mobility import (
    FieldMobilityFaceParameters, face_electric_field, linearize_field_mobility,
)
from solarlab.physics.transport.field_mobility_tangents import (
    FACE_FIELD_1D, FieldMobilityCapabilityError, require_field_measure,
)


def _frozen(value: ArrayLike) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class FieldTransportEvaluation1D:
    field_V_m: np.ndarray
    electron_diffusivity_m2_s: np.ndarray
    hole_diffusivity_m2_s: np.ndarray
    electron_d_diffusivity_d_field: np.ndarray
    hole_d_diffusivity_d_field: np.ndarray


@dataclass(frozen=True)
class FieldTransport1D:
    """State-free physical inputs; effective diffusivities are never cached."""

    nodes_m: np.ndarray
    base_electron_diffusivity_m2_s: np.ndarray
    base_hole_diffusivity_m2_s: np.ndarray
    thermal_voltage_V: float
    electron: FieldMobilityFaceParameters
    hole: FieldMobilityFaceParameters
    field_measure: str = FACE_FIELD_1D

    def __post_init__(self) -> None:
        require_field_measure(self.field_measure)
        if self.field_measure != FACE_FIELD_1D:
            raise FieldMobilityCapabilityError("CT 1D requires the complete face_field_1d_v1 field")
        nodes = _frozen(self.nodes_m)
        if nodes.ndim != 1 or nodes.size < 2 or not np.all(np.isfinite(nodes)) or np.any(np.diff(nodes) <= 0.0):
            raise ValueError("field transport requires a finite strictly increasing node grid")
        object.__setattr__(self, "nodes_m", nodes)
        if not np.isfinite(self.thermal_voltage_V) or self.thermal_voltage_V <= 0.0:
            raise ValueError("field transport requires positive finite thermal voltage")
        for name in ("base_electron_diffusivity_m2_s", "base_hole_diffusivity_m2_s"):
            value = _frozen(getattr(self, name))
            if value.shape != (nodes.size - 1,) or not np.all(np.isfinite(value)) or np.any(value <= 0.0):
                raise FieldMobilityCapabilityError("smooth CT binding requires finite positive low-field face diffusivity")
            object.__setattr__(self, name, value)
        for carrier in (self.electron, self.hole):
            if carrier.saturation_velocity_m_s.shape != (nodes.size - 1,):
                raise ValueError("CT face parameters must match the physical grid")
            if np.any(carrier.pf_gamma_sqrt_m_V != 0.0):
                raise FieldMobilityCapabilityError("Poole-Frenkel transport is not bound in the smooth CT slice")
            velocity, beta = carrier.saturation_velocity_m_s, carrier.ct_exponent
            if np.any(velocity < 0.0) or np.any((velocity > 0.0) & (beta <= 1.0)):
                raise FieldMobilityCapabilityError("smooth CT requires beta > 1 on every active saturation face")

    @property
    def identity_sha256(self) -> str:
        header = ("smooth_ct_transport_1d_v1", self.field_measure, self.thermal_voltage_V)
        digest = hashlib.sha256(json.dumps(header, separators=(",", ":"), allow_nan=False).encode())
        arrays = [self.nodes_m, self.base_electron_diffusivity_m2_s, self.base_hole_diffusivity_m2_s]
        for carrier in (self.electron, self.hole):
            arrays.extend((carrier.saturation_velocity_m_s, carrier.ct_exponent, carrier.pf_gamma_sqrt_m_V))
        for array in arrays:
            digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
        return digest.hexdigest()

    def evaluate(self, phi_V: ArrayLike) -> FieldTransportEvaluation1D:
        phi = np.asarray(phi_V, dtype=float)
        if phi.shape != self.nodes_m.shape or not np.all(np.isfinite(phi)):
            raise ValueError("CT potential must be finite on the bound node grid")
        field = face_electric_field(phi, self.nodes_m)
        values = []
        derivatives = []
        for base, carrier in ((self.base_electron_diffusivity_m2_s, self.electron),
                              (self.base_hole_diffusivity_m2_s, self.hole)):
            linear = linearize_field_mobility(
                base / self.thermal_voltage_V, field,
                carrier.saturation_velocity_m_s, carrier.ct_exponent, carrier.pf_gamma_sqrt_m_V,
            )
            if not np.all(linear.differentiable):
                raise FieldMobilityCapabilityError("CT transport encountered an unsupported nonsmooth tangent")
            value = linear.mobility_m2_V_s * self.thermal_voltage_V
            derivative = linear.field_derivative_m3_V2_s * self.thermal_voltage_V
            if not np.all(np.isfinite(value)) or np.any(value <= 0.0) or not np.all(np.isfinite(derivative)):
                raise FieldMobilityCapabilityError("CT effective diffusivity/tangent is outside the finite positive domain")
            values.append(_frozen(value))
            derivatives.append(_frozen(derivative))
        return FieldTransportEvaluation1D(_frozen(field), values[0], values[1], derivatives[0], derivatives[1])


def bind_field_transport_1d(
    nodes_m: ArrayLike, electron_diffusivity_faces_m2_s: ArrayLike,
    hole_diffusivity_faces_m2_s: ArrayLike, thermal_voltage_V: float, *,
    v_sat_n: ArrayLike, v_sat_p: ArrayLike, ct_beta_n: ArrayLike, ct_beta_p: ArrayLike,
    pf_gamma_n: ArrayLike, pf_gamma_p: ArrayLike,
) -> FieldTransport1D | None:
    """Use original arithmetic face means of the compiler's painted node arrays.

    With saturation off, return None so the existing constant-D path (including
    its identity and arithmetic) remains exact.
    """
    count = np.asarray(nodes_m).size
    arrays = [np.asarray(value, dtype=float) for value in
              (v_sat_n, v_sat_p, ct_beta_n, ct_beta_p, pf_gamma_n, pf_gamma_p)]
    if any(value.shape != (count,) or not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("CT node parameters must be finite and match the physical node grid")
    if np.any(arrays[4] != 0.0) or np.any(arrays[5] != 0.0):
        raise FieldMobilityCapabilityError("Poole-Frenkel transport is not bound in the smooth CT slice")
    if np.any(arrays[0] < 0.0) or np.any(arrays[1] < 0.0):
        raise FieldMobilityCapabilityError("saturation velocities must be nonnegative")
    if not np.any(arrays[0] > 0.0) and not np.any(arrays[1] > 0.0):
        return None
    electron = FieldMobilityFaceParameters.from_nodes(arrays[0], arrays[2], arrays[4])
    hole = FieldMobilityFaceParameters.from_nodes(arrays[1], arrays[3], arrays[5])
    return FieldTransport1D(
        np.asarray(nodes_m), np.asarray(electron_diffusivity_faces_m2_s),
        np.asarray(hole_diffusivity_faces_m2_s), thermal_voltage_V, electron, hole,
    )
