"""Smooth CT on the existing2D normal-face field, with exact SG chain terms.

x faces use -(phi_R-phi_L)/dx and y faces use -(phi_R-phi_L)/dy.
The vector field norm is a different model. BaseD is supplied by the existing
compiler; empirical node parameters retain the legacy arithmetic face means.
PF and nonsmooth CT are outside this first native binding, as in new-core1D.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from numpy.typing import ArrayLike

from solarlab.contracts import frozen_array
from solarlab.mesh.geometry_2d import RECTANGULAR_HALF_ENDPOINT_2D_V1, RectangularGeometry2D
from solarlab.physics.transport.electrochemical_flux import CarrierFaceFlux, CarrierFaceFluxes, stable_sg_particle_flux
from solarlab.physics.transport.field_mobility import FieldMobilityFaceParameters, linearize_field_mobility
from solarlab.physics.transport.field_mobility_tangents import (
    FACE_NORMAL_COMPONENT_2D, FieldMobilityCapabilityError, require_field_measure,
)


@dataclass(frozen=True)
class FieldTransportEvaluation2D:
    field_V_m: np.ndarray
    electron_diffusivity_m2_s: np.ndarray
    hole_diffusivity_m2_s: np.ndarray
    electron_d_diffusivity_d_field: np.ndarray
    hole_d_diffusivity_d_field: np.ndarray


@dataclass(frozen=True)
class FieldTransport2D:
    """Geometry-bound physical inputs; every evaluation uses the actual phi."""

    geometry: RectangularGeometry2D
    base_electron_diffusivity_m2_s: np.ndarray
    base_hole_diffusivity_m2_s: np.ndarray
    thermal_voltage_V: float
    electron: FieldMobilityFaceParameters
    hole: FieldMobilityFaceParameters
    field_measure: str = FACE_NORMAL_COMPONENT_2D

    def __post_init__(self) -> None:
        require_field_measure(self.field_measure)
        if self.field_measure != FACE_NORMAL_COMPONENT_2D:
            raise FieldMobilityCapabilityError("2D CT requires the declared normal-face-component field")
        if not isinstance(self.geometry, RectangularGeometry2D) or self.geometry.name != RECTANGULAR_HALF_ENDPOINT_2D_V1:
            raise FieldMobilityCapabilityError("2D CT requires the nonperiodic half-endpoint face geometry")
        if not np.isfinite(self.thermal_voltage_V) or self.thermal_voltage_V <= 0:
            raise ValueError("field transport requires positive finite thermal voltage")
        count = self.geometry.face_spacing_m.size
        for name in ("base_electron_diffusivity_m2_s", "base_hole_diffusivity_m2_s"):
            value = frozen_array(getattr(self, name))
            if value.shape != (count,) or np.any(value <= 0):
                raise FieldMobilityCapabilityError("2D CT requires positive physicalD on the full face set")
            object.__setattr__(self, name, value)
        for name in ("electron", "hole"):
            carrier = getattr(self, name)
            if carrier.saturation_velocity_m_s.shape != (count,):
                raise ValueError("CT parameters must match all oriented2D faces")
            velocity, beta = carrier.saturation_velocity_m_s, carrier.ct_exponent
            if np.any(carrier.pf_gamma_sqrt_m_V != 0):
                raise FieldMobilityCapabilityError("PF is not bound in the smoothCT2D slice")
            if np.any(velocity < 0) or np.any((velocity > 0) & (beta <= 1)):
                raise FieldMobilityCapabilityError("smoothCT2D requires beta>1 on every active face")
            bound = FieldMobilityFaceParameters(carrier.saturation_velocity_m_s, carrier.ct_exponent,
                                                carrier.pf_gamma_sqrt_m_V)
            for field_name in ("saturation_velocity_m_s", "ct_exponent", "pf_gamma_sqrt_m_V"):
                object.__setattr__(bound, field_name, frozen_array(getattr(bound, field_name)))
            object.__setattr__(self, name, bound)

    @property
    def identity_sha256(self) -> str:
        digest = hashlib.sha256(json.dumps(("smooth_ct_normal_face_transport_2d_v1", self.geometry.sha256,
            self.field_measure, self.thermal_voltage_V), separators=(",", ":"), allow_nan=False).encode())
        arrays = [self.base_electron_diffusivity_m2_s, self.base_hole_diffusivity_m2_s]
        for carrier in (self.electron, self.hole):
            arrays.extend((carrier.saturation_velocity_m_s, carrier.ct_exponent, carrier.pf_gamma_sqrt_m_V))
        for value in arrays:
            digest.update(np.ascontiguousarray(value, dtype="<f8").tobytes())
        return digest.hexdigest()

    def _nodes(self, value: ArrayLike, name: str) -> np.ndarray:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError(name + " must be real")
        nodes = np.asarray(raw, dtype=float)
        if nodes.shape != (self.geometry.node_count,) or not np.all(np.isfinite(nodes)):
            raise ValueError(name + " must be finite on the complete2D node set")
        return nodes

    def evaluate(self, phi_V: ArrayLike) -> FieldTransportEvaluation2D:
        phi = self._nodes(phi_V, "electrostatic potential")
        left, right = self.geometry.contract_geometry.pairs.T
        field = -(phi[right] - phi[left]) / self.geometry.face_spacing_m
        values, derivatives = [], []
        for base, carrier in ((self.base_electron_diffusivity_m2_s, self.electron),
                              (self.base_hole_diffusivity_m2_s, self.hole)):
            linear = linearize_field_mobility(base / self.thermal_voltage_V, field,
                carrier.saturation_velocity_m_s, carrier.ct_exponent, carrier.pf_gamma_sqrt_m_V)
            if not np.all(linear.differentiable):
                raise FieldMobilityCapabilityError("normal-faceCT encountered an unsupported nonsmooth tangent")
            value = linear.mobility_m2_V_s * self.thermal_voltage_V
            slope = linear.field_derivative_m3_V2_s * self.thermal_voltage_V
            if not np.all(np.isfinite(value)) or np.any(value <= 0) or not np.all(np.isfinite(slope)):
                raise FieldMobilityCapabilityError("2D CT diffusivity/tangent is outside the finite positive domain")
            values.append(frozen_array(value))
            derivatives.append(frozen_array(slope))
        return FieldTransportEvaluation2D(frozen_array(field), values[0], values[1], derivatives[0], derivatives[1])

    def fluxes(self, phi_V: ArrayLike, electron_density_m3: ArrayLike, hole_density_m3: ArrayLike,
               electron_transport_potential_V: ArrayLike, hole_transport_potential_V: ArrayLike, *,
               electron_electrochemical_step: ArrayLike | None = None,
               hole_electrochemical_step: ArrayLike | None = None) -> CarrierFaceFluxes:
        """Reuse SG density/energy derivatives and add dF/dk*dD/dphi/h."""
        evaluation = self.evaluate(phi_V)
        left, right = self.geometry.contract_geometry.pairs.T
        h = self.geometry.face_spacing_m
        fluxes = []
        for density, potential, diffusivity, slope, sign, drive in (
            (electron_density_m3, electron_transport_potential_V, evaluation.electron_diffusivity_m2_s,
             evaluation.electron_d_diffusivity_d_field, -1.0, electron_electrochemical_step),
            (hole_density_m3, hole_transport_potential_V, evaluation.hole_diffusivity_m2_s,
             evaluation.hole_d_diffusivity_d_field, 1.0, hole_electrochemical_step),
        ):
            c, psi = self._nodes(density, "carrier density"), self._nodes(potential, "transport potential")
            if np.any(c <= 0):
                raise ValueError("SG carrier densities must be positive")
            step = sign * (psi[right] - psi[left]) / self.thermal_voltage_V
            coefficient = diffusivity / h
            evaluated = stable_sg_particle_flux(c[left], c[right], step, coefficient, electrochemical_step=drive)
            per_volt = sign / self.thermal_voltage_V * evaluated.energy_step_derivative
            chain = evaluated.coefficient_derivative * slope / h**2
            fluxes.append(CarrierFaceFlux(evaluated.flux, evaluated.left_density_derivative,
                evaluated.right_density_derivative, -per_volt + chain, per_volt - chain, step, coefficient))
        return CarrierFaceFluxes(fluxes[0], fluxes[1])


def bind_field_transport_2d(geometry: RectangularGeometry2D, electron_diffusivity_faces_m2_s: ArrayLike,
                            hole_diffusivity_faces_m2_s: ArrayLike, thermal_voltage_V: float, *,
                            v_sat_n: ArrayLike, v_sat_p: ArrayLike, ct_beta_n: ArrayLike, ct_beta_p: ArrayLike,
                            pf_gamma_n: ArrayLike, pf_gamma_p: ArrayLike,
                            field_measure: str = FACE_NORMAL_COMPONENT_2D) -> FieldTransport2D | None:
    """Bind original painted node parameters; off returnsNone without changingD."""
    require_field_measure(field_measure)
    if field_measure != FACE_NORMAL_COMPONENT_2D:
        raise FieldMobilityCapabilityError("2D binding requires normal-face components, not total field")
    if not isinstance(geometry, RectangularGeometry2D):
        raise FieldMobilityCapabilityError("2D binding requires explicit nonperiodic physical geometry")
    arrays = [np.asarray(value, dtype=float) for value in
        (v_sat_n, v_sat_p, ct_beta_n, ct_beta_p, pf_gamma_n, pf_gamma_p)]
    if any(value.shape != (geometry.node_count,) or not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("CT node parameters must be finite on the full2D node set")
    if np.any(arrays[4] != 0) or np.any(arrays[5] != 0):
        raise FieldMobilityCapabilityError("PF is not bound in the smoothCT2D slice")
    if np.any(arrays[0] < 0) or np.any(arrays[1] < 0):
        raise FieldMobilityCapabilityError("saturation velocities must be nonnegative")
    if not np.any(arrays[0] > 0) and not np.any(arrays[1] > 0):
        return None
    left, right = geometry.contract_geometry.pairs.T
    faces = [0.5 * (value[left] + value[right]) for value in arrays]
    electron = FieldMobilityFaceParameters(faces[0], faces[2], faces[4])
    hole = FieldMobilityFaceParameters(faces[1], faces[3], faces[5])
    return FieldTransport2D(geometry, np.asarray(electron_diffusivity_faces_m2_s),
        np.asarray(hole_diffusivity_faces_m2_s), thermal_voltage_V, electron, hole, field_measure)
