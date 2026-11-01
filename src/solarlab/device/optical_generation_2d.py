"""Laterally uniform spectral optics on the actual2D half-endpoint cells.

The existing coherent1D TMM binding is evaluated on the y control volumes,
then extruded along x as in the legacy2D model. There is no lateral optical
propagation or density-dependent radiative recycling in this source. Optical
prefix absorption stays parasitic; the complete supplied spectrum is absolute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np

from solarlab.contracts import frozen_array
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.device.optical_generation import (
    PHOTON_BUDGET_RTOL, OpticalGeneration1D, bind_optical_generation_1d,
)
from solarlab.device.resolved import PreparedDevice
from solarlab.mesh.geometry_2d import RECTANGULAR_HALF_ENDPOINT_2D_V1, RectangularGeometry2D
from solarlab.physics.generation.profile import StaleGenerationProfileError

OPTICAL_BINDING_2D_V1 = "uniform_lateral_coherent_front_tmm_half_cells_2d_v1"


@dataclass(frozen=True)
class OpticalGeneration2D:
    """Fixed volumetric source and cell photons per unit out-of-plane depth.

    source_m3_s is flattened (Ny,Nx), x fastest, in particles/(m3*s).
    cell_generation_per_depth_m_s is photons/(m depth*s). The y profile
    already uses half-height contact cells; no endpoint factor is added here.
    """

    prepared_sha256: str
    geometry: RectangularGeometry2D
    optical_y: OpticalGeneration1D
    source_m3_s: np.ndarray = field(init=False)
    cell_generation_per_depth_m_s: np.ndarray = field(init=False)
    identity_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, RectangularGeometry2D) or self.geometry.name != RECTANGULAR_HALF_ENDPOINT_2D_V1:
            raise ValueError("optical2D requires the explicit nonperiodic half-endpoint geometry")
        if not isinstance(self.optical_y, OpticalGeneration1D):
            raise TypeError("optical2D requires the actual existing1D optical binding")
        if not self.prepared_sha256:
            raise ValueError("optical2D requires the exact prepared source identity")
        geometry, optical = self.geometry, self.optical_y
        profile, budget = optical.profile, optical.budget
        profile.require_inputs(optical.binding.sha256, geometry.y_geometry)
        source: np.ndarray = np.repeat(profile.source_m3_s, geometry.nx)
        photons = np.repeat(profile.absorbed_m2_s, geometry.nx) * np.tile(geometry.x_widths_m, geometry.ny)
        if source.shape != (geometry.node_count,) or not np.all(np.isfinite(source)) or np.any(source < 0.0):
            raise ValueError("optical2D source must be finite and nonnegative on every cell")
        assembled = source * geometry.contract_geometry.volumes
        if not np.allclose(assembled, photons, rtol=PHOTON_BUDGET_RTOL, atol=0.0):
            raise ValueError("optical2D cell source does not preserve each absorbed photon count")
        expected = budget.electrical_absorbed_m2_s * geometry.lateral_length_m
        tolerance = PHOTON_BUDGET_RTOL * budget.incident_m2_s * geometry.lateral_length_m
        if abs(float(np.sum(assembled)) - expected) > tolerance or abs(float(np.sum(photons)) - expected) > tolerance:
            raise ValueError("optical2D per-depth source does not close the electrical photon budget")
        object.__setattr__(self, "source_m3_s", frozen_array(source))
        object.__setattr__(self, "cell_generation_per_depth_m_s", frozen_array(photons))
        identity = (OPTICAL_BINDING_2D_V1, self.prepared_sha256, geometry.sha256,
                    optical.binding.sha256, profile.control_volume_sha256)
        object.__setattr__(self, "identity_sha256", hashlib.sha256(
            json.dumps(identity, separators=(",", ":")).encode()).hexdigest())

    @property
    def particle_generation_per_depth_m_s(self) -> float:
        """Actual integrated continuity source, particles/(m depth*s)."""
        return float(np.sum(self.source_m3_s * self.geometry.contract_geometry.volumes))

    @property
    def generation_current_limit_A_m(self) -> float:
        return Q * self.particle_generation_per_depth_m_s

    @property
    def generation_current_limit_A_m2(self) -> float:
        return self.generation_current_limit_A_m / self.geometry.lateral_length_m

    def require_inputs(self, prepared: PreparedDevice, geometry: RectangularGeometry2D) -> None:
        """Admission is source/mesh equality; it never reruns optics or Newton."""
        if prepared.content_sha256 != self.prepared_sha256:
            raise StaleGenerationProfileError("optical2D source was computed from a different prepared device")
        if geometry.sha256 != self.geometry.sha256:
            raise StaleGenerationProfileError("optical2D source was computed on different2D control volumes")
        self.optical_y.profile.require_inputs(self.optical_y.binding.sha256, geometry.y_geometry)


def bind_optical_generation_2d(prepared: PreparedDevice, geometry: RectangularGeometry2D) -> OpticalGeneration2D:
    """Reuse coherent normal/front1D optics, on the physical y half cells.

    The1D binding preserves its spectrum/resource, n-k support, optical prefix,
    grading/incoherent refusals and original1e-12 photon accounting checks.
    This function constructs a2D source; it does not admit an electronic model.
    """
    if not isinstance(prepared, PreparedDevice) or not isinstance(geometry, RectangularGeometry2D):
        raise TypeError("optical2D requires PreparedDevice and RectangularGeometry2D")
    optical = bind_optical_generation_1d(prepared, geometry.y_geometry)
    return OpticalGeneration2D(prepared.content_sha256, geometry, optical)
