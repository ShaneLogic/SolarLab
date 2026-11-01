"""Neutral finite-width grain SRH on the historical 2D control volumes.

A partial cell mixes *rates*, not lifetimes: R += f (R_grain - R_bulk_SRH).
The identical fraction multiplies both exact density partials. Grain row
ownership is LEFT_LAYER_WINS; the physical coefficients remain right-owned.
No surface rate, trap charge, extra state or mobility law is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math

import numpy as np
from numpy.typing import NDArray

from solarlab.contracts import frozen_array
from solarlab.device.inputs import GrainBoundaryInput
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.two_dimensional.inputs import JV2DInput
from solarlab.mesh.geometry_2d import RectangularGeometry2D
from solarlab.mesh.grid import LEFT_LAYER_WINS, node_owner
from solarlab.physics.recombination.bulk import RecombinationRate, lifetime_srh_recombination
from solarlab.physics.recombination.reaction import (
    BulkRecombinationCoefficients, BulkRecombinationEvaluation, RecombinationDomainError,
)
from solarlab.units import normalize_quantity

GRAIN_SRH_2D_V1 = "neutral_grain_fractional_lifetime_srh_left_layer_2d_v1"


def grain_declarations(prepared: PreparedDevice, spatial: JV2DInput) -> tuple[GrainBoundaryInput, ...]:
    """Resolve the existing inherit/explicit-clear contract and reject ambiguity.

    Omitted, null and {} inherit the device inventory; an explicitly supplied
    grain_boundaries array overrides it, including the empty array. layer_ids
    are the selectors; source_layer_role is retained import provenance only.
    """
    override = spatial.microstructure
    grains = (override.grain_boundaries if override is not None and override.editing_data()
              else prepared.to_input().grain_boundaries)
    if not grains:
        return ()
    if spatial.lateral_bc != "neumann" or spatial.lateral_length is None:
        raise ValueError("grain regions require explicit Neumann lateral geometry")
    length = normalize_quantity(spatial.lateral_length, "m")
    layer_ids = {layer.id for layer in prepared.layers if layer.role != "substrate"}
    seen: set[str] = set()
    bands: list[tuple[float, float, set[str]]] = []
    for grain in grains:
        if grain.id in seen:
            raise ValueError("duplicate grain-boundary ID: " + grain.id)
        seen.add(grain.id)
        owners = set(grain.layer_ids)
        if not owners or len(owners) != len(grain.layer_ids) or not owners <= layer_ids:
            raise ValueError("grain regions require unique existing electrical layer_ids")
        x, width, tau_n, tau_p = (normalize_quantity(getattr(grain, name), unit)
                                for name, unit in (("x_position", "m"), ("width", "m"),
                                                   ("tau_n", "s"), ("tau_p", "s")))
        if not all(math.isfinite(value) for value in (x, width, tau_n, tau_p)) or min(width, tau_n, tau_p) <= 0:
            raise ValueError("grain width and lifetimes must be finite and positive")
        lo, hi = x - width / 2, x + width / 2
        if lo < 0 or hi > length or hi <= lo:
            raise ValueError("grain band must lie wholly inside the lateral domain; no clipping")
        if any(owners & other and min(hi, b) > max(lo, a) for a, b, other in bands):
            raise ValueError("overlapping grain bands in the same electrical layer are unsupported")
        bands.append((lo, hi, owners))
    return tuple(grains)


@dataclass(frozen=True)
class GrainRegion2D:
    declaration: GrainBoundaryInput
    x_overlap_fraction: NDArray[np.float64]
    y_mask: NDArray[np.bool_]
    fractions: NDArray[np.float64]
    represented_width_m: float
    represented_height_m: float
    represented_area_m2: float

    def __post_init__(self) -> None:
        for name in ("x_overlap_fraction", "fractions"):
            values = frozen_array(getattr(self, name))
            if np.any(values < 0) or np.any(values > 1):
                raise ValueError("grain control-volume fractions must lie in [0,1]")
            object.__setattr__(self, name, values)
        mask = np.asarray(self.y_mask, dtype=bool)
        object.__setattr__(self, "y_mask", np.frombuffer(mask.tobytes(), dtype=bool).reshape(mask.shape))
        if not np.array_equal(self.fractions, (mask[:, None] * self.x_overlap_fraction[None, :]).ravel()):
            raise ValueError("grain fractions disagree with their declared row and lateral ownership")


@dataclass(frozen=True)
class GrainRecombination2D:
    regions: tuple[GrainRegion2D, ...]
    geometry_sha256: str
    effective_origin: str
    model_id: str = field(init=False)

    def __post_init__(self) -> None:
        digest = hashlib.sha256(json.dumps((GRAIN_SRH_2D_V1, self.geometry_sha256, self.effective_origin,
            [region.declaration.normalized_data() for region in self.regions]), sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode())
        for region in self.regions:
            digest.update(region.fractions.astype("<f8").tobytes())
        object.__setattr__(self, "model_id", GRAIN_SRH_2D_V1 + ":" + digest.hexdigest())

    def evaluate(self, evaluation: BulkRecombinationEvaluation,
                 coefficients: BulkRecombinationCoefficients) -> RecombinationRate:
        """Correct only SRH; densities, ni², n1 and p1 stay physical node values."""
        if not self.regions:
            return evaluation.total
        if evaluation.explicit is not None or not np.all(evaluation.lifetime_nodes):
            raise ValueError("grain SRH requires lifetime bulk recombination, not explicit defects")
        evaluation.require_domain()
        result = np.array(evaluation.total.rate, copy=True)
        dn = None if not evaluation.total.has_derivatives else np.array(evaluation.total.d_rate_dn, copy=True)
        dp = None if not evaluation.total.has_derivatives else np.array(evaluation.total.d_rate_dp, copy=True)
        for region in self.regions:
            if region.fractions.shape != result.shape:
                raise ValueError("grain binding and reaction node sets differ")
            selected = region.fractions > 0
            f = region.fractions[selected]
            grain = lifetime_srh_recombination(
                evaluation.electron_density_m3[selected], evaluation.hole_density_m3[selected],
                coefficients.ni_sq_m6[selected], normalize_quantity(region.declaration.tau_n, "s"),
                normalize_quantity(region.declaration.tau_p, "s"), coefficients.n1_m3[selected],
                coefficients.p1_m3[selected], derivatives=dn is not None)
            result[selected] += f * (grain.rate - evaluation.srh.rate[selected])
            if dn is not None and dp is not None:
                assert grain.d_rate_dn is not None and grain.d_rate_dp is not None
                assert evaluation.srh.d_rate_dn is not None and evaluation.srh.d_rate_dp is not None
                dn[selected] += f * (grain.d_rate_dn - evaluation.srh.d_rate_dn[selected])
                dp[selected] += f * (grain.d_rate_dp - evaluation.srh.d_rate_dp[selected])
        for value in (result, dn, dp):
            if value is not None:
                if not np.all(np.isfinite(value)):
                    raise RecombinationDomainError("grain SRH rate or exact density partial is nonfinite")
                value.setflags(write=False)
        return RecombinationRate(result, dn, dp)


def bind_grain_regions_2d(prepared: PreparedDevice, spatial: JV2DInput,
                         geometry: RectangularGeometry2D) -> GrainRecombination2D | None:
    """Bind physical band intersections to the actual half-endpoint areas."""
    grains = grain_declarations(prepared, spatial)
    if not grains:
        return None
    if geometry.x_m[0] != 0 or geometry.lateral_length_m != normalize_quantity(spatial.lateral_length, "m"):
        raise ValueError("grain declarations and actual lateral domain differ")
    electrical = tuple(layer for layer in prepared.layers if layer.role != "substrate")
    owners = node_owner(geometry.y_geometry.nodes_m, [layer.thickness for layer in electrical], LEFT_LAYER_WINS)
    x = geometry.x_m
    bounds = np.r_[x[0], .5 * (x[:-1] + x[1:]), x[-1]]
    regions = []
    for grain in grains:
        center, width = normalize_quantity(grain.x_position, "m"), normalize_quantity(grain.width, "m")
        overlap = np.maximum(0.0, np.minimum(bounds[1:], center + width / 2)
                             - np.maximum(bounds[:-1], center - width / 2))
        fractions_x = overlap / np.diff(bounds)
        rows = np.isin(owners, [i for i, layer in enumerate(electrical) if layer.id in grain.layer_ids])
        fractions = (rows[:, None] * fractions_x[None, :]).ravel()
        if not np.any(fractions):
            raise ValueError("declared grain has no represented control-volume area")
        regions.append(GrainRegion2D(grain, fractions_x, rows, fractions,
            float(fractions_x @ geometry.x_widths_m), float(np.sum(geometry.y_geometry.cell_measures_m[rows])),
            float(fractions @ geometry.contract_geometry.volumes)))
    override = spatial.microstructure
    origin = "experiment.microstructure" if override is not None and override.editing_data() else "device.grain_boundaries"
    return GrainRecombination2D(tuple(regions), geometry.sha256, origin)
