"""Nonlocal radiative reabsorption kernels, photon budgets, and switching tangents.

Separates local escape-scaled radiative recombination from nonlocal internal
emission-reabsorption on global emitter-receiver supports:

* ``CONTROL_VOLUME_CONSERVATIVE_V1`` integrates net radiative emission with
  declared emitter control-volume measures ``w_emit`` and redistributes the
  trapped photon flux ``(1 - P_esc) * R_tot`` over ``receiver_nodes`` with a
  normalized profile ``psi_recv`` satisfying ``sum(w_recv * psi_recv) == 1``,
  closing ``emitted == reabsorbed + escaped`` to floating-point round-off on
  arbitrary non-uniform meshes.
* ``HISTORICAL_TRAPEZOID_UNIFORM_V1`` preserves the legacy 1D MoL operator:
  composite-trapezoid emitter quadrature over ``x_emitter`` divided uniformly
  by the declared layer thickness ``thickness_m`` on ``receiver_nodes``.
* ``LAGGED_BAKED_GENERATION_V1`` evaluates ``G_rad`` at an explicitly supplied
  frozen state ``(n_lagged, p_lagged)`` with zero live state derivatives,
  labeling operator splitting instead of masquerading as self-consistent.

Analytical derivatives use the exact rank-1 outer-product structure
``dG_rad[receiver] / d(n, p)[emitter] = u_receiver (x) v_emitter`` and declare
non-differentiability across the clamped ``R_tot = 0`` dark-equilibrium surface,
the thermionic ``min(|J_SG|, |J_TE|)`` crossover, and Poole-Frenkel / low-beta
Caughey-Thomas field-mobility cusps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts.arrays import frozen_array, integer_indices
from solarlab.physics.generation.photon_recycling import (
    PhotonRecyclingCapabilityError,
    PhotonRecyclingError,
)
from solarlab.physics.transport.field_mobility_tangents import (
    FieldMobilityDirectionalDerivative,
    field_mobility_directional_derivative,
)

__all__ = [
    "CONTROL_VOLUME_CONSERVATIVE_V1",
    "HISTORICAL_TRAPEZOID_UNIFORM_V1",
    "LAGGED_BAKED_GENERATION_V1",
    "REABSORPTION_MODES",
    "AbsorberRank1Factor",
    "AbsorberReabsorptionSpec",
    "PhotonRecyclingBudget",
    "ReabsorptionDirectionalDerivative",
    "ReabsorptionEvaluation",
    "SwitchingSurfaceAssessment",
    "ThermionicCapDirectionalDerivative",
    "assess_switching_surfaces",
    "dense_reabsorption_jacobian",
    "evaluate_radiative_reabsorption",
    "radiative_reabsorption_directional_derivative",
    "require_smooth_ac_surfaces",
    "thermionic_cap_directional_derivative",
    "thermionic_cap_value",
    "trapezoid_quadrature_weights_1d",
]

Vector: TypeAlias = NDArray[np.float64]
IndexVector: TypeAlias = NDArray[np.intp]
Vector2D: TypeAlias = NDArray[np.float64]

CONTROL_VOLUME_CONSERVATIVE_V1 = "control_volume_conservative_v1"
HISTORICAL_TRAPEZOID_UNIFORM_V1 = "historical_trapezoid_uniform_v1"
LAGGED_BAKED_GENERATION_V1 = "lagged_baked_generation_v1"
REABSORPTION_MODES: tuple[str, ...] = (
    CONTROL_VOLUME_CONSERVATIVE_V1,
    HISTORICAL_TRAPEZOID_UNIFORM_V1,
    LAGGED_BAKED_GENERATION_V1,
)

_DEFAULT_EMISSION_SWITCH_TOL_M2_S = 1.0e-30


def trapezoid_quadrature_weights_1d(x_m: ArrayLike) -> Vector:
    """Composite-trapezoid nodal weights ``w`` [m] satisfying ``w . y = trapz(y, x)``."""
    coords = np.asarray(x_m, dtype=float)
    if coords.ndim != 1 or coords.size < 2 or not np.all(np.isfinite(coords)):
        raise PhotonRecyclingError("trapezoid quadrature requires >= 2 finite 1D coordinates")
    dx = np.diff(coords)
    if np.any(dx <= 0.0):
        raise PhotonRecyclingError("trapezoid quadrature coordinates must be strictly increasing")
    weights = np.zeros_like(coords)
    weights[:-1] += 0.5 * dx
    weights[1:] += 0.5 * dx
    return frozen_array(weights)


@dataclass(frozen=True)
class AbsorberReabsorptionSpec:
    """Nonlocal emission-reabsorption specification for one emitter-receiver pair."""

    layer_id: str
    mode: str
    node_count: int
    emitter_nodes: IndexVector
    receiver_nodes: IndexVector
    emitter_weights_m: Vector
    receiver_profile_m1: Vector
    receiver_weights_m: Vector
    escape_probability: float
    intrinsic_radiative_B_m3_s: Vector
    intrinsic_product_m6: Vector
    emitter_coordinates_m: Vector | None = None
    lagged_electron_density_m3: Vector | None = None
    lagged_hole_density_m3: Vector | None = None
    clamp_negative_net_emission: bool = True
    switching_tolerance_m2_s: float = _DEFAULT_EMISSION_SWITCH_TOL_M2_S
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.layer_id, str) or not self.layer_id.strip():
            raise PhotonRecyclingError("AbsorberReabsorptionSpec requires a non-empty layer_id")
        if self.mode not in REABSORPTION_MODES:
            raise PhotonRecyclingCapabilityError(f"unsupported reabsorption mode {self.mode!r}")
        if type(self.node_count) is not int or self.node_count < 2:
            raise PhotonRecyclingError("node_count must be an integer >= 2")

        emit = integer_indices(self.emitter_nodes)
        recv = integer_indices(self.receiver_nodes)
        if emit.ndim != 1 or emit.size < 1 or recv.ndim != 1 or recv.size < 1:
            raise PhotonRecyclingError("emitter_nodes and receiver_nodes must be non-empty 1D index arrays")
        if (
            np.any(emit < 0)
            or np.any(emit >= self.node_count)
            or np.any(recv < 0)
            or np.any(recv >= self.node_count)
        ):
            raise PhotonRecyclingError("emitter_nodes and receiver_nodes must lie in [0, node_count)")
        if len(set(emit.tolist())) != emit.size or len(set(recv.tolist())) != recv.size:
            raise PhotonRecyclingError("emitter_nodes and receiver_nodes must not contain duplicates")

        w_emit = frozen_array(self.emitter_weights_m)
        psi_recv = frozen_array(self.receiver_profile_m1)
        w_recv = frozen_array(self.receiver_weights_m)
        b_rad = frozen_array(self.intrinsic_radiative_B_m3_s)
        ni_sq = frozen_array(self.intrinsic_product_m6)

        if w_emit.shape != emit.shape or np.any(w_emit <= 0.0):
            raise PhotonRecyclingError("emitter_weights_m must be finite positive on emitter_nodes")
        if psi_recv.shape != recv.shape or np.any(psi_recv < 0.0):
            raise PhotonRecyclingError("receiver_profile_m1 must be finite non-negative on receiver_nodes")
        if w_recv.shape != recv.shape or np.any(w_recv <= 0.0):
            raise PhotonRecyclingError("receiver_weights_m must be finite positive on receiver_nodes")
        if b_rad.shape != emit.shape or np.any(b_rad < 0.0):
            raise PhotonRecyclingError("intrinsic_radiative_B_m3_s must be finite non-negative on emitter_nodes")
        if ni_sq.shape != emit.shape or np.any(ni_sq < 0.0):
            raise PhotonRecyclingError("intrinsic_product_m6 must be finite non-negative on emitter_nodes")

        p_esc = float(self.escape_probability)
        if not math.isfinite(p_esc) or p_esc < 0.0 or p_esc > 1.0:
            raise PhotonRecyclingError(f"escape_probability must lie in [0, 1], got {self.escape_probability!r}")
        tol = float(self.switching_tolerance_m2_s)
        if not math.isfinite(tol) or tol < 0.0:
            raise PhotonRecyclingError("switching_tolerance_m2_s must be finite and non-negative")

        if self.mode == CONTROL_VOLUME_CONSERVATIVE_V1:
            norm = float(np.dot(w_recv, psi_recv))
            if not math.isclose(norm, 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
                raise PhotonRecyclingError(
                    f"{self.layer_id}: conservative receiver profile must satisfy sum(w_recv * psi_recv) == 1, got {norm}"
                )

        coords: Vector | None = None
        if self.emitter_coordinates_m is not None:
            coords = frozen_array(self.emitter_coordinates_m)
            if coords.shape != emit.shape or coords.size < 2 or np.any(np.diff(coords) <= 0.0):
                raise PhotonRecyclingError(
                    "emitter_coordinates_m must be strictly increasing and match emitter_nodes"
                )
        elif self.mode == HISTORICAL_TRAPEZOID_UNIFORM_V1:
            raise PhotonRecyclingError(
                "HISTORICAL_TRAPEZOID_UNIFORM_V1 requires emitter_coordinates_m"
            )

        n_lag: Vector | None = None
        p_lag: Vector | None = None
        if self.mode == LAGGED_BAKED_GENERATION_V1:
            if self.lagged_electron_density_m3 is None or self.lagged_hole_density_m3 is None:
                raise PhotonRecyclingError(
                    "LAGGED_BAKED_GENERATION_V1 requires explicit lagged_electron_density_m3 and lagged_hole_density_m3"
                )
            n_lag = frozen_array(self.lagged_electron_density_m3)
            p_lag = frozen_array(self.lagged_hole_density_m3)
            if n_lag.shape != (self.node_count,) or p_lag.shape != (self.node_count,):
                raise PhotonRecyclingError("lagged densities must have shape (node_count,)")
            if np.any(n_lag <= 0.0) or np.any(p_lag <= 0.0):
                raise PhotonRecyclingError("lagged densities must be strictly positive")
        elif self.lagged_electron_density_m3 is not None or self.lagged_hole_density_m3 is not None:
            raise PhotonRecyclingError(
                f"lagged densities are only valid for {LAGGED_BAKED_GENERATION_V1}"
            )

        object.__setattr__(self, "emitter_nodes", emit)
        object.__setattr__(self, "receiver_nodes", recv)
        object.__setattr__(self, "emitter_weights_m", w_emit)
        object.__setattr__(self, "receiver_profile_m1", psi_recv)
        object.__setattr__(self, "receiver_weights_m", w_recv)
        object.__setattr__(self, "escape_probability", p_esc)
        object.__setattr__(self, "intrinsic_radiative_B_m3_s", b_rad)
        object.__setattr__(self, "intrinsic_product_m6", ni_sq)
        object.__setattr__(self, "emitter_coordinates_m", coords)
        object.__setattr__(self, "lagged_electron_density_m3", n_lag)
        object.__setattr__(self, "lagged_hole_density_m3", p_lag)
        object.__setattr__(self, "switching_tolerance_m2_s", tol)

        digest = hashlib.sha256()
        meta = {
            "schema": "solarlab.radiative_reabsorption.spec.v1",
            "layer_id": self.layer_id,
            "mode": self.mode,
            "node_count": self.node_count,
            "emitter_nodes": emit.tolist(),
            "receiver_nodes": recv.tolist(),
            "escape_probability": repr(p_esc),
            "clamp_negative_net_emission": bool(self.clamp_negative_net_emission),
            "switching_tolerance_m2_s": repr(tol),
        }
        digest.update(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("ascii"))
        for arr in (w_emit, psi_recv, w_recv, b_rad, ni_sq):
            digest.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
        if coords is not None:
            digest.update(np.ascontiguousarray(coords, dtype="<f8").tobytes())
        if n_lag is not None and p_lag is not None:
            digest.update(np.ascontiguousarray(n_lag, dtype="<f8").tobytes())
            digest.update(np.ascontiguousarray(p_lag, dtype="<f8").tobytes())
        object.__setattr__(self, "sha256", digest.hexdigest())

    @classmethod
    def control_volume_conservative(
        cls,
        *,
        layer_id: str,
        node_count: int,
        emitter_nodes: ArrayLike,
        cell_measures_m: ArrayLike,
        escape_probability: float,
        intrinsic_radiative_B_m3_s: ArrayLike,
        intrinsic_product_m6: ArrayLike,
        receiver_nodes: ArrayLike | None = None,
        receiver_shape: ArrayLike | None = None,
        clamp_negative_net_emission: bool = True,
        switching_tolerance_m2_s: float = _DEFAULT_EMISSION_SWITCH_TOL_M2_S,
    ) -> AbsorberReabsorptionSpec:
        """Build a control-volume photon-conserving global emitter-receiver spec."""
        measures = frozen_array(cell_measures_m)
        if measures.shape != (node_count,) or np.any(measures <= 0.0):
            raise PhotonRecyclingError("cell_measures_m must be finite positive of shape (node_count,)")
        emit = integer_indices(emitter_nodes)
        recv = emit if receiver_nodes is None else integer_indices(receiver_nodes)
        w_emit = measures[emit]
        w_recv = measures[recv]
        raw_shape = np.ones(recv.size, dtype=float) if receiver_shape is None else np.asarray(receiver_shape, dtype=float)
        if raw_shape.shape != recv.shape or not np.all(np.isfinite(raw_shape)) or np.any(raw_shape < 0.0):
            raise PhotonRecyclingError("receiver_shape must be finite non-negative on receiver_nodes")
        denom = float(np.dot(w_recv, raw_shape))
        if denom <= 0.0:
            raise PhotonRecyclingError("receiver_shape must have positive integral over receiver_nodes")
        psi_recv = raw_shape / denom
        b_full = np.asarray(intrinsic_radiative_B_m3_s, dtype=float)
        ni_full = np.asarray(intrinsic_product_m6, dtype=float)
        b_emit = b_full[emit] if b_full.shape == (node_count,) else b_full
        ni_emit = ni_full[emit] if ni_full.shape == (node_count,) else ni_full
        return cls(
            layer_id=layer_id,
            mode=CONTROL_VOLUME_CONSERVATIVE_V1,
            node_count=node_count,
            emitter_nodes=emit,
            receiver_nodes=recv,
            emitter_weights_m=w_emit,
            receiver_profile_m1=psi_recv,
            receiver_weights_m=w_recv,
            escape_probability=escape_probability,
            intrinsic_radiative_B_m3_s=b_emit,
            intrinsic_product_m6=ni_emit,
            clamp_negative_net_emission=clamp_negative_net_emission,
            switching_tolerance_m2_s=switching_tolerance_m2_s,
        )

    @classmethod
    def historical_trapezoid_uniform(
        cls,
        *,
        layer_id: str,
        nodes_m: ArrayLike,
        absorber_mask: ArrayLike,
        thickness_m: float,
        escape_probability: float,
        intrinsic_radiative_B_m3_s: ArrayLike,
        intrinsic_product_m6: ArrayLike,
        clamp_negative_net_emission: bool = True,
        switching_tolerance_m2_s: float = _DEFAULT_EMISSION_SWITCH_TOL_M2_S,
    ) -> AbsorberReabsorptionSpec:
        """Build the historical 1D MoL trapezoid-emission / uniform-thickness spec."""
        coords = frozen_array(nodes_m)
        mask = np.asarray(absorber_mask, dtype=bool)
        if coords.ndim != 1 or mask.shape != coords.shape:
            raise PhotonRecyclingError("absorber_mask must match 1D nodes_m")
        idx = np.flatnonzero(mask)
        if idx.size < 2:
            raise PhotonRecyclingError("historical trapezoid reabsorption requires >= 2 absorber nodes")
        width = float(thickness_m)
        if not math.isfinite(width) or width <= 0.0:
            raise PhotonRecyclingError("thickness_m must be finite and positive")
        x_emit = coords[idx]
        w_trap = trapezoid_quadrature_weights_1d(x_emit)
        psi_uniform: Vector = np.full(idx.size, 1.0 / width, dtype=float)
        b_full = np.asarray(intrinsic_radiative_B_m3_s, dtype=float)
        ni_full = np.asarray(intrinsic_product_m6, dtype=float)
        b_emit = b_full[idx] if b_full.shape == coords.shape else b_full
        ni_emit = ni_full[idx] if ni_full.shape == coords.shape else ni_full
        return cls(
            layer_id=layer_id,
            mode=HISTORICAL_TRAPEZOID_UNIFORM_V1,
            node_count=coords.size,
            emitter_nodes=idx,
            receiver_nodes=idx,
            emitter_weights_m=w_trap,
            receiver_profile_m1=psi_uniform,
            receiver_weights_m=w_trap,
            escape_probability=escape_probability,
            intrinsic_radiative_B_m3_s=b_emit,
            intrinsic_product_m6=ni_emit,
            emitter_coordinates_m=x_emit,
            clamp_negative_net_emission=clamp_negative_net_emission,
            switching_tolerance_m2_s=switching_tolerance_m2_s,
        )

    @classmethod
    def lagged_baked(
        cls,
        base_spec: AbsorberReabsorptionSpec,
        lagged_electron_density_m3: ArrayLike,
        lagged_hole_density_m3: ArrayLike,
    ) -> AbsorberReabsorptionSpec:
        """Freeze a reabsorption spec at an explicit lagged carrier-density pair."""
        return cls(
            layer_id=base_spec.layer_id,
            mode=LAGGED_BAKED_GENERATION_V1,
            node_count=base_spec.node_count,
            emitter_nodes=base_spec.emitter_nodes,
            receiver_nodes=base_spec.receiver_nodes,
            emitter_weights_m=base_spec.emitter_weights_m,
            receiver_profile_m1=base_spec.receiver_profile_m1,
            receiver_weights_m=base_spec.receiver_weights_m,
            escape_probability=base_spec.escape_probability,
            intrinsic_radiative_B_m3_s=base_spec.intrinsic_radiative_B_m3_s,
            intrinsic_product_m6=base_spec.intrinsic_product_m6,
            emitter_coordinates_m=base_spec.emitter_coordinates_m,
            lagged_electron_density_m3=lagged_electron_density_m3,
            lagged_hole_density_m3=lagged_hole_density_m3,
            clamp_negative_net_emission=base_spec.clamp_negative_net_emission,
            switching_tolerance_m2_s=base_spec.switching_tolerance_m2_s,
        )


@dataclass(frozen=True)
class AbsorberRank1Factor:
    """Rank-1 analytical derivative factor ``u_receiver (x) v_emitter`` for one absorber."""

    layer_id: str
    mode: str
    emitter_nodes: IndexVector
    receiver_nodes: IndexVector
    receiver_vector_m1: Vector
    d_emission_flux_dn_m_s: Vector
    d_emission_flux_dp_m_s: Vector
    active: bool
    differentiable: bool
    raw_emitted_flux_m2_s: float
    clamped_emitted_flux_m2_s: float


@dataclass(frozen=True)
class PhotonRecyclingBudget:
    """Per-absorber internal photon emission, reabsorption, and escape ledger."""

    layer_id: str
    mode: str
    escape_probability: float
    raw_emitted_flux_m2_s: float
    emitted_flux_m2_s: float
    reabsorbed_flux_m2_s: float
    escaped_flux_m2_s: float
    closure_residual_m2_s: float


@dataclass(frozen=True)
class ReabsorptionEvaluation:
    """Evaluated nonlocal ``G_rad`` [m^-3 s^-1], low-rank factors, and photon ledger."""

    generation_m3_s: Vector
    factors: tuple[AbsorberRank1Factor, ...]
    budgets: tuple[PhotonRecyclingBudget, ...]
    differentiable: bool
    non_smooth_layers: tuple[str, ...]

    @property
    def total_emitted_flux_m2_s(self) -> float:
        return float(sum(item.emitted_flux_m2_s for item in self.budgets))

    @property
    def total_reabsorbed_flux_m2_s(self) -> float:
        return float(sum(item.reabsorbed_flux_m2_s for item in self.budgets))

    @property
    def total_escaped_flux_m2_s(self) -> float:
        return float(sum(item.escaped_flux_m2_s for item in self.budgets))

    @property
    def max_budget_residual_m2_s(self) -> float:
        if not self.budgets:
            return 0.0
        return float(max(abs(item.closure_residual_m2_s) for item in self.budgets))


def _raw_emission_flux(spec: AbsorberReabsorptionSpec, n_eval: Vector, p_eval: Vector) -> tuple[Vector, float]:
    emit = spec.emitter_nodes
    net_rate = spec.intrinsic_radiative_B_m3_s * (
        n_eval[emit] * p_eval[emit] - spec.intrinsic_product_m6
    )
    if spec.emitter_coordinates_m is not None:
        raw_flux = float(np.trapezoid(net_rate, spec.emitter_coordinates_m))
    else:
        raw_flux = float(np.dot(spec.emitter_weights_m, net_rate))
    return net_rate, raw_flux


def _effective_switching_tolerance_m2_s(spec: AbsorberReabsorptionSpec) -> float:
    thermal_flux_scale = float(
        np.dot(
            spec.emitter_weights_m,
            spec.intrinsic_radiative_B_m3_s * spec.intrinsic_product_m6,
        )
    )
    return max(spec.switching_tolerance_m2_s, 1.0e-11 * thermal_flux_scale)


def evaluate_radiative_reabsorption(
    specs: Sequence[AbsorberReabsorptionSpec],
    electron_density_m3: ArrayLike,
    hole_density_m3: ArrayLike,
    *,
    derivatives: bool = True,
) -> ReabsorptionEvaluation:
    """Evaluate nonlocal radiative reabsorption ``G_rad`` and exact rank-1 tangents."""
    n = np.asarray(electron_density_m3, dtype=float)
    p = np.asarray(hole_density_m3, dtype=float)
    if n.ndim != 1 or p.shape != n.shape or not (np.all(np.isfinite(n)) and np.all(np.isfinite(p))):
        raise PhotonRecyclingError("carrier densities must be finite 1D arrays of equal length")
    node_count = n.size
    generation = np.zeros(node_count, dtype=float)
    factors: list[AbsorberRank1Factor] = []
    budgets: list[PhotonRecyclingBudget] = []
    non_smooth: list[str] = []

    for spec in specs:
        if spec.node_count != node_count:
            raise PhotonRecyclingError(
                f"{spec.layer_id}: spec node_count {spec.node_count} != density length {node_count}"
            )
        if spec.mode == LAGGED_BAKED_GENERATION_V1:
            assert spec.lagged_electron_density_m3 is not None
            assert spec.lagged_hole_density_m3 is not None
            n_eval = spec.lagged_electron_density_m3
            p_eval = spec.lagged_hole_density_m3
        else:
            n_eval = n
            p_eval = p

        _, raw_flux = _raw_emission_flux(spec, n_eval, p_eval)
        trapped_fraction = 1.0 - spec.escape_probability
        has_trapping = trapped_fraction > 0.0 and bool(np.any(spec.intrinsic_radiative_B_m3_s > 0.0))

        if spec.clamp_negative_net_emission:
            active = raw_flux > 0.0
            clamped_flux = raw_flux if active else 0.0
            tol = _effective_switching_tolerance_m2_s(spec)
            on_kink = (
                spec.mode != LAGGED_BAKED_GENERATION_V1
                and has_trapping
                and abs(raw_flux) <= tol
            )
            smooth = not on_kink
        else:
            active = True
            clamped_flux = raw_flux
            smooth = True

        if not smooth:
            non_smooth.append(spec.layer_id)

        recv = spec.receiver_nodes
        u_recv = trapped_fraction * spec.receiver_profile_m1
        if clamped_flux != 0.0 and trapped_fraction > 0.0:
            local_gen = u_recv * clamped_flux
            generation[recv] += local_gen
            reabsorbed_flux = float(np.dot(spec.receiver_weights_m, local_gen))
        else:
            reabsorbed_flux = 0.0

        escaped_flux = spec.escape_probability * clamped_flux
        residual_flux = clamped_flux - reabsorbed_flux - escaped_flux
        budgets.append(
            PhotonRecyclingBudget(
                layer_id=spec.layer_id,
                mode=spec.mode,
                escape_probability=spec.escape_probability,
                raw_emitted_flux_m2_s=raw_flux,
                emitted_flux_m2_s=clamped_flux,
                reabsorbed_flux_m2_s=reabsorbed_flux,
                escaped_flux_m2_s=escaped_flux,
                closure_residual_m2_s=residual_flux,
            )
        )

        if derivatives:
            emit = spec.emitter_nodes
            if spec.mode == LAGGED_BAKED_GENERATION_V1 or not active or trapped_fraction == 0.0:
                v_n = np.zeros(emit.size, dtype=float)
                v_p = np.zeros(emit.size, dtype=float)
                u_factor = np.zeros(recv.size, dtype=float)
            else:
                coeff = spec.emitter_weights_m * spec.intrinsic_radiative_B_m3_s
                v_n = coeff * p_eval[emit]
                v_p = coeff * n_eval[emit]
                u_factor = u_recv
            factors.append(
                AbsorberRank1Factor(
                    layer_id=spec.layer_id,
                    mode=spec.mode,
                    emitter_nodes=emit,
                    receiver_nodes=recv,
                    receiver_vector_m1=frozen_array(u_factor),
                    d_emission_flux_dn_m_s=frozen_array(v_n),
                    d_emission_flux_dp_m_s=frozen_array(v_p),
                    active=active,
                    differentiable=smooth,
                    raw_emitted_flux_m2_s=raw_flux,
                    clamped_emitted_flux_m2_s=clamped_flux,
                )
            )

    return ReabsorptionEvaluation(
        generation_m3_s=frozen_array(generation),
        factors=tuple(factors),
        budgets=tuple(budgets),
        differentiable=len(non_smooth) == 0,
        non_smooth_layers=tuple(non_smooth),
    )


def dense_reabsorption_jacobian(
    evaluation: ReabsorptionEvaluation,
    node_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Explicit dense ``(dG_rad/dn, dG_rad/dp)`` matrices for small-grid verification."""
    d_dn: Vector2D = np.zeros((node_count, node_count), dtype=float)
    d_dp: Vector2D = np.zeros((node_count, node_count), dtype=float)
    for factor in evaluation.factors:
        if not factor.active:
            continue
        rows = factor.receiver_nodes[:, None]
        cols = factor.emitter_nodes[None, :]
        d_dn[rows, cols] += np.outer(factor.receiver_vector_m1, factor.d_emission_flux_dn_m_s)
        d_dp[rows, cols] += np.outer(factor.receiver_vector_m1, factor.d_emission_flux_dp_m_s)
    return d_dn, d_dp


@dataclass(frozen=True)
class ReabsorptionDirectionalDerivative:
    """Value and exact one-sided directional derivative of ``G_rad`` across ``R_tot = 0``."""

    generation_m3_s: Vector
    directional_derivative_m3_s: Vector
    smooth: bool
    non_smooth_layers: tuple[str, ...]


def radiative_reabsorption_directional_derivative(
    specs: Sequence[AbsorberReabsorptionSpec],
    electron_density_m3: ArrayLike,
    hole_density_m3: ArrayLike,
    delta_electron_m3: ArrayLike,
    delta_hole_m3: ArrayLike,
) -> ReabsorptionDirectionalDerivative:
    """Exact one-sided directional derivative ``lim_{h->0+} [G(n+hdn, p+hdp) - G(n,p)] / h``."""
    n = np.asarray(electron_density_m3, dtype=float)
    p = np.asarray(hole_density_m3, dtype=float)
    dn = np.asarray(delta_electron_m3, dtype=float)
    dp = np.asarray(delta_hole_m3, dtype=float)
    if (
        n.ndim != 1
        or p.shape != n.shape
        or dn.shape != n.shape
        or dp.shape != n.shape
        or not (np.all(np.isfinite(dn)) and np.all(np.isfinite(dp)))
    ):
        raise PhotonRecyclingError("densities and directions must be finite 1D arrays of equal length")

    base = evaluate_radiative_reabsorption(specs, n, p, derivatives=False)
    slope = np.zeros_like(n)
    non_smooth: list[str] = []

    for spec in specs:
        if spec.mode == LAGGED_BAKED_GENERATION_V1:
            continue
        trapped_fraction = 1.0 - spec.escape_probability
        if trapped_fraction <= 0.0 or not np.any(spec.intrinsic_radiative_B_m3_s > 0.0):
            continue
        _, raw_flux = _raw_emission_flux(spec, n, p)
        emit = spec.emitter_nodes
        recv = spec.receiver_nodes
        coeff = spec.emitter_weights_m * spec.intrinsic_radiative_B_m3_s
        d_flux = float(np.dot(coeff, p[emit] * dn[emit] + n[emit] * dp[emit]))
        u_recv = trapped_fraction * spec.receiver_profile_m1

        tol = _effective_switching_tolerance_m2_s(spec)
        if not spec.clamp_negative_net_emission:
            slope[recv] += u_recv * d_flux
        elif raw_flux > tol:
            slope[recv] += u_recv * d_flux
        elif raw_flux < -tol:
            pass
        else:
            non_smooth.append(spec.layer_id)
            if d_flux > 0.0:
                slope[recv] += u_recv * d_flux

    return ReabsorptionDirectionalDerivative(
        generation_m3_s=base.generation_m3_s,
        directional_derivative_m3_s=frozen_array(slope),
        smooth=len(non_smooth) == 0,
        non_smooth_layers=tuple(non_smooth),
    )


def thermionic_cap_value(
    j_sg: ArrayLike,
    j_te: ArrayLike,
    *,
    te_softness: float = 0.0,
) -> Vector:
    """Direction-preserving thermionic magnitude cap ``copysign(min(|J_sg|, |J_te|), J_sg)``."""
    sg, te = np.broadcast_arrays(np.asarray(j_sg, dtype=float), np.asarray(j_te, dtype=float))
    if not (np.all(np.isfinite(sg)) and np.all(np.isfinite(te))):
        raise PhotonRecyclingError("j_sg and j_te must be finite")
    soft = float(te_softness)
    if not math.isfinite(soft) or soft < 0.0:
        raise PhotonRecyclingError("te_softness must be finite and non-negative")
    a_sg = np.abs(sg)
    a_te = np.abs(te)
    if soft == 0.0:
        return frozen_array(np.copysign(np.minimum(a_sg, a_te), sg))
    denom = soft * (a_sg + a_te) + 1.0e-300
    t_arg = (a_sg - a_te) / denom
    weight = np.where(
        t_arg > 40.0,
        1.0,
        np.where(t_arg < -40.0, 0.0, 1.0 / (1.0 + np.exp(-np.clip(t_arg, -40.0, 40.0)))),
    )
    magnitude = weight * a_te + (1.0 - weight) * a_sg
    return frozen_array(np.copysign(magnitude, sg))


@dataclass(frozen=True)
class ThermionicCapDirectionalDerivative:
    """Value, one-sided directional derivative, and smoothness mask of the TE cap."""

    capped_current_A_m2: Vector
    directional_derivative_A_m2: Vector
    smooth: NDArray[np.bool_]


def thermionic_cap_directional_derivative(
    j_sg: ArrayLike,
    j_te: ArrayLike,
    d_j_sg: ArrayLike,
    d_j_te: ArrayLike,
    *,
    te_softness: float = 0.0,
    crossover_tolerance_A_m2: float = 1.0e-15,
) -> ThermionicCapDirectionalDerivative:
    """Exact one-sided directional derivative across ``|J_sg| == |J_te|`` and ``J_sg == 0``."""
    sg, te, dsg, dte = np.broadcast_arrays(
        np.asarray(j_sg, dtype=float),
        np.asarray(j_te, dtype=float),
        np.asarray(d_j_sg, dtype=float),
        np.asarray(d_j_te, dtype=float),
    )
    sg_1d = sg.reshape(-1)
    te_1d = te.reshape(-1)
    dsg_1d = dsg.reshape(-1)
    dte_1d = dte.reshape(-1)
    if not (np.all(np.isfinite(sg_1d)) and np.all(np.isfinite(te_1d)) and np.all(np.isfinite(dsg_1d)) and np.all(np.isfinite(dte_1d))):
        raise PhotonRecyclingError("thermionic currents and perturbations must be finite")
    soft = float(te_softness)
    if not math.isfinite(soft) or soft < 0.0:
        raise PhotonRecyclingError("te_softness must be finite and non-negative")
    tol = float(crossover_tolerance_A_m2)
    if not math.isfinite(tol) or tol < 0.0:
        raise PhotonRecyclingError("crossover_tolerance_A_m2 must be finite and non-negative")

    capped = thermionic_cap_value(sg_1d, te_1d, te_softness=soft)
    deriv = np.zeros_like(sg_1d)
    smooth = np.ones(sg_1d.shape, dtype=bool)

    for idx in range(sg_1d.size):
        s_val = float(sg_1d[idx])
        t_val = float(te_1d[idx])
        ds = float(dsg_1d[idx])
        dt = float(dte_1d[idx])
        a_s = abs(s_val)
        a_t = abs(t_val)
        da_s = abs(ds) if a_s <= tol else math.copysign(1.0, s_val) * ds
        da_t = abs(dt) if a_t <= tol else math.copysign(1.0, t_val) * dt

        if a_s <= tol:
            # At J_sg = 0: if |J_te| > 0 and ds == 0, copysign(0, ...) stays 0;
            # if a_t == 0 too, copysign(h * min(|ds|, |dt|), ds) has a kink when |dt| < |ds|.
            if a_t <= tol:
                smooth[idx] = False
                sign_dir = 0.0 if ds == 0.0 else math.copysign(1.0, ds)
                deriv[idx] = sign_dir * min(abs(ds), abs(dt))
            else:
                smooth[idx] = soft == 0.0
                deriv[idx] = ds if soft == 0.0 else (
                    math.copysign(1.0, ds) * (1.0 / (1.0 + math.exp(1.0 / soft))) * a_t
                    if ds != 0.0
                    else 0.0
                )
                if soft > 0.0 and a_t > tol:
                    # Jump discontinuity in copysign(nonzero, J_sg) across J_sg = 0!
                    smooth[idx] = False
                    deriv[idx] = math.copysign(math.inf, ds) if ds != 0.0 else 0.0
            continue

        sign_sg = math.copysign(1.0, s_val)
        if soft == 0.0:
            if abs(a_s - a_t) <= tol:
                smooth[idx] = False
                deriv[idx] = sign_sg * min(da_s, da_t)
            elif a_s < a_t:
                smooth[idx] = True
                if a_t <= tol:
                    smooth[idx] = False
                deriv[idx] = ds
            else:
                smooth[idx] = a_t > tol
                deriv[idx] = sign_sg * da_t
        else:
            if a_t <= tol:
                smooth[idx] = False
            denom = soft * (a_s + a_t) + 1.0e-300
            t_arg = (a_s - a_t) / denom
            if t_arg > 40.0:
                weight = 1.0
                d_weight = 0.0
            elif t_arg < -40.0:
                weight = 0.0
                d_weight = 0.0
            else:
                weight = 1.0 / (1.0 + math.exp(-t_arg))
                sig_prime = weight * (1.0 - weight)
                dt_arg = (
                    (da_s - da_t) * denom - (a_s - a_t) * soft * (da_s + da_t)
                ) / (denom * denom)
                d_weight = sig_prime * dt_arg
            d_mag = d_weight * (a_t - a_s) + weight * da_t + (1.0 - weight) * da_s
            deriv[idx] = sign_sg * d_mag

    return ThermionicCapDirectionalDerivative(
        capped_current_A_m2=capped,
        directional_derivative_A_m2=frozen_array(deriv),
        smooth=smooth,
    )


@dataclass(frozen=True)
class SwitchingSurfaceAssessment:
    """Combined differentiability assessment across reabsorption, TE, and PF surfaces."""

    smooth: bool
    reabsorption_smooth: bool
    thermionic_smooth: bool
    field_mobility_smooth: bool
    reasons: tuple[str, ...]


def assess_switching_surfaces(
    *,
    reabsorption: ReabsorptionEvaluation | None = None,
    thermionic_electron: ThermionicCapDirectionalDerivative | None = None,
    thermionic_hole: ThermionicCapDirectionalDerivative | None = None,
    field_mobility_electron: FieldMobilityDirectionalDerivative | None = None,
    field_mobility_hole: FieldMobilityDirectionalDerivative | None = None,
    face_fields_V_m: ArrayLike | None = None,
    mu0_n: ArrayLike | None = None,
    v_sat_n: ArrayLike | None = None,
    beta_n: ArrayLike | None = None,
    gamma_pf_n: ArrayLike | None = None,
    pf_width_V_m: float = 0.0,
) -> SwitchingSurfaceAssessment:
    """Evaluate smoothness across declared switching surfaces before linear AC solves."""
    reasons: list[str] = []
    reabs_ok = True
    if reabsorption is not None and not reabsorption.differentiable:
        reabs_ok = False
        layers = ", ".join(reabsorption.non_smooth_layers)
        reasons.append(
            f"radiative_reabsorption clamped switching surface R_tot == 0 active on ({layers})"
        )

    te_ok = True
    for label, te_diag in (("electron", thermionic_electron), ("hole", thermionic_hole)):
        if te_diag is not None and not bool(np.all(te_diag.smooth)):
            te_ok = False
            count = int(np.sum(~te_diag.smooth))
            reasons.append(f"thermionic_cap {label} non-smooth min/sign surface on {count} face(s)")

    fm_ok = True
    if (
        field_mobility_electron is None
        and face_fields_V_m is not None
        and mu0_n is not None
        and v_sat_n is not None
        and beta_n is not None
        and gamma_pf_n is not None
    ):
        fields = np.asarray(face_fields_V_m, dtype=float)
        field_mobility_electron = field_mobility_directional_derivative(
            mu0_n,
            fields,
            np.ones_like(fields),
            v_sat_n,
            beta_n,
            gamma_pf_n,
            pf_field_regularization_width_V_m=pf_width_V_m,
        )
    for label, fm_diag in (("electron", field_mobility_electron), ("hole", field_mobility_hole)):
        if fm_diag is not None and not bool(np.all(fm_diag.smooth)):
            fm_ok = False
            count = int(np.sum(~fm_diag.smooth))
            infinite = int(np.sum(~fm_diag.finite))
            reasons.append(
                f"field_mobility {label} non-smooth surface on {count} face(s) ({infinite} infinite slope)"
            )

    return SwitchingSurfaceAssessment(
        smooth=reabs_ok and te_ok and fm_ok,
        reabsorption_smooth=reabs_ok,
        thermionic_smooth=te_ok,
        field_mobility_smooth=fm_ok,
        reasons=tuple(reasons),
    )


def require_smooth_ac_surfaces(assessment: SwitchingSurfaceAssessment) -> None:
    """Refuse linear small-signal AC assembly on non-smooth switching surfaces."""
    if not assessment.smooth:
        detail = "; ".join(assessment.reasons)
        raise PhotonRecyclingCapabilityError(
            "non_smooth_ac_switching_surface: linear AC small-signal operator is undefined at "
            f"this operating point ({detail})"
        )
