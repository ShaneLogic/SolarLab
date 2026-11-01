"""DEF-1 explicit neutral multi-species SRH and its node ownership.

An explicit defect inventory REPLACES the layer's lifetime-form SRH on the
nodes it owns; it is never added on top of it. Each neutral single-level
species contributes its own lifetime-form SRH term with capture lifetimes
``tau = 1 / (sigma v_th N_t)`` and trap reference densities
``n1 = ni exp((E_t - E_g/2)/V_T)``, ``p1 = ni exp(-(E_t - E_g/2)/V_T)``.
Species owning the same node are summed in declaration order. A species
with an infinite capture lifetime cannot close the SRH cycle and
contributes nothing.

Model identity (``explicit_neutral_multi_species_srh_v1``) and every array
operation follow the historical implementation exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from solarlab.physics.recombination._validation import frozen_bool_mask, frozen_float_array
from solarlab.physics.recombination.bulk import lifetime_srh_denominator

__all__ = [
    "EXPLICIT_NEUTRAL_MODEL_ID",
    "NeutralDefectSpecies",
    "NeutralExplicitDefectEvaluation",
    "NeutralExplicitDefectModel",
    "compile_neutral_defect_species",
    "evaluate_neutral_explicit_defects",
]

EXPLICIT_NEUTRAL_MODEL_ID = "explicit_neutral_multi_species_srh_v1"


@dataclass(frozen=True, slots=True)
class NeutralDefectSpecies:
    """One neutral single-level SRH species compiled onto the device grid."""

    identifier: str
    document_sha256: str
    active_nodes: np.ndarray
    tau_n_s: float
    tau_p_s: float
    n1_m3: np.ndarray
    p1_m3: np.ndarray

    def __post_init__(self) -> None:
        identifier = str(self.identifier).strip()
        if not identifier:
            raise ValueError("neutral defect identifier must be non-empty")
        digest = str(self.document_sha256).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("neutral defect document_sha256 must be a SHA-256 hex")
        active = np.asarray(self.active_nodes, dtype=bool)
        n1 = np.asarray(self.n1_m3, dtype=float)
        p1 = np.asarray(self.p1_m3, dtype=float)
        if active.ndim != 1 or not np.any(active):
            raise ValueError("neutral defect active_nodes must be a non-empty 1D mask")
        if n1.shape != active.shape or p1.shape != active.shape:
            raise ValueError("neutral defect reference arrays must match active_nodes")
        if (
            not np.all(np.isfinite(n1))
            or not np.all(np.isfinite(p1))
            or np.any(n1 < 0.0)
            or np.any(p1 < 0.0)
        ):
            raise ValueError("neutral defect n1/p1 arrays must be finite and non-negative")
        for name in ("tau_n_s", "tau_p_s"):
            value = float(getattr(self, name))
            if np.isnan(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive or +inf")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "document_sha256", digest)
        object.__setattr__(self, "active_nodes", frozen_bool_mask(active, name="active_nodes"))
        object.__setattr__(self, "n1_m3", frozen_float_array(n1, name="n1_m3"))
        object.__setattr__(self, "p1_m3", frozen_float_array(p1, name="p1_m3"))

    @property
    def cycle_active(self) -> bool:
        """Whether both capture legs are finite and the SRH cycle can close."""

        return bool(np.isfinite(self.tau_n_s) and np.isfinite(self.tau_p_s))


@dataclass(frozen=True, slots=True)
class NeutralExplicitDefectModel:
    """Compiled DEF-1 inventory; ``explicit_node_mask`` is its node ownership."""

    species: tuple[NeutralDefectSpecies, ...]
    explicit_node_mask: np.ndarray
    layer_document_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        species = tuple(self.species)
        if not species or not all(isinstance(item, NeutralDefectSpecies) for item in species):
            raise ValueError("neutral bulk-defect model requires compiled species")
        identifiers = [item.identifier for item in species]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("compiled neutral defect identifiers must be unique")
        mask = np.asarray(self.explicit_node_mask, dtype=bool)
        if mask.ndim != 1 or not np.any(mask):
            raise ValueError("explicit_node_mask must be a non-empty 1D mask")
        if any(item.active_nodes.shape != mask.shape for item in species):
            raise ValueError("compiled neutral defect grids must have one node count")
        union = np.logical_or.reduce([item.active_nodes for item in species])
        if not np.array_equal(mask, union):
            raise ValueError("explicit_node_mask must equal the union of species masks")
        documents = tuple(
            (str(layer), str(digest).lower()) for layer, digest in self.layer_document_sha256
        )
        if not documents:
            raise ValueError("neutral bulk-defect model requires layer document hashes")
        if len({layer for layer, _ in documents}) != len(documents):
            raise ValueError("neutral bulk-defect layer names must be unique")
        valid_hashes = {item.document_sha256 for item in species}
        for layer, digest in documents:
            if not layer or digest not in valid_hashes:
                raise ValueError("layer document hashes must match compiled species")
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "explicit_node_mask", frozen_bool_mask(mask, name="explicit_node_mask"))
        object.__setattr__(self, "layer_document_sha256", documents)

    @property
    def node_count(self) -> int:
        return int(self.explicit_node_mask.size)

    @property
    def owned_nodes(self) -> np.ndarray:
        """Nodes on which this model replaces the lifetime-form SRH law."""

        return self.explicit_node_mask

    @property
    def identity_sha256(self) -> str:
        payload = {
            "model": EXPLICIT_NEUTRAL_MODEL_ID,
            "layers": [
                {"name": layer, "document_sha256": digest}
                for layer, digest in self.layer_document_sha256
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def compile_neutral_defect_species(
    *,
    identifier: str,
    document_sha256: str,
    active_nodes: np.ndarray,
    ni_sq_m6: np.ndarray,
    band_gap_eV: np.ndarray,
    thermal_voltage_V: float,
    total_density_m3: float,
    center_eV_above_vb: float,
    sigma_n_m2: float,
    sigma_p_m2: float,
    thermal_velocity_n_m_s: float,
    thermal_velocity_p_m_s: float,
    charge_transition: str,
    distribution: str,
    degeneracy: float,
) -> NeutralDefectSpecies:
    """Compile one DEF-1 species from resolved node arrays.

    ``ni_sq_m6`` and ``band_gap_eV`` are full-grid arrays; only the
    ``active_nodes`` entries are read. DEF-1 requires a uniform positive gap
    and intrinsic density on the owned nodes, a neutral transition, a single
    level and unit degeneracy; anything else is a capability error, never a
    silent approximation.
    """

    if charge_transition != "neutral":
        raise ValueError(
            f"DEF-1 explicit execution accepts neutral species only; {identifier} "
            f"is {charge_transition!r}"
        )
    if distribution != "single_level":
        raise ValueError("DEF-1 explicit execution accepts single-level species only")
    if degeneracy != 1.0:
        raise ValueError(
            "DEF-1 does not silently ignore degeneracy; explicit neutral species "
            f"{identifier} must set degeneracy=1.0"
        )
    mask = np.asarray(active_nodes, dtype=bool)
    ni_sq = np.asarray(ni_sq_m6, dtype=float)
    gap = np.asarray(band_gap_eV, dtype=float)
    if mask.ndim != 1 or ni_sq.shape != mask.shape or gap.shape != mask.shape:
        raise ValueError("DEF-1 node arrays must share one 1D grid")
    local_gap = gap[mask]
    local_ni_sq = ni_sq[mask]
    if (
        local_gap.size == 0
        or not np.all(np.isfinite(local_gap))
        or not np.all(local_gap == local_gap[0])
        or not np.all(np.isfinite(local_ni_sq))
        or np.any(local_ni_sq <= 0.0)
    ):
        raise ValueError(
            "DEF-1 explicit neutral SRH requires a uniform positive band gap "
            f"and intrinsic density in {identifier}; spatial grading starts at D3"
        )
    if not math.isfinite(float(thermal_voltage_V)) or float(thermal_voltage_V) <= 0.0:
        raise ValueError("thermal_voltage_V must be finite and positive")
    density = float(total_density_m3)
    inverse_tau_n = float(sigma_n_m2) * float(thermal_velocity_n_m_s) * density
    inverse_tau_p = float(sigma_p_m2) * float(thermal_velocity_p_m_s) * density
    tau_n_s = 1.0 / inverse_tau_n if inverse_tau_n > 0.0 else math.inf
    tau_p_s = 1.0 / inverse_tau_p if inverse_tau_p > 0.0 else math.inf
    active = np.array(mask, dtype=bool, copy=True)
    node_count = int(mask.size)
    n1_nodes: np.ndarray = np.zeros(node_count, dtype=float)
    p1_nodes: np.ndarray = np.zeros(node_count, dtype=float)
    ni = np.sqrt(local_ni_sq)
    offset_from_midgap_eV = float(center_eV_above_vb) - 0.5 * local_gap
    ratio = np.exp(offset_from_midgap_eV / thermal_voltage_V)
    n1_nodes[active] = ni * ratio
    p1_nodes[active] = ni / ratio
    return NeutralDefectSpecies(
        identifier=identifier,
        document_sha256=document_sha256,
        active_nodes=active,
        tau_n_s=tau_n_s,
        tau_p_s=tau_p_s,
        n1_m3=n1_nodes,
        p1_m3=p1_nodes,
    )


@dataclass(frozen=True, slots=True)
class NeutralExplicitDefectEvaluation:
    """Per-species and total explicit SRH on the full grid.

    Arrays are zero on nodes a species does not own. The total is the
    declaration-order sum of the species rows.
    """

    model_identity_sha256: str
    species_identifiers: tuple[str, ...]
    total_rate_m3_s: np.ndarray
    total_d_rate_dn_s1: np.ndarray
    total_d_rate_dp_s1: np.ndarray
    per_species_rate_m3_s: np.ndarray
    per_species_d_rate_dn_s1: np.ndarray
    per_species_d_rate_dp_s1: np.ndarray
    per_species_denominators_s_m3: tuple[np.ndarray | None, ...]
    minimum_denominator_s_m3: tuple[float | None, ...]

    def to_dict(self) -> dict[str, object]:
        """The historical JSON result payload, key for key."""

        return {
            "model": EXPLICIT_NEUTRAL_MODEL_ID,
            "model_identity_sha256": self.model_identity_sha256,
            "species_identifiers": list(self.species_identifiers),
            "total_rate_m3_s": np.asarray(self.total_rate_m3_s).tolist(),
            "total_electron_derivative_s1": np.asarray(self.total_d_rate_dn_s1).tolist(),
            "total_hole_derivative_s1": np.asarray(self.total_d_rate_dp_s1).tolist(),
            "per_species_rate_m3_s": self.per_species_rate_m3_s.tolist(),
            "per_species_electron_derivative_s1": self.per_species_d_rate_dn_s1.tolist(),
            "per_species_hole_derivative_s1": self.per_species_d_rate_dp_s1.tolist(),
            "minimum_denominator_s_m3": list(self.minimum_denominator_s_m3),
            "charge_density_C_m3": None,
        }


def evaluate_neutral_explicit_defects(
    n: np.ndarray,
    p: np.ndarray,
    ni_sq: np.ndarray,
    model: NeutralExplicitDefectModel,
) -> NeutralExplicitDefectEvaluation:
    """Evaluate exact per-species neutral SRH and analytic carrier tangents."""

    n_a, p_a, ni_a = np.broadcast_arrays(
        np.asarray(n, dtype=float),
        np.asarray(p, dtype=float),
        np.asarray(ni_sq, dtype=float),
    )
    if n_a.shape != (model.node_count,):
        raise ValueError("neutral bulk-defect state must match the compiled grid")
    shape = (len(model.species), model.node_count)
    rates: np.ndarray = np.zeros(shape, dtype=float)
    derivative_n: np.ndarray = np.zeros(shape, dtype=float)
    derivative_p: np.ndarray = np.zeros(shape, dtype=float)
    denominators: list[np.ndarray | None] = []
    minima: list[float | None] = []
    for index, species in enumerate(model.species):
        if not species.cycle_active:
            denominators.append(None)
            minima.append(None)
            continue
        active = species.active_nodes
        denominator = lifetime_srh_denominator(
            n_a[active],
            p_a[active],
            species.tau_n_s,
            species.tau_p_s,
            species.n1_m3[active],
            species.p1_m3[active],
        )
        numerator = n_a[active] * p_a[active] - ni_a[active]
        rate = numerator / denominator
        rates[index, active] = rate
        derivative_n[index, active] = (p_a[active] - rate * species.tau_p_s) / denominator
        derivative_p[index, active] = (n_a[active] - rate * species.tau_n_s) / denominator
        denominator.setflags(write=False)
        denominators.append(denominator)
        finite = denominator[np.isfinite(denominator)]
        minima.append(float(np.min(finite)) if finite.size else None)
    if len(model.species) == 1:
        total_rate = rates[0].copy()
        total_n = derivative_n[0].copy()
        total_p = derivative_p[0].copy()
    else:
        total_rate = np.sum(rates, axis=0)
        total_n = np.sum(derivative_n, axis=0)
        total_p = np.sum(derivative_p, axis=0)
    for array in (rates, derivative_n, derivative_p, total_rate, total_n, total_p):
        array.setflags(write=False)
    return NeutralExplicitDefectEvaluation(
        model_identity_sha256=model.identity_sha256,
        species_identifiers=tuple(item.identifier for item in model.species),
        total_rate_m3_s=total_rate,
        total_d_rate_dn_s1=total_n,
        total_d_rate_dp_s1=total_p,
        per_species_rate_m3_s=rates,
        per_species_d_rate_dn_s1=derivative_n,
        per_species_d_rate_dp_s1=derivative_p,
        per_species_denominators_s_m3=tuple(denominators),
        minimum_denominator_s_m3=tuple(minima),
    )
