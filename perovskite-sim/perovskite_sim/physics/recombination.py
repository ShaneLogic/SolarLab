from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from perovskite_sim.physics.defect_closure import (
    MonovalentBulkDefectModel,
    evaluate_monovalent_bulk_defects,
    evaluate_monovalent_source_defect_closure,
)
from perovskite_sim.physics.multivalent_defect_device import (
    MultivalentBulkDefectModel,
    evaluate_multivalent_bulk_defects,
    evaluate_multivalent_source_defect_closure,
)


@dataclass(frozen=True, slots=True)
class RecombinationDerivatives:
    """A recombination rate and its local carrier-density derivatives."""

    rate: np.ndarray
    electron_density_derivative: np.ndarray
    hole_density_derivative: np.ndarray


def _readonly_array(value: object, *, dtype: object = float) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class CompiledNeutralDefectSpecies:
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
        object.__setattr__(self, "active_nodes", _readonly_array(active, dtype=bool))
        object.__setattr__(self, "n1_m3", _readonly_array(n1))
        object.__setattr__(self, "p1_m3", _readonly_array(p1))

    @property
    def cycle_active(self) -> bool:
        """Whether both capture legs are finite and the SRH cycle can close."""

        return bool(np.isfinite(self.tau_n_s) and np.isfinite(self.tau_p_s))


@dataclass(frozen=True, slots=True)
class NeutralBulkDefectModel:
    """Compiled DEF-1 model for exact neutral multi-species SRH."""

    species: tuple[CompiledNeutralDefectSpecies, ...]
    explicit_node_mask: np.ndarray
    layer_document_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        species = tuple(self.species)
        if not species or not all(
            isinstance(item, CompiledNeutralDefectSpecies) for item in species
        ):
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
        documents = tuple((str(layer), str(digest).lower()) for layer, digest in self.layer_document_sha256)
        if not documents:
            raise ValueError("neutral bulk-defect model requires layer document hashes")
        if len({layer for layer, _ in documents}) != len(documents):
            raise ValueError("neutral bulk-defect layer names must be unique")
        valid_hashes = {item.document_sha256 for item in species}
        for layer, digest in documents:
            if not layer or digest not in valid_hashes:
                raise ValueError("layer document hashes must match compiled species")
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "explicit_node_mask", _readonly_array(mask, dtype=bool))
        object.__setattr__(self, "layer_document_sha256", documents)

    @property
    def node_count(self) -> int:
        return int(self.explicit_node_mask.size)

    @property
    def identity_sha256(self) -> str:
        payload = {
            "model": "explicit_neutral_multi_species_srh_v1",
            "layers": [
                {"name": layer, "document_sha256": digest}
                for layer, digest in self.layer_document_sha256
            ],
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class NeutralBulkDefectEvaluation:
    """Exact total/per-species neutral SRH rates and local derivatives."""

    model_identity_sha256: str
    species_identifiers: tuple[str, ...]
    total: RecombinationDerivatives
    per_species_rate_m3_s: np.ndarray
    per_species_electron_derivative_s1: np.ndarray
    per_species_hole_derivative_s1: np.ndarray
    minimum_denominator_s_m3: tuple[float | None, ...]

    def __post_init__(self) -> None:
        identifiers = tuple(self.species_identifiers)
        rate = np.asarray(self.per_species_rate_m3_s, dtype=float)
        derivative_n = np.asarray(
            self.per_species_electron_derivative_s1, dtype=float
        )
        derivative_p = np.asarray(self.per_species_hole_derivative_s1, dtype=float)
        if (
            rate.ndim != 2
            or rate.shape[0] != len(identifiers)
            or derivative_n.shape != rate.shape
            or derivative_p.shape != rate.shape
        ):
            raise ValueError("per-species neutral defect diagnostics are mis-shaped")
        for value in (
            self.total.rate,
            self.total.electron_density_derivative,
            self.total.hole_density_derivative,
        ):
            if np.asarray(value).shape != rate.shape[1:]:
                raise ValueError("total neutral defect diagnostics are mis-shaped")
        if len(self.minimum_denominator_s_m3) != len(identifiers):
            raise ValueError("one denominator diagnostic is required per species")
        object.__setattr__(self, "species_identifiers", identifiers)
        object.__setattr__(self, "per_species_rate_m3_s", _readonly_array(rate))
        object.__setattr__(
            self,
            "per_species_electron_derivative_s1",
            _readonly_array(derivative_n),
        )
        object.__setattr__(
            self,
            "per_species_hole_derivative_s1",
            _readonly_array(derivative_p),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible optional result payload."""

        return {
            "model": "explicit_neutral_multi_species_srh_v1",
            "model_identity_sha256": self.model_identity_sha256,
            "species_identifiers": list(self.species_identifiers),
            "total_rate_m3_s": np.asarray(self.total.rate).tolist(),
            "total_electron_derivative_s1": np.asarray(
                self.total.electron_density_derivative
            ).tolist(),
            "total_hole_derivative_s1": np.asarray(
                self.total.hole_density_derivative
            ).tolist(),
            "per_species_rate_m3_s": self.per_species_rate_m3_s.tolist(),
            "per_species_electron_derivative_s1": (
                self.per_species_electron_derivative_s1.tolist()
            ),
            "per_species_hole_derivative_s1": (
                self.per_species_hole_derivative_s1.tolist()
            ),
            "minimum_denominator_s_m3": list(self.minimum_denominator_s_m3),
            "charge_density_C_m3": None,
        }


SRHDenominatorObserver = Callable[[str, np.ndarray], None]
_SRH_DENOMINATOR_OBSERVER: ContextVar[SRHDenominatorObserver | None] = (
    ContextVar("srh_denominator_observer", default=None)
)


@contextmanager
def _observe_srh_denominators(
    observer: SRHDenominatorObserver,
) -> Iterator[None]:
    """Route production SRH denominators to one per-solve observer."""

    token = _SRH_DENOMINATOR_OBSERVER.set(observer)
    try:
        yield
    finally:
        _SRH_DENOMINATOR_OBSERVER.reset(token)


def _record_srh_denominator(kind: str, denominator: np.ndarray) -> None:
    observer = _SRH_DENOMINATOR_OBSERVER.get()
    if observer is not None:
        observer(kind, np.asarray(denominator))


def bulk_srh_denominator(
    n: np.ndarray,
    p: np.ndarray,
    tau_n: float,
    tau_p: float,
    n1: float,
    p1: float,
) -> np.ndarray:
    """Bulk SRH denominator [s m^-3], without altering invalid inputs."""

    return tau_p * (n + n1) + tau_n * (p + p1)


def bulk_recombination_denominators(
    n: np.ndarray,
    p: np.ndarray,
    tau_n: float,
    tau_p: float,
    n1: float,
    p1: float,
    *,
    neutral_bulk_defects: NeutralBulkDefectModel | None = None,
) -> np.ndarray:
    """Return every active lifetime-form SRH denominator used at this state."""

    if neutral_bulk_defects is None:
        return np.asarray(bulk_srh_denominator(n, p, tau_n, tau_p, n1, p1))
    n_array, p_array, tau_n_array, tau_p_array, n1_array, p1_array = (
        np.broadcast_arrays(
            np.asarray(n, dtype=float),
            np.asarray(p, dtype=float),
            np.asarray(tau_n, dtype=float),
            np.asarray(tau_p, dtype=float),
            np.asarray(n1, dtype=float),
            np.asarray(p1, dtype=float),
        )
    )
    if n_array.shape != (neutral_bulk_defects.node_count,):
        raise ValueError("neutral bulk-defect state must match the compiled grid")
    values: list[np.ndarray] = []
    legacy = ~neutral_bulk_defects.explicit_node_mask
    if np.any(legacy):
        values.append(
            bulk_srh_denominator(
                n_array[legacy],
                p_array[legacy],
                tau_n_array[legacy],
                tau_p_array[legacy],
                n1_array[legacy],
                p1_array[legacy],
            )
        )
    for species in neutral_bulk_defects.species:
        if not species.cycle_active:
            continue
        active = species.active_nodes
        values.append(
            bulk_srh_denominator(
                n_array[active],
                p_array[active],
                species.tau_n_s,
                species.tau_p_s,
                species.n1_m3[active],
                species.p1_m3[active],
            )
        )
    if not values:
        return np.empty(0, dtype=float)
    return np.concatenate([np.ravel(value) for value in values])


def interface_srh_denominator(
    n: float,
    p: float,
    n1: float,
    p1: float,
    v_n: float,
    v_p: float,
) -> float:
    """Surface SRH denominator [s m^-4] for positive capture velocities."""

    return (n + n1) / v_p + (p + p1) / v_n


def srh_recombination(
    n: np.ndarray, p: np.ndarray, ni_sq: float,
    tau_n: float, tau_p: float, n1: float, p1: float,
    *,
    neutral_bulk_defects: NeutralBulkDefectModel | None = None,
    monovalent_bulk_defects: MonovalentBulkDefectModel | None = None,
    multivalent_bulk_defects: MultivalentBulkDefectModel | None = None,
) -> np.ndarray:
    """Shockley-Read-Hall recombination rate [m⁻³ s⁻¹]."""
    if multivalent_bulk_defects is not None:
        return _mixed_multivalent_srh_recombination(
            n,
            p,
            ni_sq,
            tau_n,
            tau_p,
            n1,
            p1,
            neutral_bulk_defects=neutral_bulk_defects,
            monovalent_bulk_defects=monovalent_bulk_defects,
            multivalent_bulk_defects=multivalent_bulk_defects,
        )
    if neutral_bulk_defects is not None and monovalent_bulk_defects is not None:
        raise ValueError("neutral and monovalent bulk-defect models are exclusive")
    if monovalent_bulk_defects is not None:
        return _mixed_monovalent_srh_recombination(
            n,
            p,
            ni_sq,
            tau_n,
            tau_p,
            n1,
            p1,
            monovalent_bulk_defects,
        )
    if neutral_bulk_defects is not None:
        return _mixed_srh_recombination(
            n,
            p,
            ni_sq,
            tau_n,
            tau_p,
            n1,
            p1,
            neutral_bulk_defects,
        )
    denominator = bulk_srh_denominator(n, p, tau_n, tau_p, n1, p1)
    _record_srh_denominator("bulk", denominator)
    return (n * p - ni_sq) / denominator


def srh_recombination_derivatives(
    n: np.ndarray,
    p: np.ndarray,
    ni_sq: float,
    tau_n: float,
    tau_p: float,
    n1: float,
    p1: float,
    *,
    neutral_bulk_defects: NeutralBulkDefectModel | None = None,
    monovalent_bulk_defects: MonovalentBulkDefectModel | None = None,
    multivalent_bulk_defects: MultivalentBulkDefectModel | None = None,
) -> RecombinationDerivatives:
    """Return bulk SRH and exact local derivatives with respect to n and p."""

    if multivalent_bulk_defects is not None:
        return _mixed_multivalent_srh_recombination_derivatives(
            n,
            p,
            ni_sq,
            tau_n,
            tau_p,
            n1,
            p1,
            neutral_bulk_defects=neutral_bulk_defects,
            monovalent_bulk_defects=monovalent_bulk_defects,
            multivalent_bulk_defects=multivalent_bulk_defects,
        )
    if neutral_bulk_defects is not None and monovalent_bulk_defects is not None:
        raise ValueError("neutral and monovalent bulk-defect models are exclusive")
    if monovalent_bulk_defects is not None:
        return _mixed_monovalent_srh_recombination_derivatives(
            n,
            p,
            ni_sq,
            tau_n,
            tau_p,
            n1,
            p1,
            monovalent_bulk_defects,
        )
    if neutral_bulk_defects is not None:
        return _mixed_srh_recombination_derivatives(
            n,
            p,
            ni_sq,
            tau_n,
            tau_p,
            n1,
            p1,
            neutral_bulk_defects,
        )

    denominator = bulk_srh_denominator(n, p, tau_n, tau_p, n1, p1)
    numerator = n * p - ni_sq
    rate = numerator / denominator
    return RecombinationDerivatives(
        rate=np.asarray(rate),
        electron_density_derivative=np.asarray(
            (p - rate * tau_p) / denominator
        ),
        hole_density_derivative=np.asarray(
            (n - rate * tau_n) / denominator
        ),
    )


def _broadcast_bulk_inputs(
    n: np.ndarray,
    p: np.ndarray,
    ni_sq: float,
    tau_n: float,
    tau_p: float,
    n1: float,
    p1: float,
    model: NeutralBulkDefectModel,
) -> tuple[np.ndarray, ...]:
    arrays = np.broadcast_arrays(
        np.asarray(n, dtype=float),
        np.asarray(p, dtype=float),
        np.asarray(ni_sq, dtype=float),
        np.asarray(tau_n, dtype=float),
        np.asarray(tau_p, dtype=float),
        np.asarray(n1, dtype=float),
        np.asarray(p1, dtype=float),
    )
    if arrays[0].shape != (model.node_count,):
        raise ValueError("neutral bulk-defect state must match the compiled grid")
    return tuple(np.asarray(value) for value in arrays)


def _mixed_srh_recombination(
    n: np.ndarray,
    p: np.ndarray,
    ni_sq: float,
    tau_n: float,
    tau_p: float,
    n1: float,
    p1: float,
    model: NeutralBulkDefectModel,
) -> np.ndarray:
    n_a, p_a, ni_a, tau_n_a, tau_p_a, n1_a, p1_a = _broadcast_bulk_inputs(
        n, p, ni_sq, tau_n, tau_p, n1, p1, model
    )
    result = np.zeros_like(n_a, dtype=float)
    legacy = ~model.explicit_node_mask
    if np.any(legacy):
        denominator = bulk_srh_denominator(
            n_a[legacy],
            p_a[legacy],
            tau_n_a[legacy],
            tau_p_a[legacy],
            n1_a[legacy],
            p1_a[legacy],
        )
        _record_srh_denominator("bulk", denominator)
        result[legacy] = (n_a[legacy] * p_a[legacy] - ni_a[legacy]) / denominator
    for species in model.species:
        if not species.cycle_active:
            continue
        active = species.active_nodes
        denominator = bulk_srh_denominator(
            n_a[active],
            p_a[active],
            species.tau_n_s,
            species.tau_p_s,
            species.n1_m3[active],
            species.p1_m3[active],
        )
        _record_srh_denominator("bulk", denominator)
        result[active] += (
            n_a[active] * p_a[active] - ni_a[active]
        ) / denominator
    return result


def evaluate_neutral_bulk_defects(
    n: np.ndarray,
    p: np.ndarray,
    ni_sq: np.ndarray,
    model: NeutralBulkDefectModel,
) -> NeutralBulkDefectEvaluation:
    """Evaluate exact per-species neutral SRH and analytic carrier tangents."""

    n_a, p_a, ni_a = np.broadcast_arrays(
        np.asarray(n, dtype=float),
        np.asarray(p, dtype=float),
        np.asarray(ni_sq, dtype=float),
    )
    if n_a.shape != (model.node_count,):
        raise ValueError("neutral bulk-defect state must match the compiled grid")
    shape = (len(model.species), model.node_count)
    rates = np.zeros(shape, dtype=float)
    derivative_n = np.zeros(shape, dtype=float)
    derivative_p = np.zeros(shape, dtype=float)
    minima: list[float | None] = []
    for index, species in enumerate(model.species):
        if not species.cycle_active:
            minima.append(None)
            continue
        active = species.active_nodes
        denominator = bulk_srh_denominator(
            n_a[active],
            p_a[active],
            species.tau_n_s,
            species.tau_p_s,
            species.n1_m3[active],
            species.p1_m3[active],
        )
        _record_srh_denominator("bulk", denominator)
        numerator = n_a[active] * p_a[active] - ni_a[active]
        rate = numerator / denominator
        rates[index, active] = rate
        derivative_n[index, active] = (
            p_a[active] - rate * species.tau_p_s
        ) / denominator
        derivative_p[index, active] = (
            n_a[active] - rate * species.tau_n_s
        ) / denominator
        finite = denominator[np.isfinite(denominator)]
        minima.append(float(np.min(finite)) if finite.size else None)
    if len(model.species) == 1:
        total_rate = rates[0].copy()
        total_derivative_n = derivative_n[0].copy()
        total_derivative_p = derivative_p[0].copy()
    else:
        total_rate = np.sum(rates, axis=0)
        total_derivative_n = np.sum(derivative_n, axis=0)
        total_derivative_p = np.sum(derivative_p, axis=0)
    return NeutralBulkDefectEvaluation(
        model_identity_sha256=model.identity_sha256,
        species_identifiers=tuple(item.identifier for item in model.species),
        total=RecombinationDerivatives(
            rate=total_rate,
            electron_density_derivative=total_derivative_n,
            hole_density_derivative=total_derivative_p,
        ),
        per_species_rate_m3_s=rates,
        per_species_electron_derivative_s1=derivative_n,
        per_species_hole_derivative_s1=derivative_p,
        minimum_denominator_s_m3=tuple(minima),
    )


def _mixed_srh_recombination_derivatives(
    n: np.ndarray,
    p: np.ndarray,
    ni_sq: float,
    tau_n: float,
    tau_p: float,
    n1: float,
    p1: float,
    model: NeutralBulkDefectModel,
) -> RecombinationDerivatives:
    n_a, p_a, ni_a, tau_n_a, tau_p_a, n1_a, p1_a = _broadcast_bulk_inputs(
        n, p, ni_sq, tau_n, tau_p, n1, p1, model
    )
    rate = np.zeros_like(n_a, dtype=float)
    derivative_n = np.zeros_like(n_a, dtype=float)
    derivative_p = np.zeros_like(n_a, dtype=float)
    legacy = ~model.explicit_node_mask
    if np.any(legacy):
        denominator = bulk_srh_denominator(
            n_a[legacy],
            p_a[legacy],
            tau_n_a[legacy],
            tau_p_a[legacy],
            n1_a[legacy],
            p1_a[legacy],
        )
        _record_srh_denominator("bulk", denominator)
        local_rate = (n_a[legacy] * p_a[legacy] - ni_a[legacy]) / denominator
        rate[legacy] = local_rate
        derivative_n[legacy] = (
            p_a[legacy] - local_rate * tau_p_a[legacy]
        ) / denominator
        derivative_p[legacy] = (
            n_a[legacy] - local_rate * tau_n_a[legacy]
        ) / denominator
    explicit = evaluate_neutral_bulk_defects(n_a, p_a, ni_a, model).total
    rate[model.explicit_node_mask] = explicit.rate[model.explicit_node_mask]
    derivative_n[model.explicit_node_mask] = (
        explicit.electron_density_derivative[model.explicit_node_mask]
    )
    derivative_p[model.explicit_node_mask] = (
        explicit.hole_density_derivative[model.explicit_node_mask]
    )
    return RecombinationDerivatives(
        rate=rate,
        electron_density_derivative=derivative_n,
        hole_density_derivative=derivative_p,
    )


def _broadcast_monovalent_bulk_inputs(
    n: np.ndarray,
    p: np.ndarray,
    ni_sq: float,
    tau_n: float,
    tau_p: float,
    n1: float,
    p1: float,
    model: MonovalentBulkDefectModel,
) -> tuple[np.ndarray, ...]:
    arrays = np.broadcast_arrays(
        np.asarray(n, dtype=float),
        np.asarray(p, dtype=float),
        np.asarray(ni_sq, dtype=float),
        np.asarray(tau_n, dtype=float),
        np.asarray(tau_p, dtype=float),
        np.asarray(n1, dtype=float),
        np.asarray(p1, dtype=float),
    )
    if arrays[0].shape != (model.node_count,):
        raise ValueError("monovalent bulk-defect state must match the compiled grid")
    return tuple(np.asarray(value) for value in arrays)


def _mixed_monovalent_srh_recombination(
    n: np.ndarray,
    p: np.ndarray,
    ni_sq: float,
    tau_n: float,
    tau_p: float,
    n1: float,
    p1: float,
    model: MonovalentBulkDefectModel,
) -> np.ndarray:
    n_a, p_a, ni_a, tau_n_a, tau_p_a, n1_a, p1_a = (
        _broadcast_monovalent_bulk_inputs(
            n,
            p,
            ni_sq,
            tau_n,
            tau_p,
            n1,
            p1,
            model,
        )
    )
    result = np.zeros_like(n_a, dtype=float)
    explicit = model.explicit_node_mask
    legacy = ~explicit
    if np.any(legacy):
        denominator = bulk_srh_denominator(
            n_a[legacy],
            p_a[legacy],
            tau_n_a[legacy],
            tau_p_a[legacy],
            n1_a[legacy],
            p1_a[legacy],
        )
        _record_srh_denominator("bulk", denominator)
        result[legacy] = (
            n_a[legacy] * p_a[legacy] - ni_a[legacy]
        ) / denominator
    evaluation = evaluate_monovalent_bulk_defects(n_a, p_a, model)
    result[explicit] = evaluation.total_recombination_rate_m3_s[explicit]
    return result


def _mixed_monovalent_srh_recombination_derivatives(
    n: np.ndarray,
    p: np.ndarray,
    ni_sq: float,
    tau_n: float,
    tau_p: float,
    n1: float,
    p1: float,
    model: MonovalentBulkDefectModel,
) -> RecombinationDerivatives:
    n_a, p_a, ni_a, tau_n_a, tau_p_a, n1_a, p1_a = (
        _broadcast_monovalent_bulk_inputs(
            n,
            p,
            ni_sq,
            tau_n,
            tau_p,
            n1,
            p1,
            model,
        )
    )
    rate = np.zeros_like(n_a, dtype=float)
    derivative_n = np.zeros_like(n_a, dtype=float)
    derivative_p = np.zeros_like(n_a, dtype=float)
    explicit = model.explicit_node_mask
    legacy = ~explicit
    if np.any(legacy):
        denominator = bulk_srh_denominator(
            n_a[legacy],
            p_a[legacy],
            tau_n_a[legacy],
            tau_p_a[legacy],
            n1_a[legacy],
            p1_a[legacy],
        )
        _record_srh_denominator("bulk", denominator)
        local_rate = (
            n_a[legacy] * p_a[legacy] - ni_a[legacy]
        ) / denominator
        rate[legacy] = local_rate
        derivative_n[legacy] = (
            p_a[legacy] - local_rate * tau_p_a[legacy]
        ) / denominator
        derivative_p[legacy] = (
            n_a[legacy] - local_rate * tau_n_a[legacy]
        ) / denominator
    evaluation = evaluate_monovalent_bulk_defects(n_a, p_a, model)
    rate[explicit] = evaluation.total_recombination_rate_m3_s[explicit]
    derivative_n[explicit] = (
        evaluation.total_recombination_derivative_n_s1[explicit]
    )
    derivative_p[explicit] = (
        evaluation.total_recombination_derivative_p_s1[explicit]
    )
    return RecombinationDerivatives(
        rate=rate,
        electron_density_derivative=derivative_n,
        hole_density_derivative=derivative_p,
    )


def _multivalent_mixed_inputs(
    n: np.ndarray,
    p: np.ndarray,
    ni_sq: float,
    tau_n: float,
    tau_p: float,
    n1: float,
    p1: float,
    *,
    neutral_bulk_defects: NeutralBulkDefectModel | None,
    monovalent_bulk_defects: MonovalentBulkDefectModel | None,
    multivalent_bulk_defects: MultivalentBulkDefectModel,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    values = np.broadcast_arrays(
        np.asarray(n, dtype=float),
        np.asarray(p, dtype=float),
        np.asarray(ni_sq, dtype=float),
        np.asarray(tau_n, dtype=float),
        np.asarray(tau_p, dtype=float),
        np.asarray(n1, dtype=float),
        np.asarray(p1, dtype=float),
    )
    n_a, p_a, ni_a, tau_n_a, tau_p_a, n1_a, p1_a = values
    # The neutral/monovalent exclusivity invariant is independent of the
    # multivalent model: the compiler nulls the neutral inventory whenever a
    # monovalent one exists, so the two arriving together means a caller built
    # its own inconsistent partition. Keep the check on this path too rather
    # than letting the multivalent dispatch silently accept it.
    if neutral_bulk_defects is not None and monovalent_bulk_defects is not None:
        raise ValueError("neutral and monovalent bulk-defect models are exclusive")
    if n_a.shape != (multivalent_bulk_defects.node_count,):
        raise ValueError("multivalent bulk-defect inputs must match the grid")
    masks = [multivalent_bulk_defects.explicit_node_mask]
    if neutral_bulk_defects is not None:
        if neutral_bulk_defects.node_count != n_a.size:
            raise ValueError("neutral and multivalent models must share one grid")
        masks.append(neutral_bulk_defects.explicit_node_mask)
    if monovalent_bulk_defects is not None:
        if monovalent_bulk_defects.node_count != n_a.size:
            raise ValueError("monovalent and multivalent models must share one grid")
        masks.append(monovalent_bulk_defects.explicit_node_mask)
    ownership = np.sum(np.stack(masks, axis=0).astype(np.int8), axis=0)
    if np.any(ownership > 1):
        raise ValueError("compiled explicit bulk-defect models overlap on grid nodes")
    return (*values, ownership.astype(bool))


def _mixed_multivalent_srh_recombination(
    n: np.ndarray,
    p: np.ndarray,
    ni_sq: float,
    tau_n: float,
    tau_p: float,
    n1: float,
    p1: float,
    *,
    neutral_bulk_defects: NeutralBulkDefectModel | None,
    monovalent_bulk_defects: MonovalentBulkDefectModel | None,
    multivalent_bulk_defects: MultivalentBulkDefectModel,
) -> np.ndarray:
    n_a, p_a, ni_a, tau_n_a, tau_p_a, n1_a, p1_a, explicit = _multivalent_mixed_inputs(
        n,
        p,
        ni_sq,
        tau_n,
        tau_p,
        n1,
        p1,
        neutral_bulk_defects=neutral_bulk_defects,
        monovalent_bulk_defects=monovalent_bulk_defects,
        multivalent_bulk_defects=multivalent_bulk_defects,
    )
    result = np.zeros_like(n_a, dtype=float)
    legacy = ~explicit
    if np.any(legacy):
        denominator = bulk_srh_denominator(
            n_a[legacy],
            p_a[legacy],
            tau_n_a[legacy],
            tau_p_a[legacy],
            n1_a[legacy],
            p1_a[legacy],
        )
        _record_srh_denominator("bulk", denominator)
        result[legacy] = (n_a[legacy] * p_a[legacy] - ni_a[legacy]) / denominator
    if neutral_bulk_defects is not None:
        mask = neutral_bulk_defects.explicit_node_mask
        neutral = evaluate_neutral_bulk_defects(
            n_a,
            p_a,
            ni_a,
            neutral_bulk_defects,
        )
        result[mask] = neutral.total.rate[mask]
    if monovalent_bulk_defects is not None:
        mask = monovalent_bulk_defects.explicit_node_mask
        monovalent = evaluate_monovalent_bulk_defects(
            n_a,
            p_a,
            monovalent_bulk_defects,
        )
        result[mask] = monovalent.total_recombination_rate_m3_s[mask]
    mask = multivalent_bulk_defects.explicit_node_mask
    multivalent = evaluate_multivalent_bulk_defects(
        n_a,
        p_a,
        multivalent_bulk_defects,
    )
    result[mask] = multivalent.total_recombination_rate_m3_s[mask]
    return result


def _mixed_multivalent_srh_recombination_derivatives(
    n: np.ndarray,
    p: np.ndarray,
    ni_sq: float,
    tau_n: float,
    tau_p: float,
    n1: float,
    p1: float,
    *,
    neutral_bulk_defects: NeutralBulkDefectModel | None,
    monovalent_bulk_defects: MonovalentBulkDefectModel | None,
    multivalent_bulk_defects: MultivalentBulkDefectModel,
) -> RecombinationDerivatives:
    n_a, p_a, ni_a, tau_n_a, tau_p_a, n1_a, p1_a, explicit = _multivalent_mixed_inputs(
        n,
        p,
        ni_sq,
        tau_n,
        tau_p,
        n1,
        p1,
        neutral_bulk_defects=neutral_bulk_defects,
        monovalent_bulk_defects=monovalent_bulk_defects,
        multivalent_bulk_defects=multivalent_bulk_defects,
    )
    rate = np.zeros_like(n_a, dtype=float)
    derivative_n = np.zeros_like(n_a, dtype=float)
    derivative_p = np.zeros_like(n_a, dtype=float)
    legacy = ~explicit
    if np.any(legacy):
        denominator = bulk_srh_denominator(
            n_a[legacy],
            p_a[legacy],
            tau_n_a[legacy],
            tau_p_a[legacy],
            n1_a[legacy],
            p1_a[legacy],
        )
        _record_srh_denominator("bulk", denominator)
        local_rate = (n_a[legacy] * p_a[legacy] - ni_a[legacy]) / denominator
        rate[legacy] = local_rate
        derivative_n[legacy] = (
            p_a[legacy] - local_rate * tau_p_a[legacy]
        ) / denominator
        derivative_p[legacy] = (
            n_a[legacy] - local_rate * tau_n_a[legacy]
        ) / denominator
    if neutral_bulk_defects is not None:
        mask = neutral_bulk_defects.explicit_node_mask
        neutral = evaluate_neutral_bulk_defects(
            n_a,
            p_a,
            ni_a,
            neutral_bulk_defects,
        ).total
        rate[mask] = neutral.rate[mask]
        derivative_n[mask] = neutral.electron_density_derivative[mask]
        derivative_p[mask] = neutral.hole_density_derivative[mask]
    if monovalent_bulk_defects is not None:
        mask = monovalent_bulk_defects.explicit_node_mask
        monovalent = evaluate_monovalent_bulk_defects(
            n_a,
            p_a,
            monovalent_bulk_defects,
        )
        rate[mask] = monovalent.total_recombination_rate_m3_s[mask]
        derivative_n[mask] = monovalent.total_recombination_derivative_n_s1[mask]
        derivative_p[mask] = monovalent.total_recombination_derivative_p_s1[mask]
    mask = multivalent_bulk_defects.explicit_node_mask
    multivalent = evaluate_multivalent_bulk_defects(
        n_a,
        p_a,
        multivalent_bulk_defects,
    )
    rate[mask] = multivalent.total_recombination_rate_m3_s[mask]
    derivative_n[mask] = multivalent.total_recombination_derivative_n_s1[mask]
    derivative_p[mask] = multivalent.total_recombination_derivative_p_s1[mask]
    return RecombinationDerivatives(
        rate=rate,
        electron_density_derivative=derivative_n,
        hole_density_derivative=derivative_p,
    )


def radiative_recombination(
    n: np.ndarray, p: np.ndarray, ni_sq: float, B_rad: float,
) -> np.ndarray:
    """Bimolecular radiative recombination rate [m⁻³ s⁻¹]."""
    return B_rad * (n * p - ni_sq)


def radiative_recombination_derivatives(
    n: np.ndarray,
    p: np.ndarray,
    ni_sq: float,
    B_rad: float,
) -> RecombinationDerivatives:
    """Return radiative recombination and its exact local derivatives."""

    rate = B_rad * (n * p - ni_sq)
    return RecombinationDerivatives(
        rate=np.asarray(rate),
        electron_density_derivative=np.asarray(B_rad * p),
        hole_density_derivative=np.asarray(B_rad * n),
    )


def auger_recombination(
    n: np.ndarray, p: np.ndarray, ni_sq: float,
    C_n: float, C_p: float,
) -> np.ndarray:
    """Auger recombination rate [m⁻³ s⁻¹]."""
    return (C_n * n + C_p * p) * (n * p - ni_sq)


def auger_recombination_derivatives(
    n: np.ndarray,
    p: np.ndarray,
    ni_sq: float,
    C_n: float,
    C_p: float,
) -> RecombinationDerivatives:
    """Return Auger recombination and its exact local derivatives."""

    numerator = n * p - ni_sq
    coefficient = C_n * n + C_p * p
    rate = coefficient * numerator
    return RecombinationDerivatives(
        rate=np.asarray(rate),
        electron_density_derivative=np.asarray(C_n * numerator + coefficient * p),
        hole_density_derivative=np.asarray(C_p * numerator + coefficient * n),
    )


def interface_recombination(
    n: float, p: float, ni_sq: float,
    n1: float, p1: float,
    v_n: float, v_p: float,
) -> float:
    """Interface (surface) SRH recombination rate [m⁻² s⁻¹].

    Parameters
    ----------
    n, p : carrier densities at the interface node [m⁻³]
    ni_sq : intrinsic carrier density squared [m⁻⁶]
    n1, p1 : SRH trap-level carrier densities [m⁻³]
    v_n, v_p : surface recombination velocities [m/s]
    """
    if v_n <= 0.0 or v_p <= 0.0:
        # A single blocked capture channel blocks the full SRH cycle: the
        # denominator diverges as v -> 0, so the physical limit is R -> 0.
        # Guarding both (not just the both-zero case) also prevents a
        # ZeroDivisionError for configs with one-sided passivation.
        return 0.0
    denominator = interface_srh_denominator(n, p, n1, p1, v_n, v_p)
    _record_srh_denominator("interface", np.asarray(denominator))
    return (n * p - ni_sq) / denominator


def interface_recombination_derivatives(
    n: float,
    p: float,
    ni_sq: float,
    n1: float,
    p1: float,
    v_n: float,
    v_p: float,
) -> RecombinationDerivatives:
    """Return surface SRH and exact local derivatives with respect to n and p."""

    if v_n <= 0.0 or v_p <= 0.0:
        zero = np.asarray(0.0)
        return RecombinationDerivatives(
            rate=zero,
            electron_density_derivative=zero,
            hole_density_derivative=zero,
        )
    denominator = interface_srh_denominator(n, p, n1, p1, v_n, v_p)
    numerator = n * p - ni_sq
    rate = numerator / denominator
    return RecombinationDerivatives(
        rate=np.asarray(rate),
        electron_density_derivative=np.asarray(
            (p - rate / v_p) / denominator
        ),
        hole_density_derivative=np.asarray(
            (n - rate / v_n) / denominator
        ),
    )


def total_recombination(
    n: np.ndarray, p: np.ndarray, ni_sq: float,
    tau_n: float, tau_p: float, n1: float, p1: float,
    B_rad: float, C_n: float, C_p: float,
    *,
    neutral_bulk_defects: NeutralBulkDefectModel | None = None,
    monovalent_bulk_defects: MonovalentBulkDefectModel | None = None,
    multivalent_bulk_defects: MultivalentBulkDefectModel | None = None,
) -> np.ndarray:
    """Sum of SRH + radiative + Auger [m⁻³ s⁻¹]."""
    return (
        srh_recombination(
            n,
            p,
            ni_sq,
            tau_n,
            tau_p,
            n1,
            p1,
            neutral_bulk_defects=neutral_bulk_defects,
            monovalent_bulk_defects=monovalent_bulk_defects,
            multivalent_bulk_defects=multivalent_bulk_defects,
        )
        + radiative_recombination(n, p, ni_sq, B_rad)
        + auger_recombination(n, p, ni_sq, C_n, C_p)
    )


def total_recombination_derivatives(
    n: np.ndarray,
    p: np.ndarray,
    ni_sq: float,
    tau_n: float,
    tau_p: float,
    n1: float,
    p1: float,
    B_rad: float,
    C_n: float,
    C_p: float,
    *,
    neutral_bulk_defects: NeutralBulkDefectModel | None = None,
    monovalent_bulk_defects: MonovalentBulkDefectModel | None = None,
    multivalent_bulk_defects: MultivalentBulkDefectModel | None = None,
) -> RecombinationDerivatives:
    """Return SRH + radiative + Auger and exact local derivatives."""

    srh = srh_recombination_derivatives(
        n,
        p,
        ni_sq,
        tau_n,
        tau_p,
        n1,
        p1,
        neutral_bulk_defects=neutral_bulk_defects,
        monovalent_bulk_defects=monovalent_bulk_defects,
        multivalent_bulk_defects=multivalent_bulk_defects,
    )
    radiative = radiative_recombination_derivatives(n, p, ni_sq, B_rad)
    auger = auger_recombination_derivatives(n, p, ni_sq, C_n, C_p)
    return RecombinationDerivatives(
        rate=srh.rate + radiative.rate + auger.rate,
        electron_density_derivative=(
            srh.electron_density_derivative
            + radiative.electron_density_derivative
            + auger.electron_density_derivative
        ),
        hole_density_derivative=(
            srh.hole_density_derivative
            + radiative.hole_density_derivative
            + auger.hole_density_derivative
        ),
    )


def total_recombination_at_node(
    n: float,
    p: float,
    ni_sq: float,
    tau_n: float,
    tau_p: float,
    n1: float,
    p1: float,
    B_rad: float,
    C_n: float,
    C_p: float,
    *,
    node: int,
    neutral_bulk_defects: NeutralBulkDefectModel | None = None,
    monovalent_bulk_defects: MonovalentBulkDefectModel | None = None,
    multivalent_bulk_defects: MultivalentBulkDefectModel | None = None,
) -> float:
    """Scalar total rate using the same lifetime/explicit node dispatch."""

    if neutral_bulk_defects is not None and monovalent_bulk_defects is not None:
        raise ValueError("neutral and monovalent bulk-defect models are exclusive")
    if multivalent_bulk_defects is not None:
        multivalent_region = next(
            (
                region
                for region in multivalent_bulk_defects.regions
                if region.active_nodes[node]
            ),
            None,
        )
        if multivalent_region is not None:
            srh = float(
                evaluate_multivalent_source_defect_closure(
                    n,
                    p,
                    multivalent_region.species,
                    band_gap_eV=multivalent_region.band_gap_eV,
                    effective_conduction_dos_m3=(
                        multivalent_region.effective_conduction_dos_m3
                    ),
                    effective_valence_dos_m3=(
                        multivalent_region.effective_valence_dos_m3
                    ),
                    temperature_K=multivalent_region.temperature_K,
                ).total_recombination_rate_m3_s
            )
            return float(
                srh
                + radiative_recombination(np.asarray(n), np.asarray(p), ni_sq, B_rad)
                + auger_recombination(np.asarray(n), np.asarray(p), ni_sq, C_n, C_p)
            )
    monovalent_region = None
    if monovalent_bulk_defects is not None:
        monovalent_region = next(
            (
                region
                for region in monovalent_bulk_defects.regions
                if region.active_nodes[node]
            ),
            None,
        )
    neutral_explicit = bool(
        neutral_bulk_defects is not None
        and neutral_bulk_defects.explicit_node_mask[node]
    )
    if not neutral_explicit and monovalent_region is None:
        srh = float(
            srh_recombination(
                np.asarray(n),
                np.asarray(p),
                ni_sq,
                tau_n,
                tau_p,
                n1,
                p1,
            )
        )
    elif neutral_explicit:
        assert neutral_bulk_defects is not None
        srh = 0.0
        for species in neutral_bulk_defects.species:
            if not species.active_nodes[node] or not species.cycle_active:
                continue
            denominator = float(
                bulk_srh_denominator(
                    np.asarray(n),
                    np.asarray(p),
                    species.tau_n_s,
                    species.tau_p_s,
                    float(species.n1_m3[node]),
                    float(species.p1_m3[node]),
                )
            )
            _record_srh_denominator("bulk", np.asarray(denominator))
            srh += (n * p - ni_sq) / denominator
    else:
        assert monovalent_region is not None
        srh = evaluate_monovalent_source_defect_closure(
            n,
            p,
            monovalent_region.species,
            band_gap_eV=monovalent_region.band_gap_eV,
            effective_conduction_dos_m3=(
                monovalent_region.effective_conduction_dos_m3
            ),
            effective_valence_dos_m3=(
                monovalent_region.effective_valence_dos_m3
            ),
            temperature_K=monovalent_region.temperature_K,
            energy_quadrature_order=(
                monovalent_region.energy_quadrature_order
            ),
            energy_expansions=monovalent_region.source_expansions,
        ).total_recombination_rate_m3_s.item()
    return float(
        srh
        + radiative_recombination(np.asarray(n), np.asarray(p), ni_sq, B_rad)
        + auger_recombination(np.asarray(n), np.asarray(p), ni_sq, C_n, C_p)
    )
