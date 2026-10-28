"""The bound bulk-recombination reaction: one instance for rate, tangent and source.

``BulkRecombinationReaction`` binds the per-node coefficients once (after
temperature, grading, band-gap-narrowing and trap-profile resolution by the
caller) together with the explicit-defect node ownership. One call to
``evaluate`` produces the channel rates, their exact local tangents, the
SRH denominators (the observation the historical code routed through a
process-wide context variable) and the carrier sources. The electron and
hole continuity equations both consume ``-R`` from the same evaluated array,
so they can never see two different reaction populations.

Node ownership: a node owned by an explicit defect model carries that
model's SRH and no lifetime-form SRH; every other node carries the
lifetime-form law. Radiative and Auger channels are independent of SRH
ownership and act on every node.

Interface switch point (S1/S2 in the W3b design note): ``evaluate`` takes the
physical electron/hole density arrays directly until the public
``StateView``/``StateIncrement`` subset and the typed ``TermSink`` sources
exist in ``solarlab.contracts``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from solarlab.physics.recombination._validation import frozen_float_array, require_finite
from solarlab.physics.recombination.bulk import (
    RecombinationRate,
    auger_recombination,
    lifetime_srh_denominator,
    lifetime_srh_from_denominator,
    radiative_recombination,
    sum_recombination_rates,
)
from solarlab.physics.recombination.explicit_neutral import (
    NeutralExplicitDefectEvaluation,
    NeutralExplicitDefectModel,
    evaluate_neutral_explicit_defects,
)

__all__ = [
    "BulkRecombinationCoefficients",
    "BulkRecombinationEvaluation",
    "BulkRecombinationReaction",
    "CARRIER_SOURCE_UNIT",
    "CarrierSources",
    "RecombinationDomainError",
]

#: Unit of every unweighted cell source returned here; the assembly layer
#: multiplies by the cell measure exactly once.
CARRIER_SOURCE_UNIT = "m^-3 s^-1"


class RecombinationDomainError(ValueError):
    """A state lies outside the rate laws' physical domain."""


def _node_coefficient(value: object, name: str, node_count: int, *, positive: bool) -> np.ndarray:
    array = frozen_float_array(np.broadcast_to(np.asarray(value, dtype=float), (node_count,)), name=name)
    if np.any(np.isnan(array)):
        raise ValueError(f"{name} must not be NaN")
    if positive:
        if np.any(array <= 0.0):
            raise ValueError(f"{name} must be positive or +inf")
    else:
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"{name} must be finite and non-negative")
    return array


@dataclass(frozen=True, slots=True)
class BulkRecombinationCoefficients:
    """Resolved per-node coefficients on one 1D node set (SI units).

    ``tau_*`` [s] may be ``+inf`` (a blocked capture leg); every other
    coefficient is finite and non-negative. Scalars broadcast to ``(N,)``.
    """

    node_count: int
    ni_sq_m6: np.ndarray
    tau_n_s: np.ndarray
    tau_p_s: np.ndarray
    n1_m3: np.ndarray
    p1_m3: np.ndarray
    radiative_m3_s: np.ndarray
    auger_n_m6_s: np.ndarray
    auger_p_m6_s: np.ndarray

    def __post_init__(self) -> None:
        if type(self.node_count) is not int or self.node_count < 1:
            raise ValueError("node_count must be a positive integer")
        count = self.node_count
        for name in ("ni_sq_m6", "n1_m3", "p1_m3", "radiative_m3_s", "auger_n_m6_s", "auger_p_m6_s"):
            object.__setattr__(self, name, _node_coefficient(getattr(self, name), name, count, positive=False))
        for name in ("tau_n_s", "tau_p_s"):
            object.__setattr__(self, name, _node_coefficient(getattr(self, name), name, count, positive=True))


@dataclass(frozen=True, slots=True)
class CarrierSources:
    """Unweighted cell sources of one reaction evaluation.

    ``electron`` and ``hole`` are ``-R`` built from the same rate array; the
    local Jacobian blocks are ``d(-R)/dn`` and ``d(-R)/dp`` for both rows.
    """

    electron: np.ndarray
    hole: np.ndarray
    d_source_dn: np.ndarray | None
    d_source_dp: np.ndarray | None
    unit: str = CARRIER_SOURCE_UNIT
    stoichiometry: tuple[tuple[str, int], ...] = (("electron", -1), ("hole", -1))


@dataclass(frozen=True, slots=True)
class BulkRecombinationEvaluation:
    """Every observable of one reaction evaluation at one state."""

    total: RecombinationRate
    srh: RecombinationRate
    radiative: RecombinationRate
    auger: RecombinationRate
    lifetime_nodes: np.ndarray
    srh_denominators_s_m3: np.ndarray
    explicit: NeutralExplicitDefectEvaluation | None
    electron_density_m3: np.ndarray
    hole_density_m3: np.ndarray

    @property
    def minimum_srh_denominator_s_m3(self) -> float | None:
        finite = self.srh_denominators_s_m3[np.isfinite(self.srh_denominators_s_m3)]
        return float(np.min(finite)) if finite.size else None

    def carrier_sources(self) -> CarrierSources:
        rate = self.total.rate
        source = -rate
        source.setflags(write=False)
        if not self.total.has_derivatives:
            return CarrierSources(source, source, None, None)
        d_dn: Any = -np.asarray(self.total.d_rate_dn)
        d_dp: Any = -np.asarray(self.total.d_rate_dp)
        d_dn.setflags(write=False)
        d_dp.setflags(write=False)
        return CarrierSources(source, source, d_dn, d_dp)

    def in_domain(self) -> bool:
        """Non-negative densities and strictly positive finite SRH denominators."""

        denominators = self.srh_denominators_s_m3
        return bool(
            np.all(self.electron_density_m3 >= 0.0)
            and np.all(self.hole_density_m3 >= 0.0)
            and np.all(denominators[np.isfinite(denominators)] > 0.0)
            and not np.any(np.isnan(denominators))
        )

    def require_domain(self) -> None:
        if not self.in_domain():
            raise RecombinationDomainError(
                "bulk recombination evaluated outside its domain: densities must be "
                "non-negative and every SRH denominator positive"
            )


@dataclass(frozen=True, slots=True)
class BulkRecombinationReaction:
    """Lifetime SRH + radiative + Auger with optional explicit SRH ownership."""

    coefficients: BulkRecombinationCoefficients
    explicit_defects: NeutralExplicitDefectModel | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.coefficients, BulkRecombinationCoefficients):
            raise TypeError("coefficients must be BulkRecombinationCoefficients")
        model = self.explicit_defects
        if model is not None:
            if not isinstance(model, NeutralExplicitDefectModel):
                raise TypeError("explicit_defects must be a NeutralExplicitDefectModel")
            if model.node_count != self.coefficients.node_count:
                raise ValueError("explicit defect model and coefficients must share one grid")

    @property
    def node_count(self) -> int:
        return self.coefficients.node_count

    @property
    def lifetime_nodes(self) -> np.ndarray:
        """Nodes whose SRH channel is the lifetime-form law."""

        if self.explicit_defects is None:
            mask: np.ndarray = np.ones(self.node_count, dtype=bool)
        else:
            mask = ~self.explicit_defects.owned_nodes
        mask.setflags(write=False)
        return mask

    def evaluate(self, n: np.ndarray, p: np.ndarray, *, derivatives: bool = True) -> BulkRecombinationEvaluation:
        """Evaluate every channel at one state (densities [m^-3], shape ``(N,)``).

        Any finite state is evaluated with the historical arithmetic; the
        physical-domain gate is the explicit ``require_domain`` call, so a
        trial state is never silently clipped here.
        """

        c = self.coefficients
        # Owned read-only copies: the evaluation keeps the state it judged.
        n_a = np.array(n, dtype=float, copy=True)
        p_a = np.array(p, dtype=float, copy=True)
        n_a.setflags(write=False)
        p_a.setflags(write=False)
        if n_a.shape != (c.node_count,) or p_a.shape != (c.node_count,):
            raise ValueError("electron and hole densities must have shape (node_count,)")
        require_finite(n_a, name="electron density")
        require_finite(p_a, name="hole density")
        model = self.explicit_defects
        lifetime = self.lifetime_nodes
        if model is None:
            denominators = lifetime_srh_denominator(n_a, p_a, c.tau_n_s, c.tau_p_s, c.n1_m3, c.p1_m3)
            srh = lifetime_srh_from_denominator(
                n_a, p_a, c.ni_sq_m6, c.tau_n_s, c.tau_p_s, denominators, derivatives=derivatives
            )
            explicit = None
        else:
            srh, denominators, explicit = self._owned_srh(n_a, p_a, lifetime, model, derivatives)
        radiative = radiative_recombination(n_a, p_a, c.ni_sq_m6, c.radiative_m3_s, derivatives=derivatives)
        auger = auger_recombination(n_a, p_a, c.ni_sq_m6, c.auger_n_m6_s, c.auger_p_m6_s, derivatives=derivatives)
        denominators = np.asarray(denominators, dtype=float)
        denominators.setflags(write=False)
        return BulkRecombinationEvaluation(
            total=sum_recombination_rates(srh, radiative, auger),
            srh=srh,
            radiative=radiative,
            auger=auger,
            lifetime_nodes=lifetime,
            srh_denominators_s_m3=denominators,
            explicit=explicit,
            electron_density_m3=n_a,
            hole_density_m3=p_a,
        )

    def _owned_srh(
        self,
        n_a: np.ndarray,
        p_a: np.ndarray,
        lifetime: np.ndarray,
        model: NeutralExplicitDefectModel,
        derivatives: bool,
    ) -> tuple[RecombinationRate, np.ndarray, NeutralExplicitDefectEvaluation]:
        c = self.coefficients
        rate: np.ndarray = np.zeros(c.node_count, dtype=float)
        d_dn: np.ndarray = np.zeros(c.node_count, dtype=float)
        d_dp: np.ndarray = np.zeros(c.node_count, dtype=float)
        parts: list[np.ndarray] = []
        if np.any(lifetime):
            denominator = lifetime_srh_denominator(
                n_a[lifetime], p_a[lifetime], c.tau_n_s[lifetime], c.tau_p_s[lifetime],
                c.n1_m3[lifetime], c.p1_m3[lifetime],
            )
            local = lifetime_srh_from_denominator(
                n_a[lifetime], p_a[lifetime], c.ni_sq_m6[lifetime],
                c.tau_n_s[lifetime], c.tau_p_s[lifetime], denominator,
            )
            rate[lifetime] = local.rate
            d_dn[lifetime] = local.d_rate_dn
            d_dp[lifetime] = local.d_rate_dp
            parts.append(np.ravel(denominator))
        explicit = evaluate_neutral_explicit_defects(n_a, p_a, c.ni_sq_m6, model)
        owned = model.owned_nodes
        rate[owned] = explicit.total_rate_m3_s[owned]
        d_dn[owned] = explicit.total_d_rate_dn_s1[owned]
        d_dp[owned] = explicit.total_d_rate_dp_s1[owned]
        parts.extend(np.ravel(item) for item in explicit.per_species_denominators_s_m3 if item is not None)
        denominators = np.concatenate(parts) if parts else np.empty(0, dtype=float)
        for array in (rate, d_dn, d_dp):
            array.setflags(write=False)
        if derivatives:
            return RecombinationRate(rate, d_dn, d_dp), denominators, explicit
        return RecombinationRate(rate, None, None), denominators, explicit
