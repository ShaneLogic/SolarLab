"""Constitutive kinetics, Wegscheider detailed balance, and exact derivatives.

Implements the five iodine reaction channels of Bitton & Tessler (Energy &
Environmental Science, 16, 2621-2628, 2023, SI Table S1 & Eqs. 3-4) in strict
SI units (m, s, m^-3) with both Wegscheider cyclic detailed balance (default)
and irreversible mass-action closures, plus exact analytical state Jacobians.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts import ContractError, Vector, frozen_array
from solarlab.physics.chemistry.stoichiometry import (
    CANONICAL_IODINE_NETWORK,
    StoichiometricNetwork,
)

__all__ = [
    "IODINE_KINETICS_SCHEMA",
    "IodineKineticsParameters",
    "IodineReactionEvaluation",
    "IodineThermodynamicMode",
    "ReactionEOrder",
    "equilibrium_iodine_populations_m3",
    "evaluate_iodine_kinetics",
    "smoluchowski_rate_m3_s",
]

IODINE_KINETICS_SCHEMA = "solarlab.iodine_kinetics.v1"

IodineThermodynamicMode = Literal[
    "wegscheider_detailed_balance",
    "irreversible_mass_action",
]
ReactionEOrder = Literal["bimolecular_n", "termolecular_n2"]

_THERMODYNAMIC_MODES: frozenset[str] = frozenset(get_args(IodineThermodynamicMode))
_REACTION_E_ORDERS: frozenset[str] = frozenset(get_args(ReactionEOrder))


def _vector(values: ArrayLike, size: int, name: str) -> Vector:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(size, float(arr), dtype=np.float64)
    if arr.shape != (size,) or not np.all(np.isfinite(arr)):
        raise ContractError(f"{name}_must_be_finite_1d_vector_of_length_{size}")
    return frozen_array(arr)


def smoluchowski_rate_m3_s(
    diffusivity_a_m2_s: ArrayLike,
    diffusivity_b_m2_s: ArrayLike,
    capture_radius_m: float,
) -> Vector:
    """Diffusion-limited bimolecular rate ``4 * pi * (D_A + D_B) * R_AB`` [m^3/s].

    For homodimerization ``2 I^0 -> I_2`` pass ``D_A = D_0`` and ``D_B = D_0``
    so ``D_A + D_B = 2 D_0``, matching Bitton & Tessler SI Table S1.
    """
    radius = float(capture_radius_m)
    if not math.isfinite(radius) or radius < 0.0:
        raise ContractError("capture_radius_m_must_be_finite_nonnegative")
    da = np.asarray(diffusivity_a_m2_s, dtype=np.float64)
    db = np.asarray(diffusivity_b_m2_s, dtype=np.float64)
    if not (np.all(np.isfinite(da)) and np.all(np.isfinite(db))):
        raise ContractError("diffusivities_must_be_finite")
    if np.any(da < 0.0) or np.any(db < 0.0):
        raise ContractError("diffusivities_must_be_nonnegative")
    rate = 4.0 * math.pi * (da + db) * radius
    return frozen_array(np.atleast_1d(np.asarray(rate, dtype=np.float64)))


@dataclass(frozen=True, slots=True)
class IodineKineticsParameters:
    """Node-aligned SI kinetic arrays on the chemical NodeSet (size ``M``)."""

    k_a_m3_s: Vector
    k_b_s_1: Vector
    k_c_m3_s: Vector
    k_d_m3_s: Vector
    k_e_m3_s: Vector
    n_i_sq_m6: Vector
    K_diss_m3: Vector
    n_e_star_m3: Vector
    k_e_3body_m6_s: Vector | None = None
    n_1_iodine_m3: Vector | None = None
    thermodynamic_mode: IodineThermodynamicMode = "wegscheider_detailed_balance"
    reaction_e_order: ReactionEOrder = "bimolecular_n"
    network: StoichiometricNetwork = CANONICAL_IODINE_NETWORK

    def __post_init__(self) -> None:
        if self.thermodynamic_mode not in _THERMODYNAMIC_MODES:
            raise ContractError("unknown_iodine_thermodynamic_mode")
        if self.reaction_e_order not in _REACTION_E_ORDERS:
            raise ContractError("unknown_reaction_e_order")
        raw_ka = np.atleast_1d(np.asarray(self.k_a_m3_s, dtype=np.float64))
        if raw_ka.ndim != 1 or raw_ka.size < 1:
            raise ContractError("k_a_m3_s_must_be_nonempty_1d_vector")
        size = int(raw_ka.size)

        k_a = _vector(self.k_a_m3_s, size, "k_a_m3_s")
        k_b = _vector(self.k_b_s_1, size, "k_b_s_1")
        k_c = _vector(self.k_c_m3_s, size, "k_c_m3_s")
        k_d = _vector(self.k_d_m3_s, size, "k_d_m3_s")
        k_e = _vector(self.k_e_m3_s, size, "k_e_m3_s")
        n_i_sq = _vector(self.n_i_sq_m6, size, "n_i_sq_m6")
        k_diss = _vector(self.K_diss_m3, size, "K_diss_m3")
        n_star = _vector(self.n_e_star_m3, size, "n_e_star_m3")
        k_e_3 = (
            _vector(0.0, size, "k_e_3body_m6_s")
            if self.k_e_3body_m6_s is None
            else _vector(self.k_e_3body_m6_s, size, "k_e_3body_m6_s")
        )

        for label, arr in (
            ("k_a_m3_s", k_a),
            ("k_b_s_1", k_b),
            ("k_c_m3_s", k_c),
            ("k_d_m3_s", k_d),
            ("k_e_m3_s", k_e),
            ("k_e_3body_m6_s", k_e_3),
        ):
            if np.any(arr < 0.0):
                raise ContractError(f"{label}_must_be_nonnegative")
        if np.any(n_i_sq <= 0.0):
            raise ContractError("n_i_sq_m6_must_be_strictly_positive")
        if np.any(k_diss <= 0.0):
            raise ContractError("K_diss_m3_must_be_strictly_positive")

        if self.thermodynamic_mode == "wegscheider_detailed_balance":
            if (
                self.reaction_e_order == "bimolecular_n"
                and np.any((k_e > 0.0) & (n_star <= 0.0))
            ):
                raise ContractError("wegscheider_bimolecular_e_requires_positive_n_e_star")
            if self.n_1_iodine_m3 is None:
                if np.any((k_a > 0.0) & (k_b <= 0.0)):
                    raise ContractError(
                        "wegscheider_detailed_balance_requires_positive_k_b_where_k_a_positive"
                    )
                with np.errstate(divide="ignore", invalid="ignore"):
                    derived_n1 = np.where(k_b > 0.0, (k_a * n_i_sq) / k_b, 0.0)
                if not np.all(np.isfinite(derived_n1)):
                    raise ContractError("nonfinite_derived_n_1_iodine_m3")
                n_1 = frozen_array(derived_n1)
            else:
                n_1 = _vector(self.n_1_iodine_m3, size, "n_1_iodine_m3")
                if np.any(n_1 < 0.0):
                    raise ContractError("n_1_iodine_m3_must_be_nonnegative")
        else:
            if np.any(n_star < 0.0):
                raise ContractError("n_e_star_m3_must_be_nonnegative")
            n_1 = (
                _vector(0.0, size, "n_1_iodine_m3")
                if self.n_1_iodine_m3 is None
                else _vector(self.n_1_iodine_m3, size, "n_1_iodine_m3")
            )

        object.__setattr__(self, "k_a_m3_s", k_a)
        object.__setattr__(self, "k_b_s_1", k_b)
        object.__setattr__(self, "k_c_m3_s", k_c)
        object.__setattr__(self, "k_d_m3_s", k_d)
        object.__setattr__(self, "k_e_m3_s", k_e)
        object.__setattr__(self, "k_e_3body_m6_s", k_e_3)
        object.__setattr__(self, "n_i_sq_m6", n_i_sq)
        object.__setattr__(self, "K_diss_m3", k_diss)
        object.__setattr__(self, "n_e_star_m3", n_star)
        object.__setattr__(self, "n_1_iodine_m3", n_1)

    @property
    def size(self) -> int:
        return int(self.k_a_m3_s.size)

    @property
    def identity(self) -> str:
        digest = sha256(IODINE_KINETICS_SCHEMA.encode())
        digest.update(self.network.identity.encode())
        digest.update(
            json.dumps(
                [self.thermodynamic_mode, self.reaction_e_order, self.size],
                separators=(",", ":"),
            ).encode()
        )
        assert self.k_e_3body_m6_s is not None and self.n_1_iodine_m3 is not None
        for arr in (
            self.k_a_m3_s,
            self.k_b_s_1,
            self.k_c_m3_s,
            self.k_d_m3_s,
            self.k_e_m3_s,
            self.k_e_3body_m6_s,
            self.n_i_sq_m6,
            self.K_diss_m3,
            self.n_e_star_m3,
            self.n_1_iodine_m3,
        ):
            digest.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class IodineReactionEvaluation:
    """Evaluated reaction rates, projected species sources, and exact Jacobians."""

    rates_m3_s: NDArray[np.float64]
    sources_m3_s: NDArray[np.float64]
    dsources_dstate: NDArray[np.float64] | None = None


def evaluate_iodine_kinetics(
    n_m3: ArrayLike,
    p_m3: ArrayLike,
    c_minus_m3: ArrayLike,
    c_atom_m3: ArrayLike,
    c_mol_m3: ArrayLike,
    params: IodineKineticsParameters,
    *,
    derivatives: bool = True,
) -> IodineReactionEvaluation:
    """Evaluate all five iodine reactions, net sources, and exact partials.

    State ordering in ``sources_m3_s`` (shape ``(5, M)``) and ``dsources_dstate``
    (shape ``(5, 5, M)``) follows :data:`IODINE_SPECIES`:
    ``(electron, hole, iodide, iodine_atom, iodine_molecule)``.
    """
    size = params.size
    n = np.asarray(n_m3, dtype=np.float64)
    p = np.asarray(p_m3, dtype=np.float64)
    cm = np.asarray(c_minus_m3, dtype=np.float64)
    c0 = np.asarray(c_atom_m3, dtype=np.float64)
    c2 = np.asarray(c_mol_m3, dtype=np.float64)
    for label, arr in (
        ("n_m3", n),
        ("p_m3", p),
        ("c_minus_m3", cm),
        ("c_atom_m3", c0),
        ("c_mol_m3", c2),
    ):
        if arr.shape != (size,) or not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
            raise ContractError(f"{label}_must_be_finite_nonnegative_vector")

    k_a = params.k_a_m3_s
    k_b = params.k_b_s_1
    k_c = params.k_c_m3_s
    k_d = params.k_d_m3_s
    k_e = params.k_e_m3_s
    assert params.k_e_3body_m6_s is not None and params.n_1_iodine_m3 is not None
    k_e3 = params.k_e_3body_m6_s
    n1 = params.n_1_iodine_m3
    k_diss = params.K_diss_m3
    n_star = params.n_e_star_m3
    reversible = params.thermodynamic_mode == "wegscheider_detailed_balance"

    # R_a = k_a * c_- * p, R_b = k_b * c_0
    r_a = k_a * cm * p
    r_b = k_b * c0

    # R_c and R_d
    if reversible:
        r_c = k_c * (c0 * n - n1 * cm)
        r_d = k_d * (c0 * c0 - k_diss * c2)
        alpha_e = (n1 * n1) / k_diss
    else:
        r_c = k_c * c0 * n
        r_d = k_d * (c0 * c0)
        alpha_e = np.zeros(size, dtype=np.float64)

    # R_e
    if params.reaction_e_order == "bimolecular_n":
        if reversible:
            denom = n + n_star
            # Safe where k_e == 0 and n == n_star == 0.
            safe_denom = np.where(denom > 0.0, denom, 1.0)
            pref = np.where(k_e > 0.0, k_e / safe_denom, 0.0)
            r_e = pref * (c2 * (n * n) - alpha_e * (cm * cm))
        else:
            r_e = k_e * c2 * n
    else:
        if reversible:
            r_e = k_e3 * (c2 * (n * n) - alpha_e * (cm * cm))
        else:
            r_e = k_e3 * c2 * (n * n)

    rates = np.stack((r_a, r_b, r_c, r_d, r_e), axis=0)
    if not np.all(np.isfinite(rates)):
        raise ContractError("nonfinite_iodine_reaction_rates")
    sources = params.network.project_sources(rates)

    if not derivatives:
        return IodineReactionEvaluation(
            rates_m3_s=frozen_array(rates),
            sources_m3_s=frozen_array(sources),
            dsources_dstate=None,
        )

    # Exact drates_dstate of shape (5 reactions, 5 state vars, M nodes)
    dr: NDArray[np.float64] = np.zeros((5, 5, size), dtype=np.float64)

    # dR_a / d(p, c_-)
    dr[0, 1] = k_a * cm
    dr[0, 2] = k_a * p

    # dR_b / dc_0
    dr[1, 3] = k_b

    # dR_c / d(n, c_-, c_0)
    dr[2, 0] = k_c * c0
    if reversible:
        dr[2, 2] = -k_c * n1
    dr[2, 3] = k_c * n

    # dR_d / d(c_0, c_2)
    dr[3, 3] = 2.0 * k_d * c0
    if reversible:
        dr[3, 4] = -k_d * k_diss

    # dR_e / d(n, c_-, c_2)
    if params.reaction_e_order == "bimolecular_n":
        if reversible:
            denom = n + n_star
            safe_denom = np.where(denom > 0.0, denom, 1.0)
            inv_d = np.where(k_e > 0.0, 1.0 / safe_denom, 0.0)
            # Cancellation-free positive form:
            # d/dn [k_e (c2 n^2 - alpha cm^2)/(n + n*)]
            #   = k_e [c2 n (n + 2 n*) + alpha cm^2] / (n + n*)^2
            dr[4, 0] = (
                k_e
                * (c2 * n * (n + 2.0 * n_star) + alpha_e * (cm * cm))
                * (inv_d * inv_d)
            )
            dr[4, 2] = -2.0 * k_e * alpha_e * cm * inv_d
            dr[4, 4] = k_e * (n * n) * inv_d
        else:
            dr[4, 0] = k_e * c2
            dr[4, 4] = k_e * n
    else:
        dr[4, 0] = 2.0 * k_e3 * c2 * n
        if reversible:
            dr[4, 2] = -2.0 * k_e3 * alpha_e * cm
        dr[4, 4] = k_e3 * (n * n)

    nu_float = params.network.nu.astype(np.float64)
    dsources = np.einsum("sr,rjm->sjm", nu_float, dr)
    if not np.all(np.isfinite(dsources)):
        raise ContractError("nonfinite_iodine_reaction_jacobian")

    return IodineReactionEvaluation(
        rates_m3_s=frozen_array(rates),
        sources_m3_s=frozen_array(sources),
        dsources_dstate=frozen_array(dsources),
    )


def equilibrium_iodine_populations_m3(
    total_iodine_atoms_m3: ArrayLike,
    n_eq_m3: ArrayLike,
    params: IodineKineticsParameters,
) -> tuple[Vector, Vector, Vector]:
    """Solve exact local thermal equilibrium ``(c_-, c_0, c_2)`` conserving iodine.

    Given local elemental iodine density ``c_tot = c_- + c_0 + 2 c_2 >= 0`` and
    equilibrium electron density ``n_eq > 0`` (with ``n_eq * p_eq = n_i^2``),
    detailed balance fixes ``c_0 = r_0 * c_-`` (``r_0 = n_1 / n_eq``) and
    ``c_2 = (r_0^2 / K_diss) * c_-^2``. Solving the monotone quadratic
    ``(2 r_0^2 / K_diss) c_-^2 + (1 + r_0) c_- - c_tot = 0`` via the
    cancellation-free Citardauq root gives exact machine-precision equilibrium.
    """
    size = params.size
    c_tot = _vector(total_iodine_atoms_m3, size, "total_iodine_atoms_m3")
    n_eq = _vector(n_eq_m3, size, "n_eq_m3")
    if np.any(c_tot < 0.0):
        raise ContractError("total_iodine_atoms_m3_must_be_nonnegative")
    if np.any(n_eq <= 0.0):
        raise ContractError("n_eq_m3_must_be_strictly_positive")
    assert params.n_1_iodine_m3 is not None
    r0 = params.n_1_iodine_m3 / n_eq
    quad_a = 2.0 * (r0 * r0) / params.K_diss_m3
    lin_b = 1.0 + r0
    disc = np.sqrt(lin_b * lin_b + 4.0 * quad_a * c_tot)
    c_minus = (2.0 * c_tot) / (lin_b + disc)
    c_atom = r0 * c_minus
    c_mol = (c_atom * c_atom) / params.K_diss_m3
    return frozen_array(c_minus), frozen_array(c_atom), frozen_array(c_mol)
