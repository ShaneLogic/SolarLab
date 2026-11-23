"""Stoichiometric network contracts and exact nullspace conservation invariants.

Enforces microscopic charge neutrality ``z^T @ nu == 0`` and elemental iodine
atom conservation ``m_I^T @ nu == 0`` at construction time on every reaction
column before any ODE or PDE evaluation can proceed.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts import ContractError

__all__ = [
    "CANONICAL_IODINE_NETWORK",
    "IODINE_REACTIONS",
    "IODINE_SPECIES",
    "STOICHIOMETRY_SCHEMA",
    "StoichiometricNetwork",
    "canonical_iodine_network",
]

STOICHIOMETRY_SCHEMA = "solarlab.iodine_stoichiometry.v1"

IODINE_SPECIES: tuple[str, ...] = (
    "electron",
    "hole",
    "iodide",
    "iodine_atom",
    "iodine_molecule",
)
IODINE_REACTIONS: tuple[str, ...] = ("R_a", "R_b", "R_c", "R_d", "R_e")


def _frozen_int_array(values: ArrayLike, *, ndim: int, name: str) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.ndim != ndim or raw.dtype.kind not in ("i", "u"):
        raise ContractError(f"{name}_must_be_{ndim}d_integer_array")
    arr = np.array(raw, dtype=np.int64, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, slots=True)
class StoichiometricNetwork:
    """Immutable stoichiometric network enforcing charge and atom conservation."""

    species_ids: tuple[str, ...]
    reaction_ids: tuple[str, ...]
    nu: NDArray[np.int64]
    charge_numbers: NDArray[np.int64]
    iodine_atom_counts: NDArray[np.int64]

    def __post_init__(self) -> None:
        species = tuple(str(item) for item in self.species_ids)
        reactions = tuple(str(item) for item in self.reaction_ids)
        if not species or len(set(species)) != len(species):
            raise ContractError("stoichiometry_requires_distinct_species_ids")
        if not reactions or len(set(reactions)) != len(reactions):
            raise ContractError("stoichiometry_requires_distinct_reaction_ids")
        nu = _frozen_int_array(self.nu, ndim=2, name="nu")
        charges = _frozen_int_array(
            self.charge_numbers, ndim=1, name="charge_numbers"
        )
        atoms = _frozen_int_array(
            self.iodine_atom_counts, ndim=1, name="iodine_atom_counts"
        )
        if nu.shape != (len(species), len(reactions)):
            raise ContractError("stoichiometric_matrix_shape_mismatch")
        if charges.shape != (len(species),) or atoms.shape != (len(species),):
            raise ContractError("conservation_vector_shape_mismatch")
        if np.any(atoms < 0):
            raise ContractError("iodine_atom_counts_must_be_nonnegative")
        charge_defect = charges @ nu
        if not np.all(charge_defect == 0):
            raise ContractError("stoichiometric_charge_conservation_violated")
        atom_defect = atoms @ nu
        if not np.all(atom_defect == 0):
            raise ContractError("stoichiometric_iodine_conservation_violated")
        object.__setattr__(self, "species_ids", species)
        object.__setattr__(self, "reaction_ids", reactions)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "charge_numbers", charges)
        object.__setattr__(self, "iodine_atom_counts", atoms)

    @property
    def identity(self) -> str:
        payload = [
            STOICHIOMETRY_SCHEMA,
            list(self.species_ids),
            list(self.reaction_ids),
            self.nu.tolist(),
            self.charge_numbers.tolist(),
            self.iodine_atom_counts.tolist(),
        ]
        return sha256(
            json.dumps(payload, separators=(",", ":")).encode()
        ).hexdigest()

    def project_sources(self, rates: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map reaction rates ``(R, M)`` or ``(R,)`` to species sources ``nu @ rates``."""
        arr = np.asarray(rates, dtype=np.float64)
        if arr.shape[0] != len(self.reaction_ids):
            raise ContractError("reaction_rate_dimension_mismatch")
        return np.asarray(self.nu.astype(np.float64) @ arr, dtype=np.float64)


def canonical_iodine_network() -> StoichiometricNetwork:
    """Return the 5-species, 5-reaction Bitton & Tessler (2023) iodine network.

    Columns correspond to ``[R_a, R_b, R_c, R_d, R_e]``:
      * ``R_a``: ``I^- + h^+ -> I^0``
      * ``R_b``: ``I^0 -> I^- + h^+``
      * ``R_c``: ``I^0 + e^- <-> I^-``
      * ``R_d``: ``2 I^0 <-> I_2``
      * ``R_e``: ``I_2 + 2 e^- <-> 2 I^-``
    """
    nu = np.array(
        [
            [0, 0, -1, 0, -2],   # electron (n)
            [-1, 1, 0, 0, 0],    # hole (p)
            [-1, 1, 1, 0, 2],    # iodide (c_-)
            [1, -1, -1, -2, 0],  # iodine_atom (c_0)
            [0, 0, 0, 1, -1],    # iodine_molecule (c_2)
        ],
        dtype=np.int64,
    )
    charges = np.array([-1, 1, -1, 0, 0], dtype=np.int64)
    atoms = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    return StoichiometricNetwork(
        species_ids=IODINE_SPECIES,
        reaction_ids=IODINE_REACTIONS,
        nu=nu,
        charge_numbers=charges,
        iodine_atom_counts=atoms,
    )


CANONICAL_IODINE_NETWORK: StoichiometricNetwork = canonical_iodine_network()
