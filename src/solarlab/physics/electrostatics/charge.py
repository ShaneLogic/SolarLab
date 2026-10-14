"""The shared charge ledger: one declaration of every charged population.

Poisson rows, Gauss jumps at interfaces and electrode charges must read their
charge from this single ledger instead of re-deriving ``rho`` locally. Each
volumetric entry declares its integer charge number and an optional
compensating background (e.g. a mobile-ion reference profile); interface
sheet charges are separate, areal entries that only enter through a Gauss
jump. Populations are non-negative densities; their charge sign comes only
from ``charge_number``.

``space_charge_C_m3`` folds the entries of one group left to right and
multiplies by ``q`` once per group, then adds the groups in declaration order.
That order reproduces the historical
``Q*(p - n + (P - P0) - N_A + N_D)`` followed by ``rho -= Q*(P_neg - P_neg0)``
bit-for-bit when declared as two groups in that order.

The bitwise equivalence holds only on the ledger's domain of non-negative
densities and backgrounds. The historical expression also accepts negative
(trial) carrier or ion densities; the ledger rejects them, so building it from
an unaccepted signed trial state is a behaviour change that must be decided
and classified explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts import ELEMENTARY_CHARGE_C, frozen_array

__all__ = ["ChargeEntry", "ChargeLedger", "SheetChargeEntry", "SPACE_CHARGE_GROUP"]

SPACE_CHARGE_GROUP = "space_charge"


@dataclass(frozen=True)
class ChargeEntry:
    """A volumetric population [m^-3] on the nodes, with its charge number."""

    id: str
    owner: str
    charge_number: int
    density_m3: NDArray[np.float64]
    background_m3: NDArray[np.float64] | None = None
    group: str = SPACE_CHARGE_GROUP

    def __post_init__(self) -> None:
        if not self.id or not self.owner or not self.group:
            raise ValueError("charge entries need an id, owner and group")
        if type(self.charge_number) is not int or self.charge_number == 0:
            raise ValueError("charge_number must be a nonzero integer")
        density = frozen_array(self.density_m3)
        if density.ndim != 1 or np.any(density < 0.0):
            raise ValueError(f"{self.id}: densities must be a non-negative 1-D array")
        object.__setattr__(self, "density_m3", density)
        if self.background_m3 is not None:
            background = frozen_array(self.background_m3)
            if background.shape != density.shape or np.any(background < 0.0):
                raise ValueError(f"{self.id}: background must match and be non-negative")
            object.__setattr__(self, "background_m3", background)

    def net_population(self) -> NDArray[np.float64]:
        if self.background_m3 is None:
            return self.density_m3
        net: NDArray[np.float64] = self.density_m3 - self.background_m3
        return net


@dataclass(frozen=True)
class SheetChargeEntry:
    """An areal charge [C/m^2] on one electrical interface (Gauss-jump source)."""

    id: str
    owner: str
    interface: int
    charge_C_m2: float

    def __post_init__(self) -> None:
        if not self.id or not self.owner:
            raise ValueError("sheet charges need an id and owner")
        if type(self.interface) is not int or self.interface < 0:
            raise ValueError("sheet charge interface must be a non-negative index")
        value = float(self.charge_C_m2)
        if not math.isfinite(value):
            raise ValueError("sheet charge must be finite")
        object.__setattr__(self, "charge_C_m2", value)


@dataclass(frozen=True)
class ChargeLedger:
    """Immutable ledger; ``with_*`` returns a new ledger and rejects duplicates."""

    node_count: int
    interface_count: int = 0
    entries: tuple[ChargeEntry, ...] = ()
    sheets: tuple[SheetChargeEntry, ...] = ()
    _groups: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.node_count) is not int or self.node_count < 1:
            raise ValueError("node_count must be a positive integer")
        if type(self.interface_count) is not int or self.interface_count < 0:
            raise ValueError("interface_count must be a non-negative integer")
        entries, sheets = tuple(self.entries), tuple(self.sheets)
        ids = [entry.id for entry in entries] + [sheet.id for sheet in sheets]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate charge ledger entry")
        for entry in entries:
            if entry.density_m3.shape != (self.node_count,):
                raise ValueError(f"{entry.id}: density does not match the node count")
        for sheet in sheets:
            if sheet.interface >= self.interface_count:
                raise ValueError(f"{sheet.id}: interface outside the stack")
        groups: list[str] = []
        for entry in entries:
            if entry.group not in groups:
                groups.append(entry.group)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "sheets", sheets)
        object.__setattr__(self, "_groups", tuple(groups))

    def with_entry(self, entry: ChargeEntry) -> ChargeLedger:
        return ChargeLedger(self.node_count, self.interface_count,
                            self.entries + (entry,), self.sheets)

    def with_sheet(self, sheet: SheetChargeEntry) -> ChargeLedger:
        return ChargeLedger(self.node_count, self.interface_count,
                            self.entries, self.sheets + (sheet,))

    @property
    def groups(self) -> tuple[str, ...]:
        return self._groups

    def entry(self, entry_id: str) -> ChargeEntry:
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        raise KeyError(entry_id)

    def space_charge_C_m3(self) -> NDArray[np.float64]:
        """Volumetric charge density [C/m^3] on the nodes."""
        rho: NDArray[np.float64] | None = None
        for group in self._groups:
            accumulated: NDArray[np.float64] | None = None
            for entry in self.entries:
                if entry.group != group:
                    continue
                term = entry.net_population()
                z = entry.charge_number
                if accumulated is None:
                    accumulated = term if z == 1 else (-term if z == -1 else z * term)
                elif z == 1:
                    accumulated = accumulated + term
                elif z == -1:
                    accumulated = accumulated - term
                else:
                    accumulated = accumulated + z * term
            assert accumulated is not None
            contribution: NDArray[np.float64] = ELEMENTARY_CHARGE_C * accumulated
            rho = contribution if rho is None else rho + contribution
        if rho is None:
            return np.zeros(self.node_count)
        return rho

    def space_charge_derivative(self, entry_id: str) -> float:
        """``d rho / d density`` [C] of one entry; its background enters with minus."""
        return float(self.entry(entry_id).charge_number * ELEMENTARY_CHARGE_C)

    def sheet_charge_C_m2(self) -> NDArray[np.float64]:
        """Total areal charge per interface, summed in declaration order."""
        sigma = np.zeros(self.interface_count)
        for sheet in self.sheets:
            sigma[sheet.interface] = sigma[sheet.interface] + sheet.charge_C_m2
        return sigma

    def total_charge_C_m2(self, cell_measures_m: ArrayLike) -> float:
        """Net charge per unit cross-section: ``sum(w*rho) + sum(sigma)``."""
        measures = np.asarray(cell_measures_m, dtype=float)
        if measures.shape != (self.node_count,) or np.any(measures <= 0.0):
            raise ValueError("cell measures must match the ledger nodes")
        volumetric = (measures * self.space_charge_C_m3()).tolist()
        return math.fsum(volumetric + self.sheet_charge_C_m2().tolist())
