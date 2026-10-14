"""Placeholders for P02-02 interfaces whose shape waits on the G2-R route decision.

Each name is importable so downstream code and reviews can see the boundary,
but nothing is frozen: calling or instantiating a placeholder raises
``ContractError("pending_g2r:<name>")``. Do not add private substitutes for
these interfaces in physics or discretization code; G2-R fixes variables,
storage, coordinates, error norms and the accepted-step semantics first.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, NoReturn

from solarlab.contracts.errors import ContractError

__all__ = ["PENDING_G2R", "PendingG2RInterface", "pending_interface"]

# Name -> why its signature depends on G2-R (ExecutionV2 §4.2 item 6, D17).
PENDING_G2R = MappingProxyType({
    "StorageLaw": "physical storage Q(t, y, a) and its partials follow the selected storage/coordinate route",
    "StoragePartials": "partials are defined on the selected solver coordinates",
    "PhysicalStorage": "storage value/delta callbacks follow the selected storage route",
    "ImplicitSystem": "dense residual/linearization composition depends on the IDA and research-step contract",
    "ImplicitProblem": "solver-facing residual protocol is part of the route decision",
    "ValidatedProblem": "binds storage, structure and conservative callbacks of the selected route",
    "SparseStructure": "sparse layout and assembly belong to P04-02 on the selected storage",
    "SparseLinearization": "complete Fy/Fydot partials depend on the selected storage route",
    "PhysicalScaling": "physical error norm and scaling are decided by G2-R",
    "AcceptedStep": "accepted-interval semantics depend on the product/research backends",
    "CoordinateMap": "production coordinates (linear/log/relative, words) are a route choice",
    "StateCodec": "state encoding and precision words are a route choice",
    "PhysicalLinearForm": "precision observation interface pending G2-R and D25",
    "RateView": "full-word rate representation pending G2-R",
    "BoundLinearSource": "precision observation interface pending G2-R and D25",
    "LinearFormResult": "precision observation interface pending G2-R and D25",
})


class PendingG2RInterface:
    """A named, unusable placeholder; see :data:`PENDING_G2R` for the reason."""

    __slots__ = ("name", "reason")

    def __init__(self, name: str):
        if name not in PENDING_G2R:
            raise ContractError("unknown_pending_interface")
        self.name = name
        self.reason = PENDING_G2R[name]

    def __call__(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise ContractError(f"pending_g2r:{self.name}")

    def __getattr__(self, attribute: str) -> NoReturn:
        # Protocol lookups (copy, inspect, pickle) must keep normal semantics.
        if attribute.startswith("__") and attribute.endswith("__"):
            raise AttributeError(attribute)
        raise ContractError(f"pending_g2r:{object.__getattribute__(self, 'name')}")

    def __repr__(self) -> str:
        return f"PendingG2RInterface({self.name!r})"


def pending_interface(name: str) -> PendingG2RInterface:
    return PendingG2RInterface(name)


StorageLaw = pending_interface("StorageLaw")
StoragePartials = pending_interface("StoragePartials")
PhysicalStorage = pending_interface("PhysicalStorage")
ImplicitSystem = pending_interface("ImplicitSystem")
ImplicitProblem = pending_interface("ImplicitProblem")
ValidatedProblem = pending_interface("ValidatedProblem")
SparseStructure = pending_interface("SparseStructure")
SparseLinearization = pending_interface("SparseLinearization")
PhysicalScaling = pending_interface("PhysicalScaling")
AcceptedStep = pending_interface("AcceptedStep")
CoordinateMap = pending_interface("CoordinateMap")
StateCodec = pending_interface("StateCodec")
PhysicalLinearForm = pending_interface("PhysicalLinearForm")
RateView = pending_interface("RateView")
BoundLinearSource = pending_interface("BoundLinearSource")
LinearFormResult = pending_interface("LinearFormResult")
