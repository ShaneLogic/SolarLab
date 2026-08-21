"""Validation helpers with explicit numerical-protocol semantics."""

from .grid_convergence import ConvergenceSeries
from .numerical_certificate import (
    CellResult,
    LaneDefinition,
    NumericalCertificate,
    load_refinement_registry,
)

__all__ = [
    "CellResult",
    "ConvergenceSeries",
    "LaneDefinition",
    "NumericalCertificate",
    "load_refinement_registry",
]
