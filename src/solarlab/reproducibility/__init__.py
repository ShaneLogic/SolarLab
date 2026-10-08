"""Source-verified registry migration, without scientific execution admission."""

from solarlab.reproducibility.registry import MigrationRegistry, load_registry, validate_registry_pair
from solarlab.reproducibility.sources import SourceRef, SourceSet, read_sources

__all__ = ["MigrationRegistry", "load_registry", "validate_registry_pair", "SourceRef", "SourceSet", "read_sources"]
