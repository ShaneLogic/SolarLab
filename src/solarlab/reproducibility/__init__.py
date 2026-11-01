"""Source-verified registry migration, without scientific execution admission."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from solarlab.reproducibility.registry import MigrationRegistry, load_registry, validate_registry_pair
    from solarlab.reproducibility.sources import SourceRef, SourceSet, read_sources

__all__ = ["MigrationRegistry", "load_registry", "validate_registry_pair", "SourceRef", "SourceSet", "read_sources"]


def __getattr__(name: str) -> object:
    if name in ("MigrationRegistry", "load_registry", "validate_registry_pair"):
        from solarlab.reproducibility import registry

        value = getattr(registry, name)
    elif name in ("SourceRef", "SourceSet", "read_sources"):
        from solarlab.reproducibility import sources

        value = getattr(sources, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
