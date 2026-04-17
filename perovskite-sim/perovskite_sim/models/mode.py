"""Simulation mode resolution.

Three tiered modes control which physics upgrades are active at runtime:

- ``legacy``: single-species ions, Beer-Lambert optics, no thermionic emission,
  uniform traps, fixed T=300 K. This reproduces the pre-upgrade behavior
  and the IonMonger-style baseline so that benchmark comparisons remain
  meaningful.
- ``fast``: identical physics flags as legacy but reserved for future
  optimisations that trade accuracy for speed (e.g. TMM → effective alpha).
  Today it behaves the same as legacy; the name is kept to pin the API.
- ``full`` (default): every physics upgrade is active. TMM optical generation,
  thermionic emission capping, dual-species ions, position-dependent traps,
  and temperature scaling all honour whatever the config supplies.

Modes act as a **ceiling**: they enable a feature only when the configuration
actually provides the required parameters. For example, ``full`` mode does not
build the TMM generation profile unless at least one layer sets
``optical_material``. The mode therefore never forces an upgrade that the
config cannot support, which keeps legacy YAML files working unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

ModeName = Literal["legacy", "fast", "full"]


@dataclass(frozen=True)
class SimulationMode:
    """Immutable bundle of physics-feature flags.

    Fields follow the "off by default in legacy, on in full" convention so
    that ``SimulationMode()`` is a safe baseline.
    """
    name: ModeName = "full"
    use_thermionic_emission: bool = True
    use_tmm_optics: bool = True
    use_dual_ions: bool = True
    use_trap_profile: bool = True
    use_temperature_scaling: bool = True
    use_photon_recycling: bool = True
    use_field_dependent_mobility: bool = True


LEGACY = SimulationMode(
    name="legacy",
    use_thermionic_emission=False,
    use_tmm_optics=False,
    use_dual_ions=False,
    use_trap_profile=False,
    use_temperature_scaling=False,
    use_photon_recycling=False,
    use_field_dependent_mobility=False,
)

FAST = SimulationMode(
    name="fast",
    use_thermionic_emission=False,
    use_tmm_optics=False,
    use_dual_ions=False,
    use_trap_profile=False,
    use_temperature_scaling=False,
    use_photon_recycling=False,
    use_field_dependent_mobility=False,
)

FULL = SimulationMode(
    name="full",
    use_thermionic_emission=True,
    use_tmm_optics=True,
    use_dual_ions=True,
    use_trap_profile=True,
    use_temperature_scaling=True,
    use_photon_recycling=True,
    use_field_dependent_mobility=True,
)


_PRESETS: dict[str, SimulationMode] = {
    "legacy": LEGACY,
    "fast": FAST,
    "full": FULL,
}


def resolve_mode(mode: Union[ModeName, SimulationMode, None]) -> SimulationMode:
    """Return a ``SimulationMode`` for the given mode argument.

    Parameters
    ----------
    mode
        Either a mode name (``"legacy" | "fast" | "full"``), an already-built
        ``SimulationMode`` instance, or ``None``. ``None`` resolves to
        ``FULL`` so that callers that ignore the mode keyword get every
        physics upgrade they have configured.

    Raises
    ------
    ValueError
        If ``mode`` is a string that is not one of the known presets.
    """
    if mode is None:
        return FULL
    if isinstance(mode, SimulationMode):
        return mode
    if isinstance(mode, str):
        key = mode.lower()
        if key not in _PRESETS:
            raise ValueError(
                f"Unknown simulation mode {mode!r}. "
                f"Expected one of {sorted(_PRESETS)}."
            )
        return _PRESETS[key]
    raise TypeError(
        f"Expected mode to be a str, SimulationMode, or None; got {type(mode).__name__}"
    )
