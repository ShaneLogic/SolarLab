"""Simulation mode resolution.

Three tiered modes control which physics upgrades are active at runtime.
All three tiers act as a **ceiling** — they enable a feature only when the
configuration actually provides the required parameters (e.g. FULL does not
build the TMM generation profile unless at least one layer sets
``optical_material``; FULL does not activate selective contacts unless at
least one of the four outer ``S_*`` values is non-null). The mode therefore
never forces an upgrade that the config cannot support, which keeps legacy
YAML files working unchanged.

Tier flag matrix:

    flag                             LEGACY  FAST  FULL
    use_thermionic_emission            off    on    on
    use_tmm_optics                     off    on    on
    use_dual_ions                      off    on    on
    use_trap_profile                   off    on    on
    use_temperature_scaling            off    on    on
    use_photon_recycling               off    on    on
    use_radiative_reabsorption         off    off   on
    use_field_dependent_mobility       off    off   on
    use_selective_contacts             off    off   on

Design intent:

- ``legacy`` — IonMonger-reproduction baseline. Every Phase 1+ physics
  upgrade is disabled so that benchmark comparisons against IonMonger /
  driftfusion / Courtier-2019 remain numerically meaningful.

- ``fast`` — every "build-once" physics upgrade (Phase 1 band-offset + TE,
  Phase 2 TMM optics, Phase 3.1 photon recycling, dual ions, trap profiles,
  temperature scaling) is ON, but the three "per-RHS" upgrades that break
  the MaterialArrays build-once invariant — Phase 3.1b radiative
  reabsorption, Phase 3.2 field-dependent mobility, Phase 3.3 selective
  contacts — are OFF. This is the best-accuracy-per-unit-time tier and
  should be the default for most interactive / benchmark work where the
  device geometry already captures selective behaviour via layer doping
  rather than outer-contact Robin BCs.

- ``full`` — every physics flag is on. Use when the config explicitly needs
  self-consistent radiative reabsorption (Phase 3.1b, per-RHS G_rad source
  on absorbers), field-dependent mobility (``v_sat_{n,p}``,
  ``pf_gamma_{n,p}`` set), or selective / Schottky contacts (any
  ``S_{n,p}_{left,right}`` set). Full tier is safe to use on legacy YAML
  too — the per-RHS hooks self-disable when the config leaves their opt-in
  parameters at None/zero — but it exercises more code paths than strictly
  necessary for pre-Phase-3.1b stacks.
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
    use_radiative_reabsorption: bool = True
    use_field_dependent_mobility: bool = True
    use_selective_contacts: bool = True


LEGACY = SimulationMode(
    name="legacy",
    use_thermionic_emission=False,
    use_tmm_optics=False,
    use_dual_ions=False,
    use_trap_profile=False,
    use_temperature_scaling=False,
    use_photon_recycling=False,
    use_radiative_reabsorption=False,
    use_field_dependent_mobility=False,
    use_selective_contacts=False,
)

FAST = SimulationMode(
    # FAST tier: every "build-once" physics upgrade is on, but the three
    # per-RHS upgrades that break the MaterialArrays build-once invariant
    # (Phase 3.1b radiative reabsorption, Phase 3.2 field-dependent
    # mobility, Phase 3.3 selective contacts) stay off. Best accuracy-per-
    # unit-time default for benchmark work.
    name="fast",
    use_thermionic_emission=True,
    use_tmm_optics=True,
    use_dual_ions=True,
    use_trap_profile=True,
    use_temperature_scaling=True,
    use_photon_recycling=True,
    use_radiative_reabsorption=False,
    use_field_dependent_mobility=False,
    use_selective_contacts=False,
)

FULL = SimulationMode(
    name="full",
    use_thermionic_emission=True,
    use_tmm_optics=True,
    use_dual_ions=True,
    use_trap_profile=True,
    use_temperature_scaling=True,
    use_photon_recycling=True,
    use_radiative_reabsorption=True,
    use_field_dependent_mobility=True,
    use_selective_contacts=True,
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
