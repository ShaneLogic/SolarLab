"""SCAPS-1D parameter compatibility adapters.

This subpackage translates SCAPS-style microscopic defect and material
parameters (``sigma``, ``v_th``, ``N_t``, ``E_t``, ``N_C``, ``N_V``) into the
SolarLab lumped parameter set (``tau_n``, ``tau_p``, ``n1``, ``p1``, ``ni``,
``v_eff`` at interfaces) without modifying any solver, model, or boundary-
condition code.

Module layout:
    defects.py   -- microscopic SRH lifetime and interface velocity
    materials.py -- ni from effective DOS plus cgs->SI helpers
    loader.py    -- read a SCAPS-shape YAML and return a ``DeviceStack``

The existing ``perovskite_sim.sweeps.device_parameter_sweep`` module already
exposes ``cm3_to_m3``, ``cms_to_ms`` and ``srh_n1_p1_from_trap_depth``; this
package re-exports them so SCAPS adapters live behind a single import
namespace.
"""
from __future__ import annotations

from perovskite_sim.sweeps.device_parameter_sweep import (
    cm3_to_m3,
    cms_to_ms,
    srh_n1_p1_from_trap_depth,
)

from perovskite_sim.scaps_compat.defects import (
    interface_surface_velocity,
    srh_lifetime,
)
from perovskite_sim.scaps_compat.loader import load_scaps_yaml
from perovskite_sim.scaps_compat.materials import ni_from_dos

__all__ = [
    "cm3_to_m3",
    "cms_to_ms",
    "interface_surface_velocity",
    "load_scaps_yaml",
    "ni_from_dos",
    "srh_lifetime",
    "srh_n1_p1_from_trap_depth",
]
