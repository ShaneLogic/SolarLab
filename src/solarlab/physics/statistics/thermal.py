"""Thermal voltage and the historical 300 K reference (M class).

``thermal_voltage`` evaluates ``k_B * T / q`` in the legacy order, so every
temperature gives the same binary64 value as the legacy engine.
``REFERENCE_THERMAL_VOLTAGE_V`` is the module constant the legacy engine uses
for its 300 K paths (``perovskite_sim.constants.V_T``), for example the
historical endpoint Fermi-level estimate.
"""

from __future__ import annotations

import math

from solarlab.contracts.units import ELEMENTARY_CHARGE_C

__all__ = [
    "BOLTZMANN_CONSTANT_J_K", "ELEMENTARY_CHARGE_C", "REFERENCE_TEMPERATURE_K",
    "REFERENCE_THERMAL_VOLTAGE_V", "thermal_voltage",
]

# Exact SI 2019 defining constant; same binary64 value as the legacy engine.
BOLTZMANN_CONSTANT_J_K = 1.380649e-23
REFERENCE_TEMPERATURE_K = 300.0
REFERENCE_THERMAL_VOLTAGE_V = (
    BOLTZMANN_CONSTANT_J_K * REFERENCE_TEMPERATURE_K / ELEMENTARY_CHARGE_C
)


def thermal_voltage(temperature_K: float) -> float:
    """Thermal voltage ``k_B T / q`` [V] for a finite positive temperature."""
    temperature = float(temperature_K)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_K must be finite and positive")
    return BOLTZMANN_CONSTANT_J_K * temperature / ELEMENTARY_CHARGE_C
