"""Electrode-referenced quasi-Fermi coordinates of the classical electronic blocks.

Blocks ``potential``, ``electron`` and ``hole`` hold ``phi`` [V] and the
dimensionless offsets ``z_n``, ``z_p`` of each carrier's quasi-Fermi level
from its majority-electrode Fermi level (``electrode_quasi_fermi_1d_v1``).
Densities come from the compiled device's own map, so they are bitwise the
historical fields. Local Terms chain through ``dn/dz_n = n``,
``dp/dz_p = -p`` and ``dc/dphi = (dc/dz)/V_T``.

The SG flux consumes the quasi-Fermi step directly; its potential tangent at
fixed ``z`` is :func:`quasi_fermi_potential_tangents`, a chain without
equilibrium cancellation, not a density chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from solarlab.discretization.stable_bernoulli import bernoulli_pair
from solarlab.equations.assembly.layout import SystemLayout
from solarlab.equations.assembly.terms import FieldValues, OperatingInputs
from solarlab.equations.dc_1d import CompiledDevice1D
from solarlab.physics.transport.electrochemical_flux import CarrierFaceFlux

__all__ = [
    "ELECTRODE_QUASI_FERMI_1D_V1", "ElectrodeQuasiFermiFields", "quasi_fermi_potential_tangents",
]

ELECTRODE_QUASI_FERMI_1D_V1 = "electrode_quasi_fermi_1d_v1"
BLOCKS = ("potential", "electron", "hole")

Vector: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class ElectrodeQuasiFermiFields:
    """Field map ``u -> (phi, n, p)`` of one compiled device; no state is cached."""

    device: CompiledDevice1D

    @property
    def identity(self) -> str:
        return sha256(f"{ELECTRODE_QUASI_FERMI_1D_V1}:{self.device.identity}".encode()).hexdigest()

    def evaluate(self, layout: SystemLayout, u: Vector, inputs: OperatingInputs) -> FieldValues:
        values = np.concatenate([layout.view(u, block) for block in BLOCKS])
        phi, n, p = self.device.fields(values, voltage_V=inputs.voltage_V)
        # Outside this domain the historical residual returns NaN rows.
        admissible = bool(np.all(np.isfinite(n)) and np.all(np.isfinite(p))
                          and not np.any(n <= 0.0) and not np.any(p <= 0.0))
        return FieldValues({"potential_V": phi, "electron_density_m3": n,
                            "hole_density_m3": p}, admissible)


def quasi_fermi_potential_tangents(
    flux: CarrierFaceFlux, drive: Vector, carrier_sign: float, thermal: float
) -> tuple[Vector, Vector]:
    """Chain the SG kernel through density(phi, z) without equilibrium cancellation.

    Public home of the private DC1D chain, with the same arithmetic (pinned
    bitwise by tests/unit/assembly). Phase 2 deletes the private copy and
    moves its importers (dc_2d, impedance, ionic AC) here.
    """
    pair = bernoulli_pair(flux.energy_step)
    forward = np.divide(
        pair.forward_derivative, pair.forward,
        out=np.zeros_like(flux.flux), where=pair.forward != 0.0,
    )
    backward = np.divide(
        pair.backward_derivative, pair.backward,
        out=np.zeros_like(flux.flux), where=pair.backward != 0.0,
    )
    # An underflowed B(x), x >> 1, still has d(log B)/dx = 1/x - 1.
    tiny_forward = pair.forward == 0.0
    tiny_backward = pair.backward == 0.0
    forward[tiny_forward] = 1.0 / flux.energy_step[tiny_forward] - 1.0
    backward[tiny_backward] = -1.0 / flux.energy_step[tiny_backward] - 1.0
    from_left = drive < 0.0
    left_factor = np.where(from_left, 1.0 + forward, -backward)
    right_factor = np.where(from_left, -forward, 1.0 + backward)
    factor = carrier_sign * flux.flux / thermal
    return factor * left_factor, factor * right_factor
