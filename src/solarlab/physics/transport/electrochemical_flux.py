"""Unified directed carrier particle flux in stable electrochemical form (N class).

The Scharfetter-Gummel particle flux on a face oriented from node L to node R is

    F = k (B(D) c_L - B(-D) c_R),        k = D_face / h  [m/s],

with ``D`` the dimensionless potential-energy rise of the carrier: electrons
``D_n = -(psi_n,R - psi_n,L)/V_T`` with ``psi_n = phi + chi_t`` and holes
``D_p = +(psi_p,R - psi_p,L)/V_T`` with ``psi_p = phi + chi_t + Eg_t``. The
transport potentials carry the DOS terms
(:class:`~solarlab.physics.statistics.BandStructure`), so ``m = ln(c_R/c_L) + D``
is the reduced electrochemical-potential step: ``m_n = dE_Fn/kT`` and
``m_p = -dE_Fp/kT``. With ``B(-D) = B(D) e^D`` the flux is exactly

    F = k B(-D) c_R expm1(-m)     (m >= 0, carriers come from R)
    F = -k B(D) c_L expm1(m)      (m < 0, carriers come from L),

a product of the upwind density, one Bernoulli value and an exponential factor
bounded by one. It is therefore exactly zero at ``m = 0`` and
keeps its relative accuracy for weak drives instead of differencing two nearly
equal terms. Extreme energy steps combine the logarithms of all factors before
exponentiation, avoiding premature underflow of an otherwise representable
flux or derivative. ``m`` may come from a state representation (for example
``StateView.electrochemical_difference``, which it equals for binary64 fields);
without one it is formed from the densities with ``log1p`` near equal states.
Derivatives are those of ``F(c_L, c_R, D, k)`` from
:func:`~solarlab.discretization.stable_bernoulli.bernoulli_pair`.

Accuracy: the evaluation adds only a few roundings to whatever error ``m``
carries. When ``m`` is rebuilt from binary64 densities it carries about
``eps (|ln c_L| + |ln c_R| + |D|)``, so the absolute flux error is that times
``k (B(D) c_L + B(-D) c_R)`` -- slightly larger than the two-term form's
``eps`` times the same scale. The form gains relative accuracy where it
matters, near equilibrium, only when ``m`` is supplied by a representation that
holds the electrochemical step precisely (for example quasi-Fermi variables).

Output channels keep their own signs: ``J_n = -q F_n`` and ``J_p = +q F_p``
(A/m^2, along +x) adapt the flux to the historical current directions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.discretization.particle_flux import ParticleFlux
from solarlab.discretization.stable_bernoulli import bernoulli_pair
from solarlab.physics.statistics.band_potentials import BandStructure

__all__ = [
    "CarrierFaceFlux", "CarrierFaceFluxes", "carrier_particle_fluxes",
    "electrochemical_step_from_densities", "electron_current_from_particle_flux",
    "hole_current_from_particle_flux", "stable_sg_particle_flux",
]

Q = ELEMENTARY_CHARGE_C


def _extreme_bernoulli_product(
    argument: NDArray[np.float64], density: NDArray[np.float64],
    factor: NDArray[np.float64], *, extra_factor: NDArray[np.float64] | None = None,
    derivative: bool = False,
) -> NDArray[np.float64]:
    """B(argument)*density*factor for |argument| > 500, without losing B first.

    All multiplicative factors are nonnegative; the derivative of B is negative.
    True values outside binary64 still under/overflow, as they should.
    """
    a = np.abs(argument)
    decay = np.exp(-a)
    if derivative:
        log_small = -a + np.log(a - 1.0 + decay) - 2.0 * np.log1p(-decay)
        log_b = np.where(argument > 0.0, log_small, 0.0)
    else:
        log_b = np.log(a) - np.log1p(-decay) - np.where(argument > 0.0, a, 0.0)
    with np.errstate(divide="ignore", over="ignore", under="ignore"):
        log_product = log_b + np.log(density) + np.log(factor)
        if extra_factor is not None:
            log_product = log_product + np.log(extra_factor)
        product = np.exp(log_product)
    return -product if derivative else product


def electrochemical_step_from_densities(c_left: ArrayLike, c_right: ArrayLike,
                                        energy_step: ArrayLike) -> NDArray[np.float64]:
    """``ln(c_R/c_L) + D`` with ``log1p`` for nearly equal densities."""
    left = np.asarray(c_left, dtype=float)
    right = np.asarray(c_right, dtype=float)
    if np.any(left <= 0.0) or np.any(right <= 0.0):
        raise ValueError("the electrochemical form requires positive densities")
    with np.errstate(over="ignore"):
        relative = (right - left) / left
    close = np.abs(relative) < 0.5
    log_ratio = np.where(close, np.log1p(np.where(close, relative, 0.0)),
                         np.log(right) - np.log(left))
    result: NDArray[np.float64] = log_ratio + np.asarray(energy_step, dtype=float)
    return result


def stable_sg_particle_flux(
    c_left: ArrayLike,
    c_right: ArrayLike,
    energy_step: ArrayLike,
    coefficient: ArrayLike,
    *,
    electrochemical_step: ArrayLike | None = None,
) -> ParticleFlux:
    """SG particle flux in upwind electrochemical form with analytic derivatives."""
    left = np.asarray(c_left, dtype=float)
    right = np.asarray(c_right, dtype=float)
    step = np.asarray(energy_step, dtype=float)
    k = np.asarray(coefficient, dtype=float)
    shape = np.broadcast_shapes(left.shape, right.shape, step.shape, k.shape)
    if (left.shape != shape or right.shape != shape or step.shape != shape
            or k.shape not in {(), shape}):
        raise ValueError("SG densities and energy steps must share one face shape")
    if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))
            and np.all(np.isfinite(step)) and np.all(np.isfinite(k))):
        raise ValueError("SG inputs must be finite")
    if np.any(k < 0.0):
        raise ValueError("SG transfer coefficient must be non-negative")
    if electrochemical_step is None:
        drive = electrochemical_step_from_densities(left, right, step)
    else:
        drive = np.asarray(electrochemical_step, dtype=float)
        if drive.shape != shape or not np.all(np.isfinite(drive)):
            raise ValueError("electrochemical steps must be finite on every face")
        if np.any(left <= 0.0) or np.any(right <= 0.0):
            raise ValueError("the electrochemical form requires positive densities")
    pair = bernoulli_pair(step)
    decay = np.expm1(-np.abs(drive))  # expm1(-m) for m >= 0, expm1(m) for m < 0
    k_faces = np.broadcast_to(k, shape)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        upwind = np.where(drive >= 0.0, pair.backward * right, -(pair.forward * left))
        unscaled = np.array(decay * upwind + 0.0)  # normalize the equilibrium -0.0
        flux = np.array(k_faces * unscaled + 0.0)
        d_left = np.array(k_faces * pair.forward)
        d_right = np.array(-k_faces * pair.backward)
        d_step = np.array(k_faces * (pair.forward_derivative * left + pair.backward_derivative * right))
    extreme = np.abs(step) > 500.0
    if np.any(extreme):
        selected_step = np.where(drive >= 0.0, -step, step)[extreme]
        selected_density = np.where(drive >= 0.0, right, left)[extreme]
        sign = np.where(drive >= 0.0, -1.0, 1.0)[extreme]
        magnitude = -decay[extreme]
        unscaled[extreme] = sign * _extreme_bernoulli_product(selected_step, selected_density, magnitude)
        # Keep k separate from the decay so their product cannot underflow first.
        flux[extreme] = sign * _extreme_bernoulli_product(
            selected_step, selected_density, k_faces[extreme], extra_factor=magnitude)
        d_left[extreme] = _extreme_bernoulli_product(step[extreme], np.ones_like(selected_density), k_faces[extreme])
        d_right[extreme] = -_extreme_bernoulli_product(-step[extreme], np.ones_like(selected_density), k_faces[extreme])
        d_step[extreme] = (
            _extreme_bernoulli_product(step[extreme], left[extreme], k_faces[extreme], derivative=True)
            + _extreme_bernoulli_product(-step[extreme], right[extreme], k_faces[extreme], derivative=True)
        )
    flux = np.where((drive == 0.0) | (k_faces == 0.0), 0.0, flux)
    unscaled = np.where(drive == 0.0, 0.0, unscaled)
    return ParticleFlux(
        flux=flux + 0.0,
        left_density_derivative=d_left,
        right_density_derivative=d_right,
        energy_step_derivative=d_step,
        coefficient_derivative=unscaled,
    )


@dataclass(frozen=True)
class CarrierFaceFlux:
    """One carrier's directed particle flux [m^-2 s^-1] and local derivatives."""

    flux: NDArray[np.float64]
    left_density_derivative: NDArray[np.float64]
    right_density_derivative: NDArray[np.float64]
    left_potential_derivative: NDArray[np.float64]
    right_potential_derivative: NDArray[np.float64]
    energy_step: NDArray[np.float64]
    coefficient: NDArray[np.float64]


@dataclass(frozen=True)
class CarrierFaceFluxes:
    """Electron and hole particle fluxes on the same oriented faces."""

    electron: CarrierFaceFlux
    hole: CarrierFaceFlux

    def electron_current(self) -> NDArray[np.float64]:
        return electron_current_from_particle_flux(self.electron.flux)

    def hole_current(self) -> NDArray[np.float64]:
        return hole_current_from_particle_flux(self.hole.flux)


def electron_current_from_particle_flux(flux: ArrayLike) -> NDArray[np.float64]:
    """Electron current density ``J_n = -q F_n`` [A/m^2] (charge -q)."""
    result: NDArray[np.float64] = -Q * np.asarray(flux, dtype=float)
    return result


def hole_current_from_particle_flux(flux: ArrayLike) -> NDArray[np.float64]:
    """Hole current density ``J_p = +q F_p`` [A/m^2] (charge +q)."""
    result: NDArray[np.float64] = Q * np.asarray(flux, dtype=float)
    return result


def _carrier_flux(density: NDArray[np.float64], potential: NDArray[np.float64],
                  coefficient: NDArray[np.float64], thermal_voltage_V: float,
                  energy_sign: float, electrochemical_step: ArrayLike | None) -> CarrierFaceFlux:
    step = energy_sign * (potential[1:] - potential[:-1]) / thermal_voltage_V
    result = stable_sg_particle_flux(density[:-1], density[1:], step, coefficient,
                                     electrochemical_step=electrochemical_step)
    # D = s (psi_R - psi_L)/V_T, so dF/dpsi_R = s/V_T dF/dD and dF/dpsi_L = -s/V_T dF/dD.
    per_volt = energy_sign / thermal_voltage_V * result.energy_step_derivative
    return CarrierFaceFlux(
        flux=result.flux,
        left_density_derivative=result.left_density_derivative,
        right_density_derivative=result.right_density_derivative,
        left_potential_derivative=-per_volt,
        right_potential_derivative=per_volt,
        energy_step=step,
        coefficient=np.broadcast_to(coefficient, step.shape).copy(),
    )


def carrier_particle_fluxes(
    x: ArrayLike,
    phi: ArrayLike,
    n: ArrayLike,
    p: ArrayLike,
    D_n_face: ArrayLike,
    D_p_face: ArrayLike,
    thermal_voltage_V: float,
    bands: BandStructure,
    *,
    electron_electrochemical_step: ArrayLike | None = None,
    hole_electrochemical_step: ArrayLike | None = None,
) -> CarrierFaceFluxes:
    """Electron and hole particle fluxes along +x on the N-1 grid faces.

    Potential derivatives are with respect to each carrier's transport
    potential, which equal those with respect to ``phi`` because the band
    terms are fixed.
    """
    grid = np.asarray(x, dtype=float)
    spacing = np.diff(grid)
    electrons = np.asarray(n, dtype=float)
    holes = np.asarray(p, dtype=float)
    thermal = float(thermal_voltage_V)
    if (grid.ndim != 1 or grid.size < 2 or electrons.shape != grid.shape
            or holes.shape != grid.shape or np.any(spacing <= 0.0)
            or not np.isfinite(thermal) or thermal <= 0.0):
        raise ValueError("flux inputs must share a strictly increasing node grid")
    electron_potential, hole_potential = bands.transport_potentials(phi)
    k_n = np.asarray(D_n_face, dtype=float) / spacing
    k_p = np.asarray(D_p_face, dtype=float) / spacing
    return CarrierFaceFluxes(
        electron=_carrier_flux(electrons, electron_potential, k_n, thermal, -1.0,
                               electron_electrochemical_step),
        hole=_carrier_flux(holes, hole_potential, k_p, thermal, 1.0,
                           hole_electrochemical_step),
    )
