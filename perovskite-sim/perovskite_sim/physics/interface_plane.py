"""Phase E3 — thermionic-emission flux primitive for interface-plane state.

Implements Pauwels-Vanhoutte 1978 (J.Phys.D 11, 649-667) eq 14a/b
translated to SolarLab's n+p heterojunction convention.

For each heterointerface k, four state-vec densities live on the
interface plane: (n_1s, p_1s, n_2s, p_2s). They couple to the bulk
nodes on each side via a thermionic-emission (TE) flux:

  J_TE = v_th_eff * (density_bulk_projected - density_state)

where ``density_bulk_projected`` is the equilibrium bulk density on the
appropriate side, Boltzmann-projected to the interface plane via the
local band-bending V_1 or V_2 partition of (|V_bi| - V_app).

Sign convention: positive flux means carriers flow INTO the interface-
plane state (state density grows). The Sprint 6 Day 8-10 work then
applies these fluxes as source terms on the dy/dt vector for the
iface_state block.

Units: m^-3 * m/s = m^-2 s^-1 (surface flux).

The default ``v_th_eff = 1.0e5`` m/s matches SCAPS Manual sec 3.8
typical thermal velocity (~ 10^7 cm/s). Per-layer override via the
material ``v_th`` field is available through the YAML loader; callers may
also supply the ``v_th_eff`` fallback explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares

from perovskite_sim.constants import Q, V_T as _V_T_300
from perovskite_sim.physics.fermi_dirac import (
    fermi_dirac_half,
    fermi_dirac_one,
    inverse_fermi_dirac_half,
)
from perovskite_sim.physics.recombination import interface_recombination
from perovskite_sim.models.device import electrical_interfaces


_DEFAULT_V_TH_MS = 1.0e-2  # m/s — matches SCAPS-mirror surface recombination
                           # velocity scale to keep iface_state ODE timescale
                           # comparable to the existing carrier transient
                           # (tau ~ dx_iface / v_th ~ 1e-9/1e-2 = 1e-7 s).
                           # Full thermal velocity (1e5 m/s) makes Newton
                           # diverge at the diode knee; Sprint 7 Day 4+
                           # will revisit with QSS algebraic reduction.
_EXP_CAP = 30.0           # cap exponent to avoid overflow at V_bi/V_T ~ 42

FERMI_RICHARDSON = "fermi_richardson"
FERMI_DIRAC_RICHARDSON = "fermi_dirac_richardson"
SCAPS_THERMIONIC = "scaps_thermionic"
SCAPS_THERMAL_VELOCITY = "scaps_thermal_velocity"
INTERFACE_TRANSPORT_MODELS = (
    FERMI_RICHARDSON,
    FERMI_DIRAC_RICHARDSON,
    SCAPS_THERMIONIC,
    SCAPS_THERMAL_VELOCITY,
)


def validate_interface_transport_model(model: str) -> str:
    """Return a normalized interface-transport model name."""
    normalized = str(model).strip().lower()
    if normalized not in INTERFACE_TRANSPORT_MODELS:
        raise ValueError(
            "interface_transport_model must be one of "
            f"{INTERFACE_TRANSPORT_MODELS}; got {model!r}"
        )
    return normalized


def _hole_density_step(mat, dE_c: float, dE_g: float) -> float:
    """Return the exponent controlling ``p_right / p_left``.

    The physical SG zero-flux invariant is
    ``p_R/p_L = exp(-(dchi+dEg)/V_T)`` because ``E_v = -chi-Eg``. The
    historical interface-state prototype used ``dchi-dEg``; keep that only
    outside the opt-in exclusive transport path so calibrated legacy runs do
    not move.
    """
    if getattr(mat, "iface_state_physical_offsets", False):
        return -(dE_c + dE_g)
    return dE_c - dE_g


def _interface_activity_steps(
    mat,
    interface_index: int,
    dE_c: float,
    dE_g: float,
    thermal_voltage: float,
) -> tuple[float, float]:
    """Return electron and hole log-density energy steps across a face.

    Volumetric carrier densities include their band-edge densities of states.
    A reciprocal thermionic law must therefore use

    ``dEc + V_T*ln(Nc_R/Nc_L)`` and
    ``-(dEc+dEg) + V_T*ln(Nv_R/Nv_L)``.

    These are exactly the zero-current density ratios implied by continuous
    quasi-Fermi potentials. The historical interface-state prototype omitted
    the DOS ratios and is retained outside the opt-in physical-offset path.
    """
    if not getattr(mat, "iface_state_physical_offsets", False):
        return dE_c, _hole_density_step(mat, dE_c, dE_g)
    if interface_index >= len(mat.interface_nodes):
        raise ValueError("interface index is not aligned with interface nodes")
    right = int(mat.interface_nodes[interface_index])
    left = right - 1
    N_C = getattr(mat, "N_C_physical", None)
    N_V = getattr(mat, "N_V_physical", None)
    if N_C is None or N_V is None or left < 0 or right >= len(N_C):
        raise ValueError(
            "physical interface transport requires adjacent Nc and Nv data"
        )
    values = (
        float(N_C[left]),
        float(N_C[right]),
        float(N_V[left]),
        float(N_V[right]),
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(
            "physical interface transport requires finite positive adjacent "
            "Nc and Nv data"
        )
    N_C_left, N_C_right, N_V_left, N_V_right = values
    electron_step = dE_c + thermal_voltage * math.log(N_C_right / N_C_left)
    hole_step = -(dE_c + dE_g) + thermal_voltage * math.log(
        N_V_right / N_V_left
    )
    return electron_step, hole_step


def _fermi_edge_occupation(log_activity: float) -> float:
    """Stable band-edge occupation from ``log(density / DOS)``."""
    if log_activity >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(log_activity, 700.0)))
    exponential = math.exp(max(log_activity, -700.0))
    return exponential / (1.0 + exponential)


def _project_density_to_plane(
    density: float,
    density_of_states: float,
    log_boltzmann_factor: float,
) -> float:
    """Project a reservoir density to a DOS-bounded band-edge population."""
    log_activity = (
        math.log(max(float(density), 1.0e-300))
        - math.log(float(density_of_states))
        + float(log_boltzmann_factor)
    )
    return float(density_of_states) * _fermi_edge_occupation(log_activity)


def _project_density_to_plane_fermi_dirac(
    density: float,
    density_of_states: float,
    reduced_energy_shift: float,
) -> float:
    """Project a 3D carrier density by shifting its reduced chemical potential."""
    ratio = max(float(density), 0.0) / float(density_of_states)
    eta = inverse_fermi_dirac_half(ratio)
    return float(density_of_states) * fermi_dirac_half(
        eta + float(reduced_energy_shift)
    )


def _project_density_for_transport_model(
    density: float,
    density_of_states: float,
    reduced_energy_shift: float,
    transport_model: str,
) -> float:
    if transport_model == FERMI_DIRAC_RICHARDSON:
        return _project_density_to_plane_fermi_dirac(
            density,
            density_of_states,
            reduced_energy_shift,
        )
    return _project_density_to_plane(
        density,
        density_of_states,
        reduced_energy_shift,
    )


def _barrier_occupation_from_state(
    density: float,
    density_of_states: float,
    barrier_normalized: float,
) -> float:
    """Evaluate a bounded band-edge supply at a barrier top.

    ``density / DOS`` is used as the non-degenerate chemical activity. The
    local state is separately bounded by ``density <= DOS``, so this closure
    remains in its declared Boltzmann-to-band-edge validity envelope instead
    of interpreting ``density == DOS`` as an infinite single-level logit.
    """
    log_activity = (
        math.log(max(float(density), 1.0e-300))
        - math.log(float(density_of_states))
        - float(barrier_normalized)
    )
    return _fermi_edge_occupation(log_activity)


def _reciprocal_fermi_edge_cross_fluxes(
    mat,
    interface_index: int,
    state: np.ndarray,
    thermal_voltage: float,
    transmission: float,
) -> tuple[float, float]:
    """Return reciprocal electron/hole particle fluxes from left to right.

    A common Richardson supply prefactor is used in both directions. The two
    one-way supplies are band-edge Fermi occupations evaluated at the common
    barrier top. This recovers Boltzmann thermionic emission at low occupation,
    gives exactly zero net flux for a common quasi-Fermi level, and prevents an
    unphysical ``n >> Nc`` Boltzmann pile-up from providing unlimited current.
    """
    right = int(mat.interface_nodes[interface_index])
    left = right - 1
    N_C = mat.N_C_physical
    N_V = mat.N_V_physical
    N_C_left = float(N_C[left])
    N_C_right = float(N_C[right])
    N_V_left = float(N_V[left])
    N_V_right = float(N_V[right])
    base = 4 * interface_index
    n_right = max(float(state[base + 0]), 1.0e-300)
    p_right = max(float(state[base + 1]), 1.0e-300)
    n_left = max(float(state[base + 2]), 1.0e-300)
    p_left = max(float(state[base + 3]), 1.0e-300)
    dE_c = float(mat.interface_chi_step[interface_index])
    dE_g = float(mat.interface_Eg_step[interface_index])

    # E_C,R - E_C,L = -dchi. The left supply is penalized when the right
    # conduction band is higher; the reverse supply sees the opposite barrier.
    electron_delta = -dE_c
    electron_barrier_left = max(electron_delta, 0.0)
    electron_barrier_right = max(-electron_delta, 0.0)
    electron_left = _barrier_occupation_from_state(
        n_left,
        N_C_left,
        electron_barrier_left / thermal_voltage,
    )
    electron_right = _barrier_occupation_from_state(
        n_right,
        N_C_right,
        electron_barrier_right / thermal_voltage,
    )

    # Hole energy is -E_V = phi + chi + Eg, hence its right-minus-left step is
    # dchi+dEg. Apply the same common-barrier construction to hole particles.
    hole_delta = dE_c + dE_g
    hole_barrier_left = max(hole_delta, 0.0)
    hole_barrier_right = max(-hole_delta, 0.0)
    hole_left = _barrier_occupation_from_state(
        p_left,
        N_V_left,
        hole_barrier_left / thermal_voltage,
    )
    hole_right = _barrier_occupation_from_state(
        p_right,
        N_V_right,
        hole_barrier_right / thermal_voltage,
    )

    temperature = float(mat.T_device)
    electron_supply = (
        transmission
        * min(float(mat.A_star_n[left]), float(mat.A_star_n[right]))
        * temperature**2
        / Q
    )
    hole_supply = (
        transmission
        * min(float(mat.A_star_p[left]), float(mat.A_star_p[right]))
        * temperature**2
        / Q
    )
    return (
        electron_supply * (electron_left - electron_right),
        hole_supply * (hole_left - hole_right),
    )


def _fermi_dirac_richardson_cross_fluxes(
    mat,
    interface_index: int,
    state: np.ndarray,
    thermal_voltage: float,
    transmission: float,
) -> tuple[float, float]:
    """Return reciprocal 3D Fermi-Dirac thermionic particle fluxes.

    Carrier densities determine reduced chemical potentials through
    ``state / DOS = F_1/2(eta)``. The one-way Richardson supplies use
    ``F_1(eta - barrier/kT)`` and therefore recover the SCAPS Boltzmann law
    when dilute without imposing the single-level ``state <= DOS`` cap.
    """
    right = int(mat.interface_nodes[interface_index])
    left = right - 1
    base = 4 * interface_index
    N_C_left = float(mat.N_C_physical[left])
    N_C_right = float(mat.N_C_physical[right])
    N_V_left = float(mat.N_V_physical[left])
    N_V_right = float(mat.N_V_physical[right])
    eta_n_left = inverse_fermi_dirac_half(
        max(float(state[base + 2]), 0.0) / N_C_left
    )
    eta_n_right = inverse_fermi_dirac_half(
        max(float(state[base + 0]), 0.0) / N_C_right
    )
    eta_p_left = inverse_fermi_dirac_half(
        max(float(state[base + 3]), 0.0) / N_V_left
    )
    eta_p_right = inverse_fermi_dirac_half(
        max(float(state[base + 1]), 0.0) / N_V_right
    )
    dE_c = float(mat.interface_chi_step[interface_index])
    dE_g = float(mat.interface_Eg_step[interface_index])
    electron_delta = -dE_c
    electron_left = fermi_dirac_one(
        eta_n_left - max(electron_delta, 0.0) / thermal_voltage
    )
    electron_right = fermi_dirac_one(
        eta_n_right - max(-electron_delta, 0.0) / thermal_voltage
    )
    hole_delta = dE_c + dE_g
    hole_left = fermi_dirac_one(
        eta_p_left - max(hole_delta, 0.0) / thermal_voltage
    )
    hole_right = fermi_dirac_one(
        eta_p_right - max(-hole_delta, 0.0) / thermal_voltage
    )
    temperature = float(mat.T_device)
    electron_supply = (
        float(transmission)
        * min(float(mat.A_star_n[left]), float(mat.A_star_n[right]))
        * temperature**2
        / Q
    )
    hole_supply = (
        float(transmission)
        * min(float(mat.A_star_p[left]), float(mat.A_star_p[right]))
        * temperature**2
        / Q
    )
    return (
        electron_supply * (electron_left - electron_right),
        hole_supply * (hole_left - hole_right),
    )


def _bounded_reciprocal_density_flux(
    density_left: float,
    density_right: float,
    log_right_over_left_at_equilibrium: float,
    velocity: float,
) -> float:
    """Classical thermionic particle flux with no growing exponential.

    The two algebraic branches are the same detailed-balance law with the
    larger one-way exponential factored out.  Positive flux is left to right.
    """
    step = float(log_right_over_left_at_equilibrium)
    left = max(float(density_left), 0.0)
    right = max(float(density_right), 0.0)
    if step >= 0.0:
        return float(velocity) * (left - right * math.exp(-min(step, 700.0)))
    return float(velocity) * (
        left * math.exp(max(step, -700.0)) - right
    )


def _interface_thermal_velocity(
    mat,
    interface_index: int,
    fallback: float,
) -> float:
    """SCAPS interface velocity: the smaller adjacent layer value."""
    values = getattr(mat, "v_th_physical", None)
    right = int(mat.interface_nodes[interface_index])
    left = right - 1
    if values is None or left < 0 or right >= len(values):
        return float(fallback)
    pair = (float(values[left]), float(values[right]))
    if any(not np.isfinite(value) or value <= 0.0 for value in pair):
        raise ValueError(
            "SCAPS thermionic transport requires finite positive adjacent "
            "thermal velocities"
        )
    return min(pair)


def _scaps_thermal_velocity_cross_fluxes(
    mat,
    interface_index: int,
    state: np.ndarray,
    thermal_voltage: float,
    transmission: float,
    fallback_velocity: float = 1.0e5,
) -> tuple[float, float]:
    """Return SCAPS-style classical thermionic particle fluxes.

    SCAPS declares the common interface velocity to be the smaller thermal
    velocity of the neighbouring layers.  The DOS ratios are retained in the
    equilibrium activity steps, so both carrier fluxes vanish for a common
    quasi-Fermi level even when the adjacent effective DOS differ.
    """
    base = 4 * interface_index
    n_right = float(state[base + 0])
    p_right = float(state[base + 1])
    n_left = float(state[base + 2])
    p_left = float(state[base + 3])
    electron_step, hole_step = _interface_activity_steps(
        mat,
        interface_index,
        float(mat.interface_chi_step[interface_index]),
        float(mat.interface_Eg_step[interface_index]),
        thermal_voltage,
    )
    velocity = float(transmission) * _interface_thermal_velocity(
        mat,
        interface_index,
        fallback_velocity,
    )
    return (
        _bounded_reciprocal_density_flux(
            n_left,
            n_right,
            electron_step / thermal_voltage,
            velocity,
        ),
        _bounded_reciprocal_density_flux(
            p_left,
            p_right,
            hole_step / thermal_voltage,
            velocity,
        ),
    )


def _scaps_boltzmann_cross_fluxes(
    mat,
    interface_index: int,
    state: np.ndarray,
    thermal_voltage: float,
    transmission: float,
) -> tuple[float, float]:
    """Published SCAPS thermionic law in Boltzmann variables.

    Verschraegen and Burgelman (2007), Eq. 1, use one common Richardson
    prefactor and the two quasi-Fermi Boltzmann factors at the maximum band
    edge.  Unlike :func:`_reciprocal_fermi_edge_cross_fluxes`, this deliberately
    does not replace the Boltzmann activity with a Fermi occupation.
    """
    right = int(mat.interface_nodes[interface_index])
    left = right - 1
    base = 4 * interface_index
    n_right = max(float(state[base + 0]), 0.0)
    p_right = max(float(state[base + 1]), 0.0)
    n_left = max(float(state[base + 2]), 0.0)
    p_left = max(float(state[base + 3]), 0.0)
    N_C_left = float(mat.N_C_physical[left])
    N_C_right = float(mat.N_C_physical[right])
    N_V_left = float(mat.N_V_physical[left])
    N_V_right = float(mat.N_V_physical[right])
    dE_c = float(mat.interface_chi_step[interface_index])
    dE_g = float(mat.interface_Eg_step[interface_index])

    electron_delta = -dE_c
    electron_left = (n_left / N_C_left) * math.exp(
        -min(max(electron_delta, 0.0) / thermal_voltage, 700.0)
    )
    electron_right = (n_right / N_C_right) * math.exp(
        -min(max(-electron_delta, 0.0) / thermal_voltage, 700.0)
    )
    hole_delta = dE_c + dE_g
    hole_left = (p_left / N_V_left) * math.exp(
        -min(max(hole_delta, 0.0) / thermal_voltage, 700.0)
    )
    hole_right = (p_right / N_V_right) * math.exp(
        -min(max(-hole_delta, 0.0) / thermal_voltage, 700.0)
    )
    temperature = float(mat.T_device)
    electron_supply = (
        float(transmission)
        * min(float(mat.A_star_n[left]), float(mat.A_star_n[right]))
        * temperature**2
        / Q
    )
    hole_supply = (
        float(transmission)
        * min(float(mat.A_star_p[left]), float(mat.A_star_p[right]))
        * temperature**2
        / Q
    )
    return (
        electron_supply * (electron_left - electron_right),
        hole_supply * (hole_left - hole_right),
    )


def _physical_cross_fluxes(
    mat,
    interface_index: int,
    state: np.ndarray,
    thermal_voltage: float,
    transmission: float,
    transport_model: str,
    fallback_velocity: float,
) -> tuple[float, float]:
    model = validate_interface_transport_model(transport_model)
    if model == FERMI_RICHARDSON:
        return _reciprocal_fermi_edge_cross_fluxes(
            mat,
            interface_index,
            state,
            thermal_voltage,
            transmission,
        )
    if model == FERMI_DIRAC_RICHARDSON:
        return _fermi_dirac_richardson_cross_fluxes(
            mat,
            interface_index,
            state,
            thermal_voltage,
            transmission,
        )
    if model == SCAPS_THERMIONIC:
        return _scaps_boltzmann_cross_fluxes(
            mat,
            interface_index,
            state,
            thermal_voltage,
            transmission,
        )
    return _scaps_thermal_velocity_cross_fluxes(
        mat,
        interface_index,
        state,
        thermal_voltage,
        transmission,
        fallback_velocity,
    )


def te_flux(
    density_bulk_projected: float,
    density_state: float,
    v_th: float,
) -> float:
    """Pure-function TE flux primitive (paper eq 14 reduced).

    Returns surface flux INTO the interface-plane state [m^-2 s^-1].
    Positive when bulk-projected > state (state fills); negative when
    state > bulk-projected (state drains).
    """
    return float(v_th) * (float(density_bulk_projected) - float(density_state))


def compute_interface_te_fluxes(
    mat,
    iface_state: np.ndarray,
    V_app: float = 0.0,
    *,
    v_th_eff: float = _DEFAULT_V_TH_MS,
    v_cross_eff: float = 0.0,
    V_T: float | None = None,
) -> np.ndarray:
    """Compute all 4*N_iface TE fluxes for one RHS evaluation.

    Per interface k, the block layout is (n_1s, p_1s, n_2s, p_2s)
    matching ``StateVec.iface_state`` and ``_compute_iface_state_dark_eq``.

    Bulk-projected densities use cached equilibrium values plus the
    V_app-dependent band-bending partition:

      V_total = |V_bi_eff| - V_app  (clamped to >= 0)
      V_2     = partition_left[k] * V_total       # PVK (light) side
      V_1     = (1 - partition_left[k]) * V_total # ETL (heavy) side

      n_1s_bulk_proj = n_R_eq * exp(-V_1 / V_T)
      p_1s_bulk_proj = p_R_eq * exp(+V_1 / V_T)
      n_2s_bulk_proj = n_L_eq * exp(+V_2 / V_T)
      p_2s_bulk_proj = p_L_eq * exp(-V_2 / V_T)

    At dark equilibrium (V_app = 0, iface_state = dark-eq init) the
    bulk-projected equals the state on every entry -> all fluxes = 0.

    Returns array shape (4 * N_iface,).
    """
    n_iface = len(mat.interface_V_partition_2)
    if n_iface == 0:
        return np.zeros(0, dtype=float)
    V_T_local = V_T if V_T is not None else (
        mat.V_T_device if hasattr(mat, "V_T_device") else _V_T_300
    )
    V_total = max(0.0, abs(float(mat.V_bi_eff)) - float(V_app))
    out = np.zeros(4 * n_iface, dtype=float)
    for k in range(n_iface):
        partition_left = float(mat.interface_V_partition_2[k])
        V_2 = partition_left * V_total
        V_1 = (1.0 - partition_left) * V_total
        v1_norm = max(-_EXP_CAP, min(_EXP_CAP, V_1 / V_T_local))
        v2_norm = max(-_EXP_CAP, min(_EXP_CAP, V_2 / V_T_local))
        n_R = float(mat.interface_n_R_eq[k])
        p_R = float(mat.interface_p_R_eq[k])
        n_L = float(mat.interface_n_L_eq[k])
        p_L = float(mat.interface_p_L_eq[k])
        n_1s_proj = n_R * math.exp(-v1_norm)
        p_1s_proj = p_R * math.exp(+v1_norm)
        # Phase E3 Day 4-6 — χ-step-consistent 2s projection. Without
        # this, single-side Boltzmann from L bulk gives an unphysical
        # 2s_proj that ignores cross-interface equilibrium (n_2s_eq =
        # n_1s_eq · exp(-ΔE_c/V_T) at flat E_F). Falls back to legacy
        # single-side when chi_step cache is empty.
        if mat.interface_chi_step and len(mat.interface_chi_step) > k:
            dE_c = float(mat.interface_chi_step[k])
            dE_g = float(mat.interface_Eg_step[k])
            dE_c_density, dE_v_density = _interface_activity_steps(
                mat,
                k,
                dE_c,
                dE_g,
                V_T_local,
            )
            ec_n = max(-_EXP_CAP, min(_EXP_CAP, dE_c_density / V_T_local))
            ev_n = max(-_EXP_CAP, min(_EXP_CAP, dE_v_density / V_T_local))
            n_2s_proj = n_1s_proj * math.exp(-ec_n)
            p_2s_proj = p_1s_proj * math.exp(-ev_n)
        else:
            n_2s_proj = n_L * math.exp(+v2_norm)
            p_2s_proj = p_L * math.exp(-v2_norm)
        base = 4 * k
        n_1s = float(iface_state[base + 0])
        p_1s = float(iface_state[base + 1])
        n_2s = float(iface_state[base + 2])
        p_2s = float(iface_state[base + 3])
        out[base + 0] = te_flux(n_1s_proj, n_1s, v_th_eff)
        out[base + 1] = te_flux(p_1s_proj, p_1s, v_th_eff)
        out[base + 2] = te_flux(n_2s_proj, n_2s, v_th_eff)
        out[base + 3] = te_flux(p_2s_proj, p_2s, v_th_eff)
        # Phase E3 Day 4-6 — cross-interface χ-step TE flux (paper eq 15).
        # Electrons: equilibrium ratio n_1s/n_2s = exp(ΔE_c/V_T) where
        # ΔE_c = chi_R − chi_L. Cross-flux drives n_1s, n_2s toward this
        # ratio across the χ-step barrier.
        # Holes: equilibrium ratio p_1s/p_2s = exp(ΔE_v/V_T) where
        # ΔE_v = (E_v_R − E_v_L) = ΔE_c − (Eg_R − Eg_L).
        if (
            v_cross_eff > 0.0
            and mat.interface_chi_step
            and len(mat.interface_chi_step) > k
        ):
            dE_c = float(mat.interface_chi_step[k])  # eV
            dE_g = float(mat.interface_Eg_step[k])   # eV
            dE_c_density, dE_v_density = _interface_activity_steps(
                mat,
                k,
                dE_c,
                dE_g,
                V_T_local,
            )
            ec_norm = dE_c_density / V_T_local
            ev_norm = dE_v_density / V_T_local
            # Cross-flux (positive flows 2s → 1s) toward the detailed-balance
            # ratio n_1s/n_2s = exp(ΔE_c/V_T). Phase E8.4 — bounded form: the
            # legacy ``n_2s·exp(ec_norm) − n_1s`` carried exp(+|ΔE|/V_T), which
            # at a large offset (HTL/PVK ΔE_c=1.54 eV → exp(59), capped at
            # exp(30)≈1e13) produced the ~1e36 cross-flux that both hung the
            # Sprint-9 wire-through and flattened the CBO response under the
            # cap. Factor out the LARGER exponential so the surviving exp arg
            # is always ≤ 0 (bounded ≤ 1). Same zero-point, physical magnitude
            # (cross-barrier emission is Boltzmann-suppressed, not enhanced).
            if ec_norm >= 0.0:
                J_cross_n = v_cross_eff * (n_2s - n_1s * math.exp(-ec_norm))
            else:
                J_cross_n = v_cross_eff * (n_2s * math.exp(ec_norm) - n_1s)
            if ev_norm >= 0.0:
                J_cross_p = v_cross_eff * (p_2s - p_1s * math.exp(-ev_norm))
            else:
                J_cross_p = v_cross_eff * (p_2s * math.exp(ev_norm) - p_1s)
            # Mass-conserving redistribution: +J on 1s side, -J on 2s side.
            out[base + 0] += J_cross_n      # n_1s gains
            out[base + 2] -= J_cross_n      # n_2s loses
            out[base + 1] += J_cross_p      # p_1s gains
            out[base + 3] -= J_cross_p      # p_2s loses
    return out


def compute_interface_srh_on_state(
    iface_state: np.ndarray,
    stack,
    mat,
    *,
    density_regularization_width_m3: float = 0.0,
) -> np.ndarray:
    """Two-sided Shockley-Read SRH evaluated on interface-plane state.

    Paper eq 12, 13 translated to SolarLab n+p convention.

    Per interface k, four state densities (n_1s, p_1s, n_2s, p_2s) pair
    up into two SRH paths:
      R_s1 = SRH(n_1s, p_2s, ...)  # ETL electron + PVK hole
      R_s2 = SRH(n_2s, p_1s, ...)  # PVK electron + ETL hole

    Sinks (negative contributions to dy/dt) on each state density:
      d(n_1s)/dt -= R_s1
      d(p_2s)/dt -= R_s1   # paired with n_1s
      d(n_2s)/dt -= R_s2
      d(p_1s)/dt -= R_s2   # paired with n_2s

    Returns array shape (4 * N_iface,) of NEGATIVE sink magnitudes (each
    entry = -R_s for the carrier consumed in its pair). Caller adds this
    array to the diface_state block to apply the loss.

    Uses cached per-interface (n_1, p_1, ni_eff_sq, calibration_factor)
    from MaterialArrays. Surface velocities (v_n, v_p) read from
    stack.interfaces, scaled by calibration_factor (Phase E1.6 pattern).
    """
    ifaces = electrical_interfaces(stack)
    n_iface = len(mat.interface_V_partition_2)
    out = np.zeros(4 * n_iface, dtype=float)
    if n_iface == 0 or not ifaces:
        return out
    for k in range(n_iface):
        if k >= len(ifaces):
            break
        v_n, v_p = ifaces[k]
        if mat.interface_calibration_factor:
            cf = float(mat.interface_calibration_factor[k])
            v_n = v_n * cf
            v_p = v_p * cf
        if v_n == 0.0 and v_p == 0.0:
            continue
        n1_k = float(mat.interface_n1[k])
        p1_k = float(mat.interface_p1[k])
        # χ-step-consistent ni_eff² for iface-plane SRH path. At equilibrium
        # n_1s_eq · p_2s_eq = n_R_eq · p_R_eq · exp(-ΔE_v/V_T) (detailed
        # balance across the χ step). Falls back to legacy cached value
        # when chi_step cache is empty.
        if (
            mat.interface_chi_step
            and len(mat.interface_chi_step) > k
            and mat.interface_n_R_eq
        ):
            dE_c_k = float(mat.interface_chi_step[k])
            dE_g_k = float(mat.interface_Eg_step[k])
            dE_v_k = _hole_density_step(mat, dE_c_k, dE_g_k)
            V_T_iface = mat.V_T_device if hasattr(mat, "V_T_device") else _V_T_300
            ev_n_k = max(-_EXP_CAP, min(_EXP_CAP, dE_v_k / V_T_iface))
            n_R_eq = float(mat.interface_n_R_eq[k])
            p_R_eq = float(mat.interface_p_R_eq[k])
            ni_eff_sq = n_R_eq * p_R_eq * math.exp(-ev_n_k)
        else:
            ni_eff_sq = (
                float(mat.interface_ni_sq_eff[k])
                if mat.interface_ni_sq_eff else 0.0
            )
        base = 4 * k
        n_1s = float(iface_state[base + 0])
        p_1s = float(iface_state[base + 1])
        n_2s = float(iface_state[base + 2])
        p_2s = float(iface_state[base + 3])
        # Zero width retains the historical hard projection. A positive
        # width is research-only and must remain paired with raw negative-
        # state diagnostics; it does not certify an invalid trajectory.
        from perovskite_sim.physics.regularization import compact_positive_part

        n_1s = float(compact_positive_part(n_1s, density_regularization_width_m3))
        p_1s = float(compact_positive_part(p_1s, density_regularization_width_m3))
        n_2s = float(compact_positive_part(n_2s, density_regularization_width_m3))
        p_2s = float(compact_positive_part(p_2s, density_regularization_width_m3))
        # R_s1: ETL-side electron capture paired with PVK-side hole.
        R_s1 = interface_recombination(
            n_1s, p_2s, ni_eff_sq, n1_k, p1_k, v_n, v_p,
        )
        # R_s2: PVK-side electron capture paired with ETL-side hole.
        R_s2 = interface_recombination(
            n_2s, p_1s, ni_eff_sq, n1_k, p1_k, v_n, v_p,
        )
        # Sinks (carrier consumed -> negative dy/dt contribution).
        out[base + 0] = -R_s1   # n_1s
        out[base + 1] = -R_s2   # p_1s
        out[base + 2] = -R_s2   # n_2s
        out[base + 3] = -R_s1   # p_2s
    return out


# =====================================================================
# QSS algebraic interface-plane closure (2026-06) — the "Sprint 7 Day 4+
# QSS algebraic reduction" anticipated above, finally built.
#
# Instead of carrying the four plane densities as ODE state (whose explicit
# TE coupling made Newton diverge at full thermal velocity and whose bounded
# cross-flux flattened the CBO response — see module header), the plane
# densities (n_s, p_s) are solved as a LOCAL steady-state flux balance per
# RHS call:
#
#     S*(nb - 2*n_s) = R(n_s, p_s)        nb = bn_L*n_L + bn_R*n_R
#     S*(pb - 2*p_s) = R(n_s, p_s)        pb = bp_L*p_L + bp_R*p_R
#
# with the P-V single-level rate on the plane densities. Plane band edges
# are the FAVOURABLE edges (electrons: lower E_C of the two sides; holes:
# higher E_V), so the plane gap Eg_s = E_C,s - E_V,s is the reduced
# interface gap — a conduction-band cliff shrinks Eg_s and raises ni_s^2
# exponentially (the SCAPS cliff mechanism). The b factors are the
# Boltzmann barrier penalties from each side's node (already projected to
# the plane potential by the caller) onto the favourable edge.
#
# Detailed balance is analytic: at dark equilibrium the SG zero-flux
# condition makes node ratios exactly Boltzmann, both sides' projected
# supplies agree, and n_s*p_s = ni_s^2 exactly -> R = 0 with no
# cross-product reference needed.
#
# Structural properties the bulk-node formulations lacked:
#   * supply limitation: R <= S*(nb + pb)/2 by construction (the measured
#     HTL/PVK six-decade SRV insensitivity emerges as saturation);
#   * trap-level visibility: plane minority densities are small enough
#     that n1_s/p1_s compete in the denominator (E_t responds);
#   * locally implicit: no sign-flipping explicit rates (the historical
#     Radau wall).
# =====================================================================

# Effusion supply velocity [m/s]: v_th/4 with the SCAPS default thermal
# velocity 1e7 cm/s = 1e5 m/s. Supply is huge except where a band offset
# blocks it, so results are weakly sensitive to the exact prefactor.
S_SUPPLY = 2.5e4

_QSS_MAX_NEWTON = 40
_QSS_TOL = 1.0e-12
_QSS_FLOOR = 1.0e-30


@dataclass(frozen=True)
class PlaneInterfaceParams:
    """Build-time constants of one defect interface's plane closure.

    Built from the POST-DOS-fold chi/Eg node arrays (the closure is part of
    the parity configuration and activates only under
    ``dos_band_potentials`` + reference-layer DOS data). Densities m^-3.
    """
    bn_L: float      # exp(-(chi_s - chi_L)/V_T) <= 1
    bn_R: float
    bp_L: float      # exp(-((chi+Eg)_L - (chi+Eg)_s)/V_T) <= 1
    bp_R: float
    ni_s_sq: float   # N_C,ref * N_V,ref * exp(-Eg_s/V_T)
    n1_s: float      # trap level vs plane edges; n1_s*p1_s == ni_s_sq
    p1_s: float


@dataclass(frozen=True)
class InterfaceStateQSSResult:
    """Locally eliminated four-state interface response."""

    state_m3: np.ndarray
    bulk_flux_m2_s: np.ndarray
    cross_flux_m2_s: np.ndarray
    state_flux_m2_s: np.ndarray
    normalized_residual: float
    evaluations: int
    transport_model: str = FERMI_RICHARDSON


def build_plane_params(
    chi_L: float, chi_R: float, ceg_L: float, ceg_R: float,
    Nc_ref: float, Nv_ref: float, chi_ref: float, E_t_eV: float,
    V_T: float,
) -> PlaneInterfaceParams:
    """Derive the closure constants from folded band quantities.

    ``chi_i`` are the folded chi values at the two interface-adjacent nodes;
    ``ceg_i = chi_i + Eg_i`` (folded). ``chi_ref`` is the RAW reference-layer
    chi (the fold shift is zero on the reference layer, so the trap energy
    E_t below the reference CB needs no fold correction).
    """
    chi_s = max(chi_L, chi_R)            # electron edge: lower E_C
    ceg_s = min(ceg_L, ceg_R)            # hole edge: higher E_V
    eg_s = ceg_s - chi_s                 # reduced interface gap [eV]
    def _b(d):                           # Boltzmann penalty, d >= 0
        return math.exp(-min(max(d, 0.0), 60.0 * V_T) / V_T)
    depth_n = E_t_eV - (chi_s - chi_ref)  # trap depth below plane E_C [eV]
    # Clamp the level to the (reduced) plane gap: a trap pushed energetically
    # outside the gap by a band offset acts as a band-edge state — emission
    # cannot outrun the band-edge DOS. Without this, a deep cliff puts the
    # trap above the plane E_C and n1_s = N_C*exp(+|depth|/V_T) explodes,
    # suppressing the cliff recombination SCAPS resolves (measured: R 8
    # orders low at dE_C = -1.0, V_oc arm collapsed 734 -> 284 mV).
    depth_n = min(max(depth_n, 0.0), max(eg_s, 0.0))
    # exponentials guarded: |args| <= ~60 in practice (Eg/V_T ~ 60 worst)
    def _e(a):
        return math.exp(max(-120.0, min(120.0, a)))
    return PlaneInterfaceParams(
        bn_L=_b(chi_s - chi_L),
        bn_R=_b(chi_s - chi_R),
        bp_L=_b(ceg_L - ceg_s),
        bp_R=_b(ceg_R - ceg_s),
        ni_s_sq=Nc_ref * Nv_ref * _e(-eg_s / V_T),
        n1_s=Nc_ref * _e(-depth_n / V_T),
        p1_s=Nv_ref * _e(-(eg_s - depth_n) / V_T),
    )


def plane_rate(n_s: float, p_s: float, prm: PlaneInterfaceParams,
               v_n: float, v_p: float, *,
               allow_generation: bool = False) -> float:
    """P-V single-level rate on plane densities [m^-2 s^-1].

    ``allow_generation=False`` (default) keeps the historical NOGEN clamp:
    ``n_s p_s < n_{i,s}^2`` returns 0 instead of a negative rate.

    ``allow_generation=True`` returns the signed rate, so a depleted plane
    is a net carrier SOURCE — the physical Shockley-Read-Hall generation
    the review's F-04 asks for. This is sound HERE and not at the
    cross-carrier bulk sampling sites because ``prm.ni_s_sq`` is the plane
    mass-action product ``N_C N_V exp(-Eg_s/V_T)`` built from the reduced
    interface gap, i.e. a physically correct detailed-balance reference
    rather than a bulk-asymptotic one whose negative excursions are partly
    a sampling artifact.

    The signed branch is BOUNDED, which is what makes it safe: as
    ``n_s, p_s -> 0`` the numerator tends to ``-n_{i,s}^2`` while the
    denominator tends to the finite positive ``n1_s/v_p + p1_s/v_n``, so

        R -> -n_{i,s}^2 / (n1_s/v_p + p1_s/v_n)

    which is exactly the textbook depletion-generation limit. There is no
    runaway branch to guard against; the rate saturates on its own.
    """
    num = n_s * p_s - prm.ni_s_sq
    if num <= 0.0 and not allow_generation:
        return 0.0
    den = (n_s + prm.n1_s) / v_p + (p_s + prm.p1_s) / v_n
    return num / den


def solve_plane_densities(
    n_L: float, n_R: float, p_L: float, p_R: float,
    prm: PlaneInterfaceParams, v_n: float, v_p: float,
    s_supply: float = S_SUPPLY,
    *,
    allow_generation: bool = False,
) -> tuple[float, float, float]:
    """Solve the 2x2 plane closure; return ``(n_s, p_s, R)``.

    Inputs are the adjacent node densities already Boltzmann-projected to
    the plane POTENTIAL (caller applies the phi factors); the band-offset
    penalties live in ``prm``. Newton in (ln n_s, ln p_s) — positivity by
    construction, analytic Jacobian, damped steps. On the rare
    non-convergence the last iterate is used: fail-bounded (|R| capped by
    the delivery flux on the recombination side and by the finite depletion
    limit on the generation side), never fail-wild.

    ``allow_generation`` propagates to :func:`plane_rate`; see its docstring
    for why the signed branch is bounded. The plane densities THEMSELVES stay
    positive either way — the Newton unknowns are their logarithms — so
    enabling generation flips the sign of ``R`` but cannot produce a negative
    ``n_s`` or ``p_s``.
    """
    nb = prm.bn_L * max(n_L, 0.0) + prm.bn_R * max(n_R, 0.0)
    pb = prm.bp_L * max(p_L, 0.0) + prm.bp_R * max(p_R, 0.0)
    s2 = 2.0 * s_supply
    ln_n = math.log(max(nb / 2.0, _QSS_FLOOR))
    ln_p = math.log(max(pb / 2.0, _QSS_FLOOR))
    for _ in range(_QSS_MAX_NEWTON):
        n_s, p_s = math.exp(ln_n), math.exp(ln_p)
        num = n_s * p_s - prm.ni_s_sq
        den = (n_s + prm.n1_s) / v_p + (p_s + prm.p1_s) / v_n
        signed = num > 0.0 or allow_generation
        R = num / den if signed else 0.0
        F1 = s_supply * nb - s2 * n_s - R
        F2 = s_supply * pb - s2 * p_s - R
        scale = s_supply * (nb + pb) + abs(R) + _QSS_FLOOR
        if abs(F1) + abs(F2) < _QSS_TOL * scale:
            break
        # The Jacobian has to follow the branch the residual actually took,
        # or Newton solves a different problem than the one it evaluates.
        if signed:
            dR_dn = (p_s * den - num / v_p) / (den * den)
            dR_dp = (n_s * den - num / v_n) / (den * den)
        else:
            dR_dn = dR_dp = 0.0
        J11 = (-s2 - dR_dn) * n_s
        J12 = (-dR_dp) * p_s
        J21 = (-dR_dn) * n_s
        J22 = (-s2 - dR_dp) * p_s
        det = J11 * J22 - J12 * J21
        if det == 0.0 or not math.isfinite(det):
            break
        d_ln_n = (-F1 * J22 + F2 * J12) / det
        d_ln_p = (-F2 * J11 + F1 * J21) / det
        d_ln_n = max(-30.0, min(30.0, d_ln_n))
        d_ln_p = max(-30.0, min(30.0, d_ln_p))
        ln_n += d_ln_n
        ln_p += d_ln_p
    n_s, p_s = math.exp(ln_n), math.exp(ln_p)
    return n_s, p_s, plane_rate(
        n_s, p_s, prm, v_n, v_p, allow_generation=allow_generation,
    )


def compute_interface_te_fluxes_live(
    mat,
    iface_state: np.ndarray,
    n: np.ndarray,
    p: np.ndarray,
    phi: np.ndarray,
    *,
    v_th_eff: float,
    v_cross_eff: float,
    V_app: float = 0.0,
    V_T: float | None = None,
    interface_transport_model: str = FERMI_RICHARDSON,
) -> np.ndarray:
    """Live-referenced variant of ``compute_interface_te_fluxes`` (2026-06,
    P1 of scaps_mode).

    The original projects the state targets from EQUILIBRIUM-cache
    densities via the V-partition — an E3 approximation that under
    illumination pins the plane states at equilibrium scale (measured:
    ETL-side eq holes ~1e-4 m^-3 vs 1e22 live; SRH through the states
    ~1e-15 A/m^2; V_oc insensitive to every interface parameter). Here the
    targets are the LIVE adjacent node densities Boltzmann-projected to
    the plane potential (the supply construction validated by the QSS
    closure), so the plane carries quasi-Fermi-consistent densities and
    the two-sided SRH on it becomes the real SCAPS-style channel.

    Block convention (matches ``compute_interface_srh_on_state`` and the
    bulk-drain mapping in ``assemble_rhs``): components are
    ``(n_1s, p_1s, n_2s, p_2s)`` with side 1 on the right and side 2 on the
    left. The opt-in physical-offset path uses the two endpoints of the
    replaced face (``idx`` and ``idx-1``). The historical state prototype
    keeps its interior samples (``idx+1`` and ``idx-1``) for compatibility.
    """
    n_iface = len(mat.interface_V_partition_2)
    if n_iface == 0:
        return np.zeros(0, dtype=float)
    V_T_local = V_T if V_T is not None else (
        mat.V_T_device if hasattr(mat, "V_T_device") else _V_T_300
    )
    transport_model = validate_interface_transport_model(
        interface_transport_model
    )
    out = np.zeros(4 * n_iface, dtype=float)
    for k in range(n_iface):
        if k >= len(mat.interface_nodes):
            break
        idx = int(mat.interface_nodes[k])
        if idx <= 0 or idx >= len(phi):
            continue
        right = (
            idx
            if getattr(mat, "iface_state_physical_offsets", False)
            else min(idx + 1, len(phi) - 1)
        )
        left = idx - 1
        physical_supply = getattr(mat, "iface_state_physical_offsets", False)
        if getattr(mat, "iface_state_partition", False):
            # V-partition x live-QFL merge (P1, 2026-06): the sub-grid
            # interface band bending — per-side, per-carrier,
            # bias-dependent, with OPPOSITE signs for majority/minority
            # (the structure the uniform-suppression scan proved
            # necessary) — applied to the LIVE node densities. E3 used
            # this bending on equilibrium densities (wrong QFLs); the
            # plain live projection used node potentials only (no
            # bending). V_1 depletes ETL-side electrons / accumulates
            # holes; V_2 the mirror on the PVK side.
            V_tot = max(0.0, abs(float(mat.V_bi_eff)) - float(V_app))
            part = float(mat.interface_V_partition_2[k])
            v2 = max(-_EXP_CAP, min(_EXP_CAP, part * V_tot / V_T_local))
            v1 = max(-_EXP_CAP, min(_EXP_CAP,
                                    (1.0 - part) * V_tot / V_T_local))
            if physical_supply and transport_model in (
                FERMI_RICHARDSON,
                FERMI_DIRAC_RICHARDSON,
            ):
                n_1s_t = _project_density_for_transport_model(
                    n[right], mat.N_C_physical[right], -v1, transport_model
                )
                p_1s_t = _project_density_for_transport_model(
                    p[right], mat.N_V_physical[right], +v1, transport_model
                )
                n_2s_t = _project_density_for_transport_model(
                    n[left], mat.N_C_physical[left], +v2, transport_model
                )
                p_2s_t = _project_density_for_transport_model(
                    p[left], mat.N_V_physical[left], -v2, transport_model
                )
            else:
                n_1s_t = max(float(n[right]), 0.0) * math.exp(-v1)
                p_1s_t = max(float(p[right]), 0.0) * math.exp(+v1)
                n_2s_t = max(float(n[left]), 0.0) * math.exp(+v2)
                p_2s_t = max(float(p[left]), 0.0) * math.exp(-v2)
        else:
            # phi-projection of the live neighbours onto the plane
            eR = (float(phi[idx]) - float(phi[right])) / V_T_local
            eL = (float(phi[idx]) - float(phi[left])) / V_T_local
            eR = max(-_EXP_CAP, min(_EXP_CAP, eR))
            eL = max(-_EXP_CAP, min(_EXP_CAP, eL))
            if physical_supply and transport_model in (
                FERMI_RICHARDSON,
                FERMI_DIRAC_RICHARDSON,
            ):
                n_1s_t = _project_density_for_transport_model(
                    n[right], mat.N_C_physical[right], eR, transport_model
                )
                p_1s_t = _project_density_for_transport_model(
                    p[right], mat.N_V_physical[right], -eR, transport_model
                )
                n_2s_t = _project_density_for_transport_model(
                    n[left], mat.N_C_physical[left], eL, transport_model
                )
                p_2s_t = _project_density_for_transport_model(
                    p[left], mat.N_V_physical[left], -eL, transport_model
                )
            else:
                n_1s_t = max(float(n[right]), 0.0) * math.exp(eR)
                p_1s_t = max(float(p[right]), 0.0) * math.exp(-eR)
                n_2s_t = max(float(n[left]), 0.0) * math.exp(eL)
                p_2s_t = max(float(p[left]), 0.0) * math.exp(-eL)
        base = 4 * k
        n_1s = float(iface_state[base + 0])
        p_1s = float(iface_state[base + 1])
        n_2s = float(iface_state[base + 2])
        p_2s = float(iface_state[base + 3])
        local_velocity = (
            _interface_thermal_velocity(mat, k, v_th_eff)
            if transport_model == SCAPS_THERMAL_VELOCITY
            else v_th_eff
        )
        out[base + 0] = te_flux(n_1s_t, n_1s, local_velocity)
        out[base + 1] = te_flux(p_1s_t, p_1s, local_velocity)
        out[base + 2] = te_flux(n_2s_t, n_2s, local_velocity)
        out[base + 3] = te_flux(p_2s_t, p_2s, local_velocity)
        # cross-plane chi-step exchange — identical bounded form to the
        # eq-referenced variant (factored-larger-exponential, arg <= 0)
        if (
            v_cross_eff > 0.0
            and mat.interface_chi_step
            and len(mat.interface_chi_step) > k
        ):
            dE_c = float(mat.interface_chi_step[k])
            dE_g = float(mat.interface_Eg_step[k])
            if getattr(mat, "iface_state_physical_offsets", False):
                transmission = v_cross_eff / v_th_eff
                J_cross_n, J_cross_p = _physical_cross_fluxes(
                    mat,
                    k,
                    iface_state,
                    V_T_local,
                    transmission,
                    transport_model,
                    v_th_eff,
                )
            else:
                dE_c_density, dE_v_density = _interface_activity_steps(
                    mat,
                    k,
                    dE_c,
                    dE_g,
                    V_T_local,
                )
                ec_norm = dE_c_density / V_T_local
                ev_norm = dE_v_density / V_T_local
                if ec_norm >= 0.0:
                    J_cross_n = v_cross_eff * (
                        n_2s - n_1s * math.exp(-ec_norm)
                    )
                else:
                    J_cross_n = v_cross_eff * (
                        n_2s * math.exp(ec_norm) - n_1s
                    )
                if ev_norm >= 0.0:
                    J_cross_p = v_cross_eff * (
                        p_2s - p_1s * math.exp(-ev_norm)
                    )
                else:
                    J_cross_p = v_cross_eff * (
                        p_2s * math.exp(ev_norm) - p_1s
                    )
            out[base + 0] += J_cross_n
            out[base + 2] -= J_cross_n
            out[base + 1] += J_cross_p
            out[base + 3] -= J_cross_p
    return out


def compute_interface_srh_shared_on_state(
    iface_state: np.ndarray,
    stack,
    mat,
    *,
    density_regularization_width_m3: float = 0.0,
) -> np.ndarray:
    """Shared-occupancy single-trap P-V on the interface-plane STATE
    densities (2026-06, P1 of scaps_mode).

    The cross-pair form (``compute_interface_srh_on_state``) pairs
    majority-side densities (n_1s*p_2s), so the trap-level densities
    never compete and the E_t response is dead; on BULK-node samples the
    shared-occupancy form collapsed the CBO trend (densities 6-13 orders
    above n1/p1 — falsified). On the live-projected PLANE states — the
    substrate SCAPS actually evaluates on — it is the correct object:

        R = (nS*pS - refS) / ((nS + n1S)/v_p + (pS + p1S)/v_n)

    with nS = n_1s + n_2s, pS = p_1s + p_2s, per-side trap-level
    densities from the existing caches (interface_n1_L/R, p1_L/R), and
    the reference evaluated on the DARK-EQUILIBRIUM STATE projections
    (same V-partition formula as ``_compute_iface_state_dark_eq``) so
    R = 0 exactly at the dark-equilibrium state block. Consumption is
    split between the two faces by capture share (n_is/nS, p_is/pS) so
    the sink layout stays compatible with the assemble_rhs drain mapping.
    """
    ifaces = electrical_interfaces(stack)
    n_iface = len(mat.interface_V_partition_2)
    out = np.zeros(4 * n_iface, dtype=float)
    if n_iface == 0 or not ifaces:
        return out
    V_T_local = mat.V_T_device if hasattr(mat, "V_T_device") else _V_T_300
    V_total = float(mat.V_bi_eff)
    CAP = 30.0
    for k in range(n_iface):
        if k >= len(ifaces):
            break
        v_n, v_p = ifaces[k]
        if mat.interface_calibration_factor:
            cf = float(mat.interface_calibration_factor[k])
            v_n = v_n * cf
            v_p = v_p * cf
        if v_n == 0.0 and v_p == 0.0:
            continue
        base = 4 * k
        from perovskite_sim.physics.regularization import compact_positive_part

        n_1s = float(compact_positive_part(
            iface_state[base + 0], density_regularization_width_m3,
        ))
        p_1s = float(compact_positive_part(
            iface_state[base + 1], density_regularization_width_m3,
        ))
        n_2s = float(compact_positive_part(
            iface_state[base + 2], density_regularization_width_m3,
        ))
        p_2s = float(compact_positive_part(
            iface_state[base + 3], density_regularization_width_m3,
        ))
        nS = n_1s + n_2s
        pS = p_1s + p_2s
        if nS <= 0.0 or pS <= 0.0:
            continue
        # dark-equilibrium state projections (the dark-eq init values)
        part = float(mat.interface_V_partition_2[k])
        v2 = max(-CAP, min(CAP, part * V_total / V_T_local))
        v1 = max(-CAP, min(CAP, (1.0 - part) * V_total / V_T_local))
        n1s_eq = float(mat.interface_n_R_eq[k]) * math.exp(-v1)
        p1s_eq = float(mat.interface_p_R_eq[k]) * math.exp(+v1)
        n2s_eq = float(mat.interface_n_L_eq[k]) * math.exp(+v2)
        p2s_eq = float(mat.interface_p_L_eq[k]) * math.exp(-v2)
        refS = (n1s_eq + n2s_eq) * (p1s_eq + p2s_eq)
        n1S = (float(mat.interface_n1_L[k]) + float(mat.interface_n1_R[k])
               if mat.interface_n1_L else 2.0 * float(mat.interface_n1[k]))
        p1S = (float(mat.interface_p1_L[k]) + float(mat.interface_p1_R[k])
               if mat.interface_p1_L else 2.0 * float(mat.interface_p1[k]))
        num = nS * pS - refS
        if num <= 0.0:
            continue                       # NOGEN clamp
        denominator = (nS + n1S) / v_p + (pS + p1S) / v_n
        from perovskite_sim.physics.recombination import _record_srh_denominator

        _record_srh_denominator("interface", np.asarray(denominator))
        R = num / denominator
        out[base + 0] = -R * (n_1s / nS)
        out[base + 1] = -R * (p_1s / pS)
        out[base + 2] = -R * (n_2s / nS)
        out[base + 3] = -R * (p_2s / pS)
    return out


def compute_interface_srh_occupancy_on_state(
    iface_state: np.ndarray,
    stack,
    mat,
    *,
    density_regularization_width_m3: float = 0.0,
) -> np.ndarray:
    """Net side-resolved capture from a shared interface-trap occupancy.

    For each interface, the steady trap occupation is

    ``f = [v_n sum(n_i) + v_p sum(p1_i)] /``
    ``    [v_n sum(n_i+n1_i) + v_p sum(p_i+p1_i)]``.

    Electron capture on side ``i`` is
    ``v_n * [n_i*(1-f) - n1_i*f]`` and hole capture is
    ``v_p * [p_i*f - p1_i*(1-f)]``. These signed capture/emission terms obey
    ``sum(R_n_i) == sum(R_p_i)`` by construction, including at thermal
    equilibrium. This is the detailed-balance closure used only by the opt-in
    physical QSS interface path; the calibrated legacy shared-rate model above
    remains unchanged.
    """
    ifaces = electrical_interfaces(stack)
    n_iface = len(mat.interface_V_partition_2)
    out = np.zeros(4 * n_iface, dtype=float)
    for k in range(min(n_iface, len(ifaces))):
        v_n, v_p = (float(value) for value in ifaces[k])
        if mat.interface_calibration_factor:
            calibration = float(mat.interface_calibration_factor[k])
            v_n *= calibration
            v_p *= calibration
        if v_n == 0.0 and v_p == 0.0:
            continue
        base = 4 * k
        from perovskite_sim.physics.regularization import compact_positive_part

        n_right = float(compact_positive_part(
            iface_state[base + 0], density_regularization_width_m3,
        ))
        p_right = float(compact_positive_part(
            iface_state[base + 1], density_regularization_width_m3,
        ))
        n_left = float(compact_positive_part(
            iface_state[base + 2], density_regularization_width_m3,
        ))
        p_left = float(compact_positive_part(
            iface_state[base + 3], density_regularization_width_m3,
        ))
        if mat.interface_n1_L and mat.interface_n1_R:
            n1_left = max(float(mat.interface_n1_L[k]), 0.0)
            n1_right = max(float(mat.interface_n1_R[k]), 0.0)
            p1_left = max(float(mat.interface_p1_L[k]), 0.0)
            p1_right = max(float(mat.interface_p1_R[k]), 0.0)
        else:
            n1_left = n1_right = max(float(mat.interface_n1[k]), 0.0)
            p1_left = p1_right = max(float(mat.interface_p1[k]), 0.0)
        numerator = v_n * (n_left + n_right) + v_p * (
            p1_left + p1_right
        )
        denominator = v_n * (
            n_left + n_right + n1_left + n1_right
        ) + v_p * (
            p_left + p_right + p1_left + p1_right
        )
        if denominator <= 0.0 or not np.isfinite(denominator):
            continue
        occupancy = min(1.0, max(0.0, numerator / denominator))
        empty = 1.0 - occupancy
        capture_n_right = v_n * (n_right * empty - n1_right * occupancy)
        capture_p_right = v_p * (p_right * occupancy - p1_right * empty)
        capture_n_left = v_n * (n_left * empty - n1_left * occupancy)
        capture_p_left = v_p * (p_left * occupancy - p1_left * empty)
        out[base + 0] = -capture_n_right
        out[base + 1] = -capture_p_right
        out[base + 2] = -capture_n_left
        out[base + 3] = -capture_p_left
    return out


def solve_interface_states_live_qss(
    mat,
    stack,
    n: np.ndarray,
    p: np.ndarray,
    phi: np.ndarray,
    *,
    V_app: float = 0.0,
    v_th_eff: float = 1.0e5,
    cross_transmission: float = 1.0,
    interface_transport_model: str = FERMI_RICHARDSON,
    residual_tolerance: float = 1.0e-7,
    max_evaluations: int = 200,
    fail_on_residual: bool = True,
    density_regularization_width_m3: float = 0.0,
) -> InterfaceStateQSSResult:
    """Eliminate the four plane densities at every interface locally.

    The algebraic balance is

    ``bulk supply + cross-interface exchange + shared-occupancy SRH = 0``.

    Solving this small positive system in log-density space removes the full
    thermal-velocity modes from the device-level Newton Jacobian. The bulk
    continuity equations receive only ``bulk_flux_m2_s``; cross-interface
    exchange remains internal to the plane and is therefore not counted as a
    second bulk drain.
    """
    n_iface = len(mat.interface_V_partition_2)
    if n_iface == 0:
        empty = np.zeros(0, dtype=float)
        return InterfaceStateQSSResult(empty, empty, empty, empty, 0.0, 0)
    if not getattr(mat, "iface_state_physical_offsets", False):
        raise ValueError(
            "live QSS interface elimination requires physical band offsets"
        )
    if not np.isfinite(v_th_eff) or v_th_eff <= 0.0:
        raise ValueError("v_th_eff must be finite and positive")
    if (
        not np.isfinite(cross_transmission)
        or cross_transmission <= 0.0
        or cross_transmission > 1.0
    ):
        raise ValueError("cross_transmission must lie in (0, 1]")
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be finite and positive")
    if max_evaluations <= 0:
        raise ValueError("max_evaluations must be positive")
    transport_model = validate_interface_transport_model(
        interface_transport_model
    )

    size = 4 * n_iface
    zero_state = np.zeros(size, dtype=float)
    state_capacity = np.empty(size, dtype=float)
    for interface_index, interface_node in enumerate(mat.interface_nodes):
        base = 4 * interface_index
        right = int(interface_node)
        left = right - 1
        state_capacity[base : base + 4] = (
            float(mat.N_C_physical[right]),
            float(mat.N_V_physical[right]),
            float(mat.N_C_physical[left]),
            float(mat.N_V_physical[left]),
        )
    bulk_at_zero = compute_interface_te_fluxes_live(
        mat,
        zero_state,
        n,
        p,
        phi,
        v_th_eff=v_th_eff,
        v_cross_eff=0.0,
        V_app=V_app,
        interface_transport_model=transport_model,
    )
    bulk_velocities = np.full(size, float(v_th_eff), dtype=float)
    if transport_model == SCAPS_THERMAL_VELOCITY:
        for interface_index in range(n_iface):
            velocity = _interface_thermal_velocity(
                mat,
                interface_index,
                v_th_eff,
            )
            bulk_velocities[4 * interface_index : 4 * interface_index + 4] = (
                velocity
            )
    bulk_target = np.maximum(bulk_at_zero / bulk_velocities, 1.0e-30)

    def te_state_flux(state: np.ndarray) -> np.ndarray:
        return compute_interface_te_fluxes_live(
            mat,
            state,
            n,
            p,
            phi,
            v_th_eff=v_th_eff,
            v_cross_eff=v_th_eff * cross_transmission,
            V_app=V_app,
            interface_transport_model=transport_model,
        )

    # Bulk coupling is diagonal and much faster than the interface capture
    # channels, so the four one-sided reservoir projections are the natural
    # zero-order state. The older linear TE seed is invalid once reciprocal
    # cross-interface supply uses bounded Fermi occupations.
    bounded_states = transport_model == FERMI_RICHARDSON
    if bounded_states:
        seed = np.minimum(
            np.maximum(bulk_target, 1.0e-30),
            state_capacity * (1.0 - 1.0e-12),
        )
    else:
        seed = np.maximum(bulk_target, 1.0e-30)

    def components(state_variable: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if bounded_states:
            occupation = np.asarray(
                [
                    _fermi_edge_occupation(float(value))
                    for value in state_variable
                ],
                dtype=float,
            )
            state = state_capacity * occupation
        else:
            state = np.exp(np.clip(state_variable, -200.0, 200.0))
        state_flux = te_state_flux(
            state
        ) + compute_interface_srh_occupancy_on_state(
            state,
            stack,
            mat,
            density_regularization_width_m3=density_regularization_width_m3,
        )
        return state, state_flux

    if bounded_states:
        seed_occupation = np.clip(
            seed / state_capacity,
            1.0e-60,
            1.0 - 1.0e-12,
        )
        variable_seed = np.log(seed_occupation) - np.log1p(-seed_occupation)
    else:
        variable_seed = np.log(seed)
    _, residual_seed = components(variable_seed)
    # Fixed scaling keeps the nested root deterministic under the outer
    # finite-difference perturbations while balancing majority/minority states.
    scale = np.maximum(np.abs(residual_seed), bulk_velocities * seed)
    scale = np.maximum(scale, 1.0)

    def normalized_residual(state_variable: np.ndarray) -> np.ndarray:
        return components(state_variable)[1] / scale

    solved = least_squares(
        normalized_residual,
        variable_seed,
        jac="3-point",
        x_scale="jac",
        bounds=(-200.0, 200.0),
        ftol=1.0e-13,
        xtol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=max_evaluations,
    )
    state, state_flux = components(solved.x)
    normalized = float(np.max(np.abs(state_flux / scale)))
    total_evaluations = int(solved.nfev)
    if normalized > residual_tolerance:
        candidates = [solved]
        for recovery_seed in (
            np.clip(variable_seed, -30.0, 80.0),
            np.clip(variable_seed, -12.0, 60.0),
            np.zeros_like(variable_seed),
        ):
            recovered = least_squares(
                normalized_residual,
                recovery_seed,
                method="lm",
                jac="2-point",
                x_scale="jac",
                ftol=1.0e-13,
                xtol=1.0e-13,
                gtol=1.0e-13,
                max_nfev=max_evaluations,
            )
            candidates.append(recovered)
            total_evaluations += int(recovered.nfev)
        solved = min(
            candidates,
            key=lambda candidate: float(
                np.max(np.abs(normalized_residual(candidate.x)))
            ),
        )
        state, state_flux = components(solved.x)
        normalized = float(np.max(np.abs(state_flux / scale)))
    non_finite = (
        not np.all(np.isfinite(state))
        or not np.all(np.isfinite(state_flux))
    )
    inaccurate = not solved.success or normalized > residual_tolerance
    if non_finite or (fail_on_residual and inaccurate):
        raise RuntimeError(
            "local interface-state QSS solve did not converge: "
            f"normalized residual={normalized:.6g}, "
            f"evaluations={total_evaluations}"
        )
    raw_bulk_flux = compute_interface_te_fluxes_live(
        mat,
        state,
        n,
        p,
        phi,
        v_th_eff=v_th_eff,
        v_cross_eff=0.0,
        V_app=V_app,
        interface_transport_model=transport_model,
    )
    thermionic_flux = te_state_flux(state)
    cross_flux = thermionic_flux - raw_bulk_flux
    srh_flux = compute_interface_srh_occupancy_on_state(
        state,
        stack,
        mat,
        density_regularization_width_m3=density_regularization_width_m3,
    )
    # Project the returned reservoir flux onto the exactly conservative local
    # manifold. ``raw_bulk_flux`` is v_th * (target - state), a subtraction of
    # nearly equal O(1e20--1e24) densities at full thermal velocity. Its tiny
    # steady component therefore carries finite floating-point noise even when
    # the constitutive root above is converged. Cross exchange and SRH do not
    # suffer that cancellation, and the QSS equation requires their negative
    # sum to equal the bulk flux. The unprojected constitutive mismatch remains
    # in ``state_flux_m2_s`` and has already passed the fail-closed residual
    # gate, so this enforces conservation without hiding a failed local solve.
    bulk_flux = -(cross_flux + srh_flux)
    return InterfaceStateQSSResult(
        state_m3=state,
        bulk_flux_m2_s=bulk_flux,
        cross_flux_m2_s=cross_flux,
        state_flux_m2_s=state_flux,
        normalized_residual=normalized,
        evaluations=total_evaluations,
        transport_model=transport_model,
    )


def compute_interface_trap_charge(
    iface_state: np.ndarray,
    stack,
    mat,
) -> np.ndarray:
    """Occupancy-derived interface trapped charge (P1 of scaps_mode,
    2026-06) — the surviving E_t mechanism after three rate-algebra
    falsifications.

    Steady-state single-level occupancy on the plane-state densities:

        f = (v_n*nS + v_p*p1S) / (v_n*(nS + n1S) + v_p*(pS + p1S))

    (capture of electrons + emission of holes fill the trap; the standard
    SRH occupancy with per-side trap-level sums). The trapped areal
    charge MAGNITUDE q*N_t*(f - f_eq) is returned per interface [C/m^2];
    the caller applies the sign convention (acceptor-like -1 / donor-like
    +1 via ``MaterialArrays.iface_state_charge``) and converts to a
    volumetric Poisson contribution at the interface node. ``f_eq`` is
    evaluated on the dark-equilibrium state projections, so equilibrium
    is exactly charge-neutral by construction.
    """
    from perovskite_sim.models.device import electrical_interface_defects

    ifaces = electrical_interfaces(stack)
    defects = electrical_interface_defects(stack)
    n_iface = len(mat.interface_V_partition_2)
    out = np.zeros(n_iface, dtype=float)
    if n_iface == 0 or not ifaces:
        return out
    def _f(nS, pS, n1S, p1S, v_n, v_p):
        den = v_n * (nS + n1S) + v_p * (pS + p1S)
        if den <= 0.0:
            return 0.0
        return (v_n * nS + v_p * p1S) / den

    for k in range(n_iface):
        if k >= len(ifaces) or k >= len(defects):
            break
        defect = defects[k]
        if defect is None or not getattr(defect, "N_t_cm2", None):
            continue
        v_n, v_p = ifaces[k]
        if mat.interface_calibration_factor:
            cf = float(mat.interface_calibration_factor[k])
            v_n = v_n * cf
            v_p = v_p * cf
        if v_n == 0.0 and v_p == 0.0:
            continue
        base = 4 * k
        nS = (max(float(iface_state[base + 0]), 0.0)
              + max(float(iface_state[base + 2]), 0.0))
        pS = (max(float(iface_state[base + 1]), 0.0)
              + max(float(iface_state[base + 3]), 0.0))
        n1S = (float(mat.interface_n1_L[k]) + float(mat.interface_n1_R[k])
               if mat.interface_n1_L else 2.0 * float(mat.interface_n1[k]))
        p1S = (float(mat.interface_p1_L[k]) + float(mat.interface_p1_R[k])
               if mat.interface_p1_L else 2.0 * float(mat.interface_p1[k]))
        f = _f(nS, pS, n1S, p1S, v_n, v_p)
        # Neutral reference IN THE SAME GAUGE as the live states: plain
        # bulk-equilibrium sums. The E3 partition projections explode
        # their accumulation factors to ~1e33, forcing f_eq = 1 while the
        # live f computes ~0 at shallow E_t — the resulting full-sheet
        # artifact charge (q*N_t, measured -1.6e-3 C/m^2 at EVERY bias)
        # reversed the terminal current. The live states at dark
        # equilibrium settle near the phi-projected node densities, whose
        # gauge the bulk-eq sums match to O(1).
        nS_eq = (float(mat.interface_n_R_eq[k])
                 + float(mat.interface_n_L_eq[k]))
        pS_eq = (float(mat.interface_p_R_eq[k])
                 + float(mat.interface_p_L_eq[k]))
        f_eq = _f(nS_eq, pS_eq, n1S, p1S, v_n, v_p)
        N_t_m2 = float(defect.N_t_cm2) * 1.0e4      # cm^-2 -> m^-2
        out[k] = 1.602176634e-19 * N_t_m2 * (f - f_eq)
    return out
