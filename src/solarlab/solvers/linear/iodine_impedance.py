"""Small-signal AC impedance solver for 1D iodine reaction-diffusion systems.

Linearizes the assembled 1D ``(phi, n, p, I^-, I^0, I_2)`` DAE operator around
any admitted steady or transient snapshot state using the ``exp(+i * omega * t)``
phasor convention and reuses SolarLab's shared
:func:`~solarlab.solvers.linear.frequency_domain.solve_frequency_systems`
without duplicating linear solver infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import lil_matrix

from solarlab.contracts import ContractError, Vector, frozen_array
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.equations.assembly.iodine_1d import (
    IODIDE,
    IODINE_BLOCKS,
    POTENTIAL,
    AssembledConstrainedIodine1DProblem,
    AssembledIodine1DProblem,
    iodine_system,
)
from solarlab.equations.assembly.quasi_fermi import quasi_fermi_potential_tangents
from solarlab.equations.dc_1d import CapabilityError, DC1DProblem
from solarlab.equations.iodine_1d import CompiledIodineDevice1D, IodineState1D
from solarlab.solvers.linear.frequency_domain import (
    FrequencyOptions,
    FrequencySolution,
    solve_frequency_systems,
)

__all__ = [
    "ChemResponseMode",
    "IodineFrequencyResponse",
    "IodineImpedanceResult",
    "solve_iodine_small_signal_1d",
]

ComplexVector: TypeAlias = NDArray[np.complex128]
ChemResponseMode: TypeAlias = Literal["mobile", "frozen"]


def _frozen_complex(values: ArrayLike) -> ComplexVector:
    arr = np.array(values, dtype=np.complex128, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, slots=True)
class IodineFrequencyResponse:
    """Complex small-signal response at one frequency ``f`` [Hz]."""

    frequency_Hz: float
    omega_rad_s: float
    voltage_phasor_V: complex
    current_density_phasor_A_m2: complex
    admittance_S_m2: complex
    impedance_Ohm_m2: complex
    apparent_capacitance_F_m2: float
    maxwell_spread_A_m2_V: float
    backward_error: float
    perturbation_u: ComplexVector
    species_inventory_perturbation_m2: dict[str, complex]


@dataclass(frozen=True, slots=True)
class IodineImpedanceResult:
    """Bounded AC impedance sweep across a distinct frequency grid."""

    frequencies_Hz: Vector
    voltage_phasor_V: complex
    chem_mode: ChemResponseMode
    operating_voltage_V: float
    generation_scale: float
    responses: tuple[IodineFrequencyResponse, ...]
    admittance_S_m2: ComplexVector
    impedance_Ohm_m2: ComplexVector
    apparent_capacitance_F_m2: Vector
    max_maxwell_spread_A_m2_V: float
    max_backward_error: float
    device_sha256: str
    scientific_qualification: bool = False


def _carrier_flux_perturbation_m2_s(
    problem: AssembledIodine1DProblem,
    state: IodineState1D,
    fixed_x: ComplexVector,
) -> tuple[ComplexVector, ComplexVector]:
    """Perturbed interior electron and hole particle fluxes [m^-2 s^-1]."""
    comp = problem.device.electronic_component
    n_c = problem.device.node_count
    u_0 = state.u[: 3 * n_c]
    cf = problem.carrier_fluxes(state.u)
    vt = comp.thermal_voltage_V
    e_state = state.electronic_state
    ft = None if comp.field_transport is None else comp.field_transport.evaluate(u_0[:n_c])

    out: list[ComplexVector] = []
    for offset, dens, flux_obj, sign in (
        (n_c, e_state.electron_density_m3, cf.electron, 1.0),
        (2 * n_c, e_state.hole_density_m3, cf.hole, -1.0),
    ):
        drive = sign * np.diff(u_0[offset : offset + n_c])
        left_phi, _ = quasi_fermi_potential_tangents(flux_obj, drive, sign, vt)
        if ft is not None:
            diff = (
                ft.electron_diffusivity_m2_s
                if offset == n_c
                else ft.hole_diffusivity_m2_s
            )
            ddiff = (
                ft.electron_d_diffusivity_d_field
                if offset == n_c
                else ft.hole_d_diffusivity_d_field
            )
            left_phi = left_phi + (flux_obj.flux / diff) * ddiff / np.diff(comp.nodes_m)
        left_z = sign * flux_obj.left_density_derivative * dens[:-1]
        df = (
            left_phi * (fixed_x[: n_c - 1] - fixed_x[1:n_c])
            + left_z
            * (fixed_x[offset : offset + n_c - 1] - fixed_x[offset + 1 : offset + n_c])
            + sign
            * flux_obj.flux
            * (fixed_x[offset + 1 : offset + n_c] + fixed_x[1:n_c] / vt)
        )
        out.append(np.asarray(df, dtype=np.complex128))
    return out[0], out[1]


def _iodide_flux_perturbation_m2_s(
    problem: AssembledIodine1DProblem,
    state: IodineState1D,
    fixed_x: ComplexVector,
    chem_mode: ChemResponseMode,
) -> ComplexVector:
    """Perturbed interior iodide particle flux embedded on all ``N - 1`` faces."""
    dev = problem.device
    n_c = dev.node_count
    out: NDArray[np.complex128] = np.zeros(n_c - 1, dtype=np.complex128)
    if chem_mode == "frozen":
        return out
    assert problem.binding is not None
    ev_state = problem.binding.assembler.state(state.u, problem.inputs)
    if ev_state is None:
        raise FloatingPointError("inadmissible operating state in iodide AC flux")
    _, left, right = problem.binding.chem_transports[IODIDE].flux_evaluation(
        ev_state, derivatives=True
    )
    idx = dev.chem_node_indices
    x_phi_chem = fixed_x[idx]
    x_iodide = fixed_x[dev.chem_slices[IODIDE]]
    d_flux_chem = (
        np.asarray(left[POTENTIAL], dtype=np.float64) * x_phi_chem[:-1]
        + np.asarray(right[POTENTIAL], dtype=np.float64) * x_phi_chem[1:]
        + np.asarray(left[IODIDE], dtype=np.float64) * x_iodide[:-1]
        + np.asarray(right[IODIDE], dtype=np.float64) * x_iodide[1:]
    )
    start = int(idx[0])
    out[start : start + d_flux_chem.size] = d_flux_chem
    return out


def solve_iodine_small_signal_1d(
    device: CompiledIodineDevice1D,
    state: IodineState1D,
    frequencies_Hz: ArrayLike,
    *,
    generation_scale: float = 0.0,
    voltage_phasor_V: complex = 1.0,
    chem_mode: ChemResponseMode = "mobile",
    inventory_node: int = 0,
    options: FrequencyOptions | None = None,
) -> IodineImpedanceResult:
    """Solve small-signal AC impedance ``Z(omega)`` on a 1D iodine aging stack.

    Parameters
    ----------
    device:
        Compiled 1D iodine-coupled device.
    state:
        Operating state ``(u_0, V_0)`` around which ``J`` and ``K = -S_u`` are
        linearized.
    frequencies_Hz:
        1D array of distinct nonnegative frequencies ``f`` [Hz].
    generation_scale:
        Operating optical generation scale ``G`` matching ``state``.
    voltage_phasor_V:
        Complex terminal voltage perturbation amplitude ``V_tilde`` [V].
    chem_mode:
        ``"mobile"`` includes full coupled ``I^- / I^0 / I_2`` reaction-diffusion
        perturbations; ``"frozen"`` locks all three chemical species at ``0`` AC
        perturbation to isolate pure electronic impedance.
    inventory_node:
        Cell index used if ``0 Hz`` is requested on a closed-domain stack.
    options:
        Linear frequency solver tolerances and iteration controls.
    """
    if state.device_sha256 != device.identity:
        raise ContractError("iodine_ac_state_device_mismatch")
    if chem_mode not in ("mobile", "frozen"):
        raise ContractError("invalid_chem_mode")
    voltage = complex(voltage_phasor_V)
    if not (math.isfinite(voltage.real) and math.isfinite(voltage.imag)) or voltage == 0.0j:
        raise ContractError("voltage_phasor_V_must_be_finite_nonzero")
    g_scale = float(generation_scale)
    if not math.isfinite(g_scale) or g_scale < 0.0:
        raise ContractError("generation_scale_must_be_finite_nonnegative")

    comp = device.electronic_component
    if any(ch.kind != "fixed_reservoir" for ch in comp.channels):
        raise CapabilityError(
            "iodine small-signal AC currently requires fixed_reservoir carrier contacts"
        )

    controls = FrequencyOptions() if options is None else options
    raw_freqs = np.asarray(frequencies_Hz)
    if np.iscomplexobj(raw_freqs):
        raise ValueError("frequencies_Hz must be real")
    freqs = np.asarray(raw_freqs, dtype=np.float64)
    if (
        freqs.ndim != 1
        or freqs.size == 0
        or freqs.size > controls.max_frequencies
        or not np.all(np.isfinite(freqs))
        or np.any(freqs < 0.0)
        or np.unique(freqs).size != freqs.size
    ):
        raise ValueError("frequencies_Hz must be a 1D array of distinct nonnegative Hz")

    v_op = state.electronic_state.voltage_V
    unconstrained_sys = iodine_system(device, constrained_steady=False)
    problem = AssembledIodine1DProblem(
        device=device,
        voltage_V=v_op,
        generation_scale=g_scale,
        binding=unconstrained_sys,
    )
    j_base = problem.jacobian(state.u)
    k_base = problem.storage_jacobian(state.u)

    n_c = device.node_count
    vt = comp.thermal_voltage_V
    phi_right = complex(-comp.voltage_polarity * voltage)
    known_base: dict[int, complex] = {
        0: 0.0j,
        n_c - 1: phi_right,
        n_c: 0.0j,
        2 * n_c - 1: -phi_right / vt,
        2 * n_c: 0.0j,
        3 * n_c - 1: -phi_right / vt,
    }

    j_work = lil_matrix(j_base, dtype=np.float64)
    k_work = lil_matrix(k_base, dtype=np.float64)
    known = dict(known_base)
    if chem_mode == "frozen":
        # Pure electronic response against frozen static iodide space charge:
        # chemical reactions & ionic motion are frozen out, preserving exact
        # Kirchhoff electronic current continuity div(J_n + J_p + i*w*D) == 0.
        dc_prob = DC1DProblem(comp, v_op, g_scale)
        j_dc = dc_prob.jacobian(state.u[: 3 * n_c])
        k_dc = dc_prob.storage_jacobian(state.u[: 3 * n_c])
        j_work[: 3 * n_c, :] = 0.0
        k_work[: 3 * n_c, :] = 0.0
        j_work[: 3 * n_c, : 3 * n_c] = j_dc
        k_work[: 3 * n_c, : 3 * n_c] = k_dc
        for row_idx in range(3 * n_c, device.state_size):
            j_work.rows[row_idx] = [row_idx]
            j_work.data[row_idx] = [1.0]
            k_work.rows[row_idx] = []
            k_work.data[row_idx] = []
            known[row_idx] = 0.0j

    j_csc = j_work.tocsc()
    k_csc = k_work.tocsc()
    prescribed_vec: NDArray[np.complex128] = np.zeros(
        device.state_size, dtype=np.complex128
    )
    for k_idx, k_val in known.items():
        prescribed_vec[k_idx] = k_val
    rhs: NDArray[np.complex128] = np.zeros(device.state_size, dtype=np.complex128)
    fixed_sorted = sorted(known)
    rhs[fixed_sorted] = (j_csc[fixed_sorted, :] @ prescribed_vec)

    # Separate 0 Hz on closed mobile domains (which requires the inventory constraint row)
    zero_mask = freqs == 0.0
    pos_freqs = freqs[~zero_mask]
    sol_by_freq: dict[float, FrequencySolution] = {}

    if pos_freqs.size > 0:
        pos_sols = solve_frequency_systems(
            j_csc,
            k_csc,
            rhs,
            pos_freqs,
            options=controls,
            fixed_coordinates=known,
        )
        for sol in pos_sols:
            sol_by_freq[sol.frequency_Hz] = sol

    if np.any(zero_mask):
        if chem_mode == "mobile" and not device.has_open_iodine_loss:
            c_prob = AssembledConstrainedIodine1DProblem(
                device=device,
                voltage_V=v_op,
                generation_scale=g_scale,
                inventory_node=inventory_node,
            )
            j_zero = c_prob.jacobian(state.u)
            k_zero = lil_matrix(k_csc.shape, dtype=np.float64).tocsc()
            rhs_zero: NDArray[np.complex128] = np.zeros(
                device.state_size, dtype=np.complex128
            )
            rhs_zero[fixed_sorted] = j_zero[fixed_sorted, :] @ prescribed_vec
            zero_sol = solve_frequency_systems(
                j_zero,
                k_zero,
                rhs_zero,
                np.array([0.0]),
                options=controls,
                fixed_coordinates=known,
            )[0]
        else:
            zero_sol = solve_frequency_systems(
                j_csc,
                k_csc,
                rhs,
                np.array([0.0]),
                options=controls,
                fixed_coordinates=known,
            )[0]
        sol_by_freq[0.0] = zero_sol

    # Tangents dc_j / du_j at operating point for species inventory readout
    _, _, _, _, _, _, mm, m0, m2 = device.fields_and_tangents(state.u, voltage_V=v_op)
    tangents_map = {
        "iodide": mm,
        "iodine_atom": m0,
        "iodine_molecule": m2,
    }
    w_chem = device.chem_geometry.cell_measures_m
    cap_faces = comp.poisson.face_capacitance_F_m2
    dx_faces = np.diff(comp.nodes_m)
    total_len = float(math.fsum(dx_faces.tolist()))
    slices = device.chem_slices

    responses: list[IodineFrequencyResponse] = []
    y_list: list[complex] = []
    z_list: list[complex] = []
    c_list: list[float] = []

    for f_hz in freqs:
        sol = sol_by_freq[float(f_hz)]
        omega = 2.0 * math.pi * float(f_hz)
        x_fixed = np.asarray(sol.solution, dtype=np.complex128)

        dfn, dfp = _carrier_flux_perturbation_m2_s(problem, state, x_fixed)
        df_iodide = _iodide_flux_perturbation_m2_s(problem, state, x_fixed, chem_mode)
        j_cond_faces = Q * (dfp - dfn - df_iodide)
        j_disp_faces = 1j * omega * (-cap_faces * np.diff(x_fixed[:n_c]))
        j_tot_faces = j_cond_faces + j_disp_faces

        # Spatial mean & all-face Maxwell uniformity spread
        j_mean = complex(
            math.fsum((dx_faces * j_tot_faces.real).tolist()) / total_len,
            math.fsum((dx_faces * j_tot_faces.imag).tolist()) / total_len,
        )
        j_in = complex(comp.voltage_polarity * j_mean)
        admittance = j_in / voltage
        impedance = (1.0 / admittance) if admittance != 0.0j else complex(math.nan, math.nan)
        c_app = float(admittance.imag / omega) if omega > 0.0 else math.nan
        spread = float(np.max(np.abs(j_tot_faces - j_tot_faces[0]))) / abs(voltage)

        sp_inv: dict[str, complex] = {}
        for sp in IODINE_BLOCKS:
            if chem_mode == "frozen":
                sp_inv[sp] = 0.0j
            else:
                dc = tangents_map[sp] * x_fixed[slices[sp]]
                sp_inv[sp] = complex(
                    math.fsum((w_chem * dc.real).tolist()),
                    math.fsum((w_chem * dc.imag).tolist()),
                )

        responses.append(
            IodineFrequencyResponse(
                frequency_Hz=float(f_hz),
                omega_rad_s=omega,
                voltage_phasor_V=voltage,
                current_density_phasor_A_m2=j_in,
                admittance_S_m2=admittance,
                impedance_Ohm_m2=impedance,
                apparent_capacitance_F_m2=c_app,
                maxwell_spread_A_m2_V=spread,
                backward_error=sol.componentwise_backward_error,
                perturbation_u=_frozen_complex(x_fixed),
                species_inventory_perturbation_m2=sp_inv,
            )
        )
        y_list.append(admittance)
        z_list.append(impedance)
        c_list.append(c_app)

    c_arr = np.array(c_list, dtype=np.float64, copy=True)
    c_arr.setflags(write=False)
    return IodineImpedanceResult(
        frequencies_Hz=frozen_array(freqs),
        voltage_phasor_V=voltage,
        chem_mode=chem_mode,
        operating_voltage_V=v_op,
        generation_scale=g_scale,
        responses=tuple(responses),
        admittance_S_m2=_frozen_complex(y_list),
        impedance_Ohm_m2=_frozen_complex(z_list),
        apparent_capacitance_F_m2=c_arr,
        max_maxwell_spread_A_m2_V=max(r.maxwell_spread_A_m2_V for r in responses),
        max_backward_error=max(r.backward_error for r in responses),
        device_sha256=device.identity,
        scientific_qualification=False,
    )
