"""R1 controls A-D (nu_I, nu_t) preparation, 0- replication, and channel suppression checks (P07-02.1).

Verifies on the public ``solarlab`` interface that:
* A single dark equilibrium ``0-`` anchor state is prepared once and replicated
  identically across controls ``A:(0,0)``, ``B:(1,0)``, ``C:(0,1)``, and ``D:(1,1)``
  without relabeling any historical ``R1CommonState`` certificate.
* When ``nu_I == 0`` (controls ``A`` and ``C``), positive-ion face flux and
  cell storage rate are identically zero while ion background/inventory remain intact.
* When ``nu_t == 0`` (controls ``A`` and ``B``), dynamic interface trap occupancy
  rate ``df_t / dt`` and sheet-charge variation ``sigma_if`` are identically zero.
* Under control ``D`` (``nu_I == 1, nu_t == 1``), both ionic and trap channels
  actively respond while conserving total ion inventory and Maxwell current.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from solarlab.config.device_inputs import DeviceInput
from solarlab.config.resolve_device import resolve_device
from solarlab.contracts import ELEMENTARY_CHARGE_C
from solarlab.device.compile_ionic_1d import CompiledIonicDevice1D, compile_ionic_device_1d
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.ionic_1d import IonicState1D
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument
from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions, solve_ionic_steady_state
from solarlab.solvers.nonlinear.problem import SolveBudget
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions

from solarlab_research.r1.contracts import R1_CONTROLS

__all__ = [
    "R1ControlAudit",
    "R1ControlSuiteReport",
    "R1ZeroMinusAnchor",
    "default_r1_prepared_device",
    "prepare_r1_zero_minus_anchor",
    "verify_r1_controls_abcd",
]

Q = ELEMENTARY_CHARGE_C


def default_r1_prepared_device(
    *,
    nu_I: int = 1,
    ion_diffusivity_m2_s: float = 2.0e-15,
    ion_reference_m3: float = 1.0e24,
    ion_limit_m3: float = 1.0e27,
) -> PreparedDevice:
    """Build a self-contained public 2-layer absorber/ETL heterojunction for R1 verification."""
    if nu_I not in (0, 1):
        raise ValueError(f"nu_I must be 0 or 1; got {nu_I!r}")
    if ion_diffusivity_m2_s <= 0.0 or not 0.0 < ion_reference_m3 < ion_limit_m3:
        raise ValueError("positive unsaturated ion reference and positive base diffusivity required")

    effective_d_ion = float(ion_diffusivity_m2_s) if nu_I == 1 else 0.0
    catalog_doc: dict[str, Any] = {
        "schema": "solarlab.default-catalog.v1",
        "material": {
            "eps_r": 24.0,
            "mu_n": 2.0e-4,
            "mu_p": 2.0e-4,
            "D_ion": 0.0,
            "P_lim": 1.0e27,
            "P0": 0.0,
            "ni": 363572901223.5858,
            "tau_n": 1.0e-6,
            "tau_p": 1.0e-6,
            "n1": 363572901223.5858,
            "p1": 363572901223.5858,
            "B_rad": 1.0e-18,
            "C_n": 1.0e-42,
            "C_p": 1.0e-42,
            "alpha": 2.0e6,
            "N_A": 0.0,
            "N_D": 0.0,
            "chi": 3.9,
            "Eg": 1.6,
            "A_star_n": 1.2e6,
            "A_star_p": 1.2e6,
            "v_th": 1.0e5,
            "D_ion_neg": 0.0,
            "P0_neg": 0.0,
            "P_lim_neg": 1.0e27,
            "Nc300": 1.0e25,
            "Nv300": 1.0e25,
            "carrier_statistics": "maxwell_boltzmann",
            "dopant_ionization_model": "fully_ionized",
            "donor_binding_energy_eV": None,
            "acceptor_binding_energy_eV": None,
            "donor_degeneracy": 2.0,
            "acceptor_degeneracy": 4.0,
            "band_gap_narrowing_model": "off",
            "bgn_reference_energy_eV": 0.0,
            "bgn_reference_density_m3": 1.0e24,
            "bgn_log_shape": 0.5,
            "bgn_conduction_band_fraction": 0.5,
            "mu_T_gamma": -1.5,
            "E_a_ion": 0.0,
            "B_rad_T_gamma": 0.0,
            "varshni_alpha": 0.0,
            "varshni_beta": 0.0,
            "trap_N_t_interface": None,
            "trap_N_t_bulk": None,
            "trap_decay_length": None,
            "trap_profile_shape": "exponential",
            "trap_edge": "both",
            "optical_material": None,
            "n_optical": None,
            "incoherent": False,
            "v_sat_n": 0.0,
            "v_sat_p": 0.0,
            "ct_beta_n": 1.0,
            "ct_beta_p": 1.0,
            "pf_gamma_n": 0.0,
            "pf_gamma_p": 0.0,
            "Eg_back": None,
            "chi_back": None,
            "grading_profile": "linear",
            "grading_direction": "front_to_back",
            "grading_bowing": 0.0,
            "grading_char_length": None,
            "grading_N_mult": 1,
            "N_A_bulk": None,
            "N_D_bulk": None,
            "doping_profile_shape": None,
            "doping_decay_length": None,
            "doping_edge": "front",
        },
        "device": {
            "phi_left": 0.0,
            "V_bi": 0.0,
            "built_in_potential_mode": "semiconductor_work_function",
            "work_function_left_eV": None,
            "work_function_right_eV": None,
            "Phi": 2.0e21,
            "T": 300.0,
            "mode": "fast",
            "interface_plane_projection": False,
            "dos_band_potentials": True,
            "te_physical_norm": False,
            "ion_steric_diffusion_only": True,
            "ion_steric_shared_site": False,
            "flat_band_contacts": False,
            "flat_band_metal_contacts": False,
            "contact_phi_B_eV": 0.0,
            "interface_two_sided": False,
            "interface_shared_occupancy": False,
            "interface_plane_closure": False,
            "interface_plane_generation": False,
            "het_recomb_despike": 0.0,
            "band_grading": False,
            "graded_optics": False,
            "interface_tunneling": False,
            "tunnel_mass_eff": 0.2,
            "jv_solver_policy": "general",
            "interface_charge_closure": "off",
            "interface_charge_rebaseline_acknowledged": False,
        },
        "scaps": {
            "D_ion": 0.0,
            "P_lim": 1.0e27,
            "P0": 0.0,
            "B_rad": 1.0e-18,
            "C_n": 1.0e-42,
            "C_p": 1.0e-42,
            "alpha": 2.0e6,
            "tau_n": 1.0e-6,
            "tau_p": 1.0e-6,
        },
        "interface": {
            "calibration_factor": 1.0,
            "iface_state_calibration_factor": 1.0,
        },
        "constants": {
            "Q": 1.602176634e-19,
            "K_B": 1.380649e-23,
            "T": 300.0,
        },
        "contacts": {
            "S_n_left": None,
            "S_p_left": None,
            "S_n_right": None,
            "S_p_right": None,
        },
        "structural": {
            "grain_boundary_layer_role": "absorber",
            "tandem_light_direction": "top_first",
            "junction_incoherent": False,
        },
        "defect_model": "effective_lifetime",
        "defect_schema_version": None,
        "defect_degeneracy": 1.0,
        "distribution_width_convention": "not_applicable",
    }
    device_doc: dict[str, Any] = {
        "schema_version": "solarlab.device-preparation.v1",
        "id": f"r1_heterojunction_nuI_{nu_I}",
        "name": "R1 1D absorber/ETL heterojunction",
        "source_format": "canonical",
        "settings": {
            "built_in_potential_mode": "semiconductor_work_function",
            "dos_band_potentials": True,
            "ion_steric_diffusion_only": True,
            "T": 300.0,
            "Phi": 2.0e21,
        },
        "layers": [
            {
                "id": "absorber",
                "name": "absorber",
                "role": "absorber",
                "thickness": 3.0e-7,
                "parameters": {
                    "chi": 3.9,
                    "Eg": 1.6,
                    "Nc300": 1.0e25,
                    "Nv300": 1.0e25,
                    "eps_r": 24.0,
                    "N_D": 0.0,
                    "N_A": 1.0e22,
                    "ni": 363572901223.5858,
                    "n1": 363572901223.5858,
                    "p1": 363572901223.5858,
                    "mu_n": 2.0e-4,
                    "mu_p": 2.0e-4,
                    "alpha": 2.0e6,
                    "P0": ion_reference_m3,
                    "P_lim": ion_limit_m3,
                    "D_ion": effective_d_ion,
                    "E_a_ion": 0.0,
                },
            },
            {
                "id": "etl",
                "name": "ETL",
                "role": "ETL",
                "thickness": 8.0e-8,
                "parameters": {
                    "chi": 4.0,
                    "Eg": 2.0,
                    "Nc300": 4.0e25,
                    "Nv300": 2.0e25,
                    "eps_r": 9.0,
                    "N_D": 2.0e23,
                    "N_A": 0.0,
                    "ni": 449039324.31171566,
                    "n1": 449039324.31171566,
                    "p1": 449039324.31171566,
                    "mu_n": 5.0e-5,
                    "mu_p": 5.0e-5,
                    "alpha": 0.0,
                    "P0": 0.0,
                    "D_ion": 0.0,
                },
            },
        ],
        "electrical_grid": [
            {"layer": "absorber", "interval_weight": 12.0, "alpha": 2.5},
            {"layer": "etl", "interval_weight": 8.0, "alpha": 2.5},
        ],
    }
    raw_bytes = json.dumps(
        {"default_catalog": catalog_doc, "device": device_doc},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    source = SourceDocument(f"solarlab_research:r1_control_nuI_{nu_I}", raw_bytes)
    catalog_with_evidence = {**catalog_doc, "evidence": [[source.id, source.sha256]]}
    return resolve_device(
        DeviceInput.model_validate(device_doc),
        DefaultCatalog.from_mapping(catalog_with_evidence),
        ResourceLibrary(),
        sources=(source,),
    )


@dataclass(frozen=True, slots=True)
class R1ZeroMinusAnchor:
    """Common dark equilibrium 0- anchor state shared across controls A-D."""

    mobile_device: CompiledIonicDevice1D
    immobile_device: CompiledIonicDevice1D
    mobile_state: IonicState1D
    immobile_state: IonicState1D
    interface_node: int
    equilibrium_trap_occupancy: float
    trap_density_m2: float
    capture_n_m_s: float
    capture_p_m_s: float
    n1_trap_m3: float
    p1_trap_m3: float


def prepare_r1_zero_minus_anchor(
    *,
    intervals_per_layer: tuple[int, int] = (6, 4),
    trap_density_m2: float = 2.0e15,
    capture_n_m_s: float = 5.0e-14,
    capture_p_m_s: float = 5.0e-14,
    equilibrium_occupancy_target: float = 0.35,
) -> R1ZeroMinusAnchor:
    """Solve the common dark 0- ionic steady state and replicate it onto immobile nu_I=0.

    Note: ``capture_n_m_s`` / ``capture_p_m_s`` are 3D SRH capture rate constants
    ``sigma * v_th`` [m^3/s] such that ``N_t * c_n = 100 m/s`` at ``N_t = 2e15 m^-2``.
    """
    prep_mobile = default_r1_prepared_device(nu_I=1)
    prep_immobile = default_r1_prepared_device(nu_I=0)

    dev_mobile = compile_ionic_device_1d(
        prep_mobile,
        intervals_per_layer=intervals_per_layer,
        ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"),
    )
    steady_opts = IonicSteadyOptions(
        electronic=SteadyOptions(
            poisson_tolerance=1.0e-11,
            continuity_tolerance_A_m2=1.0e-9,
            current_spread_tolerance_A_m2=1.0e-9,
        ),
        inventory_relative_tolerance=1.0e-12,
        ion_balance_tolerance_m2_s=1.0e6,
        ion_flux_tolerance_m2_s=1.0e6,
        ion_logit_step=1.0,
        inventory_row_node=0,
        budget=SolveBudget(100, 5000, 1000, 40),
        wall_s=30.0,
    )
    res_mobile = solve_ionic_steady_state(
        dev_mobile,
        voltage_V=0.0,
        generation_scale=0.0,
        options=steady_opts,
    )
    if not res_mobile.accepted or res_mobile.state is None:
        raise RuntimeError(f"R1 0- mobile steady state failed: {res_mobile.termination} ({res_mobile.error})")

    dev_immobile = compile_ionic_device_1d(
        prep_immobile,
        intervals_per_layer=intervals_per_layer,
        ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"),
    )
    # Replicate the exact 0- physical state vector onto the nu_I=0 device (no re-solve drift).
    immobile_state = dev_immobile.state(res_mobile.state.u, voltage_V=0.0)

    if_node = int(dev_mobile.ion_node_indices[-1])
    n0 = float(res_mobile.state.electronic_state.electron_density_m3[if_node])
    p0 = float(res_mobile.state.electronic_state.hole_density_m3[if_node])
    if not 0.0 < equilibrium_occupancy_target < 1.0:
        raise ValueError("equilibrium_occupancy_target must lie strictly in (0, 1)")
    f_t0 = float(equilibrium_occupancy_target)
    # Place the interface trap level near the dark equilibrium Fermi level at if_node
    # with exact SRH detailed balance (n1_t * p1_t == n0 * p0 and rn_0 == rp_0 == 0).
    n1_t = n0 * (1.0 - f_t0) / f_t0
    p1_t = p0 * f_t0 / (1.0 - f_t0)

    return R1ZeroMinusAnchor(
        mobile_device=dev_mobile,
        immobile_device=dev_immobile,
        mobile_state=res_mobile.state,
        immobile_state=immobile_state,
        interface_node=if_node,
        equilibrium_trap_occupancy=f_t0,
        trap_density_m2=trap_density_m2,
        capture_n_m_s=capture_n_m_s,
        capture_p_m_s=capture_p_m_s,
        n1_trap_m3=n1_t,
        p1_trap_m3=p1_t,
    )


@dataclass(frozen=True, slots=True)
class R1ControlAudit:
    """Single-control verification evidence for 0- replication and suppressed channels."""

    control: str
    nu_I: int
    nu_t: int
    zero_minus_replicated: bool
    max_ion_flux_m2_s: float
    max_ion_rate_m2_s: float
    ion_inventory_relative_drift: float
    trap_occupancy_rate_s: float
    sheet_charge_C_m2: float
    passed: bool


@dataclass(frozen=True, slots=True)
class R1ControlSuiteReport:
    """Combined verification report across controls A, B, C, and D."""

    passed: bool
    zero_minus_potential_sha256: str
    zero_minus_ion_density_sha256: str
    controls: Mapping[str, R1ControlAudit]


def verify_r1_controls_abcd(
    *,
    intervals_per_layer: tuple[int, int] = (6, 4),
    dt_s: float = 1.0e-4,
    voltage_step_V: float = 5.0e-3,
) -> R1ControlSuiteReport:
    """Run one coupled Backward-Euler step under controls A, B, C, and D from a common 0- anchor."""
    from solarlab_research.r1.short_window import solve_coupled_r1_step

    anchor = prepare_r1_zero_minus_anchor(intervals_per_layer=intervals_per_layer)
    ref_phi = anchor.mobile_state.electronic_state.potential_V
    ref_ion = anchor.mobile_state.positive_ion_density_m3
    import hashlib

    phi_sha = hashlib.sha256(np.ascontiguousarray(ref_phi, dtype="<f8").tobytes()).hexdigest()
    ion_sha = hashlib.sha256(np.ascontiguousarray(ref_ion, dtype="<f8").tobytes()).hexdigest()

    audits: dict[str, R1ControlAudit] = {}
    for ctrl, switches in R1_CONTROLS.items():
        nu_i = int(switches["nu_I"])
        nu_t = int(switches["nu_t"])
        dev = anchor.mobile_device if nu_i == 1 else anchor.immobile_device
        init_state = anchor.mobile_state if nu_i == 1 else anchor.immobile_state

        replicated = bool(
            np.array_equal(init_state.electronic_state.potential_V, ref_phi)
            and np.array_equal(init_state.positive_ion_density_m3, ref_ion)
        )

        step_res = solve_coupled_r1_step(
            device=dev,
            previous_ionic=init_state,
            previous_trap_occupancy=anchor.equilibrium_trap_occupancy,
            anchor=anchor,
            voltage_V=voltage_step_V,
            dt_s=dt_s,
            nu_t=nu_t,
            mode="compensated",
        )
        max_ion_flux = float(np.max(np.abs(step_res.ion_flux_m2_s)))
        max_ion_rate = float(np.max(np.abs(step_res.ion_rate_m2_s)))
        inv_0 = anchor.mobile_state.domain_inventory_m2
        inv_1 = step_res.ionic_state.domain_inventory_m2
        inv_drift = abs(inv_1 - inv_0) / inv_0
        trap_rate = abs(step_res.trap_occupancy - anchor.equilibrium_trap_occupancy) / dt_s
        sheet_q = step_res.sheet_charge_C_m2

        ion_ok = (max_ion_flux == 0.0 and max_ion_rate == 0.0) if nu_i == 0 else (max_ion_flux > 0.0 and max_ion_rate > 0.0)
        trap_ok = (trap_rate == 0.0 and sheet_q == 0.0) if nu_t == 0 else (trap_rate > 0.0 and abs(sheet_q) > 0.0)
        ok = bool(replicated and step_res.accepted and ion_ok and trap_ok and inv_drift <= 1.0e-12)
        audits[ctrl] = R1ControlAudit(
            control=ctrl,
            nu_I=nu_i,
            nu_t=nu_t,
            zero_minus_replicated=replicated,
            max_ion_flux_m2_s=max_ion_flux,
            max_ion_rate_m2_s=max_ion_rate,
            ion_inventory_relative_drift=inv_drift,
            trap_occupancy_rate_s=trap_rate,
            sheet_charge_C_m2=sheet_q,
            passed=ok,
        )

    all_passed = all(a.passed for a in audits.values())
    return R1ControlSuiteReport(
        passed=all_passed,
        zero_minus_potential_sha256=phi_sha,
        zero_minus_ion_density_sha256=ion_sha,
        controls=MappingProxyType(audits),
    )
