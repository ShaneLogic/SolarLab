"""Executable prospective counterexample and falsification witnesses for Stage P08 (P08-01).

Every witness in this module exercises ``solarlab`` at its pre-``P08`` ``N``-class
baseline without modifying ``src/solarlab/**``:

1. ``test_p08_02_nonuniform_series_capacitor_dielectric_witness``:
   Contrasts exact distance-weighted series capacitance ``eps_0 / (d_L/eps_L + d_R/eps_R)``
   against unweighted nodal harmonic ``2*eps_L*eps_R/(eps_L+eps_R)/h`` and arithmetic
   ``0.5*(eps_L+eps_R)/h`` face averages on non-uniform split faces, and exposes the
   ``dx_L / 2`` interface shift on shared-node ``RIGHT_LAYER_WINS`` grids.
2. ``test_p08_02_endpoint_control_volume_and_hole_step_rounding_witness``:
   Contrasts ``INTEGER_ENDPOINT_V1`` full-interval contact weights (``sum(w) > L``)
   against physical half-cell geometries (``sum(w) == L``), and reproduces the
   IEEE-754 binary64 ``hole_separate_increment`` (``BandStepsV1``) rounding mechanism
   underlying ``ScapsFixedStateResultReviewV1`` / ``Final121`` (``V1`` §19.35).
3. ``test_p08_03_unequal_dos_bare_te_equilibrium_and_unit_scaling_witness``:
   Contrasts legacy density-weighted thermionic emission (``A m^-5``, unit-system
   dependent) and single-DOS / non-reciprocal ``A*`` forms against reciprocal
   two-sided ``fermi_richardson_flux`` / ``fermi_dirac_richardson_flux`` detailed
   balance on **bare TE** across ``N_C,L != N_C,R``, plus the pre-``P08-03``
   ``CapabilityError`` gate on ``settings.te_physical_norm=True``.
4. ``test_p08_03_contact_gate_and_falsified_shared_occupancy_refusals``:
   Pins the fixed ``5 meV`` contact thermodynamic gate alongside the permanent
   ``6.025 meV`` ``inconsistent`` ``legacy_manual_v_bi_contact_offset`` label, and
   verifies pre-solve ``CapabilityError`` refusal of falsified
   ``interface_shared_occupancy=True`` across 1D, 2D DC, and 2D ionic compilers.
5. ``test_p08_04_spectral_beer_lambert_photon_budget_and_positive_log_activity_witness``:
   Demonstrates mesh-independent wavelength-resolved Beer-Lambert control-volume
   photon conservation vs monochromatic ``alpha`` mismatch, and verifies that
   log-activity coordinates + physical-inventory ``BackwardEuler1DStep`` maintain
   strictly positive ``n > 0, p > 0`` at ``q * int(G) dx = 260.14 A/m^2`` where an
   unconstrained linear-density step crosses ``p < 0``.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from solarlab.config.resolve_device import resolve_device
from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.device.compile_1d import CapabilityError, compile_device_1d
from solarlab.device.compile_2d import compile_device_2d
from solarlab.device.compile_ionic_2d import compile_ionic_device_2d
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.discretization.face_average import (
    VACUUM_PERMITTIVITY_F_M,
    arithmetic_face_mean,
    harmonic_face_mean,
    series_face_capacitance,
)
from solarlab.discretization.scharfetter_gummel import legacy_sg_current_p
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.experiments.transient import TransientOptions, TransientProtocol, solve_transient
from solarlab.experiments.two_dimensional.inputs import JV2DInput
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument
from solarlab.mesh.geometry import (
    half_endpoint_geometry,
    integer_endpoint_geometry,
    physical_boundary_two_sided_geometry,
    two_sided_trace_geometry,
)
from solarlab.physics.contacts import (
    CONTACT_THERMODYNAMIC_TOLERANCE_EV,
    LEGACY_MANUAL_CONTACT_OFFSET,
    ContactEndpoint,
    ContactThermodynamicError,
    assess_contact_thermodynamics,
    require_contact_thermodynamic_certificate,
)
from solarlab.physics.electrostatics import PoissonFactorization, PoissonOperator1D
from solarlab.physics.generation.beer_lambert import (
    beer_lambert_absorbed_photon_flux,
    beer_lambert_generation,
)
from solarlab.physics.interface_transport import (
    fermi_dirac_richardson_flux,
    fermi_richardson_flux,
    richardson_from_dos,
    scaps_thermionic_flux,
    thermionic_emission_bound_m2_s,
)
from solarlab.physics.statistics import BandStructure, fermi_dirac_half, thermal_voltage
from solarlab.physics.transport import legacy_carrier_face_currents
from solarlab.solvers.nonlinear.problem import SolveBudget
from solarlab.solvers.nonlinear.production import POISSON_CONTACT_BALANCE_V1
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_equilibrium, solve_steady_state
from solarlab.solvers.time import BackwardEuler1DStep

Q = ELEMENTARY_CHARGE_C
EPS_0 = VACUUM_PERMITTIVITY_F_M


def _canonical_default_catalog() -> dict[str, Any]:
    return {
        "schema": "solarlab.default-catalog.v1",
        "material": {
            "eps_r": 25.0,
            "mu_n": 1.9340863535916802e-3,
            "mu_p": 1.9340863535916802e-3,
            "D_ion": 0.0,
            "P_lim": 1.0e28,
            "P0": 0.0,
            "ni": 3.635729012235858e11,
            "tau_n": 1.0e-6,
            "tau_p": 1.0e-6,
            "n1": 3.635729012235858e11,
            "p1": 3.635729012235858e11,
            "B_rad": 1.0e-18,
            "C_n": 1.0e-42,
            "C_p": 1.0e-42,
            "alpha": 2.0e6,
            "N_A": 0.0,
            "N_D": 0.0,
            "chi": 3.9,
            "Eg": 1.6,
            "A_star_n": 0.0,
            "A_star_p": 0.0,
            "v_th": 1.0e5,
            "D_ion_neg": 0.0,
            "P0_neg": 0.0,
            "P_lim_neg": 1.0e28,
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
            "ion_steric_diffusion_only": False,
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
            "P_lim": 1.0e28,
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


def _prepare_pin_device(
    *,
    settings_override: dict[str, Any] | None = None,
    absorber_override: dict[str, Any] | None = None,
) -> PreparedDevice:
    device_doc: dict[str, Any] = {
        "schema_version": "solarlab.device-preparation.v1",
        "id": "p08_witness_pin",
        "name": "P08 witness p-i-n stack",
        "source_format": "canonical",
        "settings": {
            "built_in_potential_mode": "semiconductor_work_function",
            "dos_band_potentials": True,
            "T": 300.0,
            "Phi": 1.6236661838820224e21,
        },
        "layers": [
            {
                "id": "htl",
                "name": "HTL",
                "role": "HTL",
                "thickness": 4.0e-8,
                "parameters": {
                    "chi": 3.6,
                    "Eg": 1.8,
                    "Nc300": 2.0e25,
                    "Nv300": 3.0e25,
                    "eps_r": 3.0,
                    "N_D": 0.0,
                    "N_A": 1.0e24,
                    "ni": 1.8609772721447e10,
                    "n1": 1.8609772721447e10,
                    "p1": 1.8609772721447e10,
                    "mu_n": 3.8681727071833605e-5,
                    "mu_p": 3.8681727071833605e-5,
                    "alpha": 0.0,
                },
            },
            {
                "id": "absorber",
                "name": "absorber",
                "role": "absorber",
                "thickness": 3.5e-7,
                "parameters": {
                    "chi": 3.9,
                    "Eg": 1.6,
                    "Nc300": 1.0e25,
                    "Nv300": 1.0e25,
                    "eps_r": 25.0,
                    "N_D": 0.0,
                    "N_A": 0.0,
                    "ni": 3.635729012235858e11,
                    "n1": 3.635729012235858e11,
                    "p1": 3.635729012235858e11,
                    "mu_n": 1.9340863535916802e-3,
                    "mu_p": 1.9340863535916802e-3,
                    "alpha": 1.5e7,
                },
            },
            {
                "id": "etl",
                "name": "ETL",
                "role": "ETL",
                "thickness": 4.0e-8,
                "parameters": {
                    "chi": 4.0,
                    "Eg": 2.0,
                    "Nc300": 4.0e25,
                    "Nv300": 2.0e25,
                    "eps_r": 9.0,
                    "N_D": 1.0e24,
                    "N_A": 0.0,
                    "ni": 4.4903932431171566e8,
                    "n1": 4.4903932431171566e8,
                    "p1": 4.4903932431171566e8,
                    "mu_n": 3.8681727071833605e-5,
                    "mu_p": 3.8681727071833605e-5,
                    "alpha": 0.0,
                },
            },
        ],
        "electrical_grid": [
            {"layer": "htl", "interval_weight": 6.0, "alpha": 2.5},
            {"layer": "absorber", "interval_weight": 12.0, "alpha": 3.0},
            {"layer": "etl", "interval_weight": 6.0, "alpha": 2.5},
        ],
    }
    if settings_override:
        device_doc["settings"].update(deepcopy(settings_override))
    if absorber_override:
        device_doc["layers"][1]["parameters"].update(deepcopy(absorber_override))

    payload = {"default_catalog": _canonical_default_catalog(), "device": device_doc}
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    source = SourceDocument("inmemory://p08_witness_pin.json", raw)
    catalog_map = {**payload["default_catalog"], "evidence": [[source.id, source.sha256]]}
    return resolve_device(
        DeviceInput.model_validate(device_doc),
        DefaultCatalog.from_mapping(catalog_map),
        ResourceLibrary(),
        sources=(source,),
    )


def test_p08_02_nonuniform_series_capacitor_dielectric_witness() -> None:
    """P08-02a: Distance-weighted series capacitance vs unweighted harmonic/arithmetic & shared-node shift."""
    eps_left, eps_right = 3.0, 25.0
    x_if = 20.0e-9
    total_len = 200.0e-9
    phi_left, phi_right = 0.0, 1.0

    # Exact charge-free 2-layer capacitor: D = -eps_0 * eps(x) * dphi/dx = const
    inv_c_total = x_if / (EPS_0 * eps_left) + (total_len - x_if) / (EPS_0 * eps_right)
    d_exact = -(phi_right - phi_left) / inv_c_total

    def exact_phi(x_arr: np.ndarray) -> np.ndarray:
        return np.where(
            x_arr <= x_if,
            phi_left - (d_exact / (EPS_0 * eps_left)) * x_arr,
            phi_left - (d_exact / (EPS_0 * eps_left)) * x_if - (d_exact / (EPS_0 * eps_right)) * (x_arr - x_if),
        )

    # 1. Two-sided non-uniform grid: interface splits face 1 into d_L = 5 nm, d_R = 60 nm (d_L != d_R).
    x_two_sided = np.array([0.0, 15.0e-9, 80.0e-9, total_len], dtype=float)
    eps_two_sided = np.array([eps_left, eps_left, eps_right, eps_right], dtype=float)
    rho_zero = np.zeros_like(x_two_sided)

    geom_two_sided = two_sided_trace_geometry(x_two_sided, [x_if])
    op_series = PoissonOperator1D.for_geometry(geom_two_sided, eps_two_sided)
    op_nodal_harmonic = PoissonOperator1D.from_harmonic_nodes(x_two_sided, eps_two_sided)
    c_arith = EPS_0 * arithmetic_face_mean(eps_two_sided) / np.diff(x_two_sided)
    op_arithmetic = PoissonOperator1D.from_finite_volume(
        c_arith, op_nodal_harmonic.interior_widths_m, geometry_name="arithmetic_probe_v1"
    )

    phi_series = PoissonFactorization.factor(op_series).solve(rho_zero, phi_left, phi_right)
    phi_harmonic = PoissonFactorization.factor(op_nodal_harmonic).solve(rho_zero, phi_left, phi_right)
    phi_arith = PoissonFactorization.factor(op_arithmetic).solve(rho_zero, phi_left, phi_right)
    phi_ref = exact_phi(x_two_sided)

    # Exact distance-weighted series capacitance reproduces analytical potential & displacement to machine eps.
    assert np.max(np.abs(phi_series - phi_ref)) <= 1.0e-15
    assert np.max(np.abs(op_series.face_displacement(phi_series) - d_exact)) <= 1.0e-15 * abs(d_exact)

    # Unweighted harmonic (assuming d_L == d_R) and arithmetic means incur O(1) interface errors when d_L != d_R.
    c_if_exact = series_face_capacitance(5.0e-9, eps_left, 60.0e-9, eps_right)
    assert op_series.face_capacitance_F_m2[1] == pytest.approx(c_if_exact, rel=1.0e-15)
    assert abs(op_nodal_harmonic.face_capacitance_F_m2[1] - c_if_exact) / c_if_exact > 0.45
    assert abs(op_arithmetic.face_capacitance_F_m2[1] - c_if_exact) / c_if_exact > 0.10
    assert np.max(np.abs(phi_harmonic - phi_ref)) > 0.12
    assert np.max(np.abs(phi_arith - phi_ref)) > 0.01

    # On a midpoint or left-heavy split (d_L >= d_R in the low-eps layer), arithmetic mean overestimates C_f by >160%-300%.
    c_if_left_heavy = series_face_capacitance(60.0e-9, eps_left, 5.0e-9, eps_right)
    c_arith_left_heavy = EPS_0 * float(arithmetic_face_mean([eps_left, eps_right])[0]) / 65.0e-9
    assert abs(c_arith_left_heavy - c_if_left_heavy) / c_if_left_heavy > 3.0

    # Sanity check: in a homogeneous dielectric (eps_left == eps_right) or midpoint split (d_L == d_R),
    # series_face_capacitance and harmonic_face_mean agree identically.
    assert series_face_capacitance(5.0e-9, 12.0, 60.0e-9, 12.0) == pytest.approx(
        EPS_0 * float(harmonic_face_mean([12.0, 12.0])[0]) / 65.0e-9, rel=1.0e-15
    )
    assert series_face_capacitance(10.0e-9, eps_left, 10.0e-9, eps_right) == pytest.approx(
        EPS_0 * float(harmonic_face_mean([eps_left, eps_right])[0]) / 20.0e-9, rel=1.0e-15
    )

    # 2. Shared-node grid (INTEGER_ENDPOINT_V1 / HALF_ENDPOINT_V1) with RIGHT_LAYER_WINS:
    # node 2 sits on x_if = 20 nm and carries eps_right, polluting [15 nm, 20 nm] inside layer 1.
    x_shared = np.array([0.0, 15.0e-9, x_if, 80.0e-9, total_len], dtype=float)
    eps_shared = np.array([eps_left, eps_left, eps_right, eps_right, eps_right], dtype=float)
    for geom in (integer_endpoint_geometry(x_shared, [x_if]), half_endpoint_geometry(x_shared, [x_if])):
        op_shared = PoissonOperator1D.for_geometry(geom, eps_shared)
        phi_shared = PoissonFactorization.factor(op_shared).solve(np.zeros_like(x_shared), phi_left, phi_right)
        # Effective boundary is shifted by dx_L / 2 = 2.5 nm into the low-permittivity left layer,
        # so [17.5 nm, 20 nm] inside layer 1 is treated as eps_right instead of eps_left:
        shifted_inv_c = (x_if - 2.5e-9) / (EPS_0 * eps_left) + (total_len - x_if + 2.5e-9) / (EPS_0 * eps_right)
        phi_if_shifted = ((x_if - 2.5e-9) / (EPS_0 * eps_left) + 2.5e-9 / (EPS_0 * eps_right)) / shifted_inv_c
        assert phi_shared[2] == pytest.approx(phi_if_shifted, rel=1.0e-14)
        assert abs(phi_shared[2] - float(exact_phi(np.array([x_if]))[0])) > 0.025


def test_p08_02_endpoint_control_volume_and_hole_step_rounding_witness() -> None:
    """P08-02b & P08-02c: Full- vs half-cell endpoint measures and hole_separate_increment binary64 rounding."""
    x_shared = np.array([0.0, 6.0e-9, 20.0e-9, 70.0e-9, 120.0e-9], dtype=float)
    x_two_sided = np.array([0.0, 6.0e-9, 34.0e-9, 120.0e-9], dtype=float)
    span_m = float(x_shared[-1] - x_shared[0])

    geom_int = integer_endpoint_geometry(x_shared, [20.0e-9])
    geom_half = half_endpoint_geometry(x_shared, [20.0e-9])
    geom_phys = physical_boundary_two_sided_geometry(x_two_sided, [20.0e-9])

    dx = np.diff(x_shared)
    assert not geom_int.geometric_cells
    assert geom_half.geometric_cells and geom_phys.geometric_cells
    assert geom_int.total_measure_m == pytest.approx(span_m + 0.5 * float(dx[0] + dx[-1]), rel=1.0e-15)
    assert geom_half.total_measure_m == pytest.approx(span_m, rel=1.0e-15)
    assert geom_phys.total_measure_m == pytest.approx(span_m, rel=1.0e-15)

    # Absorbed photon counts on dual faces are identical between INTEGER_ENDPOINT_V1 and HALF_ENDPOINT_V1,
    # so dividing by full-interval w[0] = dx[0] halves nodal G[0] on INTEGER_ENDPOINT_V1.
    alpha = 5.0e6
    phi_inc = 1.5e21
    abs_int = beer_lambert_absorbed_photon_flux(geom_int, alpha, phi_inc)
    abs_half = beer_lambert_absorbed_photon_flux(geom_half, alpha, phi_inc)
    gen_int = beer_lambert_generation(geom_int, alpha, phi_inc)
    gen_half = beer_lambert_generation(geom_half, alpha, phi_inc)
    assert np.array_equal(abs_int, abs_half)
    assert gen_int[0] == pytest.approx(0.5 * gen_half[0], rel=1.0e-15)
    assert gen_int[-1] == pytest.approx(0.5 * gen_half[-1], rel=1.0e-15)
    assert float(np.dot(gen_int, geom_int.cell_measures_m)) == pytest.approx(
        float(np.dot(gen_half, geom_half.cell_measures_m)), rel=1.0e-15
    )

    # IEEE-754 binary64 hole_separate_increment (BandStepsV1) witness (V1 §12 L784 & §19.35 L1621):
    # Adding chi_t + Eg_t (~5.49 eV, ulp = 2^-50 V) before differencing quantizes small intra-layer dphi.
    chi_val, eg_val = 3.94, 1.55
    vt = thermal_voltage(300.0)
    dx_fine = 1.0e-9
    dphi_exact = 1.0e-7 + 2.0**-51  # exact in binary64 near phi=0, sub-ULP once shifted by +5.49 V
    phi_pair = np.array([0.0, dphi_exact], dtype=float)
    bands = BandStructure(
        np.array([chi_val, chi_val]),
        np.array([eg_val, eg_val]),
        np.array([chi_val, chi_val]),
        np.array([eg_val, eg_val]),
        dos_folded=True,
        reference_layer=0,
    )
    p_heavy = np.array([2.5e24, 2.5e24], dtype=float)
    n_min = np.array([1.0e10, 1.0e10], dtype=float)
    d_face = np.array([1.5e-4], dtype=float)

    _, jp_summed = legacy_carrier_face_currents(
        np.array([0.0, dx_fine]), phi_pair, n_min, p_heavy, d_face, d_face, vt, bands
    )
    # Separate-increment potential step: (phi_R - phi_L) + (chi_R - chi_L) + (Eg_R - Eg_L) == dphi_exact.
    dpsi_separate = (phi_pair[1] - phi_pair[0]) + 0.0 + 0.0
    jp_separate = legacy_sg_current_p(
        np.array([0.0, dpsi_separate]), p_heavy, np.array([dx_fine]), d_face, vt
    )
    rounding_gap_a_m2 = abs(float(jp_summed[0] - jp_separate[0]))
    # Quantization error is O(1e-4 A/m^2) — the exact scale of Final121's 4.0107e-4 > 1/3000 A/m^2 screen.
    assert 1.0e-4 < rounding_gap_a_m2 < 2.0e-3


def test_p08_03_unequal_dos_bare_te_equilibrium_and_unit_scaling_witness() -> None:
    """P08-03a: Density-unit scaling & bare-TE equilibrium reciprocity across unequal N_C / m*."""
    temp_k = 300.0
    vt = thermal_voltage(temp_k)
    nc_left, nc_right = 2.0e24, 1.6e25
    nv_left, nv_right = 3.0e24, 9.0e24
    a_star_n_left = richardson_from_dos(nc_left, temp_k)
    a_star_n_right = richardson_from_dos(nc_right, temp_k)
    a_star_p_left = richardson_from_dos(nv_left, temp_k)
    a_star_p_right = richardson_from_dos(nv_right, temp_k)
    assert a_star_n_left != pytest.approx(a_star_n_right)

    # 1. Unit-scaling witness: m^-3 -> cm^-3 simultaneous scaling by 1e-6.
    scale_cm3 = 1.0e-6
    n_l_sample, n_r_sample = 1.0e21, 4.0e21
    gamma_legacy_si, _, _ = thermionic_emission_bound_m2_s(
        n_l_sample, n_r_sample, 0.20, temp_k, a_star_n_left, dos_norm_m3=None
    )
    gamma_legacy_scaled, _, _ = thermionic_emission_bound_m2_s(
        n_l_sample * scale_cm3, n_r_sample * scale_cm3, 0.20, temp_k, a_star_n_left, dos_norm_m3=None
    )
    gamma_norm_si, _, _ = thermionic_emission_bound_m2_s(
        n_l_sample, n_r_sample, 0.20, temp_k, a_star_n_left, dos_norm_m3=nc_left
    )
    gamma_norm_scaled, _, _ = thermionic_emission_bound_m2_s(
        n_l_sample * scale_cm3,
        n_r_sample * scale_cm3,
        0.20,
        temp_k,
        a_star_n_left,
        dos_norm_m3=nc_left * scale_cm3,
    )
    # Legacy density-weighted bound changes by 1e-6 under density-unit scaling and is ~24 orders larger!
    assert gamma_legacy_scaled == pytest.approx(gamma_legacy_si * scale_cm3, rel=1.0e-15)
    assert gamma_norm_scaled == pytest.approx(gamma_norm_si, rel=1.0e-15)
    assert abs(gamma_legacy_si) > 1.0e20 * abs(gamma_norm_si)

    # 2. Bare-TE thermal equilibrium across an unequal-DOS heterojunction (flat E_F => net flux MUST be 0).
    delta_ec_phys = 0.24  # eV (E_C,R - E_C,L at flat potential)
    delta_ev_step = 0.18  # eV hole step
    phi_trace_l, phi_trace_r = 0.0, 0.06
    e_step = delta_ec_phys - (phi_trace_r - phi_trace_l)  # > 0
    h_step = delta_ev_step + (phi_trace_r - phi_trace_l)  # > 0

    # Exact Boltzmann equilibrium densities at flat Fermi level:
    # (n_L / N_C,L) * exp(-e_step / V_T) == (n_R / N_C,R)
    n_eq_l = 5.0e20
    n_eq_r = n_eq_l * (nc_right / nc_left) * math.exp(-e_step / vt)
    p_eq_l = 2.0e20
    p_eq_r = p_eq_l * (nv_right / nv_left) * math.exp(-h_step / vt)
    state_eq = np.array([n_eq_l, p_eq_l, n_eq_r, p_eq_r], dtype=float)

    # Counterexample A: Single shared DOS divisor (dos_norm_m3=nc_left) with true physical step e_step
    # violates equilibrium detailed balance whenever N_C,L != N_C,R!
    gamma_shared_dos_eq, _, _ = thermionic_emission_bound_m2_s(
        n_eq_l, n_eq_r, e_step, temp_k, min(a_star_n_left, a_star_n_right), dos_norm_m3=nc_left
    )
    one_way_ref = (min(a_star_n_left, a_star_n_right) * temp_k**2 / Q) * (n_eq_l / nc_left) * math.exp(-e_step / vt)
    assert abs(gamma_shared_dos_eq) / one_way_ref == pytest.approx(abs(1.0 - nc_right / nc_left), rel=1.0e-14)

    # Counterexample B: Separate side-specific Richardson constants A*_L != A*_R on forward vs reverse legs
    # also violate equilibrium detailed balance!
    non_reciprocal_eq = (temp_k**2 / Q) * (
        a_star_n_left * (n_eq_l / nc_left) * math.exp(-e_step / vt) - a_star_n_right * (n_eq_r / nc_right)
    )
    assert abs(non_reciprocal_eq) / one_way_ref > 1.0

    # Target reciprocal two-sided Bethe/Richardson & SCAPS kernels satisfy exact equilibrium detailed balance.
    boltz_eval = fermi_richardson_flux(
        state_eq,
        phi_trace_left_V=phi_trace_l,
        phi_trace_right_V=phi_trace_r,
        conduction_band_step_eV=delta_ec_phys,
        hole_transport_step_eV=delta_ev_step,
        N_C_left_m3=nc_left,
        N_C_right_m3=nc_right,
        N_V_left_m3=nv_left,
        N_V_right_m3=nv_right,
        richardson_n_A_m2_K2=min(a_star_n_left, a_star_n_right),
        richardson_p_A_m2_K2=min(a_star_p_left, a_star_p_right),
        temperature_K=temp_k,
        thermal_voltage_V=vt,
    )
    assert np.all(np.abs(boltz_eval.cross_flux_m2_s) <= 1.0e-15 * boltz_eval.one_way_scale_m2_s)

    scaps_eval = scaps_thermionic_flux(
        state_eq,
        phi_trace_left_V=phi_trace_l,
        phi_trace_right_V=phi_trace_r,
        conduction_band_step_eV=delta_ec_phys,
        hole_transport_step_eV=delta_ev_step,
        N_C_left_m3=nc_left,
        N_C_right_m3=nc_right,
        N_V_left_m3=nv_left,
        N_V_right_m3=nv_right,
        richardson_n_left_A_m2_K2=a_star_n_left,
        richardson_n_right_A_m2_K2=a_star_n_right,
        richardson_p_left_A_m2_K2=a_star_p_left,
        richardson_p_right_A_m2_K2=a_star_p_right,
        temperature_K=temp_k,
        thermal_voltage_V=vt,
    )
    assert np.all(np.abs(scaps_eval.cross_flux_m2_s) <= 1.0e-15 * scaps_eval.one_way_scale_m2_s)

    # Degenerate Fermi-Dirac reciprocal equilibrium check at flat chemical potential eta_R = eta_L - step/V_T:
    eta_nl, eta_pl = 1.2, 0.8
    eta_nr = eta_nl - e_step / vt
    eta_pr = eta_pl - h_step / vt
    dos_vec = np.array([nc_left, nv_left, nc_right, nv_right], dtype=float)
    occ_vec = np.array(
        [fermi_dirac_half(eta) for eta in (eta_nl, eta_pl, eta_nr, eta_pr)],
        dtype=float,
    )
    state_fd_eq = dos_vec * occ_vec
    fd_eval = fermi_dirac_richardson_flux(
        state_fd_eq,
        phi_trace_left_V=phi_trace_l,
        phi_trace_right_V=phi_trace_r,
        conduction_band_step_eV=delta_ec_phys,
        hole_transport_step_eV=delta_ev_step,
        N_C_left_m3=nc_left,
        N_C_right_m3=nc_right,
        N_V_left_m3=nv_left,
        N_V_right_m3=nv_right,
        richardson_n_A_m2_K2=min(a_star_n_left, a_star_n_right),
        richardson_p_A_m2_K2=min(a_star_p_left, a_star_p_right),
        temperature_K=temp_k,
        thermal_voltage_V=vt,
        base_log_to_dos=np.log(occ_vec),
        u_delta=np.zeros(4, dtype=float),
        electrochemical_steps=(0.0, 0.0),
    )
    assert np.all(np.abs(fd_eval.cross_flux_m2_s) <= 1.0e-15 * fd_eval.one_way_scale_m2_s)

    # Pre-P08-03 guard: settings.te_physical_norm=True fails closed with CapabilityError in compile_device_1d.
    prepared_te = _prepare_pin_device(settings_override={"te_physical_norm": True})
    with pytest.raises(CapabilityError, match=r"settings\.te_physical_norm"):
        compile_device_1d(prepared_te)


def test_p08_03_contact_gate_and_falsified_shared_occupancy_refusals() -> None:
    """P08-03b & P08-03c: 5 meV contact gate + 6.025 meV historical label + shared_occupancy CapabilityError."""
    vt = thermal_voltage(300.0)
    # Self-consistent left (p-type HTL) and right (n-type ETL) reservoirs:
    nc_l, nv_l, chi_l, eg_l = 2.2e24, 1.8e25, 3.50, 2.05
    nc_r, nv_r, chi_r, eg_r = 2.2e24, 1.8e25, 4.00, 3.20
    p_l = 1.0e24
    n_l = nc_l * nv_l * math.exp(-eg_l / vt) / p_l
    n_r = 1.0e24
    p_r = nc_r * nv_r * math.exp(-eg_r / vt) / n_r

    left = ContactEndpoint(n_l, p_l, nc_l, nv_l, chi_l, eg_l)
    right = ContactEndpoint(n_r, p_r, nc_r, nv_r, chi_r, eg_r)
    wf_left = chi_l + eg_l - vt * math.log(nv_l / p_l)
    wf_right = chi_r - vt * math.log(n_r / nc_r)
    v_bi_compatible = wf_left - wf_right

    # Compatible semiconductor_work_function potential passes the fixed 5 meV gate.
    certified = assess_contact_thermodynamics(
        built_in_potential_mode="semiconductor_work_function",
        left=left,
        right=right,
        V_bi_bc=v_bi_compatible,
        thermal_voltage_V=vt,
        semiconductor_potential_V=v_bi_compatible,
        has_band_reference=True,
    )
    assert CONTACT_THERMODYNAMIC_TOLERANCE_EV == 5.0e-3
    assert require_contact_thermodynamic_certificate(certified).certified
    assert certified.historical_label is None
    assert certified.fermi_level_span_eV is not None and certified.fermi_level_span_eV < 1.0e-12

    # Historical 6.024958 meV manual offset stays inconsistent and preserves its label.
    historical_offset_v = 6.024958093130195e-3
    inconsistent = assess_contact_thermodynamics(
        built_in_potential_mode="legacy_manual",
        left=left,
        right=right,
        V_bi_bc=v_bi_compatible + historical_offset_v,
        thermal_voltage_V=vt,
        semiconductor_potential_V=v_bi_compatible,
        has_band_reference=True,
    )
    assert inconsistent.status == "inconsistent" and not inconsistent.certified
    assert inconsistent.historical_label == LEGACY_MANUAL_CONTACT_OFFSET
    assert inconsistent.fermi_level_span_eV == pytest.approx(historical_offset_v, abs=1.0e-12)
    with pytest.raises(ContactThermodynamicError, match=LEGACY_MANUAL_CONTACT_OFFSET):
        require_contact_thermodynamic_certificate(inconsistent)
    with pytest.raises(ValueError, match="cannot exceed"):
        assess_contact_thermodynamics(
            built_in_potential_mode="legacy_manual",
            left=left,
            right=right,
            V_bi_bc=v_bi_compatible + historical_offset_v,
            thermal_voltage_V=vt,
            tolerance_eV=6.1e-3,
        )

    # Scientifically falsified interface_shared_occupancy=True MUST fail closed across 1D and 2D compilers.
    prepared_shared_1d = _prepare_pin_device(settings_override={"interface_shared_occupancy": True})
    with pytest.raises(CapabilityError, match=r"interface_shared_occupancy"):
        compile_device_1d(prepared_shared_1d)

    spatial_dc_2d = JV2DInput(kind="jv_2d", lateral_length="100 nm", Nx=2, Ny_per_layer=4, lateral_bc="neumann")
    with pytest.raises(CapabilityError, match=r"interface_shared_occupancy"):
        compile_device_2d(prepared_shared_1d, spatial_dc_2d)

    prepared_shared_ionic_2d = _prepare_pin_device(
        settings_override={"interface_shared_occupancy": True},
        absorber_override={"P0": 1.0e24, "D_ion": 1.0e-16, "P_lim": 1.0e27},
    )
    spatial_ionic_2d = JV2DInput(
        kind="jv_2d",
        lateral_length="100 nm",
        Nx=2,
        Ny_per_layer=4,
        lateral_bc="neumann",
        ion_dynamics="single_mobile",
    )
    with pytest.raises(CapabilityError, match=r"interface_shared_occupancy"):
        compile_ionic_device_2d(
            prepared_shared_ionic_2d,
            spatial_ionic_2d,
            ion_domain_layer="absorber",
            ion_domain_endpoints=("blocking", "blocking"),
        )


def test_p08_04_spectral_beer_lambert_photon_budget_and_positive_log_activity_witness() -> None:
    """P08-04: Spectral vs scalar Beer-Lambert photon budget & positive log-activity BE under 260.14 A/m^2."""
    thickness_m = 400.0e-9
    wavelengths_m = np.linspace(300.0e-9, 800.0e-9, 201)
    # Dispersive perovskite-like absorption spanning 4e7 m^-1 (UV/blue) down to 2.5e5 m^-1 (band edge).
    norm_wl = (wavelengths_m - 300.0e-9) / 500.0e-9
    alpha_lambda_m1 = 4.0e7 * np.exp(-5.5 * norm_wl)
    # Spectral photon flux normalized so exact single-pass absorbed current q * int(G) dx == 260.14 A/m^2.
    raw_flux = 0.6 + 0.4 * np.sin(math.pi * norm_wl)
    analytic_fraction = 1.0 - np.exp(-alpha_lambda_m1 * thickness_m)
    target_j_ph_a_m2 = 260.14
    target_absorbed_photons = target_j_ph_a_m2 / Q
    flux_scale = target_absorbed_photons / float(np.trapezoid(raw_flux * analytic_fraction, wavelengths_m))
    spectral_flux_m3_s = raw_flux * flux_scale

    def spectral_cv_generation(grid_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        geom = half_endpoint_geometry(grid_m)
        faces = geom.cell_faces_m
        tau_faces = faces[:, None] * alpha_lambda_m1[None, :]
        absorbed_per_wl = np.exp(-tau_faces[:-1]) * (-np.expm1(-np.diff(tau_faces, axis=0)))
        absorbed_cells = np.trapezoid(absorbed_per_wl * spectral_flux_m3_s[None, :], wavelengths_m, axis=1)
        return absorbed_cells / geom.cell_measures_m, geom.cell_measures_m

    # Non-uniform front-clustered coarse (N=15) vs fine (N=91) meshes:
    s_coarse = np.linspace(0.0, 1.0, 15) ** 1.8
    s_fine = np.linspace(0.0, 1.0, 91) ** 1.8
    x_coarse = thickness_m * s_coarse
    x_fine = thickness_m * s_fine

    gen_coarse, w_coarse = spectral_cv_generation(x_coarse)
    gen_fine, w_fine = spectral_cv_generation(x_fine)

    # Exact control-volume spectral Beer-Lambert is mesh-independent to machine precision.
    assert float(np.dot(gen_coarse, w_coarse)) == pytest.approx(target_absorbed_photons, rel=1.0e-14)
    assert float(np.dot(gen_fine, w_fine)) == pytest.approx(target_absorbed_photons, rel=1.0e-14)

    # Contrast against monochromatic Beer-Lambert:
    phi_incident_total = float(np.trapezoid(spectral_flux_m3_s, wavelengths_m))
    alpha_flux_weighted = float(np.trapezoid(spectral_flux_m3_s * alpha_lambda_m1, wavelengths_m)) / phi_incident_total
    geom_fine = half_endpoint_geometry(x_fine)
    gen_scalar_mean = beer_lambert_generation(geom_fine, alpha_flux_weighted, phi_incident_total)
    # 1) Jensen's inequality: scalar mean alpha overpredicts total absorbed photons.
    assert float(np.dot(gen_scalar_mean, w_fine)) > 1.05 * target_absorbed_photons
    # 2) Even if scalar alpha_eff is tuned to match total absorbed photons exactly, it severely underpredicts
    #    front-surface peak generation G[0] compared with dispersive short-wavelength absorption.
    alpha_eff_matched = -math.log(1.0 - target_absorbed_photons / phi_incident_total) / thickness_m
    gen_scalar_matched = beer_lambert_generation(geom_fine, alpha_eff_matched, phi_incident_total)
    assert float(np.dot(gen_scalar_matched, w_fine)) == pytest.approx(target_absorbed_photons, rel=1.0e-14)
    assert gen_fine[0] > 2.5 * gen_scalar_matched[0]

    # Positive-density transient witness at high front generation (q * Phi_abs ~ 260.14 A/m^2):
    # Legacy density-variable Radau hit p < 0 at the absorber front; new-core log-activity coordinates
    # + BackwardEuler1DStep guarantee strict n > 0, p > 0 and exact global particle conservation.
    prepared = _prepare_pin_device()
    device = compile_device_1d(prepared, intervals_per_layer=(6, 12, 6))
    dark_eq = solve_equilibrium(device)
    assert dark_eq.converged

    # Verify illuminated steady-state and a sharp dark-to-1-sun BackwardEuler1DStep both stay strictly positive.
    illum_ss = solve_steady_state(device, voltage_V=0.0, generation_scale=1.0)
    assert illum_ss.converged
    assert np.all(illum_ss.state.electron_density_m3 > 0.0)
    assert np.all(illum_ss.state.hole_density_m3 > 0.0)

    # First-order linear-density update witness under carrier extraction shock (1-sun -> dark reverse bias):
    # du_p < -1 on depleted nodes drives linear tangent p_0 * (1 + du_p) < 0, whereas log-activity stays > 0.
    dc_extract = DC1DProblem(device, voltage_V=-0.4, generation_scale=0.0)
    step_extract = BackwardEuler1DStep(endpoint=dc_extract, previous=illum_ss.state, dt_s=1.0e-6)
    u_extract_0 = step_extract.initial_coordinates(predictor="previous")
    du_extract = np.asarray(
        spsolve(step_extract.jacobian(u_extract_0), -step_extract.residual(u_extract_0)),
        dtype=float,
    )
    du_p_extract = du_extract[2 * device.node_count :]
    linear_tangent_p = illum_ss.state.hole_density_m3 * (1.0 + du_p_extract)
    assert np.min(linear_tangent_p) < 0.0
    unclipped_log_trial = device.state(u_extract_0 + du_extract, voltage_V=-0.4)
    assert np.all(unclipped_log_trial.hole_density_m3 > 0.0)

    # Illuminated bias-ramp transient solved in log-activity coordinates via conservative Backward Euler:
    protocol = TransientProtocol(
        times_s=np.array([0.0, 5.0e-8, 1.0e-7]),
        voltage_V=np.array([0.0, 0.05, 0.10]),
        generation_scale=np.array([1.0, 1.0, 1.0]),
    )
    tr_options = TransientOptions(
        poisson_tolerance=1.0e-11,
        continuity_tolerance_A_m2=1.0e-8,
        contact_log_tolerance=1.0e-11,
        potential_step_V=0.1,
        quasi_fermi_step=2.0,
        newton_budget=SolveBudget(60, 1000, 60, 24),
        max_steps=8,
        wall_s=10.0,
        predictor="previous",
        convergence_policy=POISSON_CONTACT_BALANCE_V1,
    )
    tr_result = solve_transient(
        device,
        illum_ss,
        protocol,
        options=tr_options,
        initial_options=SteadyOptions(),
        method="backward_euler",
    )
    assert tr_result.completed
    for accepted_state in tr_result.states:
        assert np.all(accepted_state.electron_density_m3 > 0.0)
        assert np.all(accepted_state.hole_density_m3 > 0.0)
    for attempt in tr_result.attempts:
        assert attempt.accounting is not None
        pair_total = float(np.sum(np.abs(np.asarray(attempt.accounting["net_pair_source_m2_s"], dtype=float))))
        global_balance = np.asarray(attempt.accounting["global_particle_balance_m2_s"], dtype=float)
        assert float(np.max(np.abs(global_balance))) <= 1.0e-11 * max(pair_total, 1.0)
