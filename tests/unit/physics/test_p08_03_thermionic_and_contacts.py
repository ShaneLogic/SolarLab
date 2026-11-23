"""Unit tests for P08-03 reciprocal unequal-DOS thermionic emission and 5.0 meV contact gate.

Verifies:
1. Bare thermionic emission detailed balance (``|J_TE| <= 1e-15 * J_one_way``) at thermal
   equilibrium across unequal DOS ``N_{C,L}/N_{C,R} in [1e-2, 1e2]`` and unequal effective
   masses ``A*_L != A*_R`` under both Maxwell-Boltzmann and Fermi-Dirac statistics, without
   Scharfetter-Gummel masking, contrasted against single-DOS, non-reciprocal ``A*``, and
   DOS-folded barrier counterexamples.
2. Strict sign invariant (zero spurious non-equilibrium roots), sub-zeptoampere cancellation-free
   ``expm1`` drive accuracy, exact ``left <-> right`` swap reciprocity, concentration/length
   unit-rescaling invariance, transmission scaling, SCAPS effective-velocity distinctness,
   and 0-ULP legacy parity.
3. Fourth-order central finite-difference verification (``<= 1e-9``) of analytical Jacobians
   wrt ``(phi_L, phi_R, c_L, c_R)`` and ``(ln c_L, ln c_R)`` for both bare reciprocal TE and
   composite SG + reciprocal TE face evaluations (including at exact ``m == 0`` equilibrium).
4. Contact thermodynamic ``5.0 meV`` gate invariants, parameter-free vs fitted contact mode
   classification, and permanent preservation of the historical ``6.024958093130e-3 eV``
   ``scaps_mirror_v2`` ``legacy_manual_v_bi_contact_offset`` ``inconsistent`` record.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from solarlab.config.resolve_device import resolve_device
from solarlab.config.scaps_input import import_scaps_device
from solarlab.materials.packaged import (
    load_packaged_default_catalog,
    load_packaged_resource_library,
)
from solarlab.materials.source import SourceDocument
from solarlab.physics.contacts import (
    CONTACT_ALIGNMENT_CLASSES,
    CONTACT_THERMODYNAMIC_TOLERANCE_EV,
    EXTERNAL_METAL_CONTACT_ALIGNMENT,
    FITTED_MANUAL_CONTACT_ALIGNMENT,
    LEGACY_MANUAL_CONTACT_OFFSET,
    PARAMETER_FREE_CONTACT_ALIGNMENT,
    BuiltInPotentialSources,
    ContactEndpoint,
    ContactMaterial,
    ContactThermodynamicError,
    LegacyBandContact,
    assess_contact_thermodynamics,
    certify_parameter_free_contact_equilibrium,
    classify_contact_alignment_mode,
    doping_equilibrium_densities,
    require_contact_thermodynamic_certificate,
)
from solarlab.physics.interface_transport.thermionic import (
    LEGACY_UNNORMALIZED_THERMIONIC_V1,
    RECIPROCAL_PHYSICAL_THERMIONIC_V2,
    SCAPS_EFFECTIVE_VELOCITY_THERMIONIC_V1,
    THERMIONIC_MODEL_VERSIONS,
    ReciprocalThermionicCompositeEvaluation,
    ThermionicCarrierKind,
    ThermionicCombinationMode,
    ThermionicStatisticsKind,
    combine_sg_and_reciprocal_thermionic_flux,
    physical_barrier_steps_eV,
    physical_carrier_barrier_step_eV,
    reciprocal_richardson,
    reciprocal_thermionic_flux_m2_s,
    richardson_from_dos,
    thermionic_emission_bound_m2_s,
)
from solarlab.physics.statistics.band_potentials import fold_dos_into_transport_bands
from solarlab.physics.statistics.fermi_dirac import fermi_dirac_half
from solarlab.physics.statistics.thermal import thermal_voltage
from solarlab.physics.transport.electrochemical_flux import stable_sg_particle_flux

SCAPS_MIRROR_PRESET = (
    Path(__file__).resolve().parents[3]
    / "src/solarlab/materials/data/presets/scaps_mirror_v2.yaml"
)


@pytest.fixture(scope="module", autouse=True)
def single_thread_blas() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


def _fourth_order_fd(fn: Callable[[float], float], x0: float, h: float) -> float:
    return (
        -fn(x0 + 2.0 * h) + 8.0 * fn(x0 + h) - 8.0 * fn(x0 - h) + fn(x0 - 2.0 * h)
    ) / (12.0 * h)


@pytest.mark.parametrize("dos_ratio", [1.0e-2, 1.0e-1, 0.4, 1.0, 2.5, 10.0, 1.0e2])
@pytest.mark.parametrize("carrier", ["electron", "hole"])
@pytest.mark.parametrize("band_offset_eV", [-0.24, -0.07, 0.0, 0.11, 0.29])
def test_bare_reciprocal_te_equilibrium_across_unequal_dos_and_masses_boltzmann(
    dos_ratio: float,
    carrier: ThermionicCarrierKind,
    band_offset_eV: float,
) -> None:
    temp = 300.0
    vt = thermal_voltage(temp)
    dos_r = 4.0e24
    dos_l = dos_ratio * dos_r
    a_star_l = 1.85e5
    a_star_r = 7.40e5
    phi_l = 0.08
    phi_r = 0.21

    barrier_step, _, _ = physical_carrier_barrier_step_eV(
        carrier, phi_left_V=phi_l, phi_right_V=phi_r, band_offset_eV=band_offset_eV
    )
    c_l = 1.2e20
    c_r = c_l * (dos_r / dos_l) * math.exp(-barrier_step / vt)

    # 1. Raw binary64 density equilibrium (without passing electrochemical_step):
    res_raw = reciprocal_thermionic_flux_m2_s(
        c_l,
        c_r,
        carrier=carrier,
        phi_left_V=phi_l,
        phi_right_V=phi_r,
        band_offset_eV=band_offset_eV,
        temperature_K=temp,
        a_star_left_A_m2_K2=a_star_l,
        a_star_right_A_m2_K2=a_star_r,
        dos_left_m3=dos_l,
        dos_right_m3=dos_r,
        transmission=0.85,
        statistics="maxwell_boltzmann",
        model_version=RECIPROCAL_PHYSICAL_THERMIONIC_V2,
    )
    assert res_raw.one_way_scale_m2_s > 0.0
    assert abs(res_raw.flux_m2_s) <= 1.0e-15 * res_raw.one_way_scale_m2_s
    assert abs(res_raw.electrochemical_step) <= 1.0e-15
    assert res_raw.effective_richardson_A_m2_K2 == min(a_star_l, a_star_r)

    # 2. Exact quasi-Fermi equilibrium (electrochemical_step == 0.0): exact 0.0 flux.
    res_exact = reciprocal_thermionic_flux_m2_s(
        c_l,
        c_r,
        carrier=carrier,
        phi_left_V=phi_l,
        phi_right_V=phi_r,
        band_offset_eV=band_offset_eV,
        temperature_K=temp,
        a_star_left_A_m2_K2=a_star_l,
        a_star_right_A_m2_K2=a_star_r,
        dos_left_m3=dos_l,
        dos_right_m3=dos_r,
        transmission=0.85,
        statistics="maxwell_boltzmann",
        model_version=RECIPROCAL_PHYSICAL_THERMIONIC_V2,
        electrochemical_step=0.0,
    )
    assert res_exact.flux_m2_s == 0.0
    assert res_exact.current_A_m2 == 0.0


@pytest.mark.parametrize("dos_ratio", [1.0e-2, 0.25, 1.0, 4.0, 1.0e2])
@pytest.mark.parametrize("carrier", ["electron", "hole"])
def test_bare_reciprocal_te_equilibrium_fermi_dirac(
    dos_ratio: float,
    carrier: ThermionicCarrierKind,
) -> None:
    temp = 300.0
    vt = thermal_voltage(temp)
    dos_r = 5.0e24
    dos_l = dos_ratio * dos_r
    a_star_l = 2.1e5
    a_star_r = 6.3e5
    phi_l, phi_r = -0.05, 0.09
    band_offset = 0.22

    barrier_step, _, _ = physical_carrier_barrier_step_eV(
        carrier, phi_left_V=phi_l, phi_right_V=phi_r, band_offset_eV=band_offset
    )
    # Degenerate left side (eta_l = +1.75) in equilibrium with right side across barrier_step:
    eta_l = 1.75
    eta_r = eta_l - barrier_step / vt
    c_l = dos_l * fermi_dirac_half(eta_l)
    c_r = dos_r * fermi_dirac_half(eta_r)

    res_fd_raw = reciprocal_thermionic_flux_m2_s(
        c_l,
        c_r,
        carrier=carrier,
        phi_left_V=phi_l,
        phi_right_V=phi_r,
        band_offset_eV=band_offset,
        temperature_K=temp,
        a_star_left_A_m2_K2=a_star_l,
        a_star_right_A_m2_K2=a_star_r,
        dos_left_m3=dos_l,
        dos_right_m3=dos_r,
        statistics="fermi_dirac",
        model_version=RECIPROCAL_PHYSICAL_THERMIONIC_V2,
    )
    assert res_fd_raw.one_way_scale_m2_s > 0.0
    assert abs(res_fd_raw.flux_m2_s) <= 5.0e-15 * res_fd_raw.one_way_scale_m2_s

    res_fd_exact = reciprocal_thermionic_flux_m2_s(
        c_l,
        c_r,
        carrier=carrier,
        phi_left_V=phi_l,
        phi_right_V=phi_r,
        band_offset_eV=band_offset,
        temperature_K=temp,
        a_star_left_A_m2_K2=a_star_l,
        a_star_right_A_m2_K2=a_star_r,
        dos_left_m3=dos_l,
        dos_right_m3=dos_r,
        statistics="fermi_dirac",
        model_version=RECIPROCAL_PHYSICAL_THERMIONIC_V2,
        electrochemical_step=0.0,
    )
    assert res_fd_exact.flux_m2_s == 0.0


def test_counterexamples_single_dos_nonreciprocal_mass_and_folded_barrier_break_equilibrium() -> None:
    """Contrast RECIPROCAL_PHYSICAL_THERMIONIC_V2 against the three legacy failure modes."""
    temp = 300.0
    vt = thermal_voltage(temp)
    chi_l, chi_r = 3.70, 3.92
    eg_l, eg_r = 1.75, 1.55
    nc_l, nc_r = 2.0e25, 2.5e24  # 8x unequal DOS
    nv_l, nv_r = 3.0e25, 6.0e24
    a_l, a_r = 2.0e5, 6.5e5
    phi_l, phi_r = 0.0, 0.14

    e_step_phys, _ = physical_barrier_steps_eV(
        phi_left_V=phi_l,
        phi_right_V=phi_r,
        chi_phys_left_eV=chi_l,
        chi_phys_right_eV=chi_r,
        eg_phys_left_eV=eg_l,
        eg_phys_right_eV=eg_r,
    )
    n_l = 1.0e19
    n_r = n_l * (nc_r / nc_l) * math.exp(-e_step_phys / vt)

    # Physical V2 gives machine-zero equilibrium flux:
    v2 = reciprocal_thermionic_flux_m2_s(
        n_l,
        n_r,
        carrier="electron",
        phi_left_V=phi_l,
        phi_right_V=phi_r,
        band_offset_eV=chi_l - chi_r,
        temperature_K=temp,
        a_star_left_A_m2_K2=a_l,
        a_star_right_A_m2_K2=a_r,
        dos_left_m3=nc_l,
        dos_right_m3=nc_r,
    )
    assert abs(v2.flux_m2_s) <= 1.0e-15 * v2.one_way_scale_m2_s

    # Counterexample A: single geometric-mean DOS divisor (legacy te_physical_norm):
    geom_dos = math.sqrt(nc_l * nc_r)
    single_dos = reciprocal_thermionic_flux_m2_s(
        n_l,
        n_r,
        carrier="electron",
        phi_left_V=phi_l,
        phi_right_V=phi_r,
        band_offset_eV=chi_l - chi_r,
        temperature_K=temp,
        a_star_left_A_m2_K2=a_l,
        dos_left_m3=geom_dos,
        model_version=LEGACY_UNNORMALIZED_THERMIONIC_V1,
    )
    assert abs(single_dos.flux_m2_s) > 0.6 * single_dos.one_way_scale_m2_s

    # Counterexample B: non-reciprocal per-side Richardson weighting A*_L != A*_R:
    b_l = max(e_step_phys, 0.0) / vt
    b_r = max(-e_step_phys, 0.0) / vt
    leg_l = a_l * (n_l / nc_l) * math.exp(-b_l)
    leg_r = a_r * (n_r / nc_r) * math.exp(-b_r)
    assert abs(leg_l - leg_r) / max(leg_l, leg_r) == pytest.approx(
        abs(a_l - a_r) / max(a_l, a_r), rel=1e-14
    )

    # Counterexample C: feeding DOS-folded SG transport affinities into the TE barrier:
    bands = fold_dos_into_transport_bands(
        np.array([chi_l, chi_r]),
        np.array([eg_l, eg_r]),
        [np.array([True, False]), np.array([False, True])],
        [(nc_l, nv_l), (nc_r, nv_r)],
        reference_layer=0,
        thermal_voltage_V=vt,
    )
    folded_offset = float(bands.transport_electron_affinity_eV[0] - bands.transport_electron_affinity_eV[1])
    folded_v2 = reciprocal_thermionic_flux_m2_s(
        n_l,
        n_r,
        carrier="electron",
        phi_left_V=phi_l,
        phi_right_V=phi_r,
        band_offset_eV=folded_offset,
        temperature_K=temp,
        a_star_left_A_m2_K2=a_l,
        a_star_right_A_m2_K2=a_r,
        dos_left_m3=nc_l,
        dos_right_m3=nc_r,
    )
    assert abs(folded_v2.flux_m2_s) > 0.8 * folded_v2.one_way_scale_m2_s


def test_zero_spurious_roots_weak_drive_expm1_reciprocity_unit_invariance_and_model_versions() -> None:
    assert set(THERMIONIC_MODEL_VERSIONS) == {
        LEGACY_UNNORMALIZED_THERMIONIC_V1,
        RECIPROCAL_PHYSICAL_THERMIONIC_V2,
        SCAPS_EFFECTIVE_VELOCITY_THERMIONIC_V1,
    }
    assert reciprocal_richardson(0.0, 3.0e5) == 0.0
    assert reciprocal_richardson(5.0e5, 2.2e5) == 2.2e5

    temp = 300.0
    vt = thermal_voltage(temp)
    chi_l, chi_r = 3.85, 4.02
    eg_l, eg_r = 1.62, 1.95
    nc_l, nc_r = 8.0e24, 3.2e25
    nv_l, nv_r = 1.2e25, 4.5e24
    a_l, a_r = richardson_from_dos(nc_l, temp), richardson_from_dos(nc_r, temp)
    phi_l, phi_r = 0.05, 0.24

    bands = fold_dos_into_transport_bands(
        np.array([chi_l, chi_r]),
        np.array([eg_l, eg_r]),
        [np.array([True, False]), np.array([False, True])],
        [(nc_l, nv_l), (nc_r, nv_r)],
        reference_layer=0,
        thermal_voltage_V=vt,
    )
    e_step_phys, h_step_phys = physical_barrier_steps_eV(
        phi_left_V=phi_l,
        phi_right_V=phi_r,
        chi_phys_left_eV=chi_l,
        chi_phys_right_eV=chi_r,
        eg_phys_left_eV=eg_l,
        eg_phys_right_eV=eg_r,
    )

    # 1. Strict sign invariant across 80 orders of electrochemical drive (no spurious roots)
    # and exact sign agreement with DOS-folded Scharfetter-Gummel flux:
    n_l_base = 3.0e19
    n_r_eq = n_l_base * (nc_r / nc_l) * math.exp(-e_step_phys / vt)
    sg_energy_step = float(
        -((phi_r + bands.transport_electron_affinity_eV[1]) - (phi_l + bands.transport_electron_affinity_eV[0]))
        / vt
    )
    drives = [-80.0, -12.0, -1.5, -1.0e-3, -1.0e-12, -1.0e-25, 0.0, 1.0e-25, 1.0e-12, 1.0e-3, 1.5, 12.0, 80.0]
    for m_drive in drives:
        n_r_drive = n_r_eq * math.exp(max(-300.0, min(300.0, m_drive)))
        te_eval = reciprocal_thermionic_flux_m2_s(
            n_l_base,
            n_r_drive,
            carrier="electron",
            phi_left_V=phi_l,
            phi_right_V=phi_r,
            band_offset_eV=chi_l - chi_r,
            temperature_K=temp,
            a_star_left_A_m2_K2=a_l,
            a_star_right_A_m2_K2=a_r,
            dos_left_m3=nc_l,
            dos_right_m3=nc_r,
            electrochemical_step=m_drive,
        )
        sg_eval = stable_sg_particle_flux(
            n_l_base, n_r_drive, sg_energy_step, 1.0e3, electrochemical_step=m_drive
        )
        if m_drive == 0.0:
            assert te_eval.flux_m2_s == 0.0
            assert float(sg_eval.flux) == 0.0
        else:
            expected_sign = -1.0 if m_drive > 0.0 else 1.0
            assert math.copysign(1.0, te_eval.flux_m2_s) == expected_sign
            assert math.copysign(1.0, float(sg_eval.flux)) == expected_sign
            if abs(m_drive) <= 1.0e-12:
                # Sub-zeptoampere linear accuracy without two-term cancellation:
                linear_exact = -te_eval.d_flux_d_log_c_left_m2_s * m_drive
                assert te_eval.flux_m2_s == pytest.approx(linear_exact, rel=1.0e-11)

    # 2. Exact 0-ULP directional reciprocity under L <-> R swap (electrons & holes, MB & FD):
    carrier_cases: tuple[tuple[ThermionicCarrierKind, float, float, float], ...] = (
        ("electron", chi_l - chi_r, nc_l, nc_r),
        ("hole", (chi_r + eg_r) - (chi_l + eg_l), nv_l, nv_r),
    )
    stats_cases: tuple[ThermionicStatisticsKind, ...] = ("maxwell_boltzmann", "fermi_dirac")
    for carrier, offset, d_l, d_r in carrier_cases:
        for stats in stats_cases:
            c_left, c_right = 4.2e23, 1.9e22
            fwd = reciprocal_thermionic_flux_m2_s(
                c_left,
                c_right,
                carrier=carrier,
                phi_left_V=phi_l,
                phi_right_V=phi_r,
                band_offset_eV=offset,
                temperature_K=temp,
                a_star_left_A_m2_K2=a_l,
                a_star_right_A_m2_K2=a_r,
                dos_left_m3=d_l,
                dos_right_m3=d_r,
                transmission=0.73,
                statistics=stats,
            )
            rev = reciprocal_thermionic_flux_m2_s(
                c_right,
                c_left,
                carrier=carrier,
                phi_left_V=phi_r,
                phi_right_V=phi_l,
                band_offset_eV=-offset,
                temperature_K=temp,
                a_star_left_A_m2_K2=a_r,
                a_star_right_A_m2_K2=a_l,
                dos_left_m3=d_r,
                dos_right_m3=d_l,
                transmission=0.73,
                statistics=stats,
            )
            assert rev.barrier_step_eV == -fwd.barrier_step_eV
            assert rev.effective_richardson_A_m2_K2 == fwd.effective_richardson_A_m2_K2
            if stats == "maxwell_boltzmann":
                assert rev.flux_m2_s == -fwd.flux_m2_s
                assert rev.electrochemical_step == -fwd.electrochemical_step
            else:
                assert rev.flux_m2_s == pytest.approx(-fwd.flux_m2_s, rel=1.0e-15)

    # 3. Concentration/length unit-rescaling invariance (m^-3 <-> cm^-3, scale = 1e-6):
    scale_cm3 = 1.0e-6
    v2_si = reciprocal_thermionic_flux_m2_s(
        2.5e20,
        9.0e18,
        carrier="electron",
        phi_left_V=phi_l,
        phi_right_V=phi_r,
        band_offset_eV=chi_l - chi_r,
        temperature_K=temp,
        a_star_left_A_m2_K2=a_l,
        a_star_right_A_m2_K2=a_r,
        dos_left_m3=nc_l,
        dos_right_m3=nc_r,
    )
    v2_cgs = reciprocal_thermionic_flux_m2_s(
        2.5e20 * scale_cm3,
        9.0e18 * scale_cm3,
        carrier="electron",
        phi_left_V=phi_l,
        phi_right_V=phi_r,
        band_offset_eV=chi_l - chi_r,
        temperature_K=temp,
        a_star_left_A_m2_K2=a_l,
        a_star_right_A_m2_K2=a_r,
        dos_left_m3=nc_l * scale_cm3,
        dos_right_m3=nc_r * scale_cm3,
    )
    assert v2_cgs.flux_m2_s == pytest.approx(v2_si.flux_m2_s, rel=1.0e-15)
    assert v2_cgs.current_A_m2 == pytest.approx(v2_si.current_A_m2, rel=1.0e-15)

    # Legacy unnormalized V1 scales spuriously by 1e-6 and matches thermionic_emission_bound_m2_s to 0 ULP:
    leg_si = reciprocal_thermionic_flux_m2_s(
        2.5e20,
        9.0e18,
        carrier="electron",
        phi_left_V=0.0,
        phi_right_V=0.0,
        band_offset_eV=chi_l - chi_r,
        temperature_K=temp,
        a_star_left_A_m2_K2=a_l,
        model_version=LEGACY_UNNORMALIZED_THERMIONIC_V1,
    )
    leg_cgs = reciprocal_thermionic_flux_m2_s(
        2.5e20 * scale_cm3,
        9.0e18 * scale_cm3,
        carrier="electron",
        phi_left_V=0.0,
        phi_right_V=0.0,
        band_offset_eV=chi_l - chi_r,
        temperature_K=temp,
        a_star_left_A_m2_K2=a_l,
        model_version=LEGACY_UNNORMALIZED_THERMIONIC_V1,
    )
    ref_bound, ref_dl, ref_dr = thermionic_emission_bound_m2_s(
        2.5e20, 9.0e18, chi_l - chi_r, temp, a_l, dos_norm_m3=None
    )
    assert leg_si.flux_m2_s == ref_bound
    assert leg_si.d_flux_d_c_left_m_s == ref_dl
    assert leg_si.d_flux_d_c_right_m_s == ref_dr
    assert leg_cgs.flux_m2_s == pytest.approx(scale_cm3 * leg_si.flux_m2_s, rel=1.0e-15)

    # 4. Transmission factor scaling & SCAPS effective-velocity distinctness:
    v2_zero_t = reciprocal_thermionic_flux_m2_s(
        2.5e20,
        9.0e18,
        carrier="electron",
        phi_left_V=phi_l,
        phi_right_V=phi_r,
        band_offset_eV=chi_l - chi_r,
        temperature_K=temp,
        a_star_left_A_m2_K2=a_l,
        a_star_right_A_m2_K2=a_r,
        dos_left_m3=nc_l,
        dos_right_m3=nc_r,
        transmission=0.0,
    )
    v2_quar_t = reciprocal_thermionic_flux_m2_s(
        2.5e20,
        9.0e18,
        carrier="electron",
        phi_left_V=phi_l,
        phi_right_V=phi_r,
        band_offset_eV=chi_l - chi_r,
        temperature_K=temp,
        a_star_left_A_m2_K2=a_l,
        a_star_right_A_m2_K2=a_r,
        dos_left_m3=nc_l,
        dos_right_m3=nc_r,
        transmission=0.25,
    )
    assert v2_zero_t.flux_m2_s == 0.0
    assert v2_quar_t.flux_m2_s == pytest.approx(0.25 * v2_si.flux_m2_s, rel=1.0e-15)

    scaps_v1 = reciprocal_thermionic_flux_m2_s(
        2.5e20,
        9.0e18,
        carrier="electron",
        phi_left_V=phi_l,
        phi_right_V=phi_r,
        band_offset_eV=chi_l - chi_r,
        temperature_K=temp,
        a_star_left_A_m2_K2=a_l,
        a_star_right_A_m2_K2=a_r,
        dos_left_m3=nc_l,
        dos_right_m3=nc_r,
        model_version=SCAPS_EFFECTIVE_VELOCITY_THERMIONIC_V1,
    )
    assert scaps_v1.effective_richardson_A_m2_K2 != pytest.approx(
        v2_si.effective_richardson_A_m2_K2, rel=0.05
    )
    assert abs(h_step_phys) > 0.0


@pytest.mark.parametrize("carrier", ["electron", "hole"])
@pytest.mark.parametrize("statistics", ["maxwell_boltzmann", "fermi_dirac"])
@pytest.mark.parametrize("band_offset_eV", [-0.19, -0.09, 0.09, 0.23])
def test_bare_reciprocal_te_analytical_jacobians_match_fourth_order_fd(
    carrier: ThermionicCarrierKind,
    statistics: ThermionicStatisticsKind,
    band_offset_eV: float,
) -> None:
    temp = 300.0
    vt = thermal_voltage(temp)
    dos_l, dos_r = 6.0e24, 2.0e25
    a_l, a_r = 2.4e5, 5.1e5
    phi_l, phi_r = 0.07, 0.16
    b_step, _, _ = physical_carrier_barrier_step_eV(
        carrier, phi_left_V=phi_l, phi_right_V=phi_r, band_offset_eV=band_offset_eV
    )
    # Non-equilibrium state with both over-barrier legs active and safely inside a single
    # 5e-3 Fermi-Dirac table interval so 4th-order FD sees smooth C^infinity evaluation:
    if statistics == "fermi_dirac":
        eta_l = 0.8125 if b_step >= 0.0 else (0.8125 + b_step / vt)
        eta_r = (eta_l - b_step / vt) + 0.9231
        # Snap away from k * 5e-3 grid knots to 0.0025 midpoint:
        eta_l = math.floor(eta_l / 5.0e-3) * 5.0e-3 + 2.5e-3
        eta_r = math.floor(eta_r / 5.0e-3) * 5.0e-3 + 2.5e-3
        c_l = dos_l * fermi_dirac_half(eta_l)
        c_r = dos_r * fermi_dirac_half(eta_r)
    else:
        c_l = 1.7e21
        c_r = 2.7 * c_l * (dos_r / dos_l) * math.exp(-b_step / vt)

    def eval_at(pl: float, pr: float, cl: float, cr: float) -> float:
        return reciprocal_thermionic_flux_m2_s(
            cl,
            cr,
            carrier=carrier,
            phi_left_V=pl,
            phi_right_V=pr,
            band_offset_eV=band_offset_eV,
            temperature_K=temp,
            a_star_left_A_m2_K2=a_l,
            a_star_right_A_m2_K2=a_r,
            dos_left_m3=dos_l,
            dos_right_m3=dos_r,
            transmission=0.82,
            statistics=statistics,
        ).flux_m2_s

    ana = reciprocal_thermionic_flux_m2_s(
        c_l,
        c_r,
        carrier=carrier,
        phi_left_V=phi_l,
        phi_right_V=phi_r,
        band_offset_eV=band_offset_eV,
        temperature_K=temp,
        a_star_left_A_m2_K2=a_l,
        a_star_right_A_m2_K2=a_r,
        dos_left_m3=dos_l,
        dos_right_m3=dos_r,
        transmission=0.82,
        statistics=statistics,
    )

    h_phi = 1.0e-4
    h_log = 1.0e-4
    fd_phi_l = _fourth_order_fd(lambda x: eval_at(x, phi_r, c_l, c_r), phi_l, h_phi)
    fd_phi_r = _fourth_order_fd(lambda x: eval_at(phi_l, x, c_l, c_r), phi_r, h_phi)
    fd_c_l = _fourth_order_fd(lambda x: eval_at(phi_l, phi_r, x, c_r), c_l, h_log * c_l)
    fd_c_r = _fourth_order_fd(lambda x: eval_at(phi_l, phi_r, c_l, x), c_r, h_log * c_r)
    fd_log_l = _fourth_order_fd(
        lambda z: eval_at(phi_l, phi_r, c_l * math.exp(z), c_r), 0.0, h_log
    )
    fd_log_r = _fourth_order_fd(
        lambda z: eval_at(phi_l, phi_r, c_l, c_r * math.exp(z)), 0.0, h_log
    )

    assert ana.d_flux_d_phi_left_m2_s_V == pytest.approx(fd_phi_l, rel=1.0e-9, abs=1.0e-9)
    assert ana.d_flux_d_phi_right_m2_s_V == pytest.approx(fd_phi_r, rel=1.0e-9, abs=1.0e-9)
    assert ana.d_flux_d_c_left_m_s == pytest.approx(fd_c_l, rel=1.0e-9, abs=1.0e-18)
    assert ana.d_flux_d_c_right_m_s == pytest.approx(fd_c_r, rel=1.0e-9, abs=1.0e-18)
    assert ana.d_flux_d_log_c_left_m2_s == pytest.approx(fd_log_l, rel=1.0e-9, abs=1.0e-9)
    assert ana.d_flux_d_log_c_right_m2_s == pytest.approx(fd_log_r, rel=1.0e-9, abs=1.0e-9)


@pytest.mark.parametrize("mode", ["harmonic_series", "compact_support", "legacy_logistic"])
@pytest.mark.parametrize("at_equilibrium", [False, True])
def test_composite_sg_and_reciprocal_te_jacobians_match_fd(
    mode: ThermionicCombinationMode,
    at_equilibrium: bool,
) -> None:
    temp = 300.0
    vt = thermal_voltage(temp)
    chi_l, chi_r = 3.75, 3.95
    eg_l, eg_r = 1.70, 1.55
    nc_l, nc_r = 1.5e25, 4.0e24
    nv_l, nv_r = 2.0e25, 8.0e24
    a_l, a_r = 1.6e5, 4.8e5
    phi_l, phi_r = 0.04, 0.19
    k_sg = 2.5e4

    bands = fold_dos_into_transport_bands(
        np.array([chi_l, chi_r]),
        np.array([eg_l, eg_r]),
        [np.array([True, False]), np.array([False, True])],
        [(nc_l, nv_l), (nc_r, nv_r)],
        reference_layer=0,
        thermal_voltage_V=vt,
    )
    e_step_phys, _ = physical_barrier_steps_eV(
        phi_left_V=phi_l,
        phi_right_V=phi_r,
        chi_phys_left_eV=chi_l,
        chi_phys_right_eV=chi_r,
        eg_phys_left_eV=eg_l,
        eg_phys_right_eV=eg_r,
    )
    c_l = 2.0e20
    c_r_eq = c_l * (nc_r / nc_l) * math.exp(-e_step_phys / vt)
    c_r = c_r_eq if at_equilibrium else (3.4 * c_r_eq)

    def eval_composite(
        pl: float, pr: float, cl: float, cr: float
    ) -> ReciprocalThermionicCompositeEvaluation:
        d_n = float(
            -((pr + bands.transport_electron_affinity_eV[1]) - (pl + bands.transport_electron_affinity_eV[0]))
            / vt
        )
        sg = stable_sg_particle_flux(cl, cr, d_n, k_sg)
        d_sg_d_step = float(sg.energy_step_derivative)
        te = reciprocal_thermionic_flux_m2_s(
            cl,
            cr,
            carrier="electron",
            phi_left_V=pl,
            phi_right_V=pr,
            band_offset_eV=chi_l - chi_r,
            temperature_K=temp,
            a_star_left_A_m2_K2=a_l,
            a_star_right_A_m2_K2=a_r,
            dos_left_m3=nc_l,
            dos_right_m3=nc_r,
        )
        return combine_sg_and_reciprocal_thermionic_flux(
            float(sg.flux),
            d_sg_d_phi_left_m2_s_V=d_sg_d_step / vt,
            d_sg_d_phi_right_m2_s_V=-d_sg_d_step / vt,
            d_sg_d_c_left_m_s=float(sg.left_density_derivative),
            d_sg_d_c_right_m_s=float(sg.right_density_derivative),
            c_left_m3=cl,
            c_right_m3=cr,
            thermionic=te,
            combination_mode=mode,
            te_softness=0.08,
        )

    ana = eval_composite(phi_l, phi_r, c_l, c_r)
    if at_equilibrium:
        assert abs(ana.flux_m2_s) <= 1.0e-15 * ana.thermionic.one_way_scale_m2_s

    h_phi = 5.0e-5
    h_log = 5.0e-5
    fd_phi_l = _fourth_order_fd(lambda x: eval_composite(x, phi_r, c_l, c_r).flux_m2_s, phi_l, h_phi)
    fd_phi_r = _fourth_order_fd(lambda x: eval_composite(phi_l, x, c_l, c_r).flux_m2_s, phi_r, h_phi)
    fd_log_l = _fourth_order_fd(
        lambda z: eval_composite(phi_l, phi_r, c_l * math.exp(z), c_r).flux_m2_s, 0.0, h_log
    )
    fd_log_r = _fourth_order_fd(
        lambda z: eval_composite(phi_l, phi_r, c_l, c_r * math.exp(z)).flux_m2_s, 0.0, h_log
    )

    # harmonic_series and compact_support are C^infinity across m = 0 (<= 1e-9);
    # legacy_logistic uses |sg| and |te| magnitudes which are C^1 at m = 0 (O(h) jump in f'').
    rel_tol = 5.0e-5 if (at_equilibrium and mode == "legacy_logistic") else 1.0e-9
    assert ana.d_flux_d_phi_left_m2_s_V == pytest.approx(fd_phi_l, rel=rel_tol)
    assert ana.d_flux_d_phi_right_m2_s_V == pytest.approx(fd_phi_r, rel=rel_tol)
    assert ana.d_flux_d_log_c_left_m2_s == pytest.approx(fd_log_l, rel=rel_tol)
    assert ana.d_flux_d_log_c_right_m2_s == pytest.approx(fd_log_r, rel=rel_tol)


def test_contact_5mev_gate_alignment_provenance_and_scaps_mirror_historical_record() -> None:
    assert CONTACT_THERMODYNAMIC_TOLERANCE_EV == 5.0e-3
    assert set(CONTACT_ALIGNMENT_CLASSES) == {
        PARAMETER_FREE_CONTACT_ALIGNMENT,
        EXTERNAL_METAL_CONTACT_ALIGNMENT,
        FITTED_MANUAL_CONTACT_ALIGNMENT,
    }
    assert classify_contact_alignment_mode("semiconductor_work_function") == PARAMETER_FREE_CONTACT_ALIGNMENT
    assert classify_contact_alignment_mode("metal_work_function") == EXTERNAL_METAL_CONTACT_ALIGNMENT
    assert classify_contact_alignment_mode("legacy_manual") == FITTED_MANUAL_CONTACT_ALIGNMENT
    with pytest.raises(ValueError, match="unknown built-in-potential mode"):
        classify_contact_alignment_mode("fitted_free_offset")

    defaults = load_packaged_default_catalog("device_only")
    resources = load_packaged_resource_library(source_id_prefix="src/solarlab/materials/data")
    scaps_src = SourceDocument("scaps_mirror_v2.yaml", SCAPS_MIRROR_PRESET.read_bytes())
    prepared = resolve_device(
        import_scaps_device(scaps_src, id="scaps_mirror_v2", defaults=defaults),
        defaults,
        resources,
        sources=(scaps_src,),
    )
    temp_K = float(prepared.values["settings"]["T"])
    vt_dev = thermal_voltage(temp_K)
    htl, pvk, etl = prepared.layers[1], prepared.layers[2], prepared.layers[3]
    p_l = dict(htl.parameters)
    p_r = dict(etl.parameters)
    n_l, hole_l = doping_equilibrium_densities(
        float(p_l["N_D"]), float(p_l["N_A"]), float(p_l["ni"]) ** 2
    )
    n_r, hole_r = doping_equilibrium_densities(
        float(p_r["N_D"]), float(p_r["N_A"]), float(p_r["ni"]) ** 2
    )
    left_ep = ContactEndpoint(
        n_l,
        hole_l,
        float(p_l["Nc300"]),
        float(p_l["Nv300"]),
        float(p_l["chi"]),
        float(p_l["Eg"]),
    )
    right_ep = ContactEndpoint(
        n_r,
        hole_r,
        float(p_r["Nc300"]),
        float(p_r["Nv300"]),
        float(p_r["chi"]),
        float(p_r["Eg"]),
    )
    left_mat = ContactMaterial(
        temperature_K=temp_K,
        electron_affinity_eV=float(p_l["chi"]),
        band_gap_eV=float(p_l["Eg"]),
        conduction_dos_m3=float(p_l["Nc300"]),
        valence_dos_m3=float(p_l["Nv300"]),
        donor_density_m3=float(p_l["N_D"]),
        acceptor_density_m3=float(p_l["N_A"]),
    )
    right_mat = ContactMaterial(
        temperature_K=temp_K,
        electron_affinity_eV=float(p_r["chi"]),
        band_gap_eV=float(p_r["Eg"]),
        conduction_dos_m3=float(p_r["Nc300"]),
        valence_dos_m3=float(p_r["Nv300"]),
        donor_density_m3=float(p_r["N_D"]),
        acceptor_density_m3=float(p_r["N_A"]),
    )
    left_band = LegacyBandContact(
        float(p_l["chi"]), float(p_l["Eg"]), float(p_l["N_D"]), float(p_l["N_A"]), float(p_l["ni"])
    )
    right_band = LegacyBandContact(
        float(p_r["chi"]), float(p_r["Eg"]), float(p_r["N_D"]), float(p_r["N_A"]), float(p_r["ni"])
    )
    elec_bands = tuple(
        (float(dict(lyr.parameters)["chi"]), float(dict(lyr.parameters)["Eg"]))
        for lyr in (htl, pvk, etl)
    )
    sources_manual = BuiltInPotentialSources(
        mode="legacy_manual",
        manual_V_bi=float(prepared.values["settings"]["V_bi"]),
        left_band=left_band,
        right_band=right_band,
        electrical_layer_bands=elec_bands,
        left_material=left_mat,
        right_material=right_mat,
    )
    semi_v_bi = float(sources_manual.semiconductor_potential())
    manual_v_bi_bc = float(sources_manual.poisson_potential())
    stats_kind = str(prepared.values["settings"].get("carrier_statistics") or "maxwell_boltzmann")

    # 1. Historical scaps_mirror_v2 legacy_manual (6.024958093130e-3 eV) stays permanently inconsistent:
    hist = assess_contact_thermodynamics(
        built_in_potential_mode="legacy_manual",
        left=left_ep,
        right=right_ep,
        V_bi_bc=manual_v_bi_bc,
        thermal_voltage_V=vt_dev,
        statistics=stats_kind,
        semiconductor_potential_V=semi_v_bi,
        has_band_reference=True,
    )
    assert hist.status == "inconsistent"
    assert not hist.certified
    assert not hist.is_parameter_free
    assert hist.alignment_class == FITTED_MANUAL_CONTACT_ALIGNMENT
    assert hist.historical_label == LEGACY_MANUAL_CONTACT_OFFSET
    assert hist.fermi_level_span_eV == pytest.approx(6.024958093130e-3, abs=1.0e-12)
    with pytest.raises(ContactThermodynamicError, match=LEGACY_MANUAL_CONTACT_OFFSET):
        require_contact_thermodynamic_certificate(hist)
    with pytest.raises(ContactThermodynamicError, match=LEGACY_MANUAL_CONTACT_OFFSET):
        certify_parameter_free_contact_equilibrium(hist)

    # 2. Tolerance > 5.0e-3 eV is rejected with ValueError:
    for bad_tol in (5.000001e-3, 6.1e-3, 0.05):
        with pytest.raises(ValueError, match="cannot exceed"):
            assess_contact_thermodynamics(
                built_in_potential_mode="legacy_manual",
                left=left_ep,
                right=right_ep,
                V_bi_bc=manual_v_bi_bc,
                thermal_voltage_V=vt_dev,
                statistics=stats_kind,
                semiconductor_potential_V=semi_v_bi,
                has_band_reference=True,
                tolerance_eV=bad_tol,
            )

    # 3. Even a fitted legacy_manual V_bi tuned inside 5.0 meV is refused by parameter-free gate:
    tuned_manual = assess_contact_thermodynamics(
        built_in_potential_mode="legacy_manual",
        left=left_ep,
        right=right_ep,
        V_bi_bc=semi_v_bi + 2.0e-3,
        thermal_voltage_V=vt_dev,
        statistics=stats_kind,
        semiconductor_potential_V=semi_v_bi,
        has_band_reference=True,
    )
    assert tuned_manual.certified
    assert not tuned_manual.is_parameter_free
    with pytest.raises(ContactThermodynamicError, match="fitted_manual_potential"):
        certify_parameter_free_contact_equilibrium(tuned_manual)

    # 4. Compatible parameter-free semiconductor_work_function on the same reservoirs is certified:
    sources_compat = dataclasses.replace(sources_manual, mode="semiconductor_work_function")
    compat = certify_parameter_free_contact_equilibrium(
        assess_contact_thermodynamics(
            built_in_potential_mode="semiconductor_work_function",
            left=left_ep,
            right=right_ep,
            V_bi_bc=float(sources_compat.poisson_potential()),
            thermal_voltage_V=vt_dev,
            statistics=stats_kind,
            semiconductor_potential_V=semi_v_bi,
            has_band_reference=True,
        )
    )
    assert compat.certified
    assert compat.is_parameter_free
    assert compat.alignment_class == PARAMETER_FREE_CONTACT_ALIGNMENT
    assert compat.historical_label is None
    assert compat.fermi_level_span_eV is not None and compat.fermi_level_span_eV < 1.0e-12
