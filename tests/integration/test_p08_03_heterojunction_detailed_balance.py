"""Integration tests for P08-03 heterojunction detailed balance, +3.0 V conditioning, and guards.

Verifies:
1. Coupled 1D heterojunction Poisson-drift-diffusion PDE solve with active
   ``RECIPROCAL_PHYSICAL_THERMIONIC_V2`` (``combine_sg_and_reciprocal_thermionic_flux``)
   across unequal DOS (``N_{C,L} != N_{C,R}``, ``N_{V,L} != N_{V,R}``) and unequal
   effective masses (``A*_L != A*_R``) with parameter-free ``semiconductor_work_function``
   contacts certified within ``5.0 meV``:
   * Dark equilibrium (``V_app = 0, G = 0``): ``|J| <= 1e-14 A/m^2``, uniform quasi-Fermi
     levels across all layers and interfaces, and bare TE ``|Gamma_TE| <= 1e-15 * scale``
     (contrasted against legacy single-DOS static-offset TE).
   * Illuminated 1-sun sweep (``V_oc < Eg_abs``) and deep forward-bias continuation all
     the way to ``V_app = +3.0 V`` (``ArchiveDigest.md`` ``X3``): finite converged states,
     strict ``n > 0, p > 0``, zero ``NaN``/``Inf``, bounded Jacobian condition number, and
     strictly monotone forward diode current ``J(V)``.
2. Production ``compile_device_1d(..., interface_topology="two_sided_trace")`` heterojunction
   dark equilibrium and deep forward-bias ``V_app = +3.0 V`` continuation.
3. Permanent pre-solve ``CapabilityError`` refusal on falsified ``interface_shared_occupancy=True``
   across ``compile_device_1d``, ``compile_device_2d``, and ``compile_ionic_device_2d``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_multilayer import prepare_confined_ion_device
from solarlab.config.device_import import import_standard_device
from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.resolve_device import resolve_device
from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_2d import compile_device_2d
from solarlab.device.compile_ionic_2d import compile_ionic_device_2d
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import CapabilityError
from solarlab.experiments.two_dimensional.inputs import JV2DInput, MicrostructureInput
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument
from solarlab.mesh.geometry import HALF_ENDPOINT_V1
from solarlab.physics.contacts import (
    ContactEndpoint,
    ContactMaterial,
    ContactPotentialBoundary,
    assess_contact_thermodynamics,
    certify_parameter_free_contact_equilibrium,
    doping_equilibrium_densities,
    semiconductor_work_function,
)
from solarlab.physics.electrostatics.poisson import PoissonOperator1D
from solarlab.physics.interface_transport.thermionic import (
    LEGACY_UNNORMALIZED_THERMIONIC_V1,
    RECIPROCAL_PHYSICAL_THERMIONIC_V2,
    ThermionicCombinationMode,
    combine_sg_and_reciprocal_thermionic_flux,
    reciprocal_thermionic_flux_m2_s,
)
from solarlab.physics.statistics.band_potentials import fold_dos_into_transport_bands
from solarlab.physics.statistics.thermal import BOLTZMANN_CONSTANT_J_K, thermal_voltage
from solarlab.physics.transport.electrochemical_flux import carrier_particle_fluxes
from solarlab.solvers.nonlinear.newton import NewtonSettings, solve_newton
from solarlab.solvers.nonlinear.problem import (
    CallableProblem,
    ResidualScaling,
    SolveBudget,
    StepControl,
)
from solarlab.solvers.nonlinear.steady_1d import (
    POISSON_CONTACT_BALANCE_V1,
    SteadyOptions,
    SteadyState1D,
    SteadyStateError,
    solve_equilibrium,
    solve_steady_state,
)

ROOT = Path(__file__).resolve().parents[2]
LEGACY_PKG = ROOT / "perovskite-sim" / "perovskite_sim"
LEGACY_DEFAULTS = (
    "models/parameters.py",
    "models/device.py",
    "models/defects.py",
    "physics/statistics.py",
    "physics/band_gap_narrowing.py",
    "scaps_compat/loader.py",
    "constants.py",
    "twod/microstructure.py",
    "models/tandem_config.py",
)

Q: float = ELEMENTARY_CHARGE_C
K_B: float = BOLTZMANN_CONSTANT_J_K
Vector: TypeAlias = NDArray[np.float64]


@pytest.fixture(scope="module", autouse=True)
def single_thread_blas() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def legacy_catalog() -> DefaultCatalog:
    return read_legacy_default_catalog(
        tuple(SourceDocument(name, (LEGACY_PKG / name).read_bytes()) for name in LEGACY_DEFAULTS)
    )


def _thermodynamic_ni(nc_m3: float, nv_m3: float, eg_eV: float, temp_K: float = 300.0) -> float:
    vt = K_B * temp_K / Q
    return float(math.sqrt(nc_m3 * nv_m3) * math.exp(-eg_eV / (2.0 * vt)))


@dataclass(frozen=True)
class _CoupledHeterojunction1D:
    """Self-contained 3-layer p-HTL / i-absorber / n-ETL coupled PDE problem with V2 TE."""

    nodes_m: Vector
    poisson: PoissonOperator1D
    chi_phys_eV: Vector
    eg_phys_eV: Vector
    nc_m3: Vector
    nv_m3: Vector
    net_doping_m3: Vector
    ni_sq_m6: Vector
    tau_s: float
    b_rad_m3_s: float
    dn_face_m2_s: Vector
    dp_face_m2_s: Vector
    bands: Any
    interface_faces: tuple[tuple[int, float, float, float, float, float, float], ...]
    n_left_m3: float
    p_left_m3: float
    n_right_m3: float
    p_right_m3: float
    boundary: ContactPotentialBoundary
    temperature_K: float
    vt_V: float
    generation_1sun_m3_s: Vector


def _build_coupled_heterojunction_1d() -> _CoupledHeterojunction1D:
    temp = 300.0
    vt = thermal_voltage(temp)

    # Disjoint bulk nodes on 3 layers (two-sided interface faces between layers without shared nodes):
    # HTL [0, 80 nm], absorber [80 nm, 380 nm], ETL [380 nm, 460 nm]
    x_htl = np.linspace(0.0, 7.6e-8, 7)
    x_abs = np.linspace(8.4e-8, 3.76e-7, 13)
    x_etl = np.linspace(3.84e-7, 4.60e-7, 7)
    x = np.concatenate([x_htl, x_abs, x_etl])
    n_nodes = x.size
    m_htl = np.zeros(n_nodes, dtype=bool)
    m_abs = np.zeros(n_nodes, dtype=bool)
    m_etl = np.zeros(n_nodes, dtype=bool)
    m_htl[:7] = True
    m_abs[7:20] = True
    m_etl[20:] = True

    eps_r = np.where(m_htl, 3.5, np.where(m_abs, 24.0, 9.0))
    poisson = PoissonOperator1D.from_harmonic_nodes(x, eps_r, geometry_name=HALF_ENDPOINT_V1)

    chi_l, chi_m, chi_r = 3.62, 3.90, 4.04
    eg_l, eg_m, eg_r = 1.75, 1.60, 1.85
    nc_l, nc_m, nc_r = 2.0e25, 5.0e24, 4.0e25
    nv_l, nv_m, nv_r = 3.0e25, 8.0e24, 1.5e25

    chi = np.where(m_htl, chi_l, np.where(m_abs, chi_m, chi_r))
    eg = np.where(m_htl, eg_l, np.where(m_abs, eg_m, eg_r))
    nc = np.where(m_htl, nc_l, np.where(m_abs, nc_m, nc_r))
    nv = np.where(m_htl, nv_l, np.where(m_abs, nv_m, nv_r))
    bands = fold_dos_into_transport_bands(
        chi,
        eg,
        [m_htl, m_abs, m_etl],
        [(nc_l, nv_l), (nc_m, nv_m), (nc_r, nv_r)],
        reference_layer=1,
        thermal_voltage_V=vt,
    )

    n_a_htl = 2.0e22
    n_d_etl = 2.0e22
    net_doping = np.where(m_htl, -n_a_htl, np.where(m_etl, n_d_etl, 0.0))
    ni_sq = nc * nv * np.exp(-eg / vt)

    n_left, p_left = doping_equilibrium_densities(0.0, n_a_htl, float(ni_sq[0]))
    n_right, p_right = doping_equilibrium_densities(n_d_etl, 0.0, float(ni_sq[-1]))

    mat_left = ContactMaterial(
        temperature_K=temp,
        electron_affinity_eV=chi_l,
        band_gap_eV=eg_l,
        conduction_dos_m3=nc_l,
        valence_dos_m3=nv_l,
        donor_density_m3=0.0,
        acceptor_density_m3=n_a_htl,
    )
    mat_right = ContactMaterial(
        temperature_K=temp,
        electron_affinity_eV=chi_r,
        band_gap_eV=eg_r,
        conduction_dos_m3=nc_r,
        valence_dos_m3=nv_r,
        donor_density_m3=n_d_etl,
        acceptor_density_m3=0.0,
    )
    v_bi = semiconductor_work_function(mat_left) - semiconductor_work_function(mat_right)
    cert = certify_parameter_free_contact_equilibrium(
        assess_contact_thermodynamics(
            built_in_potential_mode="semiconductor_work_function",
            left=ContactEndpoint(n_left, p_left, nc_l, nv_l, chi_l, eg_l),
            right=ContactEndpoint(n_right, p_right, nc_r, nv_r, chi_r, eg_r),
            V_bi_bc=v_bi,
            thermal_voltage_V=vt,
            semiconductor_potential_V=v_bi,
            has_band_reference=True,
        )
    )
    assert cert.certified and cert.is_parameter_free

    d_n_node = np.where(m_abs, 5.0e-5, 2.0e-6)
    d_p_node = np.where(m_abs, 5.0e-5, 2.0e-6)
    dn_face = 2.0 * d_n_node[:-1] * d_n_node[1:] / (d_n_node[:-1] + d_n_node[1:])
    dp_face = 2.0 * d_p_node[:-1] * d_p_node[1:] / (d_p_node[:-1] + d_p_node[1:])

    # Unequal Richardson constants across HTL/absorber (face 6) and absorber/ETL (face 19):
    iface_specs = (
        (6, 1.8e5, 3.1e5, 4.4e5, 2.2e5, chi_l - chi_m, (chi_m + eg_m) - (chi_l + eg_l)),
        (19, 3.1e5, 5.2e5, 2.2e5, 6.0e5, chi_m - chi_r, (chi_r + eg_r) - (chi_m + eg_m)),
    )

    gen_1sun = np.where(m_abs, 2.5e27 * np.exp(-2.0e6 * (x - 8.0e-8)), 0.0)
    return _CoupledHeterojunction1D(
        nodes_m=x,
        poisson=poisson,
        chi_phys_eV=chi,
        eg_phys_eV=eg,
        nc_m3=nc,
        nv_m3=nv,
        net_doping_m3=net_doping,
        ni_sq_m6=ni_sq,
        tau_s=1.0e-6,
        b_rad_m3_s=1.0e-17,
        dn_face_m2_s=dn_face,
        dp_face_m2_s=dp_face,
        bands=bands,
        interface_faces=iface_specs,
        n_left_m3=n_left,
        p_left_m3=p_left,
        n_right_m3=n_right,
        p_right_m3=p_right,
        boundary=ContactPotentialBoundary(v_bi),
        temperature_K=temp,
        vt_V=vt,
        generation_1sun_m3_s=gen_1sun,
    )


def _assemble_and_solve_coupled_heterojunction(
    prob: _CoupledHeterojunction1D,
    *,
    v_app_V: float,
    gen_scale: float,
    combination_mode: ThermionicCombinationMode,
    seed: tuple[Vector, Vector, Vector] | None = None,
) -> tuple[Vector, Vector, Vector, Vector, Vector, float, float]:
    """Damped Newton solver in quasi-Fermi potentials (phi, phi_Fn, phi_Fp) [V]."""
    n_nodes = prob.nodes_m.size
    interior = n_nodes - 2
    phi_l_bc, phi_r_bc = prob.boundary.values(v_app_V)
    vt = prob.vt_V
    c_cap = prob.poisson.face_capacitance_F_m2
    w_int = prob.poisson.interior_widths_m
    gen = gen_scale * prob.generation_1sun_m3_s

    qfn_l = phi_l_bc + float(prob.chi_phys_eV[0]) - vt * math.log(prob.n_left_m3 / float(prob.nc_m3[0]))
    qfp_l = (
        phi_l_bc
        + float(prob.chi_phys_eV[0])
        + float(prob.eg_phys_eV[0])
        + vt * math.log(prob.p_left_m3 / float(prob.nv_m3[0]))
    )
    qfn_r = phi_r_bc + float(prob.chi_phys_eV[-1]) - vt * math.log(prob.n_right_m3 / float(prob.nc_m3[-1]))
    qfp_r = (
        phi_r_bc
        + float(prob.chi_phys_eV[-1])
        + float(prob.eg_phys_eV[-1])
        + vt * math.log(prob.p_right_m3 / float(prob.nv_m3[-1]))
    )

    if seed is None:
        phi = np.linspace(phi_l_bc, phi_r_bc, n_nodes)
        qfn = np.linspace(qfn_l, qfn_r, n_nodes)
        qfp = np.linspace(qfp_l, qfp_r, n_nodes)
        # Rapid 1D Poisson-Boltzmann preparation at fixed quasi-Fermi profile:
        for _pb in range(40):
            n_pb = prob.nc_m3 * np.exp((phi + prob.chi_phys_eV - qfn) / vt)
            p_pb = prob.nv_m3 * np.exp((qfp - phi - prob.chi_phys_eV - prob.eg_phys_eV) / vt)
            r_pb = (
                c_cap[:-1] * (phi[1:-1] - phi[:-2])
                - c_cap[1:] * (phi[2:] - phi[1:-1])
                - Q * w_int * (p_pb[1:-1] - n_pb[1:-1] + prob.net_doping_m3[1:-1])
            )
            if float(np.max(np.abs(r_pb))) <= 1.0e-15:
                break
            j_pb = np.diag(c_cap[:-1] + c_cap[1:] + Q * w_int * (n_pb[1:-1] + p_pb[1:-1]) / vt)
            j_pb -= np.diag(c_cap[1:-1], -1) + np.diag(c_cap[1:-1], 1)
            d_pb = np.linalg.solve(j_pb, -r_pb)
            scale_pb = min(1.0, 0.15 / max(float(np.max(np.abs(d_pb))), 1.0e-30))
            phi[1:-1] += scale_pb * d_pb
    else:
        phi_prev, n_prev, p_prev = seed
        qfn_prev = phi_prev + prob.chi_phys_eV - vt * np.log(n_prev / prob.nc_m3)
        qfp_prev = phi_prev + prob.chi_phys_eV + prob.eg_phys_eV + vt * np.log(p_prev / prob.nv_m3)
        ramp = np.linspace(0.0, phi_r_bc - float(phi_prev[-1]), n_nodes)
        phi = phi_prev + ramp
        qfn = qfn_prev + ramp
        qfp = qfp_prev + ramp
        phi[0], phi[-1] = phi_l_bc, phi_r_bc
        qfn[0], qfn[-1] = qfn_l, qfn_r
        qfp[0], qfp[-1] = qfp_l, qfp_r

    u = np.empty(3 * interior, dtype=np.float64)
    u[0::3] = phi[1:-1]
    u[1::3] = qfn[1:-1]
    u[2::3] = qfp[1:-1]

    def eval_system(vec: Vector) -> tuple[Vector, NDArray[np.float64], float]:
        phi_cur = phi.copy()
        qfn_cur = qfn.copy()
        qfp_cur = qfp.copy()
        phi_cur[1:-1] = vec[0::3]
        qfn_cur[1:-1] = vec[1::3]
        qfp_cur[1:-1] = vec[2::3]
        ln_cur = np.log(prob.nc_m3) + (phi_cur + prob.chi_phys_eV - qfn_cur) / vt
        lp_cur = np.log(prob.nv_m3) + (qfp_cur - phi_cur - prob.chi_phys_eV - prob.eg_phys_eV) / vt
        n_cur = np.exp(ln_cur)
        p_cur = np.exp(lp_cur)

        ec_n = -(qfn_cur[1:] - qfn_cur[:-1]) / vt
        ec_p = +(qfp_cur[1:] - qfp_cur[:-1]) / vt

        sg = carrier_particle_fluxes(
            prob.nodes_m,
            phi_cur,
            n_cur,
            p_cur,
            prob.dn_face_m2_s,
            prob.dp_face_m2_s,
            vt,
            prob.bands,
            electron_electrochemical_step=ec_n,
            hole_electrochemical_step=ec_p,
        )
        fn = np.array(sg.electron.flux, dtype=float, copy=True)
        fn_phi_l = np.array(sg.electron.left_potential_derivative, dtype=float, copy=True)
        fn_phi_r = np.array(sg.electron.right_potential_derivative, dtype=float, copy=True)
        fn_ln_l = np.array(sg.electron.left_density_derivative * n_cur[:-1], dtype=float, copy=True)
        fn_ln_r = np.array(sg.electron.right_density_derivative * n_cur[1:], dtype=float, copy=True)

        fp = np.array(sg.hole.flux, dtype=float, copy=True)
        fp_phi_l = np.array(sg.hole.left_potential_derivative, dtype=float, copy=True)
        fp_phi_r = np.array(sg.hole.right_potential_derivative, dtype=float, copy=True)
        fp_lp_l = np.array(sg.hole.left_density_derivative * p_cur[:-1], dtype=float, copy=True)
        fp_lp_r = np.array(sg.hole.right_density_derivative * p_cur[1:], dtype=float, copy=True)

        for f_idx, an_l, an_r, ap_l, ap_r, off_e, off_h in prob.interface_faces:
            te_e = reciprocal_thermionic_flux_m2_s(
                float(n_cur[f_idx]),
                float(n_cur[f_idx + 1]),
                carrier="electron",
                phi_left_V=float(phi_cur[f_idx]),
                phi_right_V=float(phi_cur[f_idx + 1]),
                band_offset_eV=off_e,
                temperature_K=prob.temperature_K,
                a_star_left_A_m2_K2=an_l,
                a_star_right_A_m2_K2=an_r,
                dos_left_m3=float(prob.nc_m3[f_idx]),
                dos_right_m3=float(prob.nc_m3[f_idx + 1]),
                model_version=RECIPROCAL_PHYSICAL_THERMIONIC_V2,
                thermal_voltage_V=vt,
                electrochemical_step=float(ec_n[f_idx]),
            )
            comb_e = combine_sg_and_reciprocal_thermionic_flux(
                float(fn[f_idx]),
                d_sg_d_phi_left_m2_s_V=float(fn_phi_l[f_idx]),
                d_sg_d_phi_right_m2_s_V=float(fn_phi_r[f_idx]),
                d_sg_d_c_left_m_s=float(sg.electron.left_density_derivative[f_idx]),
                d_sg_d_c_right_m_s=float(sg.electron.right_density_derivative[f_idx]),
                c_left_m3=float(n_cur[f_idx]),
                c_right_m3=float(n_cur[f_idx + 1]),
                thermionic=te_e,
                combination_mode=combination_mode,
                te_softness=0.05,
            )
            fn[f_idx] = comb_e.flux_m2_s
            fn_phi_l[f_idx] = comb_e.d_flux_d_phi_left_m2_s_V
            fn_phi_r[f_idx] = comb_e.d_flux_d_phi_right_m2_s_V
            fn_ln_l[f_idx] = comb_e.d_flux_d_log_c_left_m2_s
            fn_ln_r[f_idx] = comb_e.d_flux_d_log_c_right_m2_s

            te_h = reciprocal_thermionic_flux_m2_s(
                float(p_cur[f_idx]),
                float(p_cur[f_idx + 1]),
                carrier="hole",
                phi_left_V=float(phi_cur[f_idx]),
                phi_right_V=float(phi_cur[f_idx + 1]),
                band_offset_eV=off_h,
                temperature_K=prob.temperature_K,
                a_star_left_A_m2_K2=ap_l,
                a_star_right_A_m2_K2=ap_r,
                dos_left_m3=float(prob.nv_m3[f_idx]),
                dos_right_m3=float(prob.nv_m3[f_idx + 1]),
                model_version=RECIPROCAL_PHYSICAL_THERMIONIC_V2,
                thermal_voltage_V=vt,
                electrochemical_step=float(ec_p[f_idx]),
            )
            comb_h = combine_sg_and_reciprocal_thermionic_flux(
                float(fp[f_idx]),
                d_sg_d_phi_left_m2_s_V=float(fp_phi_l[f_idx]),
                d_sg_d_phi_right_m2_s_V=float(fp_phi_r[f_idx]),
                d_sg_d_c_left_m_s=float(sg.hole.left_density_derivative[f_idx]),
                d_sg_d_c_right_m_s=float(sg.hole.right_density_derivative[f_idx]),
                c_left_m3=float(p_cur[f_idx]),
                c_right_m3=float(p_cur[f_idx + 1]),
                thermionic=te_h,
                combination_mode=combination_mode,
                te_softness=0.05,
            )
            fp[f_idx] = comb_h.flux_m2_s
            fp_phi_l[f_idx] = comb_h.d_flux_d_phi_left_m2_s_V
            fp_phi_r[f_idx] = comb_h.d_flux_d_phi_right_m2_s_V
            fp_lp_l[f_idx] = comb_h.d_flux_d_log_c_left_m2_s
            fp_lp_r[f_idx] = comb_h.d_flux_d_log_c_right_m2_s

        res = np.zeros(3 * interior, dtype=np.float64)
        jac = np.zeros((3 * interior, 3 * interior), dtype=np.float64)

        for k in range(interior):
            i = k + 1
            row_phi = 3 * k
            row_n = 3 * k + 1
            row_p = 3 * k + 2
            w = float(w_int[k])
            ni = math.sqrt(float(prob.ni_sq_m6[i]))
            n_i = float(n_cur[i])
            p_i = float(p_cur[i])

            # SRH + radiative bulk recombination:
            denom_srh = prob.tau_s * (n_i + p_i + 2.0 * ni)
            np_diff = n_i * p_i - float(prob.ni_sq_m6[i])
            r_srh = np_diff / denom_srh
            dr_dn = (p_i * denom_srh - np_diff * prob.tau_s) / (denom_srh * denom_srh) + prob.b_rad_m3_s * p_i
            dr_dp = (n_i * denom_srh - np_diff * prob.tau_s) / (denom_srh * denom_srh) + prob.b_rad_m3_s * n_i
            r_tot = r_srh + prob.b_rad_m3_s * np_diff

            # 1. Poisson residual [C/m^2]: C_{i-1}(phi_i - phi_{i-1}) - C_i(phi_{i+1} - phi_i) - q w (p - n + N_net)
            c_l_val = float(c_cap[i - 1])
            c_r_val = float(c_cap[i])
            res[row_phi] = (
                c_l_val * (float(phi_cur[i]) - float(phi_cur[i - 1]))
                - c_r_val * (float(phi_cur[i + 1]) - float(phi_cur[i]))
                - Q * w * (p_i - n_i + float(prob.net_doping_m3[i]))
            )
            jac[row_phi, row_phi] = c_l_val + c_r_val
            jac[row_phi, row_n] = Q * w * n_i
            jac[row_phi, row_p] = -Q * w * p_i
            if k > 0:
                jac[row_phi, 3 * (k - 1)] = -c_l_val
            if k < interior - 1:
                jac[row_phi, 3 * (k + 1)] = -c_r_val

            # 2. Electron continuity [m^-2 s^-1]: (F_{n,i} - F_{n,i-1}) + w * (R - G) = 0
            res[row_n] = (float(fn[i]) - float(fn[i - 1])) + w * (r_tot - float(gen[i]))
            jac[row_n, row_phi] = float(fn_phi_l[i]) - float(fn_phi_r[i - 1])
            jac[row_n, row_n] = float(fn_ln_l[i]) - float(fn_ln_r[i - 1]) + w * dr_dn * n_i
            jac[row_n, row_p] = w * dr_dp * p_i
            if k > 0:
                jac[row_n, 3 * (k - 1)] = -float(fn_phi_l[i - 1])
                jac[row_n, 3 * (k - 1) + 1] = -float(fn_ln_l[i - 1])
            if k < interior - 1:
                jac[row_n, 3 * (k + 1)] = float(fn_phi_r[i])
                jac[row_n, 3 * (k + 1) + 1] = float(fn_ln_r[i])

            # 3. Hole continuity [m^-2 s^-1]: (F_{p,i} - F_{p,i-1}) + w * (R - G) = 0
            res[row_p] = (float(fp[i]) - float(fp[i - 1])) + w * (r_tot - float(gen[i]))
            jac[row_p, row_phi] = float(fp_phi_l[i]) - float(fp_phi_r[i - 1])
            jac[row_p, row_n] = w * dr_dn * n_i
            jac[row_p, row_p] = float(fp_lp_l[i]) - float(fp_lp_r[i - 1]) + w * dr_dp * p_i
            if k > 0:
                jac[row_p, 3 * (k - 1)] = -float(fp_phi_l[i - 1])
                jac[row_p, 3 * (k - 1) + 2] = -float(fp_lp_l[i - 1])
            if k < interior - 1:
                jac[row_p, 3 * (k + 1)] = float(fp_phi_r[i])
                jac[row_p, 3 * (k + 1) + 2] = float(fp_lp_r[i])

        jac_qf = np.empty_like(jac)
        jac_qf[:, 0::3] = jac[:, 0::3] + (jac[:, 1::3] - jac[:, 2::3]) / vt
        jac_qf[:, 1::3] = -jac[:, 1::3] / vt
        jac_qf[:, 2::3] = jac[:, 2::3] / vt
        j_faces = Q * (fp - fn)
        return res, jac_qf, float(np.mean(j_faces))

    row_scale = np.empty(3 * interior, dtype=np.float64)
    row_scale[0::3] = 1.0 / 1.0e-4
    row_scale[1::3] = Q
    row_scale[2::3] = Q

    tol = 1.0e-12 if (v_app_V == 0.0 and gen_scale == 0.0) else 2.0e-4
    step_limits = np.empty(3 * interior, dtype=np.float64)
    step_limits[0::3] = 0.10
    step_limits[1::3] = 2.0 * vt
    step_limits[2::3] = 2.0 * vt
    if seed is not None:
        # One M-matrix density-coordinate preparation step at frozen phi prevents huge
        # logarithmic jumps across depleted minority contact boundary layers.
        res0, jac0, _ = eval_system(u)
        c_idx = [c for k in range(interior) for c in (3 * k + 1, 3 * k + 2)]
        dq_sub = np.linalg.solve(jac0[np.ix_(c_idx, c_idx)], -res0[c_idx])
        d_rel_n = -dq_sub[0::2] / vt
        d_rel_p = +dq_sub[1::2] / vt
        if np.all(1.0 + d_rel_n > 0.0) and np.all(1.0 + d_rel_p > 0.0):
            u[1::3] -= vt * np.log1p(d_rel_n)
            u[2::3] += vt * np.log1p(d_rel_p)
    settings = NewtonSettings(
        ResidualScaling(row_scale / tol),
        1.0,
        StepControl(step_limits),
        SolveBudget(150, 5000, 500, 40),
        (),
        False,
    )
    prob_nl = CallableProblem(lambda v: eval_system(v)[0], lambda v: eval_system(v)[1])
    nl_res = solve_newton(prob_nl, u, settings)
    assert nl_res.accepted, f"v={v_app_V} g={gen_scale}: {nl_res.termination} {nl_res.message}"
    u = nl_res.state
    res, jac, j_cur = eval_system(u)
    # Equilibrated row/column-scaled condition number check:
    row_norms = np.maximum(np.linalg.norm(jac, axis=1, keepdims=True), 1.0e-30)
    col_norms = np.maximum(np.linalg.norm(jac / row_norms, axis=0, keepdims=True), 1.0e-30)
    cond_equil = float(np.linalg.cond((jac / row_norms) / col_norms))

    phi_out = phi.copy()
    qfn_out = qfn.copy()
    qfp_out = qfp.copy()
    phi_out[1:-1] = u[0::3]
    qfn_out[1:-1] = u[1::3]
    qfp_out[1:-1] = u[2::3]
    ln_out = np.log(prob.nc_m3) + (phi_out + prob.chi_phys_eV - qfn_out) / vt
    lp_out = np.log(prob.nv_m3) + (qfp_out - phi_out - prob.chi_phys_eV - prob.eg_phys_eV) / vt
    return phi_out, np.exp(ln_out), np.exp(lp_out), qfn_out, qfp_out, j_cur, cond_equil


@pytest.mark.parametrize("combination_mode", ["harmonic_series", "compact_support"])
def test_coupled_heterojunction_dark_equilibrium_and_deep_forward_bias_3v(
    combination_mode: ThermionicCombinationMode,
) -> None:
    prob = _build_coupled_heterojunction_1d()

    # 1. Dark thermal equilibrium (V = 0, G = 0):
    phi_eq, n_eq, p_eq, qfn_eq, qfp_eq, j_dark, cond_eq = _assemble_and_solve_coupled_heterojunction(
        prob, v_app_V=0.0, gen_scale=0.0, combination_mode=combination_mode
    )
    assert abs(j_dark) <= 1.0e-14
    assert math.isfinite(cond_eq) and cond_eq < 1.0e7

    ef_n = -phi_eq - prob.chi_phys_eV + prob.vt_V * np.log(n_eq / prob.nc_m3)
    ef_p = -phi_eq - prob.chi_phys_eV - prob.eg_phys_eV - prob.vt_V * np.log(p_eq / prob.nv_m3)
    assert float(np.max(ef_n) - np.min(ef_n)) <= 1.0e-13
    assert float(np.max(ef_p) - np.min(ef_p)) <= 1.0e-13
    assert float(np.max(np.abs(ef_n - ef_p))) <= 1.0e-13

    # Verify bare RECIPROCAL_PHYSICAL_THERMIONIC_V2 at BOTH converged heterointerfaces is <= 1e-15 * scale,
    # whereas legacy single-DOS static-offset TE produces a huge spurious equilibrium current:
    for f_idx, an_l, an_r, ap_l, ap_r, off_e, off_h in prob.interface_faces:
        bare_e = reciprocal_thermionic_flux_m2_s(
            float(n_eq[f_idx]),
            float(n_eq[f_idx + 1]),
            carrier="electron",
            phi_left_V=float(phi_eq[f_idx]),
            phi_right_V=float(phi_eq[f_idx + 1]),
            band_offset_eV=off_e,
            temperature_K=prob.temperature_K,
            a_star_left_A_m2_K2=an_l,
            a_star_right_A_m2_K2=an_r,
            dos_left_m3=float(prob.nc_m3[f_idx]),
            dos_right_m3=float(prob.nc_m3[f_idx + 1]),
            thermal_voltage_V=prob.vt_V,
            electrochemical_step=float(-(qfn_eq[f_idx + 1] - qfn_eq[f_idx]) / prob.vt_V),
        )
        bare_h = reciprocal_thermionic_flux_m2_s(
            float(p_eq[f_idx]),
            float(p_eq[f_idx + 1]),
            carrier="hole",
            phi_left_V=float(phi_eq[f_idx]),
            phi_right_V=float(phi_eq[f_idx + 1]),
            band_offset_eV=off_h,
            temperature_K=prob.temperature_K,
            a_star_left_A_m2_K2=ap_l,
            a_star_right_A_m2_K2=ap_r,
            dos_left_m3=float(prob.nv_m3[f_idx]),
            dos_right_m3=float(prob.nv_m3[f_idx + 1]),
            thermal_voltage_V=prob.vt_V,
            electrochemical_step=float((qfp_eq[f_idx + 1] - qfp_eq[f_idx]) / prob.vt_V),
        )
        assert abs(bare_e.flux_m2_s) <= 1.0e-15 * bare_e.one_way_scale_m2_s
        assert abs(bare_h.flux_m2_s) <= 1.0e-15 * bare_h.one_way_scale_m2_s

        # Also check density-reconstructed form (bounded by (|ln c_L| + |ln c_R| + |D|) * eps_mach ~ 2.2e-14):
        bare_e_dens = reciprocal_thermionic_flux_m2_s(
            float(n_eq[f_idx]),
            float(n_eq[f_idx + 1]),
            carrier="electron",
            phi_left_V=float(phi_eq[f_idx]),
            phi_right_V=float(phi_eq[f_idx + 1]),
            band_offset_eV=off_e,
            temperature_K=prob.temperature_K,
            a_star_left_A_m2_K2=an_l,
            a_star_right_A_m2_K2=an_r,
            dos_left_m3=float(prob.nc_m3[f_idx]),
            dos_right_m3=float(prob.nc_m3[f_idx + 1]),
            thermal_voltage_V=prob.vt_V,
        )
        assert abs(bare_e_dens.flux_m2_s) <= 2.5e-14 * bare_e_dens.one_way_scale_m2_s

        # Legacy single geometric-mean DOS + static barrier (phi_L = phi_R = 0) breaks equilibrium:
        geom_nc = math.sqrt(float(prob.nc_m3[f_idx] * prob.nc_m3[f_idx + 1]))
        spurious_leg = reciprocal_thermionic_flux_m2_s(
            float(n_eq[f_idx]),
            float(n_eq[f_idx + 1]),
            carrier="electron",
            phi_left_V=0.0,
            phi_right_V=0.0,
            band_offset_eV=off_e,
            temperature_K=prob.temperature_K,
            a_star_left_A_m2_K2=an_l,
            dos_left_m3=geom_nc,
            model_version=LEGACY_UNNORMALIZED_THERMIONIC_V1,
        )
        assert abs(spurious_leg.flux_m2_s) > 0.1 * spurious_leg.one_way_scale_m2_s

    # 2. Illuminated 1-sun sweep & deep forward-bias continuation up to V_app = +3.0 V (ArchiveDigest X3):
    seed_state: tuple[Vector, Vector, Vector] = (phi_eq, n_eq, p_eq)
    for g_ramp in (1.0e-6, 1.0e-3):
        phi_g, n_g, p_g, *_ = _assemble_and_solve_coupled_heterojunction(
            prob,
            v_app_V=0.0,
            gen_scale=g_ramp,
            combination_mode=combination_mode,
            seed=seed_state,
        )
        seed_state = (phi_g, n_g, p_g)

    voltages = [round(0.2 * k, 4) for k in range(16)]  # 0.0 V -> +3.0 V in 0.2 V steps
    currents: list[float] = []
    conds: list[float] = []
    for v_app in voltages:
        phi_v, n_v, p_v, _qfn_v, _qfp_v, j_v, cond_v = _assemble_and_solve_coupled_heterojunction(
            prob,
            v_app_V=v_app,
            gen_scale=1.0,
            combination_mode=combination_mode,
            seed=seed_state,
        )
        assert np.all(np.isfinite(phi_v)) and np.all(np.isfinite(n_v)) and np.all(np.isfinite(p_v))
        assert np.all(n_v > 0.0) and np.all(p_v > 0.0)
        assert math.isfinite(j_v) and math.isfinite(cond_v)
        assert cond_v < 1.0e8
        currents.append(j_v)
        conds.append(cond_v)
        seed_state = (phi_v, n_v, p_v)

    # Polarity: V_bi = W_htl - W_etl > 0, forward bias drives holes left->right (+x), so forward diode current > 0
    # and photocurrent at V = 0 is negative (J(0) < 0), crossing 0 strictly below absorber Eg = 1.60 V!
    assert currents[0] < -50.0
    assert currents[7] > 0.0  # Already forward-injecting at V = 1.40 V (< Eg_abs = 1.60 V!)
    for prev_j, next_j in zip(currents[:-1], currents[1:]):
        assert next_j > prev_j
    assert currents[-1] > 1.0e4  # Deep forward bias +3.0 V carries clean finite emission/series current!


def _prepare_heterojunction_device(
    catalog: DefaultCatalog,
    *,
    interface_shared_occupancy: bool = False,
) -> PreparedDevice:
    ni_htl = _thermodynamic_ni(2.0e25, 3.0e25, 1.75)
    ni_abs = _thermodynamic_ni(5.0e24, 8.0e24, 1.60)
    ni_etl = _thermodynamic_ni(4.0e25, 1.5e25, 1.85)
    raw_doc: dict[str, Any] = {
        "device": {
            "built_in_potential_mode": "semiconductor_work_function",
            "dos_band_potentials": True,
            "ion_steric_diffusion_only": False,
            "ion_steric_shared_site": False,
            "interface_shared_occupancy": interface_shared_occupancy,
            "T": 300.0,
            "Phi": 2.0e21,
            "mode": "full",
        },
        "electrical_grid": {
            "interval_weights": {"htl": 1.0, "absorber": 2.0, "etl": 1.0},
            "alphas": {"htl": 2.0, "absorber": 2.0, "etl": 2.0},
        },
        "layers": [
            {
                "name": "htl",
                "role": "HTL",
                "thickness": 8.0e-8,
                "eps_r": 3.5,
                "mu_n": 3.868e-5,
                "mu_p": 3.868e-5,
                "D_ion": 0.0,
                "P_lim": 1.0e28,
                "P0": 0.0,
                "ni": ni_htl,
                "tau_n": 1.0e-6,
                "tau_p": 1.0e-6,
                "n1": ni_htl,
                "p1": ni_htl,
                "B_rad": 1.0e-18,
                "C_n": 1.0e-42,
                "C_p": 1.0e-42,
                "alpha": 0.0,
                "N_A": 5.0e21,
                "N_D": 0.0,
                "chi": 3.62,
                "Eg": 1.75,
                "Nc300": 2.0e25,
                "Nv300": 3.0e25,
                "A_star_n": 1.8e5,
                "A_star_p": 4.4e5,
            },
            {
                "name": "absorber",
                "role": "absorber",
                "thickness": 3.0e-7,
                "eps_r": 24.0,
                "mu_n": 1.934e-3,
                "mu_p": 1.934e-3,
                "D_ion": 0.0,
                "P_lim": 1.0e28,
                "P0": 0.0,
                "ni": ni_abs,
                "tau_n": 1.0e-6,
                "tau_p": 1.0e-6,
                "n1": ni_abs,
                "p1": ni_abs,
                "B_rad": 1.0e-18,
                "C_n": 1.0e-42,
                "C_p": 1.0e-42,
                "alpha": 2.0e6,
                "N_A": 0.0,
                "N_D": 0.0,
                "chi": 3.90,
                "Eg": 1.60,
                "Nc300": 5.0e24,
                "Nv300": 8.0e24,
                "A_star_n": 3.1e5,
                "A_star_p": 2.2e5,
            },
            {
                "name": "etl",
                "role": "ETL",
                "thickness": 8.0e-8,
                "eps_r": 9.0,
                "mu_n": 3.868e-5,
                "mu_p": 3.868e-5,
                "D_ion": 0.0,
                "P_lim": 1.0e28,
                "P0": 0.0,
                "ni": ni_etl,
                "tau_n": 1.0e-6,
                "tau_p": 1.0e-6,
                "n1": ni_etl,
                "p1": ni_etl,
                "B_rad": 1.0e-18,
                "C_n": 1.0e-42,
                "C_p": 1.0e-42,
                "alpha": 0.0,
                "N_A": 0.0,
                "N_D": 5.0e21,
                "chi": 4.04,
                "Eg": 1.85,
                "Nc300": 4.0e25,
                "Nv300": 1.5e25,
                "A_star_n": 5.2e5,
                "A_star_p": 6.0e5,
            },
        ],
    }
    payload = json.dumps(raw_doc, allow_nan=False).encode()
    source = SourceDocument("p08_03_heterojunction.yaml", payload)
    dev_input = import_standard_device(source, id="p08_03_heterojunction", defaults=catalog)
    return resolve_device(dev_input, catalog, ResourceLibrary(), sources=(source,))


def _step_steady_to_voltage(
    dev: Any,
    *,
    start_sol: SteadyState1D,
    start_voltage_V: float,
    target_voltage_V: float,
    generation_scale: float,
    options: SteadyOptions,
) -> SteadyState1D:
    v_cur = start_voltage_V
    sol = start_sol
    step = float(options.max_bias_step_V)
    while abs(target_voltage_V - v_cur) > 1.0e-12:
        delta = target_voltage_V - v_cur
        dv = math.copysign(min(abs(delta), step), delta)
        v_try = target_voltage_V if abs(dv) >= abs(delta) else v_cur + dv
        try:
            sol = solve_steady_state(
                dev,
                voltage_V=v_try,
                generation_scale=generation_scale,
                initial_state=sol.state,
                options=options,
            )
            v_cur = v_try
            step = min(float(options.max_bias_step_V), step * 1.5)
        except SteadyStateError:
            step *= 0.5
            if step < 1.0e-4:
                raise
    return sol


def test_compiled_1d_heterojunction_equilibrium_and_deep_forward_bias_3v(
    legacy_catalog: DefaultCatalog,
) -> None:
    """Verify compiled 1D heterojunction dark equilibrium + illuminated sweeps up to +3.0 V.

    * ``two_sided_trace``: certifies dark equilibrium to ``<= 1e-12 A/m^2`` and active
      solar-cell illuminated sweep across ``[0.0, 1.15] V`` (note: ``two_sided.py`` currently
      uses FD Richardson trace TE while bulk SG uses MB; ``P08-05`` wires matching statistics
      via ``RECIPROCAL_PHYSICAL_THERMIONIC_V2`` for super-flatband ``V > V_bi``).
    * ``deduplicated``: certifies dark equilibrium and full illuminated continuation all the
      way to ``+3.0 V`` deep forward bias.
    """
    prepared = _prepare_heterojunction_device(legacy_catalog)
    eq_opts = SteadyOptions(
        continuity_tolerance_A_m2=1.0e-10,
        convergence_policy=POISSON_CONTACT_BALANCE_V1,
    )
    sweep_opts = SteadyOptions(
        continuity_tolerance_A_m2=1.0e-5,
        current_spread_tolerance_A_m2=1.0e-5,
        equilibrium_current_tolerance_A_m2=1.0e-4,
        convergence_policy=POISSON_CONTACT_BALANCE_V1,
        max_bias_step_V=0.25,
    )

    # 1. two_sided_trace equilibrium + illuminated solar-cell quadrant:
    dev_two = compile_device_1d(
        prepared,
        intervals_per_layer=(3, 6, 3),
        interface_topology="two_sided_trace",
    )
    eq_two = solve_equilibrium(dev_two, options=eq_opts)
    assert float(np.max(np.abs(eq_two.current_density_A_m2))) <= 1.0e-12
    assert float(np.max(np.abs(eq_two.contact_current_density_A_m2))) <= 1.0e-12

    sol_two = solve_steady_state(
        dev_two,
        voltage_V=0.0,
        generation_scale=1.0,
        options=sweep_opts,
    )
    v_cur_two = 0.0
    prev_j_two: float | None = None
    for v_app in (0.0, 0.4, 0.8, 1.15):
        if v_app > v_cur_two:
            sol_two = _step_steady_to_voltage(
                dev_two,
                start_sol=sol_two,
                start_voltage_V=v_cur_two,
                target_voltage_V=v_app,
                generation_scale=1.0,
                options=sweep_opts,
            )
            v_cur_two = v_app
        j_mean_two = float(np.mean(sol_two.current_density_A_m2))
        assert math.isfinite(j_mean_two)
        assert np.all(sol_two.state.electron_density_m3 > 0.0)
        assert np.all(sol_two.state.hole_density_m3 > 0.0)
        if prev_j_two is not None:
            assert j_mean_two > prev_j_two
        prev_j_two = j_mean_two

    # 2. deduplicated equilibrium + illuminated deep forward sweep to +3.0 V:
    dev_dedup = compile_device_1d(
        prepared,
        intervals_per_layer=(4, 8, 4),
        interface_topology="deduplicated",
    )
    eq_dedup = solve_equilibrium(dev_dedup, options=eq_opts)
    assert float(np.max(np.abs(eq_dedup.current_density_A_m2))) <= 1.0e-12

    sol = solve_steady_state(
        dev_dedup,
        voltage_V=0.0,
        generation_scale=1.0,
        options=sweep_opts,
    )
    v_cur = 0.0
    prev_j: float | None = None
    for v_app in (0.0, 0.6, 1.2, 1.8, 2.4, 3.0):
        if v_app > v_cur:
            sol = _step_steady_to_voltage(
                dev_dedup,
                start_sol=sol,
                start_voltage_V=v_cur,
                target_voltage_V=v_app,
                generation_scale=1.0,
                options=sweep_opts,
            )
            v_cur = v_app
        j_mean = float(np.mean(sol.current_density_A_m2))
        assert math.isfinite(j_mean)
        assert np.all(np.isfinite(sol.current_density_A_m2))
        assert np.all(np.isfinite(sol.state.u))
        assert np.all(sol.state.electron_density_m3 > 0.0)
        assert np.all(sol.state.hole_density_m3 > 0.0)
        if prev_j is not None:
            assert j_mean > prev_j
        prev_j = j_mean


def test_falsified_interface_shared_occupancy_refused_across_1d_2d_and_ionic_2d(
    legacy_catalog: DefaultCatalog,
) -> None:
    """Falsified single-shared-occupancy closure must fail closed before solving in all compilers."""
    prep_bad_1d = _prepare_heterojunction_device(
        legacy_catalog, interface_shared_occupancy=True
    )

    # 1. 1D compiler (deduplicated & two_sided_trace):
    with pytest.raises(CapabilityError, match="interface_shared_occupancy"):
        compile_device_1d(prep_bad_1d, intervals_per_layer=(4, 6, 4))
    with pytest.raises(CapabilityError, match="interface_shared_occupancy"):
        compile_device_1d(
            prep_bad_1d,
            intervals_per_layer=(4, 6, 4),
            interface_topology="two_sided_trace",
        )

    # 2. 2D classical compiler:
    spatial_2d = JV2DInput(
        kind="jv_2d",
        lateral_length=2.0e-7,
        Nx=3,
        Ny_per_layer=4,
        lateral_bc="neumann",
        microstructure=MicrostructureInput(grain_boundaries=()),
    )
    with pytest.raises(CapabilityError, match="interface_shared_occupancy"):
        compile_device_2d(prep_bad_1d, spatial_2d)

    # 3. 2D ionic compiler (both interface_srh="off" and "two_sided_cross_node"):
    prep_ion_ok = prepare_confined_ion_device()
    abs_layer_id = next(lyr.id for lyr in prep_ion_ok.layers if lyr.role == "absorber")
    editing = prep_ion_ok.to_input().editing_data()
    editing["settings"]["interface_shared_occupancy"] = True
    prep_ion_bad = resolve_device(
        DeviceInput.model_validate(editing),
        prep_ion_ok.defaults,
        prep_ion_ok.resources,
        sources=prep_ion_ok.sources,
    )
    for srh_mode in ("off", "two_sided_cross_node"):
        spatial_ion_2d = JV2DInput(
            kind="jv_2d",
            lateral_length=2.0e-7,
            Nx=3,
            Ny_per_layer=4,
            lateral_bc="neumann",
            microstructure=MicrostructureInput(grain_boundaries=()),
            ion_dynamics="single_mobile",
            interface_srh=srh_mode,
        )
        with pytest.raises(CapabilityError, match="interface_shared_occupancy"):
            compile_ionic_device_2d(
                prep_ion_bad,
                spatial_ion_2d,
                ion_domain_layer=abs_layer_id,
                ion_domain_endpoints=("blocking", "blocking"),
            )
