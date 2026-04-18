from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import yaml

from perovskite_sim.constants import Q, K_B, T, V_T  # noqa: F401


@dataclass(frozen=True)
class MaterialParams:
    eps_r: float
    mu_n: float     # m²/Vs
    mu_p: float
    D_ion: float    # ion diffusion coefficient m²/s (0 if no ions)
    P_lim: float    # maximum ion vacancy density m⁻³
    P0: float       # initial (equilibrium) ion density m⁻³
    ni: float       # intrinsic carrier density m⁻³
    tau_n: float    # SRH electron lifetime s
    tau_p: float
    n1: float       # SRH trap-level carrier densities
    p1: float
    B_rad: float    # radiative recombination coefficient m³/s
    C_n: float      # Auger coefficient m⁶/s
    C_p: float
    alpha: float    # optical absorption coefficient m⁻¹
    N_A: float      # acceptor doping m⁻³
    N_D: float      # donor doping m⁻³
    chi: float = 0.0   # electron affinity [eV] (= voltage, since 1 eV/q = 1 V)
    Eg: float = 0.0    # band gap [eV]
    A_star_n: float = 1.2017e6   # Richardson constant for electrons [A/(m²·K²)]
    A_star_p: float = 1.2017e6   # Richardson constant for holes [A/(m²·K²)]
    # Negative ion species (e.g. V_MA-, halide interstitial)
    D_ion_neg: float = 0.0     # diffusion coefficient [m²/s] (0 = single species)
    P0_neg: float = 0.0        # equilibrium density [m⁻³]
    P_lim_neg: float = 1e30    # steric limit [m⁻³]
    # Temperature-dependent scaling parameters (all optional, T=300 K default)
    Nc300: float | None = None      # effective conduction-band DOS at 300 K [m⁻³]
    Nv300: float | None = None      # effective valence-band DOS at 300 K [m⁻³]
    mu_T_gamma: float = -1.5        # mobility temperature exponent
    E_a_ion: float = 0.58           # ion activation energy [eV] (Arrhenius)
    # Spatially varying trap profile (None = uniform tau).
    # ``trap_profile_shape`` selects between the two forms in
    # physics/traps.py: "exponential" (the Phase 4 default) and
    # "gaussian" (Phase 4a — faster decay into the bulk for defect
    # layers with a well-defined finite extent). ``trap_decay_length``
    # is the length parameter in both cases — the exponential 1/e scale
    # for "exponential" and the Gaussian sigma for "gaussian".
    trap_N_t_interface: float | None = None  # interface trap density [m⁻³]
    trap_N_t_bulk: float | None = None       # bulk trap density [m⁻³]
    trap_decay_length: float | None = None   # decay length / sigma [m]
    trap_profile_shape: str = "exponential"  # "exponential" | "gaussian"
    # Optical data source for TMM (None = use scalar alpha Beer-Lambert)
    optical_material: str | None = None   # e.g. "MAPbI3", "TiO2", "spiro_OMeTAD"
    n_optical: float | None = None        # constant refractive index (fallback)
    # Optical coherence flag for TMM. When True, the layer is treated as
    # incoherent (bulk Beer-Lambert + Fresnel interfaces, no interference).
    # Must be True for mm-thick substrates; defaults False (coherent).
    incoherent: bool = False
    # Field-dependent mobility parameters (Phase 3.2 — Apr 2026).
    # Caughey-Thomas velocity-saturation: at |E| ≫ v_sat / μ₀ the drift
    # velocity asymptotes to v_sat. Defaults v_sat_{n,p} = 0 disable CT at
    # this layer — the low-field μ is returned unchanged. β is the
    # Caughey-Thomas exponent; β = 2 is the Canali form used for silicon
    # electrons, β = 1 is the Thornber form used for silicon holes. We
    # default both to 2 for perovskite-ish materials where the literature
    # does not strongly favour one over the other.
    v_sat_n: float = 0.0      # electron saturation velocity [m/s]
    v_sat_p: float = 0.0      # hole saturation velocity [m/s]
    ct_beta_n: float = 2.0    # CT exponent for electrons
    ct_beta_p: float = 2.0    # CT exponent for holes
    # Poole-Frenkel field-enhanced mobility: μ = μ₀ · exp(γ · √|E|).
    # Relevant for disordered / organic transport layers (e.g. spiro).
    # γ = 0 disables the model; typical γ for spiro-OMeTAD is
    # ~3e-4 (V/m)^-0.5 (arg ~ 3 at |E| = 1e8 V/m, i.e. μ ≈ 20·μ₀).
    pf_gamma_n: float = 0.0   # PF prefactor for electrons [(V/m)^-0.5]
    pf_gamma_p: float = 0.0   # PF prefactor for holes [(V/m)^-0.5]

    @property
    def D_n(self) -> float:
        return self.mu_n * V_T

    @property
    def D_p(self) -> float:
        return self.mu_p * V_T

    @property
    def ni_sq(self) -> float:
        return self.ni ** 2


@dataclass(frozen=True)
class SolverConfig:
    N: int = 200
    alpha_grid: float = 3.0
    rtol: float = 1e-4
    atol: float = 1e-6
    T: float = 300.0


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
