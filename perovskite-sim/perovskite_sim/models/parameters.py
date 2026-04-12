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
    # Optical data source for TMM (None = use scalar alpha Beer-Lambert)
    optical_material: str | None = None   # e.g. "MAPbI3", "TiO2", "spiro_OMeTAD"
    n_optical: float | None = None        # constant refractive index (fallback)

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
