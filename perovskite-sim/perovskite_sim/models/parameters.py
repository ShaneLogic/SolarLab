from __future__ import annotations
from dataclasses import dataclass
import math
import yaml

from perovskite_sim.constants import Q, K_B, T, V_T  # noqa: F401
from perovskite_sim.physics.band_gap_narrowing import (
    BAND_GAP_NARROWING_OFF,
    SLOTBOOM,
    BandGapNarrowingModel,
    apply_band_gap_narrowing,
    normalize_band_gap_narrowing_model,
)
from perovskite_sim.physics.bulk_traps import BulkTrapDistribution
from perovskite_sim.physics.statistics import (
    CarrierStatistics,
    DISCRETE_LEVEL,
    DopantIonizationModel,
    FERMI_DIRAC,
    FULLY_IONIZED,
    MAXWELL_BOLTZMANN,
    normalize_carrier_statistics,
    normalize_dopant_ionization_model,
)


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
    # SCAPS layer thermal velocity at 300 K [m/s].  This is distinct from
    # ``A_star``: SCAPS' heterointerface thermionic boundary uses the smaller
    # declared thermal velocity of the two adjacent layers.
    v_th: float = 1.0e5
    # Negative ion species (e.g. V_MA-, halide interstitial)
    D_ion_neg: float = 0.0     # diffusion coefficient [m²/s] (0 = single species)
    P0_neg: float = 0.0        # equilibrium density [m⁻³]
    P_lim_neg: float = 1e30    # steric limit [m⁻³]
    # Temperature-dependent scaling parameters (all optional, T=300 K default)
    Nc300: float | None = None      # effective conduction-band DOS at 300 K [m⁻³]
    Nv300: float | None = None      # effective valence-band DOS at 300 K [m⁻³]
    # Bulk carrier statistics.  Maxwell-Boltzmann is the historical/default
    # constitutive law.  Fermi-Dirac is an explicit research opt-in and
    # requires physical band-edge DOS data on every activated layer.
    carrier_statistics: CarrierStatistics = MAXWELL_BOLTZMANN
    # Dopants remain fully ionized unless a discrete donor/acceptor level is
    # explicitly selected. Binding energies are measured from the adjacent
    # band edge: E_C-E_D for donors and E_A-E_V for acceptors.
    dopant_ionization_model: DopantIonizationModel = FULLY_IONIZED
    donor_binding_energy_eV: float | None = None
    acceptor_binding_energy_eV: float | None = None
    donor_degeneracy: float = 2.0
    acceptor_degeneracy: float = 4.0
    # Static heavy-doping band-edge correction. Slotboom parameters use SI
    # density and eV energy units; the conduction fraction partitions DeltaEg
    # between the two physical band edges.
    band_gap_narrowing_model: BandGapNarrowingModel = BAND_GAP_NARROWING_OFF
    bgn_reference_energy_eV: float = 0.009
    bgn_reference_density_m3: float = 1.0e23
    bgn_log_shape: float = 0.5
    bgn_conduction_band_fraction: float = 0.5
    mu_T_gamma: float = -1.5        # mobility temperature exponent
    E_a_ion: float = 0.58           # ion activation energy [eV] (Arrhenius)
    # Phase 4b temperature scaling of radiative recombination and bandgap.
    # ``B_rad_T_gamma`` is the power-law exponent in B(T) = B_300 · (T/300)^γ
    # — default 0 keeps the pre-Phase-4b bit-identical behaviour. Set to
    # ``-1.5`` to recover the detailed-balance scaling for a non-
    # degenerate semiconductor at fixed bandgap (the typical literature
    # value for MAPbI3). ``varshni_alpha`` and ``varshni_beta`` define
    # the Varshni bandgap shift Eg(T) = Eg_300 − α·[T²/(T+β) − T_REF²/(T_REF+β)].
    # α = 0 (default) disables the shift. Silicon: α ≈ 4.73e-4 eV/K,
    # β ≈ 636 K (Eg narrows with heating). MAPbI3 is opposite — its
    # bandgap *increases* with T, reproduced by α ≈ −3e-4 eV/K with a
    # positive β (~+200 K is a representative value).
    B_rad_T_gamma: float = 0.0
    varshni_alpha: float = 0.0      # Varshni α [eV/K] — 0 = disabled
    varshni_beta: float = 0.0       # Varshni β [K]
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
    # Which absorber face the trap edge profile attaches to. Default
    # "both" reproduces the original symmetric Phase 4a behaviour.
    # "left" / "right" attach the kernel to a single face only so
    # heterojunction-specific defects (e.g. SCAPS PVK/ETL Gaussian
    # interface trap) can drive asymmetric recombination that responds
    # to the band offset on that side.
    trap_edge: str = "both"                  # "both" | "left" | "right"
    # Explicit energy-resolved bulk defects. This is distinct from the legacy
    # spatial lifetime profile above: it carries an integrated density, capture
    # kinetics, and a donor/acceptor neutral-charge reference.
    bulk_trap_distribution: BulkTrapDistribution | None = None
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
    # Continuous bandgap / electron-affinity grading (2026-06). The existing
    # scalar ``chi`` / ``Eg`` are the FRONT (x=0) endpoints; these are the
    # BACK (far-face) endpoints. A layer is graded iff ``Eg_back is not None
    # or chi_back is not None`` (physics/grading.has_grading_params). All
    # default None/sentinel so legacy configs stay bit-identical, and grading
    # only activates when ``DeviceStack.band_grading`` is on (LEGACY tier
    # forces it off). SCAPS material-driven law — see physics/grading.py
    # (Burgelman & Marlein, 23rd EU PVSEC 2008).
    Eg_back: float | None = None     # band gap at far face [eV]; None = uniform
    chi_back: float | None = None    # electron affinity at far face [eV]; None = uniform
    grading_profile: str = "linear"  # "linear" | "parabolic" | "exponential"
    grading_direction: str = "front_to_back"  # or "back_to_front" (flips y)
    grading_bowing: float = 0.0      # alloy bowing b in Eg(y) law [eV]
    grading_char_length: float | None = None  # notch length L for exponential y(x) [m]
    grading_N_mult: int = 1          # per-layer mesh refinement factor (1 = unchanged)
    # Optional spatial dopant profile. N_A/N_D are the density at the selected
    # edge; a populated N_A_bulk or N_D_bulk activates a Gaussian decay toward
    # that deep-layer asymptote. All fields are inert when both bulk values are
    # None, preserving uniform-doping configs exactly.
    N_A_bulk: float | None = None
    N_D_bulk: float | None = None
    doping_profile_shape: str | None = None  # currently "gaussian"
    doping_decay_length: float | None = None  # Gaussian 1/e distance [m]
    doping_edge: str = "front"              # "front" | "back"

    def __post_init__(self) -> None:
        statistics = normalize_carrier_statistics(self.carrier_statistics)
        object.__setattr__(self, "carrier_statistics", statistics)
        ionization = normalize_dopant_ionization_model(
            self.dopant_ionization_model
        )
        object.__setattr__(self, "dopant_ionization_model", ionization)
        narrowing_model = normalize_band_gap_narrowing_model(
            self.band_gap_narrowing_model
        )
        object.__setattr__(
            self, "band_gap_narrowing_model", narrowing_model
        )
        bulk_trap = self.bulk_trap_distribution
        if bulk_trap is not None:
            if not isinstance(bulk_trap, BulkTrapDistribution):
                raise TypeError(
                    "bulk_trap_distribution must be a BulkTrapDistribution or None"
                )
            try:
                bulk_trap_gap = float(self.Eg)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "energy-resolved bulk traps require a finite positive Eg"
                ) from exc
            if not math.isfinite(bulk_trap_gap) or bulk_trap_gap <= 0.0:
                raise ValueError(
                    "energy-resolved bulk traps require a finite positive Eg"
                )
            bulk_trap.validate_band_gap(bulk_trap_gap)
            if statistics != MAXWELL_BOLTZMANN:
                raise ValueError(
                    "energy-resolved bulk traps currently require "
                    "carrier_statistics='maxwell_boltzmann'"
                )
            if ionization != FULLY_IONIZED:
                raise ValueError(
                    "energy-resolved bulk traps currently require "
                    "dopant_ionization_model='fully_ionized'"
                )
            if narrowing_model != BAND_GAP_NARROWING_OFF:
                raise ValueError(
                    "energy-resolved bulk traps currently exclude band-gap narrowing"
                )

        if ionization == FULLY_IONIZED:
            if (
                self.donor_binding_energy_eV is not None
                or self.acceptor_binding_energy_eV is not None
                or self.donor_degeneracy != 2.0
                or self.acceptor_degeneracy != 4.0
            ):
                raise ValueError(
                    "dopant level parameters require "
                    "dopant_ionization_model='discrete_level'"
                )
        elif ionization == DISCRETE_LEVEL:
            if not all(
                math.isfinite(float(value)) and float(value) >= 0.0
                for value in (self.N_A, self.N_D)
            ):
                raise ValueError(
                    "discrete_level ionization requires finite non-negative "
                    "N_A and N_D"
                )
            donor_active = self.N_D > 0.0 or (
                self.N_D_bulk is not None and self.N_D_bulk > 0.0
            )
            acceptor_active = self.N_A > 0.0 or (
                self.N_A_bulk is not None and self.N_A_bulk > 0.0
            )
            if donor_active and self.donor_binding_energy_eV is None:
                raise ValueError(
                    "active discrete donors require donor_binding_energy_eV"
                )
            if acceptor_active and self.acceptor_binding_energy_eV is None:
                raise ValueError(
                    "active discrete acceptors require "
                    "acceptor_binding_energy_eV"
                )
            for name, value in (
                ("donor_binding_energy_eV", self.donor_binding_energy_eV),
                ("acceptor_binding_energy_eV", self.acceptor_binding_energy_eV),
            ):
                if value is not None and (
                    not math.isfinite(float(value)) or float(value) < 0.0
                ):
                    raise ValueError(f"{name} must be finite and non-negative")
            for name, value in (
                ("donor_degeneracy", self.donor_degeneracy),
                ("acceptor_degeneracy", self.acceptor_degeneracy),
            ):
                if not math.isfinite(float(value)) or float(value) <= 0.0:
                    raise ValueError(f"{name} must be finite and positive")

        if narrowing_model == BAND_GAP_NARROWING_OFF:
            if (
                self.bgn_reference_energy_eV != 0.009
                or self.bgn_reference_density_m3 != 1.0e23
                or self.bgn_log_shape != 0.5
                or self.bgn_conduction_band_fraction != 0.5
            ):
                raise ValueError(
                    "BGN parameters require "
                    "band_gap_narrowing_model='slotboom'"
                )
        elif narrowing_model == SLOTBOOM:
            apply_band_gap_narrowing(
                electron_affinity_eV=float(self.chi),
                band_gap_eV=float(self.Eg),
                acceptor_density_m3=max(
                    float(self.N_A), float(self.N_A_bulk or 0.0)
                ),
                donor_density_m3=max(
                    float(self.N_D), float(self.N_D_bulk or 0.0)
                ),
                model=SLOTBOOM,
                reference_energy_eV=float(self.bgn_reference_energy_eV),
                reference_density_m3=float(self.bgn_reference_density_m3),
                log_shape=float(self.bgn_log_shape),
                conduction_band_fraction=float(
                    self.bgn_conduction_band_fraction
                ),
            )

        if (
            statistics != FERMI_DIRAC
            and ionization != DISCRETE_LEVEL
            and narrowing_model != SLOTBOOM
            and bulk_trap is None
        ):
            return
        required_positive = {
            "Eg": self.Eg,
            "Nc300": self.Nc300,
            "Nv300": self.Nv300,
        }
        invalid = [
            name
            for name, value in required_positive.items()
            if value is None
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ]
        if invalid:
            closure_name = (
                "fermi_dirac carrier statistics"
                if statistics == FERMI_DIRAC
                else (
                    "discrete_level ionization"
                    if ionization == DISCRETE_LEVEL
                    else (
                        "slotboom band-gap narrowing"
                        if narrowing_model == SLOTBOOM
                        else "energy-resolved bulk traps"
                    )
                )
            )
            raise ValueError(
                f"{closure_name} require finite positive "
                + ", ".join(invalid)
            )
        if not all(
            math.isfinite(float(value)) and float(value) >= 0.0
            for value in (self.N_A, self.N_D)
        ):
            closure_name = (
                "fermi_dirac carrier statistics"
                if statistics == FERMI_DIRAC
                else (
                    "discrete_level ionization"
                    if ionization == DISCRETE_LEVEL
                    else (
                        "slotboom band-gap narrowing"
                        if narrowing_model == SLOTBOOM
                        else "energy-resolved bulk traps"
                    )
                )
            )
            raise ValueError(
                f"{closure_name} require finite non-negative "
                "N_A and N_D"
            )

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
