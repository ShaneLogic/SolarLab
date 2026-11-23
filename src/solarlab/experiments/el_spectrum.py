"""Forward-injection reciprocity electroluminescence (EL) spectrum on 1D DC.

Evaluates Rau's optoelectronic reciprocity relation around a genuine dark
forward-injection DC operating point at ``V_inj_V > 0``:

    Phi_EL(lambda, V_inj) = A_abs(lambda) * phi_bb(lambda, T) * exp(q * V_inj / (k_B * T))

together with radiative emission current ``J_em_rad_A_m2 = q * int Phi_EL dlambda``,
total injected dark current ``J_inj_conjugate_A_m2`` (and PV-convention
``J_inj_pv_A_m2``), luminescent external quantum efficiency
``EQE_EL = J_em_rad / J_inj_conjugate``, and non-radiative open-circuit voltage
loss ``delta_V_nr_V = -(k_B * T / q) * ln(min(EQE_EL, 1))``.

Key N-class contracts (P05-03a):

1. **Refusal of invalid or unsupported inputs**: Missing tabulated ``nk``
   optical resources (Beer-Lambert-only configurations), graded/incoherent
   optics, ``V_inj_V <= 0``, fewer than 2 distinct probe wavelengths, out-of-range
   wavelengths, or non-injecting / non-converged dark DC states raise explicit
   ``ValueError``, ``CapabilityError``, or ``RuntimeError`` rather than
   fabricating zeros.
2. **Exact TMM layer absorptance by default**: ``absorptance_mode="exact_layer"``
   uses analytical Poynting-flux layer absorptance from ``PhotonBudget`` (with
   ``"node_trapezoid"`` available for spatial-grid quadrature comparisons).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Literal

import numpy as np
from scipy.integrate import trapezoid

from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.ionic_1d import IonicState1D
from solarlab.experiments.eqe import (
    build_wavelength_optical_binding,
    ensure_compilable_spectral_device,
)
from solarlab.experiments.suns_voc import (
    default_ionic_steady_options,
    solve_ionic_dc_operating_point,
)
from solarlab.physics.generation.optical_binding import PhotonBudget
from solarlab.physics.generation.spectrum import wavelength_grid_m
from solarlab.reproducibility.job_provenance import current_provenance
from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions
from solarlab.solvers.nonlinear.steady_1d import (
    SteadyOptions,
    SteadyStateError,
    solve_steady_state,
)

__all__ = [
    "ELOptions",
    "ELSpectrumResult",
    "blackbody_photon_flux_m2_s_m",
    "run_el_spectrum",
]

BOLTZMANN_J_K = 1.380649e-23
PLANCK_J_S = 6.62607015e-34
SPEED_OF_LIGHT_M_S = 299_792_458.0
ProgressCallback = Callable[[str, int, int, str], None]


def _ro_float(value: Any) -> np.ndarray:
    arr = np.array(value, dtype=np.float64, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class ELOptions:
    """Explicit numerical and optical controls for reciprocity EL spectra."""

    intervals_per_layer: int = 24
    continuation_steps: int = 12
    absorptance_mode: Literal["exact_layer", "node_trapezoid"] = "exact_layer"
    ion_domain_layer: str | None = None
    ion_domain_endpoints: tuple[str, str] = ("blocking", "blocking")
    steady: SteadyOptions = field(default_factory=SteadyOptions)
    ionic_steady: IonicSteadyOptions | None = None

    def __post_init__(self) -> None:
        if type(self.intervals_per_layer) is not int or self.intervals_per_layer < 2:
            raise ValueError("intervals_per_layer must be an integer >= 2")
        if type(self.continuation_steps) is not int or self.continuation_steps < 1:
            raise ValueError("continuation_steps must be an integer >= 1")
        if self.absorptance_mode not in ("exact_layer", "node_trapezoid"):
            raise ValueError(
                "absorptance_mode must be 'exact_layer' or 'node_trapezoid'"
            )
        if not isinstance(self.steady, SteadyOptions):
            raise ValueError("steady must be an explicit SteadyOptions instance")
        if self.ionic_steady is not None and not isinstance(
            self.ionic_steady, IonicSteadyOptions
        ):
            raise ValueError(
                "ionic_steady must be None or an explicit IonicSteadyOptions"
            )


@dataclass(frozen=True)
class ELSpectrumResult:
    """Forward-injection reciprocity EL spectrum and non-radiative V_oc loss."""

    wavelengths_nm: np.ndarray
    Phi_EL_m2_s_m: np.ndarray
    EL_spectrum_m2_s_nm: np.ndarray
    blackbody_flux_m2_s_m: np.ndarray
    absorber_absorptance: np.ndarray
    V_inj_V: float
    J_inj_pv_A_m2: float
    J_inj_conjugate_A_m2: float
    J_em_rad_A_m2: float
    EQE_EL: float
    delta_V_nr_V: float
    delta_V_nr_mV: float
    temperature_K: float
    photon_budget: PhotonBudget
    absorptance_mode: str
    optical_binding_sha256: str
    compiled_identity: str
    solver_controls: dict[str, Any]
    job_provenance: dict[str, Any] | None = field(default_factory=current_provenance)

    def __post_init__(self) -> None:
        for name in (
            "wavelengths_nm",
            "Phi_EL_m2_s_m",
            "EL_spectrum_m2_s_nm",
            "blackbody_flux_m2_s_m",
            "absorber_absorptance",
        ):
            object.__setattr__(self, name, _ro_float(getattr(self, name)))

    @property
    def EL_spectrum(self) -> np.ndarray:
        return self.EL_spectrum_m2_s_nm

    @property
    def V_inj(self) -> float:
        return self.V_inj_V

    @property
    def J_inj(self) -> float:
        return self.J_inj_pv_A_m2

    @property
    def J_em_rad(self) -> float:
        return self.J_em_rad_A_m2

    @property
    def T(self) -> float:
        return self.temperature_K


def blackbody_photon_flux_m2_s_m(
    wavelengths_m: Sequence[float] | np.ndarray, temperature_K: float
) -> np.ndarray:
    """Hemispherical Lambertian blackbody photon flux [photons m^-2 s^-1 m^-1]."""
    wl_m: np.ndarray = np.asarray(wavelengths_m, dtype=float)
    if wl_m.ndim != 1 or wl_m.size == 0 or np.any(~np.isfinite(wl_m)) or np.any(wl_m <= 0.0):
        raise ValueError("wavelengths_m must be a non-empty 1D positive finite array")
    if not math.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError("temperature_K must be finite and positive")

    x = PLANCK_J_S * SPEED_OF_LIGHT_M_S / (wl_m * BOLTZMANN_J_K * float(temperature_K))
    x_safe = np.clip(x, 0.0, 700.0)
    e_neg = np.exp(-x_safe)
    denom = 1.0 - e_neg
    denom = np.where(denom > 1e-300, denom, 1.0)
    return np.asarray(
        (2.0 * math.pi * SPEED_OF_LIGHT_M_S / (wl_m**4)) * (e_neg / denom),
        dtype=float,
    )


def _normalize_el_wavelengths(
    wavelengths_nm: Sequence[float] | np.ndarray | None,
) -> np.ndarray:
    if wavelengths_nm is None:
        return np.linspace(400.0, 1000.0, 25, dtype=float)
    raw = [float(w) for w in wavelengths_nm]
    if len(raw) < 2:
        raise ValueError("wavelengths_nm must have >= 2 distinct wavelengths for EL integration")
    if any(not math.isfinite(w) or w <= 0.0 for w in raw):
        raise ValueError(f"wavelengths_nm must be finite and positive, got {raw}")
    wl = np.array(sorted(set(raw)), dtype=float)
    if wl.size < 2:
        raise ValueError("wavelengths_nm must have >= 2 distinct wavelengths for EL integration")
    return wl


def _absorber_layer_indices(prepared: PreparedDevice) -> tuple[int, ...]:
    abs_idx = tuple(
        idx for idx, layer in enumerate(prepared.layers) if layer.role == "absorber"
    )
    if abs_idx:
        return abs_idx
    return tuple(
        idx for idx, layer in enumerate(prepared.layers) if layer.role != "substrate"
    )


def _absorber_node_mask(nodes_m: np.ndarray, prepared: PreparedDevice) -> np.ndarray:
    elec_layers = [layer for layer in prepared.layers if layer.role != "substrate"]
    if not any(layer.role == "absorber" for layer in elec_layers):
        return np.ones_like(nodes_m, dtype=bool)
    mask = np.zeros_like(nodes_m, dtype=bool)
    cursor = 0.0
    eps = 1e-15
    for layer in elec_layers:
        hi = cursor + layer.thickness
        if layer.role == "absorber":
            mask |= (nodes_m >= cursor - eps) & (nodes_m <= hi + eps)
        cursor = hi
    return mask


def _rebase_ionic_state(
    compiled: Any, state: IonicState1D, voltage_V: float
) -> IonicState1D:
    if state.voltage_V == voltage_V:
        return state
    coords = state.u.copy()
    n_elec = 3 * compiled.node_count
    coords[:n_elec] = compiled.electronic_component.rebase_coordinates(
        coords[:n_elec], state.voltage_V, voltage_V
    )
    return compiled.state(coords, voltage_V=voltage_V)


def run_el_spectrum(
    prepared: PreparedDevice,
    *,
    V_inj_V: float = 1.0,
    wavelengths_nm: Sequence[float] | np.ndarray | None = None,
    options: ELOptions | None = None,
    progress: ProgressCallback | None = None,
) -> ELSpectrumResult:
    """Compute forward-injection EL spectrum, EQE_EL, and non-radiative V_oc loss."""
    opts = ELOptions() if options is None else options
    v_inj = float(V_inj_V)
    if not math.isfinite(v_inj) or v_inj <= 0.0:
        raise ValueError(f"V_inj_V must be finite and positive, got {V_inj_V}")

    wl_nm = _normalize_el_wavelengths(wavelengths_nm)
    wl_m = wavelength_grid_m(wl_nm)
    binding, budget = build_wavelength_optical_binding(
        prepared, wl_nm, caller_name="run_el_spectrum"
    )
    if progress is not None:
        progress("el", 1, 3, "evaluated TMM optical stack")

    compilable = ensure_compilable_spectral_device(prepared, wl_nm)
    if opts.ion_domain_layer is not None:
        compiled_ion = compile_ionic_device_1d(
            compilable,
            intervals_per_layer=opts.intervals_per_layer,
            ion_domain_layer=opts.ion_domain_layer,
            ion_domain_endpoints=opts.ion_domain_endpoints,
        )
        elec = compiled_ion.electronic_component
        compiled_identity = compiled_ion.identity
    else:
        compiled_ion = None
        elec = compile_device_1d(
            compilable, intervals_per_layer=opts.intervals_per_layer
        )
        compiled_identity = elec.identity

    if opts.absorptance_mode == "exact_layer":
        abs_indices = _absorber_layer_indices(prepared)
        a_abs = np.clip(
            np.sum(budget.layer_absorptance[list(abs_indices), :], axis=0),
            0.0,
            1.0,
        )
    else:
        a_xl = binding.absorption_profile(elec.nodes_m)
        node_mask = _absorber_node_mask(elec.nodes_m, prepared)
        x_abs = elec.nodes_m[node_mask]
        if x_abs.size < 2:
            raise ValueError(
                "absorber node mask contains fewer than 2 nodes; increase intervals_per_layer"
            )
        a_abs = np.clip(
            np.asarray(trapezoid(a_xl[node_mask, :], x_abs, axis=0), dtype=float),
            0.0,
            1.0,
        )

    temperature_K = float(
        elec.thermal_voltage_V * ELEMENTARY_CHARGE_C / BOLTZMANN_J_K
    )
    v_thermal = float(elec.thermal_voltage_V)
    phi_bb = blackbody_photon_flux_m2_s_m(wl_m, temperature_K)
    boltzmann_factor = math.exp(min(v_inj / v_thermal, 700.0))
    phi_el = a_abs * phi_bb * boltzmann_factor
    el_per_nm = phi_el * 1e-9
    j_em_rad = float(ELEMENTARY_CHARGE_C * trapezoid(phi_el, wl_m))

    if progress is not None:
        progress("el", 2, 3, f"solving dark forward injection to V_inj={v_inj:.3f} V")

    polarity = float(elec.voltage_polarity)
    v_steps: np.ndarray = np.linspace(
        0.0, v_inj, opts.continuation_steps + 1, dtype=float
    )

    if compiled_ion is None:
        controls_record: dict[str, Any] = {
            "mode": "electronic_dc",
            "steady": asdict(opts.steady),
            "continuation_steps": opts.continuation_steps,
        }
        prev_state = None
        last_sol = None
        for v_step in v_steps:
            try:
                sol = solve_steady_state(
                    elec,
                    voltage_V=float(v_step),
                    generation_scale=0.0,
                    initial_state=prev_state,
                    options=opts.steady,
                )
            except SteadyStateError as exc:
                raise RuntimeError(
                    f"EL dark forward continuation failed at V={v_step:.4f} V: {exc}"
                ) from exc
            if not sol.converged:
                raise RuntimeError(
                    f"EL dark forward continuation failed to converge at V={v_step:.4f} V"
                )
            prev_state = sol.state
            last_sol = sol
        assert last_sol is not None
        faces = np.asarray(last_sol.current_density_A_m2, dtype=float)
        contacts = np.asarray(last_sol.contact_current_density_A_m2, dtype=float)
        j_inj_pv = float(-polarity * faces[0])
        j_inj_conj = float(polarity * contacts[0 if polarity > 0.0 else -1])
    else:
        ion_controls = (
            default_ionic_steady_options(
                compiled_ion.ion_node_count, electronic=opts.steady
            )
            if opts.ionic_steady is None
            else opts.ionic_steady
        )
        controls_record = {
            "mode": "ionic_dc",
            "ionic_steady": asdict(ion_controls),
            "continuation_steps": opts.continuation_steps,
        }
        prev_ion: IonicState1D | None = None
        last_ion = None
        for v_step in v_steps:
            sol_ion = solve_ionic_dc_operating_point(
                compiled_ion,
                voltage_V=float(v_step),
                generation_scale=0.0,
                options=ion_controls,
                base_state=prev_ion,
                base_generation_scale=0.0,
            )
            if (
                not sol_ion.accepted
                or sol_ion.state is None
                or sol_ion.diagnostics is None
            ):
                reason = sol_ion.error or sol_ion.termination
                raise RuntimeError(
                    f"EL ionic dark forward continuation failed at V={v_step:.4f} V ({reason})"
                )
            prev_ion = sol_ion.state
            last_ion = sol_ion
        assert last_ion is not None and last_ion.diagnostics is not None
        all_faces = np.asarray(
            last_ion.diagnostics["total_current_density_A_m2"], dtype=float
        )
        j_inj_pv = float(-polarity * all_faces[0])
        j_inj_conj = float(polarity * all_faces[0 if polarity > 0.0 else -1])

    if not math.isfinite(j_inj_conj) or j_inj_conj <= 1e-30:
        raise RuntimeError(
            f"run_el_spectrum requires positive forward dark injection current at "
            f"V_inj={v_inj} V, got J_inj_conjugate={j_inj_conj:.3e} A/m^2 "
            f"(J_inj_pv={j_inj_pv:.3e} A/m^2); refusing to fabricate zero EQE_EL."
        )

    eqe_el = float(j_em_rad / j_inj_conj)
    if eqe_el <= 0.0:
        raise RuntimeError(
            "run_el_spectrum computed zero radiative emission over the probe wavelength "
            "window (absorber_absorptance is identically zero); check wavelengths_nm."
        )
    eqe_el_clamped = min(max(eqe_el, 1e-300), 1.0)
    delta_v_nr_v = float(-v_thermal * math.log(eqe_el_clamped))

    if progress is not None:
        progress(
            "el",
            3,
            3,
            f"EQE_EL={eqe_el:.2e}, dV_nr={delta_v_nr_v * 1e3:.1f} mV",
        )

    return ELSpectrumResult(
        wavelengths_nm=wl_nm,
        Phi_EL_m2_s_m=phi_el,
        EL_spectrum_m2_s_nm=el_per_nm,
        blackbody_flux_m2_s_m=phi_bb,
        absorber_absorptance=a_abs,
        V_inj_V=v_inj,
        J_inj_pv_A_m2=j_inj_pv,
        J_inj_conjugate_A_m2=j_inj_conj,
        J_em_rad_A_m2=j_em_rad,
        EQE_EL=eqe_el,
        delta_V_nr_V=delta_v_nr_v,
        delta_V_nr_mV=float(delta_v_nr_v * 1e3),
        temperature_K=temperature_K,
        photon_budget=budget,
        absorptance_mode=opts.absorptance_mode,
        optical_binding_sha256=binding.sha256,
        compiled_identity=compiled_identity,
        solver_controls=controls_record,
    )
