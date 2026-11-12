"""Monochromatic perturbation EQE(lambda) on the unified 1D DC core.

Computes external quantum efficiency

    EQE(lambda) = (J_pv(lambda, V_bias) - J_pv_dark(V_bias)) / (q * Phi_inc(lambda))

around an accepted dark/bias DC operating point using the coherent transfer-matrix
optical stack (``OpticalStackBinding``) and photon-conserving control-volume
generation profiles (or exact linearized DC Jacobian tangent in
``perturbation_mode="linear_response"``).

Key N-class contracts (P05-03a):

1. **Explicit optical capability enforcement**: Devices lacking wavelength-resolved
   tabulated ``nk`` optical resources (e.g. scalar Beer-Lambert ``alpha``-only
   configurations), graded optics, incoherent layers, non-positive ``Phi_incident``,
   or probe wavelengths outside tabulated ``nk`` bounds raise ``CapabilityError``
   or ``ValueError`` before solver work rather than returning zeros or clamping.
2. **Sign-faithful EQE**: Never wraps ``J_k`` in ``abs()``; negative incremental
   photocurrent emits a ``RuntimeWarning`` and remains negative in ``EQE``.
3. **Closed photon budget & spectrum-bound J_sc_eqe**: Records per-wavelength
   ``PhotonBudget`` (verifying ``|R + T + sum(A) - 1| <= 1e-12``) and integrates
   ``J_sc_eqe_A_m2`` against an explicit or declared ``SpectralPhotonFlux``
   without silent edge extrapolation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from typing import Any, Literal, cast
import warnings

import numpy as np
from scipy.sparse.linalg import spsolve

from solarlab.config.resolve_device import resolve_device
from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.optical_generation import PHOTON_BUDGET_RTOL
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
from solarlab.equations.dc_1d import CapabilityError, CompiledDevice1D, DC1DProblem
from solarlab.equations.implementation import DCProblem, use_assembled
from solarlab.equations.ionic_1d import CompiledIonicDevice1D
from solarlab.experiments.suns_voc import (
    default_ionic_steady_options,
    solve_ionic_dc_operating_point,
)
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument as MaterialSourceDocument
from solarlab.physics.generation.optical_binding import (
    OpticalLayerSpec,
    OpticalStackBinding,
    PhotonBudget,
    constant_index_optical_constants,
)
from solarlab.physics.generation.profile import (
    GenerationProfile,
    control_volume_digest,
)
from solarlab.physics.generation.spectrum import (
    SpectralPhotonFlux,
    sample_nk_table,
    sample_tabulated,
    wavelength_grid_m,
)
from solarlab.physics.generation.transfer_matrix import tmm_cumulative_absorptance
from solarlab.reproducibility.job_provenance import current_provenance
from solarlab.reproducibility.sources import SourceDocument
from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions
from solarlab.solvers.nonlinear.steady_1d import (
    SteadyOptions,
    SteadyStateError,
    solve_steady_state,
)

__all__ = [
    "EQEOptions",
    "EQEResult",
    "build_wavelength_optical_binding",
    "compute_eqe",
    "ensure_compilable_spectral_device",
]

ProgressCallback = Callable[[str, int, int, str], None]


def _ro_float(value: Any) -> np.ndarray:
    arr = np.array(value, dtype=np.float64, copy=True)
    arr.setflags(write=False)
    return arr


def _ro_bool(value: Any) -> np.ndarray:
    arr = np.array(value, dtype=np.bool_, copy=True)
    arr.setflags(write=False)
    return arr


def _dc_problem(
    device: CompiledDevice1D, voltage_V: float, generation_scale: float
) -> DCProblem:
    if use_assembled():
        return AssembledDC1DProblem(device, voltage_V, generation_scale)
    return DC1DProblem(device, voltage_V, generation_scale)


@dataclass(frozen=True)
class EQEOptions:
    """Explicit optical, bias, and solver controls for monochromatic EQE."""

    intervals_per_layer: int = 24
    bias_voltage_V: float = 0.0
    perturbation_mode: Literal["finite_flux", "linear_response"] = "finite_flux"
    generation_discretization: Literal["cell_flux", "node_profile"] = "cell_flux"
    ion_domain_layer: str | None = None
    ion_domain_endpoints: tuple[str, str] = ("blocking", "blocking")
    steady: SteadyOptions = field(default_factory=SteadyOptions)
    ionic_steady: IonicSteadyOptions | None = None
    require_all_converged: bool = True
    require_reference_spectrum: bool = False

    def __post_init__(self) -> None:
        if type(self.intervals_per_layer) is not int or self.intervals_per_layer < 2:
            raise ValueError("intervals_per_layer must be an integer >= 2")
        if not math.isfinite(self.bias_voltage_V):
            raise ValueError("bias_voltage_V must be finite")
        if self.perturbation_mode not in ("finite_flux", "linear_response"):
            raise ValueError(
                "perturbation_mode must be 'finite_flux' or 'linear_response'"
            )
        if self.generation_discretization not in ("cell_flux", "node_profile"):
            raise ValueError(
                "generation_discretization must be 'cell_flux' or 'node_profile'"
            )
        if (
            self.ion_domain_layer is not None
            and self.perturbation_mode == "linear_response"
        ):
            raise ValueError(
                "linear_response EQE currently supports ion-free devices; use finite_flux for ionic"
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
class EQEResult:
    """Monochromatic EQE(lambda), IQE(lambda), photon budget, and integrated J_sc."""

    wavelengths_nm: np.ndarray
    EQE: np.ndarray
    IQE: np.ndarray
    electrical_absorptance: np.ndarray
    J_photo_per_lambda_A_m2: np.ndarray
    dark_bias_current_A_m2: float
    Phi_incident_m2_s: float
    bias_voltage_V: float
    converged_mask: np.ndarray
    point_reasons: tuple[str | None, ...]
    photon_budget: PhotonBudget
    J_sc_eqe_A_m2: float
    quadrature_weights_m: np.ndarray | None
    reference_spectrum_id: str | None
    integration_failure_reason: str | None
    max_current_spread_A_m2: float
    perturbation_mode: str
    generation_discretization: str
    optical_binding_sha256: str
    compiled_identity: str
    solver_controls: dict[str, Any]
    job_provenance: dict[str, Any] | None = field(default_factory=current_provenance)

    def __post_init__(self) -> None:
        for name in (
            "wavelengths_nm",
            "EQE",
            "IQE",
            "electrical_absorptance",
            "J_photo_per_lambda_A_m2",
        ):
            object.__setattr__(self, name, _ro_float(getattr(self, name)))
        object.__setattr__(self, "converged_mask", _ro_bool(self.converged_mask))
        if self.quadrature_weights_m is not None:
            object.__setattr__(
                self, "quadrature_weights_m", _ro_float(self.quadrature_weights_m)
            )

    @property
    def J_sc_per_lambda(self) -> np.ndarray:
        return self.J_photo_per_lambda_A_m2

    @property
    def J_sc_integrated(self) -> float:
        return self.J_sc_eqe_A_m2

    @property
    def Phi_incident(self) -> float:
        return self.Phi_incident_m2_s

    @property
    def J_spread_max(self) -> float:
        return self.max_current_spread_A_m2


def _normalize_probe_wavelengths(
    wavelengths_nm: Sequence[float] | np.ndarray | None,
    *,
    default_lo_nm: float = 300.0,
    default_hi_nm: float = 1000.0,
    default_count: int = 29,
) -> np.ndarray:
    if wavelengths_nm is None:
        return np.linspace(default_lo_nm, default_hi_nm, default_count, dtype=float)
    raw = [float(w) for w in wavelengths_nm]
    if not raw:
        raise ValueError("wavelengths_nm must be non-empty")
    if any(not math.isfinite(w) or w <= 0.0 for w in raw):
        raise ValueError(f"wavelengths_nm must be finite and positive, got {raw}")
    unique_sorted = np.array(sorted(set(raw)), dtype=float)
    return unique_sorted


def build_wavelength_optical_binding(
    prepared: PreparedDevice,
    wavelengths_nm: np.ndarray,
    *,
    caller_name: str = "compute_eqe",
) -> tuple[OpticalStackBinding, PhotonBudget]:
    """Validate wavelength-resolved optics on ``prepared`` and build TMM binding."""
    if not isinstance(prepared, PreparedDevice):
        raise TypeError("prepared must be a PreparedDevice instance")
    data = prepared.to_mapping()
    if data["settings"]["graded_optics"]:
        raise CapabilityError(
            f"{caller_name}: graded_optics is not bound in this 1D coherent TMM slice"
        )
    has_tabulated_nk = any(layer.optical is not None for layer in prepared.layers)
    if not has_tabulated_nk:
        raise ValueError(
            f"{caller_name} requires at least one layer with tabulated wavelength-resolved "
            f"optical_material (n, k) data; scalar Beer-Lambert alpha-only stacks are refused."
        )

    probe_flux = np.ones_like(wavelengths_nm, dtype=float)
    if wavelengths_nm.size >= 2:
        spec_wl = wavelengths_nm
        spec_flux = probe_flux
    else:
        # SpectralPhotonFlux requires >= 2 grid points; pad a tiny epsilon twin for
        # 1-wavelength queries and slice [0:1] when reading per-wavelength quantities.
        w0 = float(wavelengths_nm[0])
        spec_wl = np.array([w0, w0 * (1.0 + 1e-12)], dtype=float)
        spec_flux = np.ones(2, dtype=float)

    spectrum = SpectralPhotonFlux(spec_wl, spec_flux, f"{caller_name}_unit_probe")
    deps: dict[str, str] = {
        "binding_convention": f"{caller_name}_coherent_tmm_1d_v1",
        "prepared_device_sha256": prepared.content_sha256,
        "temperature_K": str(data["settings"]["T"]),
    }

    specs: list[OpticalLayerSpec] = []
    for layer in prepared.layers:
        params = dict(layer.parameters)
        if params["incoherent"]:
            raise CapabilityError(
                f"{caller_name}: layer {layer.id!r} sets incoherent=True; "
                f"only coherent TMM stacks are bound in this slice"
            )
        if layer.cigs_graded_optics_json is not None:
            raise CapabilityError(
                f"{caller_name}: layer {layer.id!r} uses unsupported graded optics"
            )
        if layer.optical is not None:
            if params["n_optical"] is not None:
                raise CapabilityError(
                    f"{caller_name}: layer {layer.id!r} cannot declare both optical_material and n_optical"
                )
            native = layer.optical.array
            n_arr, k_arr = sample_nk_table(
                native[:, 0] * 1e9,
                native[:, 1],
                native[:, 2],
                spec_wl,
                material=layer.optical.name,
            )
            deps[f"nk_resource:{layer.id}"] = layer.optical.content_sha256
        else:
            n_opt = params["n_optical"]
            eps_r = cast(float, params["eps_r"])
            n_arr, k_arr = constant_index_optical_constants(
                spec_wl,
                absorption_coefficient_m1=cast(float, params["alpha"]),
                refractive_index=None if n_opt is None else float(n_opt),
                relative_permittivity=eps_r,
            )
        specs.append(
            OpticalLayerSpec(
                name=layer.id,
                thickness_m=layer.thickness,
                n=n_arr,
                k=k_arr,
                incoherent=False,
                electrical=layer.role != "substrate",
            )
        )

    binding = OpticalStackBinding(
        tuple(specs),
        spectrum,
        ambient_index=1.0,
        exit_index=1.0,
        incidence="front",
        dependencies=tuple(deps.items()),
    )
    budget = binding.photon_budget()
    if np.max(np.abs(budget.spectral_closure_residual())) > PHOTON_BUDGET_RTOL:
        raise CapabilityError(
            f"{caller_name}: TMM spectral photon budget failed closure check"
        )
    return binding, budget


def ensure_compilable_spectral_device(
    prepared: PreparedDevice, wavelengths_nm: np.ndarray
) -> PreparedDevice:
    """Bind a zero-flux placeholder spectrum over ``wavelengths_nm`` for dark/mono compilation.

    Full-spectrum reference tables such as ASTM AM1.5G (280-1500 nm) can extend
    beyond a layer's tabulated ``nk`` window (e.g. MAPbI3 300-1100 nm) even when
    the requested monochromatic probe ``wavelengths_nm`` lie strictly inside all
    ``nk`` bounds. Binding a zero-flux spectrum over ``[min(wl), max(wl)]`` keeps
    ``compile_device_1d`` within validated ``nk`` bounds without touching
    ``prepared``'s declared reference spectrum.
    """
    lo_nm = float(np.min(wavelengths_nm))
    hi_nm = float(np.max(wavelengths_nm))
    if hi_nm <= lo_nm:
        hi_nm = lo_nm + 1.0
    csv_bytes = (
        "wavelength_nm,spectral_flux_m-2_s-1_m-1\n"
        f"{lo_nm:.6f},0.0\n"
        f"{hi_nm:.6f},0.0\n"
    ).encode("utf-8")
    spec_name = "__solarlab_eqe_zero_flux_placeholder__"
    spec_table = ResourceTable(
        spec_name,
        "spectrum",
        MaterialSourceDocument("eqe.zero_flux_placeholder.csv", csv_bytes),
    )
    merged_tables = [
        table for table in prepared.resources.tables if table.name != spec_name
    ]
    merged_tables.append(spec_table)
    lib = ResourceLibrary(tuple(merged_tables))

    editing = prepared.to_input().editing_data()
    editing["spectrum"] = spec_name
    editing["fixed_generation"] = None
    settings = dict(editing.get("settings") or {})
    settings["Phi"] = 0.0
    editing["settings"] = settings
    src = SourceDocument(
        "eqe.bind_zero_flux_spectrum",
        json.dumps(
            editing, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8"),
    )
    return resolve_device(
        DeviceInput.model_validate(editing),
        prepared.defaults,
        lib,
        sources=(*prepared.sources, src),
    )


def _resolve_reference_spectrum(
    prepared: PreparedDevice,
    reference_spectrum: SpectralPhotonFlux | None,
) -> SpectralPhotonFlux | None:
    if reference_spectrum is not None:
        if not isinstance(reference_spectrum, SpectralPhotonFlux):
            raise TypeError(
                "reference_spectrum must be None or a SpectralPhotonFlux instance"
            )
        return reference_spectrum
    spec_name = prepared.to_mapping()["declaration"].get("spectrum")
    if spec_name is None:
        return None
    table = prepared.resources.get(spec_name, "spectrum")
    if table.units != ("m", "m^-3/s"):
        return None
    return SpectralPhotonFlux(
        table.array[:, 0] * 1e9, table.array[:, 1], table.source.id
    )


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    w = np.zeros_like(x, dtype=float)
    if x.size < 2:
        return w
    dx = np.diff(x)
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    if x.size > 2:
        w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return w


def _monochromatic_device(
    base_compiled: CompiledDevice1D,
    source_m3_s: np.ndarray,
    absorbed_m2_s: np.ndarray,
    optical_binding_sha256: str,
    lam_nm: float,
    phi_inc: float,
) -> CompiledDevice1D:
    cv_digest = control_volume_digest(base_compiled.geometry)
    mono_bind_sha = hashlib.sha256(
        f"{optical_binding_sha256}|lam_nm={lam_nm!r}|phi={phi_inc!r}".encode("ascii")
    ).hexdigest()
    profile = GenerationProfile(
        method="transfer_matrix_monochromatic",
        geometry_name=base_compiled.geometry.name,
        source_m3_s=source_m3_s,
        absorbed_m2_s=absorbed_m2_s,
        binding_sha256=mono_bind_sha,
        control_volume_sha256=cv_digest,
    )
    dev_sha = hashlib.sha256(
        f"{base_compiled.sha256}|mono={mono_bind_sha}".encode("ascii")
    ).hexdigest()
    return replace(base_compiled, generation=profile, sha256=dev_sha)


def _monochromatic_ionic_device(
    base_ionic: CompiledIonicDevice1D,
    source_m3_s: np.ndarray,
    absorbed_m2_s: np.ndarray,
    optical_binding_sha256: str,
    lam_nm: float,
    phi_inc: float,
) -> CompiledIonicDevice1D:
    mono_elec = _monochromatic_device(
        base_ionic.electronic_component,
        source_m3_s,
        absorbed_m2_s,
        optical_binding_sha256,
        lam_nm,
        phi_inc,
    )
    manifest = dict(base_ionic.binding_manifest)
    comp_bind = dict(cast(dict[str, Any], manifest["electronic_component"]))
    comp_bind["compiled_sha256"] = mono_elec.identity
    manifest["electronic_component"] = comp_bind
    manifest_json = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    ion_sha = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    return replace(
        base_ionic,
        electronic_component=mono_elec,
        binding_manifest_json=manifest_json,
        sha256=ion_sha,
    )


def _slice_single_point_budget(budget: PhotonBudget, count: int) -> PhotonBudget:
    if budget.wavelengths_nm.size == count:
        return budget
    return PhotonBudget(
        wavelengths_nm=budget.wavelengths_nm[:count],
        incident_spectral_m2_s_m=budget.incident_spectral_m2_s_m[:count],
        reflectance=budget.reflectance[:count],
        transmittance=budget.transmittance[:count],
        layer_absorptance=budget.layer_absorptance[:, :count],
        dropped_multipass=budget.dropped_multipass[:count],
        layer_names=budget.layer_names,
        layer_electrical=budget.layer_electrical,
    )


def compute_eqe(
    prepared: PreparedDevice,
    wavelengths_nm: Sequence[float] | np.ndarray | None = None,
    *,
    Phi_incident: float = 4e21,
    reference_spectrum: SpectralPhotonFlux | None = None,
    options: EQEOptions | None = None,
    progress: ProgressCallback | None = None,
) -> EQEResult:
    """Compute monochromatic EQE(lambda), photon budget, and integrated J_sc_eqe."""
    opts = EQEOptions() if options is None else options
    phi_inc = float(Phi_incident)
    if not math.isfinite(phi_inc) or phi_inc <= 0.0:
        raise ValueError(f"Phi_incident must be finite and positive, got {Phi_incident}")

    wl_nm = _normalize_probe_wavelengths(wavelengths_nm)
    n_wl = int(wl_nm.size)

    binding, raw_budget = build_wavelength_optical_binding(
        prepared, wl_nm, caller_name="compute_eqe"
    )
    budget = _slice_single_point_budget(raw_budget, n_wl)
    elec_mask: np.ndarray = np.asarray(budget.layer_electrical, dtype=bool)
    elec_abs: np.ndarray = np.asarray(
        np.sum(budget.layer_absorptance[elec_mask, :n_wl], axis=0), dtype=float
    )

    compilable = ensure_compilable_spectral_device(prepared, wl_nm)
    if opts.ion_domain_layer is not None:
        base_ionic = compile_ionic_device_1d(
            compilable,
            intervals_per_layer=opts.intervals_per_layer,
            ion_domain_layer=opts.ion_domain_layer,
            ion_domain_endpoints=opts.ion_domain_endpoints,
        )
        base_elec = base_ionic.electronic_component
        compiled_identity = base_ionic.identity
    else:
        base_ionic = None
        base_elec = compile_device_1d(
            compilable, intervals_per_layer=opts.intervals_per_layer
        )
        compiled_identity = base_elec.identity

    measures = np.asarray(base_elec.geometry.cell_measures_m, dtype=float)
    faces_opt = (
        np.asarray(base_elec.geometry.cell_faces_m, dtype=float)
        + binding.electrical_offset_m
    )
    if opts.generation_discretization == "cell_flux":
        cum_abs = tmm_cumulative_absorptance(
            list(binding.tmm_layers),
            binding.wavelengths_m,
            faces_opt,
            binding.boundaries_m,
            binding.ambient_index,
            binding.exit_index,
        )[:, :n_wl]
        cell_abs = np.maximum(0.0, np.diff(cum_abs, axis=0))
        absorbed_all = cell_abs * phi_inc
        source_all = absorbed_all / measures[:, None]
    else:
        a_xl = np.maximum(
            0.0, binding.absorption_profile(base_elec.nodes_m)[:, :n_wl]
        )
        source_all = a_xl * phi_inc
        absorbed_all = source_all * measures[:, None]

    polarity = float(base_elec.voltage_polarity)
    eqe_arr: np.ndarray = np.full(n_wl, np.nan, dtype=float)
    iqe_arr: np.ndarray = np.full(n_wl, np.nan, dtype=float)
    j_photo_arr: np.ndarray = np.full(n_wl, np.nan, dtype=float)
    conv_mask: np.ndarray = np.zeros(n_wl, dtype=bool)
    reasons: list[str | None] = [None] * n_wl
    spread_max = 0.0

    if base_ionic is None:
        controls_record: dict[str, Any] = {
            "mode": "electronic_dc",
            "perturbation_mode": opts.perturbation_mode,
            "steady": asdict(opts.steady),
        }
        dark_sol = solve_steady_state(
            base_elec,
            voltage_V=opts.bias_voltage_V,
            generation_scale=0.0,
            options=opts.steady,
        )
        if not dark_sol.converged:
            raise RuntimeError(
                f"EQE dark baseline DC solve failed at V={opts.bias_voltage_V} V"
            )
        dark_faces = np.asarray(dark_sol.current_density_A_m2, dtype=float)
        j_dark = float(-polarity * dark_faces[0])
        spread_max = max(spread_max, float(np.ptp(dark_faces)))

        if opts.perturbation_mode == "linear_response":
            prob_dark = _dc_problem(base_elec, opts.bias_voltage_V, 0.0)
            u_dark = dark_sol.state.u
            jac_dark = prob_dark.jacobian(u_dark)
            f_dark = prob_dark.residual(u_dark)
            for k in range(n_wl):
                mono_dev = _monochromatic_device(
                    base_elec,
                    source_all[:, k],
                    absorbed_all[:, k],
                    binding.sha256,
                    float(wl_nm[k]),
                    phi_inc,
                )
                prob_mono = _dc_problem(mono_dev, opts.bias_voltage_V, 1.0)
                b_gen = prob_mono.residual(u_dark) - f_dark
                delta_u = np.asarray(spsolve(jac_dark, -b_gen), dtype=float)
                if not np.all(np.isfinite(delta_u)):
                    reasons[k] = "linear_response_nonfinite_tangent"
                    continue
                u_norm = max(float(np.linalg.norm(delta_u, ord=np.inf)), 1e-30)
                eps = 1e-7 / u_norm
                j_plus = prob_dark.current_density(u_dark + eps * delta_u)
                j_minus = prob_dark.current_density(u_dark - eps * delta_u)
                dj_faces = (j_plus - j_minus) / (2.0 * eps)
                spread_max = max(spread_max, float(np.ptp(dj_faces)))
                dj_pv = float(-polarity * dj_faces[0])
                j_photo_arr[k] = dj_pv
                eqe_arr[k] = dj_pv / (ELEMENTARY_CHARGE_C * phi_inc)
                iqe_arr[k] = (
                    eqe_arr[k] / elec_abs[k] if elec_abs[k] > 1e-12 else float("nan")
                )
                conv_mask[k] = True
                if progress is not None:
                    progress(
                        "eqe",
                        k + 1,
                        n_wl,
                        f"lambda={wl_nm[k]:.0f} nm, EQE={eqe_arr[k]:.3f}",
                    )
        else:
            for k in range(n_wl):
                mono_dev = _monochromatic_device(
                    base_elec,
                    source_all[:, k],
                    absorbed_all[:, k],
                    binding.sha256,
                    float(wl_nm[k]),
                    phi_inc,
                )
                rebased_seed = mono_dev.state(
                    dark_sol.state.u, voltage_V=opts.bias_voltage_V
                )
                has_gen = bool(np.max(source_all[:, k]) > 0.0)
                try:
                    if has_gen:
                        try:
                            sol_k = solve_steady_state(
                                mono_dev,
                                voltage_V=opts.bias_voltage_V,
                                generation_scale=1.0,
                                options=opts.steady,
                            )
                        except ValueError:
                            sol_k = solve_steady_state(
                                mono_dev,
                                voltage_V=opts.bias_voltage_V,
                                generation_scale=1.0,
                                initial_state=rebased_seed,
                                options=opts.steady,
                            )
                    else:
                        sol_k = solve_steady_state(
                            mono_dev,
                            voltage_V=opts.bias_voltage_V,
                            generation_scale=1.0,
                            initial_state=rebased_seed,
                            options=opts.steady,
                        )
                    if not sol_k.converged:
                        reasons[k] = "monochromatic_dc_not_converged"
                        continue
                except SteadyStateError as exc:
                    reasons[k] = f"monochromatic_dc_not_converged:{exc}"
                    continue

                faces_k = np.asarray(sol_k.current_density_A_m2, dtype=float)
                spread_max = max(spread_max, float(np.ptp(faces_k)))
                j_pv_k = float(-polarity * faces_k[0])
                dj_pv = j_pv_k - j_dark
                j_photo_arr[k] = dj_pv
                eqe_arr[k] = dj_pv / (ELEMENTARY_CHARGE_C * phi_inc)
                iqe_arr[k] = (
                    eqe_arr[k] / elec_abs[k] if elec_abs[k] > 1e-12 else float("nan")
                )
                conv_mask[k] = True
                if progress is not None:
                    progress(
                        "eqe",
                        k + 1,
                        n_wl,
                        f"lambda={wl_nm[k]:.0f} nm, EQE={eqe_arr[k]:.3f}",
                    )
    else:
        ion_controls = (
            default_ionic_steady_options(
                base_ionic.ion_node_count, electronic=opts.steady
            )
            if opts.ionic_steady is None
            else opts.ionic_steady
        )
        controls_record = {
            "mode": "ionic_dc",
            "perturbation_mode": opts.perturbation_mode,
            "ionic_steady": asdict(ion_controls),
        }
        dark_ion = solve_ionic_dc_operating_point(
            base_ionic,
            voltage_V=opts.bias_voltage_V,
            generation_scale=0.0,
            options=ion_controls,
        )
        if (
            not dark_ion.accepted
            or dark_ion.state is None
            or dark_ion.diagnostics is None
        ):
            reason = dark_ion.error or dark_ion.termination
            raise RuntimeError(
                f"EQE ionic dark baseline DC solve failed at V={opts.bias_voltage_V} V ({reason})"
            )
        dark_all_faces = np.asarray(
            dark_ion.diagnostics["total_current_density_A_m2"], dtype=float
        )
        j_dark = float(-polarity * dark_all_faces[0])
        spread_max = max(spread_max, float(np.ptp(dark_all_faces)))

        for k in range(n_wl):
            mono_ion = _monochromatic_ionic_device(
                base_ionic,
                source_all[:, k],
                absorbed_all[:, k],
                binding.sha256,
                float(wl_nm[k]),
                phi_inc,
            )
            seed_ion = mono_ion.state(
                dark_ion.state.u, voltage_V=opts.bias_voltage_V
            )
            sol_ion_k = solve_ionic_dc_operating_point(
                mono_ion,
                voltage_V=opts.bias_voltage_V,
                generation_scale=1.0,
                options=ion_controls,
                base_state=seed_ion,
                base_generation_scale=0.0,
            )
            if (
                not sol_ion_k.accepted
                or sol_ion_k.state is None
                or sol_ion_k.diagnostics is None
            ):
                reason = sol_ion_k.error or sol_ion_k.termination
                reasons[k] = f"monochromatic_dc_not_converged:{reason}"
                continue
            faces_ion_k = np.asarray(
                sol_ion_k.diagnostics["total_current_density_A_m2"], dtype=float
            )
            spread_max = max(spread_max, float(np.ptp(faces_ion_k)))
            dj_pv = float(-polarity * faces_ion_k[0]) - j_dark
            j_photo_arr[k] = dj_pv
            eqe_arr[k] = dj_pv / (ELEMENTARY_CHARGE_C * phi_inc)
            iqe_arr[k] = (
                eqe_arr[k] / elec_abs[k] if elec_abs[k] > 1e-12 else float("nan")
            )
            conv_mask[k] = True
            if progress is not None:
                progress(
                    "eqe",
                    k + 1,
                    n_wl,
                    f"lambda={wl_nm[k]:.0f} nm, EQE={eqe_arr[k]:.3f}",
                )

    if opts.require_all_converged and not bool(np.all(conv_mask)):
        bad = [
            f"{wl_nm[i]:.1f}nm ({reasons[i]})"
            for i in range(n_wl)
            if not conv_mask[i]
        ]
        raise RuntimeError(
            "compute_eqe failed to converge at probe wavelengths: " + ", ".join(bad)
        )

    neg_wl = wl_nm[conv_mask & (eqe_arr < -1e-9)]
    if neg_wl.size > 0:
        preview = ", ".join(f"{w:.0f}" for w in neg_wl[:5])
        suffix = " ..." if neg_wl.size > 5 else ""
        warnings.warn(
            f"Negative EQE at {neg_wl.size} wavelength(s) ({preview}{suffix} nm): "
            f"incremental photocurrent flows opposite to the PV output convention.",
            RuntimeWarning,
            stacklevel=2,
        )

    ref_spec = _resolve_reference_spectrum(prepared, reference_spectrum)
    jsc_int = float("nan")
    quad_weights: np.ndarray | None = None
    ref_id: str | None = None
    int_reason: str | None = None

    if ref_spec is None:
        int_reason = "no_reference_spectrum_declared"
        if opts.require_reference_spectrum:
            raise ValueError(
                "compute_eqe: require_reference_spectrum=True, but no reference_spectrum "
                "or declared PreparedDevice spectrum resource was provided"
            )
    elif n_wl < 2:
        ref_id = ref_spec.source_id
        int_reason = "fewer_than_two_wavelengths_for_quadrature"
        if opts.require_reference_spectrum:
            raise ValueError(
                "compute_eqe requires >= 2 probe wavelengths to integrate J_sc_eqe"
            )
    elif not bool(np.all(conv_mask)):
        ref_id = ref_spec.source_id
        int_reason = "incomplete_eqe_wavelength_convergence"
    else:
        ref_id = ref_spec.source_id
        try:
            phi_ref = sample_tabulated(
                ref_spec.wavelengths_nm,
                ref_spec.flux_m2_s_m,
                wl_nm,
                name=ref_spec.source_id,
            )
        except ValueError as exc:
            int_reason = f"wavelengths_outside_reference_spectrum:{exc}"
            if opts.require_reference_spectrum:
                raise
        else:
            wl_m = wavelength_grid_m(wl_nm)
            quad_weights = _trapezoid_weights(wl_m)
            jsc_int = float(
                ELEMENTARY_CHARGE_C * np.sum(eqe_arr * phi_ref * quad_weights)
            )

    return EQEResult(
        wavelengths_nm=wl_nm,
        EQE=eqe_arr,
        IQE=iqe_arr,
        electrical_absorptance=elec_abs,
        J_photo_per_lambda_A_m2=j_photo_arr,
        dark_bias_current_A_m2=j_dark,
        Phi_incident_m2_s=phi_inc,
        bias_voltage_V=opts.bias_voltage_V,
        converged_mask=conv_mask,
        point_reasons=tuple(reasons),
        photon_budget=budget,
        J_sc_eqe_A_m2=jsc_int,
        quadrature_weights_m=quad_weights,
        reference_spectrum_id=ref_id,
        integration_failure_reason=int_reason,
        max_current_spread_A_m2=float(spread_max),
        perturbation_mode=opts.perturbation_mode,
        generation_discretization=opts.generation_discretization,
        optical_binding_sha256=binding.sha256,
        compiled_identity=compiled_identity,
        solver_controls=controls_record,
    )
