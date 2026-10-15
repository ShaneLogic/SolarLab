"""Parameter binding: one cold compile of the resolved layer parameters at (T, composition).

``compile_layer_parameters`` evaluates, node by node, every temperature-,
composition- and doping-dependent material parameter the historical
material build derives (``solver/mol.build_material_arrays``), in the same
order and with the same expressions, so its arrays are bit-identical to the
historical ones (M class):

    doping profile -> Varshni gap -> grading -> band-gap narrowing
    -> mobility/diffusivity -> intrinsic product -> SRH references
    -> radiative coefficient -> band-edge DOS -> thermal velocity

then keeps the physical band edges and hands them to W3a's
``fold_dos_into_transport_bands``; the result's band edges are a W3a
``BandStructure`` (physical and DOS-folded transport potentials, never one
in place of the other).

A binding is immutable and state free. ``rebind`` always recompiles from the
stored 300 K inputs; it never derives the new arrays from the old ones, so a
binding can never carry stale coefficients into a new temperature or
composition. ``require`` rejects a binding offered for a different request.
Explicit temperature-input derivatives for an externally driven T(t) belong
to the time domain (P04-B): :data:`TIME_DOMAIN_TEMPERATURE_QUALIFIED` is
False and :func:`require_time_domain_temperature` refuses.

Interface switch point: layer inputs are explicit resolved SI values; the
WS2 configuration layer maps declared materials onto ``LayerParameterInput``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math

import numpy as np
from numpy.typing import ArrayLike

from solarlab.mesh.grid import RIGHT_LAYER_WINS, layer_node_masks
from solarlab.physics.band_structure.doping_profile import DopingProfileSpec, layer_doping_profiles
from solarlab.physics.band_structure.grading import (
    GradingSpec,
    affinity_profile,
    band_gap_profile,
    grade_n1_p1,
    grade_ni_sq,
)
from solarlab.physics.band_structure.narrowing import (
    BandGapNarrowingSpec,
    intrinsic_product_narrowing_scale,
    srh_reference_narrowing_scale,
)
from solarlab.physics.band_structure.temperature import (
    TEMPERATURE_LAWS,
    band_edge_dos_at_temperature,
    band_gap_at_temperature,
    device_temperature,
    intrinsic_density_at_temperature,
    ion_diffusivity_at_temperature,
    mobility_at_temperature,
    radiative_coefficient_at_temperature,
    srh_reference_scale,
    thermal_velocity_at_temperature,
)
from solarlab.physics.statistics.band_potentials import (
    BandStructure,
    dos_fold_reference_layer,
    fold_dos_into_transport_bands,
)

__all__ = [
    "LayerParameterInput",
    "LayerTemperatureState",
    "ParameterBinding",
    "StaleParameterBindingError",
    "TIME_DOMAIN_TEMPERATURE_QUALIFIED",
    "TemperatureCapabilityError",
    "TemperatureLawSpec",
    "compile_layer_parameters",
    "require_time_domain_temperature",
]

#: Explicit T(t) input/storage derivatives are declared, not qualified (P04-B).
TIME_DOMAIN_TEMPERATURE_QUALIFIED = False


class StaleParameterBindingError(ValueError):
    """A parameter binding was offered for a request it was not compiled for."""


class TemperatureCapabilityError(ValueError):
    """A temperature capability that is declared but not available."""


def require_time_domain_temperature() -> None:
    """Refuse externally driven T(t) in a time-domain residual until P04-B."""

    raise TemperatureCapabilityError(
        "pending_p04b: explicit temperature-input and storage derivatives for an "
        "externally driven T(t) are declared but not time-domain qualified"
    )


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


@dataclass(frozen=True, slots=True)
class TemperatureLawSpec:
    """Exponents and coefficients of one layer's temperature laws."""

    mobility_exponent: float = -1.5
    ion_activation_energy_eV: float = 0.58
    radiative_exponent: float = 0.0
    varshni_alpha_eV_K: float = 0.0
    varshni_beta_K: float = 0.0

    def __post_init__(self) -> None:
        for name in ("mobility_exponent", "ion_activation_energy_eV", "radiative_exponent",
                     "varshni_alpha_eV_K", "varshni_beta_K"):
            _finite(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class LayerParameterInput:
    """One electrical layer's resolved 300 K parameters (SI; energies in eV)."""

    name: str
    role: str
    thickness_m: float
    electron_affinity_eV: float
    band_gap_eV: float
    intrinsic_density_m3: float
    srh_n1_m3: float
    srh_p1_m3: float
    electron_mobility_m2_V_s: float
    hole_mobility_m2_V_s: float
    radiative_coefficient_m3_s: float
    doping: DopingProfileSpec
    thermal_velocity_m_s: float = 1.0e5
    ion_diffusivity_m2_s: float = 0.0
    negative_ion_diffusivity_m2_s: float = 0.0
    conduction_dos_300_m3: float | None = None
    valence_dos_300_m3: float | None = None
    temperature_laws: TemperatureLawSpec = field(default_factory=TemperatureLawSpec)
    grading: GradingSpec = field(default_factory=GradingSpec)
    narrowing: BandGapNarrowingSpec = field(default_factory=BandGapNarrowingSpec)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("layer requires a name")
        if _finite(self.thickness_m, "thickness_m") <= 0.0:
            raise ValueError("thickness_m must be positive")
        for name in ("electron_affinity_eV", "band_gap_eV", "intrinsic_density_m3", "srh_n1_m3",
                     "srh_p1_m3", "electron_mobility_m2_V_s", "hole_mobility_m2_V_s",
                     "radiative_coefficient_m3_s", "thermal_velocity_m_s", "ion_diffusivity_m2_s",
                     "negative_ion_diffusivity_m2_s"):
            _finite(getattr(self, name), name)
        for name in ("conduction_dos_300_m3", "valence_dos_300_m3"):
            value = getattr(self, name)
            if value is not None and _finite(value, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name, kind in (("doping", DopingProfileSpec), ("temperature_laws", TemperatureLawSpec),
                           ("grading", GradingSpec), ("narrowing", BandGapNarrowingSpec)):
            if not isinstance(getattr(self, name), kind):
                raise TypeError(f"{name} must be a {kind.__name__}")


@dataclass(frozen=True, slots=True)
class LayerTemperatureState:
    """Per-layer scalars the compile derived at the binding temperature."""

    band_gap_front_eV: float
    band_gap_back_eV: float | None
    intrinsic_density_m3: float
    srh_reference_scale: float
    electron_mobility_m2_V_s: float
    hole_mobility_m2_V_s: float
    graded: bool
    narrowed: bool


def _readonly(array: np.ndarray) -> np.ndarray:
    frozen = np.array(array, dtype=float, copy=True)
    frozen.setflags(write=False)
    return frozen


def _layers_digest(layers: Sequence[LayerParameterInput]) -> str:
    payload = json.dumps([asdict(layer) for layer in layers], sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ParameterBinding:
    """Node parameters of one cold compile; immutable and free of state."""

    nodes_m: np.ndarray
    layers: tuple[LayerParameterInput, ...]
    requested_temperature_K: float
    temperature_K: float
    thermal_voltage_V: float
    temperature_scaling: bool
    band_grading: bool
    dos_band_potentials: bool
    dual_ions: bool
    band_structure: BandStructure
    acceptor_density_m3: np.ndarray
    donor_density_m3: np.ndarray
    band_gap_narrowing_eV: np.ndarray
    composition_coordinate: np.ndarray
    intrinsic_product_m6: np.ndarray
    srh_n1_m3: np.ndarray
    srh_p1_m3: np.ndarray
    radiative_coefficient_m3_s: np.ndarray
    electron_mobility_m2_V_s: np.ndarray
    hole_mobility_m2_V_s: np.ndarray
    electron_diffusivity_m2_s: np.ndarray
    hole_diffusivity_m2_s: np.ndarray
    ion_diffusivity_m2_s: np.ndarray
    negative_ion_diffusivity_m2_s: np.ndarray
    conduction_dos_m3: np.ndarray
    valence_dos_m3: np.ndarray
    thermal_velocity_m_s: np.ndarray
    layer_masks: tuple[np.ndarray, ...]
    layer_states: tuple[LayerTemperatureState, ...]
    inputs_sha256: str
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        payload = json.dumps(
            {
                "model": "layer_parameter_binding_v1",
                "laws": [list(item) for item in TEMPERATURE_LAWS],
                "requested_temperature_K": repr(float(self.requested_temperature_K)),
                "temperature_K": repr(float(self.temperature_K)),
                "thermal_voltage_V": repr(float(self.thermal_voltage_V)),
                "flags": [self.temperature_scaling, self.band_grading, self.dos_band_potentials,
                          self.dual_ions],
                "inputs_sha256": self.inputs_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest = hashlib.sha256(payload)
        digest.update(np.ascontiguousarray(self.nodes_m, dtype="<f8").tobytes())
        object.__setattr__(self, "sha256", digest.hexdigest())

    @property
    def node_count(self) -> int:
        return int(self.nodes_m.size)

    def require(self, *, temperature_K: float, inputs_sha256: str | None = None) -> None:
        """Refuse use for a request this binding was not compiled for."""

        if float(temperature_K) != float(self.requested_temperature_K):
            raise StaleParameterBindingError(
                f"parameter binding was compiled for T={self.requested_temperature_K!r} K, "
                f"not {temperature_K!r} K; rebind instead of reusing its coefficients"
            )
        if inputs_sha256 is not None and inputs_sha256 != self.inputs_sha256:
            raise StaleParameterBindingError("parameter binding was compiled from different layer inputs")

    def rebind(
        self,
        *,
        temperature_K: float | None = None,
        layers: Sequence[LayerParameterInput] | None = None,
    ) -> ParameterBinding:
        """Cold recompile from the stored 300 K inputs (never from these arrays)."""

        return compile_layer_parameters(
            self.nodes_m,
            self.layers if layers is None else layers,
            temperature_K=self.requested_temperature_K if temperature_K is None else temperature_K,
            temperature_scaling=self.temperature_scaling,
            band_grading=self.band_grading,
            dos_band_potentials=self.dos_band_potentials,
            dual_ions=self.dual_ions,
        )


def _grid(x_nodes_m: ArrayLike) -> np.ndarray:
    x = np.array(x_nodes_m, dtype=float, copy=True)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0.0):
        raise ValueError("nodes must be finite and strictly increasing with at least two entries")
    return x


def compile_layer_parameters(
    x_nodes_m: ArrayLike,
    layers: Sequence[LayerParameterInput],
    *,
    temperature_K: float,
    temperature_scaling: bool,
    band_grading: bool,
    dos_band_potentials: bool,
    dual_ions: bool = True,
) -> ParameterBinding:
    """Cold-compile every T/composition/doping-dependent node parameter.

    ``temperature_scaling``, ``band_grading`` and ``dos_band_potentials`` are
    the resolved tier switches (the LEGACY tier turns all three off);
    ``dual_ions`` gates the negative-ion diffusivity as historically.
    """

    layer_inputs = tuple(layers)
    if not layer_inputs or not all(isinstance(item, LayerParameterInput) for item in layer_inputs):
        raise ValueError("at least one LayerParameterInput is required")
    requested = _finite(temperature_K, "temperature_K")
    if requested <= 0.0:
        raise ValueError("temperature_K must be positive")
    x = _grid(x_nodes_m)
    T, V_T = device_temperature(requested, temperature_scaling=bool(temperature_scaling))
    masks = tuple(layer_node_masks(x, [layer.thickness_m for layer in layer_inputs], RIGHT_LAYER_WINS))
    if np.any(np.sum(masks, axis=0) != 1):
        raise ValueError("layer thicknesses must cover every node exactly once")
    N = x.size

    N_A = np.zeros(N)
    N_D = np.zeros(N)
    narrowing_eV = np.zeros(N)
    composition = np.full(N, np.nan)
    chi = np.zeros(N)
    Eg = np.zeros(N)
    D_ion = np.zeros(N)
    D_ion_neg = np.zeros(N)
    mu_n = np.empty(N)
    mu_p = np.empty(N)
    D_n = np.empty(N)
    D_p = np.empty(N)
    ni_sq = np.empty(N)
    n1 = np.empty(N)
    p1 = np.empty(N)
    B_rad = np.empty(N)
    N_C = np.full(N, np.nan)
    N_V = np.full(N, np.nan)
    v_th = np.empty(N)
    states: list[LayerTemperatureState] = []

    offset = 0.0
    for layer, mask in zip(layer_inputs, masks):
        laws = layer.temperature_laws
        x_local = x[mask] - offset

        D_ion[mask] = ion_diffusivity_at_temperature(layer.ion_diffusivity_m2_s, T, laws.ion_activation_energy_eV)
        if dual_ions:
            D_ion_neg[mask] = ion_diffusivity_at_temperature(
                layer.negative_ion_diffusivity_m2_s, T, laws.ion_activation_energy_eV
            )
        N_A[mask], N_D[mask] = layer_doping_profiles(x_local, layer.thickness_m, layer.doping)

        Eg_T = band_gap_at_temperature(layer.band_gap_eV, T, laws.varshni_alpha_eV_K, laws.varshni_beta_K)
        graded = bool(band_grading) and layer.grading.graded
        Eg_back_T: float | None = None
        if graded:
            grading = layer.grading
            y = grading.coordinate(x_local, layer.thickness_m)
            composition[mask] = y
            Eg_back = grading.band_gap_back_eV if grading.band_gap_back_eV is not None else layer.band_gap_eV
            Eg_back_T = band_gap_at_temperature(Eg_back, T, laws.varshni_alpha_eV_K, laws.varshni_beta_K)
            chi_back = (
                grading.electron_affinity_back_eV
                if grading.electron_affinity_back_eV is not None
                else layer.electron_affinity_eV
            )
            chi[mask] = affinity_profile(y, layer.electron_affinity_eV, chi_back)
            Eg[mask] = band_gap_profile(y, Eg_T, Eg_back_T, grading.bowing_eV)
        else:
            chi[mask] = layer.electron_affinity_eV
            Eg[mask] = Eg_T

        narrowed = layer.narrowing.active
        if narrowed:
            edges = tuple(
                layer.narrowing.apply(affinity, gap, acceptors, donors)
                for affinity, gap, acceptors, donors in zip(chi[mask], Eg[mask], N_A[mask], N_D[mask])
            )
            chi[mask] = np.asarray([state.effective_electron_affinity_eV for state in edges])
            Eg[mask] = np.asarray([state.effective_band_gap_eV for state in edges])
            narrowing_eV[mask] = np.asarray([state.narrowing_eV for state in edges])

        mu_n_T = mobility_at_temperature(layer.electron_mobility_m2_V_s, T, laws.mobility_exponent)
        mu_p_T = mobility_at_temperature(layer.hole_mobility_m2_V_s, T, laws.mobility_exponent)
        mu_n[mask] = mu_n_T
        mu_p[mask] = mu_p_T
        D_n[mask] = mu_n_T * V_T
        D_p[mask] = mu_p_T * V_T

        ni_T = intrinsic_density_at_temperature(
            layer.intrinsic_density_m3, Eg_T, T, layer.conduction_dos_300_m3, layer.valence_dos_300_m3
        )
        if graded:
            ni_sq[mask] = grade_ni_sq(ni_T**2, Eg[mask], Eg_T, V_T)
        else:
            ni_sq[mask] = ni_T**2
            if narrowed:
                ni_sq[mask] *= intrinsic_product_narrowing_scale(narrowing_eV[mask], V_T)

        scale = srh_reference_scale(ni_T, layer.intrinsic_density_m3, T)
        if graded:
            n1[mask], p1[mask] = grade_n1_p1(
                layer.srh_n1_m3 * scale, layer.srh_p1_m3 * scale, Eg[mask], Eg_T, V_T
            )
        else:
            reference = srh_reference_narrowing_scale(narrowing_eV[mask], V_T)
            n1[mask] = layer.srh_n1_m3 * scale * reference
            p1[mask] = layer.srh_p1_m3 * scale * reference

        B_rad[mask] = radiative_coefficient_at_temperature(
            layer.radiative_coefficient_m3_s, T, laws.radiative_exponent
        )
        if layer.conduction_dos_300_m3:
            N_C[mask] = band_edge_dos_at_temperature(layer.conduction_dos_300_m3, T)
        if layer.valence_dos_300_m3:
            N_V[mask] = band_edge_dos_at_temperature(layer.valence_dos_300_m3, T)
        v_th[mask] = thermal_velocity_at_temperature(layer.thermal_velocity_m_s, T)

        states.append(LayerTemperatureState(
            band_gap_front_eV=Eg_T,
            band_gap_back_eV=Eg_back_T,
            intrinsic_density_m3=ni_T,
            srh_reference_scale=scale,
            electron_mobility_m2_V_s=mu_n_T,
            hole_mobility_m2_V_s=mu_p_T,
            graded=graded,
            narrowed=narrowed,
        ))
        offset += layer.thickness_m

    if dos_band_potentials:
        band_structure = fold_dos_into_transport_bands(
            chi,
            Eg,
            masks,
            [(layer.conduction_dos_300_m3, layer.valence_dos_300_m3) for layer in layer_inputs],
            reference_layer=dos_fold_reference_layer([layer.role for layer in layer_inputs]),
            thermal_voltage_V=V_T,
        )
    else:
        band_structure = BandStructure.without_dos_fold(chi, Eg)

    frozen_masks = []
    for mask in masks:
        owned = np.array(mask, dtype=bool, copy=True)
        owned.setflags(write=False)
        frozen_masks.append(owned)
    nodes = _readonly(x)
    return ParameterBinding(
        nodes_m=nodes,
        layers=layer_inputs,
        requested_temperature_K=requested,
        temperature_K=T,
        thermal_voltage_V=V_T,
        temperature_scaling=bool(temperature_scaling),
        band_grading=bool(band_grading),
        dos_band_potentials=bool(dos_band_potentials),
        dual_ions=bool(dual_ions),
        band_structure=band_structure,
        acceptor_density_m3=_readonly(N_A),
        donor_density_m3=_readonly(N_D),
        band_gap_narrowing_eV=_readonly(narrowing_eV),
        composition_coordinate=_readonly(composition),
        intrinsic_product_m6=_readonly(ni_sq),
        srh_n1_m3=_readonly(n1),
        srh_p1_m3=_readonly(p1),
        radiative_coefficient_m3_s=_readonly(B_rad),
        electron_mobility_m2_V_s=_readonly(mu_n),
        hole_mobility_m2_V_s=_readonly(mu_p),
        electron_diffusivity_m2_s=_readonly(D_n),
        hole_diffusivity_m2_s=_readonly(D_p),
        ion_diffusivity_m2_s=_readonly(D_ion),
        negative_ion_diffusivity_m2_s=_readonly(D_ion_neg),
        conduction_dos_m3=_readonly(N_C),
        valence_dos_m3=_readonly(N_V),
        thermal_velocity_m_s=_readonly(v_th),
        layer_masks=tuple(frozen_masks),
        layer_states=tuple(states),
        inputs_sha256=_layers_digest(layer_inputs),
    )
