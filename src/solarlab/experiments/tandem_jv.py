"""Monolithic two-terminal (2T) tandem J-V experiment and joint TMM optics.

Execution contract (P04-13 / P05-04):
1. A single joint transfer-matrix (TMM) optical solve runs over the complete
   tandem optical stack:
       ``[top_cell.layers, junction_stack, bottom_cell.layers, back_reflector?]``
   producing a closed ``PhotonBudget`` (``max |1 - R - T - sum A - dropped| <= 1e-12``)
   and exact cell-averaged generation profiles ``G_top(x_top)`` and ``G_bot(x_bot)``
   on each sub-cell's exact ``INTEGER_ENDPOINT_V1`` electrical control volumes.
2. Photons absorbed in optical-only layers (front substrates, recombination
   junction stack, optional back reflector) are accounted as parasitic
   absorption and never generate carriers in either sub-cell.
3. Each sub-cell is cold-rebound with its node-exact ``fixed_generation``
   ``ResourceTable`` and solved independently without cross-sub-cell state sharing.
4. ``series_match_jv`` combines the top and bottom J-V curves under ideal ohmic
   or ``V_junction``-offset series current matching, failing closed when the
   sub-cell current ranges do not overlap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Literal, cast

import numpy as np

from solarlab.config.resolve_device import resolve_device, resolve_tandem
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput, TandemInput
from solarlab.device.resolved import PreparedDevice, PreparedTandem
from solarlab.equations.dc_1d import CompiledDevice1D
from solarlab.equations.ionic_1d import CompiledIonicDevice1D
from solarlab.experiments.electrothermal import (
    ElectrothermalJVProtocol,
    has_declared_ions,
    run_fresh_jv_at_temperature,
)
from solarlab.experiments.jv.steady import (
    SteadyJVMetrics,
    run_steady_jv,
    steady_jv_metrics,
)
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab.mesh import INTEGER_ENDPOINT_V1, build_geometry, multilayer_grid
from solarlab.mesh.geometry import ControlVolumeGeometry1D
from solarlab.physics.generation.optical_binding import (
    PhotonBudget,
    constant_index_optical_constants,
)
from solarlab.physics.generation.spectrum import sample_nk_table, wavelength_grid_m
from solarlab.physics.generation.transfer_matrix import (
    TMMLayer,
    dropped_multipass_fraction,
    tmm_absorbed_photon_flux_per_cell,
    tmm_cumulative_absorptance,
    tmm_reflectance,
    tmm_transmittance,
)
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state

__all__ = [
    "PHOTON_BUDGET_TOLERANCE",
    "TandemCapabilityError",
    "TandemGenerationResult",
    "TandemJVProtocol",
    "TandemJVResult",
    "compute_tandem_generation",
    "prepare_subcell_with_fixed_generation",
    "run_tandem_jv",
    "series_match_jv",
]

PHOTON_BUDGET_TOLERANCE: float = 1e-12
PLANCK_CONSTANT_J_S: float = 6.62607015e-34
SPEED_OF_LIGHT_M_S: float = 299792458.0


class TandemCapabilityError(ValueError):
    """Unsupported tandem configuration or failed optical photon closure."""


def _is_real_number(value: object) -> bool:
    return isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(
        value, bool
    )


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TandemGenerationResult:
    """Joint TMM optical solution and control-volume sub-cell generation profiles."""

    top_nodes_m: np.ndarray
    top_cell_measures_m: np.ndarray
    top_generation_m3_s: np.ndarray
    bottom_nodes_m: np.ndarray
    bottom_cell_measures_m: np.ndarray
    bottom_generation_m3_s: np.ndarray
    photon_budget: PhotonBudget
    top_absorbed_m2_s: float
    bottom_absorbed_m2_s: float
    parasitic_fraction: float
    incident_power_W_m2: float
    optical_stack_sha256: str


@dataclass(frozen=True)
class TandemJVProtocol:
    """Explicit 2T tandem optical + electrical sweep controls."""

    intervals_per_layer: int | Sequence[int] = 24
    voltage_points_per_subcell: int = 81
    top_voltage_min_V: float = -0.2
    top_voltage_max_V: float = 1.4
    bottom_voltage_min_V: float = -0.2
    bottom_voltage_max_V: float = 1.0
    matched_points: int = 121
    junction_voltage_offset_V: float = 0.0
    k_floor: float = 1e-6
    driver: Literal["steady_jv", "transient_jv_strict"] = "steady_jv"
    scan_rate_V_s: float = 1.0
    spectrum_name: str | None = None
    incident_power_W_m2: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.intervals_per_layer, bool):
            raise ValueError("intervals_per_layer must be an int or int sequence")
        if isinstance(self.intervals_per_layer, int):
            if self.intervals_per_layer < 2:
                raise ValueError("intervals_per_layer must be >= 2")
        else:
            counts = tuple(self.intervals_per_layer)
            if not counts or any(type(c) is not int or c < 2 for c in counts):
                raise ValueError("intervals_per_layer sequence requires ints >= 2")
            object.__setattr__(self, "intervals_per_layer", counts)

        if (
            type(self.voltage_points_per_subcell) is not int
            or self.voltage_points_per_subcell < 3
        ):
            raise ValueError("voltage_points_per_subcell must be an integer >= 3")
        if type(self.matched_points) is not int or self.matched_points < 3:
            raise ValueError("matched_points must be an integer >= 3")
        for name in (
            "top_voltage_min_V",
            "top_voltage_max_V",
            "bottom_voltage_min_V",
            "bottom_voltage_max_V",
            "junction_voltage_offset_V",
            "k_floor",
            "scan_rate_V_s",
        ):
            raw = getattr(self, name)
            if not _is_real_number(raw) or not math.isfinite(float(raw)):
                raise ValueError(f"{name} must be a finite real scalar")
            object.__setattr__(self, name, float(raw))
        if self.top_voltage_max_V <= self.top_voltage_min_V:
            raise ValueError("top_voltage_max_V must exceed top_voltage_min_V")
        if self.bottom_voltage_max_V <= self.bottom_voltage_min_V:
            raise ValueError("bottom_voltage_max_V must exceed bottom_voltage_min_V")
        if self.k_floor < 0.0:
            raise ValueError("k_floor must be >= 0")
        if self.scan_rate_V_s <= 0.0:
            raise ValueError("scan_rate_V_s must be > 0")
        if self.driver not in ("steady_jv", "transient_jv_strict"):
            raise ValueError("driver must be 'steady_jv' or 'transient_jv_strict'")
        if self.incident_power_W_m2 is not None:
            if (
                not _is_real_number(self.incident_power_W_m2)
                or not math.isfinite(float(self.incident_power_W_m2))
                or float(self.incident_power_W_m2) <= 0.0
            ):
                raise ValueError("incident_power_W_m2 must be None or finite positive")
            object.__setattr__(
                self, "incident_power_W_m2", float(self.incident_power_W_m2)
            )


@dataclass(frozen=True)
class TandemJVResult:
    """Monolithic 2T series-matched tandem J-V curves, metrics, and photon budget."""

    V_fwd: np.ndarray
    J_fwd: np.ndarray
    V_rev: np.ndarray
    J_rev: np.ndarray
    metrics_fwd: SteadyJVMetrics
    metrics_rev: SteadyJVMetrics
    hysteresis_index: float | None
    top_V_fwd: np.ndarray
    top_J_fwd: np.ndarray
    top_V_rev: np.ndarray
    top_J_rev: np.ndarray
    top_metrics_fwd: SteadyJVMetrics
    bottom_V_fwd: np.ndarray
    bottom_J_fwd: np.ndarray
    bottom_V_rev: np.ndarray
    bottom_J_rev: np.ndarray
    bottom_metrics_fwd: SteadyJVMetrics
    optics: TandemGenerationResult
    incident_power_W_m2: float
    top_compiled_identity: str
    bottom_compiled_identity: str
    tandem_prepared_sha256: str
    result_sha256: str


def _subcell_geometry(
    prepared: PreparedDevice,
    intervals_per_layer: int | Sequence[int],
) -> ControlVolumeGeometry1D:
    """Build the exact ``INTEGER_ENDPOINT_V1`` electrical geometry of ``compile_device_1d``."""
    data = prepared.to_mapping()
    settings = data["settings"]
    layers = tuple(layer for layer in prepared.layers if layer.role != "substrate")
    grid = {entry["layer"]: entry for entry in data["electrical_grid"]}
    if isinstance(intervals_per_layer, int):
        weights = np.asarray(
            [grid.get(layer.id, {}).get("interval_weight", 1.0) for layer in layers],
            dtype=float,
        )
        counts = (
            np.maximum(2, np.rint(intervals_per_layer * weights / np.mean(weights)))
            .astype(int)
            .tolist()
        )
    else:
        counts = list(intervals_per_layer)
        if len(counts) != len(layers):
            raise TandemCapabilityError(
                f"sub-cell {data['id']}: interval sequence length {len(counts)} != {len(layers)} electrical layers"
            )
    for idx, layer in enumerate(layers):
        p = dict(layer.parameters)
        if settings["band_grading"] and (
            p["Eg_back"] is not None or p["chi_back"] is not None
        ):
            counts[idx] *= int(cast(int, p["grading_N_mult"]))
    alphas = [float(grid.get(layer.id, {}).get("alpha", 3.0)) for layer in layers]
    thicknesses = [layer.thickness for layer in layers]
    nodes = multilayer_grid(list(zip(thicknesses, counts)), alpha=alphas)
    return build_geometry(INTEGER_ENDPOINT_V1, nodes, np.cumsum(thicknesses)[:-1])


def _resolve_spectrum_table(
    tandem: PreparedTandem,
    explicit_name: str | None,
) -> ResourceTable:
    top_spec = tandem.top_cell.to_mapping()["declaration"].get("spectrum")
    bot_spec = tandem.bottom_cell.to_mapping()["declaration"].get("spectrum")
    if explicit_name is not None:
        chosen = explicit_name
    elif top_spec is not None and bot_spec is not None:
        if top_spec != bot_spec:
            raise TandemCapabilityError(
                f"top_cell spectrum {top_spec!r} and bottom_cell spectrum {bot_spec!r} disagree"
            )
        chosen = str(top_spec)
    elif top_spec is not None:
        chosen = str(top_spec)
    elif bot_spec is not None:
        chosen = str(bot_spec)
    else:
        raise TandemCapabilityError(
            "2T tandem TMM requires an explicit spectrum resource on top_cell, bottom_cell, or TandemJVProtocol.spectrum_name"
        )
    return tandem.resources.get(chosen, "spectrum")


def compute_tandem_generation(
    tandem: PreparedTandem,
    *,
    intervals_top: int | Sequence[int] = 24,
    intervals_bottom: int | Sequence[int] = 24,
    spectrum_name: str | None = None,
    k_floor: float = 1e-6,
) -> TandemGenerationResult:
    """Solve joint coherent TMM optics across the full 2T tandem stack."""
    if not isinstance(tandem, PreparedTandem):
        raise TypeError("tandem must be a PreparedTandem")
    if not _is_real_number(k_floor) or not math.isfinite(float(k_floor)) or float(k_floor) < 0.0:
        raise ValueError("k_floor must be finite and >= 0")
    k_min = float(k_floor)

    meta = tandem.to_mapping()
    if meta["junction_model"] != "ideal_ohmic":
        raise TandemCapabilityError(
            f"Unsupported junction_model {meta['junction_model']!r}; only 'ideal_ohmic' is supported"
        )
    if meta["light_direction"] != "top_first":
        raise TandemCapabilityError(
            f"Unsupported light_direction {meta['light_direction']!r}; only 'top_first' is supported"
        )

    spec_table = _resolve_spectrum_table(tandem, spectrum_name)
    wl_m = np.array(spec_table.array[:, 0], dtype=float, copy=True)
    wl_nm = wl_m * 1e9
    spectral_flux = np.array(spec_table.array[:, 1], dtype=float, copy=True)

    geom_top = _subcell_geometry(tandem.top_cell, intervals_top)
    geom_bot = _subcell_geometry(tandem.bottom_cell, intervals_bottom)

    tmm_layers: list[TMMLayer] = []
    layer_names: list[str] = []
    layer_electrical: list[bool] = []
    dep_records: list[dict[str, Any]] = []

    def append_device_layers(cell: PreparedDevice, prefix: str) -> tuple[float, float]:
        """Append all layers of ``cell`` and return ``(elec_start_m, elec_end_m)``."""
        sub_prefix_m = 0.0
        elec_total_m = 0.0
        seen_elec = False
        stack_start_m = sum(item.d for item in tmm_layers)
        for layer in cell.layers:
            params = dict(layer.parameters)
            if layer.cigs_graded_optics_json is not None:
                raise TandemCapabilityError(
                    f"{prefix}.{layer.id}: graded CIGS optics is not bound in 2T tandem"
                )
            incoh = bool(params["incoherent"])
            if incoh and len(tmm_layers) > 0:
                raise TandemCapabilityError(
                    f"{prefix}.{layer.id}: only the very first optical layer of the tandem may be incoherent"
                )
            if layer.optical is not None:
                if params["n_optical"] is not None:
                    raise TandemCapabilityError(
                        f"{prefix}.{layer.id}: cannot specify both optical_material and n_optical"
                    )
                native = layer.optical.array
                n_arr, k_arr = sample_nk_table(
                    native[:, 0] * 1e9,
                    native[:, 1],
                    native[:, 2],
                    wl_nm,
                    material=layer.optical.name,
                )
                nk_sha = layer.optical.content_sha256
            else:
                if params["n_optical"] is None:
                    raise TandemCapabilityError(
                        f"{prefix}.{layer.id}: requires optical_material or explicit n_optical"
                    )
                n_arr, k_arr = constant_index_optical_constants(
                    wl_nm,
                    absorption_coefficient_m1=float(cast(float, params["alpha"])),
                    refractive_index=float(cast(float, params["n_optical"])),
                )
                nk_sha = "constant_index"
            k_eff = np.maximum(k_arr, k_min)
            is_elec = layer.role != "substrate"
            if not is_elec:
                if seen_elec:
                    raise TandemCapabilityError(
                        f"{prefix}.{layer.id}: substrate layers must form a front prefix"
                    )
                sub_prefix_m += layer.thickness
            else:
                seen_elec = True
                elec_total_m += layer.thickness

            tmm_layers.append(
                TMMLayer(d=layer.thickness, n=n_arr, k=k_eff, incoherent=incoh)
            )
            layer_names.append(f"{prefix}:{layer.id}")
            layer_electrical.append(is_elec)
            dep_records.append(
                {
                    "name": f"{prefix}:{layer.id}",
                    "thickness_m": float(layer.thickness).hex(),
                    "incoherent": incoh,
                    "electrical": is_elec,
                    "nk_sha256": nk_sha,
                }
            )
        elec_start = stack_start_m + sub_prefix_m
        return elec_start, elec_start + elec_total_m

    def append_optical_only(opt_entry: Any, role_prefix: str) -> None:
        incoh = bool(opt_entry.incoherent)
        if incoh and len(tmm_layers) > 0:
            raise TandemCapabilityError(
                f"{role_prefix}.{opt_entry.id}: mid-stack or back optical layers cannot be incoherent"
            )
        res = tandem.resources.get(opt_entry.optical_material, "nk")
        native = res.array
        n_arr, k_arr = sample_nk_table(
            native[:, 0] * 1e9,
            native[:, 1],
            native[:, 2],
            wl_nm,
            material=res.name,
        )
        k_eff = np.maximum(k_arr, k_min)
        d_m = float(opt_entry.normalized_data()["thickness"])
        tmm_layers.append(TMMLayer(d=d_m, n=n_arr, k=k_eff, incoherent=incoh))
        layer_names.append(f"{role_prefix}:{opt_entry.id}")
        layer_electrical.append(False)
        dep_records.append(
            {
                "name": f"{role_prefix}:{opt_entry.id}",
                "thickness_m": d_m.hex(),
                "incoherent": incoh,
                "electrical": False,
                "nk_sha256": res.content_sha256,
            }
        )

    tandem_input = tandem.to_input()
    top_elec_start_m, _top_elec_end_m = append_device_layers(tandem.top_cell, "top")
    for j_layer in tandem_input.junction_stack:
        append_optical_only(j_layer, "junction")
    bot_elec_start_m, _bot_elec_end_m = append_device_layers(
        tandem.bottom_cell, "bottom"
    )
    if tandem_input.back_reflector is not None:
        append_optical_only(tandem_input.back_reflector, "back_reflector")

    boundaries = np.r_[0.0, np.cumsum([lyr.d for lyr in tmm_layers])]
    cum_abs = tmm_cumulative_absorptance(
        tmm_layers, wl_m, boundaries, boundaries, n_ambient=1.0, n_substrate=1.0
    )
    layer_abs = cum_abs[1:, :] - cum_abs[:-1, :]
    refl = tmm_reflectance(tmm_layers, wl_m, n_ambient=1.0, n_substrate=1.0)
    trans = tmm_transmittance(tmm_layers, wl_m, n_ambient=1.0, n_substrate=1.0)
    dropped = dropped_multipass_fraction(
        tmm_layers, wl_m, n_ambient=1.0, n_substrate=1.0
    )

    budget = PhotonBudget(
        wavelengths_nm=wl_nm,
        incident_spectral_m2_s_m=spectral_flux,
        reflectance=refl,
        transmittance=trans,
        layer_absorptance=layer_abs,
        dropped_multipass=dropped,
        layer_names=tuple(layer_names),
        layer_electrical=tuple(layer_electrical),
    )
    closure_err = float(np.max(np.abs(budget.spectral_closure_residual())))
    if not math.isfinite(closure_err) or closure_err > PHOTON_BUDGET_TOLERANCE:
        raise TandemCapabilityError(
            f"Joint 2T tandem TMM photon budget failed closure: max residual {closure_err:.3e} > {PHOTON_BUDGET_TOLERANCE:.3e}"
        )

    top_faces = geom_top.cell_faces_m + top_elec_start_m
    top_absorbed_cells = tmm_absorbed_photon_flux_per_cell(
        tmm_layers, wl_m, spectral_flux, top_faces, boundaries, 1.0, 1.0
    )
    g_top = top_absorbed_cells / geom_top.cell_measures_m

    bot_faces = geom_bot.cell_faces_m + bot_elec_start_m
    bot_absorbed_cells = tmm_absorbed_photon_flux_per_cell(
        tmm_layers, wl_m, spectral_flux, bot_faces, boundaries, 1.0, 1.0
    )
    g_bot = bot_absorbed_cells / geom_bot.cell_measures_m

    if not np.all(np.isfinite(g_top)) or not np.all(np.isfinite(g_bot)):
        raise TandemCapabilityError("Nonfinite sub-cell generation profile in 2T tandem")
    g_top = np.maximum(g_top, 0.0)
    g_bot = np.maximum(g_bot, 0.0)

    top_abs_flux = float(np.sum(g_top * geom_top.cell_measures_m))
    bot_abs_flux = float(np.sum(g_bot * geom_bot.cell_measures_m))
    inc_flux = budget.incident_m2_s
    tol_flux = PHOTON_BUDGET_TOLERANCE * max(1.0, inc_flux)
    if abs((top_abs_flux + bot_abs_flux) - budget.electrical_absorbed_m2_s) > 10.0 * tol_flux:
        raise TandemCapabilityError(
            "Sub-cell control-volume generation sum disagrees with joint TMM electrical absorption"
        )

    parasitic_frac = (
        0.0 if inc_flux <= 0.0 else float(budget.parasitic_absorbed_m2_s / inc_flux)
    )
    photon_energy_J = (PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_S) / wavelength_grid_m(
        wl_nm
    )
    inc_power_W_m2 = float(
        np.trapezoid(spectral_flux * photon_energy_J, x=wavelength_grid_m(wl_nm))
    )

    top_nodes = np.array(geom_top.nodes_m, dtype=float, copy=True)
    top_w = np.array(geom_top.cell_measures_m, dtype=float, copy=True)
    bot_nodes = np.array(geom_bot.nodes_m, dtype=float, copy=True)
    bot_w = np.array(geom_bot.cell_measures_m, dtype=float, copy=True)
    for arr in (top_nodes, top_w, g_top, bot_nodes, bot_w, g_bot):
        arr.setflags(write=False)

    opt_sha = _canonical_sha256(
        {
            "tandem_sha256": tandem.content_sha256,
            "spectrum_sha256": spec_table.content_sha256,
            "k_floor": k_min.hex(),
            "layers": dep_records,
        }
    )
    return TandemGenerationResult(
        top_nodes_m=top_nodes,
        top_cell_measures_m=top_w,
        top_generation_m3_s=g_top,
        bottom_nodes_m=bot_nodes,
        bottom_cell_measures_m=bot_w,
        bottom_generation_m3_s=g_bot,
        photon_budget=budget,
        top_absorbed_m2_s=top_abs_flux,
        bottom_absorbed_m2_s=bot_abs_flux,
        parasitic_fraction=parasitic_frac,
        incident_power_W_m2=inc_power_W_m2,
        optical_stack_sha256=opt_sha,
    )


def prepare_subcell_with_fixed_generation(
    subcell: PreparedDevice,
    nodes_m: np.ndarray,
    generation_m3_s: np.ndarray,
    *,
    resource_name: str,
) -> PreparedDevice:
    """Bind a joint-TMM ``fixed_generation`` table onto an electrical sub-cell."""
    payload_bytes = json.dumps(
        {
            "coordinate_m": [float(x) for x in nodes_m],
            "generation_m3_s": [float(g) for g in generation_m3_s],
        },
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    src_doc = SourceDocument(id=f"tandem_fixed_gen_{resource_name}", content=payload_bytes)
    gen_table = ResourceTable(name=resource_name, kind="fixed_generation", source=src_doc)
    retained_tables = tuple(
        tbl for tbl in subcell.resources.tables if tbl.name != resource_name
    )
    merged_resources = ResourceLibrary(tables=(*retained_tables, gen_table))

    editing = subcell.to_input().editing_data()
    editing["spectrum"] = None
    editing["fixed_generation"] = resource_name
    for mat in editing.get("materials", ()):
        mat_params = mat.get("parameters")
        if isinstance(mat_params, dict):
            mat_params.pop("optical_material", None)
            mat_params.pop("n_optical", None)
    for lyr in editing.get("layers", ()):
        lyr_params = lyr.setdefault("parameters", {})
        lyr_params["optical_material"] = None
        lyr_params["n_optical"] = None
        if lyr.get("role") == "substrate":
            lyr_params["alpha"] = 0.0

    retained_sources = tuple(
        src for src in subcell.sources if src.id != src_doc.id
    )
    return resolve_device(
        DeviceInput.model_validate(editing),
        subcell.defaults,
        merged_resources,
        sources=(*retained_sources, src_doc),
    )


def series_match_jv(
    top_J: object,
    top_V: object,
    bot_J: object,
    bot_V: object,
    *,
    V_junction: float = 0.0,
    n_points: int = 121,
) -> tuple[np.ndarray, np.ndarray]:
    """Series-match two sub-cell J-V curves at common current densities ``J``.

    Returns ``(V_tandem, J_tandem)`` ordered by strictly increasing ``V_tandem``,
    explicitly including ``J = 0`` whenever open circuit lies inside the overlap
    bracket ``[j_lo, j_hi]``. Raises ``ValueError`` when the sub-cell current
    ranges do not overlap.
    """
    if not _is_real_number(V_junction) or not math.isfinite(float(V_junction)):
        raise ValueError("V_junction must be a finite real scalar")
    if type(n_points) is not int or n_points < 3:
        raise ValueError("n_points must be an integer >= 3")

    j1 = np.asarray(top_J, dtype=float)
    v1 = np.asarray(top_V, dtype=float)
    j2 = np.asarray(bot_J, dtype=float)
    v2 = np.asarray(bot_V, dtype=float)
    for name, arr in (("top_J", j1), ("top_V", v1), ("bot_J", j2), ("bot_V", v2)):
        if arr.ndim != 1 or arr.size < 2 or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be a 1-D finite array with >= 2 samples")
    if j1.shape != v1.shape or j2.shape != v2.shape:
        raise ValueError("sub-cell J and V arrays must have matching shapes")

    ord1 = np.argsort(j1)
    j1_s, v1_s = j1[ord1], v1[ord1]
    ord2 = np.argsort(j2)
    j2_s, v2_s = j2[ord2], v2[ord2]

    # Collapse any exact duplicate current samples before 1D inverse interpolation
    j1_u, idx1 = np.unique(j1_s, return_index=True)
    v1_u = v1_s[idx1]
    j2_u, idx2 = np.unique(j2_s, return_index=True)
    v2_u = v2_s[idx2]
    if j1_u.size < 2 or j2_u.size < 2:
        raise ValueError("sub-cell J curves must span a nonzero current range")

    j_lo = max(float(j1_u[0]), float(j2_u[0]))
    j_hi = min(float(j1_u[-1]), float(j2_u[-1]))
    if j_lo >= j_hi:
        raise ValueError(
            f"Sub-cell J ranges do not overlap: top=[{j1_u[0]:.4g}, {j1_u[-1]:.4g}] A/m^2, "
            f"bottom=[{j2_u[0]:.4g}, {j2_u[-1]:.4g}] A/m^2"
        )

    native_overlap = np.r_[
        j1_u[(j1_u >= j_lo) & (j1_u <= j_hi)],
        j2_u[(j2_u >= j_lo) & (j2_u <= j_hi)],
    ]
    j_grid = np.unique(np.r_[np.linspace(j_lo, j_hi, n_points), native_overlap])
    if j_lo < 0.0 < j_hi and not np.any(np.abs(j_grid) <= 1e-15):
        j_grid = np.sort(np.r_[j_grid, 0.0])

    v_top_interp = np.interp(j_grid, j1_u, v1_u)
    v_bot_interp = np.interp(j_grid, j2_u, v2_u)
    v_tandem = v_top_interp + v_bot_interp + float(V_junction)

    v_order = np.argsort(v_tandem)
    v_sorted = v_tandem[v_order]
    j_sorted = j_grid[v_order]

    # Ensure strictly increasing voltages
    v_unique, u_idx = np.unique(v_sorted, return_index=True)
    j_unique = j_sorted[u_idx]
    if v_unique.size < 2:
        raise ValueError("series-matched tandem voltage grid collapsed to < 2 unique points")

    v_out = np.array(v_unique, dtype=float, copy=True)
    j_out = np.array(j_unique, dtype=float, copy=True)
    v_out.setflags(write=False)
    j_out.setflags(write=False)
    return v_out, j_out


def _run_subcell_jv(
    prepared: PreparedDevice,
    v_min: float,
    v_max: float,
    protocol: TandemJVProtocol,
    incident_power_W_m2: float,
) -> tuple[
    CompiledDevice1D | CompiledIonicDevice1D,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    has_ions, inferred_layer = has_declared_ions(prepared)
    if not has_ions and protocol.driver == "steady_jv":
        compiled = compile_device_1d(
            prepared, intervals_per_layer=protocol.intervals_per_layer
        )
        v_fwd = np.linspace(v_min, v_max, protocol.voltage_points_per_subcell)
        v_rev = v_fwd[::-1].copy()
        res_fwd = run_steady_jv(
            compiled,
            v_fwd,
            generation_scale=1.0,
            incident_power_W_m2=incident_power_W_m2,
            options=SteadyOptions(),
        )
        state = res_fwd.solutions[-1].state
        rev_currents: list[float] = []
        for v in v_rev:
            sol = solve_steady_state(
                compiled,
                initial_state=state,
                voltage_V=float(v),
                generation_scale=1.0,
                options=SteadyOptions(),
            )
            state = sol.state
            rev_currents.append(
                -float(compiled.voltage_polarity)
                * float(sol.current_density_A_m2[0])
            )
        j_rev: np.ndarray = np.asarray(rev_currents, dtype=float)
        j_rev.setflags(write=False)
        v_rev.setflags(write=False)
        return (
            compiled,
            res_fwd.voltage_V,
            res_fwd.output_current_A_m2,
            v_rev,
            j_rev,
        )

    # Ionic or transient_jv_strict sub-cell
    electrical_layers = tuple(
        layer for layer in prepared.layers if layer.role != "substrate"
    )
    compiled_any: CompiledDevice1D | CompiledIonicDevice1D
    if has_ions:
        endpoints = ("blocking", "blocking") if len(electrical_layers) > 1 else None
        compiled_any = compile_ionic_device_1d(
            prepared,
            intervals_per_layer=protocol.intervals_per_layer,
            ion_domain_layer=inferred_layer,
            ion_domain_endpoints=endpoints,
        )
    else:
        compiled_any = compile_device_1d(
            prepared, intervals_per_layer=protocol.intervals_per_layer
        )

    helper_prot = ElectrothermalJVProtocol(
        intervals_per_layer=(
            protocol.intervals_per_layer
            if isinstance(protocol.intervals_per_layer, int)
            else 24
        ),
        voltage_points_per_branch=protocol.voltage_points_per_subcell,
        scan_rate_V_s=protocol.scan_rate_V_s,
        voltage_max_V=v_max,
        incident_power_W_m2=incident_power_W_m2,
        driver=protocol.driver,
    )
    v_f, j_f, v_r, j_r = run_fresh_jv_at_temperature(compiled_any, helper_prot)
    return compiled_any, v_f, j_f, v_r, j_r


def run_tandem_jv(
    tandem: PreparedTandem | TandemInput,
    protocol: TandemJVProtocol | None = None,
    *,
    defaults: DefaultCatalog | None = None,
    resources: ResourceLibrary | None = None,
    sources: tuple[SourceDocument, ...] = (),
) -> TandemJVResult:
    """Execute joint coherent TMM optics and series-matched 2T tandem J-V sweeps."""
    if isinstance(tandem, TandemInput):
        if defaults is None or resources is None:
            raise ValueError(
                "defaults and resources are required when tandem is passed as TandemInput"
            )
        prepared_tandem = resolve_tandem(tandem, defaults, resources, sources=sources)
    elif isinstance(tandem, PreparedTandem):
        prepared_tandem = tandem
    else:
        raise TypeError("tandem must be a PreparedTandem or TandemInput")

    prot = TandemJVProtocol() if protocol is None else protocol
    if not isinstance(prot, TandemJVProtocol):
        raise TypeError("protocol must be a TandemJVProtocol")

    optics = compute_tandem_generation(
        prepared_tandem,
        intervals_top=prot.intervals_per_layer,
        intervals_bottom=prot.intervals_per_layer,
        spectrum_name=prot.spectrum_name,
        k_floor=prot.k_floor,
    )
    p_in = (
        optics.incident_power_W_m2
        if prot.incident_power_W_m2 is None
        else prot.incident_power_W_m2
    )

    top_bound = prepare_subcell_with_fixed_generation(
        prepared_tandem.top_cell,
        optics.top_nodes_m,
        optics.top_generation_m3_s,
        resource_name="tandem_top_fixed_generation",
    )
    bot_bound = prepare_subcell_with_fixed_generation(
        prepared_tandem.bottom_cell,
        optics.bottom_nodes_m,
        optics.bottom_generation_m3_s,
        resource_name="tandem_bottom_fixed_generation",
    )

    top_comp, top_vf, top_jf, top_vr, top_jr = _run_subcell_jv(
        top_bound, prot.top_voltage_min_V, prot.top_voltage_max_V, prot, p_in
    )
    bot_comp, bot_vf, bot_jf, bot_vr, bot_jr = _run_subcell_jv(
        bot_bound, prot.bottom_voltage_min_V, prot.bottom_voltage_max_V, prot, p_in
    )

    v_fwd, j_fwd = series_match_jv(
        top_jf,
        top_vf,
        bot_jf,
        bot_vf,
        V_junction=prot.junction_voltage_offset_V,
        n_points=prot.matched_points,
    )
    v_rev_asc, j_rev_asc = series_match_jv(
        top_jr,
        top_vr,
        bot_jr,
        bot_vr,
        V_junction=prot.junction_voltage_offset_V,
        n_points=prot.matched_points,
    )
    v_rev = np.array(v_rev_asc[::-1], dtype=float, copy=True)
    j_rev = np.array(j_rev_asc[::-1], dtype=float, copy=True)
    v_rev.setflags(write=False)
    j_rev.setflags(write=False)

    m_fwd = steady_jv_metrics(v_fwd, j_fwd, incident_power_W_m2=p_in)
    m_rev = steady_jv_metrics(v_rev, j_rev, incident_power_W_m2=p_in)
    top_m = steady_jv_metrics(top_vf, top_jf, incident_power_W_m2=p_in)
    bot_m = steady_jv_metrics(bot_vf, bot_jf, incident_power_W_m2=p_in)

    hi: float | None = None
    if (
        m_fwd.maximum_power_W_m2 is not None
        and m_rev.maximum_power_W_m2 is not None
        and m_rev.maximum_power_W_m2 > 0.0
    ):
        hi = float(
            (m_rev.maximum_power_W_m2 - m_fwd.maximum_power_W_m2)
            / m_rev.maximum_power_W_m2
        )

    res_sha = _canonical_sha256(
        {
            "tandem_prepared_sha256": prepared_tandem.content_sha256,
            "optical_stack_sha256": optics.optical_stack_sha256,
            "top_compiled_identity": top_comp.identity,
            "bottom_compiled_identity": bot_comp.identity,
            "metrics_fwd": asdict(m_fwd),
            "metrics_rev": asdict(m_rev),
        }
    )
    return TandemJVResult(
        V_fwd=v_fwd,
        J_fwd=j_fwd,
        V_rev=v_rev,
        J_rev=j_rev,
        metrics_fwd=m_fwd,
        metrics_rev=m_rev,
        hysteresis_index=hi,
        top_V_fwd=top_vf,
        top_J_fwd=top_jf,
        top_V_rev=top_vr,
        top_J_rev=top_jr,
        top_metrics_fwd=top_m,
        bottom_V_fwd=bot_vf,
        bottom_J_fwd=bot_jf,
        bottom_V_rev=bot_vr,
        bottom_J_rev=bot_jr,
        bottom_metrics_fwd=bot_m,
        optics=optics,
        incident_power_W_m2=p_in,
        top_compiled_identity=top_comp.identity,
        bottom_compiled_identity=bot_comp.identity,
        tandem_prepared_sha256=prepared_tandem.content_sha256,
        result_sha256=res_sha,
    )
