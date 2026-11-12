"""Native 2D J-V dwell sweeps and forward/reverse dynamic state-chain scans.

Executes 2D J-V protocols on ``solarlab``'s unified sparse 2D ``Assembler`` and
conservative Backward Euler steppers (``run_backward_euler_2d`` for
``frozen_ion_background`` and ``run_ionic_backward_euler_2d`` for
``single_positive_mobile_ion``), with:
- 1D dark-equilibrium or finite-time illuminated preconditioning (freezing
  equilibrated ``P_1d`` on ``frozen_ion_background`` ionic stacks),
- instantaneous Maxwell port current decomposition (carrier + ion + true
  Gauss-Maxwell displacement ``dD_y/dt``),
- explicit failure / partial result disposition (never seeding from a failed
  state or silently dropping physics),
- continuous preconditioning/hold + forward/reverse dynamic scan state chains,
- lossless JSON serialization with ``scientific_qualification=False``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from solarlab.config.resolve_device import resolve_device
from solarlab.contracts import frozen_array
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.device.compile_2d import compile_device_2d
from solarlab.device.compile_ionic_1d import ION_INPUTS, compile_ionic_device_1d
from solarlab.device.compile_ionic_2d import (
    build_radiative_reabsorption_2d,
    build_two_sided_interface_srh_2d,
    compile_ionic_device_2d,
)
from solarlab.device.grain_2d import grain_declarations
from solarlab.device.inputs import DeviceInput, GrainBoundaryInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.dc_2d import CompiledDevice2D, DC2DProblem, State2D
from solarlab.equations.ionic_2d import (
    CompiledIonicDevice2D,
    IonicState2D,
    MobileIonDiagnostics2D,
    RadiativeReabsorptionSpec2D,
    TwoSidedInterfaceSRHCoupling2D,
    TwoSidedInterfaceSRHReport2D,
    assess_mobile_ion_terminal_2d,
    evaluate_two_sided_interface_srh_2d,
    recompute_g_with_rad_2d,
)
from solarlab.experiments.jv.steady import SteadyJVMetrics, steady_jv_metrics
from solarlab.experiments.results import decode_array, encode_array
from solarlab.experiments.two_dimensional.inputs import (
    JV2DInput,
    JV2DProtocolInput,
    MicrostructureInput,
)
from solarlab.experiments.two_dimensional.jv_protocol_2d import (
    DynamicJVPairProtocol2D,
    ImplicitProtocolError,
    JV2DGrainBoundaryProtocol,
    JV2DProtocol,
    ProtocolMode,
    build_jv_2d_protocol,
    resolve_jv_2d_protocol,
)
from solarlab.experiments.two_dimensional.preparation import PreparedSpatialExperiment
from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions, solve_ionic_steady_state
from solarlab.solvers.nonlinear.ionic_2d import solve_ionic_steady_state_2d
from solarlab.solvers.nonlinear.problem import SolveBudget, state_digest
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state
from solarlab.solvers.time.backward_euler_2d import (
    BackwardEuler2DOptions,
    compute_carrier_only_terminal_current_2d,
    run_backward_euler_2d,
)
from solarlab.solvers.time.ionic_backward_euler_2d import (
    IonicBackwardEuler2DOptions,
    MobileIonCurrentComponents2D,
    compute_mobile_ion_current_components_2d,
    run_ionic_backward_euler_2d,
)
from solarlab.units import normalize_quantity

__all__ = [
    "JV2DDynamicPairResult",
    "JV2DMetrics",
    "JV2DPointAttempt",
    "JV2DResult",
    "JV2D_RESULT_SCHEMA",
    "ProgressCallback",
    "SpatialSnapshot2D",
    "build_jv_2d_execution_protocol",
    "compute_jv_2d_metrics",
    "decode_json_payload",
    "default_ionic_steady_options_1d",
    "device_has_ions",
    "embed_frozen_ion_excess_2d",
    "encode_json_payload",
    "extract_snapshot_2d",
    "precondition_frozen_lane",
    "project_prepared_for_electronic_2d",
    "read_jv_2d_result",
    "run_dynamic_jv_pair_2d",
    "run_jv_sweep_2d",
    "write_atomic_json",
    "write_jv_2d_result",
]

_JSON_CODEC_TAGS = {"$array", "$scalar", "$float", "$tuple"}


def encode_json_payload(value: Any) -> Any:
    """Losslessly encode dataclasses, tuples, arrays, and nonfinite floats for JSON."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: encode_json_payload(getattr(value, f.name)) for f in fields(value)}
    if type(value) is np.ndarray:
        return {"$array": encode_array(value, unit="raw diagnostic; units follow the named producer field")}
    if isinstance(value, np.generic):
        return {"$scalar": encode_array(np.asarray(value), unit="raw diagnostic")}
    if isinstance(value, Enum):
        return encode_json_payload(value.value)
    if type(value) is tuple:
        return {"$tuple": [encode_json_payload(item) for item in value]}
    if type(value) is list:
        return [encode_json_payload(item) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value) or set(value) & _JSON_CODEC_TAGS:
            raise ValueError("JSON payloads require plain string keys without codec tags")
        return {key: encode_json_payload(item) for key, item in value.items()}
    if type(value) is float and not math.isfinite(value):
        return {"$float": encode_array(np.asarray(value), unit="raw diagnostic")}
    if value is None or type(value) in {str, bool, int, float}:
        return value
    raise ValueError("unsupported JSON payload type: " + type(value).__name__)


def decode_json_payload(value: Any) -> Any:
    """Decode a payload written with :func:`encode_json_payload`."""
    if type(value) is list:
        return [decode_json_payload(item) for item in value]
    if type(value) is dict:
        tags = set(value) & _JSON_CODEC_TAGS
        if tags:
            if len(value) != 1:
                raise ValueError("ambiguous JSON payload tag")
            (tag,) = tags
            if tag == "$tuple":
                if type(value[tag]) is not list:
                    raise ValueError("invalid JSON payload tuple")
                return tuple(decode_json_payload(item) for item in value[tag])
            array = decode_array(value[tag])
            if tag != "$array":
                if array.shape != ():
                    raise ValueError("JSON payload scalar must have scalar shape")
                return array.item() if tag == "$float" else array[()]
            return array
        return {key: decode_json_payload(item) for key, item in value.items()}
    return value


def write_atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically write canonical strict JSON to ``path`` with fsync."""
    if type(value) is not dict:
        raise ValueError("an explicit strict JSON object is required")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    partial = path.with_suffix(path.suffix + ".part")
    fd = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)

Vector: TypeAlias = NDArray[np.float64]
ProgressCallback: TypeAlias = Callable[[str, int, int, str], None]
Disposition: TypeAlias = Literal["completed", "partial", "failed"]
JV2D_RESULT_SCHEMA = "solarlab.jv_2d.result.v1"
_DEFAULT_MAX_NFEV = 200_000


@dataclass(frozen=True)
class SpatialSnapshot2D:
    """2D field and face-current snapshot at one operating voltage."""

    V_app: float
    x: Vector
    y: Vector
    phi: Vector
    n: Vector
    p: Vector
    Jx_n: Vector
    Jy_n: Vector
    Jx_p: Vector
    Jy_p: Vector
    R_total: Vector
    P_ion: Vector | None = None

    def __post_init__(self) -> None:
        for name in ("x", "y", "phi", "n", "p", "Jx_n", "Jy_n", "Jx_p", "Jy_p", "R_total"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        if self.P_ion is not None:
            object.__setattr__(self, "P_ion", frozen_array(self.P_ion))


@dataclass(frozen=True)
class JV2DMetrics:
    """Extracted J_sc-positive photovoltaic figures of merit (SI units, fraction FF/PCE)."""

    V_oc: float
    J_sc: float
    FF: float
    PCE: float
    voc_bracketed: bool


@dataclass(frozen=True)
class JV2DPointAttempt:
    """Per-voltage execution record with explicit failure/acceptance provenance."""

    index: int
    voltage_V: float
    accepted: bool
    termination: str
    bisections: int
    wall_s: float
    state_sha256: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class JV2DResult:
    """Result of a 2D J-V sweep with explicit disposition and diagnostics."""

    V: Vector
    J: Vector
    snapshots: tuple[SpatialSnapshot2D, ...]
    grid_x: Vector
    grid_y: Vector
    lateral_bc: str
    metrics: JV2DMetrics = field(
        default_factory=lambda: JV2DMetrics(0.0, 0.0, 0.0, 0.0, voc_bracketed=False)
    )
    protocol: JV2DProtocol | None = None
    current_components: tuple[MobileIonCurrentComponents2D, ...] = ()
    ion_diagnostics: tuple[MobileIonDiagnostics2D, ...] = ()
    interface_srh_diagnostics: tuple[TwoSidedInterfaceSRHReport2D, ...] = ()
    output_current_A_m2: Vector = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    steady_metrics: SteadyJVMetrics | None = None
    requested_voltages_V: Vector = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    attempts: tuple[JV2DPointAttempt, ...] = ()
    disposition: Disposition = "completed"
    scientific_qualification: bool = False

    def __post_init__(self) -> None:
        for name in ("V", "J", "grid_x", "grid_y"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        out = self.output_current_A_m2
        if out.size == 0 and self.J.size > 0:
            out = -self.J
        object.__setattr__(self, "output_current_A_m2", frozen_array(out))
        req = self.requested_voltages_V
        if req.size == 0 and self.V.size > 0:
            req = self.V
        object.__setattr__(self, "requested_voltages_V", frozen_array(req))
        if self.steady_metrics is None and self.V.size >= 2 and np.unique(self.V).size == self.V.size:
            object.__setattr__(self, "steady_metrics", steady_jv_metrics(self.V, self.output_current_A_m2))


@dataclass(frozen=True)
class JV2DDynamicPairResult:
    """Continuous 2D preconditioning + forward + turnaround-hold + reverse scan result."""

    protocol: DynamicJVPairProtocol2D
    forward: JV2DResult
    reverse: JV2DResult
    preconditioned_state_sha256: str
    turnaround_state_sha256: str | None
    hysteresis_voc_delta_V: float | None
    disposition: Disposition
    scientific_qualification: bool = False


def compute_jv_2d_metrics(
    voltages_V: NDArray[np.float64],
    raw_current_A_m2: NDArray[np.float64],
    *,
    incident_power_W_m2: float = 1000.0,
) -> JV2DMetrics:
    """Extract ``(V_oc, J_sc, FF, PCE, voc_bracketed)`` in the J_sc-positive convention."""
    v = np.asarray(voltages_V, dtype=np.float64)
    j_raw = np.asarray(raw_current_A_m2, dtype=np.float64)
    if v.size < 2 or j_raw.size != v.size or np.unique(v).size != v.size:
        return JV2DMetrics(0.0, 0.0, 0.0, 0.0, voc_bracketed=False)
    order = np.argsort(v)
    v_s = v[order]
    # 2D +y current is negative at V=0 for standard n-i-p stacks; flip when J(0) < 0
    j0 = float(np.interp(0.0, v_s, j_raw[order])) if v_s[0] <= 0.0 <= v_s[-1] else float(j_raw[order][0])
    j_out = -j_raw[order] if j0 < 0.0 else j_raw[order].copy()
    sm = steady_jv_metrics(v_s, j_out, incident_power_W_m2=incident_power_W_m2)
    jsc = float(sm.short_circuit_current_A_m2 or max(0.0, float(np.interp(0.0, v_s, j_out))))
    if sm.open_circuit_voltage_V is None:
        return JV2DMetrics(0.0, jsc, 0.0, 0.0, voc_bracketed=False)
    return JV2DMetrics(
        V_oc=float(sm.open_circuit_voltage_V),
        J_sc=jsc,
        FF=float(sm.fill_factor or 0.0),
        PCE=float(sm.efficiency or 0.0),
        voc_bracketed=True,
    )


def _voltage_schedule(v_max: float, v_step: float) -> Vector:
    maximum = float(v_max)
    step = float(v_step)
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("V_max must be finite and non-negative")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("V_step must be finite and positive")
    return np.arange(0.0, maximum + step / 2.0, step, dtype=np.float64)


def _carrier_bc_label(prepared: PreparedDevice) -> Literal["ohmic", "selective_robin"]:
    contacts = prepared.to_mapping()["contacts"]
    has_robin = any(c.get(k) is not None for c in contacts for k in ("S_n", "S_p"))
    return "selective_robin" if has_robin else "ohmic"


def _coerce_microstructure_input(
    prepared: PreparedDevice,
    microstructure: object,
) -> MicrostructureInput | None:
    if microstructure is None or isinstance(microstructure, MicrostructureInput):
        return microstructure
    elec_layers = [layer for layer in prepared.layers if layer.role != "substrate"]
    if isinstance(microstructure, Mapping):
        return MicrostructureInput.model_validate(dict(microstructure))
    raw_items = getattr(microstructure, "grain_boundaries", microstructure)
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        raise TypeError("microstructure must be MicrostructureInput or sequence of grain boundaries")
    gb_inputs: list[GrainBoundaryInput] = []
    for idx, item in enumerate(raw_items):
        if isinstance(item, GrainBoundaryInput):
            gb_inputs.append(item)
            continue
        proto = JV2DGrainBoundaryProtocol.from_declaration(item)
        target_ids = tuple(lyr.id for lyr in elec_layers if lyr.role == proto.layer_role)
        if not target_ids:
            raise CapabilityError(
                f"grain_boundary_role_missing: no electrical layer has role {proto.layer_role!r}"
            )
        gb_inputs.append(
            GrainBoundaryInput.model_validate(
                {
                    "id": f"gb_{idx}",
                    "layer_ids": list(target_ids),
                    "x_position": proto.x_position_m,
                    "width": proto.width_m,
                    "tau_n": proto.tau_n_s,
                    "tau_p": proto.tau_p_s,
                }
            )
        )
    return MicrostructureInput(grain_boundaries=tuple(gb_inputs))


def _grain_protocols_from_spatial(
    prepared: PreparedDevice,
    geom_spatial: JV2DInput,
) -> tuple[JV2DGrainBoundaryProtocol, ...]:
    declarations = grain_declarations(prepared, geom_spatial)
    role_by_id = {layer.id: layer.role for layer in prepared.layers}
    result: list[JV2DGrainBoundaryProtocol] = []
    for decl in declarations:
        roles = {role_by_id[lid] for lid in decl.layer_ids}
        role = next(iter(roles)) if len(roles) == 1 else "absorber"
        result.append(
            JV2DGrainBoundaryProtocol(
                x_position_m=float(normalize_quantity(decl.x_position, "m")),
                width_m=float(normalize_quantity(decl.width, "m")),
                tau_n_s=float(normalize_quantity(decl.tau_n, "s")),
                tau_p_s=float(normalize_quantity(decl.tau_p, "s")),
                layer_role=role,
            )
        )
    return tuple(result)


def _dwell_time_grid(duration_s: float, divisor: int) -> Vector:
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    n_unif = max(int(divisor), 4)
    dt_unif = duration_s / n_unif
    t_early = min(1.0e-10, dt_unif * 0.125)
    if t_early < dt_unif * 0.5:
        early: Vector = np.geomspace(t_early, dt_unif, 6, dtype=np.float64)
        if n_unif == 1:
            return np.r_[0.0, early]
        tail: Vector = np.linspace(dt_unif, duration_s, n_unif, dtype=np.float64)[1:]
        return np.r_[0.0, early, tail]
    return np.linspace(0.0, duration_s, n_unif + 1, dtype=np.float64)


def _const_fn(val: float) -> Callable[[float], float]:
    fval = float(val)

    def _fn(_t: float) -> float:
        return fval

    return _fn


def _ramp_fn(v_start: float, v_end: float, duration_s: float) -> Callable[[float], float]:
    v0 = float(v_start)
    dv_dt = (float(v_end) - v0) / float(duration_s)

    def _fn(t: float) -> float:
        return v0 + dv_dt * float(t)

    return _fn


def default_ionic_steady_options_1d(ion_node_count: int) -> IonicSteadyOptions:
    return IonicSteadyOptions(
        SteadyOptions(), 1.0e-12, 1.0e6, 1.0e6, 1.0, ion_node_count - 1,
        SolveBudget(60, 600, 60, 24), 30.0,
    )


def device_has_ions(prepared: PreparedDevice) -> tuple[bool, str | None]:
    elec = [layer for layer in prepared.layers if layer.role != "substrate"]
    ion_layers: list[str] = []
    for layer in elec:
        p = dict(layer.parameters)
        if p["P0_neg"] != 0.0 or p["D_ion_neg"] != 0.0:
            raise CapabilityError("dual_ion_2d_unsupported: 2D execution refuses negative/dual ion species")
        if p["P0"] != 0.0 or p["D_ion"] != 0.0:
            ion_layers.append(layer.id)
    if len(ion_layers) > 1:
        raise CapabilityError("multiple_ion_domains_2d_unsupported: 2D supports at most one active ion layer")
    return (len(ion_layers) == 1, ion_layers[0] if ion_layers else None)


def project_prepared_for_electronic_2d(
    prepared: PreparedDevice,
    *,
    strip_ions: bool,
    strip_interfaces: bool,
    strip_grains: bool = False,
) -> PreparedDevice:
    raw = prepared.to_input().editing_data()
    changed = False
    if strip_grains and raw.get("grain_boundaries"):
        raw["grain_boundaries"] = []
        changed = True
    if strip_ions:
        for layer_entry in raw["layers"]:
            params = layer_entry.get("parameters")
            if isinstance(params, dict):
                for key in ION_INPUTS:
                    if key in params:
                        params.pop(key)
                        changed = True
        settings = raw.get("settings")
        if isinstance(settings, dict) and settings.get("ion_steric_diffusion_only"):
            settings["ion_steric_diffusion_only"] = False
            changed = True
    if strip_interfaces:
        settings = raw.get("settings")
        if isinstance(settings, dict) and settings.get("interface_two_sided"):
            settings["interface_two_sided"] = False
            changed = True
        if raw.get("interfaces"):
            cleaned = []
            for entry in raw["interfaces"]:
                item = dict(entry)
                for key in ("v_n", "v_p", "defect"):
                    item.pop(key, None)
                cleaned.append(item)
            raw["interfaces"] = cleaned
            changed = True
    if not changed:
        return prepared
    return resolve_device(
        DeviceInput.model_validate(raw),
        prepared.defaults,
        prepared.resources,
        sources=prepared.sources,
    )


def embed_frozen_ion_excess_2d(
    device_2d: CompiledDevice2D,
    excess_1d_m3: Vector,
) -> CompiledDevice2D:
    if not np.any(excess_1d_m3 != 0.0):
        return device_2d
    col = device_2d.column
    params = col.parameters
    new_nd = params.donor_density_m3 + np.maximum(excess_1d_m3, 0.0)
    new_na = params.acceptor_density_m3 + np.maximum(-excess_1d_m3, 0.0)
    new_params = replace(
        params,
        donor_density_m3=frozen_array(new_nd),
        acceptor_density_m3=frozen_array(new_na),
    )
    col_sha = hashlib.sha256(
        col.sha256.encode() + b":frozen_ion_1d_v1:" + np.ascontiguousarray(excess_1d_m3, dtype="<f8").tobytes()
    ).hexdigest()
    new_col = replace(col, sha256=col_sha, parameters=new_params)
    dev_sha = hashlib.sha256(
        device_2d.identity.encode() + b":frozen_ion_2d_v1:" + col_sha.encode()
    ).hexdigest()
    return replace(device_2d, sha256=dev_sha, column=new_col)


def extract_snapshot_2d(
    device: CompiledDevice2D | CompiledIonicDevice2D,
    state: State2D | IonicState2D,
    *,
    generation_scale: float,
    radiative_reabsorption: tuple[RadiativeReabsorptionSpec2D, ...] = (),
    interface_srh_couplings: tuple[TwoSidedInterfaceSRHCoupling2D, ...] = (),
) -> SpatialSnapshot2D:
    """Extract nodal fields, face currents, and recombination rate into a ``SpatialSnapshot2D``."""
    if isinstance(device, CompiledIonicDevice2D):
        assert isinstance(state, IonicState2D)
        comp = device.electronic_component
        elec_u = state.u[:3 * comp.node_count]
        p_ion_2d: Vector | None = device.full_grid_ion_density(state.positive_ion_density_m3)
        iface_couplings = device.interface_srh_couplings
    else:
        assert isinstance(state, State2D)
        comp = device
        elec_u = state.u
        p_ion_2d = None
        iface_couplings = interface_srh_couplings
    _ = radiative_reabsorption
    g = comp.geometry
    shape = g.shape
    phi, n, p = comp.fields(elec_u, voltage_V=state.voltage_V)
    prob = DC2DProblem(comp, state.voltage_V, generation_scale)
    c_flux = prob.fluxes(elec_u)
    je = -Q * c_flux.electron.flux
    jh = Q * c_flux.hole.flux
    x_cols = g.x_faces_per_row
    jx_n = je[:g.x_face_count].reshape(g.ny, x_cols)
    jy_n = je[g.x_face_count:].reshape(g.ny - 1, g.nx)
    jx_p = jh[:g.x_face_count].reshape(g.ny, x_cols)
    jy_p = jh[g.x_face_count:].reshape(g.ny - 1, g.nx)
    rec = prob.recombination_rate(n, p, derivatives=False).rate.reshape(shape).copy()
    if iface_couplings:
        rep = evaluate_two_sided_interface_srh_2d(
            n.reshape(shape), p.reshape(shape), g.y_geometry.nodes_m, iface_couplings
        )
        rec = rec + rep.volumetric_sink_m3_s
    return SpatialSnapshot2D(
        V_app=float(state.voltage_V),
        x=g.x_m,
        y=g.y_geometry.nodes_m,
        phi=phi.reshape(shape),
        n=n.reshape(shape),
        p=p.reshape(shape),
        Jx_n=jx_n,
        Jy_n=jy_n,
        Jx_p=jx_p,
        Jy_p=jy_p,
        R_total=rec,
        P_ion=p_ion_2d,
    )


def precondition_frozen_lane(
    prepared: PreparedDevice,
    geom_spatial: JV2DInput,
    protocol: JV2DProtocol,
) -> tuple[
    CompiledDevice2D,
    State2D,
    tuple[RadiativeReabsorptionSpec2D, ...],
    tuple[TwoSidedInterfaceSRHCoupling2D, ...],
]:
    """Compile frozen-ion 2D device and run 1D dark/illuminated preconditioner."""
    has_ions, ion_layer_id = device_has_ions(prepared)
    use_iface = protocol.interface_srh == "two_sided_cross_node"
    # Validate dynamic interface flags and two-sided SRH before projecting interfaces
    base_clean = project_prepared_for_electronic_2d(
        prepared, strip_ions=has_ions, strip_interfaces=use_iface
    )
    device_2d = compile_device_2d(base_clean, geom_spatial)
    iface_couplings = build_two_sided_interface_srh_2d(
        prepared, device_2d, interface_srh_mode=protocol.interface_srh
    )
    rr_specs = build_radiative_reabsorption_2d(prepared, device_2d) if protocol.illuminated else ()

    ny_per = int(cast(int, geom_spatial.Ny_per_layer))
    n_layers = len([lyr for lyr in prepared.layers if lyr.role != "substrate"])
    ny_nodes = device_2d.geometry.ny

    # 1. Solve 1D dark equilibrium (and 1D illuminated finite-time settle if requested)
    excess_1d: Vector = np.zeros(ny_nodes, dtype=np.float64)
    if has_ions and ion_layer_id is not None:
        prep_1d_ion = project_prepared_for_electronic_2d(
            prepared, strip_ions=False, strip_interfaces=use_iface, strip_grains=True
        )
        # If carrier contacts are ohmic and Neumann-x, use Nx=1 auxiliary 2D ionic solver so
        # reabsorption and two-sided SRH participate in the 1D illuminated preconditioner!
        if protocol.carrier_boundary_condition == "ohmic":
            aux_spatial = JV2DInput(
                kind="jv_2d",
                lateral_length=geom_spatial.lateral_length,
                Nx=1,
                Ny_per_layer=ny_per,
                lateral_bc="neumann",
                microstructure=MicrostructureInput(grain_boundaries=()),
                ion_dynamics="frozen",
                interface_srh=protocol.interface_srh,
            )
            aux_ionic = compile_ionic_device_2d(
                project_prepared_for_electronic_2d(prepared, strip_ions=False, strip_interfaces=False, strip_grains=True),
                aux_spatial,
                ion_domain_layer=ion_layer_id,
                ion_domain_endpoints=("blocking", "blocking"),
            )
            dark_elec_f = solve_steady_state(
                aux_ionic.electronic_component.column, voltage_V=0.0, generation_scale=0.0
            )
            if dark_elec_f.state is None:
                raise RuntimeError("1D dark electronic preconditioner failed")
            settled_ion_state = aux_ionic.state(
                np.r_[
                    aux_ionic.electronic_component.extrude_coordinates(dark_elec_f.state.u),
                    np.zeros(aux_ionic.ion_node_count, dtype=np.float64),
                ],
                voltage_V=0.0,
            )
            if protocol.illuminated and float(aux_ionic.positive_ion_diffusivity_faces_m2_s[0]) > 0.0:
                assert protocol.initial_state_settle_s is not None
                t_grid = _dwell_time_grid(protocol.initial_state_settle_s, 20)
                run_pre_ion = run_ionic_backward_euler_2d(
                    aux_ionic,
                    settled_ion_state,
                    t_grid,
                    voltage_fn=_const_fn(0.0),
                    generation_fn=lambda t: 0.0 if t <= 0.0 else 1.0,
                )
                if not run_pre_ion.accepted:
                    raise RuntimeError(f"1D illuminated ionic preconditioner failed: {run_pre_ion.error}")
                settled_ion_state = run_pre_ion.states[-1]
            elif protocol.illuminated:
                # Immobile positive background (D_ion == 0): settle carriers only
                assert protocol.initial_state_settle_s is not None
                t_grid = _dwell_time_grid(protocol.initial_state_settle_s, 20)
                aux_elec = aux_ionic.electronic_component
                elec_s0 = aux_elec.state(settled_ion_state.u[:3 * aux_elec.node_count], voltage_V=0.0)
                run_c = run_backward_euler_2d(
                    aux_elec,
                    elec_s0,
                    t_grid,
                    voltage_fn=lambda _t: 0.0,
                    generation_fn=lambda t: 0.0 if t <= 0.0 else 1.0,
                    options=BackwardEuler2DOptions(
                        radiative_reabsorption=aux_ionic.radiative_reabsorption,
                        interface_srh_couplings=aux_ionic.interface_srh_couplings,
                    ),
                )
                if not run_c.accepted:
                    raise RuntimeError(f"1D illuminated frozen-ion preconditioner failed: {run_c.error}")
                settled_elec = run_c.states[-1]
                settled_ion_state = aux_ionic.state(
                    np.r_[settled_elec.u, settled_ion_state.u[3 * aux_elec.node_count:]],
                    voltage_V=0.0,
                )
            # Extract column 0 (1D profile) from the 2-column (Nx=1) auxiliary solve
            aux_nx = aux_ionic.electronic_component.geometry.nx  # 2
            u_elec_2d = settled_ion_state.u[:3 * aux_ionic.node_count].reshape(3, ny_nodes, aux_nx)
            col_u_1d = u_elec_2d[:, :, 0].ravel()
            ion_p_col0 = settled_ion_state.positive_ion_density_m3[:, 0]
            ref_col0 = aux_ionic.positive_ion_reference_m3.reshape(aux_ionic.ion_geometry.ny, aux_nx)[:, 0]
            ratio_col0 = aux_ionic.ion_charge_weight_ratio.reshape(aux_ionic.ion_geometry.ny, aux_nx)[:, 0]
            excess_1d[aux_ionic.ion_row_indices] = (ion_p_col0 - ref_col0) * ratio_col0
            device_2d = embed_frozen_ion_excess_2d(device_2d, excess_1d)
            init_2d = device_2d.state(device_2d.extrude_coordinates(col_u_1d), voltage_V=0.0)
            return device_2d, init_2d, rr_specs, iface_couplings

        # Selective Robin contacts with frozen ion background: solve 1D ionic equilibrium
        ionic_1d = compile_ionic_device_1d(
            prep_1d_ion,
            intervals_per_layer=(ny_per,) * n_layers,
            ion_domain_layer=ion_layer_id,
            ion_domain_endpoints=("blocking", "blocking"),
        )
        dark_1d_ion = solve_ionic_steady_state(
            ionic_1d,
            voltage_V=0.0,
            generation_scale=0.0,
            options=default_ionic_steady_options_1d(ionic_1d.ion_node_count),
        )
        if not dark_1d_ion.accepted or dark_1d_ion.state is None:
            raise RuntimeError(f"1D Robin ionic dark equilibrium failed: {dark_1d_ion.error}")
        col_u_1d = dark_1d_ion.state.u[:3 * ionic_1d.node_count]
        ion_p_1d = dark_1d_ion.state.positive_ion_density_m3
        excess_1d[ionic_1d.ion_node_indices] = (
            ion_p_1d - ionic_1d.positive_ion_reference_m3
        ) * ionic_1d.ion_charge_weight_ratio
        device_2d = embed_frozen_ion_excess_2d(device_2d, excess_1d)
        if not protocol.illuminated:
            return device_2d, device_2d.state(device_2d.extrude_coordinates(col_u_1d), voltage_V=0.0), rr_specs, iface_couplings
    else:
        dark_1d = solve_steady_state(device_2d.column, voltage_V=0.0, generation_scale=0.0)
        if dark_1d.state is None:
            raise RuntimeError("1D dark equilibrium failed")
        col_u_1d = dark_1d.state.u
        if not protocol.illuminated:
            return device_2d, device_2d.state(device_2d.extrude_coordinates(col_u_1d), voltage_V=0.0), rr_specs, iface_couplings

    # Illuminated finite-time 1D settle on grain-free Nx=1 column pair
    assert protocol.initial_state_settle_s is not None
    aux_geom = JV2DInput(
        kind="jv_2d",
        lateral_length=geom_spatial.lateral_length,
        Nx=1,
        Ny_per_layer=ny_per,
        lateral_bc="neumann",
        microstructure=MicrostructureInput(grain_boundaries=()),
    )
    aux_clean = project_prepared_for_electronic_2d(
        prepared, strip_ions=has_ions, strip_interfaces=use_iface, strip_grains=True
    )
    aux_dev = embed_frozen_ion_excess_2d(compile_device_2d(aux_clean, aux_geom), excess_1d)
    aux_rr = build_radiative_reabsorption_2d(prepared, aux_dev)
    aux_if = build_two_sided_interface_srh_2d(prepared, aux_dev, interface_srh_mode=protocol.interface_srh)
    aux_s0 = aux_dev.state(aux_dev.extrude_coordinates(col_u_1d), voltage_V=0.0)
    t_grid = _dwell_time_grid(protocol.initial_state_settle_s, 20)
    run_pre = run_backward_euler_2d(
        aux_dev,
        aux_s0,
        t_grid,
        voltage_fn=lambda _t: 0.0,
        generation_fn=lambda t: 0.0 if t <= 0.0 else 1.0,
        options=BackwardEuler2DOptions(
            radiative_reabsorption=aux_rr,
            interface_srh_couplings=aux_if,
        ),
    )
    if not run_pre.accepted:
        raise RuntimeError(f"1D illuminated finite-time preconditioner failed: {run_pre.error}")
    u_settled = run_pre.states[-1].u.reshape(3, ny_nodes, aux_dev.geometry.nx)[:, :, 0].ravel()
    return device_2d, device_2d.state(device_2d.extrude_coordinates(u_settled), voltage_V=0.0), rr_specs, iface_couplings


def _precondition_mobile_ion_lane(
    prepared: PreparedDevice,
    exec_spatial: JV2DInput,
    protocol: JV2DProtocol,
) -> tuple[CompiledIonicDevice2D, IonicState2D]:
    """Compile mobile-ion 2D device and run 1D dark/illuminated preconditioner."""
    _, ion_layer_id = device_has_ions(prepared)
    ionic_2d = compile_ionic_device_2d(
        prepared,
        exec_spatial,
        ion_domain_layer=ion_layer_id,
        ion_domain_endpoints=("blocking", "blocking"),
    )
    aux_spatial = JV2DInput(
        kind="jv_2d",
        lateral_length=exec_spatial.lateral_length,
        Nx=1,
        Ny_per_layer=exec_spatial.Ny_per_layer,
        lateral_bc="neumann",
        microstructure=MicrostructureInput(grain_boundaries=()),
        ion_dynamics="single_mobile",
        interface_srh=protocol.interface_srh,
    )
    aux_prep = project_prepared_for_electronic_2d(
        prepared, strip_ions=False, strip_interfaces=False, strip_grains=True
    )
    aux_ionic = compile_ionic_device_2d(
        aux_prep,
        aux_spatial,
        ion_domain_layer=ionic_2d.domain_layer_id,
        ion_domain_endpoints=("blocking", "blocking"),
    )
    dark_elec = solve_steady_state(
        aux_ionic.electronic_component.column, voltage_V=0.0, generation_scale=0.0
    )
    if dark_elec.state is None:
        raise RuntimeError("1D mobile-ion electronic dark equilibrium failed")
    settled = aux_ionic.state(
        np.r_[
            aux_ionic.electronic_component.extrude_coordinates(dark_elec.state.u),
            np.zeros(aux_ionic.ion_node_count, dtype=np.float64),
        ],
        voltage_V=0.0,
    )
    if protocol.illuminated:
        assert protocol.initial_state_settle_s is not None
        t_grid = _dwell_time_grid(protocol.initial_state_settle_s, 24)
        run_pre = run_ionic_backward_euler_2d(
            aux_ionic,
            settled,
            t_grid,
            voltage_fn=_const_fn(0.0),
            generation_fn=lambda t: 0.0 if t <= 0.0 else 1.0,
        )
        if not run_pre.accepted:
            raise RuntimeError(f"1D mobile-ion illuminated preconditioner failed: {run_pre.error}")
        settled = run_pre.states[-1]
    ny_elec = ionic_2d.electronic_component.geometry.ny
    ny_ion = ionic_2d.ion_geometry.ny
    aux_nx = aux_ionic.electronic_component.geometry.nx
    u_elec_1d = settled.u[:3 * aux_ionic.node_count].reshape(3, ny_elec, aux_nx)[:, :, 0].ravel()
    y_ion_1d = settled.u[3 * aux_ionic.node_count:].reshape(ny_ion, aux_nx)[:, 0]
    u_2d = np.r_[
        ionic_2d.electronic_component.extrude_coordinates(u_elec_1d),
        np.repeat(y_ion_1d, ionic_2d.ion_geometry.nx),
    ]
    return ionic_2d, ionic_2d.state(u_2d, voltage_V=0.0)


def _bake_electronic_reabsorption_2d(
    device: CompiledDevice2D,
    state: State2D,
    rr_specs: tuple[RadiativeReabsorptionSpec2D, ...],
) -> CompiledDevice2D:
    g = device.geometry
    shape = g.shape
    _, n, p = device.fields(state.u, voltage_V=state.voltage_V)
    b_rad = device.recombination.coefficients.radiative_m3_s.reshape(shape)
    ni_sq = device.recombination.coefficients.ni_sq_m6.reshape(shape)
    base_g = device.generation_source_m3_s.reshape(shape)
    aug = recompute_g_with_rad_2d(
        G_optical=base_g,
        n=n.reshape(shape),
        p=p.reshape(shape),
        B_rad=b_rad,
        ni_sq=ni_sq,
        x=g.x_m,
        y=g.y_geometry.nodes_m,
        absorber_y_ranges=tuple((s.y_start, s.y_stop) for s in rr_specs),
        absorber_p_esc=tuple(s.escape_probability for s in rr_specs),
        absorber_areas=tuple(s.area_m2 for s in rr_specs),
    )
    gen_y = np.mean(aug, axis=1)
    new_gen = replace(device.generation_y, source_m3_s=frozen_array(gen_y))
    baked_sha = hashlib.sha256(
        device.identity.encode() + b":baked_rr_2d:" + np.ascontiguousarray(gen_y, dtype="<f8").tobytes()
    ).hexdigest()
    return replace(device, sha256=baked_sha, generation_y=new_gen)


def _bake_ionic_reabsorption_2d(
    device: CompiledIonicDevice2D,
    state: IonicState2D,
) -> CompiledIonicDevice2D:
    if not device.radiative_reabsorption:
        return device
    baked_elec = _bake_electronic_reabsorption_2d(
        device.electronic_component, state.electronic_state, device.radiative_reabsorption
    )
    manifest = dict(device.binding_manifest)
    manifest["radiative_reabsorption_baked_step"] = True
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return CompiledIonicDevice2D(
        device_id=device.device_id,
        sha256=hashlib.sha256(manifest_json).hexdigest(),
        original_prepared_sha256=device.original_prepared_sha256,
        electronic_component=baked_elec,
        ion_geometry=device.ion_geometry,
        domain_layer_id=device.domain_layer_id,
        ion_row_indices=device.ion_row_indices,
        positive_ion_reference_m3=device.positive_ion_reference_m3,
        positive_ion_site_limit_m3=device.positive_ion_site_limit_m3,
        positive_ion_logit_reference=device.positive_ion_logit_reference,
        positive_ion_diffusivity_faces_m2_s=device.positive_ion_diffusivity_faces_m2_s,
        steric_model=device.steric_model,
        binding_manifest_json=manifest_json.decode(),
        radiative_reabsorption=(),
        interface_srh_couplings=device.interface_srh_couplings,
    )


def build_jv_2d_execution_protocol(
    prepared: PreparedDevice,
    microstructure: object = None,
    *,
    lateral_length: float,
    Nx: int,
    V_max: float,
    V_step: float,
    illuminated: bool = True,
    lateral_bc: Literal["periodic", "neumann"] = "periodic",
    Ny_per_layer: int = 20,
    settle_t: float = 1.0e-7,
    save_snapshots: bool = True,
    ion_dynamics: Literal["frozen", "single_mobile"] = "frozen",
    interface_srh: Literal["off", "two_sided_cross_node"] = "off",
    rtol: float = 1.0e-6,
    atol: object = 1.0e-8,
    max_nfev_per_solve: int = _DEFAULT_MAX_NFEV,
    max_bisect: int = 6,
    ion_inventory_rtol: float = 1.0e-9,
    initial_state_settle_s: float = 1.0e-3,
    solver_method: Literal["Radau", "backward_euler"] = "Radau",
    implicit_legacy_protocol: bool = False,
) -> JV2DProtocol:
    """Build the exact canonical ``JV2DProtocol`` for a ``PreparedDevice`` 2D J-V sweep."""
    ms_input = _coerce_microstructure_input(prepared, microstructure)
    geom_spatial = JV2DInput(
        kind="jv_2d",
        lateral_length=lateral_length,
        Nx=Nx,
        Ny_per_layer=Ny_per_layer,
        lateral_bc=lateral_bc,
        microstructure=ms_input,
    )
    has_ions, _ = device_has_ions(prepared)
    clean_prep = project_prepared_for_electronic_2d(
        prepared,
        strip_ions=has_ions,
        strip_interfaces=(interface_srh == "two_sided_cross_node"),
    )
    dev_preview = compile_device_2d(clean_prep, geom_spatial)
    voltages = _voltage_schedule(V_max, V_step)
    grain_protos = _grain_protocols_from_spatial(prepared, geom_spatial)
    temp_K = float(prepared.to_mapping()["settings"]["T"])
    return build_jv_2d_protocol(
        temperature_K=temp_K,
        illuminated=illuminated,
        x_coordinates_m=dev_preview.geometry.x_m,
        y_coordinates_m=dev_preview.geometry.y_geometry.nodes_m,
        voltages_V=voltages,
        dwell_time_per_voltage_s=settle_t,
        ion_dynamics=ion_dynamics,
        carrier_boundary_condition=_carrier_bc_label(prepared),
        interface_srh=interface_srh,
        lateral_bc=lateral_bc,
        solver_rtol=rtol,
        solver_atol=atol,
        max_nfev_per_solve=max_nfev_per_solve,
        max_bisect=max_bisect,
        ion_inventory_rtol=ion_inventory_rtol,
        save_snapshots=save_snapshots,
        grain_boundaries=grain_protos,
        initial_state_settle_s=initial_state_settle_s,
        solver_method=solver_method,
        implicit_legacy_protocol=implicit_legacy_protocol,
    )


def _extract_prepared_and_controls(
    source: PreparedDevice | PreparedSpatialExperiment,
    spatial_input: JV2DInput | None,
    kwargs: dict[str, Any],
) -> tuple[PreparedDevice, dict[str, Any], MicrostructureInput | None, JV2DProtocol | JV2DProtocolInput | None]:
    if isinstance(source, PreparedSpatialExperiment):
        inp = source.to_input()
        if not isinstance(inp.experiment, JV2DInput):
            raise ValueError("PreparedSpatialExperiment does not contain a jv_2d experiment")
        prepared = resolve_device(inp.device, source.defaults, source.resources, sources=source.sources)
        spatial_input = inp.experiment
    elif isinstance(source, PreparedDevice):
        prepared = source
    else:
        raise TypeError("run_jv_sweep_2d requires PreparedDevice or PreparedSpatialExperiment")

    defaults: dict[str, Any] = {
        "lateral_length": 1.0e-6,
        "Nx": 10,
        "Ny_per_layer": 20,
        "V_max": 1.2,
        "V_step": 0.05,
        "illuminated": True,
        "save_snapshots": True,
        "settle_t": 1.0e-7,
        "rtol": 1.0e-6,
        "atol": 1.0e-8,
        "max_nfev_per_solve": _DEFAULT_MAX_NFEV,
        "max_bisect": 6,
        "ion_inventory_rtol": 1.0e-9,
        "initial_state_settle_s": 1.0e-3,
        "ion_dynamics": "frozen",
        "interface_srh": "off",
        "protocol_mode": "compatibility",
    }
    try:
        cat_defaults = prepared.defaults.experiment_defaults_for("jv_2d")
        defaults.update(cat_defaults)
    except Exception:
        pass

    ms_input: MicrostructureInput | None = None
    supplied_proto: JV2DProtocol | JV2DProtocolInput | None = kwargs.get("jv_2d_protocol")
    if spatial_input is not None:
        norm = spatial_input.normalized_data()
        for k in defaults:
            if k in norm and norm[k] is not None:
                defaults[k] = norm[k]
        if spatial_input.componentwise_atol is not None:
            defaults["atol"] = spatial_input.componentwise_atol
        if spatial_input.lateral_bc is not None:
            defaults["lateral_bc"] = spatial_input.lateral_bc
        ms_input = spatial_input.microstructure
        if spatial_input.jv_2d_protocol is not None and supplied_proto is None:
            supplied_proto = spatial_input.jv_2d_protocol

    for k, v in kwargs.items():
        if k in {"microstructure", "jv_2d_protocol"} or v is None:
            continue
        if k in {"lateral_length", "V_max", "V_step", "settle_t", "initial_state_settle_s"} and not isinstance(v, (int, float)):
            unit = "m" if k == "lateral_length" else ("s" if "settle" in k else "V")
            defaults[k] = normalize_quantity(v, unit)
        else:
            defaults[k] = v

    if "microstructure" in kwargs and kwargs["microstructure"] is not None:
        ms_input = _coerce_microstructure_input(prepared, kwargs["microstructure"])

    # Effective lateral_bc default rule: neumann when grains exist, else periodic
    if defaults.get("lateral_bc") is None:
        active_override = ms_input is not None and bool(ms_input.editing_data())
        eff_grains = (
            ms_input.grain_boundaries
            if active_override and ms_input is not None
            else prepared.to_input().grain_boundaries
        )
        defaults["lateral_bc"] = "neumann" if eff_grains else "periodic"

    return prepared, defaults, ms_input, supplied_proto


def run_jv_sweep_2d(
    prepared_or_experiment: PreparedDevice | PreparedSpatialExperiment,
    spatial_input: JV2DInput | None = None,
    *,
    microstructure: object = None,
    lateral_length: float | None = None,
    Nx: int | None = None,
    V_max: float | None = None,
    V_step: float | None = None,
    illuminated: bool | None = None,
    lateral_bc: Literal["periodic", "neumann"] | None = None,
    Ny_per_layer: int | None = None,
    settle_t: float | None = None,
    progress: ProgressCallback | None = None,
    save_snapshots: bool | None = None,
    ion_dynamics: Literal["frozen", "single_mobile"] | None = None,
    interface_srh: Literal["off", "two_sided_cross_node"] | None = None,
    rtol: float | None = None,
    atol: object = None,
    max_nfev_per_solve: int | None = None,
    max_bisect: int | None = None,
    ion_inventory_rtol: float | None = None,
    initial_state_settle_s: float | None = None,
    jv_2d_protocol: JV2DProtocol | JV2DProtocolInput | None = None,
    protocol_mode: ProtocolMode | None = None,
    on_point_failure: Literal["raise", "record_partial"] = "raise",
) -> JV2DResult:
    """Execute a 2D J-V dwell sweep on the native ``solarlab`` 2D core."""
    if on_point_failure not in {"raise", "record_partial"}:
        raise ValueError("on_point_failure must be 'raise' or 'record_partial'")
    prepared, ctrl, ms_input, supplied_proto = _extract_prepared_and_controls(
        prepared_or_experiment,
        spatial_input,
        {
            "microstructure": microstructure,
            "lateral_length": lateral_length,
            "Nx": Nx,
            "V_max": V_max,
            "V_step": V_step,
            "illuminated": illuminated,
            "lateral_bc": lateral_bc,
            "Ny_per_layer": Ny_per_layer,
            "settle_t": settle_t,
            "save_snapshots": save_snapshots,
            "ion_dynamics": ion_dynamics,
            "interface_srh": interface_srh,
            "rtol": rtol,
            "atol": atol,
            "max_nfev_per_solve": max_nfev_per_solve,
            "max_bisect": max_bisect,
            "ion_inventory_rtol": ion_inventory_rtol,
            "initial_state_settle_s": initial_state_settle_s,
            "jv_2d_protocol": jv_2d_protocol,
            "protocol_mode": protocol_mode,
        },
    )

    eff_ion_dyn = cast(Literal["frozen", "single_mobile"], ctrl["ion_dynamics"])
    eff_iface = cast(Literal["off", "two_sided_cross_node"], ctrl["interface_srh"])
    eff_lat_bc = cast(Literal["periodic", "neumann"], ctrl["lateral_bc"])
    eff_mode = cast(ProtocolMode, ctrl["protocol_mode"])

    extended = eff_ion_dyn != "frozen" or eff_iface != "off"
    if extended and eff_mode != "research_strict":
        raise ImplicitProtocolError(
            "mobile-ion or interface-SRH 2D J-V requires protocol_mode='research_strict' "
            "and an explicit matching jv_2d_protocol"
        )

    # Validate BC / microstructure capability rules fail-closed before solving
    geom_spatial = JV2DInput(
        kind="jv_2d",
        lateral_length=float(ctrl["lateral_length"]),
        Nx=int(ctrl["Nx"]),
        Ny_per_layer=int(ctrl["Ny_per_layer"]),
        lateral_bc=eff_lat_bc,
        microstructure=ms_input,
    )
    grain_decls = grain_declarations(prepared, geom_spatial)
    if grain_decls and eff_lat_bc != "neumann":
        raise CapabilityError("periodic_grain_2d_unsupported: grain boundaries require Neumann-x")
    if eff_ion_dyn == "single_mobile" and eff_lat_bc != "neumann":
        raise CapabilityError("periodic_ionic_2d_unsupported: 2D mobile-ion requires Neumann-x")
    if eff_iface != "off" and eff_lat_bc != "neumann":
        raise CapabilityError("periodic_interface_srh_2d_unsupported: two-sided SRH requires Neumann-x")
    carrier_bc = _carrier_bc_label(prepared)
    if extended and carrier_bc != "ohmic":
        raise CapabilityError("selective_robin_extended_2d_unsupported: requires ohmic contacts")

    expected_proto = build_jv_2d_execution_protocol(
        prepared,
        ms_input,
        lateral_length=float(ctrl["lateral_length"]),
        Nx=int(ctrl["Nx"]),
        V_max=float(ctrl["V_max"]),
        V_step=float(ctrl["V_step"]),
        illuminated=bool(ctrl["illuminated"]),
        lateral_bc=eff_lat_bc,
        Ny_per_layer=int(ctrl["Ny_per_layer"]),
        settle_t=float(ctrl["settle_t"]),
        save_snapshots=bool(ctrl["save_snapshots"]),
        ion_dynamics=eff_ion_dyn,
        interface_srh=eff_iface,
        rtol=float(ctrl["rtol"]),
        atol=ctrl["atol"],
        max_nfev_per_solve=int(ctrl["max_nfev_per_solve"]),
        max_bisect=int(ctrl["max_bisect"]),
        ion_inventory_rtol=float(ctrl["ion_inventory_rtol"]),
        initial_state_settle_s=float(ctrl["initial_state_settle_s"]),
        implicit_legacy_protocol=True,
    )
    resolved_proto = resolve_jv_2d_protocol(supplied_proto, expected_proto, mode=eff_mode)
    mobile_ions = resolved_proto.state_topology == "single_positive_mobile_ion"
    gen_scale = 1.0 if resolved_proto.illuminated else 0.0

    voltages: Vector = np.asarray(resolved_proto.voltage_values_V, dtype=np.float64)
    j_vals: list[float] = []
    v_accepted: list[float] = []
    snapshots: list[SpatialSnapshot2D] = []
    curr_reports: list[MobileIonCurrentComponents2D] = []
    ion_reports: list[MobileIonDiagnostics2D] = []
    iface_reports: list[TwoSidedInterfaceSRHReport2D] = []
    attempts: list[JV2DPointAttempt] = []

    max_it = min(max(resolved_proto.max_nfev_per_solve // 200, 25), 80)
    pt_budget = SolveBudget(max_it, max(resolved_proto.max_nfev_per_solve, max_it * 10), max_it * 2, 24)
    g_fn = _const_fn(gen_scale)

    if mobile_ions:
        exec_spatial = JV2DInput(
            kind="jv_2d",
            lateral_length=float(ctrl["lateral_length"]),
            Nx=int(ctrl["Nx"]),
            Ny_per_layer=int(ctrl["Ny_per_layer"]),
            lateral_bc=resolved_proto.lateral_bc,
            microstructure=ms_input,
            ion_dynamics="single_mobile",
            interface_srh=resolved_proto.interface_srh,
        )
        ionic_dev, ion_state = _precondition_mobile_ion_lane(prepared, exec_spatial, resolved_proto)
        polarity = float(ionic_dev.electronic_component.column.voltage_polarity)
        grid_x = ionic_dev.electronic_component.geometry.x_m
        grid_y = ionic_dev.electronic_component.geometry.y_geometry.nodes_m
        d_grid = _dwell_time_grid(
            resolved_proto.dwell_time_per_voltage_s, resolved_proto.solver_max_step_divisor
        )
        ion_opts = IonicBackwardEuler2DOptions(
            budget=pt_budget,
            max_bisect=max(resolved_proto.max_bisect, 4),
            ion_inventory_rtol=max(resolved_proto.ion_inventory_rtol, 1.0e-12),
        )
        ion_x = ionic_dev.ion_geometry.x_m
        ion_y = ionic_dev.ion_geometry.y_geometry.nodes_m
        ion_areas = ionic_dev.ion_geometry.contract_geometry.volumes.reshape(ionic_dev.ion_geometry.shape)
        ion_lim_2d = ionic_dev.positive_ion_site_limit_m3.reshape(ionic_dev.ion_geometry.shape)
        for idx, v_val in enumerate(voltages):
            pt_start = time.monotonic()
            prev_ion_2d = ion_state.positive_ion_density_m3
            v_fn = _const_fn(float(v_val))
            run_pt = run_ionic_backward_euler_2d(
                ionic_dev, ion_state, d_grid, voltage_fn=v_fn, generation_fn=g_fn, options=ion_opts
            )
            if not run_pt.accepted and ionic_dev.radiative_reabsorption and resolved_proto.illuminated:
                baked_dev = _bake_ionic_reabsorption_2d(ionic_dev, ion_state)
                baked_s0 = baked_dev.state(ion_state.u, voltage_V=ion_state.voltage_V)
                run_baked = run_ionic_backward_euler_2d(
                    baked_dev, baked_s0, d_grid, voltage_fn=v_fn, generation_fn=g_fn, options=ion_opts
                )
                if run_baked.accepted:
                    settled_u = run_baked.states[-1].u
                    ion_state = ionic_dev.state(settled_u, voltage_V=float(v_val))
                    run_pt = replace(run_baked, states=(*run_baked.states[:-1], ion_state))
            if not run_pt.accepted:
                err_msg = run_pt.error or f"2D mobile-ion dwell failed at V={v_val:.4f} V"
                attempts.append(
                    JV2DPointAttempt(
                        idx, float(v_val), False, run_pt.termination, run_pt.bisections,
                        time.monotonic() - pt_start, None, err_msg,
                    )
                )
                if on_point_failure == "raise":
                    raise RuntimeError(err_msg)
                break

            ion_state = run_pt.states[-1]
            snap = extract_snapshot_2d(ionic_dev, ion_state, generation_scale=gen_scale)
            if ionic_dev.interface_srh_couplings:
                iface_reports.append(
                    evaluate_two_sided_interface_srh_2d(
                        snap.n, snap.p, grid_y, ionic_dev.interface_srh_couplings
                    )
                )
            curr = compute_mobile_ion_current_components_2d(
                ionic_dev,
                ion_state,
                generation_scale=gen_scale,
                applied_voltage_rate_V_s=resolved_proto.applied_voltage_rate_at_sampling_V_s,
            )
            term_ion_2d = ion_state.positive_ion_density_m3
            diag = assess_mobile_ion_terminal_2d(
                ion_x,
                ion_y,
                prev_ion_2d,
                term_ion_2d,
                ion_lim_2d,
                terminal_electron_density=snap.n,
                terminal_hole_density=snap.p,
                inventory_rtol=max(resolved_proto.ion_inventory_rtol, 1.0e-12),
                control_volume_areas_m2=ion_areas,
            )
            if not diag.passed:
                err_msg = "2D J-V point failed mobile-ion diagnostics: " + ", ".join(diag.violations)
                attempts.append(
                    JV2DPointAttempt(
                        idx, float(v_val), False, "ion_diagnostics_failed", run_pt.bisections,
                        time.monotonic() - pt_start, None, err_msg,
                    )
                )
                if on_point_failure == "raise":
                    raise RuntimeError(err_msg)
                break

            v_accepted.append(float(v_val))
            j_vals.append(curr.terminal_total_A_m2)
            curr_reports.append(curr)
            ion_reports.append(diag)
            if resolved_proto.save_snapshots:
                snapshots.append(snap)
            attempts.append(
                JV2DPointAttempt(
                    idx, float(v_val), True, "converged", run_pt.bisections,
                    time.monotonic() - pt_start, state_digest(ion_state.u), None,
                )
            )
            if progress is not None:
                progress("jv_2d", idx + 1, int(voltages.size), f"V = {v_val:.3f} V")
    else:
        elec_dev, elec_state, rr_specs, iface_couplings = precondition_frozen_lane(
            prepared, geom_spatial, resolved_proto
        )
        polarity = float(elec_dev.column.voltage_polarity)
        grid_x = elec_dev.geometry.x_m
        grid_y = elec_dev.geometry.y_geometry.nodes_m
        d_grid = _dwell_time_grid(
            resolved_proto.dwell_time_per_voltage_s, resolved_proto.solver_max_step_divisor
        )
        be_opts = BackwardEuler2DOptions(
            budget=pt_budget,
            max_bisect=max(resolved_proto.max_bisect, 4),
            radiative_reabsorption=rr_specs,
            interface_srh_couplings=iface_couplings,
        )
        for idx, v_val in enumerate(voltages):
            pt_start = time.monotonic()
            fv = float(v_val)
            v_fn = _const_fn(fv)
            run_pt_e = run_backward_euler_2d(
                elec_dev,
                elec_state,
                d_grid,
                voltage_fn=v_fn,
                generation_fn=g_fn,
                options=be_opts,
            )
            if not run_pt_e.accepted and rr_specs and resolved_proto.illuminated:
                baked_elec = _bake_electronic_reabsorption_2d(elec_dev, elec_state, rr_specs)
                baked_s0_e = baked_elec.state(elec_state.u, voltage_V=elec_state.voltage_V)
                baked_opts = replace(be_opts, radiative_reabsorption=())
                run_baked_e = run_backward_euler_2d(
                    baked_elec,
                    baked_s0_e,
                    d_grid,
                    voltage_fn=v_fn,
                    generation_fn=g_fn,
                    options=baked_opts,
                )
                if run_baked_e.accepted:
                    elec_state = elec_dev.state(run_baked_e.states[-1].u, voltage_V=fv)
                    run_pt_e = replace(run_baked_e, states=(*run_baked_e.states[:-1], elec_state))
            if not run_pt_e.accepted:
                err_msg = run_pt_e.error or f"2D frozen-ion dwell failed at V={fv:.4f} V"
                attempts.append(
                    JV2DPointAttempt(
                        idx, fv, False, run_pt_e.termination, run_pt_e.bisections,
                        time.monotonic() - pt_start, None, err_msg,
                    )
                )
                if on_point_failure == "raise":
                    raise RuntimeError(err_msg)
                break

            elec_state = run_pt_e.states[-1]
            snap = extract_snapshot_2d(
                elec_dev,
                elec_state,
                generation_scale=gen_scale,
                radiative_reabsorption=rr_specs,
                interface_srh_couplings=iface_couplings,
            )
            if iface_couplings:
                iface_reports.append(
                    evaluate_two_sided_interface_srh_2d(snap.n, snap.p, grid_y, iface_couplings)
                )
            ports_j = compute_carrier_only_terminal_current_2d(
                elec_dev,
                elec_state,
                generation_scale=gen_scale,
                radiative_reabsorption=rr_specs,
                interface_srh_couplings=iface_couplings,
            )
            v_accepted.append(fv)
            j_vals.append(float(ports_j[1]))
            if resolved_proto.save_snapshots:
                snapshots.append(snap)
            attempts.append(
                JV2DPointAttempt(
                    idx, fv, True, "converged", run_pt_e.bisections,
                    time.monotonic() - pt_start, state_digest(elec_state.u), None,
                )
            )
            if progress is not None:
                progress("jv_2d", idx + 1, int(voltages.size), f"V = {fv:.3f} V")

    v_out: Vector = np.asarray(v_accepted, dtype=np.float64)
    j_arr: Vector = np.asarray(j_vals, dtype=np.float64)
    j_out = -polarity * j_arr
    disp: Disposition = (
        "completed"
        if v_out.size == voltages.size
        else ("partial" if v_out.size > 0 else "failed")
    )
    metrics = compute_jv_2d_metrics(v_out, j_arr)
    sm = steady_jv_metrics(v_out, j_out) if v_out.size >= 2 else None
    return JV2DResult(
        V=v_out,
        J=j_arr,
        snapshots=tuple(snapshots),
        grid_x=grid_x,
        grid_y=grid_y,
        lateral_bc=resolved_proto.lateral_bc,
        metrics=metrics,
        protocol=resolved_proto,
        current_components=tuple(curr_reports),
        ion_diagnostics=tuple(ion_reports),
        interface_srh_diagnostics=tuple(iface_reports),
        output_current_A_m2=j_out,
        steady_metrics=sm,
        requested_voltages_V=voltages,
        attempts=tuple(attempts),
        disposition=disp,
        scientific_qualification=False,
    )


def run_dynamic_jv_pair_2d(
    prepared: PreparedDevice,
    protocol: DynamicJVPairProtocol2D,
    *,
    microstructure: object = None,
) -> JV2DDynamicPairResult:
    """Execute a continuous 2D preconditioning + forward + turnaround-hold + reverse scan."""
    if not isinstance(prepared, PreparedDevice) or not isinstance(protocol, DynamicJVPairProtocol2D):
        raise TypeError("PreparedDevice and DynamicJVPairProtocol2D are required")
    ms_input = _coerce_microstructure_input(prepared, microstructure)
    nx_intervals = len(protocol.x_coordinates_m) - 1
    elec_layers = [lyr for lyr in prepared.layers if lyr.role != "substrate"]
    ny_per = (len(protocol.y_coordinates_m) - 1) // max(len(elec_layers), 1)
    lat_len = float(protocol.x_coordinates_m[-1] - protocol.x_coordinates_m[0])
    gen_scale = 1.0 if protocol.illuminated else 0.0
    mobile = protocol.state_topology == "single_positive_mobile_ion"

    if not mobile:
        raise CapabilityError(
            "dynamic_pair_2d_requires_mobile_ions: forward/reverse hysteresis pair requires single_positive_mobile_ion"
        )

    exec_spatial = JV2DInput(
        kind="jv_2d",
        lateral_length=lat_len,
        Nx=nx_intervals,
        Ny_per_layer=ny_per,
        lateral_bc=protocol.lateral_bc,
        microstructure=ms_input,
        ion_dynamics="single_mobile",
        interface_srh=protocol.interface_srh,
    )
    _, ion_layer_id = device_has_ions(prepared)
    ionic_dev = compile_ionic_device_2d(
        prepared,
        exec_spatial,
        ion_domain_layer=ion_layer_id,
        ion_domain_endpoints=("blocking", "blocking"),
    )
    if not np.allclose(ionic_dev.electronic_component.geometry.x_m, protocol.x_coordinates_m, rtol=0.0, atol=1.0e-15):
        raise ValueError("protocol.x_coordinates_m disagrees with compiled 2D lateral mesh")
    if not np.allclose(ionic_dev.electronic_component.geometry.y_geometry.nodes_m, protocol.y_coordinates_m, rtol=0.0, atol=1.0e-15):
        raise ValueError("protocol.y_coordinates_m disagrees with compiled 2D vertical mesh")

    # 1. Precondition at precondition_voltage_V
    steady_res = solve_ionic_steady_state_2d(
        ionic_dev, voltage_V=protocol.precondition_voltage_V, generation_scale=0.0
    )
    if not steady_res.accepted or steady_res.state is None:
        raise RuntimeError(f"2D dynamic pair dark preconditioning failed: {steady_res.error}")
    curr_state = steady_res.state
    ion_opts = IonicBackwardEuler2DOptions(
        ion_inventory_rtol=max(protocol.ion_inventory_rtol, 1.0e-12),
    )
    if protocol.precondition_hold_s > 0.0:
        hold_grid = _dwell_time_grid(protocol.precondition_hold_s, max(protocol.substeps_per_point, 8))
        hold_run = run_ionic_backward_euler_2d(
            ionic_dev,
            curr_state,
            hold_grid,
            voltage_fn=_const_fn(protocol.precondition_voltage_V),
            generation_fn=lambda t: 0.0 if t <= 0.0 else gen_scale,
            options=ion_opts,
        )
        if not hold_run.accepted:
            raise RuntimeError(f"2D preconditioning hold failed: {hold_run.error}")
        curr_state = hold_run.states[-1]
    precond_sha = state_digest(curr_state.u)
    polarity = float(ionic_dev.electronic_component.column.voltage_polarity)
    grid_x = ionic_dev.electronic_component.geometry.x_m
    grid_y = ionic_dev.electronic_component.geometry.y_geometry.nodes_m
    g_dyn_fn = _const_fn(gen_scale)

    def _run_branch(voltages_seq: tuple[float, ...], seed_state: IonicState2D) -> tuple[JV2DResult, IonicState2D]:
        state = seed_state
        ion_x = ionic_dev.ion_geometry.x_m
        ion_y = ionic_dev.ion_geometry.y_geometry.nodes_m
        ion_areas = ionic_dev.ion_geometry.contract_geometry.volumes.reshape(ionic_dev.ion_geometry.shape)
        ion_lim_2d = ionic_dev.positive_ion_site_limit_m3.reshape(ionic_dev.ion_geometry.shape)
        v_list: list[float] = []
        j_list: list[float] = []
        snaps: list[SpatialSnapshot2D] = []
        comps: list[MobileIonCurrentComponents2D] = []
        diags: list[MobileIonDiagnostics2D] = []
        att: list[JV2DPointAttempt] = []
        for idx, v_target in enumerate(voltages_seq):
            t0 = time.monotonic()
            prev_p = state.positive_ion_density_m3
            if protocol.step_mode == "dwell":
                t_sub = _dwell_time_grid(protocol.dwell_or_step_time_s, protocol.substeps_per_point)
                v_sub_fn = _const_fn(float(v_target))
                dv_dt = 0.0
            else:
                t_sub = np.linspace(0.0, protocol.dwell_or_step_time_s, protocol.substeps_per_point + 1, dtype=np.float64)
                v_start = float(state.voltage_V)
                dv_dt = (float(v_target) - v_start) / protocol.dwell_or_step_time_s
                v_sub_fn = _ramp_fn(v_start, float(v_target), protocol.dwell_or_step_time_s)
            run_b = run_ionic_backward_euler_2d(
                ionic_dev,
                state,
                t_sub,
                voltage_fn=v_sub_fn,
                generation_fn=g_dyn_fn,
                options=ion_opts,
            )
            if not run_b.accepted:
                raise RuntimeError(run_b.error or f"2D dynamic branch failed at V={v_target:.4f} V")
            state = run_b.states[-1]
            snap = extract_snapshot_2d(ionic_dev, state, generation_scale=gen_scale)
            curr_c = compute_mobile_ion_current_components_2d(
                ionic_dev, state, generation_scale=gen_scale, applied_voltage_rate_V_s=dv_dt
            )
            term_p = state.positive_ion_density_m3
            diag = assess_mobile_ion_terminal_2d(
                ion_x, ion_y, prev_p, term_p, ion_lim_2d,
                terminal_electron_density=snap.n, terminal_hole_density=snap.p,
                inventory_rtol=max(protocol.ion_inventory_rtol, 1.0e-12),
                control_volume_areas_m2=ion_areas,
            )
            if not diag.passed:
                raise RuntimeError("2D dynamic point failed ion diagnostics: " + ", ".join(diag.violations))
            v_list.append(float(v_target))
            j_list.append(curr_c.terminal_total_A_m2)
            comps.append(curr_c)
            diags.append(diag)
            if protocol.save_snapshots:
                snaps.append(snap)
            att.append(
                JV2DPointAttempt(
                    idx, float(v_target), True, "converged", run_b.bisections,
                    time.monotonic() - t0, state_digest(state.u), None,
                )
            )
        v_arr: Vector = np.asarray(v_list, dtype=np.float64)
        j_arr: Vector = np.asarray(j_list, dtype=np.float64)
        j_out = -polarity * j_arr
        res = JV2DResult(
            V=v_arr,
            J=j_arr,
            snapshots=tuple(snaps),
            grid_x=grid_x,
            grid_y=grid_y,
            lateral_bc=protocol.lateral_bc,
            metrics=compute_jv_2d_metrics(v_arr, j_arr),
            protocol=None,
            current_components=tuple(comps),
            ion_diagnostics=tuple(diags),
            output_current_A_m2=j_out,
            requested_voltages_V=np.asarray(voltages_seq, dtype=np.float64),
            attempts=tuple(att),
            disposition="completed",
            scientific_qualification=False,
        )
        return res, state

    fwd_res, after_fwd = _run_branch(protocol.forward_voltages_V, curr_state)
    turn_state = after_fwd
    if protocol.turnaround_hold_s > 0.0:
        t_turn = _dwell_time_grid(protocol.turnaround_hold_s, max(protocol.substeps_per_point, 8))
        hold_turn = run_ionic_backward_euler_2d(
            ionic_dev,
            turn_state,
            t_turn,
            voltage_fn=_const_fn(float(protocol.forward_voltages_V[-1])),
            generation_fn=g_dyn_fn,
            options=ion_opts,
        )
        if not hold_turn.accepted:
            raise RuntimeError(f"2D turnaround hold failed: {hold_turn.error}")
        turn_state = hold_turn.states[-1]
    turn_sha = state_digest(turn_state.u)
    rev_res, _ = _run_branch(protocol.reverse_voltages_V, turn_state)
    voc_delta = (
        float(rev_res.metrics.V_oc - fwd_res.metrics.V_oc)
        if fwd_res.metrics.voc_bracketed and rev_res.metrics.voc_bracketed
        else None
    )
    return JV2DDynamicPairResult(
        protocol=protocol,
        forward=fwd_res,
        reverse=rev_res,
        preconditioned_state_sha256=precond_sha,
        turnaround_state_sha256=turn_sha,
        hysteresis_voc_delta_V=voc_delta,
        disposition="completed",
        scientific_qualification=False,
    )


def write_jv_2d_result(path: str | Path, result: JV2DResult) -> str:
    """Write a lossless JSON artifact for ``JV2DResult`` and return its SHA-256."""
    target = Path(path)
    payload: dict[str, Any] = {
        "schema": JV2D_RESULT_SCHEMA,
        "scientific_qualification": False,
        "disposition": result.disposition,
        "lateral_bc": result.lateral_bc,
        "V": result.V,
        "J": result.J,
        "output_current_A_m2": result.output_current_A_m2,
        "requested_voltages_V": result.requested_voltages_V,
        "grid_x": result.grid_x,
        "grid_y": result.grid_y,
        "metrics": asdict(result.metrics),
        "protocol": None if result.protocol is None else result.protocol.to_dict(),
        "attempts": [asdict(a) for a in result.attempts],
        "ion_diagnostics": [asdict(d) for d in result.ion_diagnostics],
    }
    write_atomic_json(target, encode_json_payload(payload))
    return hashlib.sha256(target.read_bytes()).hexdigest()


def read_jv_2d_result(path: str | Path, *, expected_sha256: str | None = None) -> JV2DResult:
    """Read and verify a lossless JSON ``JV2DResult`` artifact."""
    raw = Path(path).read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError("JV2DResult SHA-256 mismatch")
    doc = decode_json_payload(json.loads(raw))
    if doc.get("schema") != JV2D_RESULT_SCHEMA or doc.get("scientific_qualification") is not False:
        raise ValueError("invalid JV2DResult schema or qualification flag")
    proto = None if doc["protocol"] is None else JV2DProtocol.from_dict(doc["protocol"])
    return JV2DResult(
        V=np.asarray(doc["V"], dtype=np.float64),
        J=np.asarray(doc["J"], dtype=np.float64),
        snapshots=(),
        grid_x=np.asarray(doc["grid_x"], dtype=np.float64),
        grid_y=np.asarray(doc["grid_y"], dtype=np.float64),
        lateral_bc=str(doc["lateral_bc"]),
        metrics=JV2DMetrics(**doc["metrics"]),
        protocol=proto,
        ion_diagnostics=tuple(MobileIonDiagnostics2D(**d) for d in doc["ion_diagnostics"]),
        output_current_A_m2=np.asarray(doc["output_current_A_m2"], dtype=np.float64),
        requested_voltages_V=np.asarray(doc["requested_voltages_V"], dtype=np.float64),
        attempts=tuple(JV2DPointAttempt(**a) for a in doc["attempts"]),
        disposition=cast(Disposition, doc["disposition"]),
        scientific_qualification=False,
    )
