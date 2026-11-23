"""2D V_oc(L_g) grain-boundary sweep via 2D J-V dwell or direct 2D steady V_oc root-finding.

For each lateral grain size ``L_g``, constructs a Neumann-x unit cell with one
centered vertical absorber grain-boundary band (``x_position = L_g / 2``,
``width = gb_width``). Grain-boundary recombination ALWAYS mixes SRH rates by
exact geometric cell-overlap fraction ``f`` (``R = R_bulk + f * (R_gb - R_bulk)``),
never averaging lifetimes first.

Supports both:
- ``mode="jv_sweep"`` (default): full 2D J-V sweep extracting ``(V_oc, J_sc, FF)``.
- ``mode="direct_voc"`` / :func:`solve_direct_voc_2d`: direct 2D steady
  continuation + safeguarded secant/Dekker root-finding on ``J_out(V) = 0``
  across grain width ``L_g`` and ``tau_gb`` or GB recombination velocity ``S_gb``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from solarlab.config.resolve_device import resolve_device
from solarlab.contracts import frozen_array
from solarlab.device.compile_2d import compile_device_2d
from solarlab.device.inputs import GrainBoundaryInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.dc_2d import CompiledDevice2D, State2D
from solarlab.experiments.two_dimensional.inputs import (
    GrainSweepInput,
    JV2DInput,
    MicrostructureInput,
)
from solarlab.experiments.two_dimensional.jv_sweep_2d import (
    Disposition,
    ProgressCallback,
    build_jv_2d_execution_protocol,
    decode_json_payload,
    device_has_ions,
    encode_json_payload,
    precondition_frozen_lane,
    project_prepared_for_electronic_2d,
    run_jv_sweep_2d,
    write_atomic_json,
)
from solarlab.experiments.two_dimensional.preparation import PreparedSpatialExperiment
from solarlab.solvers.nonlinear.steady_1d import solve_steady_state
from solarlab.solvers.nonlinear.steady_2d import Steady2DOptions, solve_steady_state_2d

__all__ = [
    "DirectVoc2DResult",
    "VOC_GRAIN_SWEEP_SCHEMA",
    "VocGrainPointResult",
    "VocGrainSweepResult",
    "read_voc_grain_sweep",
    "resolve_gb_lifetimes_s",
    "run_voc_grain_sweep",
    "solve_direct_voc_2d",
    "write_voc_grain_sweep",
]

Vector: TypeAlias = NDArray[np.float64]
VOC_GRAIN_SWEEP_SCHEMA = "solarlab.voc_grain_sweep_2d.result.v1"


@dataclass(frozen=True)
class DirectVoc2DResult:
    """Direct 2D steady V_oc bracket + safeguarded secant-Dekker root outcome."""

    accepted: bool
    V_oc_V: float
    J_sc_Am2: float
    residual_current_A_m2: float
    bracket_low_V: float
    bracket_high_V: float
    steady_solves: int
    wall_s: float
    termination: str
    error: str | None = None
    scientific_qualification: bool = False


@dataclass(frozen=True)
class VocGrainPointResult:
    """Per-grain-size execution record with explicit failure/acceptance provenance."""

    index: int
    grain_size_m: float
    accepted: bool
    V_oc_V: float
    J_sc_Am2: float
    FF: float
    termination: str
    wall_s: float
    error: str | None = None


@dataclass(frozen=True)
class VocGrainSweepResult:
    """Aligned ``(grain_sizes_m, V_oc_V, J_sc_Am2, FF)`` result of ``run_voc_grain_sweep``."""

    grain_sizes_m: Vector
    V_oc_V: Vector
    J_sc_Am2: Vector
    FF: Vector
    points: tuple[VocGrainPointResult, ...] = ()
    mode: Literal["jv_sweep", "direct_voc"] = "jv_sweep"
    disposition: Disposition = "completed"
    scientific_qualification: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "grain_sizes_m", frozen_array(self.grain_sizes_m))
        for name in ("V_oc_V", "J_sc_Am2", "FF"):
            arr: Vector = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if arr.ndim != 1 or np.isinf(arr).any():
                raise ValueError(f"{name} must be a 1D array without infinities")
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)


def resolve_gb_lifetimes_s(
    *,
    gb_width_m: float,
    tau_gb: tuple[float, float] | None = (1.0e-9, 1.0e-9),
    s_gb_m_s: tuple[float, float] | float | None = None,
) -> tuple[float, float]:
    """Resolve effective GB band SRH lifetimes from ``tau_gb`` or velocity ``s_gb_m_s``."""
    if not math.isfinite(gb_width_m) or gb_width_m <= 0.0:
        raise ValueError("gb_width_m must be finite and positive")
    if s_gb_m_s is not None:
        if tau_gb is not None:
            raise ValueError("supply either tau_gb or s_gb_m_s, not both")
        if isinstance(s_gb_m_s, (int, float)):
            sn = sp = float(s_gb_m_s)
        else:
            if len(s_gb_m_s) != 2:
                raise ValueError("s_gb_m_s must be a positive scalar or (S_n, S_p) pair")
            sn, sp = float(s_gb_m_s[0]), float(s_gb_m_s[1])
        if not (math.isfinite(sn) and math.isfinite(sp) and sn > 0.0 and sp > 0.0):
            raise ValueError("GB recombination velocities must be finite and positive")
        return (gb_width_m / sn, gb_width_m / sp)
    if tau_gb is None or len(tau_gb) != 2:
        raise ValueError("tau_gb must be a positive (tau_n, tau_p) pair")
    tn, tp = float(tau_gb[0]), float(tau_gb[1])
    if not (math.isfinite(tn) and math.isfinite(tp) and tn > 0.0 and tp > 0.0):
        raise ValueError("tau_gb lifetimes must be finite and positive")
    return (tn, tp)


def _centered_absorber_microstructure(
    prepared: PreparedDevice,
    *,
    lateral_length_m: float,
    gb_width_m: float,
    tau_n_s: float,
    tau_p_s: float,
) -> MicrostructureInput:
    if gb_width_m > lateral_length_m:
        raise ValueError(
            f"gb_width_m ({gb_width_m:.3e} m) exceeds grain size lateral_length_m ({lateral_length_m:.3e} m)"
        )
    target_ids = [
        layer.id for layer in prepared.layers if layer.role != "substrate" and layer.role == "absorber"
    ]
    if not target_ids:
        raise CapabilityError("voc_grain_sweep_missing_absorber: device has no electrical 'absorber' layer")
    gb = GrainBoundaryInput.model_validate(
        {
            "id": "centered_gb",
            "layer_ids": target_ids,
            "x_position": 0.5 * lateral_length_m,
            "width": gb_width_m,
            "tau_n": tau_n_s,
            "tau_p": tau_p_s,
        }
    )
    return MicrostructureInput(grain_boundaries=(gb,))


def _rebase_state_2d(device: CompiledDevice2D, state: State2D, new_voltage_V: float) -> State2D:
    if state.voltage_V == new_voltage_V:
        return state
    u = state.u.copy()
    n_nodes = device.node_count
    old_e = device.column.carrier_reference_energies(state.voltage_V)
    new_e = device.column.carrier_reference_energies(new_voltage_V)
    for offset, old_val, new_val in zip((n_nodes, 2 * n_nodes), old_e, new_e):
        u[offset:offset + n_nodes] += (old_val - new_val) / device.thermal_voltage_V
    bottom, top = device.column.contact_potentials(new_voltage_V)
    u[:device.geometry.nx] = bottom
    u[n_nodes - device.geometry.nx:n_nodes] = top
    return device.state(u, voltage_V=new_voltage_V)


def solve_direct_voc_2d(
    prepared: PreparedDevice,
    *,
    lateral_length_m: float,
    gb_width_m: float = 10.0e-9,
    tau_gb: tuple[float, float] | None = (1.0e-9, 1.0e-9),
    s_gb_m_s: tuple[float, float] | float | None = None,
    Nx: int = 10,
    Ny_per_layer: int = 10,
    v_max_bracket_V: float = 1.5,
    continuation_step_V: float = 0.1,
    v_tol_V: float = 1.0e-6,
    j_tol_A_m2: float = 1.0e-4,
    initial_state_settle_s: float = 1.0e-3,
    options: Steady2DOptions | None = None,
) -> DirectVoc2DResult:
    """Solve direct 2D illuminated steady ``V_oc`` via continuation bracket + secant-Dekker."""
    started = time.monotonic()
    if not math.isfinite(lateral_length_m) or lateral_length_m <= 0.0:
        raise ValueError("lateral_length_m must be finite and positive")
    tau_n, tau_p = resolve_gb_lifetimes_s(
        gb_width_m=gb_width_m, tau_gb=tau_gb, s_gb_m_s=s_gb_m_s
    )
    ms_input = _centered_absorber_microstructure(
        prepared,
        lateral_length_m=lateral_length_m,
        gb_width_m=gb_width_m,
        tau_n_s=tau_n,
        tau_p_s=tau_p,
    )
    geom_spatial = JV2DInput(
        kind="jv_2d",
        lateral_length=lateral_length_m,
        Nx=Nx,
        Ny_per_layer=Ny_per_layer,
        lateral_bc="neumann",
        microstructure=ms_input,
    )
    has_ions, _ = device_has_ions(prepared)
    if has_ions:
        proto = build_jv_2d_execution_protocol(
            prepared,
            ms_input,
            lateral_length=lateral_length_m,
            Nx=Nx,
            V_max=max(v_max_bracket_V, 0.2),
            V_step=max(continuation_step_V, 0.05),
            illuminated=True,
            lateral_bc="neumann",
            Ny_per_layer=Ny_per_layer,
            initial_state_settle_s=initial_state_settle_s,
            implicit_legacy_protocol=True,
        )
        device_2d, seed_2d, rr_specs, iface_couplings = precondition_frozen_lane(
            prepared, geom_spatial, proto
        )
        if rr_specs or iface_couplings:
            raise CapabilityError(
                "direct_voc_2d_unsupported_extensions: direct steady V_oc requires standard local 2D recombination"
            )
    else:
        clean = project_prepared_for_electronic_2d(prepared, strip_ions=False, strip_interfaces=False)
        device_2d = compile_device_2d(clean, geom_spatial)
        col_illum = solve_steady_state(device_2d.column, voltage_V=0.0, generation_scale=1.0)
        if col_illum.state is None:
            return DirectVoc2DResult(
                False, 0.0, 0.0, math.nan, 0.0, 0.0, 0, time.monotonic() - started,
                "column_seed_failed", "1D illuminated seed failed",
            )
        seed_2d = device_2d.state(device_2d.extrude_coordinates(col_illum.state.u), voltage_V=0.0)

    steady_opts = Steady2DOptions() if options is None else options
    polarity = float(device_2d.column.voltage_polarity)
    solves = 0

    def eval_point(v_target: float, ref_state: State2D) -> tuple[bool, float, State2D | None, str | None]:
        nonlocal solves
        solves += 1
        guess = _rebase_state_2d(device_2d, ref_state, v_target)
        res = solve_steady_state_2d(
            device_2d, voltage_V=v_target, generation_scale=1.0, options=steady_opts, initial_state=guess
        )
        if not res.accepted or res.state is None or res.diagnostics is None:
            return False, math.nan, None, res.error or res.termination
        term_j = float(res.diagnostics["terminal_mean_current_density_A_m2"][1])
        j_out = -polarity * term_j
        return True, j_out, res.state, None

    ok0, jsc_out, s0, err0 = eval_point(0.0, seed_2d)
    if not ok0 or s0 is None or jsc_out <= 0.0:
        return DirectVoc2DResult(
            False, 0.0, max(0.0, jsc_out if math.isfinite(jsc_out) else 0.0),
            math.nan, 0.0, 0.0, solves, time.monotonic() - started,
            "short_circuit_failed", err0 or "non-positive short-circuit output current",
        )

    # Walk forward in continuation steps until output current crosses <= 0
    v_lo, j_lo, s_lo = 0.0, jsc_out, s0
    v_hi: float | None = None
    j_hi: float = math.nan
    s_hi: State2D | None = None
    step = max(float(continuation_step_V), 0.02)
    v_cur = 0.0
    while v_cur < v_max_bracket_V - 1.0e-12:
        v_next = min(v_cur + step, v_max_bracket_V)
        ok, j_val, s_val, err = eval_point(v_next, s_lo)
        if not ok or s_val is None:
            if step > 0.0125:
                step *= 0.5
                continue
            return DirectVoc2DResult(
                False, 0.0, jsc_out, math.nan, v_lo, v_next, solves,
                time.monotonic() - started, "bracket_continuation_failed", err,
            )
        v_cur = v_next
        if j_val <= 0.0:
            v_hi, j_hi, s_hi = v_next, j_val, s_val
            break
        v_lo, j_lo, s_lo = v_next, j_val, s_val

    if v_hi is None or s_hi is None:
        return DirectVoc2DResult(
            False, 0.0, jsc_out, j_lo, v_lo, v_max_bracket_V, solves,
            time.monotonic() - started, "voc_not_bracketed",
            f"V_oc exceeds v_max_bracket_V={v_max_bracket_V:.3f} V",
        )

    bracket_lo_init, bracket_hi_init = v_lo, v_hi
    if abs(j_hi) <= j_tol_A_m2:
        return DirectVoc2DResult(
            True, v_hi, jsc_out, j_hi, bracket_lo_init, bracket_hi_init, solves,
            time.monotonic() - started, "converged",
        )

    # Safeguarded secant-Dekker root polish inside [v_lo, v_hi]
    for _ in range(24):
        if abs(v_hi - v_lo) <= v_tol_V:
            v_best = v_lo + j_lo * (v_hi - v_lo) / (j_lo - j_hi)
            j_best = j_lo if abs(j_lo) <= abs(j_hi) else j_hi
            return DirectVoc2DResult(
                True, float(v_best), jsc_out, float(j_best), bracket_lo_init, bracket_hi_init,
                solves, time.monotonic() - started, "converged",
            )
        denom = j_lo - j_hi
        v_sec = v_lo + j_lo * (v_hi - v_lo) / denom if abs(denom) > 1.0e-30 else 0.5 * (v_lo + v_hi)
        margin = 0.05 * (v_hi - v_lo)
        v_cand = v_sec if (v_lo + margin) < v_sec < (v_hi - margin) else 0.5 * (v_lo + v_hi)
        ref_s = s_lo if abs(v_cand - v_lo) <= abs(v_hi - v_cand) else s_hi
        ok_c, j_c, s_c, err_c = eval_point(v_cand, ref_s)
        if not ok_c or s_c is None:
            # Fall back to midpoint if secant candidate failed
            v_cand = 0.5 * (v_lo + v_hi)
            ok_c, j_c, s_c, err_c = eval_point(v_cand, s_lo)
            if not ok_c or s_c is None:
                return DirectVoc2DResult(
                    False, 0.0, jsc_out, math.nan, bracket_lo_init, bracket_hi_init, solves,
                    time.monotonic() - started, "root_polish_failed", err_c,
                )
        if abs(j_c) <= j_tol_A_m2 and abs(v_hi - v_lo) <= max(10.0 * v_tol_V, 1.0e-5):
            return DirectVoc2DResult(
                True, float(v_cand), jsc_out, float(j_c), bracket_lo_init, bracket_hi_init,
                solves, time.monotonic() - started, "converged",
            )
        if j_c > 0.0:
            v_lo, j_lo, s_lo = v_cand, j_c, s_c
        else:
            v_hi, j_hi, s_hi = v_cand, j_c, s_c

    v_est = float(v_lo + j_lo * (v_hi - v_lo) / (j_lo - j_hi))
    return DirectVoc2DResult(
        True, v_est, jsc_out, float(min(abs(j_lo), abs(j_hi))), bracket_lo_init, bracket_hi_init,
        solves, time.monotonic() - started, "converged",
    )


def run_voc_grain_sweep(
    prepared_or_experiment: PreparedDevice | PreparedSpatialExperiment,
    grain_sizes: Sequence[float] | None = None,
    *,
    tau_gb: tuple[float, float] | None = (1.0e-9, 1.0e-9),
    s_gb_m_s: tuple[float, float] | float | None = None,
    gb_width: float = 10.0e-9,
    Nx: int = 10,
    Ny_per_layer: int = 10,
    V_max: float = 1.2,
    V_step: float = 0.05,
    illuminated: bool = True,
    settle_t: float = 1.0e-3,
    mode: Literal["jv_sweep", "direct_voc"] = "jv_sweep",
    progress: ProgressCallback | None = None,
    on_point_failure: Literal["raise", "record_partial"] = "raise",
) -> VocGrainSweepResult:
    """Sweep lateral grain size ``L_g`` with one centered absorber GB band."""
    if mode not in {"jv_sweep", "direct_voc"}:
        raise ValueError("mode must be 'jv_sweep' or 'direct_voc'")
    if on_point_failure not in {"raise", "record_partial"}:
        raise ValueError("on_point_failure must be 'raise' or 'record_partial'")

    if isinstance(prepared_or_experiment, PreparedSpatialExperiment):
        inp = prepared_or_experiment.to_input()
        if not isinstance(inp.experiment, GrainSweepInput):
            raise ValueError("PreparedSpatialExperiment does not contain a voc_grain_sweep experiment")
        prepared = resolve_device(
            inp.device,
            prepared_or_experiment.defaults,
            prepared_or_experiment.resources,
            sources=prepared_or_experiment.sources,
        )
        mapping = prepared_or_experiment.to_mapping()["experiment"]["controls"]
        sizes_seq: Sequence[float] = grain_sizes if grain_sizes is not None else mapping["grain_sizes_m"]
        gb_width = float(mapping["gb_width"])
        tau_gb = (float(mapping["tau_gb_n"]), float(mapping["tau_gb_p"])) if s_gb_m_s is None else None
        Nx = int(mapping["Nx"])
        Ny_per_layer = int(mapping["Ny_per_layer"])
        V_max = float(mapping["V_max"])
        V_step = float(mapping["V_step"])
        illuminated = bool(mapping["illuminated"])
        settle_t = float(mapping["settle_t"])
    elif isinstance(prepared_or_experiment, PreparedDevice):
        prepared = prepared_or_experiment
        if grain_sizes is None:
            raise ValueError("grain_sizes must be supplied when running from PreparedDevice")
        sizes_seq = grain_sizes
    else:
        raise TypeError("run_voc_grain_sweep requires PreparedDevice or PreparedSpatialExperiment")

    raw_sizes = [float(g) for g in sizes_seq]
    if not raw_sizes or any(not math.isfinite(g) or g <= 0.0 for g in raw_sizes):
        raise ValueError("grain_sizes must be a non-empty sequence of finite positive lengths [m]")
    tau_n, tau_p = resolve_gb_lifetimes_s(gb_width_m=gb_width, tau_gb=tau_gb, s_gb_m_s=s_gb_m_s)
    for g_len in raw_sizes:
        if gb_width > g_len:
            raise ValueError(f"gb_width ({gb_width:.3e} m) exceeds grain size ({g_len:.3e} m)")

    l_accepted: list[float] = []
    voc_list: list[float] = []
    jsc_list: list[float] = []
    ff_list: list[float] = []
    points: list[VocGrainPointResult] = []
    total = len(raw_sizes)

    for idx, l_g in enumerate(raw_sizes):
        t0 = time.monotonic()
        try:
            if mode == "direct_voc":
                if not illuminated:
                    raise ValueError("direct_voc mode requires illuminated=True")
                d_res = solve_direct_voc_2d(
                    prepared,
                    lateral_length_m=l_g,
                    gb_width_m=gb_width,
                    tau_gb=(tau_n, tau_p),
                    Nx=Nx,
                    Ny_per_layer=Ny_per_layer,
                    v_max_bracket_V=max(V_max, 1.3),
                    continuation_step_V=max(V_step, 0.05),
                )
                if not d_res.accepted:
                    raise RuntimeError(d_res.error or d_res.termination)
                voc_val, jsc_val, ff_val = d_res.V_oc_V, d_res.J_sc_Am2, math.nan
            else:
                ms_in = _centered_absorber_microstructure(
                    prepared,
                    lateral_length_m=l_g,
                    gb_width_m=gb_width,
                    tau_n_s=tau_n,
                    tau_p_s=tau_p,
                )
                jv_res = run_jv_sweep_2d(
                    prepared,
                    microstructure=ms_in,
                    lateral_length=l_g,
                    Nx=Nx,
                    Ny_per_layer=Ny_per_layer,
                    V_max=V_max,
                    V_step=V_step,
                    illuminated=illuminated,
                    lateral_bc="neumann",
                    settle_t=settle_t,
                    save_snapshots=False,
                )
                voc_val = jv_res.metrics.V_oc
                jsc_val = jv_res.metrics.J_sc
                ff_val = jv_res.metrics.FF
        except CapabilityError:
            raise
        except Exception as exc:
            points.append(
                VocGrainPointResult(
                    idx, l_g, False, math.nan, math.nan, math.nan,
                    "failed", time.monotonic() - t0, str(exc),
                )
            )
            if on_point_failure == "raise":
                raise
            continue

        l_accepted.append(l_g)
        voc_list.append(voc_val)
        jsc_list.append(jsc_val)
        ff_list.append(ff_val)
        points.append(
            VocGrainPointResult(
                idx, l_g, True, voc_val, jsc_val, ff_val,
                "converged", time.monotonic() - t0, None,
            )
        )
        if progress is not None:
            progress(
                "voc_grain_sweep",
                idx + 1,
                total,
                f"L_g={l_g * 1.0e9:.0f} nm  V_oc={voc_val * 1.0e3:.1f} mV",
            )

    disp: Disposition = (
        "completed"
        if len(l_accepted) == total
        else ("partial" if l_accepted else "failed")
    )
    return VocGrainSweepResult(
        grain_sizes_m=np.asarray(l_accepted, dtype=np.float64),
        V_oc_V=np.asarray(voc_list, dtype=np.float64),
        J_sc_Am2=np.asarray(jsc_list, dtype=np.float64),
        FF=np.asarray(ff_list, dtype=np.float64),
        points=tuple(points),
        mode=mode,
        disposition=disp,
        scientific_qualification=False,
    )


def write_voc_grain_sweep(path: str | Path, result: VocGrainSweepResult) -> str:
    """Write a lossless JSON artifact for ``VocGrainSweepResult`` and return SHA-256."""
    target = Path(path)
    payload: dict[str, Any] = {
        "schema": VOC_GRAIN_SWEEP_SCHEMA,
        "scientific_qualification": False,
        "mode": result.mode,
        "disposition": result.disposition,
        "grain_sizes_m": result.grain_sizes_m,
        "V_oc_V": result.V_oc_V,
        "J_sc_Am2": result.J_sc_Am2,
        "FF": result.FF,
        "points": [asdict(p) for p in result.points],
    }
    write_atomic_json(target, encode_json_payload(payload))
    return hashlib.sha256(target.read_bytes()).hexdigest()


def read_voc_grain_sweep(path: str | Path, *, expected_sha256: str | None = None) -> VocGrainSweepResult:
    """Read and verify a lossless JSON ``VocGrainSweepResult`` artifact."""
    raw = Path(path).read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError("VocGrainSweepResult SHA-256 mismatch")
    doc = decode_json_payload(json.loads(raw))
    if doc.get("schema") != VOC_GRAIN_SWEEP_SCHEMA or doc.get("scientific_qualification") is not False:
        raise ValueError("invalid VocGrainSweepResult schema or qualification flag")
    return VocGrainSweepResult(
        grain_sizes_m=np.asarray(doc["grain_sizes_m"], dtype=np.float64),
        V_oc_V=np.asarray(doc["V_oc_V"], dtype=np.float64),
        J_sc_Am2=np.asarray(doc["J_sc_Am2"], dtype=np.float64),
        FF=np.asarray(doc["FF"], dtype=np.float64),
        points=tuple(VocGrainPointResult(**p) for p in doc["points"]),
        mode=cast(Literal["jv_sweep", "direct_voc"], doc["mode"]),
        disposition=cast(Disposition, doc["disposition"]),
        scientific_qualification=False,
    )
