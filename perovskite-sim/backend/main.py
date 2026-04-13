import asyncio
import json
import os
from typing import Any, Optional

import numpy as np
import traceback
import yaml
from dataclasses import asdict, is_dataclass
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from perovskite_sim.experiments import degradation, impedance, jv_sweep
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.mode import resolve_mode
from perovskite_sim.models.parameters import MaterialParams
from backend.jobs import JobRegistry, JobStatus, _DRAIN_TIMEOUT
from backend.progress import ProgressReporter

def _describe_active_physics(stack) -> str:
    """Return a short human-readable description of the active physics tier.

    Used by the SSE result payload so the frontend solver console can
    show which Phase 1-4 upgrades ran without re-deriving the flags.
    """
    mode_name = str(getattr(stack, "mode", "full")).lower()
    mode = resolve_mode(mode_name)
    if mode_name == "legacy":
        return "LEGACY  Beer-Lambert · single ion · uniform τ · T=300K"
    if mode_name == "fast":
        return "FAST  Beer-Lambert · single ion · uniform τ · T-scaling"
    # full (or any unrecognised-but-valid mode falls through to full defaults)
    parts = []
    if mode.use_thermionic_emission:
        parts.append("band offsets · TE")
    if mode.use_tmm_optics:
        parts.append("TMM")
    else:
        parts.append("Beer-Lambert")
    if mode.use_dual_ions:
        parts.append("dual ions")
    if mode.use_trap_profile:
        parts.append("trap profile")
    if mode.use_temperature_scaling:
        parts.append("T-scaling")
    return "FULL  " + " · ".join(parts)


_JOB_REGISTRY = JobRegistry()


CONFIGS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "configs")
)


def _coerce_numbers(obj: Any) -> Any:
    """Recursively convert strings that look like numbers into floats.

    PyYAML's 1.1 resolver leaves scientific-notation literals without a decimal
    point (e.g. ``1e-9``) as strings; the frontend numeric editor then fails.
    """
    if isinstance(obj, dict):
        return {k: _coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numbers(v) for v in obj]
    if isinstance(obj, str):
        try:
            return float(obj)
        except (ValueError, TypeError):
            return obj
    return obj


def resolve_config_path(config_path: str) -> str:
    """Resolve config_path to an absolute path inside perovskite-sim/configs if needed."""
    if os.path.isabs(config_path):
        return config_path
    backend_dir = os.path.dirname(__file__)
    candidate1 = os.path.abspath(os.path.join(backend_dir, config_path))
    if os.path.exists(candidate1):
        return candidate1
    candidate2 = os.path.join(CONFIGS_DIR, os.path.basename(config_path))
    if os.path.exists(candidate2):
        return candidate2
    return config_path


def stack_from_dict(cfg: dict) -> DeviceStack:
    """Build a DeviceStack from a dict with the same schema as the YAML files."""
    dev = cfg.get("device", {}) or {}
    layers: list[LayerSpec] = []
    for layer_cfg in cfg.get("layers", []) or []:
        p = MaterialParams(
            eps_r=float(layer_cfg["eps_r"]),
            mu_n=float(layer_cfg["mu_n"]),
            mu_p=float(layer_cfg["mu_p"]),
            D_ion=float(layer_cfg["D_ion"]),
            P_lim=float(layer_cfg["P_lim"]),
            P0=float(layer_cfg["P0"]),
            ni=float(layer_cfg["ni"]),
            tau_n=float(layer_cfg["tau_n"]),
            tau_p=float(layer_cfg["tau_p"]),
            n1=float(layer_cfg["n1"]),
            p1=float(layer_cfg["p1"]),
            B_rad=float(layer_cfg["B_rad"]),
            C_n=float(layer_cfg["C_n"]),
            C_p=float(layer_cfg["C_p"]),
            alpha=float(layer_cfg["alpha"]),
            N_A=float(layer_cfg["N_A"]),
            N_D=float(layer_cfg["N_D"]),
            chi=float(layer_cfg.get("chi", 0.0)),
            Eg=float(layer_cfg.get("Eg", 0.0)),
        )
        layers.append(
            LayerSpec(
                name=str(layer_cfg["name"]),
                thickness=float(layer_cfg["thickness"]),
                params=p,
                role=str(layer_cfg["role"]),
            )
        )
    interfaces = tuple(
        (float(pair[0]), float(pair[1]))
        for pair in (dev.get("interfaces") or [])
    )
    mode_name = str(dev.get("mode", "full"))
    # Validate early so an unknown mode fails the HTTP request rather than
    # blowing up inside the worker thread where the error is harder to surface.
    resolve_mode(mode_name)
    return DeviceStack(
        layers=tuple(layers),
        V_bi=float(dev.get("V_bi", 1.1)),
        Phi=float(dev.get("Phi", 2.5e21)),
        interfaces=interfaces,
        T=float(dev.get("T", 300.0)),
        mode=mode_name,
    )


def build_stack(config_path: Optional[str], device: Optional[dict]) -> DeviceStack:
    """Return a DeviceStack from either an inline device dict (preferred) or a YAML path."""
    if device is not None:
        return stack_from_dict(device)
    if not config_path:
        raise HTTPException(status_code=400, detail="Either 'device' or 'config_path' must be provided")
    return load_device_from_yaml(resolve_config_path(config_path))


def to_serializable(obj):
    """Recursively convert dataclasses and numpy arrays to JSON-serializable types."""
    if is_dataclass(obj):
        return {k: to_serializable(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return [{"real": float(x.real), "imag": float(x.imag)} for x in obj.flat]
        return obj.tolist()
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    else:
        return obj


app = FastAPI(title="Perovskite Solar Cell Simulator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/configs")
def list_configs():
    """List available YAML config files bundled with the project."""
    try:
        names = sorted(
            f for f in os.listdir(CONFIGS_DIR)
            if f.endswith((".yaml", ".yml"))
        )
        return {"status": "ok", "configs": names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/configs/{name}")
def get_config(name: str):
    """Return the parsed YAML device config so the frontend can edit it."""
    safe_name = os.path.basename(name)
    path = os.path.join(CONFIGS_DIR, safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Config '{safe_name}' not found")
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return {"status": "ok", "name": safe_name, "config": _coerce_numbers(cfg)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class JVRequest(BaseModel):
    config_path: Optional[str] = None
    device: Optional[dict] = None
    N_grid: int = 80
    n_points: int = 40
    v_rate: float = 1.0
    V_max: Optional[float] = None


@app.post("/api/jv")
def run_jv(req: JVRequest):
    try:
        stack = build_stack(req.config_path, req.device)
        result = jv_sweep.run_jv_sweep(
            stack, N_grid=req.N_grid, n_points=req.n_points, v_rate=req.v_rate,
            V_max=req.V_max,
        )
        return {"status": "ok", "result": to_serializable(result)}
    except HTTPException:
        raise
    except Exception as e:
        print("[JV API Exception]", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class ISRequest(BaseModel):
    config_path: Optional[str] = None
    device: Optional[dict] = None
    N_grid: int = 40
    V_dc: float = 0.9
    n_freq: int = 15
    f_min: float = 10.0
    f_max: float = 1e5


@app.post("/api/impedance")
def run_impedance_api(req: ISRequest):
    try:
        stack = build_stack(req.config_path, req.device)
        frequencies = np.logspace(np.log10(req.f_min), np.log10(req.f_max), req.n_freq)
        result = impedance.run_impedance(
            stack, frequencies, V_dc=req.V_dc, N_grid=req.N_grid,
        )
        out = to_serializable(result)
        if "Z" in out:
            Z = np.array(result.Z)
            out["Z_real"] = Z.real.tolist()
            out["Z_imag"] = Z.imag.tolist()
            del out["Z"]
        return {"status": "ok", "result": out}
    except HTTPException:
        raise
    except Exception as e:
        print("[Impedance API Exception]", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class DegRequest(BaseModel):
    config_path: Optional[str] = None
    device: Optional[dict] = None
    t_end: float = 100.0
    n_snapshots: int = 10
    N_grid: int = 40
    V_bias: float = 0.9
    metric_V_max: Optional[float] = None
    metric_settle_time: float = 1e-3


class JobRequest(BaseModel):
    kind: str  # "jv" | "impedance" | "degradation"
    config_path: Optional[str] = None
    device: Optional[dict] = None
    params: dict = {}


@app.post("/api/jobs")
def start_job(req: JobRequest):
    """Start an experiment on a worker thread and return a job ID.

    The caller then opens GET /api/jobs/{id}/events to receive
    Server-Sent-Events with incremental progress and the final result.
    """
    try:
        stack = build_stack(req.config_path, req.device)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"stack build failed: {e}")

    kind = req.kind
    p = req.params

    if kind == "jv":
        def _run(reporter: ProgressReporter) -> dict:
            result = jv_sweep.run_jv_sweep(
                stack,
                N_grid=int(p.get("N_grid", 60)),
                n_points=int(p.get("n_points", 30)),
                v_rate=float(p.get("v_rate", 1.0)),
                V_max=float(p["V_max"]) if p.get("V_max") is not None else None,
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = to_serializable(result)
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "impedance":
        def _run(reporter: ProgressReporter) -> dict:
            freqs = np.logspace(
                np.log10(float(p.get("f_min", 10.0))),
                np.log10(float(p.get("f_max", 1e5))),
                int(p.get("n_freq", 15)),
            )
            result = impedance.run_impedance(
                stack, frequencies=freqs,
                V_dc=float(p.get("V_dc", 0.9)),
                N_grid=int(p.get("N_grid", 40)),
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = to_serializable(result)
            if "Z" in out:
                Z = np.array(result.Z)
                out["Z_real"] = Z.real.tolist()
                out["Z_imag"] = Z.imag.tolist()
                del out["Z"]
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "degradation":
        def _run(reporter: ProgressReporter) -> dict:
            result = degradation.run_degradation(
                stack,
                t_end=float(p.get("t_end", 100.0)),
                n_snapshots=int(p.get("n_snapshots", 10)),
                V_bias=float(p.get("V_bias", 0.9)),
                N_grid=int(p.get("N_grid", 40)),
                metric_V_max=float(p["metric_V_max"]) if p.get("metric_V_max") is not None else None,
                metric_settle_time=float(p.get("metric_settle_time", 1e-3)),
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = to_serializable(result)
            if "t" in out:
                out["times"] = out.pop("t")
            out["active_physics"] = _describe_active_physics(stack)
            return out
    else:
        raise HTTPException(status_code=400, detail=f"unknown kind: {kind}")

    job_id = _JOB_REGISTRY.submit(_run)
    return {"status": "ok", "job_id": job_id}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    """Stream progress events, the final result, and a done marker."""
    try:
        _JOB_REGISTRY.status(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown job_id: {job_id}")

    async def _gen():
        loop = asyncio.get_event_loop()
        while True:
            ev = await loop.run_in_executor(
                None, lambda: _JOB_REGISTRY.next_event(job_id, timeout=0.5)
            )
            if ev is _DRAIN_TIMEOUT:
                yield ": keepalive\n\n"
                status, _, _ = _JOB_REGISTRY.status(job_id)
                if status == JobStatus.RUNNING:
                    continue
                # Fallthrough: worker finished between drain and status
                # check — loop once more to pick up the done sentinel.
                continue
            if ev is None:
                status, result, error = _JOB_REGISTRY.status(job_id)
                if status == JobStatus.DONE:
                    yield f"event: result\ndata: {json.dumps(result)}\n\n"
                elif status == JobStatus.ERROR:
                    yield f"event: error\ndata: {json.dumps({'message': error})}\n\n"
                yield "event: done\ndata: {}\n\n"
                return
            payload = {
                "stage": ev.stage,
                "current": ev.current,
                "total": ev.total,
                "eta_s": ev.eta_s,
                "message": ev.message,
            }
            yield f"event: progress\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.post("/api/degradation")
def run_degradation_api(req: DegRequest):
    try:
        stack = build_stack(req.config_path, req.device)
        result = degradation.run_degradation(
            stack, t_end=req.t_end, n_snapshots=req.n_snapshots,
            N_grid=req.N_grid, V_bias=req.V_bias,
            metric_V_max=req.metric_V_max,
            metric_settle_time=req.metric_settle_time,
        )
        out = to_serializable(result)
        if "t" in out:
            out["times"] = out.pop("t")
        return {"status": "ok", "result": out}
    except HTTPException:
        raise
    except Exception as e:
        print("[Degradation API Exception]", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
