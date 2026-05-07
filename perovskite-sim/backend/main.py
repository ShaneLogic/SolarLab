import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# BLAS thread pinning — defensive fix. The Radau solver hits ~300x300 dense LU
# factors thousands of times per J-V sweep. On a multi-core machine, OpenBLAS
# and MKL try to parallelise each call across every core; at this matrix size
# thread-creation + contention overhead can dominate and turn a ~3 s sweep into
# several minutes. The slow test suite (tests/conftest.py) pins BLAS for the
# same reason, but the backend previously inherited no such guard, so the first
# TMM J-V sweep from the UI could intermittently stall.
#
# Set the env vars BEFORE importing numpy so BLAS reads them on library load.
# Opt out with PEROVSKITE_BLAS_PIN=0 if running on a dedicated box and you want
# the solver to use all cores for something larger (e.g. parallel sweeps).
if os.environ.get("PEROVSKITE_BLAS_PIN", "1") != "0":
    for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_var, "1")

import numpy as np
import traceback
import yaml
from dataclasses import asdict, is_dataclass
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from perovskite_sim.experiments import degradation, impedance, jv_sweep
from perovskite_sim.experiments import dark_jv as dark_jv_exp
from perovskite_sim.experiments import suns_voc as suns_voc_exp
from perovskite_sim.experiments import eqe as eqe_exp
from perovskite_sim.experiments import mott_schottky as ms_exp
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.mode import resolve_mode
from perovskite_sim.models.parameters import MaterialParams
from backend.jobs import JobRegistry, JobStatus, _DRAIN_TIMEOUT
from backend.progress import ProgressReporter
from backend.user_configs import (
    is_shipped_name,
    validate_user_filename,
    write_user_config,
)

def _describe_active_physics(stack) -> str:
    """Return a short human-readable description of the active physics tier.

    Used by the SSE result payload so the frontend solver console can
    show which Phase 1–3 upgrades ran without re-deriving the flags.
    """
    mode_name = str(getattr(stack, "mode", "full")).lower()
    mode = resolve_mode(mode_name)
    # Drive every label fragment off the mode flags so the indicator can't
    # silently drift from the physics that actually ran. Missing labels
    # (e.g. PR when use_photon_recycling is False) are left out rather than
    # rendered as "no PR" to keep the string short.
    parts: list[str] = [
        "band offsets · TE" if mode.use_thermionic_emission else "flat bands",
        "TMM" if mode.use_tmm_optics else "Beer-Lambert",
        "dual ions" if mode.use_dual_ions else "single ion",
        "trap profile" if mode.use_trap_profile else "uniform τ",
        "T-scaling" if mode.use_temperature_scaling else "T=300K",
    ]
    # Phase 3.x extras appended only when on, so the label stays short for
    # LEGACY / FAST and expands only for FULL or custom modes that opt in.
    if mode.use_photon_recycling:
        parts.append("photon recycling")
    if mode.use_radiative_reabsorption:
        parts.append("PR reabsorption")
    if mode.use_field_dependent_mobility:
        parts.append("μ(E)")
    if mode.use_selective_contacts:
        parts.append("Robin contacts")
    return f"{mode.name.upper()}  " + " · ".join(parts)


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


def _opt_S(v) -> Optional[float]:
    """Parse a Robin contact S value. ``None`` and missing → None
    (= ohmic Dirichlet, the documented "absent / disabled" sentinel);
    every other value is coerced to float, including 0.0 (= Neumann
    blocking — distinct from absent)."""
    if v is None:
        return None
    return float(v)


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
            # Stage B(c.2) field-dependent mobility μ(E). Defaults match
            # MaterialParams: v_sat / pf_gamma at 0 (= disabled); ct_beta
            # at 2 (= Canali silicon-electron form, the safe default
            # documented in field_mobility.py).
            v_sat_n=float(layer_cfg.get("v_sat_n", 0.0)),
            v_sat_p=float(layer_cfg.get("v_sat_p", 0.0)),
            ct_beta_n=float(layer_cfg.get("ct_beta_n", 2.0)),
            ct_beta_p=float(layer_cfg.get("ct_beta_p", 2.0)),
            pf_gamma_n=float(layer_cfg.get("pf_gamma_n", 0.0)),
            pf_gamma_p=float(layer_cfg.get("pf_gamma_p", 0.0)),
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
    # Stage B(a) microstructure — mirror load_device_from_yaml's behaviour
    # so the inline-device path round-trips the ``microstructure:`` block
    # the same way the YAML loader does. Without this, loading a preset
    # like configs/twod/nip_MAPbI3_singleGB.yaml in the workstation and
    # submitting via ``device:`` would silently drop the GB block at the
    # backend boundary (the frontend's startJob always sends device: and
    # never config_path:, so the load_device_from_yaml microstructure
    # path is never used at runtime). Lazy import keeps FastAPI startup
    # cost unchanged when no jv_2d run is dispatched.
    ms_block = cfg.get("microstructure")
    if ms_block:
        from perovskite_sim.twod.microstructure import (
            load_microstructure_from_yaml_block,
        )
        microstructure = load_microstructure_from_yaml_block(ms_block)
    else:
        from perovskite_sim.twod.microstructure import Microstructure
        microstructure = Microstructure()
    return DeviceStack(
        layers=tuple(layers),
        V_bi=float(dev.get("V_bi", 1.1)),
        Phi=float(dev.get("Phi", 2.5e21)),
        interfaces=interfaces,
        T=float(dev.get("T", 300.0)),
        mode=mode_name,
        # Stage B(c.1) Robin / selective contacts. None = ohmic Dirichlet
        # (the pre-3.3 default); 0 = Neumann blocking; positive finite =
        # Robin. The frontend distinguishes these three states via
        # parseNumOrNull in config-editor.ts.
        S_n_left=_opt_S(dev.get("S_n_left")),
        S_p_left=_opt_S(dev.get("S_p_left")),
        S_n_right=_opt_S(dev.get("S_n_right")),
        S_p_right=_opt_S(dev.get("S_p_right")),
        microstructure=microstructure,
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
    """List YAML configs available to the frontend.

    Each entry carries a ``namespace`` tag so the frontend can render the
    dropdown as two ``<optgroup>``s — shipped (top-level configs/) and user
    (configs/user/). Returning a list of dicts is a deliberate breaking
    change vs the Phase 2a flat-list shape; the frontend api wrapper updates
    in lockstep.
    """
    def _peek_metadata(path: str) -> tuple[str, list[str]]:
        # Cheap YAML peek — returns (device_type, tier_compat).
        #
        # device_type: tandem presets have no stack.layers and must be routed
        # to the Tandem pane instead of the single-cell Device editor.
        #
        # tier_compat: list of physics tiers this preset runs correctly under.
        # Every preset supports 'legacy' and 'fast' — both tiers no-op safely
        # when layers leave the opt-in Phase 1/2/3.1 parameters unset: TE
        # needs non-zero band offsets between neighbouring layers (chi/Eg),
        # TMM needs `optical_material` on a layer, PR needs TMM to be active,
        # dual-ion and trap-profile and T-scaling each need their own opt-in
        # config keys. 'full' is gated on *every* electrical layer having
        # chi > 0 AND Eg > 0 — FULL tier derives the built-in potential from
        # Fermi-level matching across the heterostack, so a single missing
        # band alignment collapses compute_V_bi() and the diode fails to
        # turn on at V=0 (Stage 1a diagnosis, commit c93d854). Tandem
        # presets are single-cell-only for now and only advertise legacy/fast.
        legacy_tiers = ["legacy", "fast"]
        try:
            with open(path) as fh:
                data = yaml.safe_load(fh) or {}
        except Exception:
            return "single", legacy_tiers
        device_type = str(data.get("device_type", "single"))
        if device_type != "single":
            return device_type, legacy_tiers
        layers = data.get("layers") or []
        electrical = [l for l in layers if l.get("role") != "substrate"]
        if electrical and all(
            float(l.get("chi", 0.0) or 0.0) > 0.0
            and float(l.get("Eg", 0.0) or 0.0) > 0.0
            for l in electrical
        ):
            return device_type, [*legacy_tiers, "full"]
        return device_type, legacy_tiers

    try:
        entries: list[dict] = []
        seen_names: set[str] = set()
        for f in sorted(os.listdir(CONFIGS_DIR)):
            if f.endswith((".yaml", ".yml")):
                full = os.path.join(CONFIGS_DIR, f)
                if os.path.isfile(full):
                    device_type, tier_compat = _peek_metadata(full)
                    entries.append({
                        "name": f,
                        "namespace": "shipped",
                        "device_type": device_type,
                        "tier_compat": tier_compat,
                    })
                    seen_names.add(f)
        user_dir = os.path.join(CONFIGS_DIR, "user")
        if os.path.isdir(user_dir):
            for f in sorted(os.listdir(user_dir)):
                if f.endswith((".yaml", ".yml")):
                    full = os.path.join(user_dir, f)
                    device_type, tier_compat = _peek_metadata(full)
                    entries.append({
                        "name": f,
                        "namespace": "user",
                        "device_type": device_type,
                        "tier_compat": tier_compat,
                    })
                    seen_names.add(f)
        # Phase 6 shipped 2D presets live under ``configs/twod/`` (Stage A
        # baseline, Stage B(a) microstructure, T7 B(c.x) demo). They are
        # listed in the same ``shipped`` namespace as top-level presets so
        # the existing dropdown surfaces them without a UI redesign.
        # Collision policy: top-level precedence is preserved — a basename
        # already listed (top-level or user/) shadows the twod entry, and
        # the shadowing is reported on stderr so a maintainer can dedupe.
        twod_dir = os.path.join(CONFIGS_DIR, "twod")
        if os.path.isdir(twod_dir):
            for f in sorted(os.listdir(twod_dir)):
                if not f.endswith((".yaml", ".yml")):
                    continue
                if f in seen_names:
                    print(
                        f"[list_configs] basename collision: 'configs/twod/{f}' "
                        f"shadowed by an earlier entry; skipping",
                        file=sys.stderr,
                    )
                    continue
                full = os.path.join(twod_dir, f)
                if os.path.isfile(full):
                    device_type, tier_compat = _peek_metadata(full)
                    entries.append({
                        "name": f,
                        "namespace": "shipped",
                        "device_type": device_type,
                        "tier_compat": tier_compat,
                    })
                    seen_names.add(f)
        return {"status": "ok", "configs": entries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/optical-materials")
def list_optical_materials() -> dict:
    """Auto-scan ``perovskite_sim/data/nk/`` and return the sorted material list.

    The frontend optical-material picker calls this to populate its dropdown,
    so dropping a new ``<name>.csv`` in the nk directory makes it visible with
    no code change (same convention as ``/api/configs``).
    """
    nk_dir = Path(__file__).resolve().parent.parent / "perovskite_sim" / "data" / "nk"
    return {"materials": sorted(p.stem for p in nk_dir.glob("*.csv"))}


@app.get("/api/layer-templates")
def list_layer_templates() -> dict:
    """Return the parsed layer templates library used by the Add Layer dialog.

    The library lives in ``perovskite_sim/data/layer_templates.yaml`` so the
    frontend can populate the dialog without re-deriving material defaults.
    """
    path = (
        Path(__file__).resolve().parent.parent
        / "perovskite_sim"
        / "data"
        / "layer_templates.yaml"
    )
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail="layer_templates.yaml missing — Phase 2b data file not installed",
        )
    with path.open() as f:
        templates = yaml.safe_load(f) or {}
    return {"status": "ok", "templates": _coerce_numbers(templates)}


@app.get("/api/configs/{name}")
def get_config(name: str):
    """Return the parsed YAML device config so the frontend can edit it.

    Search order — top-level ``configs/`` → ``configs/user/`` →
    ``configs/twod/``. Top-level precedence is preserved on basename
    collision so a user-renamed top-level preset always wins, matching
    the listing order in :func:`list_configs`. ``os.path.basename``
    strips any leading path components in case a caller URL-encodes a
    slash.
    """
    safe_name = os.path.basename(name)
    candidates = [
        os.path.join(CONFIGS_DIR, safe_name),
        os.path.join(CONFIGS_DIR, "user", safe_name),
        os.path.join(CONFIGS_DIR, "twod", safe_name),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Config '{safe_name}' not found")
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return {"status": "ok", "name": safe_name, "config": _coerce_numbers(cfg)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UserConfigPayload(BaseModel):
    name: str
    config: dict
    overwrite: bool = False


@app.post("/api/configs/user")
def save_user_config(payload: UserConfigPayload):
    """Write a user-edited DeviceConfig to ``configs/user/<name>.yaml``.

    The frontend Save-As dialog calls this. ``user_configs`` owns filename
    validation, shipped-name reservation, and atomic writes.
    """
    try:
        validate_user_filename(payload.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if is_shipped_name(payload.name):
        raise HTTPException(
            status_code=409,
            detail=f"{payload.name!r} is reserved by a shipped preset",
        )
    try:
        write_user_config(payload.name, payload.config, overwrite=payload.overwrite)
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "saved": payload.name}


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


# ---------------------------------------------------------------------------
# Tandem endpoint
# ---------------------------------------------------------------------------

class TandemRequest(BaseModel):
    config_path: str
    N_grid: int = 40
    n_points: int = 15


@app.post("/api/tandem")
def run_tandem(req: TandemRequest):
    """Run a series-connected 2T tandem J-V sweep from a tandem YAML config.

    Loads the tandem config, builds the AM1.5G wavelength grid using the
    same parameters as ``_compute_tmm_generation`` (300–1000 nm, 200 points),
    calls ``run_tandem_jv``, and returns the series-matched J-V together with
    per-sub-cell voltages and four tandem metrics.
    """
    import dataclasses

    import numpy as np
    from perovskite_sim.data import load_am15g
    from perovskite_sim.experiments.tandem_jv import run_tandem_jv
    from perovskite_sim.models.tandem_config import load_tandem_from_yaml

    # --- load config -------------------------------------------------------
    try:
        cfg = load_tandem_from_yaml(resolve_config_path(req.config_path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # --- build wavelength grid (same defaults as _compute_tmm_generation) --
    try:
        lam_min, lam_max, n_wl = 300.0, 1000.0, 200
        wavelengths_nm = np.linspace(lam_min, lam_max, n_wl)
        wavelengths_m = wavelengths_nm * 1e-9
        _, spectral_flux = load_am15g(wavelengths_nm)

        result = run_tandem_jv(
            cfg,
            wavelengths_m,
            spectral_flux,
            wavelengths_nm,
            N_grid=req.N_grid,
            n_points=req.n_points,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        # series_match_jv raises when sub-cell J ranges do not overlap, which
        # happens when the stub FA_Cs_1p77 / SnPb_1p22 n,k CSVs mis-match the
        # real Lin 2019 spectral response. Surface a 400 with a clear pointer.
        msg = str(exc)
        if "Sub-cell J ranges do not overlap" in msg:
            detail = (
                f"{msg} — this tandem preset ships with stub n,k data "
                "(rigid bandgap shifts of MAPbI3). Replace "
                "perovskite_sim/data/nk/FA_Cs_1p77.csv and SnPb_1p22.csv "
                "with real Lin 2019 SI data before expecting physical results."
            )
            raise HTTPException(status_code=400, detail=detail)
        raise HTTPException(status_code=400, detail=msg)
    except Exception as exc:
        print("[Tandem API Exception]", exc)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))

    # --- serialise metrics (frozen dataclass → dict) -----------------------
    metrics_dict = dataclasses.asdict(result.metrics)

    return {
        "V": result.V.tolist(),
        "J": result.J.tolist(),
        "V_top": result.V_top.tolist(),
        "V_bot": result.V_bot.tolist(),
        "metrics": metrics_dict,
        "benchmark": cfg.benchmark,
    }


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
    kind: str  # "jv" | "impedance" | "degradation" | "tpv" | "current_decomp" | "spatial"
               # | "dark_jv" | "suns_voc" | "voc_t" | "eqe" | "el" | "mott_schottky"
               # | "tandem"
    config_path: Optional[str] = None
    device: Optional[dict] = None
    params: dict = {}


@app.post("/api/jobs")
def start_job(req: JobRequest):
    """Start an experiment on a worker thread and return a job ID.

    The caller then opens GET /api/jobs/{id}/events to receive
    Server-Sent-Events with incremental progress and the final result.
    """
    kind = req.kind
    p = req.params

    # Tandem is config-only (no single DeviceStack), so it skips build_stack.
    if kind != "tandem":
        try:
            stack = build_stack(req.config_path, req.device)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"stack build failed: {e}")

    if kind == "jv":
        def _run(reporter: ProgressReporter) -> dict:
            # illuminated defaults to True; frontend sends False for dark J-V
            _illum = p.get("illuminated", True)
            illuminated = bool(_illum) if not isinstance(_illum, str) else _illum.lower() != "false"
            result = jv_sweep.run_jv_sweep(
                stack,
                N_grid=int(p.get("N_grid", 60)),
                n_points=int(p.get("n_points", 30)),
                v_rate=float(p.get("v_rate", 1.0)),
                V_max=float(p["V_max"]) if p.get("V_max") is not None else None,
                illuminated=illuminated,
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
    elif kind == "current_decomp":
        def _run(reporter: ProgressReporter) -> dict:
            _illum = p.get("illuminated", True)
            illuminated = bool(_illum) if not isinstance(_illum, str) else _illum.lower() != "false"
            result = jv_sweep.run_jv_sweep(
                stack,
                N_grid=int(p.get("N_grid", 60)),
                n_points=int(p.get("n_points", 30)),
                v_rate=float(p.get("v_rate", 1.0)),
                V_max=float(p["V_max"]) if p.get("V_max") is not None else None,
                illuminated=illuminated,
                decompose_currents=True,
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = {}
            out["V_fwd"] = result.V_fwd.tolist()
            out["V_rev"] = result.V_rev.tolist()
            if result.decomp_fwd:
                out["Jn_fwd"] = result.decomp_fwd.J_n.tolist()
                out["Jp_fwd"] = result.decomp_fwd.J_p.tolist()
                out["Jion_fwd"] = result.decomp_fwd.J_ion.tolist()
                out["Jdisp_fwd"] = result.decomp_fwd.J_disp.tolist()
                out["Jtotal_fwd"] = result.decomp_fwd.J_total.tolist()
            if result.decomp_rev:
                out["Jn_rev"] = result.decomp_rev.J_n.tolist()
                out["Jp_rev"] = result.decomp_rev.J_p.tolist()
                out["Jion_rev"] = result.decomp_rev.J_ion.tolist()
                out["Jdisp_rev"] = result.decomp_rev.J_disp.tolist()
                out["Jtotal_rev"] = result.decomp_rev.J_total.tolist()
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "spatial":
        def _run(reporter: ProgressReporter) -> dict:
            _illum = p.get("illuminated", True)
            illuminated = bool(_illum) if not isinstance(_illum, str) else _illum.lower() != "false"
            result = jv_sweep.run_jv_sweep(
                stack,
                N_grid=int(p.get("N_grid", 60)),
                n_points=int(p.get("n_points", 15)),
                v_rate=float(p.get("v_rate", 1.0)),
                V_max=float(p["V_max"]) if p.get("V_max") is not None else None,
                illuminated=illuminated,
                save_snapshots=True,
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            # Convert snapshots to serialisable dicts with x in nm for readability
            def snap_to_dict(s):
                return {
                    "x": (s.x * 1e9).tolist(),        # nm
                    "phi": s.phi.tolist(),              # V
                    "E": s.E.tolist(),                  # V/m
                    "n": s.n.tolist(),                  # m^-3
                    "p": s.p.tolist(),                  # m^-3
                    "P": s.P.tolist(),                  # m^-3
                    "rho": s.rho.tolist(),              # C/m^3 (charge density * q)
                    "V_app": s.V_app,
                }
            out = {
                "V_fwd": result.V_fwd.tolist(),
                "V_rev": result.V_rev.tolist(),
                "snapshots_fwd": [snap_to_dict(s) for s in (result.snapshots_fwd or [])],
                "snapshots_rev": [snap_to_dict(s) for s in (result.snapshots_rev or [])],
            }
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "tpv":
        from perovskite_sim.experiments.tpv import run_tpv

        def _run(reporter: ProgressReporter) -> dict:
            result = run_tpv(
                stack,
                N_grid=int(p.get("N_grid", 80)),
                delta_G_frac=float(p.get("delta_G_frac", 0.05)),
                t_pulse=float(p.get("t_pulse", 1e-6)),
                t_decay=float(p.get("t_decay", 50e-6)),
                n_points=int(p.get("n_points", 200)),
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = to_serializable(result)
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "dark_jv":
        def _run(reporter: ProgressReporter) -> dict:
            result = dark_jv_exp.run_dark_jv(
                stack,
                V_max=float(p.get("V_max", 1.2)),
                n_points=int(p.get("n_points", 60)),
                N_grid=int(p.get("N_grid", 60)),
                v_rate=float(p.get("v_rate", 1.0)),
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = to_serializable(result)
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "suns_voc":
        def _run(reporter: ProgressReporter) -> dict:
            suns_raw = p.get("suns_levels")
            if suns_raw is None:
                suns_levels = suns_voc_exp.DEFAULT_SUNS
            else:
                suns_levels = tuple(float(x) for x in suns_raw)
            result = suns_voc_exp.run_suns_voc(
                stack,
                suns_levels=suns_levels,
                N_grid=int(p.get("N_grid", 60)),
                t_settle=float(p.get("t_settle", 1e-3)),
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = to_serializable(result)
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "voc_t":
        from perovskite_sim.experiments.voc_t import run_voc_t

        def _run(reporter: ProgressReporter) -> dict:
            result = run_voc_t(
                stack,
                T_min=float(p.get("T_min", 250.0)),
                T_max=float(p.get("T_max", 350.0)),
                n_points=int(p.get("n_points", 6)),
                N_grid=int(p.get("N_grid", 60)),
                jv_n_points=int(p.get("jv_n_points", 30)),
                v_rate=float(p.get("v_rate", 1.0)),
                V_max=float(p["V_max"]) if p.get("V_max") is not None else None,
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = to_serializable(result)
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "eqe":
        def _run(reporter: ProgressReporter) -> dict:
            lam_min = float(p.get("lambda_min_nm", 300.0))
            lam_max = float(p.get("lambda_max_nm", 1000.0))
            n_lam = int(p.get("n_lambda", 29))
            if n_lam < 2 or lam_max <= lam_min:
                raise ValueError(
                    "EQE sweep needs n_lambda >= 2 and lambda_max > lambda_min"
                )
            wavelengths_nm = np.linspace(lam_min, lam_max, n_lam)
            result = eqe_exp.compute_eqe(
                stack,
                wavelengths_nm=wavelengths_nm,
                Phi_incident=float(p.get("Phi_incident", 1e20)),
                N_grid=int(p.get("N_grid", 60)),
                t_settle=float(p.get("t_settle", 1e-3)),
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = to_serializable(result)
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "el":
        from perovskite_sim.experiments.el_spectrum import run_el_spectrum

        def _run(reporter: ProgressReporter) -> dict:
            lam_min = float(p.get("lambda_min_nm", 400.0))
            lam_max = float(p.get("lambda_max_nm", 1000.0))
            n_lam = int(p.get("n_lambda", 25))
            if n_lam < 2 or lam_max <= lam_min:
                raise ValueError(
                    "EL sweep needs n_lambda >= 2 and lambda_max > lambda_min"
                )
            wavelengths_nm = np.linspace(lam_min, lam_max, n_lam)
            result = run_el_spectrum(
                stack,
                V_inj=float(p.get("V_inj", 1.0)),
                wavelengths_nm=wavelengths_nm,
                N_grid=int(p.get("N_grid", 60)),
                n_points_dark=int(p.get("n_points_dark", 30)),
                v_rate=float(p.get("v_rate", 1.0)),
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = to_serializable(result)
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "mott_schottky":
        def _run(reporter: ProgressReporter) -> dict:
            V_lo = float(p.get("V_lo", -0.3))
            V_hi = float(p.get("V_hi", 0.4))
            n_pts = int(p.get("n_points", 8))
            if n_pts < 3 or V_hi <= V_lo:
                raise ValueError(
                    "Mott-Schottky needs n_points >= 3 and V_hi > V_lo"
                )
            V_range = np.linspace(V_lo, V_hi, n_pts)
            result = ms_exp.run_mott_schottky(
                stack,
                V_range=V_range,
                frequency=float(p.get("frequency", 1e5)),
                delta_V=float(p.get("delta_V", 0.01)),
                N_grid=int(p.get("N_grid", 40)),
                n_cycles=int(p.get("n_cycles", 5)),
                n_extract=int(p.get("n_extract", 2)),
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = to_serializable(result)
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "tandem":
        from perovskite_sim.data import load_am15g
        from perovskite_sim.experiments.tandem_jv import run_tandem_jv
        from perovskite_sim.models.tandem_config import load_tandem_from_yaml

        if not req.config_path:
            raise HTTPException(status_code=400, detail="tandem kind requires config_path")
        try:
            cfg = load_tandem_from_yaml(resolve_config_path(req.config_path))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        def _run(reporter: ProgressReporter) -> dict:
            wavelengths_nm = np.linspace(300.0, 1000.0, 200)
            wavelengths_m = wavelengths_nm * 1e-9
            _, spectral_flux = load_am15g(wavelengths_nm)
            try:
                result = run_tandem_jv(
                    cfg,
                    wavelengths_m,
                    spectral_flux,
                    wavelengths_nm,
                    N_grid=int(p.get("N_grid", 40)),
                    n_points=int(p.get("n_points", 15)),
                    progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
                )
            except ValueError as exc:
                msg = str(exc)
                if "Sub-cell J ranges do not overlap" in msg:
                    raise RuntimeError(
                        f"{msg} — this tandem preset ships with stub n,k data "
                        "(rigid bandgap shifts of MAPbI3). Replace "
                        "perovskite_sim/data/nk/FA_Cs_1p77.csv and "
                        "SnPb_1p22.csv with real Lin 2019 SI data before "
                        "expecting physical results."
                    )
                raise

            return {
                "V": result.V.tolist(),
                "J": result.J.tolist(),
                "V_top": result.V_top.tolist(),
                "V_bot": result.V_bot.tolist(),
                "metrics": dataclasses.asdict(result.metrics),
                "benchmark": cfg.benchmark,
            }
    elif kind == "jv_2d":
        from perovskite_sim.twod.experiments.jv_sweep_2d import run_jv_sweep_2d
        from perovskite_sim.twod.microstructure import (
            Microstructure, load_microstructure_from_yaml_block,
        )

        def _run(reporter: ProgressReporter) -> dict:
            _illum = p.get("illuminated", True)
            illuminated = bool(_illum) if not isinstance(_illum, str) else _illum.lower() != "false"
            _save = p.get("save_snapshots", True)
            save_snapshots = bool(_save) if not isinstance(_save, str) else _save.lower() != "false"

            # Microstructure resolution order:
            #   1. params.microstructure block (UI-supplied) — wins
            #   2. stack.microstructure (auto-attached by the YAML loader)
            #   3. Microstructure() (Stage-A lateral-uniform fallback)
            ms_block = p.get("microstructure")
            if ms_block:
                ms = load_microstructure_from_yaml_block(ms_block)
            else:
                ms = getattr(stack, "microstructure", None) or Microstructure()

            result = run_jv_sweep_2d(
                stack=stack,
                microstructure=ms,
                lateral_length=float(p.get("lateral_length", 500e-9)),
                Nx=int(p.get("Nx", 10)),
                V_max=float(p.get("V_max", 1.2)),
                V_step=float(p.get("V_step", 0.05)),
                illuminated=illuminated,
                lateral_bc=str(p.get("lateral_bc", "periodic")),
                Ny_per_layer=int(p.get("Ny_per_layer", 20)),
                settle_t=float(p.get("settle_t", 1e-7)),
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
                save_snapshots=save_snapshots,
            )

            def snap2d_to_dict(s):
                return {
                    "V": float(s.V),
                    "x": (s.x * 1e9).tolist(),
                    "y": (s.y * 1e9).tolist(),
                    "phi": s.phi.tolist(),
                    "n": s.n.tolist(),
                    "p": s.p.tolist(),
                    "Jx_n": s.Jx_n.tolist(),
                    "Jy_n": s.Jy_n.tolist(),
                    "Jx_p": s.Jx_p.tolist(),
                    "Jy_p": s.Jy_p.tolist(),
                }

            out = {
                "V": result.V.tolist(),
                "J": result.J.tolist(),
                "grid_x": (result.grid_x * 1e9).tolist(),
                "grid_y": (result.grid_y * 1e9).tolist(),
                "lateral_bc": result.lateral_bc,
                "snapshots": [snap2d_to_dict(s) for s in result.snapshots],
                # Centralised V_oc / J_sc / FF / PCE extraction (Layer 1+2
                # of the Phase 6 acceptance follow-up). Carries the
                # ``voc_bracketed`` flag so the frontend can warn the
                # user when V_max stopped short of V_oc; raw V/J above
                # are unchanged.
                "metrics": dataclasses.asdict(result.metrics),
            }
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "voc_grain_sweep":
        from perovskite_sim.twod.experiments.voc_grain_sweep import run_voc_grain_sweep

        def _run(reporter: ProgressReporter) -> dict:
            raw_sizes = p.get("grain_sizes_nm") or p.get("grain_sizes")
            if not raw_sizes:
                raise HTTPException(
                    status_code=400,
                    detail="voc_grain_sweep requires grain_sizes_nm (list of nm)",
                )
            grain_sizes_m = [float(s) * 1e-9 for s in raw_sizes]
            tau_n_gb = float(p.get("tau_gb_n", 1e-9))
            tau_p_gb = float(p.get("tau_gb_p", 1e-9))
            _illum = p.get("illuminated", True)
            illuminated = bool(_illum) if not isinstance(_illum, str) else _illum.lower() != "false"

            result = run_voc_grain_sweep(
                stack=stack,
                grain_sizes=grain_sizes_m,
                tau_gb=(tau_n_gb, tau_p_gb),
                gb_width=float(p.get("gb_width", 10e-9)),
                Nx=int(p.get("Nx", 10)),
                Ny_per_layer=int(p.get("Ny_per_layer", 10)),
                V_max=float(p.get("V_max", 1.2)),
                V_step=float(p.get("V_step", 0.05)),
                illuminated=illuminated,
                settle_t=float(p.get("settle_t", 1e-3)),
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            return {
                "grain_sizes_nm": (result.grain_sizes_m * 1e9).tolist(),
                "V_oc_V": result.V_oc_V.tolist(),
                "J_sc_Am2": result.J_sc_Am2.tolist(),
                "FF": result.FF.tolist(),
                "active_physics": _describe_active_physics(stack),
            }
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
