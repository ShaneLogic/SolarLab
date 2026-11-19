"""Source-bound short solarlab steady registration for real HTTP/worker tests."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess

from pydantic import BaseModel, ConfigDict, Field

from examples.research_1d.steady import load_device
from solarlab.config.identity import BuildSource, ExecutionEngine
from solarlab.experiments.jv.worker import SteadyJVProtocol, prepare_steady_jv_job
from solarlab.experiments.registry import ExperimentRegistry
from solarlab.experiments.worker import ResourceLimits
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab_server.steady_preparation import SteadyPreparationContext, steady_execution_catalog

ROOT = Path(__file__).resolve().parents[2]


class Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    N_grid: int = Field(default=8, ge=4, le=8)
    n_points: int = Field(default=3, ge=1, le=3)
    V_max: float = Field(default=0.02, ge=0.0, le=0.03)
    v_rate: float = Field(default=1000.0, gt=0.0)


def _working_tree_implementation() -> tuple[BuildSource, str]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
    pkg_root = ROOT / "src" / "solarlab"
    sources = tuple(
        SourceDocument("solarlab/" + path.relative_to(pkg_root).as_posix(), path.read_bytes())
        for path in sorted(pkg_root.rglob("*.py"))
        if path.stat().st_size > 0
    )
    return BuildSource(head, sources), head


def catalog(*, denied: bool = False, wall_seconds: float = 30.0, ceiling: ResourceLimits | None = None):
    implementation, head = _working_tree_implementation()
    base_device = load_device()
    sample_table = ResourceTable(
        "fixture_nk",
        "nk",
        SourceDocument("fixture_nk.csv", b"wavelength_nm,n,k\n500,2.5,0.1\n600,2.4,0.05\n"),
    )
    prepared = replace(base_device, resources=ResourceLibrary((sample_table,)))
    context = SteadyPreparationContext((("research_1d", prepared),), implementation, head)
    cap = ceiling or ResourceLimits(wall_seconds, 1536 * 1024**2, 96 * 1024**2, None, None)
    built = steady_execution_catalog(context, ceiling=cap)
    if denied:
        defn = built.registry.get("steady_jv", 1)
        denied_reg = ExperimentRegistry(
            (replace(defn, capability=replace(defn.capability, dimensions=(2,))),)
        )
        built = replace(built, registry=denied_reg, expected_registry_sha256=denied_reg.content_sha256)
    job = prepare_steady_jv_job(
        prepared,
        SteadyJVProtocol((0.0, 0.05), 6, 1.0, None),
        implementation=implementation,
        current_repository_head=head,
        engine=ExecutionEngine("solarlab"),
    )
    return built, job


def request(run_id: str = "http-case", *, n_points: int = 2, intervals_per_layer: int = 6) -> dict:
    voltages = [0.0] if n_points == 1 else [round(0.05 * i, 4) for i in range(n_points)]
    return {
        "schema_version": "solarlab.job-submission.v1",
        "run_id": run_id,
        "experiment_id": "steady_jv",
        "version": 1,
        "engine": "solarlab",
        "input": {
            "device_id": "research_1d",
            "voltage_V": voltages,
            "intervals_per_layer": intervals_per_layer,
            "generation_scale": 1.0,
            "incident_power_W_m2": None,
        },
        "limits": {
            "wall_seconds": 30.0,
            "rss_bytes": 1536 * 1024**2,
            "output_bytes": 96 * 1024**2,
            "accepted_steps": None,
            "calls": None,
        },
    }
