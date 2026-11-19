"""Production CLI entrypoint for ``solarlab_server`` (atomic V2 cutover).

Launches ``solarlab_server.app:create_app`` with single-thread BLAS, locked
assembled equations, packaged/trusted configuration resources, installed
``solarlab`` execution routes, and optional compiled SPA static serving.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import importlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from solarlab.config.identity import BuildSource
    from solarlab.device.defaults import DefaultCatalog
    from solarlab.device.resolved import PreparedDevice
    from solarlab.materials.resources import ResourceLibrary
    from solarlab.materials.source import SourceDocument
    from solarlab_server.configuration import ConfigurationPreviewContext
    from solarlab_server.job_service import ExecutionCatalog

_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def enforce_server_environment() -> None:
    """Pin single-thread BLAS and refuse legacy equation overrides before engine imports."""
    for name in _THREAD_ENV:
        os.environ[name] = "1"
    eq = os.environ.get("NEWCORE_EQUATIONS")
    if eq is not None and eq != "assembled":
        raise ValueError(
            f"solarlab_server requires NEWCORE_EQUATIONS='assembled' (received {eq!r}); "
            "legacy worker equation selection is retired"
        )
    os.environ["NEWCORE_EQUATIONS"] = "assembled"


def resolve_default_web_dist() -> Path | None:
    """Locate compiled SPA bundle in packaged ``_web_dist`` or checkout ``web/dist``."""
    here = Path(__file__).resolve().parent
    candidates = (
        here / "_web_dist",
        here.parents[1] / "web" / "dist",
    )
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "index.html").is_file():
            return candidate
    return None


def _installed_build_source(commit: str | None = None) -> tuple[BuildSource, str]:
    """Build ``BuildSource`` from explicit Git commit archive when requested, else installed package files."""
    from solarlab.config.identity import BuildSource
    from solarlab.materials.source import SourceDocument

    repo_root = Path(__file__).resolve().parents[2]
    explicit_commit = commit
    resolved_commit = commit or os.environ.get("SOLARLAB_BUILD_COMMIT")
    head = os.environ.get("SOLARLAB_REPOSITORY_HEAD")

    if (repo_root / ".git").exists():
        try:
            git_head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, timeout=5
            ).strip()
            if resolved_commit is None:
                resolved_commit = git_head
            if head is None:
                head = git_head
            if explicit_commit is not None and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", explicit_commit):
                archive_bytes = subprocess.check_output(
                    ["git", "archive", explicit_commit, "--", "src/solarlab"],
                    cwd=repo_root,
                    timeout=15,
                )
                sources: list[SourceDocument] = []
                with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive:
                    for item in archive.getmembers():
                        if item.isfile() and item.name.endswith(".py") and item.size:
                            stream = archive.extractfile(item)
                            if stream is not None:
                                sources.append(SourceDocument(item.name.removeprefix("src/"), stream.read()))
                if sources and head is not None:
                    return BuildSource(explicit_commit, tuple(sources)), head
        except (OSError, subprocess.SubprocessError):
            pass

    if resolved_commit is None or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", resolved_commit) is None:
        resolved_commit = "0" * 40
    if head is None or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", head) is None:
        head = resolved_commit

    pkg = importlib.import_module("solarlab")
    pkg_file = getattr(pkg, "__file__", None)
    if pkg_file is None:
        raise RuntimeError("unable to locate installed solarlab package files")
    pkg_root = Path(pkg_file).resolve().parent
    installed_sources = tuple(
        SourceDocument("solarlab/" + path.relative_to(pkg_root).as_posix(), path.read_bytes())
        for path in sorted(pkg_root.rglob("*.py"))
        if path.stat().st_size > 0
    )
    return BuildSource(resolved_commit, installed_sources), head


def _baseline_research_1d_yaml() -> bytes:
    """Shipped baseline ion-free classical 1D pin preset (zero legacy package reads)."""
    return (
        b"device:\n"
        b"  ion_steric_diffusion_only: false\n"
        b"  ion_steric_shared_site: false\n"
        b"  Phi: 2.5e21\n"
        b"  T: 300.0\n"
        b"  mode: full\n"
        b"  built_in_potential_mode: semiconductor_work_function\n"
        b"layers:\n"
        b"  - name: HTL\n"
        b"    role: HTL\n"
        b"    thickness: 1.0e-7\n"
        b"    eps_r: 3.5\n"
        b"    mu_n: 1.0e-6\n"
        b"    mu_p: 2.0e-5\n"
        b"    ni: 1.0e8\n"
        b"    N_D: 0.0\n"
        b"    N_A: 1.0e24\n"
        b"    D_ion: 0.0\n"
        b"    P0: 0.0\n"
        b"    P_lim: 1.0e28\n"
        b"    tau_n: 1.0e-6\n"
        b"    tau_p: 1.0e-6\n"
        b"    n1: 1.0e8\n"
        b"    p1: 1.0e8\n"
        b"    B_rad: 0.0\n"
        b"    C_n: 0.0\n"
        b"    C_p: 0.0\n"
        b"    alpha: 0.0\n"
        b"    Eg: 2.2\n"
        b"    chi: 3.2\n"
        b"  - name: Absorber\n"
        b"    role: absorber\n"
        b"    thickness: 4.0e-7\n"
        b"    eps_r: 24.1\n"
        b"    mu_n: 2.0e-4\n"
        b"    mu_p: 2.0e-4\n"
        b"    ni: 1.0e11\n"
        b"    N_D: 0.0\n"
        b"    N_A: 0.0\n"
        b"    D_ion: 0.0\n"
        b"    P0: 0.0\n"
        b"    P_lim: 1.0e28\n"
        b"    tau_n: 5.0e-7\n"
        b"    tau_p: 5.0e-7\n"
        b"    n1: 1.0e11\n"
        b"    p1: 1.0e11\n"
        b"    B_rad: 5.0e-17\n"
        b"    C_n: 1.0e-40\n"
        b"    C_p: 1.0e-40\n"
        b"    alpha: 1.0e7\n"
        b"    Eg: 1.6\n"
        b"    chi: 3.9\n"
        b"  - name: ETL\n"
        b"    role: ETL\n"
        b"    thickness: 1.0e-7\n"
        b"    eps_r: 9.0\n"
        b"    mu_n: 5.0e-5\n"
        b"    mu_p: 1.0e-6\n"
        b"    ni: 1.0e7\n"
        b"    N_D: 1.0e24\n"
        b"    N_A: 0.0\n"
        b"    D_ion: 0.0\n"
        b"    P0: 0.0\n"
        b"    P_lim: 1.0e28\n"
        b"    tau_n: 1.0e-6\n"
        b"    tau_p: 1.0e-6\n"
        b"    n1: 1.0e7\n"
        b"    p1: 1.0e7\n"
        b"    B_rad: 0.0\n"
        b"    C_n: 0.0\n"
        b"    C_p: 0.0\n"
        b"    alpha: 0.0\n"
        b"    Eg: 3.2\n"
        b"    chi: 4.0\n"
    )


def load_packaged_or_example_bundle(
    *,
    repo_root: Path | None = None,
) -> tuple[
    DefaultCatalog,
    ResourceLibrary,
    tuple[SourceDocument, ...],
    tuple[tuple[str, PreparedDevice], ...],
]:
    """Load packaged defaults/resources (when available) and prepare installed research_1d."""
    from solarlab.config.resolve_device import resolve_device
    from solarlab.device.defaults import DefaultCatalog
    from solarlab.device.inputs import DeviceInput
    from solarlab.materials import (
        load_packaged_default_catalog,
        load_packaged_preset_device_mapping,
        load_packaged_preset_sources,
        load_packaged_resource_library,
    )
    from solarlab.materials.source import SourceDocument

    resolved_root = Path(__file__).resolve().parents[2] if repo_root is None else repo_root
    example_json = resolved_root / "examples" / "research_1d" / "device.json"

    defaults: DefaultCatalog = load_packaged_default_catalog()
    resources: ResourceLibrary = load_packaged_resource_library()
    extra_presets: tuple[SourceDocument, ...] = tuple(load_packaged_preset_sources())

    yaml_doc = SourceDocument("research_1d.yaml", _baseline_research_1d_yaml())
    seen_ids = {yaml_doc.id}
    dedup_presets = tuple(doc for doc in extra_presets if doc.id not in seen_ids and not seen_ids.add(doc.id))  # type: ignore[func-returns-value]
    if example_json.is_file():
        json_source = SourceDocument("examples/research_1d/device.json", example_json.read_bytes())
        parsed = json.loads(json_source.content)
        example_catalog = DefaultCatalog.from_mapping(
            {**parsed["default_catalog"], "evidence": [[json_source.id, json_source.sha256]]}
        )
        prepared = resolve_device(
            DeviceInput.model_validate(parsed["device"]),
            example_catalog,
            resources,
            sources=(json_source,),
        )
        return defaults, resources, (yaml_doc, *dedup_presets, json_source), (("research_1d", prepared),)

    preset_map = load_packaged_preset_device_mapping("classical_pin")
    preset_doc = next(
        (doc for doc in extra_presets if doc.id == "classical_pin.device.json"),
        SourceDocument("classical_pin.device.json", json.dumps(preset_map, sort_keys=True).encode()),
    )
    prepared = resolve_device(
        DeviceInput.model_validate({**preset_map, "id": "research_1d"}),
        defaults,
        resources,
        sources=(preset_doc,),
    )
    return defaults, resources, (yaml_doc, *dedup_presets), (("research_1d", prepared),)


def build_default_server_components(
    *,
    build_source_commit: str | None = None,
    repo_root: Path | None = None,
) -> tuple[ConfigurationPreviewContext, ExecutionCatalog]:
    """Assemble V2 preview context and solarlab ExecutionCatalog (1D steady, 1D transient, 2D steady)."""
    from solarlab.experiments.registry import ExperimentRegistry
    from solarlab.experiments.worker import ResourceLimits
    from solarlab_server.configuration import ConfigurationPreviewContext
    from solarlab_server.job_service import ExecutionCatalog
    from solarlab_server.steady_2d_preparation import Steady2DPreparationContext, steady_2d_execution_catalog
    from solarlab_server.steady_preparation import SteadyPreparationContext, steady_execution_catalog
    from solarlab_server.transient_preparation import TransientPreparationContext, transient_execution_catalog

    enforce_server_environment()
    implementation, head = _installed_build_source(build_source_commit)
    defaults, resources, sources, devices = load_packaged_or_example_bundle(repo_root=repo_root)
    preview = ConfigurationPreviewContext(defaults, resources, sources)

    ceiling = ResourceLimits(30.0, 1536 * 1024**2, 96 * 1024**2, None, None)
    steady_cat = steady_execution_catalog(
        SteadyPreparationContext(devices, implementation, head), ceiling=ceiling
    )
    transient_cat = transient_execution_catalog(
        TransientPreparationContext(devices, implementation, head), ceiling=ceiling
    )
    steady_2d_cat = steady_2d_execution_catalog(
        Steady2DPreparationContext(devices, implementation, head), ceiling=ceiling
    )

    sub_catalogs = (steady_cat, transient_cat, steady_2d_cat)
    all_sources = {doc.id: doc for cat in sub_catalogs for doc in cat.sources}
    registry = ExperimentRegistry(tuple(defn for cat in sub_catalogs for defn in cat.registry.definitions))
    combined = ExecutionCatalog(
        registry=registry,
        models=steady_cat.models,
        sources=tuple(all_sources.values()),
        registrations=tuple(reg for cat in sub_catalogs for reg in cat.registrations),
        expected_registry_sha256=registry.content_sha256,
        expected_model_sha256=steady_cat.models.content_sha256,
        ceiling=ceiling,
    )
    return preview, combined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="solarlab-server",
        description="Run the SolarLab V2 HTTP/SSE service and optional compiled web workstation.",
    )
    parser.add_argument(
        "--store-dir",
        "--store-root",
        dest="store_dir",
        type=Path,
        default=Path(os.environ.get("SOLARLAB_STORE_DIR", "./var/solarlab/store")),
        help="local RunStore directory (created if absent unless --read-only)",
    )
    parser.add_argument(
        "--work-dir",
        dest="work_dir",
        type=Path,
        default=None,
        help="local ProcessSupervisor scratch directory (default: <store-dir>-work)",
    )
    parser.add_argument("--host", default=os.environ.get("SOLARLAB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SOLARLAB_PORT", "8000")))
    parser.add_argument("--build-source-commit", default=None, help="optional pinned build-source commit SHA")
    parser.add_argument("--web-dist", type=Path, default=None, help="compiled SPA dist directory containing index.html")
    parser.add_argument("--no-web-dist", action="store_true", help="serve API routes only without mounting SPA assets")
    parser.add_argument("--read-only", action="store_true", help="open RunStore read-only without JobService execution")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify preview context, execution catalog, RunStore, and FastAPI app without binding a socket",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    enforce_server_environment()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")

    import uvicorn
    from solarlab.experiments.execution import ProcessSupervisor
    from solarlab.io.run_store import RunStore
    from solarlab_server.app import create_app
    from solarlab_server.job_service import JobService

    store_dir = args.store_dir.expanduser().resolve()
    work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir is not None
        else store_dir.with_name(store_dir.name + "-work")
    )
    web_dist: Path | None = None
    if not args.no_web_dist:
        web_dist = args.web_dist.expanduser().resolve() if args.web_dist is not None else resolve_default_web_dist()

    preview, catalog = build_default_server_components(build_source_commit=args.build_source_commit)

    if args.check_only:
        if not args.read_only:
            store_dir.parent.mkdir(parents=True, exist_ok=True)
            with RunStore(store_dir):
                pass
        create_app(store_dir, configuration_preview=preview, web_dist_dir=web_dist)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "engine": "solarlab",
                    "equation_implementation": "assembled",
                    "store_dir": str(store_dir),
                    "installed_experiments": [reg.experiment_id for reg in catalog.registrations],
                    "execution_catalog_sha256": catalog.document.sha256,
                    "job_schema_sha256": catalog.schema.sha256,
                    "web_dist_dir": str(web_dist) if web_dist is not None else None,
                    "scientific_qualification": False,
                }
            ),
            flush=True,
        )
        return 0

    if args.read_only:
        app = create_app(store_dir, configuration_preview=preview, web_dist_dir=web_dist)
        uvicorn.run(app, host=args.host, port=args.port, workers=1, loop="asyncio", http="h11")
        return 0

    store_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    with RunStore(store_dir) as store, ProcessSupervisor(
        store,
        work_dir,
        catalog.worker_bindings(),
        max_active=1,
        max_pending=4,
        shared_rss_bytes=3 * 1024**3,
        shared_output_bytes=384 * 1024**2,
    ) as supervisor:
        app = create_app(
            store.root,
            configuration_preview=preview,
            job_service=JobService(catalog, supervisor),
            web_dist_dir=web_dist,
        )
        summary: dict[str, Any] = {
            "base_url": f"http://{args.host}:{args.port}",
            "engine": "solarlab",
            "equation_implementation": "assembled",
            "installed_experiments": [reg.experiment_id for reg in catalog.registrations],
            "execution_catalog_sha256": catalog.document.sha256,
            "job_schema_sha256": catalog.schema.sha256,
            "web_dist_dir": str(web_dist) if web_dist is not None else None,
            "scientific_qualification": False,
        }
        print(json.dumps(summary), flush=True)
        uvicorn.run(app, host=args.host, port=args.port, workers=1, loop="asyncio", http="h11")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
