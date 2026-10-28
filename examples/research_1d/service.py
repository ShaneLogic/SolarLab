"""Run the explicitly installed steady J-V service on localhost.

Supply a full committed source revision and explicit local store/work paths.
GET /experiments supplies the device ID, input schema and required identity
headers for POST /jobs. Responses name the existing SSE and result readers.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from solarlab.config.identity import BuildSource

ROOT = Path(__file__).resolve().parents[2]


def committed_implementation(commit: str) -> BuildSource:
    """Read exact Git object bytes; never label working-tree bytes as a commit.

    This checkout example requires Git and the explicitly supplied commit.
    Installed-distribution hosts can instead supply their own verified
    BuildSource to SteadyPreparationContext. No HTTP field reaches this code.
    """
    from solarlab.config.identity import BuildSource
    from solarlab.materials.source import SourceDocument

    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None:
        raise ValueError("supply a full immutable committed source revision")
    content = subprocess.check_output(["git", "archive", commit, "--", "src/solarlab"], cwd=ROOT, timeout=15)
    sources = []
    with tarfile.open(fileobj=io.BytesIO(content)) as archive:
        for item in archive.getmembers():
            if item.isfile() and item.name.endswith(".py") and item.size:
                stream = archive.extractfile(item)
                if stream is None:
                    raise ValueError("missing committed source member")
                sources.append(SourceDocument(item.name.removeprefix("src/"), stream.read()))
    return BuildSource(commit, tuple(sources))


def main() -> None:
    import uvicorn
    from solarlab.experiments.execution import ProcessSupervisor
    from solarlab.experiments.worker import ResourceLimits
    from solarlab.io.run_store import RunStore
    from solarlab_server.app import create_app
    from solarlab_server.job_service import JobService
    from solarlab_server.steady_preparation import SteadyPreparationContext, steady_execution_catalog

    # Spawn re-executes this script as __mp_main__. Import the example's
    # numerical stack only in the owner, never ahead of the child's environment.
    if __package__:
        from .steady import load_device
    else:
        from steady import load_device

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True, help="private local RunStore directory")
    parser.add_argument("--work", type=Path, required=True, help="private local supervisor work directory")
    parser.add_argument("--build-source-commit", required=True, help="full committed core revision")
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args()
    if not 1 <= arguments.port <= 65535:
        parser.error("port must be between 1 and 65535")
    implementation = committed_implementation(arguments.build_source_commit)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
    context = SteadyPreparationContext((("research_1d", load_device()),), implementation, head)
    ceiling = ResourceLimits(60.0, 1536 * 1024**2, 96 * 1024**2, None, None)
    catalog = steady_execution_catalog(context, ceiling=ceiling)
    store_path, work = arguments.store.expanduser().resolve(), arguments.work.expanduser().resolve()
    store_path.parent.mkdir(parents=True, exist_ok=True)
    work.parent.mkdir(parents=True, exist_ok=True)
    with RunStore(store_path) as store, ProcessSupervisor(store, work,
            (catalog.worker_binding("solarlab"),), max_active=1, max_pending=2,
            shared_rss_bytes=3 * 1024**3, shared_output_bytes=384 * 1024**2) as supervisor:
        app = create_app(store.root, job_service=JobService(catalog, supervisor))
        print(json.dumps({"base_url": f"http://127.0.0.1:{arguments.port}",
            "device_id": "research_1d", "build_source_commit": implementation.build_source_commit,
            "current_repository_head": head, "execution_catalog_sha256": catalog.document.sha256,
            "job_schema_sha256": catalog.schema.sha256, "scientific_qualification": False}), flush=True)
        uvicorn.run(app, host="127.0.0.1", port=arguments.port, workers=1, loop="asyncio", http="h11")


if __name__ == "__main__":
    main()
