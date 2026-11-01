"""Real bounded old-driver inputs; no mock executor or numerical replacement."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

from solarlab.config.behavior import behavior_source_requirements, inspect_historical_behavior
from solarlab.config.device_import import import_standard_device
from solarlab.config.identity import BuildSource, ExecutionEngine, execution_manifest
from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.legacy_execution import LegacyExecutionContext
from solarlab.config.resolve_device import resolve_device
from solarlab.config.scaps_input import import_scaps_device
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab_server.legacy_engine import legacy_description, legacy_environment, prepare_legacy_job

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "perovskite-sim/perovskite_sim"
sys.path.insert(0, str(ROOT / "tests/refactor"))
from test_execution_identity_preparation import identity_inputs  # noqa: E402
from test_behavior_context import selection  # noqa: E402

_CATALOG = ("models/parameters.py", "models/device.py", "models/defects.py", "physics/statistics.py",
            "physics/band_gap_narrowing.py", "scaps_compat/loader.py", "constants.py",
            "twod/microstructure.py", "models/tandem_config.py")


def make_job(driver="scaps_transient", *, arguments=None, environment=(), runtime_change=None, changed_resource=False):
    sources = tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name, _ in behavior_source_requirements())
    default = read_legacy_default_catalog(tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name in _CATALOG))
    tables = ResourceLibrary(tuple(ResourceTable(path.stem, "nk", SourceDocument(path.name, path.read_bytes()))
                                   for path in sorted((LEGACY / "data/nk").glob("*.csv"))))
    if changed_resource:
        first = tables.tables[0]
        tables = ResourceLibrary((replace(first, source=replace(first.source, content=first.source.content + b"\n# changed resource\n")),
                                  *tables.tables[1:]))
    if driver == "ion_hysteresis":
        raw = json.loads((ROOT / "perovskite-sim/tests/fixtures/IonDiffusivityFigureReference.json").read_bytes())["request"]
        original = SourceDocument("ion-original.json", json.dumps(raw["device"]).encode())
        supplied = {**raw["params"], "N_grid": 8, "V_max": .02, "v_rate": 1000., "n_points": 3,
            "save_snapshots": True, "decompose_currents": True,
            "waveform": {**raw["params"]["waveform"], "branch_dwell_s": 1e-6, "dark_prep_s": 1e-6,
                         "dark_seed_s": 1e-6, "turnaround_s": 1e-6}}
        editable = import_standard_device(original, id="legacy_job", defaults=default)
    else:
        original = SourceDocument("scaps-original.yaml", (ROOT / "perovskite-sim/configs/scaps_mirror_v2.yaml").read_bytes())
        supplied = ({"N_grid": 30, "V_hi": 1.3, "tol_v": .02, "iface_states": False}
                    if driver == "scaps_steady_state" else
                    {"N_grid": 8, "n_points": 3, "v_rate": 1000., "V_max": .02,
                     "save_snapshots": True, "decompose_currents": True})
        editable = import_scaps_device(original, id="legacy_job", defaults=default)
    supplied.update(arguments or {})
    device = resolve_device(editable, default, tables, sources=(original,))
    problem = "dc" if driver == "scaps_steady_state" else "transient"
    behavior = inspect_historical_behavior(device, selection(device, problem), driver=driver, sources=sources,
        environment=environment, arguments=SourceDocument("original-arguments.json", json.dumps(supplied).encode()))
    inputs = identity_inputs(device, problem=problem, effective_behavior=behavior)
    invocation = LegacyExecutionContext(behavior, original)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    # All old Python source bytes, with their real unchanged build association.
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "perovskite-sim/perovskite_sim"],
                   cwd=ROOT, check=True, capture_output=True)
    build = BuildSource(commit, tuple(SourceDocument("perovskite_sim/" + str(path.relative_to(LEGACY)), path.read_bytes())
                                     for path in sorted(LEGACY.rglob("*.py")) if path.stat().st_size))
    runtime = legacy_environment(invocation)
    if runtime_change:
        runtime = replace(runtime, **runtime_change)
    description = legacy_description(invocation, implementation=build, environment=runtime, current_repository_head=commit)
    engine = ExecutionEngine("legacy", behavior)
    manifest = execution_manifest(inputs, description, engine=engine)
    job = prepare_legacy_job(invocation, inputs, description, engine=engine, manifest=manifest, expected_sha256=manifest.sha256)
    return job, invocation
