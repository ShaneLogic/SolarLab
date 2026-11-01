"""Opposed contexts run real isolated old material preparation, without a solve.

This checks the import/environment boundary and every material word. It does
not stand in for the remaining full driver/history comparisons in P03-04.
"""

from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys

from test_behavior_context import ROOT, context, edit
from test_behavior_context import defaults as defaults, resources as resources
from test_behavior_context import scaps as scaps, sources as sources


_OBSERVER = r'''
from dataclasses import fields, is_dataclass, replace
import hashlib
import json
from pathlib import Path
import os
import resource
import struct
import sys
import time

started = time.monotonic_ns()
request = json.loads(sys.argv[2])
assert all(os.environ.get(key) == value for key, value in request.items())
from perovskite_sim.solver import mol
from perovskite_sim.experiments import steady_state
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.scaps_compat.loader import load_scaps_yaml
import numpy as np

def words(value):
    if is_dataclass(value):
        return {"type": type(value).__qualname__,
                "fields": {f.name: words(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": list(value.shape),
                "raw_hex": value.tobytes(order="C").hex()}
    if isinstance(value, np.generic):
        return {"dtype": value.dtype.str, "raw_hex": value.tobytes().hex()}
    if type(value) is float:
        return {"binary64_be": struct.pack(">d", value).hex()}
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, (tuple, list)):
        return [words(item) for item in value]
    if isinstance(value, dict):
        return {str(key): words(item) for key, item in value.items()}
    raise TypeError(type(value).__qualname__)

source = Path(sys.argv[1])
stack = replace(load_scaps_yaml(source), dos_band_potentials=False)
grid = build_electrical_grid(stack, 8)
material = mol.build_material_arrays(grid, stack)
payload = words({"stack": stack, "grid": grid, "material": material})
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform != "darwin":
    rss *= 1024
print(json.dumps({
    "payload": payload,
    "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "environment": request,
    "imports": {"ss_jacobian_reuse": steady_state._SS_JAC_REUSE,
                "ss_gummel": steady_state._SS_GUMMEL,
                "rhs_finite_check": mol._RHS_FINITE_CHECK,
                "QSS_velocity_effective_m_s": mol._QSS_V_TH_MS.hex()},
    "origins": {name: {"path": str(Path(module.__file__).resolve()),
                       "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
                for name, module in (("solver/mol.py", mol), ("experiments/steady_state.py", steady_state))},
    "pid": os.getpid(), "started_ns": started, "finished_ns": time.monotonic_ns(),
    "rss_bytes": rss,
}, allow_nan=False))
'''


def test_opposed_contexts_preserve_all_words_in_concurrent_old_material_calls(scaps, sources):
    parent_environment = dict(os.environ)
    package = find_spec("perovskite_sim")
    assert package is not None and len(package.submodule_search_locations) == 1
    package_root = Path(next(iter(package.submodule_search_locations))).resolve()
    device = edit(scaps, lambda data: data["settings"].update(dos_band_potentials=False))
    supplied = [
        {"SOLARLAB_DOS_BAND": value, "SOLARLAB_SS_JAC_REUSE": value,
         "SOLARLAB_SS_GUMMEL": value, "PEROVSKITE_RHS_FINITE_CHECK": value,
         "SOLARLAB_QSS_VTH": "-0" if value == "0" else "1"}
        for value in ("0", "1")
    ]
    contexts = [context(device, sources, environment=tuple(env.items())).export() for env in supplied]
    keys = tuple(contexts[0]["environment_effective_snapshot"])

    def start(snapshot):
        child_environment = dict(parent_environment)
        for key in keys:
            child_environment.pop(key, None)
        child_environment.update({key: value for key, value in snapshot.items() if value is not None})
        child_environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                 VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        return subprocess.Popen(
            [sys.executable, "-I", "-B", "-c", _OBSERVER,
             str(ROOT / "perovskite-sim/configs/scaps_mirror_v2.yaml"), json.dumps(snapshot)],
            env=child_environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    def finish(process):
        try:
            stdout, stderr = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
        assert process.returncode == 0, stdout + stderr
        result = json.loads(stdout)
        assert result["rss_bytes"] <= 4 * 1024**3
        return {"returncode": process.returncode, "stderr": stderr, "result": result}

    direct = [finish(start({key: env.get(key) for key in keys})) for env in supplied]
    children = [start(item["environment_effective_snapshot"]) for item in contexts]
    try:
        concurrent = [finish(child) for child in children]
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.communicate()
    assert len({row["result"]["pid"] for row in concurrent}) == 2
    assert max(row["result"]["started_ns"] for row in concurrent) < min(row["result"]["finished_ns"] for row in concurrent)
    assert direct[0]["result"]["imports"] != direct[1]["result"]["imports"]
    assert direct[0]["result"]["payload"] != direct[1]["result"]["payload"]
    for original, actual, expected in zip(direct, concurrent, contexts):
        original, actual = original["result"], actual["result"]
        assert actual["payload"] == original["payload"]
        assert actual["input_sha256"] == original["input_sha256"]
        prediction = expected["source_inspection"]["import_time_policy"]
        assert actual["imports"] == {key: value.hex() if type(value) is float else value
                                     for key, value in prediction.items() if key in actual["imports"]}
        for name, origin in actual["origins"].items():
            assert Path(origin["path"]) == package_root / name
            assert origin["sha256"] == next(source.sha256 for source in sources if source.id == name)
    assert dict(os.environ) == parent_environment
    output = os.environ.get("SOLARLAB_BEHAVIOR_EXECUTION_OUTPUT")
    if output:
        Path(output).write_text(json.dumps({
            "scope": "real concurrent old material preparation, no state solve or full driver history",
            "passed": True, "parent_environment_unchanged": True,
            "contexts": contexts, "direct": direct, "concurrent": concurrent,
        }, indent=2) + "\n")
