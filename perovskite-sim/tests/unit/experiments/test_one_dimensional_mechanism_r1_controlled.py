"""Real isolated subprocess counterexamples for bounded R1 startup."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import zipfile

import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_checkout as checkout


PROJECT = Path(__file__).resolve().parents[3]
LAUNCHER = "scripts/run_one_dimensional_mechanism_r1_controlled.py"
BINDING = "perovskite_sim/experiments/one_dimensional_mechanism_r1_binding.py"
PROBE = "perovskite_sim/probe.py"
INPUT = checkout.STUDY_INPUT_RELATIVE_PATH
REFERENCE = "0a6532a436dd07e6b27f01da3fc39aef906e6fd882057f0109c1bc5bdf77e39b"
ENV = {**os.environ, **{key: "1" for key in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}}


def git(root, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], env=checkout.git_environment(), stderr=subprocess.PIPE,
    ).decode().strip()


def commit(root):
    git(root, "add", "-A")
    git(root, "-c", "user.name=R1 Test", "-c", "user.email=r1-test@example.invalid",
        "commit", "-q", "-m", "committed test fixture")
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path):
    """Tiny committed project; exercises startup and binding without a solve."""
    root = tmp_path / "repository"
    project = root / "perovskite-sim"
    project.mkdir(parents=True)
    for name in checkout.REQUIRED_SOURCE_ANCHORS:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in (checkout._CHECKOUT_MODULE, LAUNCHER, INPUT,
                    "docs/OneDimensionalMechanismR1DynamicsV1.md"):
            path.write_bytes((PROJECT / name).read_bytes())
        else:
            path.write_text("# test fixture\n")
    (project / "perovskite_sim/__init__.py").write_text("")
    (project / "perovskite_sim/experiments/__init__.py").write_text("")
    (project / BINDING).write_bytes((PROJECT / BINDING).read_bytes())
    evidence = "perovskite_sim/experiments/one_dimensional_mechanism_r1_evidence.py"
    (project / evidence).write_bytes((PROJECT / evidence).read_bytes())
    (project / "perovskite_sim/experiments/one_dimensional_mechanism_r1.py").write_text(
        "def validate_binding(binding, stack):\n    pass\n")
    (project / PROBE).write_text("VALUE = 'trusted source'\n")
    (project / "scripts/run_one_dimensional_mechanism_r1_stage_one.py").write_text('''
import argparse
import json
from pathlib import Path
from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import current_execution_context, record_frozen_source
from perovskite_sim.experiments.one_dimensional_mechanism_r1_binding import validate_r1_study_binding, study_input_identity
parser = argparse.ArgumentParser()
parser.add_argument('--reference-hash', required=True)
parser.add_argument('--output', type=Path)
parser.add_argument('--mutate-after-freeze', action='store_true')
args = parser.parse_args()
context = current_execution_context()
assert context.run_class == 'formal'
validate_r1_study_binding({'sha256': args.reference_hash}, None)
if args.mutate_after_freeze:
    (context.project / 'perovskite_sim/probe.py').write_text("VALUE = 'changed on disk'\\n")
    context.study_input.write_text('{}')
from perovskite_sim.probe import VALUE
if args.output:
    args.output.mkdir()
    record_frozen_source(args.output, context)
print(json.dumps({'value': VALUE, 'context': context.to_dict(), 'input': study_input_identity()}))
''')
    (root / ".gitignore").write_text("__pycache__/\noutputs/\n")
    git(root, "init", "-q")
    return root, project, commit(root)


def launch(repository, *arguments, environment=None, source_sha256=None, flags=("-I", "-S"), deps=()):
    root, project, revision = repository
    command = [sys.executable, *flags, str(project / LAUNCHER), "--project", str(project),
               "--source-commit", revision]
    if source_sha256 is not None:
        command.extend(["--source-sha256", source_sha256])
    for dependency in deps:
        command.extend(["--dependency-path", str(dependency)])
    command.extend(["--", "--reference-hash", REFERENCE, *arguments])
    return subprocess.run(command, env=environment or ENV, cwd=root,
                          capture_output=True, text=True, timeout=30)


def assert_passed(result):
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["value"] == "trusted source"
    assert data["context"]["run_class"] == "formal"
    assert data["context"]["runtime"]["project_bytecode_cache_used"] is False
    return data


def test_controlled_source_and_optional_caller_content_anchor(repository):
    data = assert_passed(launch(repository))
    digest = data["context"]["source_content_sha256"]
    assert digest == checkout.source_content_digest(data["context"]["required_sources"])
    assert_passed(launch(repository, source_sha256=digest))
    failure = launch(repository, source_sha256="0" * 64)
    assert failure.returncode == 1
    assert "caller SHA-256 anchor" in failure.stderr


def test_formal_snapshot_does_not_misreport_unrelated_worktree_changes(repository):
    root, _, _ = repository
    (root / "unrelated_notes.txt").write_text("untracked local notes\n")
    data = assert_passed(launch(repository))
    context = data["context"]
    assert context["dirty"]["untracked"] is True
    assert context["required_source_snapshot_matches_commit"] is True
    assert context["archived_source_is_commit_snapshot"] is True
    assert context["status_scope"] == "advisory working-tree observation at capture"


def test_real_runner_and_helpers_cannot_reopen_project_dependency_path(repository):
    root, project, _ = repository
    for relative in ("scripts/run_one_dimensional_mechanism_r1.py",
                     "scripts/run_one_dimensional_mechanism_r0.py"):
        (project / relative).write_bytes((PROJECT / relative).read_bytes())
    runner_code = (PROJECT / "scripts/run_one_dimensional_mechanism_r1_stage_one.py").read_text()
    wrapper = (
        "import sys\n"
        "from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import current_execution_context\n"
        "context = current_execution_context()\n"
        "namespace = {'__file__': __file__, '__name__': 'tested_production_runner'}\n"
        "exec(compile(" + repr(runner_code) + ", __file__, 'exec'), namespace)\n"
        "assert str(context.project) not in sys.path, sys.path\n"
        "try:\n    import ambient_dependency_probe\n"
        "except ModuleNotFoundError:\n    print('project dependency path stayed closed')\n"
        "else:\n    raise AssertionError('untracked dependency was loaded')\n"
    )
    (project / "scripts/run_one_dimensional_mechanism_r1_stage_one.py").write_text(wrapper)
    revision = commit(root)
    (project / "ambient_dependency_probe.py").write_text("VALUE = 'untracked code'\n")
    result = launch((root, project, revision))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "project dependency path stayed closed" in result.stdout


@pytest.mark.parametrize("index_flag", ["--skip-worktree", "--assume-unchanged"])
@pytest.mark.parametrize("relative", [INPUT, PROBE])
def test_hidden_index_changes_cannot_claim_committed_source(repository, index_flag, relative):
    root, project, _ = repository
    git(root, "update-index", index_flag, "perovskite-sim/" + relative)
    (project / relative).write_bytes((project / relative).read_bytes() + b"\n")
    assert git(root, "status", "--porcelain") == ""
    failure = launch(repository)
    assert failure.returncode == 1
    assert "differs from caller commit blob" in failure.stderr
    assert relative in failure.stderr


def test_committed_repin_in_another_repository_still_fails_policy_digest(repository):
    root, project, _ = repository
    study = json.loads((project / INPUT).read_text())
    study["fixed_reference_binding_sha256"] = "0" * 64
    (project / INPUT).write_text(json.dumps(study))
    forged_commit = commit(root)
    failure = launch((root, project, forged_commit), "--reference-hash", "0" * 64)
    assert failure.returncode == 1
    assert "pinned complete input digest" in failure.stderr


def test_clone_under_ignored_outputs_does_not_bypass_content_binding(repository):
    root, project, revision = repository
    clone = project / "outputs/shadow"
    clone.parent.mkdir()
    git(root, "clone", "-q", "--local", "--no-hardlinks", str(root), str(clone))
    clone_project = clone / "perovskite-sim"
    assert_passed(launch((clone, clone_project, revision)))
    git(clone, "update-index", "--skip-worktree", "perovskite-sim/" + INPUT)
    (clone_project / INPUT).write_text("{}")
    assert git(clone, "status", "--porcelain") == ""
    failure = launch((clone, clone_project, revision))
    assert failure.returncode == 1
    assert "differs from caller commit blob" in failure.stderr


def test_sitecustomize_and_pth_are_not_run(repository, tmp_path):
    _, project, _ = repository
    hook = tmp_path / "hook"
    hook.mkdir()
    marker = tmp_path / "hook_ran"
    body = "from pathlib import Path; Path(" + repr(str(marker)) + ").write_text('ran')\n"
    body += '''
import sys, importlib.util, importlib.machinery
class PatchedLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        raw = Path(self.path).read_text()
        raw = raw.replace('raise ValueError("R1-1 reference binding is not the approved study reference")', 'pass')
        return compile(raw, self.path, 'exec')
class Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'perovskite_sim.experiments.one_dimensional_mechanism_r1_binding':
            filename = REAL_BINDING
            return importlib.util.spec_from_file_location(fullname, filename, loader=PatchedLoader(fullname, filename))
sys.meta_path.insert(0, Finder())
'''.replace("REAL_BINDING", repr(str(project / BINDING)))
    (hook / "sitecustomize.py").write_text(body)
    (hook / "injected.pth").write_text("import pathlib; pathlib.Path(" + repr(str(marker)) + ").write_text('ran')\n")
    environment = {**ENV, "PYTHONPATH": str(hook), "PYTHONSTARTUP": str(hook / "sitecustomize.py")}
    # Positive attack control: the hook really bypasses the unisolated binding.
    control = subprocess.run(
        [sys.executable, "-B", "-c", "import sys; sys.path.insert(0, " + repr(str(project)) + "); "
         "from perovskite_sim.experiments.one_dimensional_mechanism_r1_binding import validate_r1_study_binding; "
         "validate_r1_study_binding({'sha256': '0'*64}, None)"],
        cwd=project, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert control.returncode == 0, control.stderr
    assert marker.exists()
    marker.unlink()
    assert_passed(launch(repository, environment=environment, deps=(hook,)))
    assert not marker.exists()
    failure = launch(repository, "--reference-hash", "0" * 64, environment=environment, deps=(hook,))
    assert failure.returncode != 0
    assert "not the approved study reference" in failure.stderr
    assert not marker.exists()


@pytest.mark.parametrize("kind", ["unchecked", "timestamp", "genuine_source_hash"])
def test_project_pyc_never_supplies_executed_code(repository, tmp_path, kind):
    _, project, _ = repository
    actual = project / PROBE
    fake = tmp_path / "fake.py"
    fake.write_text("VALUE = 'forged bytecode'\n")
    cache = Path(importlib.util.cache_from_source(str(actual)))
    cache.parent.mkdir()
    mode = (py_compile.PycInvalidationMode.TIMESTAMP if kind == "timestamp"
            else py_compile.PycInvalidationMode.UNCHECKED_HASH)
    py_compile.compile(str(fake), cfile=str(cache), dfile=str(actual), invalidation_mode=mode, doraise=True)
    raw = cache.read_bytes()
    if kind == "genuine_source_hash":
        raw = raw[:8] + importlib.util.source_hash(actual.read_bytes()) + raw[16:]
    elif kind == "timestamp":
        import struct
        raw = raw[:8] + struct.pack("<II", int(actual.stat().st_mtime), actual.stat().st_size) + raw[16:]
    cache.write_bytes(raw)
    # Prove each poisoned cache is executable by the normal source loader.
    control = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", "import sys; sys.path.insert(0, " + repr(str(project)) + "); "
         "from perovskite_sim.probe import VALUE; print(VALUE)"],
        env=ENV, capture_output=True, text=True, timeout=30,
    )
    assert control.returncode == 0, control.stderr
    assert control.stdout.strip() == "forged bytecode"
    assert_passed(launch(repository))


def test_frozen_execution_and_evidence_survive_later_disk_changes(repository, tmp_path):
    _, project, _ = repository
    original_probe = (project / PROBE).read_bytes()
    original_input = (project / INPUT).read_bytes()
    output = tmp_path / "frozen_evidence"
    data = assert_passed(launch(repository, "--mutate-after-freeze", "--output", str(output)))
    assert (project / PROBE).read_bytes() != original_probe
    assert (project / INPUT).read_bytes() != original_input
    assert data["input"]["sha256"] == hashlib.sha256(original_input).hexdigest()
    with zipfile.ZipFile(output / "SourceV1.zip") as archive:
        assert archive.read("perovskite-sim/" + PROBE) == original_probe
        assert archive.read("perovskite-sim/" + INPUT) == original_input
    assert (output / "SourceChangesV1.patch").read_bytes() == b""


def test_normal_python_does_not_acquire_controlled_identity(repository):
    failure = launch(repository, flags=())
    assert failure.returncode != 0
    assert "Python -I -S" in failure.stderr


def test_no_environment_variable_can_install_the_runtime():
    result = subprocess.run(
        [sys.executable, "-c", "from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import current_execution_context; assert current_execution_context() is None"],
        cwd=PROJECT, env={**ENV, "PYTHONPATH": str(PROJECT), "R1_CONTROLLED": "1", "R1_SOURCE_COMMIT": "f" * 40},
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_git_environment_drops_config_and_replacement_overrides(monkeypatch):
    for name in ("GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_REPLACE_REF_BASE", "GIT_EXEC_PATH",
                 "GIT_CONFIG_PARAMETERS", "GIT_INDEX_FILE", "GIT_TRACE", "GIT_CONFIG_KEY_0"):
        monkeypatch.setenv(name, "hostile")
    environment = checkout.git_environment()
    assert environment["GIT_CONFIG_GLOBAL"] == environment["GIT_CONFIG_SYSTEM"] == os.devnull
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert "hostile" not in [value for key, value in environment.items() if key.startswith("GIT_")]


@pytest.mark.parametrize("existing", [False, True])
def test_startup_rejection_seals_failure_without_overwriting_prior_run(repository, tmp_path, existing):
    root, project, revision = repository
    git(root, "update-index", "--skip-worktree", "perovskite-sim/" + INPUT)
    (project / INPUT).write_text("{}")
    output = tmp_path / "rejected"
    if existing:
        output.mkdir()
        (output / "prior.json").write_text('{"keep": true}')
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(project / LAUNCHER), "--project", str(project),
         "--source-commit", revision, "--", "prepare", "--output-dir", str(output)],
        env=ENV, cwd=root, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1
    if existing:
        assert {path.name for path in output.iterdir()} == {"prior.json"}
        assert (output / "prior.json").read_text() == '{"keep": true}'
    else:
        completion = json.loads((output / "CompletionV1.json").read_text())
        assert completion["status"] == "failed"
        assert completion["run_class"] == "rejected_before_execution"
        assert completion["evidence_revision"] == 4
        assert completion["physical_execution_started"] is False
        assert completion["source_identity_verified"] is False
        assert not (output / "SourceManifestV1.json").exists()
        entries = json.loads((output / "ManifestV1.json").read_text())
        assert set(entries) == {"FailureV1.json", "CompletionV1.json"}
        for name, entry in entries.items():
            raw = (output / name).read_bytes()
            assert entry == {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
