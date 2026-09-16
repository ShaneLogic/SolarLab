#!/usr/bin/env python3
"""Launch R1 from a caller-anchored source snapshot using Python -I -S.

This launcher, interpreter, standard library, Git and explicit dependency
directories are trusted inputs. No general sandbox or independently approved
commit is claimed. Run this trusted script directly, not through an ambient
Python process that has already executed sitecustomize or arbitrary hooks.
Project modules are compiled from the exact frozen bytes archived by the CLI.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import types


CHECKOUT_NAME = "perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout"
CHECKOUT_PATH = "perovskite_sim/experiments/one_dimensional_mechanism_r1_checkout.py"
RUNNER_PATH = "scripts/run_one_dimensional_mechanism_r1_stage_one.py"
LAUNCHER_PATH = "scripts/run_one_dimensional_mechanism_r1_controlled.py"
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _bootstrap_git(project, *arguments):
    executable = shutil.which("git", path=os.defpath)
    if executable is None:
        raise ValueError("controlled R1 startup requires trusted system Git")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                        "GIT_CONFIG_SYSTEM": os.devnull, "GIT_NO_REPLACE_OBJECTS": "1",
                        "GIT_OPTIONAL_LOCKS": "0"})
    result = subprocess.run(
        [executable, "-c", "core.fsmonitor=false", "-c", "core.excludesFile=" + os.devnull,
         "-c", "core.hooksPath=" + os.devnull, "-C", str(project), *arguments],
        env=environment, capture_output=True, check=False,
    )
    if result.returncode:
        raise ValueError("controlled R1 startup cannot read the caller's source commit")
    return result.stdout


def _bootstrap_checkout(project, source_commit):
    if (len(source_commit) not in (40, 64)
            or any(c not in "0123456789abcdef" for c in source_commit)):
        raise ValueError("--source-commit must be a full lowercase Git commit object id")
    root = Path(_bootstrap_git(project, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if project != root / "perovskite-sim":
        raise ValueError("--project must be the perovskite-sim directory of its checkout")
    head = _bootstrap_git(project, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    if head != source_commit:
        raise ValueError("checkout HEAD differs from the caller source commit")
    for name, actual in ((LAUNCHER_PATH, Path(__file__)), (CHECKOUT_PATH, project / CHECKOUT_PATH)):
        expected = _bootstrap_git(project, "cat-file", "blob", source_commit + ":perovskite-sim/" + name)
        if actual.is_symlink() or actual.read_bytes() != expected:
            raise ValueError("trusted startup source differs from caller commit blob: " + name)
        if name == CHECKOUT_PATH:
            checkout_bytes = expected
    module = types.ModuleType(CHECKOUT_NAME)
    module.__file__ = str(project / CHECKOUT_PATH)
    module.__package__ = CHECKOUT_NAME.rpartition(".")[0]
    sys.modules[CHECKOUT_NAME] = module
    exec(compile(checkout_bytes, module.__file__, "exec", dont_inherit=True), module.__dict__)
    return module


class FrozenSourceLoader(importlib.abc.Loader):
    """A source-only loader: no bytecode cache, filesystem re-read or fallback."""

    def __init__(self, context, name, relative, package):
        self.context, self.name, self.relative, self.package = context, name, relative, package

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        filename = str(self.context.project / self.relative)
        module.__file__ = filename
        module.__cached__ = None
        if self.package:
            module.__path__ = [str(Path(filename).parent)]
        exec(compile(self.context.read_bytes(self.relative), filename, "exec", dont_inherit=True), module.__dict__)

    def get_filename(self, fullname):
        return str(self.context.project / self.relative)

    def get_source(self, fullname):
        return importlib.util.decode_source(self.context.read_bytes(self.relative))


class FrozenSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, context):
        self.context = context

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "perovskite_sim" or fullname.startswith("perovskite_sim."):
            stem = fullname.replace(".", "/")
            candidates = ((stem + "/__init__.py", True), (stem + ".py", False))
        elif fullname in ("run_one_dimensional_mechanism_r1", "run_one_dimensional_mechanism_r0"):
            candidates = (("scripts/" + fullname + ".py", False),)
        else:
            return None
        for relative, package in candidates:
            try:
                self.context.read_bytes(relative)
            except ValueError:
                continue
            loader = FrozenSourceLoader(self.context, fullname, relative, package)
            spec = importlib.util.spec_from_loader(fullname, loader, is_package=package)
            spec.origin = str(self.context.project / relative)
            return spec
        raise ModuleNotFoundError("project module is absent from the frozen source snapshot: " + fullname)


def _write_startup_failure(arguments, source_commit, exc, started, started_utc):
    """Seal a rejected request without asserting source or execution identity."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("stage", nargs="?")
    parser.add_argument("--output-dir", type=Path)
    try:
        request, _ = parser.parse_known_args(arguments)
    except SystemExit:
        return
    if request.stage not in ("prepare", "zero-check", "step") or request.output_dir is None:
        return
    output = request.output_dir.resolve()
    # A refusal must never overwrite an earlier run, even a partially written one.
    if output.exists():
        return
    failure = {"type": type(exc).__name__, "message": str(exc), "phase": "controlled_startup"}
    completion = {
        "schema": "R1StageOneCompletionV1", "evidence_revision": 5,
        "stage_scope": "R1-1", "stage": request.stage, "status": "failed",
        "run_class": "rejected_before_execution", "failure": failure,
        "requested_source_commit": source_commit,
        "source_identity_verified": False, "physical_execution_started": False,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "duration_s": time.monotonic() - started,
        "observed_record_count": 0, "persisted_record_count": 0,
        "accepted_record_count": 0, "persisted_finite_step_count": 0,
        "physical_passed_finite_step_count": 0,
    }
    def write_json(path, value):
        raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output, prefix=".r1-rejected-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(raw)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    try:
        output.mkdir(parents=True, exist_ok=False)
        entries = {
            "FailureV1.json": write_json(output / "FailureV1.json", failure),
            "CompletionV1.json": write_json(output / "CompletionV1.json", completion),
        }
        write_json(output / "ManifestV1.json", entries)
    except OSError as persistence_error:
        print("startup rejection could not be fully persisted: " + str(persistence_error), file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-sha256")
    parser.add_argument("--dependency-path", action="append", default=[], type=Path,
                        help="trusted dependency directory; .pth files are not processed")
    parser.add_argument("runner_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.runner_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    if not (sys.flags.isolated and sys.flags.no_site):
        parser.error("start the trusted launcher directly with Python -I -S")
    if any(name == "perovskite_sim" or name.startswith("perovskite_sim.") for name in sys.modules):
        parser.error("project modules were loaded before controlled startup")
    expected_finders = (importlib.machinery.BuiltinImporter, importlib.machinery.FrozenImporter,
                        importlib.machinery.PathFinder)
    if tuple(sys.meta_path) != expected_finders:
        parser.error("unexpected import hooks before controlled startup")
    execution_started = False
    try:
        project = args.project.resolve(strict=True)
        dependency_paths = tuple(str(path.resolve(strict=True)) for path in args.dependency_path)
        if any(not Path(path).is_dir() for path in dependency_paths):
            raise ValueError("dependency paths must be explicitly trusted directories")
        checkout = _bootstrap_checkout(project, args.source_commit)
        context = checkout.require_r1_checkout(project=project, formal=True,
                                               source_commit=args.source_commit,
                                               expected_source_sha256=args.source_sha256)
        for key in THREAD_VARIABLES:
            os.environ[key] = "1"
        # -I -S leaves only interpreter-controlled startup paths. Append explicit
        # dependencies as directories, without site.addsitedir or .pth handling.
        sys.path.extend(path for path in dependency_paths if path not in sys.path)
        sys.dont_write_bytecode = True
        finder = FrozenSourceFinder(context)
        sys.meta_path.insert(0, finder)
        checkout.__spec__ = finder.find_spec(CHECKOUT_NAME)
        checkout.__loader__ = checkout.__spec__.loader
        runtime = {
            "launcher": str(Path(__file__).resolve()), "python_executable": sys.executable,
            "python_version": sys.version, "isolated": True, "no_site": True,
            "dependency_paths": list(dependency_paths), "pythonpath_ignored": True,
            "project_loader": "FrozenSourceLoader", "project_bytecode_cache_used": False,
            "trust_assumptions": "trusted launcher, interpreter, standard library, Git and explicit dependencies; no arbitrary in-process tampering",
            "source_anchor_scope": "caller supplied commit/content identity; independent approval not asserted",
        }
        context = checkout._install_controlled_context(context, runtime)
        finder.context = context
        # Validate the pinned complete input before running any stage, including
        # when the candidate was committed in a different repository.
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_binding import (
            study_input_identity, execution_contract_identity, operator_criterion_identity,
            additional_failures_identity,
        )
        study_input_identity()
        execution_contract_identity()
        operator_criterion_identity()
        additional_failures_identity()
        filename = str(project / RUNNER_PATH)
        module = types.ModuleType("__main__")
        module.__file__ = filename
        module.__package__ = ""
        module.__loader__ = FrozenSourceLoader(context, "__main__", RUNNER_PATH, False)
        sys.modules["__main__"] = module
        sys.argv = [filename, *arguments]
        execution_started = True
        exec(compile(context.read_bytes(RUNNER_PATH), filename, "exec", dont_inherit=True), module.__dict__)
        return 0
    except (OSError, ValueError, ImportError) as exc:
        phase = "runner" if execution_started else "startup"
        print("controlled R1 " + phase + " failed: " + str(exc), file=sys.stderr)
        if not execution_started:
            _write_startup_failure(arguments, args.source_commit, exc, started, started_utc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
