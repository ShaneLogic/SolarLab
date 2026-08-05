"""Machine-verifiable configuration and benchmark reproducibility contracts."""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

import yaml

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.models.tandem_config import load_tandem_from_yaml
from perovskite_sim.scaps_compat.loader import load_scaps_yaml


class ReproducibilityError(RuntimeError):
    """A frozen source, config schema, or benchmark contract drifted."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ReproducibilityError(f"{path}: expected a YAML mapping")
    return value


def _canonical(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        # Walk fields ourselves instead of using asdict(), which recursively
        # erases the DeviceStack type before a tandem's nested cells reach this
        # function and would defeat the optional-field compatibility rule.
        mapping = {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
        }
        # Optional fields added after the P0 freeze must not churn every
        # historical semantic hash when they are jointly absent and therefore
        # behaviorally inert. Populated grid tuples remain part of the hash.
        if isinstance(value, DeviceStack) and not (
            value.grid_interval_weights or value.grid_alphas
        ):
            mapping.pop("grid_interval_weights", None)
            mapping.pop("grid_alphas", None)
        return _canonical(mapping)
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReproducibilityError("semantic state contains a non-finite float")
        return float(format(value, ".15g"))
    return value


def semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def detect_config_schema(raw: dict[str, Any]) -> str:
    if raw.get("device_type") == "tandem_2T_monolithic":
        return "tandem-v1"
    layers = raw.get("layers") or []
    if any(
        isinstance(layer, dict)
        and any(key in layer for key in ("mu_n_cm2", "thickness_nm", "N_C_cm3"))
        for layer in layers
    ):
        return "scaps-device-v1"
    return "standard-device-v1"


def _check_required(
    mapping: dict[str, Any], required: list[str], where: str, errors: list[str],
) -> None:
    missing = sorted(key for key in required if key not in mapping)
    if missing:
        errors.append(f"{where}: missing required keys {missing}")


def _finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _mapping_value_at_path(mapping: dict[str, Any], dotted_path: str) -> Any:
    """Resolve a dotted YAML mapping path without evaluating expressions."""
    current: Any = mapping
    for component in dotted_path.split("."):
        if not isinstance(current, dict) or component not in current:
            raise KeyError(dotted_path)
        current = current[component]
    return current


def _validate_stack(stack: DeviceStack, where: str, errors: list[str]) -> None:
    if not stack.layers:
        errors.append(f"{where}: no layers")
        return
    if not any(layer.role == "absorber" for layer in stack.layers):
        errors.append(f"{where}: no absorber layer")
    if stack.interfaces and len(stack.interfaces) != len(stack.layers) - 1:
        errors.append(
            f"{where}: {len(stack.interfaces)} interfaces for {len(stack.layers)} layers"
        )
    if stack.interface_defects and len(stack.interface_defects) != len(stack.layers) - 1:
        errors.append(
            f"{where}: {len(stack.interface_defects)} defect slots for "
            f"{len(stack.layers)} layers"
        )
    for index, layer in enumerate(stack.layers):
        prefix = f"{where}: layer[{index}] {layer.name!r}"
        if not math.isfinite(layer.thickness) or layer.thickness <= 0.0:
            errors.append(f"{prefix}: thickness must be finite and positive")
        if layer.params is None:
            errors.append(f"{prefix}: missing material parameters")
            continue
        values = dataclasses.asdict(layer.params)
        for key, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    errors.append(f"{prefix}.{key}: non-finite")
        if layer.params.eps_r <= 0.0:
            errors.append(f"{prefix}.eps_r: must be positive")
        if layer.params.mu_n < 0.0 or layer.params.mu_p < 0.0:
            errors.append(f"{prefix}: mobility must be non-negative")
        if layer.params.D_ion < 0.0 or layer.params.D_ion_neg < 0.0:
            errors.append(f"{prefix}: ion diffusivity must be non-negative")
        if layer.params.P_lim <= 0.0 or layer.params.P0 < 0.0:
            errors.append(f"{prefix}: ion densities outside their domain")
        if layer.params.P0 > layer.params.P_lim:
            errors.append(f"{prefix}: P0 exceeds P_lim")
        if layer.params.tau_n <= 0.0 or layer.params.tau_p <= 0.0:
            errors.append(f"{prefix}: carrier lifetime must be positive")


def _load_declared_config(path: Path, schema_id: str) -> Any:
    if schema_id == "standard-device-v1":
        return load_device_from_yaml(str(path))
    if schema_id == "scaps-device-v1":
        return load_scaps_yaml(str(path))
    if schema_id == "tandem-v1":
        return load_tandem_from_yaml(str(path))
    raise ReproducibilityError(f"{path}: unknown schema {schema_id!r}")


def _optical_materials(value: Any) -> set[str]:
    """Collect explicit optical-library keys from a nested YAML value."""
    if isinstance(value, dict):
        found = {
            str(item) for key, item in value.items()
            if key == "optical_material" and item
        }
        for item in value.values():
            found.update(_optical_materials(item))
        return found
    if isinstance(value, list):
        found: set[str] = set()
        for item in value:
            found.update(_optical_materials(item))
        return found
    return set()


def validate_matrix(root: Path | None = None) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    matrix = _load_mapping(root / "reproducibility/config_benchmark_matrix.yaml")
    registry_path = root / str(matrix.get("schema_registry", ""))
    registry = _load_mapping(registry_path)
    schemas = registry.get("schemas") or {}
    benchmarks = matrix.get("benchmarks") or {}
    entries = matrix.get("configs") or []
    resource_entries = matrix.get("resources") or []
    errors: list[str] = []

    if matrix.get("schema_version") != 1:
        errors.append("matrix: schema_version must equal 1")
    if registry.get("schema_version") != 1:
        errors.append("schema registry: schema_version must equal 1")
    reference_environment = matrix.get("reference_environment")
    if not isinstance(reference_environment, dict):
        errors.append("matrix: missing reference_environment")
    else:
        _check_required(
            reference_environment,
            ["os", "python", "numpy", "scipy", "blas_threads"],
            "matrix.reference_environment",
            errors,
        )
        if reference_environment.get("blas_threads") != 1:
            errors.append("matrix.reference_environment: blas_threads must equal 1")
    if not isinstance(entries, list):
        raise ReproducibilityError("matrix configs must be a list")

    actual_resources = {
        "perovskite_sim/data/am15g.csv",
        "perovskite_sim/data/layer_templates.yaml",
        "perovskite_sim/data/nk/manifest.yaml",
        *{
            path.relative_to(root).as_posix()
            for path in (root / "perovskite_sim/data/nk").glob("*.csv")
        },
        *{
            path.relative_to(root).as_posix()
            for pattern in ("*.yaml", "*.yml")
            for path in (root / "perovskite_sim/data/references").rglob(pattern)
        },
    }
    declared_resources = [str(item.get("path")) for item in resource_entries]
    if len(declared_resources) != len(set(declared_resources)):
        errors.append("matrix: duplicate resource paths")
    missing_resources = sorted(actual_resources - set(declared_resources))
    extra_resources = sorted(set(declared_resources) - actual_resources)
    if missing_resources or extra_resources:
        errors.append(
            "matrix resource set drift: "
            f"missing={missing_resources}, extra={extra_resources}"
        )
    for item in resource_entries:
        relpath = str(item.get("path"))
        path = root / relpath
        if not path.is_file():
            errors.append(f"resource {relpath}: file does not exist")
        elif sha256_file(path) != item.get("sha256"):
            errors.append(f"resource {relpath}: SHA-256 drift")

    nk_directory = root / "perovskite_sim/data/nk"
    nk_manifest_path = nk_directory / "manifest.yaml"
    if nk_manifest_path.is_file():
        nk_manifest = _load_mapping(nk_manifest_path)
        csv_stems = {path.stem for path in nk_directory.glob("*.csv")}
        manifest_stems = set(nk_manifest)
        missing_nk = sorted(csv_stems - manifest_stems)
        extra_nk = sorted(manifest_stems - csv_stems)
        if missing_nk or extra_nk:
            errors.append(
                "n,k manifest coverage drift: "
                f"missing={missing_nk}, extra={extra_nk}"
            )
        for stem, provenance in nk_manifest.items():
            if not isinstance(provenance, dict):
                errors.append(f"n,k manifest {stem}: provenance must be a mapping")
                continue
            for required in ("source", "wavelength_range_nm", "notes"):
                if not provenance.get(required):
                    errors.append(f"n,k manifest {stem}: missing {required}")

    actual_paths = {
        path.relative_to(root).as_posix()
        for pattern in ("*.yaml", "*.yml")
        for path in (root / "configs").rglob(pattern)
        if "user" not in path.relative_to(root / "configs").parts
    }
    declared_paths = [str(entry.get("path")) for entry in entries]
    if len(declared_paths) != len(set(declared_paths)):
        errors.append("matrix: duplicate config paths")
    missing = sorted(actual_paths - set(declared_paths))
    extra = sorted(set(declared_paths) - actual_paths)
    if missing or extra:
        errors.append(f"matrix config set drift: missing={missing}, extra={extra}")

    status_counts: dict[str, int] = {}
    schema_counts: dict[str, int] = {}
    semantic_hashes: dict[str, str] = {}
    referenced_optical_materials: set[str] = set()
    config_benchmark_refs: dict[str, set[str]] = {}
    allowed_statuses = {"certified", "partial", "unvalidated", "demo", "load_only"}
    for entry in entries:
        relpath = str(entry.get("path"))
        path = root / relpath
        schema_id = str(entry.get("schema"))
        status = str(entry.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
        schema_counts[schema_id] = schema_counts.get(schema_id, 0) + 1
        if status not in allowed_statuses:
            errors.append(f"{relpath}: unknown status {status!r}")
        if schema_id not in schemas:
            errors.append(f"{relpath}: unknown schema {schema_id!r}")
            continue
        if not path.is_file():
            errors.append(f"{relpath}: file does not exist")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != entry.get("sha256"):
            errors.append(f"{relpath}: SHA-256 drift ({actual_hash})")
        raw = _load_mapping(path)
        referenced_optical_materials.update(_optical_materials(raw))
        detected = detect_config_schema(raw)
        if detected != schema_id:
            errors.append(
                f"{relpath}: declared schema {schema_id}, detected {detected}"
            )
        schema = schemas[schema_id]
        _check_required(raw, schema.get("required_top_keys") or [], relpath, errors)
        device = raw.get("device") or {}
        if isinstance(device, dict):
            _check_required(
                device, schema.get("required_device_keys") or [],
                f"{relpath}.device", errors,
            )
        layers = raw.get("layers") or []
        for index, layer in enumerate(layers):
            if isinstance(layer, dict):
                _check_required(
                    layer, schema.get("required_layer_keys") or [],
                    f"{relpath}.layers[{index}]", errors,
                )
        try:
            loaded = _load_declared_config(path, schema_id)
        except Exception as exc:  # report every config in one pass
            errors.append(f"{relpath}: loader failed: {type(exc).__name__}: {exc}")
            continue
        stacks = (
            (loaded.top_cell, loaded.bottom_cell)
            if schema_id == "tandem-v1" else (loaded,)
        )
        for index, stack in enumerate(stacks):
            _validate_stack(stack, f"{relpath} stack[{index}]", errors)
        semantic_hashes[relpath] = semantic_sha256(loaded)
        expected_semantic = entry.get("semantic_sha256")
        if not expected_semantic:
            errors.append(f"{relpath}: missing semantic_sha256")
        elif expected_semantic != semantic_hashes[relpath]:
            errors.append(f"{relpath}: semantic SHA-256 drift")
        benchmark_ids = entry.get("benchmarks") or []
        if not isinstance(benchmark_ids, list):
            errors.append(f"{relpath}: benchmarks must be a list")
            benchmark_ids = []
        if len(benchmark_ids) != len(set(benchmark_ids)):
            errors.append(f"{relpath}: duplicate benchmark references")
        config_benchmark_refs[relpath] = set(benchmark_ids)
        if not benchmark_ids:
            errors.append(f"{relpath}: no benchmark or loader contract declared")
        for benchmark_id in benchmark_ids:
            if benchmark_id not in benchmarks:
                errors.append(f"{relpath}: unknown benchmark {benchmark_id!r}")
        declared_benchmarks = [
            benchmarks[benchmark_id]
            for benchmark_id in benchmark_ids
            if benchmark_id in benchmarks
        ]
        if status == "certified" and any(
            benchmark.get("status") != "pass"
            for benchmark in declared_benchmarks
        ):
            errors.append(f"{relpath}: certified config has a non-passing benchmark")
        if status == "certified" and not any(
            benchmark.get("kind") in {"external", "internal", "numerical"}
            for benchmark in declared_benchmarks
        ):
            errors.append(f"{relpath}: certified status has no physical benchmark")
        certified_evidence = [
            benchmark for benchmark in declared_benchmarks
            if benchmark.get("kind") in {"external", "internal", "numerical"}
        ]
        if status == "certified" and certified_evidence and all(
            benchmark.get("claim_level") == "calibrated_reproduction"
            for benchmark in certified_evidence
        ):
            errors.append(
                f"{relpath}: calibrated reproduction cannot be promoted to certified"
            )
        if status == "load_only" and any(
            benchmark.get("kind") not in {"schema", "gap"}
            for benchmark in declared_benchmarks
        ):
            errors.append(f"{relpath}: load_only config claims physical validation")
        if status == "partial" and not any(
            benchmark.get("kind") in {"external", "internal", "numerical"}
            for benchmark in declared_benchmarks
        ):
            errors.append(f"{relpath}: partial status has no physical benchmark")
        if status == "unvalidated" and not any(
            benchmark.get("kind") == "gap" and benchmark.get("status") == "open"
            for benchmark in declared_benchmarks
        ):
            errors.append(f"{relpath}: unvalidated status has no open gap")

    for benchmark_id, benchmark in benchmarks.items():
        benchmark_kind = benchmark.get("kind")
        benchmark_status = benchmark.get("status")
        allowed_benchmark_statuses = {
            "schema": {"pass"},
            "external": {"pass", "partial"},
            "internal": {"pass", "partial"},
            "numerical": {"pass", "partial"},
            "gap": {"open", "closed"},
        }
        if benchmark_kind not in allowed_benchmark_statuses:
            errors.append(f"benchmark {benchmark_id}: unknown kind {benchmark_kind!r}")
        elif benchmark_status not in allowed_benchmark_statuses[benchmark_kind]:
            errors.append(
                f"benchmark {benchmark_id}: status {benchmark_status!r} is invalid "
                f"for kind {benchmark_kind!r}"
            )
        benchmark_configs = benchmark.get("configs") or []
        if not isinstance(benchmark_configs, list) or not benchmark_configs:
            errors.append(f"benchmark {benchmark_id}: missing explicit configs")
            benchmark_configs = []
        if len(benchmark_configs) != len(set(benchmark_configs)):
            errors.append(f"benchmark {benchmark_id}: duplicate configs")
        for relpath in benchmark_configs:
            if relpath not in set(declared_paths):
                errors.append(
                    f"benchmark {benchmark_id}: unknown config {relpath!r}"
                )
        reverse_configs = {
            relpath for relpath, references in config_benchmark_refs.items()
            if benchmark_id in references
        }
        if set(benchmark_configs) != reverse_configs:
            errors.append(
                f"benchmark {benchmark_id}: config mapping is not bidirectional "
                f"(declared={sorted(benchmark_configs)}, "
                f"referenced_by={sorted(reverse_configs)})"
            )

        node_ids = benchmark.get("node_ids") or []
        if not isinstance(node_ids, list) or not node_ids:
            errors.append(f"benchmark {benchmark_id}: missing explicit node_ids")
            node_ids = []
        if len(node_ids) != len(set(node_ids)):
            errors.append(f"benchmark {benchmark_id}: duplicate node_ids")
        node_paths: set[str] = set()
        for node_id in node_ids:
            node_id = str(node_id)
            if "::" not in node_id:
                errors.append(
                    f"benchmark {benchmark_id}: node ID is not explicit: {node_id!r}"
                )
                continue
            test_path = node_id.split("::", 1)[0]
            node_paths.add(test_path)
            if not (root / test_path).is_file():
                errors.append(
                    f"benchmark {benchmark_id}: missing node file {test_path}"
                )
                continue
            node_name = node_id.split("::", 1)[1].split("[", 1)[0]
            try:
                syntax = ast.parse((root / test_path).read_text(encoding="utf-8"))
            except (OSError, SyntaxError) as exc:
                errors.append(
                    f"benchmark {benchmark_id}: cannot inspect {test_path}: {exc}"
                )
                continue
            function_names = {
                item.name for item in ast.walk(syntax)
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if node_name not in function_names:
                errors.append(
                    f"benchmark {benchmark_id}: missing node function {node_id}"
                )
        command = str(benchmark.get("command", ""))
        for test_path in node_paths:
            if test_path not in command:
                errors.append(
                    f"benchmark {benchmark_id}: command does not execute {test_path}"
                )
        for test_path in benchmark.get("tests") or []:
            if not (root / test_path).is_file():
                errors.append(f"benchmark {benchmark_id}: missing test {test_path}")
        if set(benchmark.get("tests") or []) != node_paths:
            errors.append(
                f"benchmark {benchmark_id}: tests must equal node_ids file set"
            )
        if benchmark.get("kind") == "external":
            source = str(benchmark.get("source_url", ""))
            if not source.startswith("https://doi.org/"):
                errors.append(f"benchmark {benchmark_id}: missing DOI source")
            if benchmark.get("claim_level") not in {
                "calibrated_reproduction", "partial_external_comparison",
            }:
                errors.append(f"benchmark {benchmark_id}: external claim is ambiguous")
            for required in ("protocol", "limitations"):
                if not benchmark.get(required):
                    errors.append(
                        f"benchmark {benchmark_id}: missing external {required}"
                    )
            protocol = benchmark.get("protocol")
            if not isinstance(protocol, dict) or not all(
                isinstance(protocol.get(key), dict) for key in ("local", "source")
            ):
                errors.append(
                    f"benchmark {benchmark_id}: external protocol must split local/source"
                )
            if not (benchmark.get("observed") or benchmark.get("observed_reverse")):
                errors.append(f"benchmark {benchmark_id}: missing observed metrics")
            observed = benchmark.get("observed_reverse") or benchmark.get("observed")
            tolerance = benchmark.get("regression_tolerance")
            if (
                benchmark.get("claim_level") == "calibrated_reproduction"
                and not isinstance(tolerance, dict)
            ):
                errors.append(
                    f"benchmark {benchmark_id}: calibrated reproduction is "
                    "missing regression_tolerance"
                )
            elif isinstance(tolerance, dict) and isinstance(observed, dict):
                if set(tolerance) != set(observed):
                    errors.append(
                        f"benchmark {benchmark_id}: regression_tolerance keys "
                        "must match observed metric keys"
                    )
                for metric, value in tolerance.items():
                    if not _finite_positive(value):
                        errors.append(
                            f"benchmark {benchmark_id}: tolerance for {metric} "
                            "must be finite and positive"
                        )
            if benchmark.get("claim_level") == "calibrated_reproduction":
                if benchmark.get("evidence_tier") != "calibration_only":
                    errors.append(
                        f"benchmark {benchmark_id}: calibrated reproduction must "
                        "declare evidence_tier=calibration_only"
                    )
                calibration = benchmark.get("calibration")
                checks = benchmark.get("non_calibration_checks")
                if not isinstance(calibration, dict):
                    errors.append(
                        f"benchmark {benchmark_id}: missing calibration contract"
                    )
                else:
                    targets = calibration.get("targets")
                    parameters = calibration.get("parameters")
                    if not isinstance(parameters, dict) or not parameters:
                        errors.append(
                            f"benchmark {benchmark_id}: missing calibrated parameters"
                        )
                    else:
                        for relpath in benchmark_configs:
                            raw_config = _load_mapping(root / relpath)
                            for parameter, expected in parameters.items():
                                try:
                                    actual = _mapping_value_at_path(
                                        raw_config, str(parameter)
                                    )
                                except KeyError:
                                    errors.append(
                                        f"benchmark {benchmark_id}: calibrated parameter "
                                        f"{parameter!r} is missing from {relpath}"
                                    )
                                    continue
                                if isinstance(expected, (int, float)) and isinstance(
                                    actual, (int, float)
                                ):
                                    matches = math.isclose(
                                        float(actual), float(expected),
                                        rel_tol=0.0, abs_tol=0.0,
                                    )
                                else:
                                    matches = actual == expected
                                if not matches:
                                    errors.append(
                                        f"benchmark {benchmark_id}: calibrated parameter "
                                        f"{parameter!r} declares {expected!r} but "
                                        f"{relpath} contains {actual!r}"
                                    )
                    if not isinstance(targets, list) or not targets:
                        errors.append(
                            f"benchmark {benchmark_id}: missing calibration targets"
                        )
                    elif isinstance(checks, list) and set(targets).intersection(checks):
                        errors.append(
                            f"benchmark {benchmark_id}: calibration targets overlap "
                            "non-calibration checks"
                        )
                if not isinstance(checks, list) or not checks:
                    errors.append(
                        f"benchmark {benchmark_id}: missing non-calibration checks"
                    )

    declared_resource_set = set(declared_resources)
    for material in sorted(referenced_optical_materials):
        resource = f"perovskite_sim/data/nk/{material}.csv"
        if resource not in declared_resource_set:
            errors.append(f"optical material {material!r}: undeclared resource {resource}")

    if errors:
        raise ReproducibilityError("reproducibility matrix failed:\n- " + "\n- ".join(errors))
    return {
        "configs": len(entries),
        "benchmarks": len(benchmarks),
        "resources": len(resource_entries),
        "schemas": schema_counts,
        "statuses": status_counts,
        "semantic_sha256": semantic_hashes,
    }


def verify_baseline(
    baseline_id: str, root: Path | None = None, *, check_worktree: bool = False,
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    directory = root / "reproducibility/baselines" / baseline_id
    manifest = _load_mapping(directory / "manifest.yaml")
    errors: list[str] = []
    if manifest.get("baseline_id") != baseline_id:
        errors.append("baseline_id does not match its directory")
    frozen_files = manifest.get("files") or []
    if not isinstance(frozen_files, list) or not frozen_files:
        errors.append("baseline manifest must declare at least one frozen file")
        frozen_files = []
    frozen_paths = [str(item.get("path", "")) for item in frozen_files]
    if len(frozen_paths) != len(set(frozen_paths)):
        errors.append("baseline manifest contains duplicate frozen files")
    patch = manifest.get("patch") or {}
    patch_path = root / str(patch.get("path", ""))
    if not patch_path.is_file():
        errors.append(f"missing baseline patch {patch_path}")
    else:
        if sha256_file(patch_path) != patch.get("sha256"):
            errors.append("baseline patch SHA-256 drift")
        if patch_path.stat().st_size != patch.get("size_bytes"):
            errors.append("baseline patch size drift")

    base_commit = str(manifest.get("base_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", base_commit):
        errors.append("base_commit must be a full 40-character lowercase Git hash")

    checked_files = 0
    if not errors:
        try:
            git_root_result = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            )
            git_root = Path(git_root_result.stdout.strip()).resolve()
            project_prefix = root.relative_to(git_root)
            resolved_result = subprocess.run(
                ["git", "-C", str(git_root), "rev-parse", f"{base_commit}^{{commit}}"],
                check=True,
                capture_output=True,
                text=True,
            )
            if resolved_result.stdout.strip() != base_commit:
                errors.append("base_commit does not resolve to the declared commit")
            else:
                with tempfile.TemporaryDirectory(prefix=f"{baseline_id}-") as temp_name:
                    temp = Path(temp_name)
                    archive_path = temp / "base.tar"
                    reconstructed_root = temp / "tree"
                    reconstructed_root.mkdir()
                    subprocess.run(
                        [
                            "git", "-C", str(git_root), "archive", "--format=tar",
                            f"--output={archive_path}", base_commit,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    with tarfile.open(archive_path, "r") as archive:
                        archive_root = reconstructed_root.resolve()
                        for member in archive.getmembers():
                            destination = (archive_root / member.name).resolve()
                            if not destination.is_relative_to(archive_root):
                                raise ReproducibilityError(
                                    f"unsafe path in Git archive: {member.name}"
                                )
                        extraction_options = (
                            {"filter": "data"} if sys.version_info >= (3, 12) else {}
                        )
                        archive.extractall(reconstructed_root, **extraction_options)
                    numstat_result = subprocess.run(
                        ["git", "apply", "--numstat", str(patch_path)],
                        cwd=reconstructed_root,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    patch_files = {
                        line.split("\t", 2)[2]
                        for line in numstat_result.stdout.splitlines()
                        if line.count("\t") >= 2
                    }
                    declared_patch_files = {
                        (project_prefix / relpath).as_posix()
                        for relpath in frozen_paths
                    }
                    if patch_files != declared_patch_files:
                        errors.append(
                            "baseline patch/file manifest mismatch: "
                            f"patch_only={sorted(patch_files - declared_patch_files)}, "
                            f"manifest_only={sorted(declared_patch_files - patch_files)}"
                        )
                    subprocess.run(
                        ["git", "apply", "--check", str(patch_path)],
                        cwd=reconstructed_root,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    subprocess.run(
                        ["git", "apply", str(patch_path)],
                        cwd=reconstructed_root,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    reconstructed_project = reconstructed_root / project_prefix
                    for item in frozen_files:
                        relpath = str(item.get("path", ""))
                        path = reconstructed_project / relpath
                        checked_files += 1
                        if not path.is_file():
                            errors.append(f"reconstructed P0 file missing: {relpath}")
                        elif sha256_file(path) != item.get("sha256"):
                            errors.append(f"reconstructed P0 file drift: {relpath}")
        except (
            OSError,
            ValueError,
            subprocess.CalledProcessError,
            tarfile.TarError,
            ReproducibilityError,
        ) as exc:
            detail = getattr(exc, "stderr", "") or str(exc)
            errors.append(f"baseline reconstruction failed: {detail.strip()}")

    worktree_checked_files = 0
    if check_worktree:
        for item in frozen_files:
            path = root / str(item.get("path", ""))
            worktree_checked_files += 1
            if not path.is_file() or sha256_file(path) != item.get("sha256"):
                errors.append(f"P0 worktree file drift: {item.get('path')}")
    if errors:
        raise ReproducibilityError("baseline verification failed:\n- " + "\n- ".join(errors))
    return {
        "baseline_id": baseline_id,
        "base_commit": base_commit,
        "patch_sha256": patch.get("sha256"),
        "checked_files": checked_files,
        "worktree_checked_files": worktree_checked_files,
        "reconstruction_verified": checked_files == len(frozen_files),
        "git_tag_created": bool(manifest.get("git_tag_created")),
    }
