"""Genuine frozen Git sources, target wheel/sdist installation and content checks.

The working authoring validator is not relabelled as part of an older commit.
The isolated target package is built exclusively from the real named Git
snapshot without ``perovskite_sim``; the current validator independently checks
the resulting bytes, packaged material data, web assets, and module origins.
"""

from dataclasses import FrozenInstanceError, replace
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

from solarlab.config.identity import (
    ExecutionEngine,
    execution_identity,
    execution_manifest,
    verify_execution_manifest,
)
from solarlab.materials.source import SourceDocument
from solarlab.reproducibility.release import (
    ReleaseBundle,
    ReleaseFile,
    read_release_bundle,
    snapshot_release_bundle,
    verify_release_installation,
    write_release_bundle,
)
from test_execution_identity_preparation import (
    device as device,
    execution as execution,
    identity_inputs,
)

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = (
    ("uv.lock", "dependency_lock"),
    ("pyproject.toml", "build_configuration"),
    ("src/solarlab/config/identity.py", "validator"),
    ("src/solarlab/reproducibility/sources.py", "validator"),
    ("src/solarlab/reproducibility/PublicAPI.json", "contract"),
    ("reproducibility/RefactorTransitionBoundariesV1.json", "contract"),
    ("src/solarlab/materials/data/nk/MAPbI3.csv", "resource"),
    ("scripts/generate_configuration_schema.py", "build_configuration"),
    ("generated/configuration-schema.json", "generated_schema"),
)
MODULES = (
    "solarlab.cli",
    "solarlab.config.identity",
    "solarlab.config.schema",
    "solarlab.experiments.jv.preparation",
    "solarlab.materials",
    "solarlab.sweeps.preparation",
    "solarlab_server",
    "solarlab_research",
)


def command(args, cwd):
    result = subprocess.run(
        [str(item) for item in args], cwd=cwd, capture_output=True, timeout=120
    )
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()
    return result.stdout


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    root = tmp_path_factory.mktemp("actual-build-source")
    commit = command(["git", "rev-parse", "HEAD"], ROOT).decode().strip()
    paths = (
        "src",
        "uv.lock",
        "pyproject.toml",
        "reproducibility/RefactorTransitionBoundariesV1.json",
        "reproducibility/ScapsComparisonContractV1.json",
        "reproducibility/HysteresisReferenceContractV1.json",
        "reproducibility/HysteresisReferenceContractV2.json",
        "reproducibility/PerformanceCasesV1.json",
        "scripts/generate_configuration_schema.py",
        "examples/research_1d",
        "examples/research_2d",
        "tests/integration/test_research_1d_workflow.py",
        "tests/refactor",
        "tests/runtime",
        "web",
        ".gitignore",
    )
    archive = command(["git", "archive", commit, "--", *paths], ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            path = root / member.name
            assert path.resolve().is_relative_to(root) and (
                member.isdir() or member.isfile()
            )
            if member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                extracted = tar.extractfile(member)
                assert extracted is not None
                path.write_bytes(extracted.read())
    # Include prebuilt web/dist if present in checkout, or materialize minimal
    # deterministic SPA entry so target wheel/sdist web_dist bundling is verified.
    checkout_dist = ROOT / "web/dist"
    target_dist = root / "web/dist"
    if checkout_dist.is_dir():
        shutil.copytree(checkout_dist, target_dist)
    else:
        target_dist.mkdir(parents=True, exist_ok=True)
        (target_dist / "index.html").write_bytes(
            b"<!doctype html><title>SolarLab</title>\n"
        )
    content = command(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            """
import sys
sys.path.insert(0, sys.argv[1])
from solarlab.config.schema import configuration_schema_representation
sys.stdout.buffer.write(configuration_schema_representation()[0])
""",
            root / "src",
        ],
        root,
    )
    (root / "generated").mkdir()
    (root / "generated/configuration-schema.json").write_bytes(content)
    bundle = snapshot_release_bundle(
        root, build_source_commit=commit, profile="target", required_files=REQUIRED
    )
    return root, bundle


def test_complete_actual_source_bundle_is_immutable_and_preserves_commit_meaning(
    frozen,
):
    source, bundle = frozen
    assert bundle.profile == "target"
    assert not any(
        item.path.startswith("site/perovskite_sim/") for item in bundle.files
    )
    assert any(
        item.path == "site/solarlab_server/_web_dist/index.html"
        for item in bundle.files
    )
    records = bundle.export()
    assert records["build_source_commit"] == bundle.build_source_commit
    assert (
        records["historical_source_algorithms"] == "unchanged"
        and not records["scientific_qualification"]
    )
    assert hashlib.sha256(bundle.canonical_json).hexdigest() == bundle.sha256
    assert not bundle.can_execute
    expected = {
        path.relative_to(source).as_posix()
        for name in ("solarlab", "solarlab_server", "solarlab_research")
        for path in (source / "src" / name).rglob("*")
        if path.is_file()
    }
    assert expected <= {item.source_path for item in bundle.files}
    assert all(
        item.content == (source / item.source_path).read_bytes()
        for item in bundle.files
    )
    assert any(
        item.path.endswith("/py.typed") and item.content == b""
        for item in bundle.files
    )
    build = bundle.implementation()
    assert build.build_source_commit == bundle.build_source_commit
    assert any(
        item.id == "src/solarlab/config/identity.py" for item in build.files
    )
    assert (
        next(
            item.content
            for item in build.files
            if item.id == "release:content-manifest.v1"
        )
        == bundle.canonical_json
    )
    assert any(
        item.id == "generated/configuration-schema.json" for item in build.files
    )
    assert {item.role for item in bundle.files} == {
        role for _, role in REQUIRED
    } | {"package"}
    with pytest.raises(FrozenInstanceError):
        bundle.profile = "transition"  # type: ignore[misc]
    assert replace(bundle, build_source_commit="3" * 40).sha256 != bundle.sha256
    records["files"].clear()
    assert bundle.export()["files"]


def test_complete_release_implementation_fits_the_real_execution_manifest(
    frozen, device, execution
):
    bundle = frozen[1]
    description = replace(execution, implementation=bundle.implementation())
    inputs, engine = identity_inputs(device), ExecutionEngine("solarlab")
    manifest = execution_manifest(inputs, description, engine=engine)
    assert len(manifest.canonical_json) > 2**21
    assert len(manifest.export()["document"]["implementation_content"]) == len(
        description.implementation.files
    )
    result = verify_execution_manifest(
        manifest,
        inputs,
        description,
        engine=engine,
        environment=description.environment,
        expected_sha256=manifest.sha256,
    )
    assert result.sha256 == manifest.execution_sha256 and not result.cache_eligible


def test_generated_schema_and_empty_marker_records_remain_execution_bound(
    frozen, device, execution
):
    bundle = frozen[1]
    changed = replace(
        bundle,
        files=tuple(
            replace(item, content=item.content + b"\n")
            if item.role == "generated_schema"
            else item
            for item in bundle.files
        ),
    )
    inputs, engine = identity_inputs(device), ExecutionEngine("solarlab")
    first = execution_identity(
        inputs, replace(execution, implementation=bundle.implementation()), engine=engine
    )
    second = execution_identity(
        inputs,
        replace(execution, implementation=changed.implementation()),
        engine=engine,
    )
    assert first.sha256 != second.sha256
    assert {
        item["source_path"] for item in bundle.export()["files"] if item["bytes"] == 0
    }


def test_bundle_roundtrip_preserves_every_byte_and_rejects_existing_destination(
    frozen, tmp_path
):
    bundle = frozen[1]
    output = tmp_path / "release"
    write_release_bundle(bundle, output)
    manifest = SourceDocument(
        "ReleaseBundle.json", (output / "ReleaseBundle.json").read_bytes()
    )
    loaded = read_release_bundle(manifest, output, expected_sha256=bundle.sha256)
    assert loaded == bundle
    with pytest.raises(FileExistsError):
        write_release_bundle(bundle, output)
    with pytest.raises(ValueError, match="expected"):
        read_release_bundle(manifest, output, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    "role", ["dependency_lock", "generated_schema", "validator", "contract", "resource"]
)
def test_missing_required_role_refuses(frozen, role):
    bundle = frozen[1]
    with pytest.raises(ValueError, match="missing"):
        replace(bundle, files=tuple(item for item in bundle.files if item.role != role))


def test_target_profile_excludes_legacy_and_paths_are_explicit(frozen):
    bundle = frozen[1]
    with pytest.raises(ValueError, match="legacy"):
        replace(bundle, profile="transition")
    with pytest.raises(ValueError, match="legacy"):
        replace(
            bundle,
            files=(
                *bundle.files,
                ReleaseFile(
                    "perovskite-sim/perovskite_sim/__init__.py",
                    "site/perovskite_sim/__init__.py",
                    "package",
                    b"",
                ),
            ),
        )
    for source, path in (
        ("../code.py", "share/code.py"),
        ("src/solarlab/a.py", "site/solarlab/b.py"),
    ):
        with pytest.raises(ValueError):
            ReleaseFile(source, path, "package", b"pass\n")
    with pytest.raises(ValueError, match="duplicate"):
        replace(bundle, files=(*bundle.files, bundle.files[0]))


def test_historical_transition_profile_manifest_remains_verifiable(frozen, tmp_path):
    """Historical transition bundles remain readable without changing v1 schema."""
    bundle = frozen[1]
    legacy_init = ReleaseFile(
        "perovskite-sim/perovskite_sim/__init__.py",
        "site/perovskite_sim/__init__.py",
        "package",
        b"# historical transition marker\n",
    )
    transition_bundle = ReleaseBundle(
        bundle.build_source_commit,
        "transition",
        (*bundle.files, legacy_init),
    )
    assert transition_bundle.profile == "transition"
    out = tmp_path / "historical-transition"
    write_release_bundle(transition_bundle, out)
    manifest = SourceDocument(
        "ReleaseBundle.json", (out / "ReleaseBundle.json").read_bytes()
    )
    assert (
        read_release_bundle(
            manifest, out, expected_sha256=transition_bundle.sha256
        )
        == transition_bundle
    )


@pytest.fixture(scope="module")
def installation(frozen, tmp_path_factory):
    source, bundle = frozen
    root = tmp_path_factory.mktemp("isolated-wheel")
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip(
            "release installation requires uv on PATH; CI installs the pinned uv build tool"
        )
    command(
        [
            uv,
            "build",
            "--offline",
            "--python",
            sys.executable,
            "--out-dir",
            root / "dist",
            source,
        ],
        root,
    )
    (wheel,) = (root / "dist").glob("*.whl")
    (sdist,) = (root / "dist").glob("*.tar.gz")
    with zipfile.ZipFile(wheel) as zf:
        wnames = zf.namelist()
    assert not any(
        n.startswith(("perovskite_sim/", "perovskite-sim/", "backend/"))
        for n in wnames
    ), wnames
    assert "solarlab/materials/data/default_catalog_v1.json" in wnames
    assert "solarlab_server/_web_dist/index.html" in wnames
    with tarfile.open(sdist) as tf:
        snames = tf.getnames()
    assert not any(
        "perovskite-sim/" in n or "perovskite_sim/" in n for n in snames
    ), snames
    assert any(n.endswith("/web/dist/index.html") for n in snames), snames

    installed = root / "installed"
    command(
        [
            uv,
            "pip",
            "install",
            "--offline",
            "--no-deps",
            "--python",
            sys.executable,
            "--target",
            installed / "site",
            wheel,
        ],
        root,
    )
    for item in bundle.files:
        if item.path.startswith("share/"):
            path = installed / item.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(item.content)
    script = """
import hashlib, importlib, importlib.metadata, importlib.util, json, pathlib, sys
site = pathlib.Path(sys.argv[1])
sys.path = [str(site)] + [p for p in sys.path if p]
assert importlib.util.find_spec('perovskite_sim') is None
assert importlib.util.find_spec('backend') is None
for forbidden in ('perovskite_sim', 'backend'):
    try:
        importlib.import_module(forbidden)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError(f'{forbidden} must not be importable from target wheel')
for name in json.loads(sys.argv[2]): importlib.import_module(name)
from solarlab.config.schema import configuration_schema_representation
from solarlab.materials import (
    load_packaged_default_catalog,
    load_packaged_resource_library,
    load_packaged_template_library,
)
cat = load_packaged_default_catalog()
res = load_packaged_resource_library()
tmpl = load_packaged_template_library()
assert res.get('MAPbI3', 'nk').shape[1] == 3
assert res.get('am15g', 'spectrum').shape[1] == 2
assert len(tmpl.templates) > 0
assert pathlib.Path(importlib.metadata.distribution('solarlab').locate_file('')).resolve() == site.resolve()
roots = {'solarlab', 'solarlab_research', 'solarlab_server', 'perovskite_sim'}
origins = sorted((name, str(pathlib.Path(module.__file__).resolve())) for name, module in sys.modules.items()
                 if name.split('.')[0] in roots)
print(json.dumps(dict(
    module_origins=origins,
    schema_sha256=hashlib.sha256(configuration_schema_representation()[0]).hexdigest(),
    default_catalog_sha256=cat.content_sha256,
)))
"""
    actual = json.loads(
        command(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                script,
                installed / "site",
                json.dumps(MODULES),
            ],
            root,
        )
    )
    return installed, actual, wheel


def test_actual_wheel_installed_outside_checkout_matches_every_source_and_schema(
    frozen, installation, record_property
):
    bundle = frozen[1]
    root, actual, _ = installation
    maintained_head = command(["git", "rev-parse", "HEAD"], ROOT).decode().strip()
    result = verify_release_installation(
        bundle,
        root,
        expected_sha256=bundle.sha256,
        module_origins=tuple(map(tuple, actual["module_origins"])),
        required_modules=MODULES,
        current_repository_head=maintained_head,
    )
    assert result["source_installation_verified"] and not result["scientific_qualification"]
    assert result["build_source_commit"] == bundle.build_source_commit
    assert result["current_repository_head"] == maintained_head
    assert result["files_verified"] == len(bundle.files)
    (schema,) = (item for item in bundle.files if item.role == "generated_schema")
    assert actual["schema_sha256"] == hashlib.sha256(schema.content).hexdigest()
    assert actual["default_catalog_sha256"]
    assert all(
        not Path(origin).is_relative_to(ROOT) for _, origin in actual["module_origins"]
    )
    record_property(
        "release_check",
        json.dumps(
            {
                **result,
                "bundle": bundle.export(),
                "wheel_sha256": hashlib.sha256(installation[2].read_bytes()).hexdigest(),
                "schema_sha256": actual["schema_sha256"],
                "all_source_bytes_equal": True,
            }
        ),
    )


def test_maintenance_head_observation_cannot_replace_the_build_commit(
    frozen, installation
):
    bundle = frozen[1]
    root, actual, _ = installation
    previous = command(["git", "rev-parse", "HEAD^"], ROOT).decode().strip()
    result = verify_release_installation(
        bundle,
        root,
        expected_sha256=bundle.sha256,
        module_origins=tuple(map(tuple, actual["module_origins"])),
        required_modules=MODULES,
        current_repository_head=previous,
    )
    assert result["current_repository_head"] == previous
    assert (
        result["bundle_sha256"] == bundle.sha256
        and result["build_source_commit"] == bundle.build_source_commit
    )


def test_checkout_fallback_and_missing_entry_origin_refuse(frozen, installation):
    bundle = frozen[1]
    root, actual, _ = installation
    origins = dict(actual["module_origins"])
    origins["solarlab.config.identity"] = str(ROOT / "src/solarlab/config/identity.py")
    with pytest.raises(ValueError, match="editable/unbound"):
        verify_release_installation(
            bundle,
            root,
            expected_sha256=bundle.sha256,
            module_origins=tuple(origins.items()),
            required_modules=MODULES,
        )
    del origins["solarlab.config.identity"]
    with pytest.raises(ValueError, match="required module"):
        verify_release_installation(
            bundle,
            root,
            expected_sha256=bundle.sha256,
            module_origins=tuple(origins.items()),
            required_modules=MODULES,
        )


@pytest.mark.parametrize(
    "change",
    [
        "package",
        "lock",
        "schema",
        "resource",
        "contract",
        "extra",
        "bytecode",
        "symlink",
    ],
)
def test_changed_installed_source_data_and_unbound_content_refuse(
    frozen, tmp_path, change
):
    bundle = frozen[1]
    root = tmp_path / "installed"
    write_release_bundle(bundle, root)
    selected = {
        "package": "site/solarlab/config/identity.py",
        "lock": "share/uv.lock",
        "schema": "share/generated/configuration-schema.json",
        "resource": "site/solarlab/materials/data/nk/MAPbI3.csv",
        "contract": "share/reproducibility/RefactorTransitionBoundariesV1.json",
    }
    if change in selected:
        path = root / selected[change]
        path.write_bytes(path.read_bytes() + b"\n")
    elif change in {"extra", "bytecode"}:
        path = root / "site/solarlab" / (
            "unbound.py" if change == "extra" else "unbound.pyc"
        )
        path.write_bytes(b"unbound")
    else:
        path = root / "site/solarlab/config/identity.py"
        path.unlink()
        path.symlink_to(ROOT / "src/solarlab/config/identity.py")
    with pytest.raises(ValueError, match="changed|unbound|escapes"):
        verify_release_installation(
            bundle,
            root,
            expected_sha256=bundle.sha256,
            module_origins=(("solarlab", str(root / "site/solarlab/__init__.py")),),
            required_modules=("solarlab",),
        )


def test_release_job_provenance_keeps_full_pins_and_strict_source_policy(
    frozen, tmp_path
):
    bundle = frozen[1]
    original = bundle.canonical_json
    recorded = bundle.job_provenance(
        {"declared_input": "source-only test"}, operator="test.release"
    )
    assert recorded.export()["git_commit"] == bundle.build_source_commit
    assert recorded.export()["dirty_policy"] == "refuse"
    assert len(recorded.export()["source_files"]) == len(bundle.files)
    from solarlab.experiments.worker import observe_environment

    root = tmp_path / "installed"
    write_release_bundle(bundle, root)
    environment = observe_environment(("numpy",), ("OMP_NUM_THREADS",))
    checks = dict(
        installation_root=root,
        expected_bundle_sha256=bundle.sha256,
        module_origins=(
            (
                "solarlab.config.identity",
                str(root / "site/solarlab/config/identity.py"),
            ),
        ),
        required_modules=("solarlab.config.identity",),
        environment=environment,
    )
    verified = bundle.verify_job_provenance(recorded.envelope(), recorded, **checks)
    assert verified["status"] == "same" and verified["scientific_qualification"] is False
    with pytest.raises(ValueError, match="environment"):
        bundle.verify_job_provenance(
            recorded.envelope(),
            recorded,
            **{**checks, "environment": replace(environment, python_version="0.0.0")},
        )
    changed = replace(
        bundle,
        files=tuple(
            replace(item, content=item.content + b"\n")
            if item.source_path == "src/solarlab/config/identity.py"
            else item
            for item in bundle.files
        ),
    )
    current = changed.job_provenance({}, operator="test.release")
    with pytest.raises(ValueError, match="strict_same_source"):
        bundle.verify_job_provenance(recorded.envelope(), current, **checks)
    assert bundle.canonical_json == original


def test_target_profile_bundle_builds_roundtrips_and_verifies_without_web_dist(
    frozen, tmp_path
):
    """Target bundles also verify cleanly when optional web/dist is absent."""
    source, _ = frozen
    target_source = tmp_path / "target-build-source"
    for rel in (
        "src",
        "uv.lock",
        "pyproject.toml",
        "reproducibility/RefactorTransitionBoundariesV1.json",
        "scripts/generate_configuration_schema.py",
        "generated/configuration-schema.json",
    ):
        src_item = source / rel
        dst_item = target_source / rel
        if src_item.is_dir():
            shutil.copytree(src_item, dst_item)
        else:
            dst_item.parent.mkdir(parents=True, exist_ok=True)
            dst_item.write_bytes(src_item.read_bytes())
    commit = command(["git", "rev-parse", "HEAD"], ROOT).decode().strip()
    target_bundle = snapshot_release_bundle(
        target_source,
        build_source_commit=commit,
        profile="target",
        required_files=REQUIRED,
    )
    assert target_bundle.profile == "target" and not target_bundle.can_execute
    assert not any(
        item.path.startswith("site/perovskite_sim/") for item in target_bundle.files
    )
    installed = tmp_path / "target-installed"
    write_release_bundle(target_bundle, installed)
    manifest = SourceDocument(
        "ReleaseBundle.json", (installed / "ReleaseBundle.json").read_bytes()
    )
    assert (
        read_release_bundle(manifest, installed, expected_sha256=target_bundle.sha256)
        == target_bundle
    )
    verified = verify_release_installation(
        target_bundle,
        installed,
        expected_sha256=target_bundle.sha256,
        module_origins=(
            (
                "solarlab.config.identity",
                str(installed / "site/solarlab/config/identity.py"),
            ),
        ),
        required_modules=("solarlab.config.identity",),
    )
    assert verified["source_installation_verified"] is True
    assert verified["bundle_sha256"] == target_bundle.sha256
    assert verified["files_verified"] == len(target_bundle.files)


def test_historical_result_provenance_verifies_against_original_source_without_relabelling(
    frozen,
):
    from solarlab.reproducibility.job_provenance import result_provenance

    bundle = frozen[1]
    older_commit = command(["git", "rev-parse", "HEAD^"], ROOT).decode().strip()
    historical_bundle = replace(
        bundle,
        build_source_commit=older_commit,
        files=tuple(
            replace(item, content=item.content + b"# historical\n")
            if item.source_path == "src/solarlab/config/identity.py"
            else item
            for item in bundle.files
        ),
    )
    inputs = {"declared_input": "historical-standalone-case"}
    historical_prov = historical_bundle.job_provenance(
        inputs, operator="test.historical"
    )
    historical_envelope = historical_prov.envelope()
    saved_identity_sha = next(
        row["sha256"]
        for row in historical_prov.export()["source_files"]
        if row["path"] == "src/solarlab/config/identity.py"
    )
    saved_record = {
        "job_provenance": historical_envelope,
        "source_sha256": {"solarlab.config.identity": saved_identity_sha},
        "source_identity": {"git_commit": older_commit},
    }
    current_prov = bundle.job_provenance(inputs, operator="test.historical")
    compared = result_provenance(saved_record, current_prov, expected_inputs=inputs)
    assert compared["status"] == "different" and compared["scientific_qualification"] is False
    assert "git_commit" in compared["different_fields"]
    assert "source_tree_sha256" in compared["different_fields"]
    assert compared["producer_identity"]["record"]["git_commit"] == older_commit
    assert (
        compared["current_identity"]["record"]["git_commit"]
        == bundle.build_source_commit
    )
    assert saved_record["job_provenance"] == historical_envelope
    with pytest.raises(ValueError, match="strict_same_source"):
        result_provenance(
            saved_record,
            current_prov,
            strict_same_source=True,
            expected_inputs=inputs,
        )
    with pytest.raises(ValueError, match="different inputs"):
        result_provenance(
            saved_record, current_prov, expected_inputs={"declared_input": "wrong"}
        )
