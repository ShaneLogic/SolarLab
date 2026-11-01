"""Offline, source-pinned preparation and separately admitted IDA wheel pilot.

No dependencies or SDKs are downloaded or installed. The pilot compiles
_cy_ida and its source-owned SuperLUMT constructor/free, preserving the
RECORD-verified runtime libraries and other extensions byte for byte.
freeze requires every SDK, private source, tool and runtime identity; build
requires the independently reviewed plan digest and Root's start message ID.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import tarfile
import time
import zipfile

HERE = Path(__file__).resolve().parent
MAX_BYTES = 1024**3
MAX_RSS = 3 * 1024**3
MAX_SECONDS = 300
SHA = re.compile(r"[0-9a-f]{64}\Z")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pinned(path: Path, expected: str) -> dict:
    if not SHA.fullmatch(expected or "") or digest(path) != expected:
        raise ValueError("missing or mismatched external identity")
    return json.loads(path.read_text())


def pins() -> dict:
    data = json.loads((HERE / "SourcePinsV1.json").read_text())
    if digest(HERE / "scikit_sundae_1_1_3.patch") != data["patch_sha256"]:
        raise ValueError("patch differs from source manifest")
    if digest(HERE / "ida_observation_copy.h") != data["copy_header_sha256"]:
        raise ValueError("copy header differs from source manifest")
    for name, identity in data["native_cleanup"]["files"].items():
        if name not in {"ida_superlumt_cleanup.c", "ida_superlumt_cleanup.h"} or digest(HERE / name) != identity:
            raise ValueError("native cleanup source differs from source manifest")
    if set(data["native_cleanup"]["files"]) != {"ida_superlumt_cleanup.c", "ida_superlumt_cleanup.h"}:
        raise ValueError("incomplete native cleanup source manifest")
    return data


def apply_patch(source: Path, manifest: dict) -> None:
    """Apply only this exact three-file unified patch, with no fuzzy matching."""
    lines = (HERE / "scikit_sundae_1_1_3.patch").read_text().splitlines(keepends=True)
    i, changed = 0, set()
    while i < len(lines):
        if not lines[i].startswith("--- a/") or not lines[i+1].startswith("+++ b/"):
            raise ValueError("invalid patch header")
        name = lines[i][6:].strip()
        if lines[i+1][6:].strip() != name or name not in manifest["files"] or name in changed:
            raise ValueError("unexpected or repeated patch target")
        path = source / name
        spec = manifest["files"][name]
        if digest(path) != spec["original_sha256"]:
            raise ValueError("upstream file identity changed: " + name)
        old, new, cursor = path.read_text().splitlines(keepends=True), [], 0
        i += 2
        while i < len(lines) and lines[i].startswith("@@ "):
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[i])
            if match is None:
                raise ValueError("invalid hunk")
            begin = int(match[1]) - 1
            if begin < cursor:
                raise ValueError("overlapping hunks")
            new.extend(old[cursor:begin])
            cursor, removed, added = begin, 0, 0
            i += 1
            while i < len(lines) and not lines[i].startswith(("@@ ", "--- a/")):
                kind, text = lines[i][0], lines[i][1:]
                if kind in " -":
                    if cursor >= len(old) or old[cursor] != text:
                        raise ValueError("patch context changed: " + name)
                    cursor += 1
                    removed += 1
                if kind in " +":
                    new.append(text)
                    added += 1
                if kind not in " +-":
                    raise ValueError("unsupported patch line")
                i += 1
            if removed != int(match[2] or 1) or added != int(match[4] or 1):
                raise ValueError("hunk counts disagree")
        new.extend(old[cursor:])
        output = "".join(new).encode()
        if hashlib.sha256(output).hexdigest() != spec["patched_sha256"]:
            raise ValueError("patched identity mismatch: " + name)
        path.write_bytes(output)
        changed.add(name)
    if changed != set(manifest["files"]):
        raise ValueError("incomplete patch")


def prepare(archive: Path, work: Path, build_id: str | None = None) -> dict:
    manifest = pins()
    if digest(archive) != manifest["source_archive"]["sha256"]:
        raise ValueError("source archive identity changed")
    if work.exists():
        raise ValueError("refuse to reuse or overwrite a preparation directory")
    if build_id is not None and not SHA.fullmatch(build_id):
        raise ValueError("invalid build recipe identity")
    # Verify all archive paths and sizes before writing any source.
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        if sum(m.size for m in members) > 8 * 1024**2:
            raise ValueError("unexpected source archive size")
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or ".." in path.parts
                    or path.parts[0] != "scikit_sundae-1.1.3"
                    or not (member.isfile() or member.isdir())):
                raise ValueError("unsafe or unexpected source archive entry")
        source = work / "source"
        source.mkdir(parents=True)
        for member in members:
            path = source.joinpath(*PurePosixPath(member.name).parts[1:])
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(tar.extractfile(member).read())
    apply_patch(source, manifest)
    (source / "src/sksundae/ida_observation_copy.h").write_bytes((HERE / "ida_observation_copy.h").read_bytes())
    for name in manifest["native_cleanup"]["files"]:
        (source / "src/sksundae" / name).write_bytes((HERE / name).read_bytes())
    if build_id is not None:
        path = source / "src/sksundae/_cy_ida.pyx"
        code = path.read_text()
        token = manifest["build_binding_token"]
        if code.count(token) != 1:
            raise ValueError("build binding token is not unique")
        path.write_text(code.replace(token, "_OBSERVATION_BUILD_ID = " + repr(build_id)))
    receipt = {"source": str(source), "archive_sha256": digest(archive),
               "patch_sha256": manifest["patch_sha256"], "build_recipe_sha256": build_id,
               "source_hashes": {name: digest(source / name) for name in manifest["files"]},
               "native_cleanup_hashes": dict(manifest["native_cleanup"]["files"]),
               "compiler_run": False, "native_calls": 0}
    (work / "Preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def verify_files(root: Path, expected: dict, *, complete=False) -> None:
    if complete and {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} != set(expected):
        raise ValueError("incomplete or changed file inventory: " + str(root))
    for name, identity in expected.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not SHA.fullmatch(identity):
            raise ValueError("unsafe or unpinned file inventory")
        if digest(root / name) != identity:
            raise ValueError("input identity changed: " + str(root / name))


def verify_record(root: Path, record: Path, expected_sha: str) -> dict:
    if digest(record) != expected_sha:
        raise ValueError("installed RECORD identity changed")
    verified = {}
    for name, encoded, size in csv.reader(io.StringIO(record.read_text())):
        if not encoded:
            continue  # RECORD itself is externally pinned.
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe installed RECORD path")
        data = (root / name).read_bytes()
        actual = hashlib.sha256(data).digest()
        if (encoded != "sha256=" + base64.urlsafe_b64encode(actual).decode().rstrip("=")
                or len(data) != int(size)):
            raise ValueError("installed payload differs from RECORD: " + name)
        verified[name] = actual.hex()
    return verified


def validate_recipe(recipe: dict) -> None:
    required = {"schema", "python", "python_sha256", "python_version", "cython",
                "compiler", "platform_tools", "sdk_prefix", "sdk_files", "private_files",
                "python_include", "python_headers", "numpy_include", "numpy_headers",
                "native_site", "native_record_sha256", "native_files", "link_libraries",
                "environment", "wheel_filename", "accounting_roots"}
    if set(recipe) != required or recipe["schema"] != "solarlab.ida75-overlay-recipe.v1":
        raise ValueError("incomplete or unknown recipe fields")
    sdk = Path(recipe["sdk_prefix"])
    config = sdk / "include/sundials/sundials_config.h"
    if not config.is_file():
        raise ValueError("matched SUNDIALS SDK unavailable; no build started")
    verify_files(sdk / "include", recipe["sdk_files"], complete=True)
    for name, identity in pins()["nonlinear_guard"]["headers"].items():
        if recipe["sdk_files"].get(name) != identity:
            raise ValueError("nonlinear guard header identity changed: " + name)
    macros = dict(re.findall(r"^#define\s+(\w+)[ \t]*(.*)$", config.read_text(), re.M))
    if (macros.get("SUNDIALS_VERSION", "").strip('"') != "7.5.0"
            or macros.get("SUNDIALS_DOUBLE_PRECISION") != "1"
            or macros.get("SUNDIALS_INT32_T") != "1"
            or "SUNDIALS_SUPERLUMT_ENABLED" not in macros
            or macros.get("SUNDIALS_SUPERLUMT_THREAD_TYPE", "").strip('"') != "OPENMP"
            or "SUNDIALS_BLAS_LAPACK_ENABLED" not in macros
            or not (sdk / "include/superlu_mt").is_dir()):
        raise ValueError("SDK version, types, or optional solvers differ from qualification")
    for name, identity in recipe["private_files"].items():
        if digest(Path(name)) != identity:
            raise ValueError("private source identity changed")
    private = [p for p in recipe["private_files"] if Path(p).name == "ida_impl.h"]
    if len(private) != 1 or recipe["private_files"][private[0]] != pins()["private_header_sha256"]:
        raise ValueError("missing exact IDA private source declaration")
    for stem in ("python", "numpy"):
        verify_files(Path(recipe[stem + "_include"]), recipe[stem + "_headers"], complete=True)
    executable = Path(recipe["python"])
    if digest(executable) != recipe["python_sha256"]:
        raise ValueError("build interpreter identity changed")
    output = subprocess.run([str(executable), "-I", "-B", "-c", "import sys;print(sys.version)"],
                            check=True, capture_output=True, text=True, timeout=5)
    if output.stdout.strip() != recipe["python_version"] or not recipe["python_version"].startswith("3.13."):
        raise ValueError("pilot requires the pinned Python3.13 interpreter")
    tool = recipe["cython"]
    if set(tool) != {"path", "version", "record", "record_sha256", "wheel", "wheel_sha256"}:
        raise ValueError("incomplete Cython pin")
    if tool["version"] != "3.1.2" or digest(Path(tool["wheel"])) != tool["wheel_sha256"]:
        raise ValueError("Cython tool pin changed")
    verify_record(Path(tool["path"]), Path(tool["record"]), tool["record_sha256"])
    compiler = recipe["compiler"]
    if set(compiler) != {"path", "sha256", "version"} or digest(Path(compiler["path"])) != compiler["sha256"]:
        raise ValueError("compiler identity changed")
    output = subprocess.run([compiler["path"], "--version"], check=True,
                            capture_output=True, text=True, timeout=5)
    if output.stdout.strip() != compiler["version"]:
        raise ValueError("compiler version changed")
    expected_tools = {"/usr/bin/install_name_tool", "/usr/bin/codesign", "/usr/bin/otool", "/usr/bin/nm"}
    if set(recipe["platform_tools"]) != expected_tools:
        raise ValueError("incomplete platform-tool pin")
    for path, identity in recipe["platform_tools"].items():
        if digest(Path(path)) != identity:
            raise ValueError("platform-tool identity changed")
    site = Path(recipe["native_site"])
    native = verify_record(site, site / "scikit_sundae-1.1.3.dist-info/RECORD", recipe["native_record_sha256"])
    if native != recipe["native_files"]:
        raise ValueError("qualified runtime payload changed")
    config_values = {line.split("=", 1)[0].strip(): ast.literal_eval(line.split("=", 1)[1].strip())
                     for line in (site / "sksundae/py_config.pxi").read_text().splitlines() if "=" in line}
    if config_values != pins()["required_config"]:
        raise ValueError("qualified configuration changed")
    libraries = recipe["link_libraries"]
    if len(libraries) != 14 or len({i["path"] for i in libraries}) != 14:
        raise ValueError("complete IDA and optional-solver link inputs required")
    for item in libraries:
        if set(item) != {"path", "install_name"} or item["path"] not in native:
            raise ValueError("unpinned link library")
    if {Path(i["path"]).name.split(".")[0][3:] for i in libraries} != set(pins()["required_libraries"]) - {"sundials_cvode", "superlu_mt_OPENMP"}:
        raise ValueError("qualified link capabilities changed")
    if set(recipe["environment"]) != {"SDKROOT", "MACOSX_DEPLOYMENT_TARGET", "SDKSettings_sha256"}:
        raise ValueError("platform environment must be explicit")
    if (recipe["environment"]["MACOSX_DEPLOYMENT_TARGET"] != "11.0"
            or digest(Path(recipe["environment"]["SDKROOT"]) / "SDKSettings.json") != recipe["environment"]["SDKSettings_sha256"]):
        raise ValueError("macOS SDK or deployment target changed")
    if recipe["wheel_filename"] != "scikit_sundae-1.1.3-4ida75nlsguard-cp313-cp313-macosx_11_0_arm64.whl":
        raise ValueError("unexpected pilot wheel tag")


def tree_bytes(root: Path) -> int:
    """Count retained files, excluding explicitly reused symlink targets."""
    if root.is_symlink():
        return 0
    if root.is_file():
        return root.stat().st_size
    return sum(p.stat().st_size for p in root.rglob("*") if not p.is_symlink() and p.is_file())


def combined_bytes(roots: list[str]) -> int:
    paths = [Path(p).absolute() for p in roots]
    if len(set(paths)) != len(paths) or any(a in b.parents for a in paths for b in paths if a != b):
        raise ValueError("accounting roots must be disjoint")
    return sum(tree_bytes(p) for p in paths)


def freeze(recipe_path: Path, recipe_sha: str, archive: Path, work: Path) -> Path:
    recipe = load_pinned(recipe_path, recipe_sha)
    validate_recipe(recipe)
    if not any(Path(p) in work.parents for p in recipe["accounting_roots"]):
        raise ValueError("build directory is outside the admitted accounting roots")
    if combined_bytes(recipe["accounting_roots"]) > MAX_BYTES - 32 * 1024**2:
        raise ValueError("insufficient combined source/tool/build/wheel envelope")
    prepared = prepare(archive, work, recipe_sha)
    source = Path(prepared["source"])
    site = Path(recipe["native_site"])
    for name in ("c_config.pxi", "py_config.pxi"):
        (source / "src/sksundae" / name).write_bytes((site / "sksundae" / name).read_bytes())
    code_path = source / "src/sksundae/_cy_ida.pyx"
    code = code_path.read_text()
    libraries = tuple(sorted((Path(name).name, identity) for name, identity in recipe["native_files"].items()
                             if name.startswith("sksundae/.dylibs/") and name.endswith(".dylib")))
    replacements = {"_OBSERVATION_LIBRARIES = ()": "_OBSERVATION_LIBRARIES = " + repr(libraries),
                    "_OBSERVATION_CONFIG_SHA256 = None": "_OBSERVATION_CONFIG_SHA256 = " + repr(recipe["sdk_files"]["sundials/sundials_config.h"])}
    for token, value in replacements.items():
        if code.count(token) != 1:
            raise ValueError("build binding token is not unique")
        code = code.replace(token, value)
    code_path.write_text(code)
    prepared["source_hashes"] = {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()}
    generated, extension = work / "_cy_ida.c", work / "_cy_ida.cpython-313-darwin.so"
    generate = [recipe["python"], "-I", "-B", "-c",
                "import sys;sys.path.insert(0," + repr(recipe["cython"]["path"]) + ");from Cython.Compiler.Main import main;main(command_line=True)",
                "-3", "-I", str(source / "src"), "-o", str(generated), str(code_path)]
    compile_command = [recipe["compiler"]["path"], "-std=c11", "-O2", "-fno-fast-math", "-ffp-contract=off",
                       "-fPIC", "-bundle", "-undefined", "dynamic_lookup", "-arch", "arm64", "-g0",
                       "-isysroot", recipe["environment"]["SDKROOT"], "-mmacosx-version-min=11.0",
                       "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION", "-D__OPENMP",
                       "-DSUNDIALS_HAS_SUPERLUMT", "-DSUNDIALS_HAS_LAPACK"]
    includes = [recipe["python_include"], recipe["numpy_include"], str(Path(recipe["sdk_prefix"]) / "include"),
                str(Path(recipe["sdk_prefix"]) / "include/superlu_mt"), str(source / "src/sksundae")]
    includes += sorted({str(Path(p).parent) for p in recipe["private_files"]})
    for path in includes:
        compile_command += ["-I", path]
    compile_command += [str(generated), str(source / "src/sksundae/ida_superlumt_cleanup.c"),
                        "-o", str(extension)]
    compile_command += [str(site / item["path"]) for item in recipe["link_libraries"]]
    repair = ["/usr/bin/install_name_tool"]
    for item in recipe["link_libraries"]:
        repair += ["-change", item["install_name"], "@loader_path/.dylibs/" + Path(item["path"]).name]
    repair += [str(extension)]
    commands = [generate, compile_command, repair,
                ["/usr/bin/codesign", "--force", "--sign", "-", str(extension)],
                ["/usr/bin/codesign", "--verify", "--strict", str(extension)]]
    env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
           "TMPDIR": str(work / "tmp"), "SDKROOT": recipe["environment"]["SDKROOT"],
           "MACOSX_DEPLOYMENT_TARGET": "11.0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "VECLIB_MAXIMUM_THREADS": "1", "LC_ALL": "C", "SOURCE_DATE_EPOCH": "315532800"}
    plan = {"schema": "solarlab.ida75-frozen-overlay-build.v1", "recipe": recipe,
            "recipe_path": str(recipe_path), "recipe_sha256": recipe_sha, "prepared": prepared,
            "work": str(work), "commands": commands, "environment": env, "extension": str(extension),
            "helper_sha256": digest(Path(__file__)), "source_pins_sha256": digest(HERE / "SourcePinsV1.json"),
            "requires_external_process_group_supervision": True,
            "limits": {"seconds": MAX_SECONDS, "rss_bytes": MAX_RSS, "combined_retained_bytes": MAX_BYTES}}
    path = work / "FrozenBuild.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    return path


def pack_overlay(plan: dict) -> dict:
    """Stream qualified files once into a wheel; no second library payload copy."""
    work, site = Path(plan["work"]), Path(plan["recipe"]["native_site"])
    source = Path(plan["prepared"]["source"])
    files = {name: site / name for name in plan["recipe"]["native_files"]}
    for name in pins()["files"]:
        files[name.removeprefix("src/")] = source / name
    files["sksundae/ida_observation_copy.h"] = source / "src/sksundae/ida_observation_copy.h"
    for name in pins()["native_cleanup"]["files"]:
        files["sksundae/" + name] = source / "src/sksundae" / name
    files["sksundae/_cy_ida.cpython-313-darwin.so"] = Path(plan["extension"])
    wheel_meta = "scikit_sundae-1.1.3.dist-info/WHEEL"
    record_name = "scikit_sundae-1.1.3.dist-info/RECORD"
    overrides = {wheel_meta: ("Wheel-Version: 1.0\nGenerator: solarlab-ida75-nls-guard-binding.v1\n"
                             "Root-Is-Purelib: false\nBuild: 4ida75nlsguard\nTag: cp313-cp313-macosx_11_0_arm64\n").encode()}
    overlay = work / "overlay"
    overlay.mkdir()
    (work / "wheel").mkdir()
    wheel = work / "wheel" / plan["recipe"]["wheel_filename"]
    rows, hashes = [], {}
    with zipfile.ZipFile(wheel, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(files):
            data = overrides.get(name)
            if data is None:
                data = files[name].read_bytes()
            identity = hashlib.sha256(data).digest()
            hashes[name] = identity.hex()
            rows.append((name, "sha256=" + base64.urlsafe_b64encode(identity).decode().rstrip("="), len(data)))
            item = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = (0o100644 << 16)
            archive.writestr(item, data)
            target = overlay / name
            target.parent.mkdir(parents=True, exist_ok=True)
            # dyld resolves an extension symlink before expanding @loader_path.
            # Keep the newly linked image next to the overlay's .dylibs directory.
            if name in overrides or name == "sksundae/_cy_ida.cpython-313-darwin.so":
                target.write_bytes(data)
            else:
                target.symlink_to(files[name])
        rows.append((record_name, "", ""))
        text = io.StringIO()
        csv.writer(text, lineterminator="\n").writerows(rows)
        data = text.getvalue().encode()
        item = zipfile.ZipInfo(record_name, (1980, 1, 1, 0, 0, 0))
        item.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(item, data)
        (overlay / record_name).write_bytes(data)
    return {"wheel": str(wheel), "wheel_sha256": digest(wheel), "wheel_files": hashes,
            "overlay": str(overlay), "overlay_reuses_qualified_payload_read_only": True}


def install_verified_wheel(packed: dict, destination: Path) -> dict:
    """Materialize a new installation so every loader-relative path is local.

    An overlay symlink to an old extension may expand @loader_path at its old
    location. Native checks use this complete immutable installation instead.
    The wheel is SHA-bound and every extracted payload is checked by RECORD.
    """
    wheel = Path(packed["wheel"])
    if destination.exists() or digest(wheel) != packed["wheel_sha256"]:
        raise ValueError("installation exists or wheel identity changed")
    record_name = "scikit_sundae-1.1.3.dist-info/RECORD"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(packed["wheel_files"]) | {record_name}:
            raise ValueError("wheel inventory differs from packed payload")
        for name in names:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe wheel path")
        destination.mkdir()
        for name in names:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    record = destination / record_name
    verified = verify_record(destination, record, digest(record))
    if verified != packed["wheel_files"]:
        raise ValueError("installed payload differs from the frozen built wheel")
    return {"installed": str(destination), "installed_record_sha256": digest(record),
            "installed_files": verified, "installation_has_no_symlinks": True}


def cleanup_owned(process, *, external_process_group=False) -> dict:
    """Retain ownership and errors even when termination or reaping fails."""
    if process is None:
        return {"status": "no_active_process"}
    result = {"pid": process.pid, "pgid": os.getpgrp() if external_process_group else process.pid,
              "status": "cleanup-unverified", "errors": [],
              "group_cleanup_owner": "external_supervisor" if external_process_group else "builder"}
    try:
        returncode = process.poll()
    except BaseException as error:
        result["errors"].append({"operation": "poll", "type": type(error).__name__, "message": str(error)})
        return result
    if returncode is None:
        try:
            if external_process_group:
                # Killing this shared group would kill the builder before its receipt.
                # The admitted outer owner terminates/checks all remaining group members.
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except BaseException as error:
            result["errors"].append({"operation": "kill_child" if external_process_group else "killpg",
                                     "type": type(error).__name__, "message": str(error)})
        try:
            returncode = process.wait(timeout=1)
        except BaseException as error:
            result["errors"].append({"operation": "wait", "type": type(error).__name__, "message": str(error)})
    result["returncode"] = returncode
    if returncode is not None and not result["errors"]:
        result["status"] = "exited"
    result["direct_child_reaped"] = returncode is not None
    if external_process_group:
        result["group_empty"] = None  # Only the external owner's receipt establishes this.
    return result


def run_build(plan_path: Path, expected: str, start_message: str, *, external_process_group=False) -> dict:
    if not re.fullmatch(r"msg_[0-9a-f]+", start_message or ""):
        raise ValueError("Root start-message identity is required")
    plan = load_pinned(plan_path, expected)
    if plan.get("requires_external_process_group_supervision") and not external_process_group:
        raise ValueError("this build requires the admitted external process-group supervisor")
    if external_process_group and os.getpgrp() != os.getpid():
        raise ValueError("externally supervised builder must be its newly owned group leader")
    if plan["helper_sha256"] != digest(Path(__file__)) or plan["source_pins_sha256"] != digest(HERE / "SourcePinsV1.json"):
        raise ValueError("helper/source manifest changed since freeze")
    recipe = load_pinned(Path(plan["recipe_path"]), plan["recipe_sha256"])
    if recipe != plan["recipe"]:
        raise ValueError("recipe changed")
    work = Path(plan["work"])
    receipt_path = work / "BuildReceipt.json"
    if receipt_path.exists():
        raise ValueError("no retry or overwrite of an attempted pilot")
    started = time.monotonic()
    receipt = {"status": "failed", "root_start_message": start_message, "plan_sha256": expected,
               "commands": [], "peak_group_rss_bytes": 0, "retries": 0, "native_solver_calls": 0,
               "external_process_group_supervision": external_process_group,
               "timing_scope": "builder command period; whole launch/finalization belongs to external receipt"}
    process = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def deadline(signum, frame):
        raise TimeoutError("pilot deadline reached; reserve time for cleanup/receipt")
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS - 2)
    try:
        validate_recipe(recipe)
        verify_files(Path(plan["prepared"]["source"]), plan["prepared"]["source_hashes"], complete=True)
        (work / "tmp").mkdir()
        for number, argv in enumerate(plan["commands"]):
            log = work / f"command{number}.log"
            with log.open("xb") as output:
                process = subprocess.Popen(argv, cwd=work, env=plan["environment"],
                                           stdout=output, stderr=subprocess.STDOUT,
                                           start_new_session=not external_process_group)
                pgid = os.getpgrp() if external_process_group else process.pid
                entry = {"argv": argv, "pid": process.pid, "owned_process_group": pgid, "log": str(log)}
                receipt["commands"].append(entry)
                while process.poll() is None:
                    listing = subprocess.run(["/bin/ps", "-axo", "pgid=,rss="],
                                             capture_output=True, text=True, timeout=2)
                    rss = sum(int(row.split()[1])*1024 for row in listing.stdout.splitlines()
                              if len(row.split()) == 2 and int(row.split()[0]) == pgid)
                    receipt["peak_group_rss_bytes"] = max(receipt["peak_group_rss_bytes"], rss)
                    if rss > MAX_RSS or combined_bytes(recipe["accounting_roots"]) > MAX_BYTES - 65536:
                        raise RuntimeError("combined pilot resource limit exceeded")
                    time.sleep(0.1)
                entry["returncode"] = process.returncode
                process = None
                if entry["returncode"] != 0:
                    raise RuntimeError("build/repair failed; preserve first error without retry")
        linked = subprocess.run(["/usr/bin/otool", "-L", plan["extension"]], check=True,
                                capture_output=True, text=True, timeout=5).stdout
        receipt["linker_output"] = linked
        for item in recipe["link_libraries"]:
            if "@loader_path/.dylibs/" + Path(item["path"]).name not in linked:
                raise ValueError("required qualified link absent from compiled extension")
        if any(line.strip().startswith(("/private/", "@rpath/")) for line in linked.splitlines()[1:]):
            raise ValueError("unrepaired external native dependency")
        symbols = subprocess.run(["/usr/bin/nm", "-u", plan["extension"]], check=True,
                                 capture_output=True, text=True, timeout=5).stdout
        receipt["undefined_symbols"] = symbols
        for symbol in ("_SUNLinSolSetup_SuperLUMT", "_SUNLinSolSolve_SuperLUMT",
                       "_SUNLinSol_LapackDense", "_SUNLinSol_LapackBand"):
            if symbol not in symbols:
                raise ValueError("optional solver compiled to an unavailable stub")
        if re.search(r"\b_SUNLinSol_SuperLUMT\s*$", symbols, re.M):
            raise ValueError("the owning extension still calls the unpatched sparse constructor")
        defined = subprocess.run(["/usr/bin/nm", "-gU", plan["extension"]], check=True,
                                 capture_output=True, text=True, timeout=5).stdout
        receipt["defined_symbols"] = defined
        if "_sl_SUNLinSol_SuperLUMT" not in defined:
            raise ValueError("source-owned sparse lifetime entry is absent")
        receipt.update(pack_overlay(plan))
        receipt.update(install_verified_wheel(receipt, work / "installed"))
        receipt.update(status="wheel_built_unqualified", compiled_config=pins()["required_config"],
                       scientific_qualification=False, private_abi_independently_qualified=False)
    except BaseException as error:
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        receipt["cleanup"] = cleanup_owned(process, external_process_group=external_process_group)
        if receipt["cleanup"]["status"] == "cleanup-unverified":
            receipt["status"] = "failed_cleanup_unverified"
        receipt["elapsed_s"] = time.monotonic() - started
        try:
            receipt["combined_retained_bytes_before_receipt"] = combined_bytes(recipe["accounting_roots"])
            if receipt["elapsed_s"] > MAX_SECONDS or receipt["combined_retained_bytes_before_receipt"] + 65536 > MAX_BYTES:
                receipt["status"] = "failed_resource_limit"
        except BaseException as error:
            receipt["resource_readback_failure"] = {"type": type(error).__name__, "message": str(error)}
            receipt["status"] = "failed_resource_readback"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--archive", type=Path, required=True)
    prepare_parser.add_argument("--work", type=Path, required=True)
    freeze_parser = commands.add_parser("freeze")
    for name in ("recipe", "archive", "work"):
        freeze_parser.add_argument("--"+name, type=Path, required=True)
    freeze_parser.add_argument("--recipe-sha256", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--plan", type=Path, required=True)
    build_parser.add_argument("--plan-sha256", required=True)
    build_parser.add_argument("--start-message", required=True)
    build_parser.add_argument("--external-process-group", action="store_true")
    args = parser.parse_args()
    if args.operation == "prepare":
        print(json.dumps(prepare(args.archive.resolve(), args.work.absolute()), indent=2))
    elif args.operation == "freeze":
        path = freeze(args.recipe.resolve(), args.recipe_sha256, args.archive.resolve(), args.work.absolute())
        print(json.dumps({"plan": str(path), "sha256": digest(path)}))
    else:
        result = run_build(args.plan.resolve(), args.plan_sha256, args.start_message,
                           external_process_group=args.external_process_group)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "wheel_built_unqualified" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
