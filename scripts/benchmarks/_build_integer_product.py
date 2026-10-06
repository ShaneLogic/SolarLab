"""Explicit one-shot build of the private checker into a new external directory."""
from hashlib import sha256
import json
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig


def hashed(path):
    with Path(path).open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: _build_integer_product.py COMPILER OUTPUT_DIRECTORY")
    compiler, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    here = Path(__file__).resolve().parent
    repo = here.parent.parent
    if output.is_relative_to(repo):
        raise SystemExit("compiled artifacts must remain outside the repository")
    if output.exists():
        raise SystemExit("refuse to overwrite an existing build attempt")
    if sys.implementation.name != "cpython" or sys.platform != "darwin":
        raise SystemExit("this explicit build command supports CPython on macOS only")
    source = here / "_integer_product_native.c"
    include = Path(sysconfig.get_path("include")).resolve()
    sdk = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk").resolve()
    flags = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-fPIC",
             "-fno-fast-math", "-shared", "-undefined", "dynamic_lookup",
             "-isysroot", str(sdk), "-I", str(include)]
    build = {
        "source_path": str(source), "source_sha256": hashed(source),
        "loader_sha256": hashed(here / "_integer_product_backend.py"),
        "builder_sha256": hashed(__file__), "compiler_path": str(compiler),
        "compiler_sha256": hashed(compiler), "flags": flags,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_executable_sha256": hashed(Path(sys.executable).resolve()),
        "python_cache_tag": sys.implementation.cache_tag, "python_version": sys.version,
        "python_headers": {str(p): hashed(p) for p in sorted(include.rglob("*.h"))},
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
    }
    build_id = sha256(json.dumps(build, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.mkdir(parents=True)
    binary = output / ("_integer_product_native" + build["extension_suffix"])
    dependencies = output / "HeaderDependencies.d"
    command = [str(compiler), *flags,
               '-DINTEGER_SOURCE_SHA256="' + build["source_sha256"] + '"',
               '-DINTEGER_BUILD_ID="' + build_id + '"',
               "-MD", "-MF", str(dependencies), "-MT", "integer_product",
               str(source), "-o", str(binary)]
    (output / "BuildRequest.json").write_text(json.dumps(
        {"build": build, "build_id": build_id, "command": command}, indent=2) + "\n")
    subprocess.run(command, check=True, timeout=2.0)
    dependency_names = shlex.split(dependencies.read_text().replace("\\\n", "").split(":", 1)[1])
    manifest = {"schema": "solarlab.integer-product-build.v1", "build": build,
                "build_id": build_id, "command": command,
                "binary_path": str(binary), "binary_sha256": hashed(binary),
                "actual_dependency_sha256": {name: hashed(name) for name in dependency_names}}
    (output / "Manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(output / "Manifest.json"),
                      "binary_sha256": manifest["binary_sha256"], "build_id": build_id}))


if __name__ == "__main__":
    main()
