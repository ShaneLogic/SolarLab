"""Optional exact checker; explicit immutable build manifest, never auto-build."""
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import BuiltinFunctionType

HERE = Path(__file__).resolve().parent


def _hashed(path):
    with Path(path).open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def _load():
    mode = os.environ.get("SOLARLAB_INTEGER_PRODUCT", "auto")
    if mode not in {"auto", "python", "required"}:
        raise RuntimeError("invalid integer-product backend selection")
    source = HERE / "_integer_product_native.c"
    info = {"schema": "solarlab.integer-product-backend.v1", "backend": "python",
            "selection": mode, "loader_sha256": _hashed(__file__),
            "c_source_sha256": _hashed(source) if source.is_file() else None,
            "python_cache_tag": sys.implementation.cache_tag}
    if mode == "python":
        info["reason"] = "explicit_python"
        return None, info
    manifest_name = os.environ.get("SOLARLAB_INTEGER_PRODUCT_MANIFEST")
    try:
        if not manifest_name:
            raise ImportError("no explicit integer-product manifest")
        manifest_path = Path(manifest_name).resolve()
        manifest = json.loads(manifest_path.read_text())
        if manifest["schema"] != "solarlab.integer-product-build.v1":
            raise ImportError("unsupported integer-product manifest")
        build = manifest["build"]
        build_id = sha256(json.dumps(build, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (build_id != manifest["build_id"]
                or build["source_sha256"] != info["c_source_sha256"]
                or build["loader_sha256"] != info["loader_sha256"]
                or build["python_cache_tag"] != sys.implementation.cache_tag
                or Path(build["python_executable"]).resolve() != Path(sys.executable).resolve()
                or build["python_executable_sha256"] != _hashed(Path(sys.executable).resolve())):
            raise ImportError("integer-product source/interpreter binding mismatch")
        binary = Path(manifest["binary_path"]).resolve()
        if binary.parent != manifest_path.parent or _hashed(binary) != manifest["binary_sha256"]:
            raise ImportError("integer-product binary binding mismatch")
        spec = importlib.util.spec_from_file_location(
            "scripts.benchmarks._integer_product_native", binary)
        if spec is None or spec.loader is None:
            raise ImportError("integer-product extension loader unavailable")
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        if (native.SOURCE_SHA256 != info["c_source_sha256"] or native.BUILD_ID != build_id
                or not native.ABI["supported"] or native.ABI["integer_bits"] != 128
                or not isinstance(native.matches, BuiltinFunctionType)
                or _hashed(binary) != manifest["binary_sha256"]):
            raise ImportError("integer-product extension ABI/provenance mismatch")
        info.update(backend="integer128", binary_path=str(binary),
                    binary_sha256=manifest["binary_sha256"], build_id=build_id,
                    manifest_path=str(manifest_path), manifest_sha256=_hashed(manifest_path),
                    abi=dict(native.ABI))
        return native.matches, info
    except (ImportError, OSError, KeyError, TypeError, ValueError, AttributeError) as error:
        if mode == "required":
            raise RuntimeError("required integer-product backend unavailable") from error
        info["reason"] = type(error).__name__ + ": " + str(error)
        return None, info


CHECKER, _identity = _load()
_IDENTITY_JSON = json.dumps(_identity, sort_keys=True, separators=(",", ":"))
del _identity


def backend_identity():
    """Return fresh metadata for the selected code, not cached numeric answers."""
    return json.loads(_IDENTITY_JSON)
