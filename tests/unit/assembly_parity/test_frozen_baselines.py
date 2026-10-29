"""Frozen M-class oracles, independent of the lifetime of the old classes.

Normal tests only read assembly_baselines.json. Never re-freeze to make an
M-class failure pass: a new baseline requires a reviewed P-class change,
its review reference, and an explanation of every changed quantity.

From the checkout root, with BLAS pinned to one thread and PYTHONPATH set to
"$PWD/src:$PWD/tests:$PWD", use the project Python to run this file with
--initial for the first authorized capture at 2ec8d1ba. Subsequent captures
require --refreeze --p-class-review <review-reference>; that argument records
the operator's review attestation, not an automatic approval mechanism.

Vectors hash raw little-endian float64 words (complex extras use interleaved
float64 real/imaginary words; masks use uint8). Matrices use sorted, duplicate-
coalesced CSR row blocks, with raw little-endian int64 indptr/indices and
float64 values. Explicit stored zeros are retained. Shapes and nnz are part
of the oracle. Identical SHA-256 strings are interned solely to keep the JSON
below 1 MB; comparison expands them and names the failing state/quantity/block.
Source hashes are historical provenance, not required to equal migrated code.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import numpy as np
import pytest
import scipy
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from support.assembly_parity import Block, Extraction
from support.assembly_references import REFERENCE_FACTORIES, STATE_NAMES

ROOT = Path(__file__).resolve().parents[3]
BASELINE_PATH = ROOT / "tests/support/assembly_baselines.json"
REFERENCE_COMMIT = "2ec8d1ba017d98174287124790d58770299dbcb0"
SCHEMA = "solarlab.assembly_baselines.v1"
MAX_BYTES = 1_000_000


def words_digest(value: Any, dtype: str) -> str:
    """Hash a specified byte order, without changing the represented precision."""
    array = np.ascontiguousarray(value, dtype=np.dtype(dtype))
    return hashlib.sha256(array.tobytes()).hexdigest()


def quantity_digest(value: Any, blocks: tuple[Block, ...]) -> dict[str, Any]:
    shape = tuple(value.shape)
    dtype = np.dtype(value.dtype)
    if dtype.kind == "b":
        encoded_dtype = "|u1"
    elif dtype.kind == "f" and dtype.itemsize == 8:
        encoded_dtype = "<f8"
    elif dtype.kind == "c" and dtype.itemsize == 16:
        encoded_dtype = "<c16"
    else:
        raise ValueError("oracle quantities must be float64, complex128 or bool")
    result: dict[str, Any] = {"shape": list(shape), "dtype": encoded_dtype, "blocks": []}
    if len(shape) == 1:
        for block in blocks:
            part = np.asarray(value)[block.indices]
            if not np.all(np.isfinite(part)):
                raise ValueError("nonfinite oracle vector")
            result["blocks"].append({"shape": list(part.shape), "nnz": int(np.count_nonzero(part)),
                                      "values_sha256": words_digest(part, encoded_dtype)})
        return result
    if len(shape) != 2 or shape != (blocks[-1].stop, blocks[-1].stop):
        raise ValueError("oracle matrix does not match the named layout")
    matrix = csr_matrix(value, copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("nonfinite oracle matrix")
    result.update(format="csr", nnz=int(matrix.nnz))
    for block in blocks:
        part = matrix[block.indices, :]
        result["blocks"].append({"shape": list(part.shape), "nnz": int(part.nnz),
            "indptr_sha256": words_digest(part.indptr, "<i8"),
            "indices_sha256": words_digest(part.indices, "<i8"),
            "values_sha256": words_digest(part.data, encoded_dtype)})
    return result


def extraction_digest(extraction: Extraction) -> dict[str, Any]:
    assert isinstance(extraction.blocks, tuple)
    return {name: quantity_digest(value, extraction.blocks)
            for name, value in sorted(extraction.quantities().items())}


def factory_digest(name: str) -> tuple[dict[str, Any], dict[str, str]]:
    case = REFERENCE_FACTORIES[name]()
    states: dict[str, Any] = {}
    sources = {}
    for state in case.states:
        extracted = case.extract(state)
        states[state.name] = {
            "state_sha256": words_digest(state.u, "<f8"),
            "history_sha256": words_digest(state.previous_u, "<f8"),
            "rate_sha256": words_digest(state.udot, "<f8"),
            "voltage_V": state.voltage_V, "generation_scale": state.generation_scale,
            "previous_voltage_V": state.previous_voltage_V, "dt_s": state.dt_s,
            "alpha": state.alpha, "frequency_Hz": state.frequency_Hz,
            "coordinate_form": state.coordinate_form,
            "quantities": extraction_digest(extracted),
        }
        sources[extracted.provenance["target_source"]] = extracted.provenance["target_sha256"]
        sources.update({key.removeprefix("fixture:"): value for key, value in extracted.provenance.items()
                        if key.startswith("fixture:")})
    return {"blocks": [[block.name, block.start, block.stop] for block in case.blocks],
            "states": states}, sources


def intern_digests(value: Any, pool: list[str], lookup: dict[str, int]) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key.endswith("_sha256") and isinstance(item, str):
                if item not in lookup:
                    lookup[item] = len(pool)
                    pool.append(item)
                result[key] = lookup[item]
            else:
                result[key] = intern_digests(item, pool, lookup)
        return result
    if isinstance(value, list):
        return [intern_digests(item, pool, lookup) for item in value]
    return value


def expand_digests(value: Any, pool: list[str]) -> Any:
    if isinstance(value, dict):
        return {key: pool[item] if key.endswith("_sha256") and type(item) is int
                else expand_digests(item, pool) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_digests(item, pool) for item in value]
    return value


def compare_factory(name: str, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    assert actual["blocks"] == expected["blocks"], f"{name}: named layout changed"
    assert actual["states"].keys() == expected["states"].keys(), f"{name}: state inventory changed"
    for state_name, state in actual["states"].items():
        old = expected["states"][state_name]
        assert {key: value for key, value in state.items() if key != "quantities"} == {
            key: value for key, value in old.items() if key != "quantities"
        }, f"{name}/{state_name}: probe inputs changed"
        assert state["quantities"].keys() == old["quantities"].keys(), f"{name}/{state_name}: quantities changed"
        for quantity, current in state["quantities"].items():
            reference = old["quantities"][quantity]
            label = f"{name}/{state_name}/{quantity}"
            assert {key: value for key, value in current.items() if key != "blocks"} == {
                key: value for key, value in reference.items() if key != "blocks"
            }, label + ": shape/dtype/nnz changed"
            assert len(current["blocks"]) == len(reference["blocks"]) == len(actual["blocks"])
            for block, now, before in zip(actual["blocks"], current["blocks"], reference["blocks"]):
                assert now == before, label + "/" + block[0] + ": frozen words or pattern changed"


@pytest.fixture(scope="module")
def baselines() -> dict[str, Any]:
    raw = BASELINE_PATH.read_bytes()
    assert len(raw) <= MAX_BYTES
    document = json.loads(raw)
    assert document["schema"] == SCHEMA
    assert len(document["sha256"]) == len(set(document["sha256"]))
    assert all(len(digest) == 64 and set(digest) <= set("0123456789abcdef") for digest in document["sha256"])
    document["factories"] = expand_digests(document["factories"], document["sha256"])
    return document


def test_frozen_inventory(baselines: dict[str, Any]) -> None:
    assert set(baselines["factories"]) == set(REFERENCE_FACTORIES)
    assert len(baselines["factories"]) == 40
    assert sum(len(item["states"]) for item in baselines["factories"].values()) == 160
    for item in baselines["factories"].values():
        assert set(item["states"]) == set(STATE_NAMES)
    if baselines["capture"]["kind"] == "initial":
        assert baselines["capture"]["source_commit"] == REFERENCE_COMMIT
    else:
        assert baselines["capture"]["kind"] == "reviewed_p_class"
        assert baselines["capture"]["review_reference"].strip()


@pytest.mark.parametrize("name", tuple(REFERENCE_FACTORIES))
def test_frozen_factory(name: str, baselines: dict[str, Any]) -> None:
    current, _ = factory_digest(name)
    compare_factory(name, current, baselines["factories"][name])


def test_digest_detects_one_ulp_signed_zero_and_pattern() -> None:
    blocks = (Block("a", 0, 2), Block("b", 2, 4))
    values = np.array([1.0, 0.0, 2.0, 0.0])
    before = quantity_digest(values, blocks)
    changed = values.copy()
    changed[0] = np.nextafter(changed[0], np.inf)
    assert quantity_digest(changed, blocks)["blocks"][0] != before["blocks"][0]
    changed = values.copy()
    changed[3] = -0.0
    assert quantity_digest(changed, blocks)["blocks"][1] != before["blocks"][1]
    diagonal = csr_matrix(np.eye(4))
    permuted = csr_matrix(np.eye(4)[:, [1, 0, 2, 3]])
    first, second = quantity_digest(diagonal, blocks), quantity_digest(permuted, blocks)
    assert first["blocks"][0]["values_sha256"] == second["blocks"][0]["values_sha256"]
    assert first["blocks"][0]["indices_sha256"] != second["blocks"][0]["indices_sha256"]
    assert first == quantity_digest(diagonal.tocsc(), blocks)
    assert words_digest(values, "<f8") == words_digest(values.astype(">f8"), "<f8")


def test_digest_pool_roundtrip() -> None:
    digest = hashlib.sha256(b"fixture").hexdigest()
    record = {"values_sha256": digest, "nested": [{"state_sha256": digest, "nnz": 2}]}
    pool: list[str] = []
    packed = intern_digests(record, pool, {})
    assert pool == [digest]
    assert expand_digests(packed, pool) == record


def freeze(*, initial: bool, review_reference: str = "") -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if initial:
        if BASELINE_PATH.exists() or head != REFERENCE_COMMIT:
            raise ValueError("initial freeze requires the untouched 2ec8d1ba reference and no existing baseline")
        subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src"], cwd=ROOT, check=True)
    elif not BASELINE_PATH.exists() or not review_reference.strip():
        raise ValueError("re-freeze requires an existing baseline and a reviewed P-class reference")
    started = time.perf_counter()
    factories, sources = {}, {}
    with threadpool_limits(limits=1):
        for name in REFERENCE_FACTORIES:
            factories[name], source = factory_digest(name)
            sources.update(source)
    pool: list[str] = []
    packed = intern_digests(factories, pool, {})
    document = {
        "schema": SCHEMA,
        "capture": {"kind": "initial" if initial else "reviewed_p_class", "source_commit": head,
                    "review_reference": review_reference, "python": platform.python_version(),
                    "numpy": np.__version__, "scipy": scipy.__version__},
        "encoding": "row-block sorted CSR; int64 indices, float64/complex128 words little-endian; bool uint8; stored zeros retained",
        "source_files": dict(sorted(sources.items())), "sha256": pool, "factories": packed,
    }
    encoded = (json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(encoded) > MAX_BYTES:
        raise ValueError("frozen baseline exceeds the 1 MB limit")
    BASELINE_PATH.write_bytes(encoded)
    return {"path": str(BASELINE_PATH), "sha256": hashlib.sha256(encoded).hexdigest(),
            "bytes": len(encoded), "factories": len(factories), "states": sum(len(item["states"]) for item in factories.values()),
            "wall_s": time.perf_counter() - started}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initial", action="store_true")
    mode.add_argument("--refreeze", action="store_true")
    parser.add_argument("--p-class-review", default="")
    arguments = parser.parse_args()
    print(json.dumps(freeze(initial=arguments.initial, review_reference=arguments.p_class_review), indent=2))
