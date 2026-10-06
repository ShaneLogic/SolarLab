"""Finite-word exactness and optional compiled-checker provenance, without devices."""
from __future__ import annotations

import ast
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys

import numpy as np
import pytest

from scripts.benchmarks import contract_prototype as C

REFERENCE_SHA256 = "36a371199e94098803fc80200b98b1b743fbf4e92eddd5fc6f8680e9090a614e"


def record(request, data):
    if name := os.environ.get("INTEGER_PRODUCT_CASE_LOG"):
        with Path(name).open("a") as stream:
            stream.write(json.dumps({"test": request.node.nodeid, **data}, sort_keys=True) + "\n")


def load_reference():
    name = os.environ.get("INTEGER_PRODUCT_SCALAR_REFERENCE")
    if not name:
        pytest.skip("the differential gate supplies the immutable scalar reference")
    path = Path(name)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_SHA256
    module_name = "_integer_product_original_scalar"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


@pytest.fixture(scope="module")
def native():
    if C._integer_product_checker is None:
        pytest.skip("optional checker has not been explicitly built and selected")
    return C._integer_product_checker


def finite_word(rng):
    bits = rng.getrandbits(64)
    if (bits >> 52) & 0x7ff == 0x7ff:
        bits ^= 1 << 52
    return struct.unpack(">d", bits.to_bytes(8, "big"))[0]


def pairs():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    normal = float.fromhex("0x1p-1022")
    maximum = float.fromhex("0x1.fffffffffffffp1023")
    values = [0.0, -0.0, tiny, -tiny, normal, -normal, 0.5, -0.5,
              1.0, -1.0, float.fromhex("0x1.0000000000001p0"),
              maximum, -maximum]
    rng = random.Random(0x1A2B3C4D)
    return [(a, b) for a in values for b in values] + [
        (finite_word(rng), finite_word(rng)) for _ in range(192)]


def product_outcome(module, a, b):
    try:
        result = module._finite_two_product(a, b)
    except module.ContractError as error:
        return ("rejected", error.reason)
    return ("returned", struct.pack(">dd", *result).hex())


def oracle(a, b, high, low):
    return Fraction(a) * Fraction(b) == Fraction(high) + Fraction(low)


def test_adversarial_random_and_independent_oracles(native, request):
    tiny = float.fromhex("0x0.0000000000001p-1022")
    maximum = float.fromhex("0x1.fffffffffffffp1023")
    cases = [
        (0.0, -0.0, 0.0, -0.0), (0.0, 1.0, maximum, -maximum),
        (tiny, tiny, 0.0, 0.0), (tiny, 2.0, tiny, tiny),
        (maximum, 2.0, maximum, maximum), (-maximum, 2.0, -maximum, -maximum),
        (1.0, 1.0, 1.0, 1.0),       # Three equal exponents, odd total.
        (1.0, 1.0, 1.0, tiny),      # Unique minimum exponent.
        (1.0, 1.0, 1.0, maximum),   # Two minima cancel; third does not.
        (1.0, 1.0, 2.0, -1.0), (3.0, 1.0, 1.0, 2.0),
        (-3.0, 1.0, -1.0, -2.0), (1.5, 1.0, 1.0, 0.5),
        (0.5, 0.5, 0.25, 2.0**-54),
    ]
    rng = random.Random(0x504F3249)
    cases += [tuple(finite_word(rng) for _ in range(4)) for _ in range(512)]
    reference = load_reference()
    for a, b in pairs():
        try:
            high, low = reference._finite_two_product(a, b)
        except reference.ContractError:
            continue
        cases.append((a, b, high, low))
        cases.append((a, b, high, math.nextafter(low, math.inf)))
    accepted = rejected = 0
    with localcontext() as context:
        context.prec = 2500
        for values in cases:
            expected = oracle(*values)
            a, b, high, low = map(Decimal.from_float, values)
            assert (a * b == high + low) == expected
            assert native(*values) is expected
            assert C._dyadic_product_matches(*values) is expected
            accepted += expected
            rejected += not expected
    assert accepted and rejected
    record(request, {"quadruples": len(cases), "exact": accepted, "inexact": rejected,
                     "oracles": ["independent Fraction", "independent Decimal(2500)",
                                 "unchanged Python dyadic predicate"]})


def test_original_words_signed_zeros_and_first_errors(native, request):
    reference = load_reference()
    inputs = pairs() + [(0.0, math.inf), (math.nan, 1.0), (1.0, -math.inf)]
    observed = set()
    for a, b in inputs:
        candidate = product_outcome(C, a, b)
        assert candidate == product_outcome(reference, a, b)
        observed.add(candidate[1] if candidate[0] == "rejected" else "returned")
    assert observed == {"returned", "linear_product_nonfinite", "linear_product_overflow",
                        "linear_product_underflow_or_inexact"}
    record(request, {"pairs": len(inputs), "outcomes": sorted(observed),
                     "all_returned_bits_and_first_errors_equal": True})


def test_actual_dispatch_and_unsupported_type_fallback(native, monkeypatch, request):
    original = C._dyadic_product_matches
    def forbidden(*values):
        raise AssertionError("finite builtin case fell back from the selected C checker")
    monkeypatch.setattr(C, "_dyadic_product_matches", forbidden)
    assert C._finite_two_product(0.5, 0.5) == (0.25, 0.0)
    assert C._exact_product_matches(1.0, 1.0, 1.0, 2.0**-54) is False
    monkeypatch.setattr(C, "_dyadic_product_matches", original)
    class FloatSubclass(float):
        pass
    unsupported = [1, FloatSubclass(1.0), np.float64(1.0)]
    for value in unsupported:
        assert native(value, 1.0, 1.0, 0.0) is NotImplemented
        assert C._exact_product_matches(value, 1.0, 1.0, 0.0) is True
    assert native(math.inf, 1.0, 1.0, 0.0) is NotImplemented
    record(request, {"actual_C_dispatch_proven": True, "unsupported_types": len(unsupported),
                     "nonfinite_direct_call": "NotImplemented"})


def test_unchanged_arithmetic_and_real_dependency_identity(native, request):
    reference = load_reference()
    old = ast.parse(Path(reference.__file__).read_text())
    new = ast.parse(Path(C.__file__).read_text())
    old_functions = {n.name: n for n in old.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: n for n in new.body if isinstance(n, ast.FunctionDef)}
    protected = ["_dyadic_product_matches", "_word_integer", "_linear_sum",
                 "_linear_quotient", "_upper_ratio", "_linear_reduce"]
    for name in protected:
        assert ast.dump(new_functions[name]) == ast.dump(old_functions[name]), name
    candidate = new_functions["_finite_two_product"]
    substitutions = 0
    for node in ast.walk(candidate):
        if isinstance(node, ast.Name) and node.id == "_exact_product_matches":
            node.id = "_dyadic_product_matches"
            substitutions += 1
    assert substitutions == 1
    assert ast.dump(candidate) == ast.dump(old_functions["_finite_two_product"])
    identity = C._linear_arithmetic_identity()
    backend = identity["product_backend"]
    assert backend["backend"] == "integer128" and backend["abi"]["supported"]
    assert backend["abi"]["integer_bits"] == 128 and backend["abi"]["layout_witnesses"] == 7
    assert identity["contract_sha256"] == hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest()
    assert backend["binary_sha256"] == hashlib.sha256(Path(backend["binary_path"]).read_bytes()).hexdigest()
    source = Path(C.__file__).with_name("_integer_product_native.c")
    assert backend["c_source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    coupled_path = Path(C.__file__).with_name("coupled_device_prototype.py")
    coupled = ast.parse(coupled_path.read_text())
    assignments = [n for n in ast.walk(coupled) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Attribute) and t.attr == "source_identity"
                           for t in n.targets)]
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "_linear_arithmetic_identity"
               for assignment in assignments for n in ast.walk(assignment.value))
    record(request, {"protected_functions": protected, "only_product_call_target_changed": True,
                     "identity": identity, "coupled_identity_wiring": "actual source AST; no device constructed"})


def test_frozen_small_forms_words_and_bounds(native, request):
    reference = load_reference()
    support_path = Path(os.environ["INTEGER_PRODUCT_CORPUS_SUPPORT"])
    spec = importlib.util.spec_from_file_location("_integer_product_corpus_support_test", support_path)
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    corpus = json.loads(Path(os.environ["ORDERED_PRODUCT_CORPUS"]).read_text())
    assert support.build_corpus() == corpus
    checked = 0
    for descriptor in corpus["correctness"]:
        form, components = support.make_case(descriptor)
        for words in (1, 2):
            expected = reference._linear_reduce(form, components, words)
            actual = C._linear_reduce(form, components, words)
            assert support.bits(actual) == support.bits(expected), descriptor["id"]
            checked += 1
    record(request, {"unchanged_corpus": True, "full_reducer_word_and_bound_comparisons": checked})


CHILD_CHECK = r"""
from pathlib import Path
import ctypes, hashlib, json, os, platform, runpy, struct, sys
path = sys.argv[1]
loaded = runpy.run_path(path)
checker = loaded["CHECKER"]
assert checker is not None
abi = loaded["backend_identity"]()["abi"]
assert abi["supported"] and abi["integer_bits"] == 128
assert sys.platform == "darwin" and platform.machine() == "arm64"
lib = ctypes.PyDLL("/usr/lib/libSystem.B.dylib")
for name in ("fegetround", "fesetround", "feclearexcept", "fetestexcept", "feraiseexcept"):
    fn = getattr(lib, name)
    fn.argtypes = [] if name == "fegetround" else [ctypes.c_int]
    fn.restype = ctypes.c_int
environment = (ctypes.c_uint64 * 2)()  # Exact local arm64 fenv_t layout.
for name in ("fegetenv", "fesetenv"):
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.c_void_p]
    fn.restype = ctypes.c_int
assert lib.fegetenv(ctypes.byref(environment)) == 0
values = (0.5, 0.5, 0.25, 0.0)
signaling_nan = struct.unpack(">d", (0x7ff0000000000001).to_bytes(8, "big"))[0]
rows = []
try:
    for mode in (0x00000000, 0x00400000, 0x00800000, 0x00c00000):
        assert lib.fesetround(mode) == 0 and lib.fegetround() == mode
        for sentinel in (0, 0x0002):
            assert lib.feclearexcept(0x009f) == 0
            if sentinel:
                assert lib.feraiseexcept(sentinel) == 0
            before = lib.fetestexcept(0x009f)
            assert checker(*values) is True
            assert checker(signaling_nan, 1.0, 1.0, 0.0) is NotImplemented
            after = lib.fetestexcept(0x009f)
            assert before == after == sentinel
            assert lib.fegetround() == mode
            rows.append({"rounding_mode": mode, "flags_before": before, "flags_after": after})
finally:
    assert lib.fesetenv(ctypes.byref(environment)) == 0
selected_identity = loaded["backend_identity"]()
os.environ["SOLARLAB_INTEGER_PRODUCT"] = "python"
assert runpy.run_path(path)["CHECKER"] is None
os.environ["SOLARLAB_INTEGER_PRODUCT"] = "auto"
os.environ["SOLARLAB_INTEGER_PRODUCT_MANIFEST"] = str(Path(path).parent / "_absent_integer_product_manifest.json")
fallback = runpy.run_path(path)
assert fallback["CHECKER"] is None and fallback["backend_identity"]()["backend"] == "python"
os.environ["SOLARLAB_INTEGER_PRODUCT"] = "required"
try:
    runpy.run_path(path)
except RuntimeError as error:
    assert str(error) == "required integer-product backend unavailable"
else:
    raise AssertionError("explicitly required backend silently fell back")
print(json.dumps({"ABI": abi, "selected_identity": selected_identity, "flag_checks": rows,
                  "mode_and_flags_restored": True, "unknown_backend_fallback": True,
                  "required_backend_fails_closed": True}))
"""


def test_actual_ABI_flags_and_backend_unavailability_in_isolated_child(native, request):
    loader = Path(C.__file__).with_name("_integer_product_backend.py")
    result = subprocess.run([sys.executable, "-B", "-c", CHILD_CHECK, str(loader)],
                            text=True, capture_output=True, check=False, timeout=1.5)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert len(payload["flag_checks"]) == 8
    record(request, {"isolated_child_returncode": result.returncode, "direct_child_reaped": True,
                     "child_inherits_outer_owned_group": True, **payload})


def test_python_fallback_remains_independent(monkeypatch):
    monkeypatch.setattr(C, "_integer_product_checker", None)
    assert C._exact_product_matches(0.5, 0.5, 0.25, 0.0) is True
    assert C._exact_product_matches(0.5, 0.5, 0.25, 2.0**-54) is False
