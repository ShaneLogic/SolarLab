"""Historical golden (legacy b1f953c) and GateSpec completeness.

* The stored fixture must be reproduced bit-for-bit by the legacy engine at
  the recorded source digests (proves the golden is a faithful history).
* A candidate is compared bitwise (M) or within P04-GOLDEN-N (N); P-class
  changes are not compared to the golden at all.  The golden never replaces
  the analytic oracles: those run independently in the other test files.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

import p04_adapter as adapter
import p04_legacy as legacy
from p04_capture_golden import FIXTURE, SCHEMA, capture_cases, from_hex
from p04_gates import GATES, gate, missing_fields

DOCUMENT = json.loads(FIXTURE.read_text())
CASES = sorted(DOCUMENT["cases"])
CAPABILITY = {
    "bernoulli": "bernoulli",
    "sg_jacobian": "scharfetter_gummel",
    "poisson_layered": "poisson_1d",
    "bl_synthetic": "beer_lambert",
    "bl_ionmonger_benchmark_N30": "beer_lambert",
    "tmm_ionmonger_benchmark_tmm_N30": "tmm",
    "tmm_synthetic_three_layer": "tmm",
    "recombination_total": "bulk_recombination",
    "trap_profiles": "trap_profile",
}


@pytest.mark.platform_bound
def test_fixture_identity_and_sources():
    assert DOCUMENT["schema"] == SCHEMA
    assert DOCUMENT["captured_at_commit"].startswith("b1f953c")
    extra = tuple(legacy.REPO_ROOT / k for k in DOCUMENT["source_sha256"] if not k.startswith("perovskite_sim/"))
    now = legacy.source_digests(extra)
    changed = sorted(k for k, v in DOCUMENT["source_sha256"].items() if now.get(k) != v)
    assert not changed, f"legacy sources changed since capture (re-review the golden): {changed}"
    assert set(CAPABILITY) == set(CASES)


@pytest.fixture(scope="module")
def recaptured():
    with np.errstate(all="ignore"):
        return capture_cases()


@pytest.mark.platform_bound
@pytest.mark.parametrize("case", CASES)
@pytest.mark.skipif(sys.platform != "darwin", reason="bitwise golden was captured on macOS/Accelerate; live legacy parity runs on Linux")
def test_legacy_reproduces_golden_bitwise(case, recaptured):
    stored = DOCUMENT["cases"][case]["outputs"]
    fresh = recaptured[case]["outputs"]
    assert stored.keys() == fresh.keys()
    for key in stored:
        assert stored[key] == fresh[key], f"{case}/{key} differs from the recorded legacy output"


@pytest.mark.platform_bound
def test_quoted_existing_values_are_reproduced():
    """Values already quoted as evidence in perovskite-sim/CLAUDE.md."""

    bl = float.fromhex(DOCUMENT["cases"]["bl_ionmonger_benchmark_N30"]["outputs"]["q_sum_G_w_A_m2"])
    phi = float.fromhex(DOCUMENT["cases"]["bl_ionmonger_benchmark_N30"]["inputs"]["Phi"])
    assert abs(1.602176634e-19 * phi - 224.30473) <= 5e-6  # "q*Phi = 224.30473 A/m^2"
    assert bl < 1.602176634e-19 * phi
    tmm = float.fromhex(DOCUMENT["cases"]["tmm_ionmonger_benchmark_tmm_N30"]["outputs"]["q_sum_G_w_A_m2"])
    assert abs(tmm - 225.001) <= 5e-4  # "exact absorbed budget 225.001 A/m^2"


@pytest.mark.platform_bound
@pytest.mark.parametrize("case", CASES)
@pytest.mark.skipif(sys.platform != "darwin", reason="bitwise golden was captured on macOS/Accelerate; independent analytic gates run on Linux")
def test_candidate_against_golden(case):
    cls = adapter.change_class(CAPABILITY[case])
    if cls == "P":
        pytest.skip("P-class change: compared by new evidence, not by the legacy golden")
    entry = DOCUMENT["cases"][case]
    out = adapter.golden_case(case, entry["inputs"])
    gn = gate("P04-GOLDEN-N")
    compared = 0
    for key, stored in entry["outputs"].items():
        if (case, key) in adapter.GOLDEN_NOT_MIGRATED:
            # Reported in the review with its reason; never silently dropped.
            assert adapter.GOLDEN_NOT_MIGRATED[(case, key)].strip()
            assert key not in out, f"{case}/{key} is declared not migrated but was returned"
            continue
        if key not in out:
            pytest.fail(f"{case}: candidate did not return {key}")
        compared += 1
        ref = from_hex(stored if isinstance(stored, list) else [stored])
        new = np.asarray(out[key], dtype=float).ravel()
        if cls == "M":
            assert [float(v).hex() for v in new] == [float(v).hex() for v in ref], f"{case}/{key} not bitwise"
        else:
            assert np.all(np.abs(new - ref) <= gn.abs_limit + gn.rel_limit * np.abs(ref)), f"{case}/{key}"
    assert compared > 0, f"{case}: nothing compared"


def test_every_gate_is_complete_and_unique():
    ids = [g.gate_id for g in GATES]
    assert len(ids) == len(set(ids))
    for g in GATES:
        assert not missing_fields(g), f"{g.gate_id} missing {missing_fields(g)}"
        assert g.abs_limit >= 0.0 and g.rel_limit >= 0.0
