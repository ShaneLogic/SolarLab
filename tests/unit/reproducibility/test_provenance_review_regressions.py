"""Exact Q6 implementation-review witnesses; no numerical recomputation."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from solarlab.reproducibility import job_provenance as provenance


FIXTURE = Path(__file__).with_name("fixtures") / "q6_readback_witnesses.tar.gz"
FIXTURE_SHA256 = "7fb7643022cfd15880efe747af6be01d30d19421b0a8bc8460f89f661e4abaf9"
WITNESSES = {
    "OriginalCurve.json": "4402897c96366099f6e02a73eafc67db454632392c232daeb4ef589069ebc6a2",
    "wrong_input_no_attempt.json": "f0388fdade31c77b33e1a9d34370ce49cfae30f70ce8858cff7a0c33ab49d097",
    "wrong_input_unvalidated_attempt.json": "cbd52a420317dc9ffd7f074aa24597bd1b3c003af03a285ace1c8f2d0b683e78",
    "wrong_input_full_foreign_attempt.json": "a09ced7c9142359f8f2897afa39802aa7bf1911451d80284547db4fc3e1f18ad",
}


@pytest.fixture
def saved_witnesses(tmp_path, monkeypatch):
    from solarlab.experiments.jv import ionic_steady

    def forbidden(*args, **kwargs):
        pytest.fail("saved witness must not solve or inspect current source")

    monkeypatch.setattr(ionic_steady, "solve_ionic_steady_state", forbidden)
    monkeypatch.setattr(provenance, "capture_job_provenance", forbidden)
    monkeypatch.setattr(provenance, "_git", forbidden)
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    files = {}
    with tarfile.open(FIXTURE) as archive:
        assert set(archive.getnames()) == set(WITNESSES)
        for name, expected in WITNESSES.items():
            member = archive.getmember(name)
            assert member.isfile() and member.size < 2 * 1024**2
            stream = archive.extractfile(member)
            assert stream is not None
            with stream:
                content = stream.read()
            assert hashlib.sha256(content).hexdigest() == expected
            path = tmp_path / name
            path.write_bytes(content)
            files[name] = path
    original = json.loads(files["OriginalCurve.json"].read_bytes())
    return ionic_steady, files, original


@pytest.mark.parametrize("strict", [False, True])
def test_review_original_curve_keeps_exact_identity_and_bytes(saved_witnesses, strict):
    reader, files, original = saved_witnesses
    path = files["OriginalCurve.json"]
    before = path.read_bytes()
    current = original["result"]["job_provenance"]
    result = reader.read_ionic_steady_jv(path, WITNESSES[path.name],
        current_identity=current, strict_same_source=strict)
    assert result["completed"] is True
    assert result["provenance_comparison"]["status"] == "same"
    assert result["provenance_comparison"]["producer_identity"] == current
    assert path.read_bytes() == before


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("name", [name for name in WITNESSES if name != "OriginalCurve.json"])
def test_review_wrong_input_cannot_be_admitted_by_an_attempt_claim(saved_witnesses, name, strict):
    reader, files, original = saved_witnesses
    path = files[name]
    before = path.read_bytes()
    with pytest.raises(ValueError, match="different inputs|RunStore"):
        reader.read_ionic_steady_jv(path, WITNESSES[name],
            current_identity=original["result"]["job_provenance"], strict_same_source=strict)
    assert path.read_bytes() == before


@pytest.mark.parametrize("strict", [False, True])
def test_correct_input_does_not_authenticate_a_foreign_attempt(saved_witnesses, tmp_path, strict):
    reader, files, original = saved_witnesses
    record = json.loads(files["wrong_input_full_foreign_attempt.json"].read_bytes())
    value = record["result"]["job_provenance"]["record"]
    value["input_sha256"] = original["result"]["job_provenance"]["record"]["input_sha256"]
    record["result"]["job_provenance"] = provenance.JobProvenance.from_record(value).envelope()
    content = json.dumps(record).encode()
    path = tmp_path / "foreign_attempt_with_correct_input.json"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="RunStore"):
        reader.read_ionic_steady_jv(path, hashlib.sha256(content).hexdigest(),
            current_identity=original["result"]["job_provenance"], strict_same_source=strict)


@contextmanager
def admitted_owner():
    with provenance.provenance_owner(source_files=(("solarlab/source.py", b"body"),),
            git_commit="a" * 40, environment_lock=b"lock", source_attested=True, dirty_policy="refuse"):
        yield


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("changed", ["operator", "assembler_identity", "analysis_identity"])
def test_review_owner_refuses_changed_declared_binding(nested, changed):
    binding = {"operator": "outer", "assembler_identity": "b" * 64, "analysis_identity": "c" * 64}
    second = {**binding, changed: "different" if changed == "operator" else "d" * 64}

    def refuse():
        with pytest.raises(ValueError, match="binding"):
            with provenance.job_provenance({}, **second):
                pytest.fail("incompatible operation entered the admitted job")

    with admitted_owner():
        with provenance.job_provenance({}, **binding):
            if nested:
                refuse()
        if not nested:
            refuse()


@pytest.mark.parametrize("nested", [False, True])
def test_identical_declared_binding_reuses_the_one_admission(nested):
    binding = {"operator": "outer", "assembler_identity": "b" * 64, "analysis_identity": "c" * 64}
    with admitted_owner():
        with provenance.job_provenance({}, **binding) as first:
            if nested:
                with provenance.job_provenance({}, **binding) as second:
                    assert second is first
        if not nested:
            with provenance.job_provenance({}, **binding) as second:
                assert second is first
        assert provenance.compare_provenance(first.envelope(), second, strict_same_source=True)["status"] == "same"


def test_assembled_context_cannot_be_reused_without_a_binding_declaration():
    with admitted_owner():
        with provenance.job_provenance({}, operator="outer", assembler_identity="b" * 64, analysis_identity="c" * 64):
            with pytest.raises(ValueError, match="binding"):
                with provenance.job_provenance({}, operator=None):
                    pytest.fail("an assembled stage inherited an undeclared binding")


@pytest.mark.parametrize("declaration", [
    {"operator": "outer"},
    {"operator": "outer", "assembler_identity": "b" * 64},
    {"operator": None, "assembler_identity": "b" * 64, "analysis_identity": "c" * 64},
])
def test_reuse_cannot_omit_an_admitted_assembler_or_its_analysis(declaration):
    with admitted_owner():
        with provenance.job_provenance({}, operator="outer", assembler_identity="b" * 64, analysis_identity="c" * 64):
            with pytest.raises(ValueError, match="binding"):
                with provenance.job_provenance({}, **declaration):
                    pytest.fail("incomplete declaration entered the admitted job")


def test_explicit_inheritance_needs_an_existing_unassembled_job():
    with admitted_owner():
        with pytest.raises(ValueError, match="binding"):
            with provenance.job_provenance({}, operator=None):
                pytest.fail("inheritance created an undeclared admission")
        with provenance.job_provenance({}, operator="initial") as first:
            with pytest.raises(ValueError, match="binding"):
                with provenance.job_provenance({}, operator="different"):
                    pytest.fail("changed unassembled operator was silently reused")
            with provenance.job_provenance({}, operator=None) as inherited:
                assert inherited is first


@pytest.mark.parametrize("strict", [False, True])
def test_manifest_without_attempt_cannot_claim_standalone_admission(saved_witnesses, tmp_path, strict):
    reader, _, original = saved_witnesses
    value = original["result"]["job_provenance"]["record"]
    value["execution_manifest_sha256"] = "d" * 64
    original["result"]["job_provenance"] = provenance.JobProvenance.from_record(value).envelope()
    content = json.dumps(original).encode()
    path = tmp_path / "unverified_manifest.json"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="RunStore"):
        reader.read_ionic_steady_jv(path, hashlib.sha256(content).hexdigest(),
            current_identity=original["result"]["job_provenance"], strict_same_source=strict)
