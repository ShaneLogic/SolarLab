"""Supported reads classify scientific input bytes outside docs/*.md too."""
import hashlib
import json
from pathlib import Path

import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_checkout as checkout
from perovskite_sim.experiments.one_dimensional_mechanism_r1_evidence import verify_source_reads
from tests.unit.experiments.test_one_dimensional_mechanism_r1_controlled import repository, launch, commit


@pytest.mark.parametrize("relative", ["reproducibility/waiver.md", "data/limit.json", "docs/limit.txt"])
def test_formal_supported_read_requires_registration_in_any_namespace(repository, relative):
    root, project, _ = repository
    path = project / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("0.1\n")
    runner = project / checkout._RUNNER
    runner.write_text(runner.read_text() + "\ncontext.read_bytes(" + repr(relative) + ")\n")
    result = launch((root, project, commit(root)))
    assert result.returncode == 1
    assert "requires dependency classification" in result.stderr
    assert relative in result.stderr


def test_registered_data_is_frozen_and_enters_identity_and_read_receipt(repository, tmp_path):
    root, project, _ = repository
    relative = "data/physical-limit.json"
    path = project / relative
    path.parent.mkdir()
    path.write_bytes(b'{"limit": 2e-6}\n')
    helper = project / checkout._CHECKOUT_MODULE
    helper.write_text(helper.read_text() + "\nR1_SOURCE_DEPENDENCIES[" + repr(relative) + "] = 'physical_standard'\n")
    runner = project / checkout._RUNNER
    runner.write_text(runner.read_text() + "\n" + f'''
raw = context.read_bytes({relative!r}, role="physical_standard")
(context.project / {relative!r}).write_bytes(b'changed after capture')
assert context.read_bytes({relative!r}) == raw
from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import record_source_reads
record_source_reads(args.output, context)
''')
    revision = commit(root)
    output = tmp_path / "recorded"
    result = launch((root, project, revision), "--output", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    execution = json.loads((output / "ExecutionSourceV1.json").read_text())
    name = "perovskite-sim/" + relative
    identity = execution["required_sources"][name]
    assert identity["sha256"] == hashlib.sha256(b'{"limit": 2e-6}\n').hexdigest()
    receipt = json.loads((output / "SourceReadsV1.json").read_text())
    assert receipt["reads"][relative] == {"role": "physical_standard", **identity}
    # A changed disk input is refused under the original source anchor.
    second = launch((root, project, revision))
    assert second.returncode == 1
    assert "source differs from caller commit blob" in second.stderr


def test_receipt_cannot_reclassify_a_read_or_change_its_digest(tmp_path):
    source = {"perovskite-sim/" + relative: {"sha256": "a" * 64, "bytes": 3}
              for relative in checkout.R1_SOURCE_DEPENDENCIES}
    registry = {name: {"role": role, **source["perovskite-sim/" + name]}
                for name, role in checkout.R1_SOURCE_DEPENDENCIES.items()}
    first = next(iter(registry))
    receipt = {"schema": "R1SourceReadsV1", "source_commit": "source", "registry": registry,
               "reads": {first: dict(registry[first])},
               "scope": "supported_frozen_data_reads; not_arbitrary_external_IO"}
    path = tmp_path / "SourceReadsV1.json"
    path.write_text(json.dumps(receipt))
    assert verify_source_reads(tmp_path, {"source_commit": "source"}, source)["supported_reads_checked"] == 1
    receipt["reads"][first]["sha256"] = "b" * 64
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="unclassified or changed"):
        verify_source_reads(tmp_path, {"source_commit": "source"}, source)
