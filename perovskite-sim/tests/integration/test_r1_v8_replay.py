"""Actual N16 fine preparation and accepted short-trajectory tampering."""
import json
from pathlib import Path
import shutil

from threadpoolctl import threadpool_limits

from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import require_r1_checkout
from perovskite_sim.experiments.one_dimensional_mechanism_r1_precision import precision_context
from perovskite_sim.models.config_loader import load_device_from_yaml
from scripts.check_r1_v8_replay import (
    canonical, run_campaign, sha, write,
)


def test_actual_preparation_and_short_trajectory_resealed_low_word_rejection(tmp_path):
    project = Path(__file__).resolve().parents[2]
    stack = load_device_from_yaml(project / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
    binding = json.loads((project / "tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json").read_text())
    source_before = require_r1_checkout(project=project)
    with threadpool_limits(1), precision_context():
        prepared = states.prepare_common_state(stack, 16, binding)
        states.verify_prepared_physics(prepared, stack, binding)
        result = states.json_data(protocol.run_r1_step(stack, 16, binding, prepared,
            times_s=[0., 1e-9], policy=protocol.r1_policy(.1), physics_evidence=True))
        assert len(result["accepted_steps"]) == 10 and result["certificate"]["certified"]
    source_after = require_r1_checkout(project=project)
    assert source_before.to_dict() == source_after.to_dict()
    run = tmp_path / "ActualShortRun"
    run.mkdir()
    write(run / "PreparedV1.json", prepared.to_dict())
    write(run / "ResultV1.json", result)
    write(run / "RequestV1.json", {"scope": "actual_short_integration_test", "times_s": [0., 1e-9]})
    (run / "AcceptedStepsV1.jsonl").write_text("".join(canonical(row) + "\n" for row in result["accepted_steps"]))
    shutil.copyfile(project / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml",
                    run / "SourceFixtureV1.yaml")
    shutil.copyfile(project / "tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json",
                    run / "ReferenceBindingV1.json")
    write(run / "SourceReceiptV1.json", {"source_commit": source_before.source_commit,
        "r1_source_content_sha256": source_before.source_content_sha256, "development_only": True})
    write(run / "SummaryV1.json", {"mode": "compensated", "source_unchanged": True,
        "source_commit": source_before.source_commit, "source_content_sha256": source_before.source_content_sha256,
        "request_sha256": sha(run / "RequestV1.json"), "development_only": True})
    write(run / "ManifestV1.json", {path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
                                    for path in sorted(run.iterdir())})
    report = run_campaign(run, tmp_path / "ReplayCampaign", allow_development=True)
    assert report["passed"], report
    assert report["development_only"] and report["checked_prefix_rows"] == 2
    assert report["healthy_prefix"]["evidence_matches_equations"]
    assert not report["healthy_prefix"]["certified"]
    assert report["preparation_mutation"]["required_physical_rejection"]
    assert report["trajectory_mutation"]["required_physical_rejection"]
