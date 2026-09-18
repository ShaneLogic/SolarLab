"""A stable dirty run must not inherit its parent's execution identity."""
from perovskite_sim.experiments.one_dimensional_mechanism_r1_run_record import capture_run_source, compare_run_snapshots
from tests.unit.experiments.test_one_dimensional_mechanism_r1_controlled import repository


def test_stable_uncommitted_source_records_a_base_and_exact_bytes(repository):
    root, project, base = repository
    before = capture_run_source(project)
    assert before["base_commit"] == base and before["source_matches_base_commit"]
    path = project / "perovskite_sim/probe.py"
    path.write_text("VALUE = 'uncommitted experiment'\n")
    changed = capture_run_source(project, test_paths=[path])
    assert changed["base_commit"] == base
    assert changed["source_matches_base_commit"] is False
    assert "perovskite-sim/perovskite_sim/probe.py" in changed["base_content_mismatches"]
    assert changed["executed_source_sha256"] != before["executed_source_sha256"]
    assert changed["working_tree_patch"]
    assert changed["test_sources"][0]["matches_base_commit"] is False
    comparison = compare_run_snapshots(changed, capture_run_source(project, test_paths=[path]))
    assert comparison["source_files_unchanged_during_run"]
    assert comparison["source_matched_base_commit_before_run"] is False


def test_untracked_package_source_is_not_lost_from_the_execution_table(repository):
    root, project, _ = repository
    path = project / "perovskite_sim/new_experiment.py"
    path.write_text("RESULT = 1\n")
    record = capture_run_source(project)
    assert record["untracked_execution_sources"] == ["perovskite-sim/perovskite_sim/new_experiment.py"]
    assert not record["source_matches_base_commit"]
