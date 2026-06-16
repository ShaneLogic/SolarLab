# tests/unit/autoloop/test_ablation.py
from perovskite_sim.autoloop.types import Gap
from perovskite_sim.autoloop.ablation import (
    CANDIDATE_FLAGS, bucket_for_gap, run_ablation,
)


def _gap(sweep="Nd_ETL", kind="trend"):
    return Gap(id=f"trend:{sweep}:V_oc", metric="V_oc", sweep=sweep, sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind=kind,
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


class _FakeRunner:
    """Returns canned badness keyed by a stable variant signature."""
    def __init__(self, table):
        self.table = table
        self.calls = []

    def run(self, variant):
        self.calls.append(variant)
        key = _key(variant)
        if isinstance(self.table.get(key), Exception):
            raise self.table[key]
        return self.table[key]


def _key(variant):
    flags = ",".join(sorted(variant.get("env_flags", {})))
    jv = ",".join(f"{k}={v}" for k, v in sorted(variant.get("jv_overrides", {}).items()))
    return f"{variant.get('measure')}|{flags}|{jv}"


def test_bucket_for_gap():
    assert bucket_for_gap(_gap("Nd_ETL")) == "interface"
    assert bucket_for_gap(_gap("CHI_ETL")) == "interface"
    base = _gap("base", "absolute")
    assert bucket_for_gap(base) == "base"


def test_run_ablation_builds_matrix_with_flag_grid_dark_probes():
    g = _gap("Nd_ETL")
    table = {
        "gap||": 40.0,                                   # baseline
        "gap|SOLARLAB_IFACE_PROJ|": 22.0,                # physics lever (improves)
        "gap|SOLARLAB_IFACE_PLANE|": 41.0,
        "gap|SOLARLAB_INTERFACE_PLANE_STATE|": 39.0,
        "gap||n_points=80": 40.5,                        # grid stable
        "dark||illuminated=False": 0.01,                 # dark ~0
    }
    m = run_ablation(g, _FakeRunner(table))
    kinds = {p.kind for p in m.probes}
    assert kinds == {"flag", "grid", "limiting"}
    assert m.baseline_val == 40.0
    proj = next(p for p in m.probes if p.name == "SOLARLAB_IFACE_PROJ")
    assert proj.delta == -18.0                           # 22 - 40


def test_run_ablation_marks_failed_probe_not_ok():
    g = _gap("Nd_ETL")
    table = {
        "gap||": 40.0,
        "gap|SOLARLAB_IFACE_PROJ|": RuntimeError("solver diverged"),
        "gap|SOLARLAB_IFACE_PLANE|": 41.0,
        "gap|SOLARLAB_INTERFACE_PLANE_STATE|": 39.0,
        "gap||n_points=80": 40.5,
        "dark||illuminated=False": 0.01,
    }
    m = run_ablation(g, _FakeRunner(table))
    proj = next(p for p in m.probes if p.name == "SOLARLAB_IFACE_PROJ")
    assert proj.ok is False
    assert "solver diverged" in proj.note
