# tests/unit/autoloop/test_types_ablation.py
import dataclasses
import pytest
from perovskite_sim.autoloop.types import Gap, AblationProbe, AblationMatrix


def _gap():
    return Gap(id="g", metric="V_oc", sweep="Nd_ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def test_gap_with_mechanism_returns_new_instance():
    g = _gap()
    g2 = g.with_mechanism("flag SOLARLAB_IFACE_PROJ term")
    assert g.mechanism is None                     # original unchanged
    assert g2.mechanism == "flag SOLARLAB_IFACE_PROJ term"
    assert g2.id == g.id


def test_ablation_probe_is_frozen():
    p = AblationProbe(name="SOLARLAB_IFACE_PROJ", kind="flag", variant={"flag": "X"},
                      baseline_val=40.0, variant_val=25.0, delta=-15.0, ok=True)
    assert p.delta == -15.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.delta = 0.0  # type: ignore[misc]


def test_ablation_matrix_holds_probes():
    p = AblationProbe(name="grid_n80", kind="grid", variant={"n_points": 80},
                      baseline_val=40.0, variant_val=41.0, delta=1.0, ok=True)
    m = AblationMatrix(gap_id="g", baseline_val=40.0, probes=(p,))
    assert m.probes[0].kind == "grid"
    assert m.skipped == ()
