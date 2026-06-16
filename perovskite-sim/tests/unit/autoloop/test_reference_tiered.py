# tests/unit/autoloop/test_reference_tiered.py
import json
from pathlib import Path
from perovskite_sim.autoloop.reference import (
    TieredReferenceSource, ScapsReferenceSource, LabReferenceSource,
    build_reference_source,
)

INTEG = Path(__file__).resolve().parents[2] / "integration"


def test_tiered_base_from_lab_sweeps_from_scaps():
    scaps = ScapsReferenceSource(INTEG / "scaps_reference.json")
    lab = LabReferenceSource(INTEG / "lab_data_example")
    t = TieredReferenceSource(base_source=lab, sweep_source=scaps)
    assert t.base_metrics() == lab.base_metrics()            # base from lab
    assert t.sweep("Nd_ETL") == scaps.sweep("Nd_ETL")        # sweeps from scaps
    assert t.sweep_sheets() == scaps.sweep_sheets()


def test_factory_dispatches_descriptor_to_tiered():
    src = build_reference_source(INTEG / "scaps_lab_tiered.json")
    assert isinstance(src, TieredReferenceSource)
    # base comes from the lab fixtures, not the scaps base_model
    scaps_base = ScapsReferenceSource(INTEG / "scaps_reference.json").base_metrics()
    assert src.base_metrics()["Voc_V"] != scaps_base["Voc_V"]
    assert src.sweep("Nd_ETL") is not None                   # SCAPS sweep present
