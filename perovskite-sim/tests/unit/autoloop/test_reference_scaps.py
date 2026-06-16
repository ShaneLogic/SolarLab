# tests/unit/autoloop/test_reference_scaps.py
import json
import pytest
from perovskite_sim.autoloop.reference import (
    ScapsReferenceSource, build_reference_source,
)

_SCAPS = {
    "base_model": {"Voc_V": 1.17, "Jsc_mA_cm2": 26.28, "FF_percent": 87.0, "PCE_percent": 26.7},
    "sweeps": {"CHI_ETL": {"x_name": "dEc", "n_points": 1,
                           "points": [{"x": 0.0, "Voc_V": 1.25, "Jsc_mA_cm2": 26.3,
                                       "FF_percent": 90.0, "PCE_percent": 29.6}]}},
}


def _write(tmp_path, obj, name="scaps_reference.json"):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def test_scaps_source_base_and_sweep(tmp_path):
    src = ScapsReferenceSource(_write(tmp_path, _SCAPS))
    assert src.base_metrics()["Voc_V"] == 1.17
    assert src.sweep("CHI_ETL")["points"][0]["Voc_V"] == 1.25
    assert src.sweep("MISSING") is None
    assert src.sweep_sheets() == ["CHI_ETL"]


def test_factory_dispatches_scaps_json(tmp_path):
    src = build_reference_source(_write(tmp_path, _SCAPS))
    assert isinstance(src, ScapsReferenceSource)


def test_factory_raises_on_junk(tmp_path):
    with pytest.raises(ValueError):
        build_reference_source(_write(tmp_path, {"nonsense": 1}, "junk.json"))
