# tests/unit/autoloop/test_reference_lab.py
import pytest
from pathlib import Path
from perovskite_sim.autoloop.reference import LabReferenceSource

FIX = Path(__file__).resolve().parents[2] / "integration" / "lab_data_example"
D1 = FIX / "device_01.csv"


def test_single_device_base_metrics_sane():
    src = LabReferenceSource(D1)
    b = src.base_metrics()
    assert 1.0 <= b["Voc_V"] <= 1.2
    assert 20.0 <= b["Jsc_mA_cm2"] <= 28.0
    assert 0.0 < b["FF_percent"] < 100.0
    assert b["PCE_percent"] > 0.0
    assert src.sweep("CHI_ETL") is None and src.sweep_sheets() == []


def test_units_conversion_agrees():
    # Same curve declared in A/m2 (x10) must give the same metrics.
    import csv, tempfile, os
    rows = [(float(r[0]), float(r[1]) * 10.0)
            for r in csv.reader(D1.read_text().splitlines()) if _num(r)]
    d = tempfile.mkdtemp()
    p = Path(d) / "dev_am2.csv"
    p.write_text("V,J\n" + "\n".join(f"{v},{j}" for v, j in rows), encoding="utf-8")
    a = LabReferenceSource(D1, units="mA/cm2").base_metrics()
    b = LabReferenceSource(p, units="A/m2").base_metrics()
    assert abs(a["Voc_V"] - b["Voc_V"]) < 1e-6
    assert abs(a["Jsc_mA_cm2"] - b["Jsc_mA_cm2"]) < 1e-6


def _num(r):
    try:
        float(r[0]); float(r[1]); return True
    except (ValueError, IndexError):
        return False


def test_directory_aggregate_median_vs_champion():
    med = LabReferenceSource(FIX, aggregate="median").base_metrics()
    champ = LabReferenceSource(FIX, aggregate="champion").base_metrics()
    # the unbracketed.csv is skipped; 3 valid devices remain
    assert med["Jsc_mA_cm2"] != champ["Jsc_mA_cm2"] or med["PCE_percent"] <= champ["PCE_percent"]
    assert champ["PCE_percent"] >= med["PCE_percent"]   # champion = best PCE


def test_unbracketed_only_raises(tmp_path):
    p = tmp_path / "only_bad.csv"
    p.write_text("V,J\n0.0,24\n0.5,23\n1.0,20\n", encoding="utf-8")
    with pytest.raises(ValueError):
        LabReferenceSource(p)


def test_unknown_config_raises():
    with pytest.raises(ValueError):
        LabReferenceSource(D1, units="furlongs")
    with pytest.raises(ValueError):
        LabReferenceSource(D1, sign="sideways")
    with pytest.raises(ValueError):
        LabReferenceSource(D1, aggregate="vibes")
