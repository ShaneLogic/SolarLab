"""Geometry contracts for the optional PDF vector-reference extractor."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def mod():
    pytest.importorskip("fitz")
    path = Path(__file__).resolve().parents[3] / "scripts/extract_calado_fig1_reference.py"
    spec = importlib.util.spec_from_file_location("calado_vector_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_outline_closes_and_slices_line_segments(mod):
    outline = mod.outline_points([
        ("l", (0, 0), (2, 0)), ("l", (2, 0), (2, 1)),
        ("l", (2, 1), (0, 1)),
    ])
    np.testing.assert_array_equal(outline[0], outline[-1])
    assert mod.slice_outline(outline, 1) == (0, 1)
    assert mod.slice_outline(outline, 3) is None


def test_outline_rejects_disconnected_segments(mod):
    with pytest.raises(ValueError, match="disconnected"):
        mod.outline_points([("l", (0, 0), (1, 0)), ("l", (2, 0), (2, 1))])


def test_multivalued_slice_requires_explicit_envelope(mod):
    outline = np.array([(0, 0), (2, 0), (2, 1), (1, 1),
                        (1, 2), (2, 2), (2, 3), (0, 3), (0, 0)])
    assert mod.slice_outline(outline, 1.5) is None
    assert mod.slice_outline(outline, 1.5, envelope=True) == (0, 3)


def test_cubic_outline_samples_endpoints(mod):
    outline = mod.outline_points([("c", (0, 0), (0, 1), (1, 1), (1, 0))])
    assert outline.shape == (66, 2)
    np.testing.assert_allclose(outline[32], [0.5, 0.75])
    np.testing.assert_array_equal(outline[-1], outline[0])


def test_stroke_domain_uses_cap_centers_not_outline_extent(mod):
    outline = np.array([[0., 0.], [100., 100.], [101., 99.], [1., -1.], [0., 0.]])
    caps = mod.stroke_endcaps(outline)
    np.testing.assert_allclose(caps.mean(axis=1), [[0.5, -0.5], [100.5, 99.5]])


def test_duplicate_vertices_do_not_change_endcaps(mod):
    outline = np.array([[0., 0.], [100., 100.], [101., 99.], [1., -1.], [0., 0.]])
    repeated = np.repeat(outline, [2, 3, 1, 2, 2], axis=0)
    np.testing.assert_array_equal(mod.stroke_endcaps(repeated), mod.stroke_endcaps(outline))


def test_all_identical_vertices_do_not_form_endcaps(mod):
    with pytest.raises(ValueError, match="unambiguous"):
        mod.stroke_endcaps(np.zeros((5, 2)))


def test_unresolved_caps_fail_instead_of_guessing(mod):
    with pytest.raises(ValueError, match="unambiguous"):
        mod.stroke_endcaps(np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0., 0.]]))


def test_cap_only_values_do_not_become_curve_data_but_remain_counted(mod):
    reference = np.array([[0.1, 1, 0, 2], [0.9, 1, 0, 2], [0.99, 1, 0, 2], [1.1, 1, 0, 2]])
    stroke = {"centerline_voltage_interval_V": [0, 1], "endcap_voltage_intervals_V": [[0, 0], [0.95, 1.05]]}
    result = mod.compare_reference_samples(reference, np.array([1, 1, 5, 1e6]), stroke)
    assert result["raw_samples"] == 4
    assert result["endcap_only_samples"] == 1
    assert result["midpoint_comparison_samples"] == 2
    assert result["cap_influenced_domain_samples"] == 1
    assert result["rmse_A_m2"] == 0
    assert result["raw_outline_midpoint_rmse_A_m2"] > 1e5
    assert result["max_graphical_envelope_violation_A_m2"] == 3
    assert result["fraction_within_graphical_envelope"] == pytest.approx(2/3)
