"""Pure identity regression: explicit 2D geometry, unchanged absent 1D domain."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import pytest

from examples.research_1d.process_ionic_transient import source_build
from examples.research_1d.steady import load_device
from examples.research_2d.dc_assembly import spatial_input
from solarlab.config.identity import input_identity, physics_identity
from solarlab.experiments.jv.worker import steady_jv_identity_inputs
from solarlab.experiments.two_dimensional.inputs import JV2DInput
from solarlab.experiments.two_dimensional.steady_worker import _identity_inputs
from solarlab.materials.source import SourceDocument

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def inputs():
    commit = os.environ.get("SOLARLAB_STEADY2D_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    archive = os.environ.get("SOLARLAB_STEADY2D_ARCHIVE")
    source = source_build(commit, Path(archive) if archive else None)
    prepared = load_device()
    return prepared, source, steady_jv_identity_inputs(prepared, source), _identity_inputs(prepared, spatial_input(), source)


def test_omitted_and_none_preserve_all_three_1d_documents(inputs):
    _, _, one, _ = inputs
    explicit = replace(one, spatial_geometry=None)
    for name in ("_input_document", "_physical_document", "_execution_binding"):
        assert getattr(one, name) == getattr(explicit, name)
        assert "spatial_geometry" not in json.loads(getattr(one, name))
    assert input_identity(one).sha256 == input_identity(explicit).sha256
    assert physics_identity(one).sha256 == physics_identity(explicit).sha256


def test_dimension_two_requires_actual_geometry_and_survives_child_replace(inputs):
    _, _, one, two = inputs
    assert two.selection.context.dimension == 2
    assert input_identity(two).sha256 != input_identity(one).sha256
    assert physics_identity(two).sha256 != physics_identity(one).sha256
    assert json.loads(two._physical_document)["model_context"]["dimension"] == 2
    for name in ("_input_document", "_physical_document", "_execution_binding"):
        assert getattr(replace(two), name) == getattr(two, name)
        assert "spatial_geometry" in json.loads(getattr(two, name))
    with pytest.raises(ValueError, match="explicit uniform lateral"):
        replace(two, spatial_geometry=None)
    with pytest.raises(ValueError, match="dimension1"):
        replace(one, spatial_geometry=two.spatial_geometry)


def test_lateral_physics_changes_but_mesh_counts_stay_numerical(inputs):
    prepared, source, _, two = inputs
    refined = _identity_inputs(prepared, JV2DInput(kind="jv_2d", lateral_length="1 um", Nx=8,
        Ny_per_layer=4, lateral_bc="neumann"), source)
    wider = _identity_inputs(prepared, JV2DInput(kind="jv_2d", lateral_length="2 um", Nx=4,
        Ny_per_layer=4, lateral_bc="neumann"), source)
    assert input_identity(refined).sha256 != input_identity(two).sha256
    assert refined._execution_binding != two._execution_binding
    assert physics_identity(refined).sha256 == physics_identity(two).sha256
    assert physics_identity(wider).sha256 != physics_identity(two).sha256


@pytest.mark.parametrize("change", ["extra_grain", "missing", "boolean_count", "layer", "boundary", "nonfinite", "policy"])
def test_unbound_geometry_or_grain_declarations_rejected(inputs, change):
    _, _, _, two = inputs
    data = json.loads(two.spatial_geometry.content)
    if change == "extra_grain":
        data["grain_boundaries"] = []
    elif change == "missing":
        del data["lateral_length_m"]
    elif change == "boolean_count":
        data["Nx"] = True
    elif change == "layer":
        data["electrical_layer_ids"] = ["different"]
    elif change == "boundary":
        data["lateral_bc"] = "periodic"
    elif change == "nonfinite":
        data["lateral_length_m"] = float("nan")
    else:
        data["geometry_policy"] = "unknown"
    with pytest.raises(ValueError):
        replace(two, spatial_geometry=SourceDocument(two.spatial_geometry.id, json.dumps(data).encode()))


def test_duplicate_geometry_keys_rejected(inputs):
    _, _, _, two = inputs
    data = two.spatial_geometry.content
    duplicate = data[:-1] + b',"Nx":4}'
    with pytest.raises(ValueError, match="duplicate"):
        replace(two, spatial_geometry=SourceDocument(two.spatial_geometry.id, duplicate))
