"""Backend/YAML round-trip for the graded CIGS optical schema."""
from __future__ import annotations

from backend.main import _stack_to_config_dict, stack_from_dict
from perovskite_sim.models.config_loader import load_device_from_yaml


def test_shipped_cigs_optics_round_trips_through_inline_device_path() -> None:
    loaded = load_device_from_yaml("configs/cigs_graded_optics.yaml")
    serialized = _stack_to_config_dict(loaded)
    rebuilt = stack_from_dict(serialized)
    assert serialized["device"]["graded_optics"] is True
    assert rebuilt.graded_optics is True
    assert rebuilt.band_grading is True

    loaded_model = next(
        layer.params.cigs_graded_optics
        for layer in loaded.layers
        if layer.params.cigs_graded_optics is not None
    )
    rebuilt_model = next(
        layer.params.cigs_graded_optics
        for layer in rebuilt.layers
        if layer.params.cigs_graded_optics is not None
    )
    assert rebuilt_model == loaded_model


def test_historical_cigs_config_defaults_graded_optics_off() -> None:
    loaded = load_device_from_yaml("configs/cigs_graded_notch.yaml")
    assert loaded.graded_optics is False
    assert all(
        layer.params.cigs_graded_optics is None for layer in loaded.layers
    )
    rebuilt = stack_from_dict(_stack_to_config_dict(loaded))
    assert rebuilt.graded_optics is False
    assert all(
        layer.params.cigs_graded_optics is None for layer in rebuilt.layers
    )
