"""Fail-closed scope contracts for CBO sensitivity screening."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest


def _load_module():
    path = Path("scripts/run_interface_cbo_sensitivity.py")
    spec = importlib.util.spec_from_file_location(
        "run_interface_cbo_sensitivity_test_module",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_grid_match_is_only_a_screening_candidate():
    module = _load_module()

    classification = module._classify_single_grid_candidate(
        numerical_certified=True,
        statistics_certified=True,
        external_certified=True,
    )

    assert classification["single_grid_screen_passed"] is True
    assert classification["certified"] is False
    assert any(
        "grid convergence" in reason
        for reason in classification["certification_reasons"]
    )


def test_failed_statistics_rejects_single_grid_screen():
    module = _load_module()

    classification = module._classify_single_grid_candidate(
        numerical_certified=True,
        statistics_certified=False,
        external_certified=None,
    )

    assert classification["single_grid_screen_passed"] is False
    assert "interface statistics certificate failed" in (
        classification["certification_reasons"]
    )


def test_external_property_is_explicitly_serialized():
    module = _load_module()

    @dataclass
    class ExternalResult:
        reason: str

        @property
        def certified(self) -> bool:
            return False

    payload = module._external_validation_payload(ExternalResult("failed"))

    assert payload == {"reason": "failed", "certified": False}


def test_two_sided_sensitivity_rejects_non_fd_model_before_loading_config():
    module = _load_module()

    with pytest.raises(ValueError, match="currently support only"):
        module.main(
            [
                "--out",
                "/tmp/not-written.json",
                "--interface-topology",
                "two_sided_trace",
                "--models",
                "fermi_richardson",
            ]
        )


def test_two_sided_sensitivity_defaults_to_fd_model():
    module = _load_module()
    args = module.build_parser().parse_args(
        [
            "--out",
            "/tmp/not-written.json",
            "--interface-topology",
            "two_sided_trace",
        ]
    )

    assert args.models is None
    assert args.interface_topology == "two_sided_trace"


def test_two_sided_sensitivity_requires_explicit_legacy_despike_disable():
    module = _load_module()

    with pytest.raises(
        ValueError,
        match="disable-legacy-heterojunction-despike",
    ):
        module.main(
            [
                "--out",
                "/tmp/not-written.json",
                "--interface-topology",
                "two_sided_trace",
            ]
        )
