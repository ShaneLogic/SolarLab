"""CLI-level fail-closed contracts for the physical-interface CBO scan."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.experiments.cbo_scan import InterfaceCBOScanError


def _load_cli_module():
    path = Path("scripts/run_interface_cbo_scan.py")
    spec = importlib.util.spec_from_file_location(
        "run_interface_cbo_scan_test_module",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_writes_a_failed_certificate_when_the_first_grid_fails(
    monkeypatch,
    tmp_path,
):
    module = _load_cli_module()

    def fail_scan(*args, **kwargs):
        raise InterfaceCBOScanError("deliberate numerical failure")

    monkeypatch.setattr(module, "solve_interface_cbo_scan", fail_scan)
    output = tmp_path / "failed-scan.json"

    exit_code = module.main(
        [
            "--out",
            str(output),
            "--delta-min",
            "0",
            "--delta-max",
            "0.1",
            "--short-circuit-only",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["schema_version"] == "1.5"
    assert not payload["complete"]
    assert not payload["certified"]
    assert not payload["statistics_validity"]["certified"]
    assert payload["settings"]["interface_topology"] == "deduplicated_qss"
    assert payload["grid_failure"] == {
        "requested_grid": 30,
        "error_type": "InterfaceCBOScanError",
        "message": "deliberate numerical failure",
    }


def test_cli_requires_transmission_continuation_to_end_at_target():
    module = _load_cli_module()

    with pytest.raises(ValueError, match="must end"):
        module.main(
            [
                "--out",
                "/tmp/not-written.json",
                "--interface-transmission",
                "0.01",
                "--transmission-continuation",
                "0.1",
                "0.03",
            ]
        )


def test_cli_rejects_voltage_ladder_in_short_circuit_mode():
    module = _load_cli_module()

    with pytest.raises(ValueError, match="incompatible with short-circuit-only"):
        module.main(
            [
                "--out",
                "/tmp/not-written.json",
                "--short-circuit-only",
                "--voltage-grid-ladder",
                "29",
                "57",
                "113",
            ]
        )


def test_cli_rejects_non_nested_uniform_voltage_ladder():
    module = _load_cli_module()

    with pytest.raises(ValueError, match="exactly nested"):
        module.main(
            [
                "--out",
                "/tmp/not-written.json",
                "--voltage-grid-ladder",
                "20",
                "40",
                "80",
            ]
        )


def test_cli_rejects_nonphysical_transmission_continuation():
    module = _load_cli_module()

    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        module.main(
            [
                "--out",
                "/tmp/not-written.json",
                "--interface-transmission",
                "0.0",
                "--transmission-continuation",
                "0.0",
            ]
        )


def test_cli_requires_fd_transport_for_two_sided_topology():
    module = _load_cli_module()

    with pytest.raises(ValueError, match="requires --interface-transport-model"):
        module.main(
            [
                "--out",
                "/tmp/not-written.json",
                "--interface-topology",
                "two_sided_trace",
            ]
        )


def test_cli_requires_explicit_legacy_despike_disable_for_two_sided_topology():
    module = _load_cli_module()

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
                "--interface-transport-model",
                "fermi_dirac_richardson",
            ]
        )


def test_cli_grid_builder_removes_shared_nodes_for_two_sided_topology(
    monkeypatch,
):
    module = _load_cli_module()
    monkeypatch.setattr(
        module,
        "build_electrical_grid",
        lambda stack, N: np.arange(4.0),
    )
    monkeypatch.setattr(
        module,
        "build_two_sided_trace_grid",
        lambda grid, stack: grid[[0, 1, 3]],
    )

    grid = module._build_scan_grid(object(), 30, "two_sided_trace")

    assert grid.tolist() == [0.0, 1.0, 3.0]


def test_cli_explicitly_disables_legacy_despike_and_records_protocol(
    monkeypatch,
    tmp_path,
):
    module = _load_cli_module()

    @dataclass(frozen=True)
    class FakeStack:
        het_recomb_despike: float = 0.53

    captured = {}
    monkeypatch.setattr(module, "load_scaps_yaml", lambda path: FakeStack())
    monkeypatch.setattr(module, "electrical_layers", lambda stack: (object(),))

    def fail_scan(stack, *args, **kwargs):
        captured["despike"] = stack.het_recomb_despike
        raise InterfaceCBOScanError("stop after protocol capture")

    monkeypatch.setattr(module, "solve_interface_cbo_scan", fail_scan)
    output = tmp_path / "disabled-despike.json"

    exit_code = module.main(
        [
            "--out",
            str(output),
            "--delta-min",
            "0",
            "--delta-max",
            "0",
            "--short-circuit-only",
            "--interface-topology",
            "two_sided_trace",
            "--interface-transport-model",
            "fermi_dirac_richardson",
            "--disable-legacy-heterojunction-despike",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert captured["despike"] == 0.0
    assert payload["settings"][
        "input_heterojunction_recombination_despike"
    ] == pytest.approx(0.53)
    assert payload["settings"]["heterojunction_recombination_despike"] == 0.0
    assert payload["settings"][
        "legacy_heterojunction_despike_explicitly_disabled"
    ] is True
