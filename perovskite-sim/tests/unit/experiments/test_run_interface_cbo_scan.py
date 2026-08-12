"""CLI-level fail-closed contracts for the physical-interface CBO scan."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

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
    assert payload["schema_version"] == "1.7"
    assert not payload["complete"]
    assert not payload["certified"]
    assert not payload["statistics_validity"]["certified"]
    assert payload["settings"]["interface_topology"] == "deduplicated_qss"
    assert payload["settings"]["adaptive_full_jv_metrics"] == []
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


def test_cli_rejects_adaptive_full_jv_in_short_circuit_mode():
    module = _load_cli_module()

    with pytest.raises(ValueError, match="incompatible with short-circuit-only"):
        module.main(
            [
                "--out",
                "/tmp/not-written.json",
                "--short-circuit-only",
                "--adaptive-full-jv-metrics",
                "FF",
            ]
        )


def test_cli_requires_both_grid_ladders_for_adaptive_full_jv():
    module = _load_cli_module()

    with pytest.raises(ValueError, match="requires --voltage-grid-ladder"):
        module.main(
            [
                "--out",
                "/tmp/not-written.json",
                "--grid-ladder",
                "20",
                "30",
                "40",
                "--adaptive-full-jv-metrics",
                "FF",
                "PCE",
            ]
        )
    with pytest.raises(ValueError, match="three unique --grid-ladder"):
        module.main(
            [
                "--out",
                "/tmp/not-written.json",
                "--voltage-grid-ladder",
                "5",
                "9",
                "17",
                "--adaptive-full-jv-metrics",
                "FF",
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


def test_cli_requires_shifted_point_local_voltage_refinement_ladder():
    module = _load_cli_module()

    with pytest.raises(ValueError, match=r"base \[a,b,c\]"):
        module.main(
            [
                "--out",
                "/tmp/not-written.json",
                "--voltage-grid-ladder",
                "5",
                "9",
                "17",
                "--voltage-refinement-grid-ladder",
                "5",
                "17",
                "65",
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


def test_cli_adaptive_full_jv_certifies_every_spatial_grid(
    monkeypatch,
    tmp_path,
):
    module = _load_cli_module()

    @dataclass(frozen=True)
    class FakeStack:
        het_recomb_despike: float = 0.0

    @dataclass(frozen=True)
    class FakeCertificate:
        certified: bool = True
        metric: str = ""

    solve_calls = []
    grid_metrics = []
    voltage_grids = []
    seed = SimpleNamespace()

    monkeypatch.setattr(module, "load_scaps_yaml", lambda path: FakeStack())
    monkeypatch.setattr(module, "electrical_layers", lambda stack: (object(),))
    monkeypatch.setattr(
        module,
        "_build_scan_grid",
        lambda stack, N_grid, topology: np.linspace(0.0, 1.0, N_grid + 1),
    )

    def fake_scan(stack, values, **kwargs):
        solve_calls.append(
            (
                kwargs["N_grid"],
                kwargs["calculate_jv_metrics"],
                kwargs["adaptive_jv_metrics"],
                tuple(
                    len(grid)
                    for grid in kwargs["voltage_refinement_grids_V"]
                ),
            )
        )
        return SimpleNamespace(
            grid_interval_count=kwargs["N_grid"],
            certified=True,
            complete=True,
            points=(
                SimpleNamespace(
                    delta_ec_eV=0.0,
                    short_circuit_state=seed,
                ),
            ),
            reference_delta_ec_eV=0.0,
            boundary_policy="fixed_contacts",
            interface_transport_model="scaps_thermionic",
            interface_topology="deduplicated_qss",
            critical_intervals=(),
            terminations=(),
        )

    def fake_summary(result):
        return {
            "schema": "solarlab.interface_cbo_scan",
            "schema_version": "1.7",
            "complete": True,
            "certified": True,
            "settings": {},
            "points": [],
            "short_circuit_trace": [],
            "critical_intervals": [],
            "metric_refinement_trace": [],
            "terminations": [],
        }

    def fake_grid_certificate(results, *, metric="Jsc", **kwargs):
        grid_metrics.append(metric)
        return FakeCertificate(metric=metric)

    def fake_voltage_certificate(result, **kwargs):
        voltage_grids.append(result.grid_interval_count)
        return FakeCertificate()

    monkeypatch.setattr(module, "solve_interface_cbo_scan", fake_scan)
    monkeypatch.setattr(module, "_summary", fake_summary)
    monkeypatch.setattr(
        module,
        "certify_cbo_statistics_validity",
        lambda result, **kwargs: FakeCertificate(),
    )
    monkeypatch.setattr(
        module,
        "certify_cbo_grid_convergence",
        fake_grid_certificate,
    )
    monkeypatch.setattr(
        module,
        "certify_cbo_voltage_grid_convergence",
        fake_voltage_certificate,
    )
    output = tmp_path / "adaptive-full-jv.json"

    exit_code = module.main(
        [
            "--out",
            str(output),
            "--delta-min",
            "0",
            "--delta-max",
            "0.4",
            "--delta-step",
            "0.4",
            "--grid-ladder",
            "10",
            "20",
            "30",
            "--voltage-grid-ladder",
            "5",
            "9",
            "17",
            "--voltage-refinement-grid-ladder",
            "9",
            "17",
            "33",
            "--adaptive-full-jv-metrics",
            "FF",
            "PCE",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert solve_calls == [
        (10, True, ("FF", "PCE"), (9, 17, 33)),
        (20, True, ("FF", "PCE"), (9, 17, 33)),
        (30, True, ("FF", "PCE"), (9, 17, 33)),
    ]
    assert grid_metrics == ["Jsc", "FF", "PCE"]
    assert voltage_grids == [10, 20, 30]
    assert payload["metric_grid_convergence"]["certified"]
    assert set(payload["metric_grid_convergence"]["certificates"]) == {
        "Jsc",
        "FF",
        "PCE",
    }
    assert payload["spatial_voltage_grid_convergence"]["certified"]
    assert len(
        payload["spatial_voltage_grid_convergence"]["certificates"]
    ) == 3
    assert payload["settings"][
        "requested_voltage_refinement_grid_point_counts"
    ] == [9, 17, 33]
