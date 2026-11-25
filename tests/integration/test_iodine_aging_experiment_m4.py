"""Integration tests for M4 iodine aging protocols, J-V probes, and RunStore I/O."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from examples.research_1d.iodine_aging import prepare_bitton_tessler_stack, run_comparison
from solarlab.contracts import ContractError
from solarlab.device.compile_iodine_1d import (
    bitton_tessler_2023_table_s1_chemistry,
    compile_iodine_device_1d,
)
from solarlab.experiments.iodine_aging import (
    IodineAgingProtocol,
    IodineProbeSpec,
    StressRecoveryStage,
    load_iodine_aging_bundle,
    record_iodine_aging_in_run_store,
    run_iodine_aging_protocol,
    save_iodine_aging_bundle,
)
from solarlab.io.run_store import RunStore


def test_multi_stage_aging_both_probes_bundle_and_run_store(tmp_path: Path) -> None:
    prepared = prepare_bitton_tessler_stack()
    chem = bitton_tessler_2023_table_s1_chemistry(
        domain_mode="absorber_blocking",
        rate_mode="smoluchowski_capture_radius",
        perovskite_ka_cm3_s=5.0e-11,
        right_escape_m_s=2.0e-2,
    )
    device = compile_iodine_device_1d(prepared, chem, intervals_per_layer=(5, 16, 5))

    protocol = IodineAgingProtocol(
        stages=(
            StressRecoveryStage(
                name="stress_1sun_0p9V",
                duration_s=30.0,
                voltage_V=0.9,
                generation_scale=1.0,
                max_step_s=10.0,
            ),
            StressRecoveryStage(
                name="dark_rest_0V",
                duration_s=50.0,
                voltage_V=0.0,
                generation_scale=0.0,
                max_step_s=25.0,
            ),
        ),
        checkpoint_times_s=(0.0, 30.0, 80.0),
        probe=IodineProbeSpec(
            mode="both",
            voltages_V=tuple(float(v) for v in np.linspace(0.0, 1.32, 13)),
            scan_rate_V_s=2.0,
            generation_scale=1.0,
        ),
    )

    result = run_iodine_aging_protocol(device, protocol)
    assert result.completed is True
    assert result.scientific_qualification is False
    assert len(result.checkpoints) == 3
    assert result.max_relative_conservation_error <= 1.0e-9

    # Verify non-destructive frozen-chemistry & dynamic J-V probes at every checkpoint
    for ckpt in result.checkpoints:
        assert ckpt.frozen_jv is not None
        assert ckpt.dynamic_jv is not None
        assert ckpt.frozen_jv.metrics.short_circuit_current_A_m2 is not None
        assert ckpt.frozen_jv.metrics.short_circuit_current_A_m2 > 50.0
        assert ckpt.frozen_jv.metrics.open_circuit_voltage_V is not None
        assert 0.5 < ckpt.frozen_jv.metrics.open_circuit_voltage_V < 1.32
        assert ckpt.dynamic_jv.hysteresis_index is not None
        assert math.isfinite(ckpt.dynamic_jv.hysteresis_index)

    # Open outgassing permanently reduced elemental iodine during light stress
    assert result.checkpoints[1].cumulative_escaped_iodine_atoms_m2 > 0.0
    assert result.checkpoints[-1].in_device_iodine_m2 < result.reference_iodine_inventory_m2

    # Save & checksum-verify bundle readback
    bundle_dir = tmp_path / "aging_bundle"
    manifest = save_iodine_aging_bundle(bundle_dir, result)
    assert manifest["scientific_qualification"] is False
    with pytest.raises(FileExistsError):
        save_iodine_aging_bundle(bundle_dir, result)

    loaded = load_iodine_aging_bundle(bundle_dir)
    np.testing.assert_allclose(
        loaded["arrays"]["checkpoint_times_s"], np.array([0.0, 30.0, 80.0])
    )
    # Tampering with arrays.npz must fail SHA-256 verification
    (bundle_dir / "arrays.npz").write_bytes(b"corrupted")
    with pytest.raises(ContractError):
        load_iodine_aging_bundle(bundle_dir)

    # SQLite RunStore lifecycle persistence & verification
    store_root = tmp_path / "run_store"
    with RunStore(store_root) as store:
        run_rec = record_iodine_aging_in_run_store(
            store, "iodine_aging_m4_test", device, protocol, result
        )
        assert run_rec["state"] == "succeeded"
        attempts = store.attempts("iodine_aging_m4_test")
        assert len(attempts) == 1 and attempts[0]["state"] == "succeeded"


def test_research_1d_iodine_aging_example_workflow(tmp_path: Path) -> None:
    out_dir = tmp_path / "example_out"
    summary = run_comparison(out_dir)
    assert summary["scientific_qualification"] is False
    closed_info = summary["closed"]
    open_info = summary["open"]
    assert isinstance(closed_info, dict) and isinstance(open_info, dict)
    assert float(closed_info["final_escaped_iodine_m2"]) == 0.0
    assert float(open_info["final_escaped_iodine_m2"]) > 0.0
    assert float(open_info["final_in_device_iodine_m2"]) < float(
        closed_info["final_in_device_iodine_m2"]
    )
