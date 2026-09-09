"""Spatial refinement of the same physical TPV pulse and open-circuit history."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import pytest
import scipy

from perovskite_sim.experiments.tpv import run_tpv
from perovskite_sim.models.config_loader import load_device_from_yaml
from tests.integration.test_tpv_open_circuit_refinement import (
    _require_converged_waveform,
)


pytestmark = pytest.mark.slow


def test_three_spatial_levels_resolve_the_same_physical_tpv(request):
    root = Path(__file__).resolve().parents[2]
    stack = load_device_from_yaml(
        root / "tests/fixtures/configs/tpv_physical_reference.yaml"
    )
    resolutions = (44, 88, 176)
    results = []
    evidence_path = os.environ.get("SOLARLAB_F7_TPV_SPATIAL_EVIDENCE")
    if evidence_path:

        def save_evidence():
            sources = (
                "perovskite_sim/experiments/tpv.py",
                "perovskite_sim/experiments/open_circuit.py",
                "perovskite_sim/experiments/steady_state.py",
                "perovskite_sim/experiments/jv_sweep.py",
                "perovskite_sim/solver/mol.py",
                "tests/fixtures/configs/tpv_physical_reference.yaml",
                "tests/integration/test_foundation_tpv_spatial_refinement.py",
                "tests/integration/test_tpv_open_circuit_refinement.py",
            )
            payload = {
                "kind": "foundation_physical_tpv_spatial_refinement",
                "acceptance_record": "paired pytest/JUnit result",
                "expected_cases": len(resolutions),
                "captured_cases": len(results),
                "environment": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                },
                "source_sha256": {
                    name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in sources
                },
                "cases": [
                    dict(N_grid=count, **asdict(result))
                    for count, result in zip(resolutions, results)
                ],
            }

            def serialize(value):
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                raise TypeError(f"unsupported evidence value {type(value).__name__}")

            destination = Path(evidence_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(payload, default=serialize, allow_nan=False) + "\n"
            )

        request.addfinalizer(save_evidence)
    for count in resolutions:
        results.append(
            run_tpv(
                stack,
                N_grid=count,
                delta_G_frac=0.005,
                t_pulse=1e-6,
                t_decay=8e-6,
                n_points=120,
                max_step=5e-8,
                rtol=1e-6,
                voltage_atol=1e-10,
            )
        )
    assert len({result.protocol.protocol_hash for result in results}) == 1
    _require_converged_waveform(results[-2], results[-1])
