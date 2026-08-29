from __future__ import annotations

from pathlib import Path

import numpy as np

from perovskite_sim.experiments.dynamic_defect_transient import (
    build_dynamic_defect_transient_protocol,
    run_dynamic_defect_transient,
)
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    build_two_sided_trace_grid,
)
from perovskite_sim.models.config_loader import load_device_from_yaml


ROOT = Path(__file__).resolve().parents[2]


def test_public_dynamic_defect_transient_is_protocol_bound_and_certified():
    stack = load_device_from_yaml(
        ROOT / "configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"
    )
    grid = build_two_sided_trace_grid(build_electrical_grid(stack, 4), stack)
    times = (0.0, 1.0e-8, 1.0e-6, 1.0e-4)
    voltage = (0.0, 0.05, 0.05, 0.05)
    protocol = build_dynamic_defect_transient_protocol(
        stack,
        grid,
        times,
        voltage,
        requested_grid_intervals=4,
    )

    result = run_dynamic_defect_transient(grid, stack, protocol)

    assert result.evidence.certified
    assert result.evidence.reasons == ()
    assert result.evidence.protocol == protocol
    assert result.evidence.protocol_sha256 == protocol.protocol_hash
    assert result.evidence.engine_certificate.certified
    assert result.evidence.dc_operating_point_certified
    assert result.evidence.dark_reference_certified
    assert result.evidence.microscopic_binding_certified
    assert result.evidence.public_projection_certified
    assert result.terminal_total_current_A_m2.shape == (4,)
    assert result.total_current_faces_A_m2.shape[0] == 4
    assert result.interface_occupancy.shape == (4, 1)
    assert result.positive_ion_centroid_shift_m.shape == (4,)
    assert result.electron_density_m3.shape == result.positive_ion_density_m3.shape
    assert np.max(np.abs(result.interface_occupancy_change)) > 1.0e-9
    assert result.evidence.maximum_positive_ion_relative_motion > 1.0e-5
    assert np.all(np.isfinite(result.terminal_total_current_A_m2))
    assert not result.grid_m.flags.writeable
    assert not result.interface_occupancy.flags.writeable
