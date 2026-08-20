import numpy as np

from perovskite_sim.experiments.ion_aware_dc import (
    build_ion_aware_dc_protocol,
    solve_ion_aware_dc,
)
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.models.config_loader import load_device_from_yaml


def test_ionmonger_fixed_bias_reaches_two_residual_certified_endpoints():
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    x = build_electrical_grid(stack, 30)
    protocol = build_ion_aware_dc_protocol(
        stack,
        V_dc=0.9,
        illuminated=True,
    )

    result = solve_ion_aware_dc(x, stack, protocol)
    certificate = result.state_certificate

    assert result.numerically_certified
    assert result.consecutive_certified_steps >= 2
    assert result.steps[-2].accepted_for_closure
    assert result.steps[-1].accepted_for_closure
    assert result.total_settle_time_s in protocol.settle_end_times_s
    assert result.total_settle_time_s <= protocol.settle_end_times_s[-1]
    assert certificate.carrier_area_rate_A_m2 <= (
        protocol.max_carrier_area_rate_A_m2
    )
    assert certificate.ion_area_rate_A_m2 <= protocol.max_ion_area_rate_A_m2
    assert certificate.max_ionic_face_current_A_m2 <= (
        protocol.max_ionic_face_current_A_m2
    )
    assert certificate.dc_face_current_spread_A_m2 <= (
        protocol.max_dc_face_current_spread_A_m2
    )
    assert certificate.max_ion_inventory_relative_drift <= (
        protocol.max_ion_inventory_relative_drift
    )
    assert 0.0 < certificate.maximum_site_occupancy_fraction < 1.0
    assert np.isfinite(certificate.dc_current_density_A_m2)
    assert not result.thermodynamically_certified
    assert certificate.contact_thermodynamics.status == "compatible_unverified"
    assert not result.certified
