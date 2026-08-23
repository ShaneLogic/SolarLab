from __future__ import annotations

import numpy as np

from perovskite_sim.experiments.ion_aware_dc import (
    build_ion_aware_dc_protocol,
    solve_ion_aware_dc,
)
from perovskite_sim.experiments.ion_aware_impedance import (
    build_ion_aware_impedance_protocol,
    run_ion_aware_impedance,
)
from perovskite_sim.experiments.ion_aware_impedance_lockin import (
    build_ion_aware_transient_lockin_protocol,
    run_ion_aware_transient_lockin_crosscheck,
)
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.models.config_loader import load_device_from_yaml


def test_exact_dc_transient_lockin_matches_frequency_domain_ionic_points():
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    x = build_electrical_grid(stack, 12)
    dc_state = solve_ion_aware_dc(
        x,
        stack,
        build_ion_aware_dc_protocol(
            stack,
            V_dc=0.9,
            illuminated=True,
        ),
    )
    frequencies = np.logspace(-6, 1, 29)
    frequency_protocol = build_ion_aware_impedance_protocol(
        dc_state,
        frequencies,
    )
    frequency_domain = run_ion_aware_impedance(
        x,
        stack,
        frequency_protocol,
        dc_state=dc_state,
    )
    lockin_protocol = build_ion_aware_transient_lockin_protocol(
        frequency_domain,
        np.array([1.0e-3, 1.0e-2, 1.0]),
        cycles=4,
        extraction_cycles=2,
        points_per_cycle_levels=(20, 40, 80),
        rtol=1.0e-5,
    )

    result = run_ion_aware_transient_lockin_crosscheck(
        frequency_domain,
        lockin_protocol,
        dc_state=dc_state,
    )

    certificate = result.certificate
    assert certificate.numerically_certified
    assert certificate.frequency_domain_agreement_certified
    assert certificate.time_resolution_certified
    assert certificate.periodicity_certified
    assert certificate.inventory_certified
    assert certificate.frequency_window_certified
    assert not certificate.thermodynamically_certified
    assert not certificate.certified
    assert certificate.numerical_reasons == ()
    assert certificate.reasons == ("contact_thermodynamics_not_certified",)
    assert certificate.max_frequency_domain_magnitude_relative_difference < 0.01
    assert certificate.max_frequency_domain_phase_difference_deg < 0.01
    assert certificate.max_time_resolution_magnitude_relative_change < 1.0e-3
    assert certificate.max_time_resolution_phase_change_deg < 1.0e-3
    assert certificate.max_cycle_current_relative_change < 2.0e-3
    assert certificate.max_state_periodicity_relative_change < 1.0e-4
    assert certificate.max_ion_inventory_relative_drift < 1.0e-12
    assert tuple(
        level.points_per_cycle for level in result.resolution_evidence
    ) == (20, 40, 80)
    assert all(
        len(level.frequency_evidence) == 3
        for level in result.resolution_evidence
    )
