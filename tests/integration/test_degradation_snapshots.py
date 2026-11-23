"""Integration tests for P05-04a frozen-ion J-V snapshots & coupled degradation trajectories."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.research_1d.ionic_multilayer import (  # noqa: E402
    prepare_confined_dual_ion_device,
    prepare_confined_ion_device,
)
from examples.research_1d.shared_site_ionic_fixture import (  # noqa: E402
    prepare_confined_shared_site_ion_device,
)
from solarlab.config.resolve_device import resolve_device  # noqa: E402
from solarlab.device.inputs import DeviceInput  # noqa: E402
from solarlab.device.resolved import PreparedDevice  # noqa: E402
from solarlab.experiments.degradation import (  # noqa: E402
    DegradationDamageProtocol,
    DegradationProtocol,
    freeze_prepared_ions,
    run_degradation,
)


@pytest.fixture(scope="module", autouse=True)
def single_thread_blas() -> object:
    with threadpool_limits(limits=1):
        yield


def _with_fast_diffusivities(
    prepared: PreparedDevice,
    *,
    d_pos: float,
    d_neg: float | None = None,
) -> PreparedDevice:
    editing = prepared.to_input().editing_data()
    absorber = next(lyr for lyr in editing["layers"] if lyr["role"] == "absorber")
    absorber["parameters"]["D_ion"] = d_pos
    if d_neg is not None:
        absorber["parameters"]["D_ion_neg"] = d_neg
    return resolve_device(
        DeviceInput.model_validate(editing),
        prepared.defaults,
        prepared.resources,
        sources=prepared.sources,
    )


@pytest.mark.parametrize(
    "species_mode",
    ["single_species", "dual_independent", "dual_shared_site"],
)
def test_frozen_ion_snapshots_preserve_exact_profiles_and_zero_rebind_across_all_steric_modes(
    species_mode: str,
) -> None:
    if species_mode == "single_species":
        dev = _with_fast_diffusivities(prepare_confined_ion_device(), d_pos=1e-13)
    elif species_mode == "dual_independent":
        dev = prepare_confined_dual_ion_device(
            positive_diffusivity_m2_s=1e-13,
            negative_diffusivity_m2_s=5e-14,
        )
    else:
        dev = prepare_confined_shared_site_ion_device(
            positive_diffusivity_m2_s=1e-13,
            negative_diffusivity_m2_s=5e-14,
        )

    frozen = freeze_prepared_ions(dev)
    for lyr_live, lyr_froz in zip(dev.layers, frozen.layers, strict=True):
        p_live = dict(lyr_live.parameters)
        p_froz = dict(lyr_froz.parameters)
        assert p_froz["D_ion"] == 0.0
        assert p_froz["D_ion_neg"] == 0.0
        assert p_froz["P0"] == p_live["P0"]
        assert p_froz["P0_neg"] == p_live["P0_neg"]
        assert p_froz["P_lim"] == p_live["P_lim"]
        assert p_froz["P_lim_neg"] == p_live["P_lim_neg"]

    prot = DegradationProtocol(
        V_bias_V=0.85,
        generation_scale=1.0,
        precondition_time_s=1e-3,
        dt_max_s=0.05,
        n_jv_points=17,
        V_max_V=1.24,
        intervals_per_layer=8,
    )
    res = run_degradation(dev, [0.0, 0.05, 0.15], prot)

    # 1. Zero irreversible damage -> zero material rebinds, single compiled identity
    assert res.material_rebind_count == 0
    assert len(set(res.snapshot_live_compiled_identities)) == 1
    assert len(set(res.snapshot_frozen_compiled_identities)) == 1
    np.testing.assert_array_equal(res.damage, np.zeros(3))

    # 2. Every snapshot certifies 0-ULP frozen-ion invariance & identically zero ion flux
    for snap in res.snapshots:
        assert snap.frozen_profile_exact is True
        assert snap.max_ion_flux_m2_s == 0.0
        assert snap.max_current_spread_A_m2 <= prot.max_current_error_A_m2
        assert snap.metrics.open_circuit_voltage_V is not None
        assert snap.metrics.efficiency is not None

    # 3. Live stress between snapshots moves mobile ions away from t=0 profile
    pos_delta = float(
        np.max(
            np.abs(
                res.snapshots[-1].positive_ion_density_m3
                - res.snapshots[0].positive_ion_density_m3
            )
        )
    )
    assert pos_delta > 1e19
    if species_mode != "single_species":
        neg_delta = float(
            np.max(
                np.abs(
                    res.snapshots[-1].negative_ion_density_m3
                    - res.snapshots[0].negative_ion_density_m3
                )
            )
        )
        assert neg_delta > 1e19


def test_coupled_irreversible_lifetime_damage_rebinds_absorber_and_degrades_voc_and_pce() -> None:
    dev = _with_fast_diffusivities(prepare_confined_ion_device(), d_pos=2e-13)
    prot = DegradationProtocol(
        V_bias_V=0.6,
        generation_scale=1.0,
        precondition_time_s=1e-3,
        dt_max_s=0.05,
        n_jv_points=21,
        V_max_V=1.24,
        intervals_per_layer=8,
        damage_protocol=DegradationDamageProtocol(
            motion_gain_m2_C=80.0,
            segregation_gain_s_1=4.0,
            lifetime_degradation_alpha=3.5,
            damage_rebind_threshold=0.0,
        ),
    )
    res = run_degradation(dev, [0.0, 0.1, 0.25], prot)

    assert res.damage[0] == 0.0
    assert res.damage[1] > 0.0
    assert res.damage[2] > res.damage[1]
    assert res.material_rebind_count >= 2
    assert len(set(res.snapshot_live_compiled_identities)) == 3
    assert len(set(res.snapshot_frozen_compiled_identities)) == 3

    # Reduced SRH lifetime monotonically degrades V_oc and PCE
    assert res.V_oc_V[2] < res.V_oc_V[0]
    assert res.PCE[2] < res.PCE[0]
