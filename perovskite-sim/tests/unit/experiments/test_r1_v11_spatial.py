"""Pair spatial projection retains low words before response subtraction."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from perovskite_sim.experiments import one_dimensional_mechanism_r1_spatial as spatial
from perovskite_sim.experiments import one_dimensional_mechanism_r1_pair_spatial as pair
from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
from perovskite_sim.experiments import one_dimensional_mechanism_r1_pair_codec as codec
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.compensated import DD


@pytest.fixture(scope="module")
def small_pair():
    project = Path(__file__).resolve().parents[3]
    stack = load_device_from_yaml(project / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
    binding = json.loads((project / "tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json").read_text())
    with threadpool_limits(1):
        prepared = states.prepare_common_state(stack, 16, binding, backend="pair")
        record = protocol.run_r1_step(stack, 16, binding, prepared, times_s=[0., 1e-9], backend="pair")
    return prepared.to_dict(), states.json_data(record)


@pytest.mark.parametrize("carrier", ["electron", "hole"])
def test_sg_projection_has_original_zero_field_limit_and_uses_low_words(carrier):
    density = DD([1., 1.], [1e-20, 3e-20])
    actual = pair.sample_sg_pair([0., 1.], density, DD([0., 0.]), [.25, .75], .025, carrier=carrier)
    difference = actual-DD([1., 1.])
    np.testing.assert_allclose(difference.to_float(), [1.5e-20, 2.5e-20], rtol=1e-14)
    rounded = spatial.sample_sg_density([0., 1.], density.to_float(), [0., 0.], [.25, .75], .025, carrier=carrier)
    assert np.array_equal(rounded, [1., 1.])


def test_conservative_moments_consume_low_words_and_respect_breaks():
    profile = pair.PairIonProfile([0., 1., 2.], DD([1., 1.], [1e-20, -1e-20]), (1,))
    mass, first = profile.moments(0., 2.)
    assert bool(mass == DD(2.))
    assert float((first-DD(2.)).to_float()) == pytest.approx(-1e-20)
    assert np.array_equal(profile.slope.to_float(), [0., 0.])
    centroid = pair.centroids_from_profile([0., 1., 2.], profile.density, ((0, 1),), (1,))
    assert float((centroid[0]-1.).to_float()) == pytest.approx(-5e-21)


def test_real_pair_projection_is_finite_and_original_legacy_path_is_unchanged(small_pair):
    prepared, record = small_pair
    adapted = spatial.spatial_responses_from_step(prepared, record)
    assert adapted.metadata["representation"] == codec.REPRESENTATION
    assert adapted.metadata["rounding_boundary"] == "after_pair_projection_and_response_difference"
    assert set(adapted.responses) == {"response_potential", "carrier_log_density", "ion_density_over_p0",
                                     "trap_occupancy_change", "ion_centroid_change"}
    assert all(np.isfinite(item.values).all() for item in adapted.responses.values())
    a = spatial.sample_r1_state(prepared)
    b = pair.sample_pair_state(prepared)
    for key, value in b.items():
        np.testing.assert_array_equal(a[key], value.to_float() if isinstance(value, DD) else value)
    legacy = spatial.sample_r1_state(prepared["seed_preparation"])
    assert legacy["n_m3"].shape == a["n_m3"].shape


@pytest.mark.parametrize("field_name", ["phi_V", "n_m3", "positive_m3", "occupancy"])
def test_low_only_response_survives_identical_high_words(small_pair, field_name):
    prepared, record = deepcopy(small_pair)
    base = prepared["state"]
    for row in record["accepted_steps"]:
        row["state"] = deepcopy(base)
        if row["time_s"]:
            high = np.asarray(row["state"]["precision_"+field_name+"_hi"])
            low = np.asarray(row["state"]["precision_"+field_name+"_lo"])
            delta = np.where(high != 0, np.abs(high)*1e-20, 0.)
            changed = DD(high, low) + DD(delta)
            assert np.array_equal(changed.hi, high)
            row["state"]["precision_"+field_name+"_lo"] = changed.lo.tolist()
    record["sha256"] = states.digest({key: value for key, value in record.items() if key != "sha256"})
    responses = spatial.spatial_responses_from_step(prepared, record)
    key = {"phi_V": "response_potential", "n_m3": "carrier_log_density",
           "positive_m3": "ion_density_over_p0", "occupancy": "trap_occupancy_change"}[field_name]
    assert np.max(np.abs(responses.responses[key].values[1])) > 0
    assert np.max(np.abs(responses.responses[key].values[0])) == 0


@pytest.mark.parametrize("change", ["missing_low", "representation", "digest", "duplicate_time", "legacy_prepared"])
def test_pair_projection_refuses_loss_or_mixed_identity(small_pair, change):
    prepared, record = deepcopy(small_pair)
    if change == "missing_low":
        record["accepted_steps"][-1]["state"].pop("precision_n_m3_lo")
    elif change == "representation":
        record["representation"] = "float64-baseline"
    elif change == "digest":
        record["sha256"] = "0"*64
    elif change == "duplicate_time":
        record["accepted_steps"].append(deepcopy(record["accepted_steps"][-1]))
    else:
        prepared = prepared["seed_preparation"]
    if change != "digest":
        record["sha256"] = states.digest({key: value for key, value in record.items() if key != "sha256"})
    with pytest.raises(ValueError):
        spatial.spatial_responses_from_step(prepared, record)


def test_standalone_sampler_rejects_pair_words_with_legacy_preparation(small_pair):
    prepared, record = small_pair
    with pytest.raises(ValueError, match="legacy spatial"):
        spatial.sample_r1_state(prepared["seed_preparation"], record["accepted_steps"][0]["state"])


def test_n256_dependency_identity_includes_low_words_and_baseline_is_inapplicable(small_pair):
    from scripts import run_r1_v11_cases as runner
    from scripts.analyze_r1_v11_responses import preparation_identity
    prepared, _ = small_pair
    required = {"required_pair_preparation_identity_sha256": states.physical_preparation_identity(prepared)["sha256"]}
    assert runner.validate_required_preparation(prepared, required, pair=True)["matched"]
    altered = deepcopy(prepared)
    altered["state"]["precision_n_m3_lo"][0] += 1e-25
    with pytest.raises(ValueError, match="physical preparation differs"):
        runner.validate_required_preparation(altered, required, pair=True)
    assert runner.validate_required_preparation(prepared, required, pair=False)["matched"] is None
    assert preparation_identity(prepared)["physical_identity"] == states.physical_preparation_identity(prepared)
