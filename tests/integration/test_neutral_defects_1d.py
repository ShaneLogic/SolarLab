"""Actual compiler/equation binding; native witness is read, never rerun."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from examples.research_1d import neutral_defects as example
from examples.research_1d.steady import load_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import CapabilityError, CompiledDevice1D, DC1DProblem
from solarlab.materials.source import SourceDocument
from solarlab.mesh import RIGHT_LAYER_WINS
from solarlab.physics.recombination.reaction import BulkRecombinationReaction


@pytest.fixture(scope="module")
def prepared() -> PreparedDevice:
    return example.prepare_case()


@pytest.fixture(scope="module")
def device(prepared: PreparedDevice) -> CompiledDevice1D:
    return compile_device_1d(prepared, intervals_per_layer=example.INTERVALS)


def edited(prepared: PreparedDevice, document: dict[str, Any]) -> PreparedDevice:
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    source = SourceDocument("neutral_binding_test", raw)
    sources = tuple(item for item in prepared.sources if item.id != source.id) + (source,)
    return resolve_device(DeviceInput.model_validate(document), prepared.defaults, prepared.resources, sources=sources)


def test_owned_nodes_replace_only_srh_and_preserve_prepared_input(
    prepared: PreparedDevice, device: CompiledDevice1D,
) -> None:
    original = prepared.to_mapping()
    again = compile_device_1d(prepared, intervals_per_layer=example.INTERVALS)
    assert original == prepared.to_mapping() and device.identity == again.identity
    model = device.recombination.explicit_defects
    assert model is not None and device.geometry.material_owner_convention == RIGHT_LAYER_WINS
    np.testing.assert_array_equal(np.flatnonzero(model.owned_nodes), np.arange(12, 24))
    assert model.owned_nodes[12] and not model.owned_nodes[24]
    assert model.species[0].identifier == "absorber/neutral_1"
    assert model.layer_document_sha256 == (("absorber", model.species[0].document_sha256),)
    assert not model.owned_nodes.flags.writeable
    n, p = np.full(37, 1e20), np.full(37, 2e20)
    actual = device.recombination.evaluate(n, p)
    lifetime = BulkRecombinationReaction(device.recombination.coefficients).evaluate(n, p)
    assert actual.explicit is not None
    np.testing.assert_array_equal(actual.srh.rate[model.owned_nodes], actual.explicit.total_rate_m3_s[model.owned_nodes])
    np.testing.assert_array_equal(actual.srh.rate[~model.owned_nodes], lifetime.srh.rate[~model.owned_nodes])
    for channel in ("radiative", "auger"):
        np.testing.assert_array_equal(getattr(actual, channel).rate, getattr(lifetime, channel).rate)
    assert np.any(actual.srh.rate[model.owned_nodes] != lifetime.srh.rate[model.owned_nodes])
    assert np.all(actual.explicit.total_rate_m3_s[~model.owned_nodes] == 0.0)


def test_nonzero_rate_exact_partials_and_physical_temperature_binding(prepared: PreparedDevice) -> None:
    document = prepared.to_input().editing_data()
    document["settings"]["T"] = 330.0
    device = compile_device_1d(edited(prepared, document), intervals_per_layer=example.INTERVALS)
    model = device.recombination.explicit_defects
    assert model is not None
    species = model.species[0]
    mask = species.active_nodes
    ni_sq = device.parameters.intrinsic_product_m6
    gap = device.parameters.band_structure.physical_band_gap_eV
    expected_n1 = np.sqrt(ni_sq[mask]) * np.exp((0.75 - 0.5 * gap[mask]) / device.thermal_voltage_V)
    np.testing.assert_array_equal(species.n1_m3[mask], expected_n1)
    n, p = np.full(37, 1e20), np.full(37, 2e20)
    actual = device.recombination.evaluate(n, p).explicit
    assert actual is not None
    denominator = species.tau_p_s * (n + species.n1_m3) + species.tau_n_s * (p + species.p1_m3)
    numerator = n * p - ni_sq
    expected = numerator / denominator
    np.testing.assert_allclose(actual.total_rate_m3_s[mask], expected[mask], rtol=2e-15)
    np.testing.assert_allclose(actual.total_d_rate_dn_s1[mask],
        ((p * denominator - numerator * species.tau_p_s) / denominator**2)[mask], rtol=2e-15)
    np.testing.assert_allclose(actual.total_d_rate_dp_s1[mask],
        ((n * denominator - numerator * species.tau_n_s) / denominator**2)[mask], rtol=2e-15)
    for carrier, derivative in (("n", actual.total_d_rate_dn_s1), ("p", actual.total_d_rate_dp_s1)):
        delta = 1e-6
        high = device.recombination.evaluate(n * (1 + delta) if carrier == "n" else n,
                                            p * (1 + delta) if carrier == "p" else p).explicit
        low = device.recombination.evaluate(n * (1 - delta) if carrier == "n" else n,
                                           p * (1 - delta) if carrier == "p" else p).explicit
        assert high is not None and low is not None
        numerical = (high.total_rate_m3_s - low.total_rate_m3_s) / (2 * delta * (n if carrier == "n" else p))
        np.testing.assert_allclose(numerical[mask], derivative[mask], rtol=5e-9)
    assert np.min(actual.total_rate_m3_s[mask]) > 0.0


def test_literal_energy_refuses_asymmetric_owned_dos_only(
    prepared: PreparedDevice, device: CompiledDevice1D,
) -> None:
    model = device.recombination.explicit_defects
    assert model is not None
    nc, nv = device.parameters.conduction_dos_m3, device.parameters.valence_dos_m3
    assert np.all(nc[model.owned_nodes] == nv[model.owned_nodes])
    assert np.any(nc[~model.owned_nodes] != nv[~model.owned_nodes])
    document = prepared.to_input().editing_data()
    parameters = document["layers"][1]["parameters"]
    parameters["Nv300"] *= 2.0
    # Keep Maxwell-Boltzmann mass action consistent, so the DOS scope guard
    # itself rejects the literal energy convention rather than another gate.
    for name in ("ni", "n1", "p1"):
        parameters[name] *= 2.0**0.5
    with pytest.raises(CapabilityError, match="equal physical conduction and valence DOS"):
        compile_device_1d(edited(prepared, document), intervals_per_layer=example.INTERVALS)


def test_prepared_conduction_referenced_energy_is_bound_losslessly(prepared: PreparedDevice) -> None:
    document = prepared.to_input().editing_data()
    defect = document["layers"][1]["bulk_defects"][0]
    defect["distribution"]["center_eV_above_vb"] = None
    defect["energy_level"] = {"reference": "below_conduction_band", "value_eV": 0.85}
    resolved = edited(prepared, document)
    assert resolved.layers[1].defects[0].values["distribution"]["center_eV_above_vb"] == 0.75
    reference = compile_device_1d(prepared, intervals_per_layer=example.INTERVALS).recombination.explicit_defects
    actual = compile_device_1d(resolved, intervals_per_layer=example.INTERVALS).recombination.explicit_defects
    assert reference is not None and actual is not None
    np.testing.assert_array_equal(reference.species[0].n1_m3, actual.species[0].n1_m3)
    np.testing.assert_array_equal(reference.species[0].p1_m3, actual.species[0].p1_m3)
    assert reference.layer_document_sha256 != actual.layer_document_sha256


def test_actual_dc_jacobian_and_poisson_have_no_neutral_trap_charge(device: CompiledDevice1D) -> None:
    count = device.node_count
    u = np.r_[device.reference_potential_V, np.full(count, 0.11), np.full(count, -0.07)]
    problem = DC1DProblem(device, generation_scale=0.0)
    lifetime = replace(device, recombination=BulkRecombinationReaction(device.recombination.coefficients))
    other = DC1DProblem(lifetime, generation_scale=0.0)
    _, n, p = device.fields(u)
    evaluation = device.recombination.evaluate(n, p)
    assert evaluation.explicit is not None
    assert evaluation.explicit.to_dict()["charge_density_C_m3"] is None
    np.testing.assert_array_equal(problem.recombination_rate(n, p).rate, evaluation.total.rate)
    np.testing.assert_array_equal(device.charge_ledger(n, p).space_charge_C_m3(), lifetime.charge_ledger(n, p).space_charge_C_m3())
    np.testing.assert_array_equal(problem.residual(u)[:count], other.residual(u)[:count])
    matrix = problem.jacobian(u)
    np.testing.assert_array_equal(matrix.toarray()[:count], other.jacobian(u).toarray()[:count])
    assert np.any(problem.residual(u)[count:] != other.residual(u)[count:])
    rng = np.random.default_rng(531)
    direction = np.r_[0.02 * rng.normal(size=count), rng.normal(size=2 * count)]
    step = 1e-5
    numerical = (problem.residual(u + step * direction) - problem.residual(u - step * direction)) / (2 * step)
    scale = np.maximum(abs(matrix) @ abs(direction), 1e-20)
    assert np.max(abs(numerical - matrix @ direction) / scale) < 2e-8


@pytest.mark.parametrize("zero_leg", ["n", "p", "both"])
def test_zero_capture_keeps_ownership_without_lifetime_fallback(prepared: PreparedDevice, zero_leg: str) -> None:
    document = prepared.to_input().editing_data()
    kinetics = document["layers"][1]["bulk_defects"][0]["kinetics"]
    for carrier in (("n", "p") if zero_leg == "both" else (zero_leg,)):
        kinetics[f"sigma_{carrier}_m2"] = 0.0
    device = compile_device_1d(edited(prepared, document), intervals_per_layer=example.INTERVALS)
    model = device.recombination.explicit_defects
    assert model is not None and not model.species[0].cycle_active
    value = device.recombination.evaluate(np.full(37, 1e20), np.full(37, 2e20))
    assert value.explicit is not None
    np.testing.assert_array_equal(value.srh.rate[model.owned_nodes], 0.0)
    np.testing.assert_array_equal(value.explicit.total_d_rate_dn_s1, 0.0)
    np.testing.assert_array_equal(value.explicit.total_d_rate_dp_s1, 0.0)
    assert np.all(value.total.rate[model.owned_nodes] > 0.0)  # Radiative/Auger remain.


def test_species_order_stable_ids_and_document_identity(prepared: PreparedDevice) -> None:
    document = prepared.to_input().editing_data()
    first = document["layers"][1]["bulk_defects"][0]
    second = deepcopy(first)
    second["id"] = "neutral_2"
    second["distribution"]["total_density_m3"] = 2e20
    document["layers"][1]["bulk_defects"] = (first, second)
    device = compile_device_1d(edited(prepared, document), intervals_per_layer=example.INTERVALS)
    model = device.recombination.explicit_defects
    assert model is not None
    assert [s.identifier for s in model.species] == ["absorber/neutral_1", "absorber/neutral_2"]
    value = device.recombination.evaluate(np.full(37, 1e20), np.full(37, 2e20)).explicit
    assert value is not None
    np.testing.assert_array_equal(value.total_rate_m3_s, np.sum(value.per_species_rate_m3_s, axis=0))
    document["layers"][1]["bulk_defects"][0]["kinetics"]["sigma_n_m2"] *= 1.1
    changed = compile_device_1d(edited(prepared, document), intervals_per_layer=example.INTERVALS)
    assert changed.recombination.explicit_defects is not None
    assert model.identity_sha256 != changed.recombination.explicit_defects.identity_sha256
    assert device.identity != changed.identity
    provenance = resolve_device(prepared.to_input(), prepared.defaults, prepared.resources,
        sources=(*prepared.sources, SourceDocument("independent-source", b"explicit provenance")))
    assert compile_device_1d(provenance, intervals_per_layer=example.INTERVALS).identity != compile_device_1d(prepared, intervals_per_layer=example.INTERVALS).identity


@pytest.mark.parametrize("kind", ["charged", "distributed", "spatial", "degeneracy", "interface", "multivalent", "metastable"])
def test_unsupported_declarations_refuse_before_solver(
    prepared: PreparedDevice, kind: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    document = prepared.to_input().editing_data()
    layer = document["layers"][1]
    defect = layer["bulk_defects"][0]
    kinetics = deepcopy(defect["kinetics"])
    if kind == "charged":
        defect.update(charge_transition="donor", neutral_reference="filled")
    elif kind == "distributed":
        layer["defect_schema_version"] = "solarlab-explicit-bulk-defects-v2"
        defect["distribution"].update(kind="uniform", width_eV=0.1,
            width_convention="uniform_full_width", energy_reference="above_valence_band")
    elif kind == "spatial":
        layer["defect_schema_version"] = "solarlab-explicit-bulk-defects-v3"
        defect["distribution"]["energy_reference"] = "above_valence_band"
        defect["spatial_profile"] = {"coordinate": "normalized_layer_coordinate", "interpolation": "piecewise_linear",
            "density_normalization": "layer_average_unity", "knots": [
                {"position_fraction": 0.0, "density_multiplier": 1.0}, {"position_fraction": 1.0, "density_multiplier": 1.0}]}
    elif kind == "degeneracy":
        defect["degeneracy"] = 2.0
    elif kind == "interface":
        document["interfaces"] = [{"id": "interface_trap", "left": "htl", "right": "absorber", "defect": {
            "id": "surface", "trap_depth_eV": 0.5, "energy_reference": "above_valence_band",
            "total_density_m2": 1e15, "kinetics": kinetics, "calibration_factor": 1.0, "iface_state_calibration_factor": 1.0}}]
    else:
        donor = {"family": "single_donor", "charge_states_e": [1, 0], "degeneracy_convention": "unity",
            "state_degeneracies": [1.0, 1.0], "energy_levels": {"first_transition_eV_above_vb": 0.75,
                "correlation_energies_eV": [], "energy_reference": "above_valence_band"}, "transition_kinetics": [kinetics]}
        if kind == "multivalent":
            layer["defect_schema_version"] = "solarlab-explicit-bulk-defects-v4"
            layer["bulk_defects"] = [{"id": "multi", "name": "Multivalent", "total_density_m3": 1e20, "configuration": donor}]
        else:
            acceptor = {**deepcopy(donor), "family": "single_acceptor", "charge_states_e": [0, -1]}
            layer.pop("defect_model")
            layer.pop("defect_schema_version")
            layer["bulk_defects"] = []
            layer["metastable_document"] = {"schema_version": "solarlab-metastable-bulk-defects-v1",
                "defect_model": "explicit_metastable_frozen", "metastable_defects": [{
                    "id": "meta", "name": "Metastable", "total_density_m3": 1e20,
                    "donor_configuration": donor, "acceptor_configuration": acceptor,
                    "donor_conversion_state_index": 0, "acceptor_conversion_state_index": 1,
                    "conversion_kinetics": {"transition_energy_eV_above_vb": 0.8,
                        "electron_capture_activation_eV": 0.1, "electron_emission_activation_eV": 1.7,
                        "hole_capture_activation_eV": 0.2, "hole_emission_activation_eV": 1.8,
                        "electron_capture_path": "double_electron_capture", "hole_capture_path": "double_hole_capture",
                        "capture_n_m3_s": 1e-14, "capture_p_m3_s": 1e-14, "phonon_frequency_Hz": 1e13}}]}
    unsupported = edited(prepared, document)
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("unsupported defect reached the nonlinear solver")
    monkeypatch.setattr(example, "prepare_case", lambda config: unsupported)
    monkeypatch.setattr(example, "solve_steady_state", forbidden)
    monkeypatch.setattr(sys, "argv", ["neutral_defects", "--output", str(tmp_path / "result.json")])
    with pytest.raises(CapabilityError):
        example.main()
    assert not (tmp_path / "result.json").exists()


def test_no_defect_branch_and_saved_before_edit_algebra() -> None:
    device = compile_device_1d(load_device(), intervals_per_layer=example.INTERVALS)
    assert device.recombination.explicit_defects is None
    baseline = os.environ.get("SOLARLAB_NEUTRAL_DEFECTS_BASELINE")
    if baseline is None:
        return
    directory = Path(baseline)
    arrays = np.load(directory / "NoDefectBaseline.npz")
    metadata = json.loads((directory / "Baseline.json").read_text())
    problem = DC1DProblem(device, generation_scale=0.0)
    assert device.identity == metadata["compiled_identity"]
    np.testing.assert_array_equal(problem.residual(arrays["u"]), arrays["residual"])
    np.testing.assert_array_equal(problem.jacobian(arrays["u"]).toarray(), arrays["jacobian"])
    np.testing.assert_array_equal(problem.current_density(arrays["u"]), arrays["current"])


def test_saved_actual_neutral_witness_without_resimulation() -> None:
    path = os.environ.get("SOLARLAB_NEUTRAL_DEFECTS_RESULT")
    if path is None:
        pytest.skip("bind the actual saved witness explicitly; never run an additional solve")
        return
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == os.environ["SOLARLAB_NEUTRAL_DEFECTS_RESULT_SHA256"]
    record = json.loads(raw)
    summary = record["summary"]
    assert summary["accepted"] and summary["node_count"] == 37
    assert summary["maximum_explicit_rate_m3_s"] > 0.0
    assert summary["maximum_explicit_d_rate_dn_s1"] > 0.0 and summary["maximum_explicit_d_rate_dp_s1"] > 0.0
    assert summary["trap_charge_density_C_m3"] is None
    assert all(gate["passed"] for gate in record["result"]["newton"]["gates"])
    assert summary["current_spread_A_m2"] < 1e-8
    assert summary["physical_certificate_status"] == "not_evaluated" and not summary["qualification"]
