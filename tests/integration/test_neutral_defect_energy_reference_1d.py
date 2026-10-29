"""Independent band-edge/occupancy oracle for literal neutral defect energies."""

from __future__ import annotations

from decimal import Decimal, localcontext
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from examples.research_1d import neutral_defects as example
from examples.research_1d.steady import load_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.neutral_defects import LITERAL_ABOVE_VB_DOS_REFERENCE_V1, bind_neutral_defects
from solarlab.equations.dc_1d import CapabilityError, CompiledDevice1D, DC1DProblem
from solarlab.materials.source import SourceDocument
from solarlab.physics.recombination.explicit_neutral import (
    NeutralDefectSpecies, NeutralExplicitDefectModel, compile_neutral_defect_species,
    evaluate_neutral_explicit_defects,
)
from solarlab.physics.recombination.reaction import BulkRecombinationReaction


def band_edge_oracle(
    nc: float, nv: float, gap: float, center: float, thermal: float,
    n: float = 1e13, p: float = 2e12,
) -> dict[str, float]:
    """80-digit Boltzmann band edges and stationary capture/emission balance.

    This does not use the DEF1 intrinsic-level formula or lifetime quotient.
    The neutral convention changes recombination, never the Poisson charge.
    """
    with localcontext() as context:
        context.prec = 80
        nc_d, nv_d, gap_d, center_d, thermal_d, n_d, p_d = (
            Decimal.from_float(float(value)) for value in (nc, nv, gap, center, thermal, n, p)
        )
        n1 = nc_d * ((center_d - gap_d) / thermal_d).exp()
        p1 = nv_d * (-center_d / thermal_d).exp()
        ni_sq = nc_d * nv_d * (-gap_d / thermal_d).exp()
        cn, cp, density = Decimal("1e-14"), Decimal("2e-14"), Decimal("1e20")
        denominator = cn * (n_d + n1) + cp * (p_d + p1)
        occupied = (cn * n_d + cp * p1) / denominator
        d_occupied_dn = cn * (1 - occupied) / denominator
        d_occupied_dp = -cp * occupied / denominator
        rate_n = density * cn * (n_d * (1 - occupied) - n1 * occupied)
        rate_p = density * cp * (p_d * occupied - p1 * (1 - occupied))
        d_rate_dn = density * cn * (1 - occupied - (n_d + n1) * d_occupied_dn)
        d_rate_dp = -density * cn * (n_d + n1) * d_occupied_dp
        return {name: float(value) for name, value in (
            ("n1", n1), ("p1", p1), ("ni_sq", ni_sq), ("rate_n", rate_n),
            ("rate_p", rate_p), ("d_rate_dn", d_rate_dn), ("d_rate_dp", d_rate_dp),
        )}


def intrinsic_reference_species(nc: float, nv: float, center: float, *, thermal: float = .026) -> NeutralDefectSpecies:
    gap = 1.6
    oracle = band_edge_oracle(nc, nv, gap, center, thermal)
    # Ei-Ev = Eg/2 + VT/2*ln(Nv/Nc). The legacy argument is measured
    # from its midgap reference, so map the energy without changing the law.
    mapped = center if nc == nv else center + .5 * thermal * np.log(nc / nv)
    return compile_neutral_defect_species(
        identifier="absorber/neutral_1", document_sha256="a" * 64,
        active_nodes=np.array([False, True, True, False]),
        ni_sq_m6=np.full(4, oracle["ni_sq"]), band_gap_eV=np.full(4, gap),
        thermal_voltage_V=thermal, total_density_m3=1e20, center_eV_above_vb=float(mapped),
        sigma_n_m2=1e-19, sigma_p_m2=2e-19,
        thermal_velocity_n_m_s=1e5, thermal_velocity_p_m_s=1e5,
        charge_transition="neutral", distribution="single_level", degeneracy=1.0,
    )


@pytest.mark.parametrize("nc,nv,center,thermal", [
    (2e25, 8e25, .75, .026), (8e25, 2e25, .75, .026),
    (2e25, 8e25, 0.0, .026), (8e25, 2e25, 1.6, .026),
    (2e25, 8e25, .75, .030), (2e25, 2e25, .75, .026),
])
def test_exact_intrinsic_reference_mapping_against_band_edges(
    nc: float, nv: float, center: float, thermal: float,
) -> None:
    reference = band_edge_oracle(nc, nv, 1.6, center, thermal)
    species = intrinsic_reference_species(nc, nv, center, thermal=thermal)
    mask = species.active_nodes
    np.testing.assert_allclose(species.n1_m3[mask], reference["n1"], rtol=3e-14)
    np.testing.assert_allclose(species.p1_m3[mask], reference["p1"], rtol=3e-14)
    np.testing.assert_allclose(species.n1_m3[mask] * species.p1_m3[mask], reference["ni_sq"], rtol=5e-16)
    np.testing.assert_array_equal(species.n1_m3[~mask], 0.0)
    np.testing.assert_array_equal(species.p1_m3[~mask], 0.0)


@pytest.mark.parametrize("nc,nv", [(2e25, 8e25), (8e25, 2e25)])
def test_unchanged_srh_partials_match_stationary_capture_emission(nc: float, nv: float) -> None:
    reference = band_edge_oracle(nc, nv, 1.6, .75, .026)
    species = intrinsic_reference_species(nc, nv, .75)
    model = NeutralExplicitDefectModel((species,), species.active_nodes,
                                      (("absorber", species.document_sha256),))
    result = evaluate_neutral_explicit_defects(
        np.full(4, 1e13), np.full(4, 2e12), np.full(4, reference["ni_sq"]), model,
    )
    mask = species.active_nodes
    assert reference["rate_n"] == reference["rate_p"]
    np.testing.assert_allclose(result.total_rate_m3_s[mask], reference["rate_n"], rtol=3e-14)
    np.testing.assert_allclose(result.total_d_rate_dn_s1[mask], reference["d_rate_dn"], rtol=3e-14)
    np.testing.assert_allclose(result.total_d_rate_dp_s1[mask], reference["d_rate_dp"], rtol=3e-14)
    assert result.to_dict()["charge_density_C_m3"] is None


def compiled_variant(
    *, nc_factor: float = 1.0, nv_factor: float = 2.0, temperature: float = 300.0,
    center: float = .75,
) -> CompiledDevice1D:
    base = example.prepare_case()
    document = base.to_input().editing_data()
    parameters = document["layers"][1]["parameters"]
    parameters["Nc300"] *= nc_factor
    parameters["Nv300"] *= nv_factor
    for name in ("ni", "n1", "p1"):
        parameters[name] *= math.sqrt(nc_factor * nv_factor)
    document["settings"]["T"] = temperature
    document["layers"][1]["bulk_defects"][0]["distribution"]["center_eV_above_vb"] = center
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    prepared = resolve_device(DeviceInput.model_validate(document), base.defaults, base.resources,
        sources=(*base.sources, SourceDocument("neutral_DOS_test", raw)))
    return compile_device_1d(prepared, intervals_per_layer=example.INTERVALS)


@pytest.mark.parametrize("nc_factor,nv_factor,temperature,center", [
    (1.0, 2.0, 300.0, .75), (2.0, 1.0, 330.0, .75),
    (1.0, 4.0, 300.0, 0.0), (4.0, 1.0, 300.0, 1.6),
])
def test_actual_binding_and_partials_follow_physical_band_edge_oracle(
    nc_factor: float, nv_factor: float, temperature: float, center: float,
) -> None:
    device = compiled_variant(nc_factor=nc_factor, nv_factor=nv_factor,
                              temperature=temperature, center=center)
    model = device.recombination.explicit_defects
    assert model is not None
    species = model.species[0]
    mask = model.owned_nodes
    node = int(np.flatnonzero(mask)[0])
    parameters = device.parameters
    oracle = band_edge_oracle(float(parameters.conduction_dos_m3[node]), float(parameters.valence_dos_m3[node]),
                             float(parameters.band_structure.physical_band_gap_eV[node]), center, device.thermal_voltage_V)
    np.testing.assert_allclose(species.n1_m3[mask], oracle["n1"], rtol=3e-14)
    np.testing.assert_allclose(species.p1_m3[mask], oracle["p1"], rtol=3e-14)
    np.testing.assert_allclose(species.n1_m3[mask] * species.p1_m3[mask],
                               parameters.intrinsic_product_m6[mask], rtol=1e-15)
    np.testing.assert_array_equal(species.n1_m3[~mask], 0.0)
    np.testing.assert_array_equal(species.p1_m3[~mask], 0.0)
    explicit = device.recombination.evaluate(np.full(device.node_count, 1e13), np.full(device.node_count, 2e12)).explicit
    assert explicit is not None
    np.testing.assert_allclose(explicit.total_rate_m3_s[mask], oracle["rate_n"], rtol=3e-14)
    np.testing.assert_allclose(explicit.total_d_rate_dn_s1[mask], oracle["d_rate_dn"], rtol=3e-14)
    np.testing.assert_allclose(explicit.total_d_rate_dp_s1[mask], oracle["d_rate_dp"], rtol=3e-14)


def test_dos_ratio_changes_binding_identity_without_rewriting_defect_document() -> None:
    symmetric = compiled_variant(nc_factor=1.0, nv_factor=1.0)
    asymmetric = compiled_variant(nc_factor=2.0, nv_factor=.5)
    before, after = symmetric.recombination.explicit_defects, asymmetric.recombination.explicit_defects
    assert before is not None and after is not None
    assert before.layer_document_sha256 == after.layer_document_sha256
    assert before.identity_sha256 == after.identity_sha256
    assert symmetric.identity != asymmetric.identity
    np.testing.assert_array_equal(symmetric.parameters.intrinsic_product_m6, asymmetric.parameters.intrinsic_product_m6)
    np.testing.assert_allclose(after.species[0].n1_m3, 2.0 * before.species[0].n1_m3, rtol=1e-15)
    np.testing.assert_allclose(after.species[0].p1_m3, .5 * before.species[0].p1_m3, rtol=1e-15)


def test_nodewise_reference_map_preserves_equal_entries_and_unowned_zeros() -> None:
    prepared = example.prepare_case()
    device = compile_device_1d(prepared, intervals_per_layer=example.INTERVALS)
    original = device.recombination.explicit_defects
    assert original is not None
    mask = original.owned_nodes
    factor = np.ones(device.node_count)
    factor[mask] = np.linspace(1.0, 2.0, int(mask.sum()))
    nc = device.parameters.conduction_dos_m3 * factor
    nv = device.parameters.valence_dos_m3 / factor
    masks = (np.zeros_like(mask), mask, np.zeros_like(mask))
    converted = bind_neutral_defects(prepared.layers, masks,
        ni_sq_m6=device.parameters.intrinsic_product_m6,
        band_gap_eV=device.parameters.band_structure.physical_band_gap_eV,
        thermal_voltage_V=device.thermal_voltage_V, conduction_dos_m3=nc, valence_dos_m3=nv)
    assert converted is not None
    expected_n1 = original.species[0].n1_m3 * factor
    expected_p1 = original.species[0].p1_m3 / factor
    np.testing.assert_allclose(converted.species[0].n1_m3, expected_n1, rtol=1e-15)
    np.testing.assert_allclose(converted.species[0].p1_m3, expected_p1, rtol=1e-15)
    unchanged = ~mask | (factor == 1.0)
    np.testing.assert_array_equal(converted.species[0].n1_m3[unchanged], original.species[0].n1_m3[unchanged])
    np.testing.assert_array_equal(converted.species[0].p1_m3[unchanged], original.species[0].p1_m3[unchanged])


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_owned_physical_dos_still_refused(invalid: float) -> None:
    prepared = example.prepare_case()
    device = compile_device_1d(prepared, intervals_per_layer=example.INTERVALS)
    model = device.recombination.explicit_defects
    assert model is not None
    nc = np.array(device.parameters.conduction_dos_m3, copy=True)
    nc[model.owned_nodes] = invalid
    masks = (np.zeros_like(model.owned_nodes), model.owned_nodes, np.zeros_like(model.owned_nodes))
    with pytest.raises(CapabilityError, match="finite positive physical DOS"):
        bind_neutral_defects(prepared.layers, masks, ni_sq_m6=device.parameters.intrinsic_product_m6,
            band_gap_eV=device.parameters.band_structure.physical_band_gap_eV,
            thermal_voltage_V=device.thermal_voltage_V, conduction_dos_m3=nc,
            valence_dos_m3=device.parameters.valence_dos_m3)


def test_full_unequal_dos_dc_jacobian_and_neutral_poisson() -> None:
    device = compiled_variant()
    count = device.node_count
    u = np.r_[device.reference_potential_V, np.full(count, .11), np.full(count, -.07)]
    problem = DC1DProblem(device)
    lifetime = DC1DProblem(replace(device, recombination=BulkRecombinationReaction(device.recombination.coefficients)))
    np.testing.assert_array_equal(problem.residual(u)[:count], lifetime.residual(u)[:count])
    matrix = problem.jacobian(u)
    np.testing.assert_array_equal(matrix.toarray()[:count], lifetime.jacobian(u).toarray()[:count])
    rng = np.random.default_rng(531)
    direction = np.r_[.02 * rng.normal(size=count), rng.normal(size=2 * count)]
    step = 1e-5
    numerical = (problem.residual(u + step * direction) - problem.residual(u - step * direction)) / (2.0 * step)
    scale = np.maximum(abs(matrix) @ abs(direction), 1e-20)
    assert np.max(abs(numerical - matrix @ direction) / scale) < 2e-8


def test_pre_edit_symmetric_and_no_defect_identity_and_algebra_are_bitwise() -> None:
    location = os.environ.get("SOLARLAB_NEUTRAL_DOS_BASELINE")
    if location is None:
        pytest.skip("explicitly supply the immutable pre-edit algebra baseline")
        return
    directory = Path(location)
    metadata = json.loads((directory / "Baseline.json").read_text())
    arrays = np.load(directory / "BeforeBinding.npz")
    for name, prepared in (("symmetric", example.prepare_case()), ("no_defect", load_device())):
        device = compile_device_1d(prepared, intervals_per_layer=example.INTERVALS)
        assert device.identity == metadata["branches"][name]["compiled_identity"]
        assert prepared.content_sha256 == metadata["branches"][name]["prepared_identity"]
        problem = DC1DProblem(device)
        u = arrays[name + "_u"]
        for key, value in (("residual", problem.residual(u)), ("jacobian", problem.jacobian(u).toarray()),
                            ("current", problem.current_density(u)), ("contacts", problem.contact_current_density(u))):
            np.testing.assert_array_equal(value, arrays[name + "_" + key])
        if name == "symmetric":
            model = device.recombination.explicit_defects
            assert model is not None and model.identity_sha256 == metadata["branches"][name]["model_identity"]
            np.testing.assert_array_equal(model.species[0].n1_m3, arrays["symmetric_n1"])
            np.testing.assert_array_equal(model.species[0].p1_m3, arrays["symmetric_p1"])


def test_saved_unequal_dos_steady_without_resimulation() -> None:
    location = os.environ.get("SOLARLAB_NEUTRAL_UNEQUAL_DOS_RESULT")
    if location is None:
        pytest.skip("read the one explicitly supplied native result; never solve here")
        return
    raw = Path(location).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == os.environ["SOLARLAB_NEUTRAL_UNEQUAL_DOS_RESULT_SHA256"]
    record = json.loads(raw)
    summary = record["summary"]
    assert summary["accepted"] and summary["unequal_dos_owned_node_count"] == 12
    assert summary["defect_reference_binding"] == LITERAL_ABOVE_VB_DOS_REFERENCE_V1
    assert summary["maximum_explicit_rate_m3_s"] > 0.0
    assert summary["trap_charge_density_C_m3"] is None
    assert summary["current_spread_A_m2"] < 1e-8
    assert all(gate["passed"] for gate in record["result"]["newton"]["gates"])
    physical = record["physical_reference_inputs"]
    bound = record["binding"]["species"][0]
    mask = np.asarray(bound["active_nodes"])
    nc, nv, gap = (np.asarray(physical[key]) for key in ("conduction_dos_m3", "valence_dos_m3", "band_gap_eV"))
    np.testing.assert_allclose(np.asarray(bound["n1_m3"])[mask], nc[mask] * np.exp((.75 - gap[mask]) / physical["thermal_voltage_V"]), rtol=3e-14)
    np.testing.assert_allclose(np.asarray(bound["p1_m3"])[mask], nv[mask] * np.exp(-.75 / physical["thermal_voltage_V"]), rtol=3e-14)
    assert summary["physical_certificate_status"] == "not_evaluated" and not summary["qualification"]
