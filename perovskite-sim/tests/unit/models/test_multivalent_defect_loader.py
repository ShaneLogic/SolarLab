"""D7-E1 v4 layer parsing and MaterialParams document dispatch."""

from __future__ import annotations

import math

import pytest

from perovskite_sim.models.config_loader import material_params_from_dict
from perovskite_sim.models.defects import (
    EXPLICIT_DEFECT_SCHEMA_VERSION,
    EXPLICIT_QUASI_STEADY,
    ExplicitDefectSchemaError,
    BulkDefectKinetics,
)
from perovskite_sim.models.multivalent_defects import (
    MULTIVALENT_DEFECT_SCHEMA_VERSION,
    MultivalentBulkDefectDocument,
    MultivalentBulkDefectSpecies,
    multivalent_bulk_defect_document_from_layer_mapping,
)
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.physics.temperature import thermal_voltage


GAP_EV = 0.80
NC_M3 = 1.0e24
NV_M3 = 8.0e23


def _species_dict() -> dict:
    return {
        "name": "double_donor_bulk",
        "total_density_m3": 2.0e21,
        "configuration": {
            "family": "double_donor",
            "charge_states_e": [2, 1, 0],
            "degeneracy_convention": "unity",
            "state_degeneracies": [1.0, 1.0, 1.0],
            "energy_levels": {
                "first_transition_eV_above_vb": 0.30,
                "correlation_energies_eV": [0.15],
                "energy_reference": "above_valence_band",
            },
            "transition_kinetics": [
                {
                    "sigma_n_m2": 2.0e-19,
                    "sigma_p_m2": 7.0e-20,
                    "thermal_velocity_n_m_s": 1.0e5,
                    "thermal_velocity_p_m_s": 8.0e4,
                },
                {
                    "sigma_n_m2": 1.0e-19,
                    "sigma_p_m2": 5.0e-20,
                    "thermal_velocity_n_m_s": 1.0e5,
                    "thermal_velocity_p_m_s": 8.0e4,
                },
            ],
        },
    }


def _layer_dict(**overrides) -> dict:
    intrinsic = math.sqrt(NC_M3 * NV_M3 * math.exp(-GAP_EV / thermal_voltage(300.0)))
    layer = {
        "eps_r": 20.0,
        "mu_n": 2.0e-3,
        "mu_p": 2.0e-3,
        "D_ion": 0.0,
        "P_lim": 1.0e30,
        "P0": 0.0,
        "ni": intrinsic,
        "tau_n": 1.0e-6,
        "tau_p": 1.0e-6,
        "n1": intrinsic,
        "p1": intrinsic,
        "B_rad": 0.0,
        "C_n": 0.0,
        "C_p": 0.0,
        "alpha": 4.0e5,
        "N_A": 0.0,
        "N_D": 0.0,
        "chi": 4.0,
        "Eg": GAP_EV,
        "Nc300": NC_M3,
        "Nv300": NV_M3,
        "defect_schema_version": MULTIVALENT_DEFECT_SCHEMA_VERSION,
        "defect_model": EXPLICIT_QUASI_STEADY,
        "bulk_defects": [_species_dict()],
    }
    layer.update(overrides)
    return layer


def _species() -> MultivalentBulkDefectSpecies:
    return MultivalentBulkDefectSpecies.from_dict(_species_dict())


def test_layer_parser_builds_the_canonical_v4_document():
    params = material_params_from_dict(_layer_dict())

    assert params.defect_schema_version == MULTIVALENT_DEFECT_SCHEMA_VERSION
    assert params.defect_model == EXPLICIT_QUASI_STEADY
    assert len(params.bulk_defects) == 1
    assert isinstance(params.bulk_defects[0], MultivalentBulkDefectSpecies)
    document = params.defect_document
    assert isinstance(document, MultivalentBulkDefectDocument)
    assert (
        document.sha256
        == MultivalentBulkDefectDocument.from_dict(
            {
                "schema_version": MULTIVALENT_DEFECT_SCHEMA_VERSION,
                "defect_model": EXPLICIT_QUASI_STEADY,
                "bulk_defects": [_species_dict()],
            }
        ).sha256
    )


def test_layer_parser_requires_the_complete_v4_key_set():
    incomplete = _layer_dict()
    del incomplete["defect_model"]
    with pytest.raises(ExplicitDefectSchemaError, match="together"):
        material_params_from_dict(incomplete)

    with pytest.raises(
        ExplicitDefectSchemaError,
        match="canonical v4 schema",
    ):
        multivalent_bulk_defect_document_from_layer_mapping(
            _layer_dict(defect_schema_version=EXPLICIT_DEFECT_SCHEMA_VERSION)
        )
    assert multivalent_bulk_defect_document_from_layer_mapping({"eps_r": 1.0}) is None


def test_v4_and_legacy_species_do_not_cross_dispatch():
    with pytest.raises(ExplicitDefectSchemaError):
        material_params_from_dict(
            _layer_dict(defect_schema_version=EXPLICIT_DEFECT_SCHEMA_VERSION)
        )

    with pytest.raises(
        ExplicitDefectSchemaError,
        match="multivalent species",
    ):
        MaterialParams(
            eps_r=20.0,
            mu_n=2.0e-3,
            mu_p=2.0e-3,
            D_ion=0.0,
            P_lim=1.0e30,
            P0=0.0,
            ni=1.0e16,
            tau_n=1.0e-6,
            tau_p=1.0e-6,
            n1=1.0e16,
            p1=1.0e16,
            B_rad=0.0,
            C_n=0.0,
            C_p=0.0,
            alpha=0.0,
            N_A=0.0,
            N_D=0.0,
            Eg=GAP_EV,
            defect_schema_version=MULTIVALENT_DEFECT_SCHEMA_VERSION,
            defect_model=EXPLICIT_QUASI_STEADY,
            bulk_defects=(),
        )


def test_material_params_validates_v4_levels_against_the_band_gap():
    with pytest.raises(
        ExplicitDefectSchemaError,
        match="inside the band gap",
    ):
        MaterialParams(
            eps_r=20.0,
            mu_n=2.0e-3,
            mu_p=2.0e-3,
            D_ion=0.0,
            P_lim=1.0e30,
            P0=0.0,
            ni=1.0e16,
            tau_n=1.0e-6,
            tau_p=1.0e-6,
            n1=1.0e16,
            p1=1.0e16,
            B_rad=0.0,
            C_n=0.0,
            C_p=0.0,
            alpha=0.0,
            N_A=0.0,
            N_D=0.0,
            Eg=0.40,
            defect_schema_version=MULTIVALENT_DEFECT_SCHEMA_VERSION,
            defect_model=EXPLICIT_QUASI_STEADY,
            bulk_defects=(_species(),),
        )


def test_material_params_defect_document_property_round_trips_v4():
    params = material_params_from_dict(_layer_dict())
    document = params.defect_document
    rebuilt = MultivalentBulkDefectDocument.from_dict(document.to_dict())
    assert rebuilt.sha256 == document.sha256
    assert rebuilt.bulk_defects[0].configuration.transition_kinetics[
        0
    ] == BulkDefectKinetics(
        sigma_n_m2=2.0e-19,
        sigma_p_m2=7.0e-20,
        thermal_velocity_n_m_s=1.0e5,
        thermal_velocity_p_m_s=8.0e4,
    )


def test_backend_inline_device_parses_v4_then_fails_closed_on_execution():
    """The UI/backend parser shares this loader, so v4 must reach the gate.

    A v4 layer must not be silently degraded to an effective-lifetime device
    by the inline path; it parses into the canonical document and then the
    ordinary (non-QF/DC) material build refuses to execute it.
    """
    from backend.main import stack_from_dict
    from perovskite_sim.discretization.grid import Layer, multilayer_grid
    from perovskite_sim.models.defects import ExplicitDefectCapabilityError
    from perovskite_sim.solver.mol import build_material_arrays

    layer = _layer_dict()
    layer.update({"name": "defective", "thickness": 300.0e-9, "role": "absorber"})
    stack = stack_from_dict(
        {
            "layers": [layer],
            "V_bi": 0.0,
            "Phi": 0.0,
            "mode": "legacy",
            "built_in_potential_mode": "semiconductor_work_function",
        }
    )
    params = stack.layers[0].params
    assert params.defect_schema_version == MULTIVALENT_DEFECT_SCHEMA_VERSION
    assert isinstance(params.bulk_defects[0], MultivalentBulkDefectSpecies)

    with pytest.raises(ExplicitDefectCapabilityError, match="multivalent"):
        build_material_arrays(
            multilayer_grid([Layer(300.0e-9, 8)]),
            stack,
        )


def test_default_layers_carry_no_multivalent_document():
    layer = _layer_dict()
    del layer["defect_schema_version"]
    del layer["defect_model"]
    del layer["bulk_defects"]
    params = material_params_from_dict(layer)
    assert params.defect_schema_version is None
    assert params.defect_document is None
    assert params.bulk_defects == ()
