"""Contact thermodynamics: the fixed 5 meV gate and the historical 6.025 meV label.

V1 P04-03: the historical scaps inconsistency keeps its label and the
threshold cannot be changed to certify it. On ``scaps_mirror_v2`` (manual
``V_bi = 1.300 V`` against the band-derived 1.29398 V) the four contact
quasi-Fermi levels span exactly that offset; the assessment stays
``inconsistent`` with the label ``legacy_manual_v_bi_contact_offset``, the
gate stays 5e-3 eV and a looser caller tolerance is refused. The compatible
semiconductor work-function source on the same reservoirs is certified.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import warnings

import pytest

from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import build_material_arrays

from solarlab.physics.contacts import (
    CONTACT_THERMODYNAMIC_TOLERANCE_EV, LEGACY_MANUAL_CONTACT_OFFSET, ContactEndpoint,
    ContactThermodynamicError, assess_contact_thermodynamics,
    require_contact_thermodynamic_certificate,
)

PRESET = Path(__file__).resolve().parents[4] / "perovskite-sim/configs/scaps_mirror_v2.yaml"


def inputs(stack, **override):
    x = build_electrical_grid(stack, 24)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat = build_material_arrays(x, stack)

    def endpoint(node, n, p):
        return ContactEndpoint(float(n), float(p), float(mat.N_C_physical[node]),
                               float(mat.N_V_physical[node]), float(mat.chi_phys[node]),
                               float(mat.Eg_phys[node]))

    mode = stack.resolved_built_in_potential_mode()
    values = dict(
        built_in_potential_mode=mode, left=endpoint(0, mat.n_L, mat.p_L),
        right=endpoint(-1, mat.n_R, mat.p_R), V_bi_bc=float(mat.V_bi_bc),
        thermal_voltage_V=mat.V_T_device, statistics=mat.carrier_statistics,
        semiconductor_potential_V=stack.compute_semiconductor_V_bi(),
        metal_work_functions_eV=(stack.work_function_left_eV, stack.work_function_right_eV)
        if mode == "metal_work_function" else None,
        has_band_reference=True,
    )
    values.update(override)
    return stack, mat, values


def test_the_historical_scaps_inconsistency_keeps_its_label_and_status():
    stack, mat, values = inputs(load_scaps_yaml(str(PRESET)))
    assessment = assess_contact_thermodynamics(**values)
    assert CONTACT_THERMODYNAMIC_TOLERANCE_EV == 5.0e-3
    assert assessment.built_in_potential_mode == "legacy_manual"
    assert assessment.status == "inconsistent" and not assessment.certified
    assert assessment.historical_label == LEGACY_MANUAL_CONTACT_OFFSET
    assert assessment.tolerance_eV == 5.0e-3
    # 6.025 meV, and it is exactly the manual-vs-band contact offset (an identity).
    # The recorded value is a macOS capture; the span is quantized to ulp(5.5 eV) ~ 9e-16,
    # so compare at the identity tolerance below rather than at one ulp (libm differs).
    assert assessment.fermi_level_span_eV == pytest.approx(6.024958093130e-3, abs=1e-12)
    offset = abs(float(mat.V_bi_bc) - stack.compute_V_bi())
    assert abs(assessment.fermi_level_span_eV - offset) <= 1e-12
    levels = assessment.contact_quasi_fermi_levels_eV
    assert abs(levels[0] - levels[1]) <= 1e-12 and abs(levels[2] - levels[3]) <= 1e-12
    assert "0.00602496 eV equilibrium quasi-Fermi-level span" in assessment.message
    with pytest.raises(ContactThermodynamicError, match=LEGACY_MANUAL_CONTACT_OFFSET):
        require_contact_thermodynamic_certificate(assessment)


@pytest.mark.parametrize("tolerance", [5.0e-3 * (1 + 1e-15), 6.1e-3, 0.2, float("inf")])
def test_the_gate_cannot_be_relaxed_to_certify_it(tolerance):
    _, _, values = inputs(load_scaps_yaml(str(PRESET)), tolerance_eV=tolerance)
    with pytest.raises(ValueError, match="cannot exceed|finite and positive"):
        assess_contact_thermodynamics(**values)


def test_a_tighter_gate_keeps_it_inconsistent_and_labelled():
    _, _, values = inputs(load_scaps_yaml(str(PRESET)), tolerance_eV=1.0e-3)
    assessment = assess_contact_thermodynamics(**values)
    assert assessment.status == "inconsistent"
    assert assessment.historical_label == LEGACY_MANUAL_CONTACT_OFFSET


def test_the_compatible_source_on_the_same_reservoirs_is_certified_without_a_label():
    stack = dataclasses.replace(load_scaps_yaml(str(PRESET)),
                                built_in_potential_mode="semiconductor_work_function")
    _, _, values = inputs(stack)
    assessment = require_contact_thermodynamic_certificate(assess_contact_thermodynamics(**values))
    assert assessment.certified and assessment.historical_label is None
    assert assessment.fermi_level_span_eV < 1.0e-12


def test_other_inconsistencies_are_not_given_the_historical_label():
    stack = dataclasses.replace(load_scaps_yaml(str(PRESET)),
                                built_in_potential_mode="metal_work_function",
                                work_function_left_eV=5.2, work_function_right_eV=4.1)
    _, _, values = inputs(stack)
    assessment = assess_contact_thermodynamics(**values)
    assert assessment.status == "inconsistent" and assessment.historical_label is None
    # Missing endpoint DOS: the band-derived mismatch alone disproves compatibility.
    _, _, values = inputs(load_scaps_yaml(str(PRESET)))
    values["left"] = dataclasses.replace(values["left"], conduction_dos_m3=None)
    partial = assess_contact_thermodynamics(**values)
    assert partial.status == "inconsistent" and partial.fermi_level_span_eV is None
    assert partial.historical_label is None
    values["semiconductor_potential_V"] = values["V_bi_bc"]
    assert assess_contact_thermodynamics(**values).status == "compatible_unverified"
    values["has_band_reference"] = False
    assert assess_contact_thermodynamics(**values).status == "not_assessable"
