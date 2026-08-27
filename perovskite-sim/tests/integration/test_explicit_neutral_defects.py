"""DEF-1 execution tests across transient and structured 1D paths."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.defects import (
    EXPLICIT_DEFECT_SCHEMA_VERSION,
    EXPLICIT_QUASI_STEADY,
    INTEGRATED_TOTAL,
    NEUTRAL,
    NEUTRAL_ALL_OCCUPANCIES,
    SINGLE_LEVEL,
    BulkDefectDistribution,
    BulkDefectDocument,
    BulkDefectKinetics,
    BulkDefectSpecies,
    ExplicitDefectCapabilityError,
)
from perovskite_sim.solver.dae import (
    build_consistent_initial_condition,
    build_no_ion_no_interface_dae,
    finite_difference_state_jacobian,
    project_algebraic_state,
)
from perovskite_sim.solver.dae_jacobian import build_structured_state_jacobian
from perovskite_sim.solver.mol import (
    StateVec,
    build_material_arrays,
    run_transient,
)
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.twod.grid_2d import build_grid_2d
from perovskite_sim.twod.microstructure import Microstructure
from perovskite_sim.twod.solver_2d import build_material_arrays_2d


ROOT = Path(__file__).resolve().parents[2]


def _neutral_species(*, name: str, center_eV: float) -> BulkDefectSpecies:
    return BulkDefectSpecies(
        name=name,
        distribution=BulkDefectDistribution(
            kind=SINGLE_LEVEL,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=2.0e20,
            center_eV_above_vb=center_eV,
        ),
        charge_transition=NEUTRAL,
        neutral_reference=NEUTRAL_ALL_OCCUPANCIES,
        kinetics=BulkDefectKinetics(
            sigma_n_m2=3.0e-19,
            sigma_p_m2=7.0e-20,
            thermal_velocity_n_m_s=1.0e5,
            thermal_velocity_p_m_s=8.0e4,
        ),
        degeneracy=1.0,
    )


def _one_layer_explicit_stack(
    species: tuple[BulkDefectSpecies, ...] | None = None,
):
    source = load_device_from_yaml("configs/csi_vannijen2025_pn_cv.yaml")
    source_layer = source.layers[1]
    selected_species = species or (
        _neutral_species(name="neutral_low", center_eV=0.35),
        _neutral_species(name="neutral_high", center_eV=0.82),
    )
    layer = replace(
        source_layer,
        params=replace(
            source_layer.params,
            defect_schema_version=EXPLICIT_DEFECT_SCHEMA_VERSION,
            defect_model=EXPLICIT_QUASI_STEADY,
            bulk_defects=selected_species,
        ),
    )
    return replace(
        source,
        layers=(layer,),
        V_bi=0.0,
        built_in_potential_mode="legacy_manual",
        interfaces=(),
        interface_defects=(),
        grid_interval_weights=(),
        grid_alphas=(),
    )


def test_short_transient_attaches_per_species_diagnostics_without_charge():
    stack = _one_layer_explicit_stack()
    grid = multilayer_grid([Layer(stack.layers[0].thickness, 8)], alpha=1.0)
    material = build_material_arrays(grid, stack)
    n = np.full(grid.shape, material.n_L)
    p = np.full(grid.shape, material.p_L)
    n[1:-1] *= np.linspace(1.02, 1.08, grid.size - 2)
    p[1:-1] *= np.linspace(1.07, 1.01, grid.size - 2)
    state = StateVec.pack(n, p, material.P_ion0.copy())

    solution = run_transient(
        grid,
        state,
        (0.0, 1.0e-10),
        np.asarray([1.0e-10]),
        stack,
        illuminated=False,
        V_app=0.0,
        mat=material,
        rtol=1.0e-7,
        atol=1.0e-3,
    )

    assert solution.success
    diagnostics = solution.explicit_bulk_defect_diagnostics
    assert diagnostics.per_species_rate_m3_s.shape == (2, grid.size)
    assert diagnostics.species_identifiers == (
        "layer[0]/n_base/neutral_low",
        "layer[0]/n_base/neutral_high",
    )
    payload = diagnostics.to_dict()
    assert payload["charge_density_C_m3"] is None
    assert payload["model_identity_sha256"] == (
        material.neutral_bulk_defects.identity_sha256
    )


def test_standard_yaml_loader_reaches_explicit_neutral_execution(tmp_path):
    raw = yaml.safe_load((ROOT / "configs/csi_vannijen2025_pn_cv.yaml").read_text())
    document = BulkDefectDocument(
        schema_version=EXPLICIT_DEFECT_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=(
            _neutral_species(name="yaml_neutral", center_eV=0.56),
        ),
    ).to_dict()
    raw["layers"][1].update(
        defect_schema_version=document["schema_version"],
        defect_model=document["defect_model"],
        bulk_defects=document["bulk_defects"],
    )
    path = tmp_path / "explicit-neutral.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    stack = load_device_from_yaml(str(path))
    grid = multilayer_grid(
        [Layer(layer.thickness, 4) for layer in stack.layers],
        alpha=1.0,
    )
    material = build_material_arrays(grid, stack)

    assert material.neutral_bulk_defects is not None
    assert material.neutral_bulk_defects.species[0].identifier.endswith(
        "/yaml_neutral"
    )


def test_explicit_neutral_qf_dark_equilibrium_is_residual_certified():
    stack = _one_layer_explicit_stack(
        (_neutral_species(name="qf_neutral", center_eV=0.56),)
    )
    grid = multilayer_grid([Layer(stack.layers[0].thickness, 10)], alpha=1.0)

    result = solve_quasi_fermi_steady_state(
        grid,
        stack,
        V_app=0.0,
        illuminated=False,
    )

    assert result.certified
    assert result.max_normalized_cell_residual < 1.0e-8
    assert result.electron_continuity_bound_A_m2 < 1.0e-8
    assert result.hole_continuity_bound_A_m2 < 1.0e-8


def test_compiled_level_references_preserve_mass_action_at_every_node():
    stack = _one_layer_explicit_stack()
    grid = multilayer_grid([Layer(stack.layers[0].thickness, 8)], alpha=1.0)
    material = build_material_arrays(grid, stack)

    for species in material.neutral_bulk_defects.species:
        active = species.active_nodes
        np.testing.assert_allclose(
            species.n1_m3[active] * species.p1_m3[active],
            material.ni_sq[active],
            rtol=2.0e-15,
            atol=0.0,
        )
        assert species.tau_n_s > 0.0
        assert species.tau_p_s > 0.0


def test_midgap_and_band_edge_reference_limits_are_finite_and_symmetric():
    gap_eV = 1.12
    stack = _one_layer_explicit_stack(
        (
            _neutral_species(name="vb", center_eV=0.0),
            _neutral_species(name="midgap", center_eV=gap_eV / 2.0),
            _neutral_species(name="cb", center_eV=gap_eV),
        )
    )
    grid = multilayer_grid([Layer(stack.layers[0].thickness, 5)], alpha=1.0)
    material = build_material_arrays(grid, stack)
    vb, midgap, cb = material.neutral_bulk_defects.species
    active = midgap.active_nodes
    intrinsic = np.sqrt(material.ni_sq[active])

    np.testing.assert_array_equal(midgap.n1_m3[active], intrinsic)
    np.testing.assert_array_equal(midgap.p1_m3[active], intrinsic)
    np.testing.assert_allclose(vb.n1_m3[active], cb.p1_m3[active], rtol=1.0e-15)
    np.testing.assert_allclose(vb.p1_m3[active], cb.n1_m3[active], rtol=1.0e-15)
    assert np.all(np.isfinite(vb.n1_m3[active]))
    assert np.all(np.isfinite(vb.p1_m3[active]))
    assert np.all(np.isfinite(cb.n1_m3[active]))
    assert np.all(np.isfinite(cb.p1_m3[active]))


def test_2d_material_build_rejects_def1_model_instead_of_falling_back():
    stack = _one_layer_explicit_stack()
    grid = build_grid_2d(
        [Layer(stack.layers[0].thickness, 5)],
        lateral_length=200.0e-9,
        Nx=3,
        lateral_uniform=True,
    )

    with pytest.raises(ExplicitDefectCapabilityError, match="1D-only"):
        build_material_arrays_2d(grid, stack, Microstructure())


def test_explicit_neutral_structured_jacobian_matches_independent_central_reference():
    stack = _one_layer_explicit_stack()
    grid = multilayer_grid([Layer(stack.layers[0].thickness, 8)], alpha=1.0)
    reference = solve_equilibrium(grid, stack)
    model = build_no_ion_no_interface_dae(
        grid,
        stack,
        reference,
        V_app_V=0.01,
        reference_time_s=1.0e-8,
    )
    initial = build_consistent_initial_condition(model)
    coordinate = np.array(initial.coordinate, copy=True)
    count = grid.size
    coordinate[1 : count - 1] += np.linspace(-0.03, 0.05, count - 2)
    coordinate[count + 1 : 2 * count - 1] += np.linspace(
        0.04, -0.02, count - 2
    )
    coordinate = project_algebraic_state(model, coordinate)

    structured = build_structured_state_jacobian(
        model,
        coordinate,
        initial.derivative,
    )
    central = finite_difference_state_jacobian(
        model,
        coordinate,
        initial.derivative,
        relative_step=2.0e-6,
    )
    analytic = structured.matrix.toarray()
    difference = np.abs(analytic - central)
    column_scale = np.maximum(np.max(np.abs(central), axis=0), 1.0e-12)

    assert model.material.neutral_bulk_defects is not None
    assert float(np.max(difference / column_scale)) < 2.0e-5
    np.testing.assert_allclose(analytic, central, rtol=3.0e-5, atol=2.0e-8)
