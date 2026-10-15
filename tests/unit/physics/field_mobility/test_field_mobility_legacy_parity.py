"""M-class parity: migrated field-dependent mobility against the legacy engine."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from perovskite_sim.physics import field_mobility as legacy
from perovskite_sim.physics.regularization import compact_sqrt_abs as legacy_compact_sqrt_abs
from solarlab.physics.transport import field_mobility as new


def _same(a, b) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _faces(seed=3, n=257):
    rng = np.random.default_rng(seed)
    E = np.concatenate([
        rng.normal(0.0, 1e6, n), [0.0, -0.0, 1e-300, -1e-12, 5e4, -5e4, 1e9, -1e9, 7.1e10],
    ])
    m = E.size
    mu0 = 10.0 ** rng.uniform(-6.0, -2.0, m)
    mu0[:5] = 0.0
    v_sat = np.where(rng.random(m) < 0.3, 0.0, 10.0 ** rng.uniform(2.0, 6.0, m))
    beta = rng.choice([0.0, 0.5, 1.0, 2.0, 1.7, -1.0], m)
    gamma = np.where(rng.random(m) < 0.3, 0.0, rng.choice([-1.0, 1.0], m) * 10.0 ** rng.uniform(-5.0, -2.0, m))
    return mu0, E, v_sat, beta, gamma


@pytest.mark.parametrize("width", [0.0, 1e2, 3e5])
def test_compact_sqrt_abs_is_bitwise_legacy(width):
    _, E, *_ = _faces()
    assert _same(new.compact_sqrt_abs(E, width), legacy_compact_sqrt_abs(E, width))
    with pytest.raises(ValueError) as old_error:
        legacy_compact_sqrt_abs(E, -1.0)
    with pytest.raises(ValueError) as new_error:
        new.compact_sqrt_abs(E, -1.0)
    assert str(new_error.value) == str(old_error.value)


@pytest.mark.parametrize("width", [0.0, 2e4])
def test_laws_and_composition_are_bitwise_legacy(width):
    mu0, E, v_sat, beta, gamma = _faces()
    assert _same(new.caughey_thomas(mu0, E, v_sat, beta), legacy.caughey_thomas(mu0, E, v_sat, beta))
    assert _same(
        new.poole_frenkel(mu0, E, gamma, field_regularization_width_V_m=width),
        legacy.poole_frenkel(mu0, E, gamma, field_regularization_width_V_m=width),
    )
    assert _same(
        new.apply_field_mobility(mu0, E, v_sat, beta, gamma, pf_field_regularization_width_V_m=width),
        legacy.apply_field_mobility(mu0, E, v_sat, beta, gamma, pf_field_regularization_width_V_m=width),
    )


@pytest.mark.parametrize("width", [0.0, 2e4])
def test_linearization_is_bitwise_legacy(width):
    mu0, E, v_sat, beta, gamma = _faces(seed=8)
    beta = np.abs(beta)  # the legacy tangent check rejects nothing here, but keep CT active-or-off
    mine = new.linearize_field_mobility(mu0, E, v_sat, beta, gamma, pf_field_regularization_width_V_m=width)
    old = legacy.linearize_field_mobility(mu0, E, v_sat, beta, gamma, pf_field_regularization_width_V_m=width)
    assert _same(mine.mobility_m2_V_s, old.mobility_m2_V_s)
    assert _same(mine.field_derivative_m3_V2_s, old.field_derivative_m3_V2_s)
    assert _same(mine.differentiable, old.differentiable)


def test_linearization_rejections_match_legacy():
    for args in (
        (np.array([-1.0]), np.array([1.0]), 0.0, 2.0, 0.0),
        (np.array([1.0]), np.array([np.nan]), 0.0, 2.0, 0.0),
        (np.ones(3), np.ones(2), 0.0, 2.0, 0.0),
    ):
        with pytest.raises(ValueError) as old_error:
            legacy.linearize_field_mobility(*args)
        with pytest.raises(ValueError) as new_error:
            new.linearize_field_mobility(*args)
        assert str(new_error.value) == str(old_error.value)


def test_face_diffusivity_and_face_parameters_are_bitwise_the_legacy_rhs():
    """``mol.build_material_arrays`` face means and the ``assemble_rhs`` recompute."""

    import field_mobility_test_support as support
    from perovskite_sim.models.device import electrical_layers
    from perovskite_sim.solver.mol import _layer_node_masks

    stack = support.load_stack("field_mobility_demo.yaml")
    x = support.electrical_grid(stack, 60)
    mat = support.legacy_material_arrays(x, stack)
    assert mat.has_field_mobility
    layers = electrical_layers(stack)
    masks = _layer_node_masks(x, layers)
    nodes = {name: np.zeros(x.size) for name in ("v_n", "v_p", "b_n", "b_p", "g_n", "g_p")}
    for layer, mask in zip(layers, masks):
        p = layer.params
        for key, value in (("v_n", p.v_sat_n), ("v_p", p.v_sat_p), ("b_n", p.ct_beta_n), ("b_p", p.ct_beta_p),
                           ("g_n", p.pf_gamma_n), ("g_p", p.pf_gamma_p)):
            nodes[key][mask] = value
    electrons = new.FieldMobilityFaceParameters.from_nodes(nodes["v_n"], nodes["b_n"], nodes["g_n"])
    holes = new.FieldMobilityFaceParameters.from_nodes(nodes["v_p"], nodes["b_p"], nodes["g_p"])
    assert _same(electrons.saturation_velocity_m_s, mat.v_sat_n_face)
    assert _same(electrons.ct_exponent, mat.ct_beta_n_face)
    assert _same(electrons.pf_gamma_sqrt_m_V, mat.pf_gamma_n_face)
    assert _same(holes.saturation_velocity_m_s, mat.v_sat_p_face)
    assert electrons.opted_in or holes.opted_in

    phi = np.linspace(0.0, 1.1, x.size) ** 2
    E_face = new.face_electric_field(phi, x)
    assert _same(E_face, -(phi[1:] - phi[:-1]) / np.diff(x))
    for width in (0.0, 5e3):
        mine = new.field_dependent_face_diffusivity(
            mat.D_n_face, mat.V_T_device, E_face, electrons, pf_field_regularization_width_V_m=width
        )
        old = legacy.apply_field_mobility(
            mat.D_n_face / mat.V_T_device, E_face, mat.v_sat_n_face, mat.ct_beta_n_face, mat.pf_gamma_n_face,
            pf_field_regularization_width_V_m=width,
        ) * mat.V_T_device
        assert _same(mine, old)


def test_legacy_rhs_face_diffusivity_matches_on_the_live_material_build():
    import field_mobility_test_support as support
    from perovskite_sim.solver.mol import build_material_arrays

    stack = support.load_stack("field_mobility_demo.yaml")
    x = support.electrical_grid(stack, 40)
    mat = build_material_arrays(x, stack)
    stack_off = dataclasses.replace(stack, mode="legacy")
    assert not build_material_arrays(x, stack_off).has_field_mobility
    parameters = new.FieldMobilityFaceParameters(mat.v_sat_p_face, mat.ct_beta_p_face, mat.pf_gamma_p_face)
    E_face = np.linspace(-3e7, 3e7, x.size - 1)
    old = legacy.apply_field_mobility(
        mat.D_p_face / mat.V_T_device, E_face, mat.v_sat_p_face, mat.ct_beta_p_face, mat.pf_gamma_p_face
    ) * mat.V_T_device
    assert _same(new.field_dependent_face_diffusivity(mat.D_p_face, mat.V_T_device, E_face, parameters), old)
