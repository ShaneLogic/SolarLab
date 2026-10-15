"""M class: identities of the field-dependent mobility composition."""

from __future__ import annotations

import numpy as np

from solarlab.physics.transport import field_mobility as fm


def test_zero_field_is_the_exact_identity():
    mu0 = np.array([2e-4, 1e-6, 3e-3])
    zero = np.zeros(3)
    for width in (0.0, 1e4):
        result = fm.apply_field_mobility(mu0, zero, np.array([1e5, 0.0, 3e4]), np.array([2.0, 1.0, 0.5]),
                                         np.array([3e-4, -1e-4, 0.0]), pf_field_regularization_width_V_m=width)
        assert np.array_equal(result, mu0)


def test_disabled_laws_return_the_low_field_mobility_bitwise():
    rng = np.random.default_rng(1)
    mu0 = 10.0 ** rng.uniform(-6, -2, 50)
    E = rng.normal(0.0, 1e7, 50)
    assert np.array_equal(fm.apply_field_mobility(mu0, E, 0.0, 2.0, 0.0), mu0)
    assert np.array_equal(fm.caughey_thomas(mu0, E, 1e5, 0.0), mu0)  # beta <= 0 disables CT


def test_poole_frenkel_precedes_caughey_thomas():
    mu0 = np.array([2e-4])
    E = np.array([4e6])
    v_sat, beta, gamma = np.array([1e3]), np.array([2.0]), np.array([3e-4])
    composed = fm.apply_field_mobility(mu0, E, v_sat, beta, gamma)
    pf_then_ct = fm.caughey_thomas(fm.poole_frenkel(mu0, E, gamma), E, v_sat, beta)
    ct_then_pf = fm.poole_frenkel(fm.caughey_thomas(mu0, E, v_sat, beta), E, gamma)
    assert np.array_equal(composed, pf_then_ct)
    assert not np.allclose(composed, ct_then_pf, rtol=1e-3)
    assert fm.COMPOSITION_ORDER == ("poole_frenkel", "caughey_thomas")


def test_laws_depend_on_the_field_magnitude_only():
    mu0 = np.full(5, 2e-4)
    E = np.array([1e3, 1e5, 3e6, 7e7, 2e9])
    args = (np.full(5, 1e4), np.full(5, 1.0), np.full(5, 2e-4))
    assert np.array_equal(fm.apply_field_mobility(mu0, E, *args), fm.apply_field_mobility(mu0, -E, *args))


def test_face_evaluation_is_recomputed_and_never_cached():
    parameters = fm.FieldMobilityFaceParameters.from_nodes(np.full(6, 1e4), np.full(6, 2.0), np.full(6, 3e-4))
    D = np.full(5, 5e-6)
    first = fm.field_dependent_face_diffusivity(D, 0.0259, np.full(5, 1e6), parameters)
    other = fm.field_dependent_face_diffusivity(D, 0.0259, np.full(5, 3e7), parameters)
    again = fm.field_dependent_face_diffusivity(D, 0.0259, np.full(5, 1e6), parameters)
    assert first is not again and np.array_equal(first, again) and not np.array_equal(first, other)
    assert not hasattr(parameters, "__dict__")  # frozen, slot-free state: nothing to memoize into
    for name in ("saturation_velocity_m_s", "ct_exponent", "pf_gamma_sqrt_m_V"):
        assert not getattr(parameters, name).flags.writeable


def test_einstein_relation_is_preserved_by_the_face_recompute():
    parameters = fm.FieldMobilityFaceParameters(np.array([1e4, 0.0]), np.array([2.0, 2.0]), np.array([0.0, 3e-4]))
    D = np.array([5e-6, 2e-7])
    V_T = 0.0302
    E = np.array([2e6, -4e5])
    mobility = fm.apply_field_mobility(D / V_T, E, parameters.saturation_velocity_m_s, parameters.ct_exponent,
                                       parameters.pf_gamma_sqrt_m_V)
    assert np.array_equal(fm.field_dependent_face_diffusivity(D, V_T, E, parameters), mobility * V_T)
