from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim._compat.numpy_compat import trapezoid
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.physics import beer_lambert_spectral as bl
from perovskite_sim.physics.generation import dual_cell_integral
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver import mol


@pytest.mark.parametrize("count", [5, 13, 31, 127])
def test_spectral_budget_matches_analytic_slab_with_substrate(monkeypatch, count):
    wavelengths = np.linspace(300, 1000, 200) * 1e-9
    flux = np.full(200, 2e27)
    substrate_alpha, absorber_alpha = 2e3, 8e6
    substrate_d, absorber_d = 1e-3, 800e-9
    optical = SimpleNamespace(
        layers=[SimpleNamespace(k=a * wavelengths / (4*np.pi), d=d) for a, d in [(substrate_alpha, substrate_d), (absorber_alpha, absorber_d)]],
        boundaries_m=np.array([0, substrate_d, substrate_d+absorber_d]),
    )
    stack = SimpleNamespace(layers=[SimpleNamespace(role="substrate", thickness=substrate_d)])
    monkeypatch.setattr(bl, "has_wavelength_resolved_optics", lambda _: True)
    monkeypatch.setattr(bl, "build_device_optical_stack", lambda *_: optical)
    monkeypatch.setattr(bl, "load_am15g", lambda _: (None, flux))
    x = np.linspace(0, absorber_d, count)
    generation = bl.spectral_beer_lambert_generation(x, stack)
    expected = trapezoid(flux, wavelengths) * np.exp(-substrate_alpha*substrate_d) * -np.expm1(-absorber_alpha*absorber_d)
    assert np.all(generation >= 0)
    assert dual_cell_integral(x, generation) == pytest.approx(expected, rel=1e-11)


def test_selected_beer_lambert_never_calls_tmm(monkeypatch):
    stack = load_scaps_yaml("configs/scaps_mirror_v2.yaml")
    monkeypatch.setattr(mol, "_compute_tmm_generation", lambda *_a, **_k: pytest.fail("TMM called for BL"))
    x = build_electrical_grid(stack, 30)
    mat = mol.build_material_arrays(x, stack)
    assert stack.optical_model == "beer_lambert_spectral"
    assert dual_cell_integral(x, mat.G_optical) > 0
    assert not mat.has_radiative_reabsorption


def test_optical_model_is_explicit_validated_and_defaults_to_scalar_bl():
    stack = load_scaps_yaml("configs/scaps_mirror_v2.yaml")
    with pytest.raises(ValueError, match="optical_model"):
        replace(stack, optical_model="TMM-typo")
    scalar = replace(stack, optical_model="beer_lambert")
    assert mol.build_material_arrays(build_electrical_grid(scalar, 30), scalar).G_optical is None


def test_real_stack_budget_does_not_change_with_electrical_mesh():
    stack = load_scaps_yaml("configs/scaps_mirror_v2.yaml")
    budgets = []
    for count in (15, 30, 60, 120):
        x = build_electrical_grid(stack, count)
        budgets.append(dual_cell_integral(x, bl.spectral_beer_lambert_generation(x, stack)))
    np.testing.assert_allclose(budgets, budgets[0], rtol=1e-13)


def test_real_optical_changes_have_distinct_semantic_hashes():
    from perovskite_sim.reproducibility import semantic_sha256

    stack = load_scaps_yaml("configs/scaps_mirror_v2.yaml")
    assert len({semantic_sha256(replace(stack, optical_model=model)) for model in ("tmm", "beer_lambert", "beer_lambert_spectral")}) == 3
