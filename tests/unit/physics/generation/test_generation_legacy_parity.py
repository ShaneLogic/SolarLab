"""M-class parity: migrated generation and optics against the legacy engine.

Every comparison is bitwise (``tobytes``). Control volumes come from W3a's
named ``legacy_mol_integer_endpoint_v1`` geometry, never from a copied
``dual_cell_widths``; the legacy functions are called only as references.
"""

from __future__ import annotations

import numpy as np
import pytest

import generation_test_support as support
from perovskite_sim.data import load_am15g, load_nk
from perovskite_sim.experiments.jv_sweep import _fixed_generation_reference
from perovskite_sim.physics import generation as legacy_generation
from perovskite_sim.physics import optics as legacy_optics
from perovskite_sim.solver.mol import _compute_tmm_generation, build_material_arrays
from solarlab.mesh.geometry import half_endpoint_geometry, integer_endpoint_geometry
from solarlab.physics import generation as new

Q = 1.602176634e-19
MESHES = (10, 15, 31, 60, 100)
TMM_PRESETS = ("ionmonger_benchmark_tmm.yaml", "nip_MAPbI3_tmm.yaml", "pin_MAPbI3_tmm.yaml",
               "driftfusion_benchmark_tmm.yaml")
BL_PRESETS = ("ionmonger_benchmark.yaml", "nip_MAPbI3.yaml", "pin_MAPbI3.yaml", "driftfusion_benchmark.yaml",
              "cSi_homojunction.yaml", "cigs_baseline.yaml")


def _same(a, b) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _scaps(name):
    from perovskite_sim.scaps_compat.loader import load_scaps_yaml

    path = support.CONFIGS / name if name == "scaps_mirror.yaml" else support.LEGACY_ROOT / "configs" / name
    return load_scaps_yaml(str(path))


@pytest.fixture(scope="module")
def tmm_stacks():
    stacks = {name: support.load_stack(name) for name in TMM_PRESETS}
    stacks["scaps_mirror.yaml"] = _scaps("scaps_mirror.yaml")
    stacks["scaps_mirror_v2.yaml"] = _scaps("scaps_mirror_v2.yaml")
    return stacks


@pytest.mark.parametrize("name", TMM_PRESETS + ("scaps_mirror.yaml", "scaps_mirror_v2.yaml"))
def test_tmm_generation_is_bitwise_the_legacy_material_build(tmm_stacks, name):
    stack = tmm_stacks[name]
    binding = support.binding_from_stack(stack)
    for n_grid in MESHES:
        x = support.electrical_grid(stack, n_grid)
        profile = binding.generation_profile(integer_endpoint_geometry(x))
        assert _same(profile.source_m3_s, _compute_tmm_generation(x, stack)), (name, n_grid)
        assert profile.geometry_name == "legacy_mol_integer_endpoint_v1"


def test_graded_cigs_slices_reproduce_the_legacy_generation():
    stack = support.load_stack("cigs_graded_optics.yaml")
    binding = support.binding_from_legacy_optical_stack(stack)
    assert any(part.stop - part.start > 1 for _, part in binding.physical_layer_slices)
    for n_grid in (10, 40, 100):
        x = support.electrical_grid(stack, n_grid)
        assert _same(binding.generation_profile(integer_endpoint_geometry(x)).source_m3_s,
                     _compute_tmm_generation(x, stack))


def test_graded_slice_geometry_is_the_legacy_one():
    from perovskite_sim.physics.optical_stack import build_device_optical_stack

    stack = support.load_stack("cigs_graded_optics.yaml")
    optical = build_device_optical_stack(stack, support.WAVELENGTHS_NM)
    for layer, part in zip(stack.layers, optical.physical_layer_slices):
        model = layer.params.cigs_graded_optics
        if model is None or part.stop - part.start == 1:
            continue
        centres, thickness = new.graded_slice_centres(layer.thickness, model.slices)
        assert all(tmm.d == thickness for tmm in optical.layers[part])
        assert _same(centres, (np.arange(model.slices, dtype=float) + 0.5) * (layer.thickness / model.slices))


@pytest.mark.parametrize("name", BL_PRESETS)
def test_beer_lambert_generation_is_bitwise_legacy(name):
    stack = support.load_stack(name)
    for n_grid in MESHES:
        x = support.electrical_grid(stack, n_grid)
        mat = build_material_arrays(x, stack)
        legacy = legacy_generation.beer_lambert_generation(x, mat.alpha, stack.Phi)
        geometry = integer_endpoint_geometry(x)
        assert _same(new.beer_lambert_generation(geometry, mat.alpha, stack.Phi), legacy)
        profile = new.BeerLambertBinding(x, mat.alpha, stack.Phi).generation_profile(geometry)
        assert _same(profile.source_m3_s, legacy)


def test_scalar_absorption_beer_lambert_is_bitwise_legacy():
    for n_nodes in (2, 3, 7, 50, 401):
        x = np.linspace(0.0, 4e-7, n_nodes)
        assert _same(new.beer_lambert_generation(integer_endpoint_geometry(x), 1.3e7, 2.5e21),
                     legacy_generation.beer_lambert_generation(x, 1.3e7, 2.5e21))


def test_half_endpoint_geometry_equals_the_legacy_2d_boundary_rescale():
    """2D rows: the legacy rescale ``G1d * dx/(dx/2)`` equals ``absorbed / (dx/2)``."""

    from perovskite_sim.twod.solver_2d import _to_2d_dual_cell

    stack = support.load_stack("cSi_homojunction.yaml")
    x = support.electrical_grid(stack, 61)
    mat = build_material_arrays(x, stack)
    legacy_1d = legacy_generation.beer_lambert_generation(x, mat.alpha, stack.Phi)
    expected = _to_2d_dual_cell(legacy_1d)
    assert _same(new.beer_lambert_generation(half_endpoint_geometry(x), mat.alpha, stack.Phi), expected)


def _test_stacks():
    rng = np.random.default_rng(17)
    n_wl = 37
    wl_nm = np.linspace(320.0, 940.0, n_wl)
    coherent = [
        legacy_optics.TMMLayer(d=60e-9, n=2.0 + 0.2 * rng.random(n_wl), k=0.02 * rng.random(n_wl)),
        legacy_optics.TMMLayer(d=350e-9, n=np.full(n_wl, 2.5), k=0.5 * np.exp(-(wl_nm - 320.0) / 160.0)),
        legacy_optics.TMMLayer(d=120e-9, n=np.full(n_wl, 1.9), k=np.zeros(n_wl)),
    ]
    glass = legacy_optics.TMMLayer(d=1e-3, n=np.full(n_wl, 1.5), k=np.full(n_wl, 1e-7), incoherent=True)
    return wl_nm * 1e-9, [coherent, [glass, *coherent]]


@pytest.mark.parametrize("which", [0, 1])
def test_transfer_matrix_primitives_are_bitwise_legacy(which):
    wl_m, stacks = _test_stacks()
    old_layers = stacks[which]
    new_layers = [new.TMMLayer(d=lyr.d, n=lyr.n, k=lyr.k, incoherent=lyr.incoherent) for lyr in old_layers]
    boundaries = np.concatenate([[0.0], np.cumsum([lyr.d for lyr in old_layers])])
    x = np.linspace(boundaries[0], boundaries[-1], 233)
    flux = np.linspace(1e27, 4e27, wl_m.size)
    assert _same(new.tmm_reflectance(new_layers, wl_m), legacy_optics.tmm_reflectance(old_layers, wl_m))
    assert _same(new.tmm_absorption_profile(new_layers, wl_m, x, boundaries),
                 legacy_optics.tmm_absorption_profile(old_layers, wl_m, x, boundaries))
    assert _same(new.tmm_cumulative_absorptance(new_layers, wl_m, x, boundaries),
                 legacy_optics.tmm_cumulative_absorptance(old_layers, wl_m, x, boundaries))
    assert _same(new.tmm_absorbed_photon_flux_per_cell(new_layers, wl_m, flux, x, boundaries),
                 legacy_optics.tmm_absorbed_photon_flux_per_cell(old_layers, wl_m, flux, x, boundaries))
    point, absorbance = new.tmm_point_generation(new_layers, wl_m, flux, x, boundaries, return_absorbance=True)
    old_point, old_absorbance = legacy_optics.tmm_generation(old_layers, wl_m, flux, x, boundaries,
                                                             return_absorbance=True)
    assert _same(point, old_point) and _same(absorbance, old_absorbance)


def test_spectrum_sampling_is_bitwise_the_legacy_loader():
    table = np.loadtxt(support.DATA / "am15g.csv", delimiter=",", comments="#")
    for grid in (support.WAVELENGTHS_NM, np.linspace(280.0, 1500.0, 977), np.array([500.0, 777.7])):
        _, legacy_flux = load_am15g(grid)
        assert _same(new.sample_tabulated(table[:, 0], table[:, 1], grid, name="am15g"), legacy_flux)
        assert _same(new.SpectralPhotonFlux.from_table(table[:, 0], table[:, 1], grid, source_id="am15g").flux_m2_s_m,
                     legacy_flux)
    with pytest.raises(ValueError, match="outside the native range"):
        new.sample_tabulated(table[:, 0], table[:, 1], np.array([270.0, 500.0]), name="am15g")


@pytest.mark.parametrize("material", sorted(p.stem for p in (support.DATA / "nk").glob("*.csv")))
def test_optical_constant_sampling_is_bitwise_the_legacy_loader(material):
    table = support.nk_table(material)
    lo, hi = max(table[0, 0], 300.0), min(table[-1, 0], 1000.0)
    grid = np.linspace(lo, hi, 157)
    _, legacy_n, legacy_k = load_nk(material, grid)
    n, k = new.sample_nk_table(table[:, 0], table[:, 1], table[:, 2], grid, material=material)
    assert _same(n, legacy_n) and _same(k, legacy_k)
    with pytest.raises(ValueError, match="outside the native range"):
        new.sample_nk_table(table[:, 0], table[:, 1], table[:, 2], np.array([table[0, 0] - 1.0]), material=material)


def test_fixed_generation_provenance_is_the_legacy_string():
    rng = np.random.default_rng(2)
    values = rng.random(37) * 1e27
    assert new.fixed_generation_reference(None) == _fixed_generation_reference(None)
    assert new.fixed_generation_reference(values) == _fixed_generation_reference(values)
    assert new.FixedGenerationProfile(values, 37).source_reference == _fixed_generation_reference(values)
    with pytest.raises(ValueError, match="only finite values"):
        new.fixed_generation_reference(np.array([1.0, np.inf]))


def test_documented_photon_budgets_are_reproduced():
    """Values quoted by the legacy sources, reproduced by the new path.

    ``optics.py`` / ``mol.py`` quote 225.001 A/m^2 absorbed and 226.672 A/m^2
    for ``q Phi (1-R-T)`` on ionmonger_benchmark_tmm; CLAUDE.md quotes the
    mesh-independent 219.01 A/m^2 electrical budget of scaps_mirror; the
    Beer-Lambert ionmonger ceiling is q Phi = 224.30473 A/m^2.
    """

    stack = support.load_stack("ionmonger_benchmark_tmm.yaml")
    budget = support.binding_from_stack(stack).photon_budget()
    assert round(Q * budget.electrical_absorbed_m2_s, 3) == 225.001
    ceiling = Q * (budget.incident_m2_s - budget.reflected_m2_s - budget.transmitted_m2_s)
    assert 226.672 <= ceiling < 226.673  # quoted truncated as 226.672
    scaps = support.binding_from_stack(_scaps("scaps_mirror.yaml")).photon_budget()
    assert round(Q * scaps.electrical_absorbed_m2_s, 2) == 219.01
    bl = support.load_stack("ionmonger_benchmark.yaml")
    assert round(Q * bl.Phi, 5) == 224.30473
    for n_grid in (10, 60, 100):
        x = support.electrical_grid(bl, n_grid)
        mat = build_material_arrays(x, bl)
        areal = float(np.sum(new.beer_lambert_absorbed_photon_flux(integer_endpoint_geometry(x), mat.alpha, bl.Phi)))
        assert Q * areal < Q * bl.Phi
