import numpy as np
import pytest

from perovskite_sim.physics.tandem_optics import partition_absorption


def test_compute_tandem_generation_happy_path(monkeypatch):
    """End-to-end test with mocked load_nk and a synthetic TandemConfig."""
    import perovskite_sim.physics.tandem_optics as tandem_optics_mod
    from perovskite_sim.physics.tandem_optics import compute_tandem_generation, TandemGeneration
    from perovskite_sim.models.tandem_config import JunctionLayer

    # Stub load_nk: return constant n=2.0, k=0.1 (weakly absorbing) across all wavelengths.
    def fake_load_nk(material, wavelengths_nm):
        n = np.full_like(wavelengths_nm, 2.0, dtype=float)
        k = np.full_like(wavelengths_nm, 0.1, dtype=float)
        return wavelengths_nm, n, k
    monkeypatch.setattr(tandem_optics_mod, "load_nk", fake_load_nk)

    from perovskite_sim.models.config_loader import load_device_from_yaml
    top_cell = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    bottom_cell = load_device_from_yaml("configs/nip_MAPbI3.yaml")

    from perovskite_sim.models.tandem_config import TandemConfig
    cfg = TandemConfig(
        top_cell=top_cell,
        bottom_cell=bottom_cell,
        junction_stack=(
            JunctionLayer(name="recomb", thickness=20e-9,
                          optical_material="stub", incoherent=False),
        ),
        junction_model="ideal_ohmic",
        light_direction="top_first",
        benchmark=None,
    )

    wavelengths_nm = np.linspace(400.0, 800.0, 20)
    wavelengths_m = wavelengths_nm * 1e-9
    spectral_flux = np.full_like(wavelengths_m, 1e21)

    # Pass the grids the sub-cell solves will actually integrate on, exactly as
    # run_tandem_jv does. Node COUNTS are not enough: G is a spatial density.
    from perovskite_sim.experiments.jv_sweep import build_electrical_grid
    x_top = build_electrical_grid(top_cell, 30)
    x_bot = build_electrical_grid(bottom_cell, 30)

    gen = compute_tandem_generation(
        cfg, wavelengths_m, spectral_flux, wavelengths_nm,
        x_top=x_top, x_bot=x_bot,
    )

    assert isinstance(gen, TandemGeneration)
    assert gen.G_top.shape == (len(x_top),)
    assert gen.G_bot.shape == (len(x_bot),)
    assert np.all(gen.G_top >= 0)
    assert np.all(gen.G_bot >= 0)
    assert 0.0 <= gen.parasitic_absorption <= 1.0


def test_compute_tandem_generation_empty_junction_stack(monkeypatch):
    """When junction_stack is empty, grid has no phantom junction points."""
    import perovskite_sim.physics.tandem_optics as tandem_optics_mod
    from perovskite_sim.physics.tandem_optics import compute_tandem_generation

    def fake_load_nk(material, wavelengths_nm):
        n = np.full_like(wavelengths_nm, 2.0, dtype=float)
        k = np.full_like(wavelengths_nm, 0.05, dtype=float)
        return wavelengths_nm, n, k
    monkeypatch.setattr(tandem_optics_mod, "load_nk", fake_load_nk)

    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.models.tandem_config import TandemConfig

    top_cell = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    bottom_cell = load_device_from_yaml("configs/nip_MAPbI3.yaml")

    cfg = TandemConfig(
        top_cell=top_cell,
        bottom_cell=bottom_cell,
        junction_stack=(),
        junction_model="ideal_ohmic",
        light_direction="top_first",
        benchmark=None,
    )

    wavelengths_nm = np.linspace(400.0, 800.0, 15)
    wavelengths_m = wavelengths_nm * 1e-9
    spectral_flux = np.full_like(wavelengths_m, 1e21)

    from perovskite_sim.experiments.jv_sweep import build_electrical_grid
    x_top = build_electrical_grid(top_cell, 25)
    x_bot = build_electrical_grid(bottom_cell, 25)

    gen = compute_tandem_generation(
        cfg, wavelengths_m, spectral_flux, wavelengths_nm,
        x_top=x_top, x_bot=x_bot,
    )

    n_top, n_bot = len(x_top), len(x_bot)
    assert gen.G_top.shape == (n_top,)
    assert gen.G_bot.shape == (n_bot,)
    # Empty junction → parasitic_absorption should be ~0 (only numerical noise)
    assert gen.parasitic_absorption < 1e-6
    assert gen.top_layer_slice == slice(0, n_top)
    assert gen.bottom_layer_slice == slice(n_top, n_top + n_bot)


def test_tandem_generation_offsets_electrical_grids_after_substrate_prefixes(
    monkeypatch,
):
    """Combined-TMM coordinates must not map carrier grids into substrates."""
    import perovskite_sim.physics.tandem_optics as tandem_optics_mod
    from perovskite_sim.experiments.jv_sweep import build_electrical_grid
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.models.device import DeviceStack, LayerSpec
    from perovskite_sim.models.tandem_config import TandemConfig
    from perovskite_sim.physics.generation import dual_cell_faces
    from perovskite_sim.physics.tandem_optics import compute_tandem_generation

    reference = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    top_substrate = LayerSpec(
        "top_glass", 2.0e-6, reference.layers[0].params, "substrate"
    )
    bot_substrate = LayerSpec(
        "bottom_optical_prefix",
        3.0e-6,
        reference.layers[0].params,
        "substrate",
    )
    top_cell = DeviceStack(
        layers=(top_substrate, *reference.layers),
        V_bi=reference.V_bi,
    )
    bottom_cell = DeviceStack(
        layers=(bot_substrate, *reference.layers),
        V_bi=reference.V_bi,
    )
    cfg = TandemConfig(
        top_cell=top_cell,
        bottom_cell=bottom_cell,
        junction_stack=(),
        junction_model="ideal_ohmic",
        light_direction="top_first",
        benchmark=None,
    )

    def fake_load_nk(material, wavelengths_nm):
        return (
            wavelengths_nm,
            np.full_like(wavelengths_nm, 2.0, dtype=float),
            np.full_like(wavelengths_nm, 0.05, dtype=float),
        )

    queried_faces = []

    def record_absorbed(
        layers,
        wavelengths,
        spectral_flux,
        faces,
        layer_boundaries,
        n_ambient=1.0,
        n_substrate=1.0,
    ):
        queried_faces.append(np.asarray(faces, dtype=float).copy())
        return np.ones(len(faces) - 1, dtype=float)

    monkeypatch.setattr(tandem_optics_mod, "load_nk", fake_load_nk)
    monkeypatch.setattr(
        tandem_optics_mod,
        "tmm_absorbed_photon_flux_per_cell",
        record_absorbed,
    )

    x_top = build_electrical_grid(top_cell, 30)
    x_bot = build_electrical_grid(bottom_cell, 30)
    wavelengths_nm = np.linspace(400.0, 800.0, 9)
    wavelengths_m = wavelengths_nm * 1.0e-9
    spectral_flux = np.full_like(wavelengths_m, 1.0e21)
    compute_tandem_generation(
        cfg,
        wavelengths_m,
        spectral_flux,
        wavelengths_nm,
        x_top=x_top,
        x_bot=x_bot,
    )

    top_faces, bot_faces, whole_stack_faces = queried_faces
    top_offset = top_substrate.thickness
    top_full_thickness = sum(layer.thickness for layer in top_cell.layers)
    bot_offset = top_full_thickness + bot_substrate.thickness

    assert top_faces == pytest.approx(dual_cell_faces(x_top) + top_offset)
    assert bot_faces == pytest.approx(dual_cell_faces(x_bot) + bot_offset)
    assert top_faces[0] == pytest.approx(top_substrate.thickness)
    assert bot_faces[0] == pytest.approx(bot_offset)
    assert whole_stack_faces == pytest.approx(
        [
            0.0,
            top_full_thickness
            + sum(layer.thickness for layer in bottom_cell.layers),
        ]
    )


def test_partition_assigns_layer_ranges_correctly():
    # 10 grid points, 3 wavelengths. Points 0-3: top, 4-5: junction, 6-9: bottom.
    A = np.ones((10, 3))  # uniform absorption rate [m^-1]
    x = np.linspace(0.0, 1.0, 10)
    spectral_flux = np.array([1.0, 2.0, 3.0])
    wavelengths = np.array([400e-9, 500e-9, 600e-9])
    top_slice = slice(0, 4)
    junction_slice = slice(4, 6)
    bot_slice = slice(6, 10)

    G_top, G_bot, parasitic = partition_absorption(
        A, x, wavelengths, spectral_flux,
        top_slice, junction_slice, bot_slice,
    )

    assert G_top.shape == (4,)
    assert G_bot.shape == (4,)
    assert np.all(G_top > 0)
    assert np.all(G_bot > 0)
    assert 0.0 < parasitic < 1.0


def test_partition_conserves_photon_count():
    rng = np.random.default_rng(0)
    N = 30
    n_wl = 8
    A = rng.uniform(0.0, 1.0, size=(N, n_wl))
    x = np.linspace(0.0, 500e-9, N)
    wavelengths = np.linspace(400e-9, 800e-9, n_wl)
    spectral_flux = np.full(n_wl, 1e21)

    top = slice(0, 10)
    junc = slice(10, 14)
    bot = slice(14, N)

    G_top, G_bot, parasitic = partition_absorption(
        A, x, wavelengths, spectral_flux, top, junc, bot,
    )

    top_photons = float(np.trapezoid(G_top, x[top]))
    bot_photons = float(np.trapezoid(G_bot, x[bot]))
    total_incident = float(np.trapezoid(spectral_flux, wavelengths))
    junc_photons = parasitic * total_incident

    integrand = A * spectral_flux[None, :]
    G_full = np.trapezoid(integrand, wavelengths, axis=1)
    full_photons = float(np.trapezoid(G_full, x))

    assert top_photons + junc_photons + bot_photons == pytest.approx(
        full_photons, rel=1e-9
    )
