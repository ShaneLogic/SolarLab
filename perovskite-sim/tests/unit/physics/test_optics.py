"""Unit tests for the Transfer-Matrix Method (TMM) optics engine."""
import numpy as np
import pytest

from perovskite_sim.physics.optics import (
    TMMLayer,
    _transfer_matrix_stack,
    _electric_field_profile,
    tmm_absorption_profile,
    tmm_generation,
    tmm_reflectance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_single_layer(n_val=1.5, k_val=0.0, d=100e-9, n_wl=50):
    """Create a single non-absorbing layer for analytical checks."""
    wl_nm = np.linspace(400, 700, n_wl)
    wl_m = wl_nm * 1e-9
    n_arr = np.full(n_wl, n_val)
    k_arr = np.full(n_wl, k_val)
    layer = TMMLayer(d=d, n=n_arr, k=k_arr)
    return [layer], wl_m


# ---------------------------------------------------------------------------
# TMMLayer
# ---------------------------------------------------------------------------

class TestTMMLayer:
    def test_n_complex(self):
        n = np.array([2.0, 2.5])
        k = np.array([0.1, 0.2])
        layer = TMMLayer(d=100e-9, n=n, k=k)
        expected = n + 1j * k
        np.testing.assert_array_equal(layer.n_complex, expected)

    def test_frozen(self):
        layer = TMMLayer(d=100e-9, n=np.ones(5), k=np.zeros(5))
        with pytest.raises(AttributeError):
            layer.d = 200e-9


# ---------------------------------------------------------------------------
# Single-layer Fabry-Perot reflectance
# ---------------------------------------------------------------------------

class TestSingleLayerReflectance:
    """For a single dielectric slab (k=0) in air, the Fabry-Perot formula
    gives an analytical reflectance that oscillates between 0 and
    4*r^2 / (1+r^2)^2 where r = (1-n)/(1+n).
    """

    def test_reflectance_range(self):
        """R must be between 0 and the Fabry-Perot max."""
        layers, wl_m = _make_single_layer(n_val=2.0, k_val=0.0, d=200e-9)
        R = tmm_reflectance(layers, wl_m)
        r_fresnel = (1.0 - 2.0) / (1.0 + 2.0)
        R_max = 4 * r_fresnel**2 / (1 + r_fresnel**2)**2
        assert np.all(R >= -1e-10)
        assert np.all(R <= R_max + 1e-10)

    def test_reflectance_at_half_wave(self):
        """At half-wave thickness (n*d = lambda/2), R should be zero
        because the reflected beams interfere destructively.
        """
        n_val = 2.0
        d = 200e-9
        # lambda where n*d = lambda/2 → lambda = 2*n*d = 800nm
        wl_m = np.array([2 * n_val * d])  # exactly 800nm
        layer = TMMLayer(d=d, n=np.array([n_val]), k=np.array([0.0]))
        R = tmm_reflectance([layer], wl_m)
        np.testing.assert_allclose(R, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Energy conservation: R + T + A = 1
# ---------------------------------------------------------------------------

class TestEnergyConservation:
    """The total reflectance + transmittance + absorption must equal 1."""

    @pytest.fixture
    def three_layer_stack(self):
        n_wl = 100
        wl_nm = np.linspace(350, 750, n_wl)
        wl_m = wl_nm * 1e-9
        layers = [
            TMMLayer(d=50e-9,
                     n=np.full(n_wl, 2.4),
                     k=np.where(wl_nm < 380, 0.05, 0.001)),
            TMMLayer(d=400e-9,
                     n=np.full(n_wl, 2.5),
                     k=0.4 * np.exp(-(wl_nm - 400) / 150)),
            TMMLayer(d=150e-9,
                     n=np.full(n_wl, 1.8),
                     k=np.where(wl_nm < 420, 0.02, 0.001)),
        ]
        return layers, wl_m

    def test_rta_equals_one(self, three_layer_stack):
        layers, wl_m = three_layer_stack

        # R
        R = tmm_reflectance(layers, wl_m)

        # T (from transfer matrix)
        S_total, _ = _transfer_matrix_stack(layers, wl_m)
        t = 1.0 / S_total[:, 0, 0]
        T = np.abs(t) ** 2

        # A (integrate absorption profile over space)
        boundaries = np.array([0, 50e-9, 450e-9, 600e-9])
        x = np.linspace(0.5e-9, 599.5e-9, 800)
        A_profile = tmm_absorption_profile(layers, wl_m, x, boundaries)
        A_total = np.trapezoid(A_profile, x, axis=0)

        total = R + T + A_total
        np.testing.assert_allclose(total, 1.0, atol=0.03,
                                   err_msg="R+T+A should equal 1")

    def test_transparent_stack_no_absorption(self):
        """With k=0 everywhere, A should be zero and R+T=1."""
        n_wl = 30
        wl_m = np.linspace(400e-9, 700e-9, n_wl)
        layers = [
            TMMLayer(d=100e-9, n=np.full(n_wl, 2.0), k=np.zeros(n_wl)),
            TMMLayer(d=300e-9, n=np.full(n_wl, 2.5), k=np.zeros(n_wl)),
        ]
        R = tmm_reflectance(layers, wl_m)
        S_total, _ = _transfer_matrix_stack(layers, wl_m)
        T = np.abs(1.0 / S_total[:, 0, 0]) ** 2
        np.testing.assert_allclose(R + T, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Generation profile
# ---------------------------------------------------------------------------

class TestTMMGeneration:

    def test_generation_positive(self):
        """G(x) must be non-negative everywhere."""
        n_wl = 50
        wl_m = np.linspace(350e-9, 750e-9, n_wl)
        layers = [
            TMMLayer(d=400e-9,
                     n=np.full(n_wl, 2.5),
                     k=0.3 * np.exp(-(wl_m * 1e9 - 400) / 200)),
        ]
        boundaries = np.array([0.0, 400e-9])
        x = np.linspace(1e-9, 399e-9, 100)
        flux = np.full(n_wl, 3e24)
        G = tmm_generation(layers, wl_m, flux, x, boundaries)
        assert np.all(G >= 0)

    def test_generation_decays_with_depth(self):
        """For a single absorbing layer, G should generally decrease
        with depth (exponential decay modified by interference)."""
        n_wl = 100
        wl_m = np.linspace(350e-9, 750e-9, n_wl)
        # Strong absorption so decay dominates interference
        layers = [
            TMMLayer(d=500e-9,
                     n=np.full(n_wl, 2.5),
                     k=np.full(n_wl, 0.5)),
        ]
        boundaries = np.array([0.0, 500e-9])
        x = np.linspace(1e-9, 499e-9, 200)
        flux = np.full(n_wl, 3e24)
        G = tmm_generation(layers, wl_m, flux, x, boundaries)
        # Front half should have more generation than back half
        mid = len(x) // 2
        assert G[:mid].mean() > G[mid:].mean()

    def test_generation_order_of_magnitude(self):
        """With realistic-ish parameters, G_max should be ~1e27-1e29."""
        n_wl = 100
        wl_m = np.linspace(350e-9, 780e-9, n_wl)
        wl_nm = wl_m * 1e9
        layers = [
            TMMLayer(d=50e-9,
                     n=np.full(n_wl, 2.4),
                     k=np.where(wl_nm < 370, 0.05, 0.0)),
            TMMLayer(d=400e-9,
                     n=np.full(n_wl, 2.5),
                     k=np.where(wl_nm < 780, 0.5 * np.exp(-(wl_nm - 400) / 200), 0.0)),
            TMMLayer(d=200e-9,
                     n=np.full(n_wl, 1.8),
                     k=np.where(wl_nm < 420, 0.02, 0.0)),
        ]
        boundaries = np.array([0, 50e-9, 450e-9, 650e-9])
        x = np.linspace(1e-9, 649e-9, 200)
        # Realistic spectral photon flux ~ 1e24 - 1e25 per m per m^2/s
        flux = np.full(n_wl, 3e24)
        G = tmm_generation(layers, wl_m, flux, x, boundaries)
        assert G.max() > 1e20, f"G_max too low: {G.max():.2e}"
        assert G.max() < 1e30, f"G_max too high: {G.max():.2e}"


# ---------------------------------------------------------------------------
# Optical data loaders
# ---------------------------------------------------------------------------

class TestOpticalData:

    def test_load_nk_mapbi3(self):
        from perovskite_sim.data import load_nk
        wl, n, k = load_nk("MAPbI3")
        assert len(wl) > 10
        assert np.all(n > 0)
        assert np.all(k >= 0)

    def test_load_nk_interpolation(self):
        from perovskite_sim.data import load_nk
        target_wl = np.array([400.0, 500.0, 600.0, 700.0])
        wl, n, k = load_nk("MAPbI3", target_wl)
        np.testing.assert_array_equal(wl, target_wl)
        assert len(n) == 4
        assert len(k) == 4

    def test_load_am15g(self):
        from perovskite_sim.data import load_am15g
        wl, flux = load_am15g()
        assert len(wl) > 10
        assert np.all(flux > 0)
        # Integrated flux should be ~2.5e21 m^-2 s^-1
        total = np.trapezoid(flux, wl * 1e-9)
        assert 1e21 < total < 5e21, f"Integrated flux {total:.2e} out of range"

    def test_am15g_shockley_queisser_integral(self):
        """AM1.5G 300-800 nm photon flux must match authoritative ASTM G-173.

        Regression guard for a prior bug where am15g.csv was ~1.5x too high
        vs canonical ASTM G-173-03 and truncated to 300-800 nm, inflating
        J_sc across the board. See Task 7.5 in the TMM activation plan.
        """
        from perovskite_sim.data import load_am15g
        q = 1.602176634e-19
        wl_nm = np.linspace(300.0, 800.0, 501)
        _, phi = load_am15g(wl_nm)
        j_sq = q * np.trapezoid(phi, wl_nm * 1e-9)
        assert 270.0 <= j_sq <= 290.0, (
            f"AM1.5G 300-800nm integral = {j_sq:.1f} A/m^2, expected ~275. "
            "File may be out of calibration vs ASTM G-173."
        )

    def test_am15g_spot_check_500nm(self):
        """Spot-check: photon flux at 500 nm must match ASTM G-173 within 5%."""
        from perovskite_sim.data import load_am15g
        _, phi = load_am15g(np.array([500.0]))
        # ASTM G-173: E(500nm) ~= 1.545 W/m^2/nm -> Phi ~= 3.89e27
        assert 3.7e27 <= phi[0] <= 4.1e27, (
            f"Phi(500nm) = {phi[0]:.3e}, expected ~3.89e27 photons/m^2/s/m"
        )

    def test_am15g_native_range_covers_ir(self):
        """Native file must span at least 280-1200 nm for c-Si / future IR work."""
        from perovskite_sim.data import load_am15g
        wl, _ = load_am15g()
        assert wl[0] <= 280.0, f"AM1.5G lower bound {wl[0]} nm > 280 nm"
        assert wl[-1] >= 1200.0, f"AM1.5G upper bound {wl[-1]} nm < 1200 nm"

    def test_am15g_raises_outside_native_range(self):
        """load_am15g must raise on extrapolation (no silent np.interp clamping)."""
        from perovskite_sim.data import load_am15g
        with pytest.raises(ValueError, match="outside"):
            load_am15g(np.array([100.0, 500.0]))  # 100 nm below native range
        with pytest.raises(ValueError, match="outside"):
            load_am15g(np.array([500.0, 5000.0]))  # 5000 nm above native range

    def test_load_nk_missing_material(self):
        from perovskite_sim.data import load_nk
        with pytest.raises(FileNotFoundError):
            load_nk("NonExistentMaterial")

    def test_load_nk_raises_below_native_range(self):
        """load_nk must raise when any wavelength is below the CSV minimum.

        Regression guard: np.interp silently clamps to the edge value, which
        for k (imaginary index) means extending the near-edge absorption into
        a spectral region where the material was never measured -- a
        J_sc-biasing bug that has already bitten load_am15g.
        """
        from perovskite_sim.data import load_nk
        with pytest.raises(ValueError, match="outside the native range"):
            load_nk("MAPbI3", np.array([250.0, 500.0]))  # 250 nm below 300

    def test_load_nk_raises_above_native_range(self):
        """load_nk must raise when any wavelength is above the CSV maximum."""
        from perovskite_sim.data import load_nk
        with pytest.raises(ValueError, match="outside the native range"):
            load_nk("MAPbI3", np.array([500.0, 1200.0]))  # 1200 nm above 1100

    def test_load_nk_error_names_material(self):
        """The ValueError message must identify the material for actionable debugging."""
        from perovskite_sim.data import load_nk
        with pytest.raises(ValueError, match="'TiO2'"):
            load_nk("TiO2", np.array([1500.0]))

    def test_load_nk_boundary_inclusive(self):
        """Requests exactly at the native endpoints must succeed (closed interval)."""
        from perovskite_sim.data import load_nk
        wl_native, _, _ = load_nk("MAPbI3")
        lo, hi = float(wl_native[0]), float(wl_native[-1])
        wl, n, k = load_nk("MAPbI3", np.array([lo, hi]))
        assert len(n) == 2 and len(k) == 2
        assert np.all(n > 0) and np.all(k >= 0)

    def test_load_shipped_fto_csv(self):
        """FTO.csv should load via load_nk() with monotonic wavelengths in 300-1000 nm."""
        from perovskite_sim.data import load_nk

        wl, n, k = load_nk("FTO")
        assert wl.shape == n.shape == k.shape
        assert np.all(np.diff(wl) > 0), "wavelengths must be strictly increasing"
        assert wl[0] <= 305.0 and wl[-1] >= 995.0, "must cover 300-1000 nm range"
        assert np.all(n > 0.5) and np.all(n < 5.0), "FTO n in reasonable range"
        assert np.all(k >= 0.0), "k must be non-negative"

    @pytest.mark.parametrize("material", [
        "MAPbI3", "TiO2", "spiro_OMeTAD",
        "FTO", "ITO", "SnO2", "C60", "PCBM", "PEDOT_PSS", "Ag", "Au", "glass",
        "NiOx",
    ])
    def test_load_all_shipped_materials(self, material):
        """Every shipped nk CSV must load cleanly, cover 300-1000 nm, and have n>0, k>=0."""
        from perovskite_sim.data import load_nk

        wl, n, k = load_nk(material)
        assert np.all(np.diff(wl) > 0), f"{material}: wavelengths not monotonic"
        assert wl[0] <= 305.0 and wl[-1] >= 995.0, f"{material}: range too narrow"
        assert np.all(n > 0.0), f"{material}: n must be positive"
        assert np.all(k >= 0.0), f"{material}: k must be non-negative"


# ---------------------------------------------------------------------------
# Incoherent first layer (thick glass substrate)
# ---------------------------------------------------------------------------

def test_glass_substrate_incoherent_suppresses_fringes():
    """1 mm glass with incoherent=True must produce smooth R(lambda), not fringes."""
    from perovskite_sim.physics.optics import TMMLayer, tmm_reflectance
    from perovskite_sim.data import load_nk

    wl_nm = np.linspace(500.0, 502.0, 201)  # 0.01 nm spacing
    wl_m = wl_nm * 1e-9

    _, n_glass, k_glass = load_nk("glass", wl_nm)
    _, n_fto, k_fto = load_nk("FTO", wl_nm)

    glass = TMMLayer(d=1.0e-3, n=n_glass, k=k_glass, incoherent=True)
    fto = TMMLayer(d=500e-9, n=n_fto, k=k_fto)

    R = tmm_reflectance([glass, fto], wl_m)
    # Smooth: peak-to-peak variation over 2 nm window < 5% absolute
    assert (R.max() - R.min()) < 0.05, f"R variation {R.max()-R.min():.3f} suggests fringes"


def test_incoherent_not_first_raises():
    from perovskite_sim.physics.optics import TMMLayer, _transfer_matrix_stack
    wl = np.linspace(400e-9, 800e-9, 10)
    ones = np.ones(10)
    bad = [
        TMMLayer(100e-9, ones * 1.5, ones * 0.0),
        TMMLayer(1e-3, ones * 1.5, ones * 0.0, incoherent=True),
    ]
    with pytest.raises(ValueError, match="first layer"):
        _transfer_matrix_stack(bad, wl)


def test_tmm_generation_maps_back_to_electrical_grid():
    """_compute_tmm_generation must return G array sized to the electrical grid.

    TMM walks the full stack (substrate + absorber) but returns G(x) on the
    electrical grid, which only covers the non-substrate layers.
    """
    from dataclasses import replace

    from perovskite_sim.solver.mol import _compute_tmm_generation
    from perovskite_sim.models.device import (
        DeviceStack,
        LayerSpec,
        electrical_layers,
    )
    from perovskite_sim.models.parameters import MaterialParams
    from perovskite_sim.discretization.grid import multilayer_grid, Layer
    from perovskite_sim.models.config_loader import load_device_from_yaml

    real = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    absorber_params = replace(real.layers[1].params, optical_material="MAPbI3")

    glass_p = MaterialParams(
        eps_r=2.25, mu_n=0.0, mu_p=0.0, D_ion=0.0,
        P_lim=1e30, P0=0.0,
        ni=1.0, tau_n=1e-9, tau_p=1e-9, n1=1.0, p1=1.0,
        B_rad=0.0, C_n=0.0, C_p=0.0, alpha=0.0, N_A=0.0, N_D=0.0,
        optical_material="glass", incoherent=True,
    )
    stack = DeviceStack(layers=(
        LayerSpec("glass", 1.0e-3, glass_p, "substrate"),
        LayerSpec("MAPbI3", 400e-9, absorber_params, "absorber"),
    ))

    elec = electrical_layers(stack)
    layers_grid = [Layer(l.thickness, 30) for l in elec]
    x = multilayer_grid(layers_grid)
    G = _compute_tmm_generation(x, stack)

    assert G is not None
    assert G.shape == x.shape
    assert np.all(G >= 0.0)
    assert G.max() > 1e24
