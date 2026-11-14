"""Unit tests for spectral control-volume Beer-Lambert optics and SCAPS catalog."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from solarlab.config.device_import import import_standard_device
from solarlab.config.identity import (
    IdentityInputs,
    ModelInputs,
    input_identity,
    physics_identity,
)
from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.model_parameters import device_topology, layer_parameter_source
from solarlab.config.resolve_device import resolve_device
from solarlab.config.scaps_input import import_scaps_device
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.compile_multiple_ionic_1d import compile_multiple_ionic_device_1d
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab.mesh import (
    HALF_ENDPOINT_V1,
    INTEGER_ENDPOINT_V1,
    TWO_SIDED_TRACE_V1,
    build_geometry,
    multilayer_grid,
)
from solarlab.physics.generation import (
    SPECTRAL_BEER_LAMBERT_MODEL_ID,
    OpticalLayerSpec,
    OpticalStackBinding,
    SpectralBeerLambertBinding,
    SpectralPhotonFlux,
    constant_index_optical_constants,
    sample_nk_table,
    sample_tabulated,
    spectral_beer_lambert_absorption_coefficients,
    wavelength_grid_m,
)
from solarlab.physics.generation import transfer_matrix as tm_module
from solarlab.physics.statistics.thermal import thermal_voltage
from solarlab.physics.interface_transport import remove_shared_interface_nodes
from solarlab.physics.registry import (
    CapabilityContext,
    CapabilityRule,
    EvidenceRef,
    ModelDefinition,
    ModelRef,
    ModelRegistry,
    Scope,
)
from solarlab.physics.selection import ModelChoice, select_models
from solarlab.physics.semantic_projection import SemanticProjection
from solarlab.reproducibility.scaps_beer_lambert import (
    HIGH_DOPING_MIN_N_GRID,
    SCAPS_BEER_LAMBERT_STUDY_ID,
    STANDARD_JV_SWEEP_HINTS,
    SWEEPS,
    apply_reference_point,
    load_scaps_beer_lambert_catalog,
    reference_catalog,
    validate_reference_point,
)

ROOT = Path(__file__).resolve().parents[3]
LEGACY = ROOT / "perovskite-sim" / "perovskite_sim"
DATA_DIR = LEGACY / "data"
CATALOG_PATHS = (
    "models/parameters.py",
    "models/device.py",
    "models/defects.py",
    "physics/statistics.py",
    "physics/band_gap_narrowing.py",
    "scaps_compat/loader.py",
    "constants.py",
    "twod/microstructure.py",
    "models/tandem_config.py",
)


@pytest.fixture(scope="module")
def defaults() -> DefaultCatalog:
    return read_legacy_default_catalog(
        tuple(
            SourceDocument(name, (LEGACY / name).read_bytes())
            for name in CATALOG_PATHS
        )
    )


def _am15g_spectrum(n_wavelengths: int = 200) -> SpectralPhotonFlux:
    raw = np.loadtxt(DATA_DIR / "am15g.csv", delimiter=",", comments="#")
    wl_nm = np.linspace(300.0, 1000.0, n_wavelengths)
    flux = sample_tabulated(raw[:, 0], raw[:, 1], wl_nm, name="am15g")
    return SpectralPhotonFlux(wl_nm, flux, f"am15g_{n_wavelengths}")


def _am15g_resource(n_wavelengths: int = 200) -> ResourceTable:
    spec = _am15g_spectrum(n_wavelengths)
    lines = ["wavelength_nm,spectral_flux_m-2_s-1_m-1\n"]
    lines.extend(
        f"{repr(float(w))},{repr(float(f))}\n"
        for w, f in zip(spec.wavelengths_nm, spec.flux_m2_s_m)
    )
    name = f"am15g_{n_wavelengths}"
    return ResourceTable(
        name,
        "spectrum",
        SourceDocument(f"{name}.csv", "".join(lines).encode("utf-8")),
    )


def _nk_tables(*materials: str) -> tuple[ResourceTable, ...]:
    return tuple(
        ResourceTable(
            mat,
            "nk",
            SourceDocument(f"nk:{mat}.csv", (DATA_DIR / "nk" / f"{mat}.csv").read_bytes()),
        )
        for mat in materials
    )


def _scaps_mirror_v2_optical_layers(
    wl_nm: NDArray[np.float64],
    *,
    incoherent_glass: bool = True,
) -> tuple[OpticalLayerSpec, ...]:
    specs = (
        ("glass", 1.0e-3, "glass", incoherent_glass, False),
        ("SCAPS_HTL", 20.0e-9, "spiro_OMeTAD", False, True),
        ("SCAPS_PVK", 800.0e-9, "MAPbI3", False, True),
        ("SCAPS_ETL", 25.0e-9, "TiO2", False, True),
    )
    out: list[OpticalLayerSpec] = []
    for layer_id, d_m, mat_name, incoh, elec in specs:
        raw = np.loadtxt(DATA_DIR / "nk" / f"{mat_name}.csv", delimiter=",", skiprows=1)
        n, k = sample_nk_table(raw[:, 0], raw[:, 1], raw[:, 2], wl_nm, material=mat_name)
        out.append(
            OpticalLayerSpec(
                layer_id,
                d_m,
                n,
                k,
                incoherent=incoh,
                electrical=elec,
            )
        )
    return tuple(out)


@pytest.mark.parametrize("count", [5, 13, 31, 127])
@pytest.mark.parametrize("geom_kind", [INTEGER_ENDPOINT_V1, HALF_ENDPOINT_V1])
def test_analytic_slab_with_front_substrate_closes_photon_budget(
    count: int,
    geom_kind: str,
) -> None:
    wl_nm = np.linspace(300.0, 1000.0, 200)
    wl_m = wavelength_grid_m(wl_nm)
    flux: NDArray[np.float64] = np.full(200, 2.0e27, dtype=np.float64)
    spectrum = SpectralPhotonFlux(wl_nm, flux, "flat_2e27")
    substrate_alpha, absorber_alpha = 2.0e3, 8.0e6
    substrate_d, absorber_d = 1.0e-3, 800.0e-9

    n_sub, k_sub = constant_index_optical_constants(
        wl_nm, absorption_coefficient_m1=substrate_alpha, refractive_index=1.5
    )
    n_abs, k_abs = constant_index_optical_constants(
        wl_nm, absorption_coefficient_m1=absorber_alpha, refractive_index=2.5
    )
    binding = SpectralBeerLambertBinding(
        layers=(
            OpticalLayerSpec(
                "glass", substrate_d, n_sub, k_sub, incoherent=True, electrical=False
            ),
            OpticalLayerSpec(
                "absorber", absorber_d, n_abs, k_abs, incoherent=False, electrical=True
            ),
        ),
        spectrum=spectrum,
    )
    nodes = multilayer_grid([(absorber_d, count - 1)], alpha=[3.0])
    geometry = build_geometry(geom_kind, nodes, np.asarray([], dtype=float))
    profile = binding.generation_profile(geometry)
    budget = binding.photon_budget()

    incident_expected = float(np.trapezoid(flux, x=wl_m))
    expected_electrical = (
        incident_expected
        * float(np.exp(-substrate_alpha * substrate_d))
        * float(-np.expm1(-absorber_alpha * absorber_d))
    )
    expected_parasitic = incident_expected * float(
        -np.expm1(-substrate_alpha * substrate_d)
    )
    expected_transmitted = incident_expected * float(
        np.exp(-(substrate_alpha * substrate_d + absorber_alpha * absorber_d))
    )

    assert profile.method == "beer_lambert_spectral"
    assert np.all(profile.source_m3_s >= 0.0)
    assert np.max(np.abs(budget.spectral_closure_residual())) <= 1.0e-15
    assert budget.reflected_m2_s == 0.0
    assert budget.dropped_m2_s == 0.0
    assert profile.areal_generation_m2_s == pytest.approx(
        expected_electrical, rel=1.0e-13
    )
    assert float(np.sum(profile.source_m3_s * geometry.cell_measures_m)) == pytest.approx(
        expected_electrical, rel=1.0e-13
    )
    assert budget.parasitic_absorbed_m2_s == pytest.approx(
        expected_parasitic, rel=1.0e-13
    )
    assert budget.transmitted_m2_s == pytest.approx(expected_transmitted, rel=1.0e-13)
    total_accounted = (
        profile.areal_generation_m2_s
        + budget.parasitic_absorbed_m2_s
        + budget.transmitted_m2_s
    )
    assert abs(total_accounted - budget.incident_m2_s) <= 1.0e-14 * budget.incident_m2_s


def test_scaps_mirror_v2_photon_budget_mesh_invariance_and_spectral_convergence() -> None:
    spectrum_200 = _am15g_spectrum(200)
    layers_incoh = _scaps_mirror_v2_optical_layers(
        spectrum_200.wavelengths_nm, incoherent_glass=True
    )
    layers_coh = _scaps_mirror_v2_optical_layers(
        spectrum_200.wavelengths_nm, incoherent_glass=False
    )
    bl_binding = SpectralBeerLambertBinding(layers_incoh, spectrum_200)
    tmm_incoh = OpticalStackBinding(layers_incoh, spectrum_200)
    tmm_coh = OpticalStackBinding(layers_coh, spectrum_200)

    # Construct via from_optical_stack must reproduce cell fluxes identically.
    from_stack = SpectralBeerLambertBinding.from_optical_stack(tmm_incoh)
    assert from_stack.sha256 != tmm_incoh.sha256

    thicknesses = [20.0e-9, 800.0e-9, 25.0e-9]
    interfaces: NDArray[np.float64] = np.cumsum(thicknesses, dtype=np.float64)[:-1]
    currents_A_m2: list[float] = []
    for n_per_layer in (5, 10, 20, 40, 80):
        base_nodes = multilayer_grid(
            [(d, n_per_layer) for d in thicknesses], alpha=[3.0, 3.0, 3.0]
        )
        for geom in (
            build_geometry(INTEGER_ENDPOINT_V1, base_nodes, interfaces),
            build_geometry(HALF_ENDPOINT_V1, base_nodes, interfaces),
            build_geometry(
                TWO_SIDED_TRACE_V1,
                remove_shared_interface_nodes(base_nodes, interfaces),
                interfaces,
            ),
        ):
            prof = bl_binding.generation_profile(geom)
            currents_A_m2.append(Q * prof.areal_generation_m2_s)
            np.testing.assert_array_equal(
                prof.absorbed_m2_s,
                from_stack.generation_profile(geom).absorbed_m2_s,
            )

    np.testing.assert_allclose(currents_A_m2, currents_A_m2[0], rtol=1.0e-13)
    # Historical archive/wip-beer-lambert (658baa84) reference: ~260.14 A/m^2 (BL)
    # vs ~235.06 A/m^2 (coherent-glass TMM, legacy SCAPS loader effective) and
    # ~232.52 A/m^2 (front-incoherent-glass TMM, declared incoherent=True).
    assert currents_A_m2[0] == pytest.approx(260.14, abs=0.15)
    tmm_incoh_j = Q * tmm_incoh.photon_budget().electrical_absorbed_m2_s
    tmm_coh_j = Q * tmm_coh.photon_budget().electrical_absorbed_m2_s
    assert tmm_coh_j == pytest.approx(235.06, abs=0.25)
    assert tmm_incoh_j == pytest.approx(232.52, abs=0.25)
    assert currents_A_m2[0] > tmm_coh_j > tmm_incoh_j > 220.0

    budget = bl_binding.photon_budget()
    assert np.max(np.abs(budget.spectral_closure_residual())) <= 1.0e-15
    closed = (
        budget.electrical_absorbed_m2_s
        + budget.parasitic_absorbed_m2_s
        + budget.transmitted_m2_s
    )
    assert abs(closed - budget.incident_m2_s) <= 1.0e-14 * budget.incident_m2_s

    # Independent spectral quadrature refinement (100 -> 200 -> 400 -> 800 -> 1600).
    spectral_j: list[float] = []
    for n_wl in (100, 200, 400, 800, 1600):
        spec_n = _am15g_spectrum(n_wl)
        lay_n = _scaps_mirror_v2_optical_layers(spec_n.wavelengths_nm)
        b_n = SpectralBeerLambertBinding(lay_n, spec_n).photon_budget()
        spectral_j.append(Q * b_n.electrical_absorbed_m2_s)
    diffs = [abs(spectral_j[i + 1] - spectral_j[i]) for i in range(len(spectral_j) - 1)]
    assert diffs[1] < diffs[0]
    assert diffs[2] < diffs[1]
    assert diffs[3] < diffs[2]
    assert diffs[3] < 0.05


def test_spectral_beer_lambert_never_calls_tmm_and_validates_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for fn_name in (
        "tmm_absorbed_photon_flux_per_cell",
        "tmm_absorption_profile",
        "tmm_cumulative_absorptance",
        "tmm_reflectance",
        "tmm_transmittance",
    ):
        monkeypatch.setattr(
            tm_module,
            fn_name,
            lambda *_a, **_k: pytest.fail(f"TMM {fn_name} invoked by Beer-Lambert"),
        )

    spec = _am15g_spectrum(64)
    layers = _scaps_mirror_v2_optical_layers(spec.wavelengths_nm)
    binding = SpectralBeerLambertBinding(layers, spec)
    thicknesses = [20.0e-9, 800.0e-9, 25.0e-9]
    nodes = multilayer_grid([(d, 8) for d in thicknesses], alpha=[3.0, 3.0, 3.0])
    geom = build_geometry(
        INTEGER_ENDPOINT_V1, nodes, np.cumsum(thicknesses, dtype=float)[:-1]
    )
    prof = binding.generation_profile(geom)
    diag = binding.absorption_profile(nodes)
    assert prof.areal_generation_m2_s > 0.0
    assert diag.shape == (nodes.size, 64)
    assert np.all(diag >= 0.0)
    assert binding.sha256 and SPECTRAL_BEER_LAMBERT_MODEL_ID == (
        "beer_lambert_spectral_front_incidence_v1"
    )

    with pytest.raises(ValueError, match="electrical window"):
        binding.absorbed_photon_flux_per_cell(np.array([-1.0e-9, 100.0e-9]))
    with pytest.raises(ValueError, match="electrical window"):
        binding.absorption_profile(np.array([0.0, 900.0e-9]))
    with pytest.raises(ValueError, match="strictly increasing"):
        spectral_beer_lambert_absorption_coefficients(
            layers, np.array([500.0e-9, 400.0e-9])
        )


def test_opt_in_compilers_preserve_default_parity_and_separate_identities(
    defaults: DefaultCatalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = SourceDocument(
        "fixture:nip_MAPbI3",
        (ROOT / "perovskite-sim/tests/fixtures/configs/nip_MAPbI3.yaml").read_bytes(),
    )
    am15g = _am15g_resource(60)
    nk = _nk_tables("TiO2", "MAPbI3", "spiro_OMeTAD")
    lib = ResourceLibrary((am15g, *nk))

    data = import_standard_device(
        original, id="opt_in_optics", defaults=defaults
    ).editing_data()
    data["spectrum"] = am15g.name
    data["settings"].update(
        mode="full",
        T=300.0,
        built_in_potential_mode="semiconductor_work_function",
        dos_band_potentials=True,
        ion_steric_diffusion_only=False,
        ion_steric_shared_site=False,
        Phi=2.5e21,
    )
    optical_map = {"ETL": "TiO2", "absorber": "MAPbI3", "HTL": "spiro_OMeTAD"}
    band_map = {
        "ETL": (4.0, 3.2),
        "absorber": (3.9, 1.6),
        "HTL": (2.45, 3.0),
    }
    vt = float(thermal_voltage(300.0))
    for layer in data["layers"]:
        role = layer["role"]
        chi_eV, eg_eV = band_map[role]
        ni = float(np.sqrt(2.2e24 * 1.8e24) * np.exp(-eg_eV / (2.0 * vt)))
        p = layer.setdefault("parameters", {})
        p.update(
            chi=chi_eV,
            Eg=eg_eV,
            Nc300=2.2e24,
            Nv300=1.8e24,
            ni=ni,
            n1=ni,
            p1=ni,
            optical_material=optical_map[role],
            n_optical=None,
            alpha=1.0e7 if role == "absorber" else 1.0e5,
            P0=0.0,
            P0_neg=0.0,
            D_ion=0.0,
            D_ion_neg=0.0,
        )
    prepared = resolve_device(
        DeviceInput.model_validate(data), defaults, lib, sources=(original,)
    )

    default_dc = compile_device_1d(prepared, intervals_per_layer=8)
    explicit_tmm = compile_device_1d(
        prepared, intervals_per_layer=8, optical_model="tmm"
    )
    explicit_sbl = compile_device_1d(
        prepared, intervals_per_layer=8, optical_model="beer_lambert_spectral"
    )
    explicit_bl = compile_device_1d(
        prepared, intervals_per_layer=8, optical_model="beer_lambert"
    )

    # Default (optical_model=None) is bit-identical to historical TMM dispatch.
    assert default_dc.identity == explicit_tmm.identity
    np.testing.assert_array_equal(
        default_dc.generation.source_m3_s, explicit_tmm.generation.source_m3_s
    )
    # Three optical models produce three distinct cryptographic identities.
    assert (
        len({explicit_tmm.identity, explicit_sbl.identity, explicit_bl.identity}) == 3
    )
    assert explicit_tmm.generation.method == "transfer_matrix"
    assert explicit_sbl.generation.method == "beer_lambert_spectral"
    assert explicit_bl.generation.method == "beer_lambert"

    with pytest.raises(ValueError, match="optical_model"):
        compile_device_1d(prepared, intervals_per_layer=8, optical_model="TMM-typo")  # type: ignore[arg-type]

    # Guard: compiling with beer_lambert_spectral never invokes transfer_matrix.
    monkeypatch.setattr(
        tm_module,
        "tmm_absorbed_photon_flux_per_cell",
        lambda *_a, **_k: pytest.fail("TMM called during beer_lambert_spectral"),
    )
    recompiled_sbl = compile_device_1d(
        prepared, intervals_per_layer=8, optical_model="beer_lambert_spectral"
    )
    assert recompiled_sbl.identity == explicit_sbl.identity
    monkeypatch.undo()

    # Single-domain and two-domain ionic compilers preserve default identity
    # and distinguish all three optical models.
    ion_data = deepcopy(data)
    ion_data["layers"][1]["parameters"].update(
        P0=1.0e24, D_ion=1.0e-16, P_lim=1.0e26, E_a_ion=0.4
    )
    ion_prepared = resolve_device(
        DeviceInput.model_validate(ion_data), defaults, lib, sources=(original,)
    )
    ion_default = compile_ionic_device_1d(
        ion_prepared,
        intervals_per_layer=6,
        ion_domain_layer=ion_data["layers"][1]["id"],
        ion_domain_endpoints=("blocking", "blocking"),
    )
    ion_models = [
        compile_ionic_device_1d(
            ion_prepared,
            intervals_per_layer=6,
            ion_domain_layer=ion_data["layers"][1]["id"],
            ion_domain_endpoints=("blocking", "blocking"),
            optical_model=m,  # type: ignore[arg-type]
        )
        for m in ("tmm", "beer_lambert_spectral", "beer_lambert")
    ]
    assert ion_default.electronic_component.identity == (
        ion_models[0].electronic_component.identity
    )
    elec_manifest = ion_default.binding_manifest["electronic_component"]
    assert isinstance(elec_manifest, dict)
    assert "optical_model" not in elec_manifest
    assert len({c.identity for c in ion_models}) == 3

    multi_data = deepcopy(data)
    for idx in (0, 2):
        multi_data["layers"][idx]["parameters"].update(
            P0=1.0e24, D_ion=1.0e-16, P_lim=1.0e26, E_a_ion=0.4
        )
    multi_prepared = resolve_device(
        DeviceInput.model_validate(multi_data), defaults, lib, sources=(original,)
    )
    domains = (
        (multi_data["layers"][0]["id"], ("blocking", "blocking")),
        (multi_data["layers"][2]["id"], ("blocking", "blocking")),
    )
    multi_default = compile_multiple_ionic_device_1d(
        multi_prepared, intervals_per_layer=6, ion_domains=domains
    )
    multi_sbl = compile_multiple_ionic_device_1d(
        multi_prepared,
        intervals_per_layer=6,
        ion_domains=domains,
        optical_model="beer_lambert_spectral",
    )
    assert multi_default.identity != multi_sbl.identity

    # H_input and H_physics distinguish explicit optical model overrides while
    # keeping un-overridden default selection untouched.
    src_bl = SourceDocument(
        "src/solarlab/physics/generation/beer_lambert.py",
        (ROOT / "src/solarlab/physics/generation/beer_lambert.py").read_bytes(),
    )
    src_sbl = SourceDocument(
        "src/solarlab/physics/generation/beer_lambert_spectral.py",
        (
            ROOT / "src/solarlab/physics/generation/beer_lambert_spectral.py"
        ).read_bytes(),
    )
    src_tmm = SourceDocument(
        "src/solarlab/physics/generation/optical_binding.py",
        (ROOT / "src/solarlab/physics/generation/optical_binding.py").read_bytes(),
    )
    defs = []
    projs = []
    refs: dict[str, ModelRef] = {}
    for key_name, label, src in (
        ("beer_lambert", "Scalar Beer-Lambert", src_bl),
        ("beer_lambert_spectral", "Spectral Beer-Lambert", src_sbl),
        ("tmm", "Transfer-Matrix Optics", src_tmm),
    ):
        ref = ModelRef(f"generation.{key_name}", 1)
        refs[key_name] = ref
        cap = CapabilityRule(
            "source_data_only",
            (1,),
            ("dc", "transient"),
            (EvidenceRef(src.id + "#binding", src.sha256, "source"),),
            backends=("metadata_only",),
        )
        defs.append(
            ModelDefinition(
                ref,
                label,
                "generation",
                ("layer",),
                "node",
                ("alpha",),
                (),
                (),
                cap,
                parameter_schema="full_layer",
            )
        )
        projs.append(SemanticProjection(ref, "all_values_v1", (src,)))
    registry = ModelRegistry(tuple(defs))
    model_defaults = {
        "layer": (ModelChoice("optics", refs["tmm"], parameter_schema="full_layer"),)
    }
    topo = device_topology(prepared)
    param_sources = {
        Scope("layer", (layer.id,)): layer_parameter_source(layer)
        for layer in prepared.layers
        if layer.id in topo.electrical_layer_ids
    }
    ctx = CapabilityContext(
        1,
        "dc",
        "metadata_only",
        "maxwell_boltzmann",
        temperature_K=prepared.values["settings"]["T"],
    )

    def _id_for(override_key: str | None) -> tuple[str, str]:
        ov: dict[Scope, dict[str, tuple[ModelChoice, ...]]] = {}
        if override_key is not None:
            choice = ModelChoice(
                "optics", refs[override_key], parameter_schema="full_layer"
            )
            ov = {
                Scope("layer", (lid,)): {"generation": (choice,)}
                for lid in topo.electrical_layer_ids
            }
        sel = select_models(registry, topo, ctx, model_defaults, param_sources, ov)
        active_refs = {inst.model for inst in sel.instances}
        active_projs = tuple(p for p in projs if p.model in active_refs)
        inp = IdentityInputs(
            prepared,
            sel,
            registry,
            ModelInputs.from_mappings(defaults=model_defaults, overrides=ov),
            active_projs,
        )
        return input_identity(inp).sha256, physics_identity(inp).sha256

    h_default = _id_for(None)
    h_by_model = {k: _id_for(k) for k in ("tmm", "beer_lambert_spectral", "beer_lambert")}
    assert h_default[1] == h_by_model["tmm"][1]
    assert len({pair[0] for pair in h_by_model.values()}) == 3
    assert len({pair[1] for pair in h_by_model.values()}) == 3


def test_scaps_beer_lambert_catalog_provenance_and_sweep_application(
    defaults: DefaultCatalog,
) -> None:
    catalog = load_scaps_beer_lambert_catalog()
    assert catalog == reference_catalog()
    assert catalog["study"] == SCAPS_BEER_LAMBERT_STUDY_ID
    assert len(catalog["sweeps"]) == 11
    points = [p for s in catalog["sweeps"].values() for p in s["points"]]
    assert sum(p.get("origin") != "solarlab_extension" for p in points) == 93
    assert len(points) == 95
    workbook_path = ROOT / "perovskite-sim" / catalog["source"]
    assert (
        catalog["source_sha256"]
        == hashlib.sha256(workbook_path.read_bytes()).hexdigest()
    )
    assert catalog["sweeps"]["CHI_ETL"]["points"][-1]["reference"]["V_oc"] is None

    scaps_src = SourceDocument(
        "configs/scaps_mirror_v2.yaml",
        (ROOT / "perovskite-sim/configs/scaps_mirror_v2.yaml").read_bytes(),
    )
    base_input = import_scaps_device(scaps_src, id="scaps_mirror_v2", defaults=defaults)
    nk_lib = ResourceLibrary(_nk_tables("glass", "spiro_OMeTAD", "MAPbI3", "TiO2"))

    # High-doping Nd_PVK extension points carry null references and min_N_grid=90.
    for ext_val in (1.0e19, 1.0e20):
        pt = next(
            p for p in catalog["sweeps"]["Nd_PVK"]["points"] if p["x"] == ext_val
        )
        assert pt["origin"] == "solarlab_extension"
        assert pt["row"] is None
        assert all(v is None for v in pt["reference"].values())
        edited, meta = apply_reference_point(base_input, "Nd_PVK", ext_val)
        assert meta["scope"] == "high_doping_extrapolation"
        assert meta["min_N_grid"] == HIGH_DOPING_MIN_N_GRID == 90
        assert meta["jv_sweep"]["N_grid"] == 90
        assert meta["optical_model"] == "beer_lambert_spectral"
        resolved = resolve_device(edited, defaults, nk_lib)
        absorber = next(lay for lay in resolved.layers if lay.role == "absorber")
        assert dict(absorber.parameters)["N_D"] == pytest.approx(ext_val * 1.0e6)

    # Every one of the 11 sweep families applies cleanly and resolves via PreparedDevice.
    for sweep_name in SWEEPS:
        first_x = float(catalog["sweeps"][sweep_name]["points"][0]["x"])
        edited, meta = apply_reference_point(base_input, sweep_name, first_x)
        assert meta["jv_sweep"] == STANDARD_JV_SWEEP_HINTS
        assert meta["computed_V_bi"] is not None
        resolved = resolve_device(edited, defaults, nk_lib)
        assert isinstance(resolved, PreparedDevice)

    # Verify species_only vs historical_effective_lifetime on Nt_C_PVK.
    eff_in, _ = apply_reference_point(
        base_input, "Nt_C_PVK", 1.0e15, bulk_defect_mode="historical_effective_lifetime"
    )
    sp_in, _ = apply_reference_point(
        base_input, "Nt_C_PVK", 1.0e15, bulk_defect_mode="species_only"
    )
    eff_res = resolve_device(eff_in, defaults, nk_lib)
    sp_res = resolve_device(sp_in, defaults, nk_lib)
    eff_pvk = dict(next(lay for lay in eff_res.layers if lay.role == "absorber").parameters)
    sp_pvk = dict(next(lay for lay in sp_res.layers if lay.role == "absorber").parameters)
    assert eff_pvk["trap_N_t_bulk"] == pytest.approx(1.0e21)
    assert sp_pvk["trap_N_t_bulk"] == pytest.approx(1.0e21 + 1.0e18)

    # Fail-closed validation on typos, booleans, nonfinite values, and out-of-range x.
    for bad_sweep, bad_val in (
        ("typo", 0.0),
        ("CHI_ETL", 10.0),
        ("Nd_PVK", 1.0e21),
        ("CHI_ETL", float("nan")),
        ("CHI_ETL", float("inf")),
    ):
        with pytest.raises(ValueError):
            validate_reference_point(bad_sweep, bad_val)
    with pytest.raises(ValueError):
        validate_reference_point("CHI_ETL", True)  # type: ignore[arg-type]
