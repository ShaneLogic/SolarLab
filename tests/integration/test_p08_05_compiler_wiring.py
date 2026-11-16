"""Integration tests for P08-05 opt-in compiler/registry wiring and Gate G8 closeout.

Verifies:
(a) 0-byte manifest and exact SHA-256 identity invariance under default ``v1``
    across all 5 compilers (1D deduplicated & two-sided trace, single/dual ionic
    1D, 2D, and ionic 2D) relative to the Wave 2 baseline ``582f019e``.
(b) Pairwise-distinct ``compiled.identity`` and ``H_physics`` (``physics_identity``)
    digests under every P08 ``v2`` option, ``HistoricalDisable`` / ``D20`` pre-P08
    CBO pinning enforcement, permanent ``CapabilityError`` refusal of falsified
    ``interface_shared_occupancy``, and strict certificate non-reuse across
    ``v1 <-> v2`` in ``solve_transient``, ``solve_impedance``, ``solve_ionic_transient``,
    and ``solve_ionic_impedance``.
(c) End-to-end compiled 1D all-V2 (``LAYER_SERIES_CAPACITANCE_V2`` +
    ``PHYSICAL_HALF_ENDPOINT_V2`` + ``HOLE_SEPARATE_INCREMENT_V1`` +
    ``RECIPROCAL_PHYSICAL_THERMIONIC_V2`` + ``beer_lambert_spectral``):
    0-ULP ``DC1DProblem`` vs ``AssembledDC1DProblem`` parity, FD Jacobian
    consistency, dark equilibrium, illuminated steady J-V, ``two_sided_trace``
    +3.0 V super-flatband continuation, Backward Euler transient step-accounting
    conservation, and 3-grid spatial refinement convergence.
(d) Compiled 2D all-V2: 0-ULP ``DC2DProblem`` vs ``AssembledDC2DProblem`` and
    ``Ionic2DProblem`` vs ``AssembledIonic2DProblem`` parity, exact 2D series
    capacitor, and 1D-vs-2D all-V2 column steady-state consistency.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
import pytest

from examples.research_1d.ionic_steady import controls as ionic_steady_controls
from examples.research_1d.transient import json_value
from solarlab.config.device_inputs import DeviceInput
from solarlab.config.identity import BuildSource, physics_identity
from solarlab.config.model_parameters import device_topology, layer_parameter_source
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import CapabilityError, compile_device_1d, p08_identity_inputs
from solarlab.device.compile_2d import compile_device_2d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.compile_ionic_2d import compile_ionic_device_2d
from solarlab.device.compile_multiple_ionic_1d import compile_multiple_ionic_device_1d
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.resolved import PreparedDevice
from solarlab.discretization import LAYER_SERIES_CAPACITANCE_V2, VACUUM_PERMITTIVITY_F_M
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
from solarlab.equations.assembly.ionic_2d import AssembledIonic2DProblem
from solarlab.equations.assembly.two_dimensional import AssembledDC2DProblem
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.equations.dc_2d import DC2DProblem
from solarlab.equations.ionic_2d import Ionic2DProblem
from solarlab.experiments.impedance import solve_impedance
from solarlab.experiments.ionic_impedance import solve_ionic_impedance
from solarlab.experiments.ionic_transient import (
    IonicTransientOptions,
    load_ionic_steady_initial,
    solve_ionic_transient,
)
from solarlab.experiments.jv.steady import run_steady_jv
from solarlab.experiments.transient import TransientOptions, TransientProtocol, solve_transient
from solarlab.experiments.two_dimensional.inputs import JV2DInput
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab.mesh import PHYSICAL_HALF_ENDPOINT_V2
from solarlab.physics.contacts import (
    CONTACT_THERMODYNAMIC_TOLERANCE_EV,
    ContactEndpoint,
    ContactThermodynamicError,
    assess_contact_thermodynamics,
    require_contact_thermodynamic_certificate,
)
from solarlab.physics.generation import sample_tabulated
from solarlab.physics.interface_transport import (
    LEGACY_UNNORMALIZED_THERMIONIC_V1,
    RECIPROCAL_PHYSICAL_THERMIONIC_V2,
    SCAPS_EFFECTIVE_VELOCITY_THERMIONIC_V1,
)
from solarlab.physics.presets import (
    P08_MODEL_VERSION_MAPPINGS,
    P08_SOURCE_PATHS,
    P08_V1_MODEL_REFS,
    P08_V2_MODEL_REFS,
    P08CompilerOptions,
    build_p08_model_registry,
    build_p08_preset,
    select_preset,
)
from solarlab.physics.registry import CapabilityContext, Scope
from solarlab.physics.selection import ModelChoice
from solarlab.physics.transport.electrochemical_flux import HOLE_SEPARATE_INCREMENT_V1
from solarlab.solvers.nonlinear.ionic_1d import solve_ionic_steady_state
from solarlab.solvers.nonlinear.problem import SolveBudget
from solarlab.solvers.nonlinear.production import POISSON_CONTACT_BALANCE_V1
from solarlab.solvers.nonlinear.steady_1d import (
    SteadyOptions,
    solve_equilibrium,
    solve_steady_state,
)
from solarlab.solvers.nonlinear.steady_2d import Steady2DOptions, solve_steady_state_2d
from solarlab_research.cbo_scan import InterfaceCBOScanError, enforce_pre_p08_cbo_pinning

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "perovskite-sim" / "perovskite_sim" / "data"

# Portable checkout-independent SHA-256 digests captured on
# refactor/wave2-n-baseline @ 582f019ef5743d43546219590a5ae094ca300594
# using repo-relative SourceDocument("examples/research_1d/device.json", ...).
PINNED_V1_COMPILE_1D_SHA256 = "f61600398765ade465c1eea97a892aa10d8014fa564ac75a5b8d41e645c38c46"
PINNED_V1_HET_DEDUP_DEVICE_SHA256 = "2908b0632e5a96e512e5d77d24a483152589bce082372b5f909db87b6313fee4"
PINNED_V1_HET_DEDUP_IFACE_SHA256 = "4b60ec8744e26ea34fc32a8441bdbc35d35ee151653cb7052414db34d641880b"
PINNED_V1_HET_TRACE_DEVICE_SHA256 = "43c649df0ec56f7ff4f0bb0046e2629c88bb25f332bd1785841d0872e07a8c39"
PINNED_V1_HET_TRACE_IFACE_SHA256 = "372af71b2cbd97b3cbc66b6971e4c32818f6c5a9a4391ae02da913a40f0f8aa4"
PINNED_V1_IONIC_1D_SHA256 = "80dbb4f30a2820a2a73219fdd62a0117374b842b22b5781758c2a492f2183215"
PINNED_V1_MULTI_IONIC_1D_SHA256 = "76483953cfbdc4d53d1acb5d0c1db06d61ed4a24dd6d200f89ba19c9886b2986"
PINNED_V1_COMPILE_2D_SHA256 = "1a9d23b527817d742bc7c974b32d7527ef06b3b8678bde231e7fe76e444d1408"
PINNED_V1_IONIC_2D_SHA256 = "d62f674f33dec4298a3c8319c13f23c80749f081b742320b5f8ce776312e719d"


def load_device() -> PreparedDevice:
    """Load the research 1D stack with a portable repo-relative SourceDocument ID."""
    rel_id = "examples/research_1d/device.json"
    raw = (ROOT / rel_id).read_bytes()
    source = SourceDocument(rel_id, raw)
    document = json.loads(raw)
    catalog = {
        **document["default_catalog"],
        "evidence": [[source.id, source.sha256]],
    }
    return resolve_device(
        DeviceInput.model_validate(document["device"]),
        DefaultCatalog.from_mapping(catalog),
        ResourceLibrary(),
        sources=(source,),
    )


def prepare_confined_ion_device(*, diffusion_only: bool = True) -> PreparedDevice:
    """Prepare the confined absorber ion stack with portable repo-relative source IDs."""
    template = load_device()
    data = template.to_input().editing_data()
    data["id"] = "confined_absorber_positive_ion"
    data["settings"]["ion_steric_diffusion_only"] = diffusion_only
    absorber = next(layer for layer in data["layers"] if layer["role"].lower() == "absorber")
    absorber["parameters"].update(P0=1.0e24, P_lim=1.0e27, D_ion=1.0e-16, E_a_ion=0.58)
    return resolve_device(
        DeviceInput.model_validate(data),
        template.defaults,
        template.resources,
        sources=template.sources,
    )


def _with_edits(prepared: PreparedDevice, mutate_fn: Any, *, resources: ResourceLibrary | None = None) -> PreparedDevice:
    data = prepared.to_input().editing_data()
    mutate_fn(data)
    return resolve_device(
        DeviceInput.model_validate(data),
        prepared.defaults,
        prepared.resources if resources is None else resources,
        sources=prepared.sources,
    )


def _heterojunction_astar_device(base: PreparedDevice) -> PreparedDevice:
    def mutate(data: dict[str, Any]) -> None:
        astars_n = (2.0e5, 4.5e5, 6.0e5)
        astars_p = (3.0e5, 5.0e5, 7.0e5)
        for layer, an, ap in zip(data["layers"], astars_n, astars_p):
            layer.setdefault("parameters", {})["A_star_n"] = an
            layer["parameters"]["A_star_p"] = ap

    return _with_edits(base, mutate)


def _two_domain_ionic_device(base_ion: PreparedDevice) -> PreparedDevice:
    def mutate(data: dict[str, Any]) -> None:
        for layer in data["layers"]:
            p = layer.setdefault("parameters", {})
            if layer["id"] in ("htl", "etl"):
                p.update(P0=1.0e23, P_lim=1.0e26, D_ion=1.0e-15, E_a_ion=0.0)
            else:
                p.update(P0=0.0, D_ion=0.0)

    return _with_edits(base_ion, mutate)


def _spectral_heterojunction_device(base: PreparedDevice, *, two_sided_trace: bool = False) -> PreparedDevice:
    """3-layer heterojunction with unequal DOS/permittivity/A* and spectral nk + AM1.5G."""
    raw = np.loadtxt(DATA_DIR / "am15g.csv", delimiter=",", comments="#")
    wl_nm = np.linspace(320.0, 920.0, 61)
    flux = sample_tabulated(raw[:, 0], raw[:, 1], wl_nm, name="am15g")
    spec_lines = ["wavelength_nm,spectral_flux_m-2_s-1_m-1\n"] + [
        f"{repr(float(w))},{repr(float(f))}\n" for w, f in zip(wl_nm, flux)
    ]
    spec_table = ResourceTable(
        "am15g_61",
        "spectrum",
        SourceDocument("am15g_61.csv", "".join(spec_lines).encode("utf-8")),
    )
    nk_tables = tuple(
        ResourceTable(
            mat,
            "nk",
            SourceDocument(f"nk:{mat}.csv", (DATA_DIR / "nk" / f"{mat}.csv").read_bytes()),
        )
        for mat in ("spiro_OMeTAD", "MAPbI3", "TiO2")
    )
    merged_resources = ResourceLibrary((*base.resources.tables, spec_table, *nk_tables))

    def mutate(data: dict[str, Any]) -> None:
        data["spectrum"] = "am15g_61"
        specs = (
            ("spiro_OMeTAD", 2.0e5, 3.0e5),
            ("MAPbI3", 4.5e5, 5.0e5),
            ("TiO2", 6.0e5, 7.0e5),
        )
        for layer, (nk_name, an, ap) in zip(data["layers"], specs):
            p = layer.setdefault("parameters", {})
            p.update(
                optical_material=nk_name,
                A_star_n=an,
                A_star_p=ap,
                incoherent=False,
            )
            if layer["role"] == "absorber":
                p.update(tau_n=2.0e-8, tau_p=2.0e-8)
        if two_sided_trace:
            for iface in data.get("interfaces", []):
                iface["v_n"] = None
                iface["v_p"] = None

    return _with_edits(base, mutate, resources=merged_resources)


def _transient_options(
    *,
    predictor: Literal["previous", "generation_inventory"] = "previous",
) -> TransientOptions:
    return TransientOptions(
        poisson_tolerance=1.0e-11,
        continuity_tolerance_A_m2=1.0e-6,
        contact_log_tolerance=1.0e-11,
        potential_step_V=0.1,
        quasi_fermi_step=2.0,
        newton_budget=SolveBudget(
            max_iterations=80,
            max_residual_evaluations=5000,
            max_jacobian_evaluations=1000,
            max_backtracks=30,
        ),
        max_steps=64,
        wall_s=15.0,
        predictor=predictor,
        convergence_policy=POISSON_CONTACT_BALANCE_V1,
    )


def _p08_build_source() -> tuple[BuildSource, dict[str, SourceDocument]]:
    docs = {
        rel_path: SourceDocument(rel_path, (ROOT / "src" / rel_path).read_bytes())
        for rel_path in P08_SOURCE_PATHS
    }
    build = BuildSource(
        "582f019ef5743d43546219590a5ae094ca300594",
        tuple(docs[k] for k in sorted(docs)),
    )
    return build, docs


# ---------------------------------------------------------------------------
# (a) Default v1 0-Byte Manifest & Exact SHA-256 Baseline Invariance
# ---------------------------------------------------------------------------


def test_p08_05_default_v1_preserves_exact_wave2_baseline_digests() -> None:
    """Default v1 compilation matches 582f019e SHA-256 digests bit-for-bit across all 5 compilers."""
    base = load_device()
    c1d = compile_device_1d(base, intervals_per_layer=(8, 12, 8))
    assert c1d.identity == PINNED_V1_COMPILE_1D_SHA256

    # Default P08CompilerOptions() forwards zero extra kwargs and produces the exact same digest
    v1_opts = P08CompilerOptions()
    assert v1_opts.is_v1_default
    assert v1_opts.active_model_refs() == P08_V1_MODEL_REFS
    assert v1_opts.compiler_kwargs_1d() == {}
    assert v1_opts.compiler_kwargs_2d() == {}
    assert (
        compile_device_1d(base, intervals_per_layer=(8, 12, 8), **v1_opts.compiler_kwargs_1d()).identity
        == PINNED_V1_COMPILE_1D_SHA256
    )

    # Deduplicated & two-sided trace heterojunctions
    het_dedup = _heterojunction_astar_device(base)
    c_dedup = compile_device_1d(het_dedup, intervals_per_layer=(8, 12, 8))
    assert c_dedup.identity == PINNED_V1_HET_DEDUP_DEVICE_SHA256
    assert c_dedup.interface_model is not None
    assert c_dedup.interface_model.identity_sha256 == PINNED_V1_HET_DEDUP_IFACE_SHA256
    assert c_dedup.interface_model.thermionic_mode == LEGACY_UNNORMALIZED_THERMIONIC_V1

    c_trace = compile_device_1d(
        het_dedup,
        intervals_per_layer=(8, 12, 8),
        interface_topology="two_sided_trace",
    )
    assert c_trace.identity == PINNED_V1_HET_TRACE_DEVICE_SHA256
    assert c_trace.interface_model is not None
    assert c_trace.interface_model.identity_sha256 == PINNED_V1_HET_TRACE_IFACE_SHA256
    assert c_trace.interface_model.thermionic_mode == LEGACY_UNNORMALIZED_THERMIONIC_V1

    # Single-domain and multi-domain 1D mobile ions
    ion_dev = prepare_confined_ion_device()
    c_ion1d = compile_ionic_device_1d(
        ion_dev,
        intervals_per_layer=(6, 10, 6),
        ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"),
    )
    assert c_ion1d.identity == PINNED_V1_IONIC_1D_SHA256
    elec_comp_manifest = cast(dict[str, Any], c_ion1d.binding_manifest["electronic_component"])
    assert "p08_options" not in elec_comp_manifest

    multi_ion_dev = _two_domain_ionic_device(ion_dev)
    c_multi1d = compile_multiple_ionic_device_1d(
        multi_ion_dev,
        intervals_per_layer=(6, 8, 6),
        ion_domains=(("htl", ("blocking", "blocking")), ("etl", ("blocking", "blocking"))),
    )
    assert c_multi1d.identity == PINNED_V1_MULTI_IONIC_1D_SHA256

    # 2D electronic and 2D ionic compilers
    sp = JV2DInput(kind="jv_2d", lateral_length="0.4 um", Nx=2, Ny_per_layer=6, lateral_bc="neumann")
    c2d = compile_device_2d(het_dedup, sp)
    assert c2d.identity == PINNED_V1_COMPILE_2D_SHA256

    c_ion2d = compile_ionic_device_2d(
        ion_dev,
        sp,
        ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"),
    )
    assert c_ion2d.identity == PINNED_V1_IONIC_2D_SHA256


# ---------------------------------------------------------------------------
# (b) Distinct V2 Identities, H_physics Separation, D20 Pinning & Non-Reuse
# ---------------------------------------------------------------------------


def test_p08_05_v2_options_shift_compiled_and_physics_identities_and_enforce_g8_guards(
    tmp_path: Path,
) -> None:
    """Every P08 V2 option produces distinct compiled.identity & H_physics and blocks certificate reuse."""
    base = load_device()
    spec_dev = _spectral_heterojunction_device(base, two_sided_trace=False)
    build, docs = _p08_build_source()

    option_variants: dict[str, P08CompilerOptions] = {
        "v1_default": P08CompilerOptions(),
        "p08_02a_series_eps": P08CompilerOptions(dielectric_rule=LAYER_SERIES_CAPACITANCE_V2),
        "p08_02b_half_endpoint": P08CompilerOptions(geometry_name=PHYSICAL_HALF_ENDPOINT_V2),
        "p08_02c_hole_increment": P08CompilerOptions(band_step_mode=HOLE_SEPARATE_INCREMENT_V1),
        "p08_03_reciprocal_harmonic": P08CompilerOptions(
            thermionic_mode=RECIPROCAL_PHYSICAL_THERMIONIC_V2,
            combination_mode="harmonic_series",
        ),
        "p08_03_reciprocal_compact": P08CompilerOptions(
            thermionic_mode=RECIPROCAL_PHYSICAL_THERMIONIC_V2,
            combination_mode="compact_support",
        ),
        "p08_03_scaps_velocity": P08CompilerOptions(
            thermionic_mode=SCAPS_EFFECTIVE_VELOCITY_THERMIONIC_V1,
        ),
        "p08_04_spectral_bl": P08CompilerOptions(optical_model="beer_lambert_spectral"),
        "all_v2": P08CompilerOptions.all_prospective_v2(),
    }

    compiled_digests: dict[str, str] = {}
    h_physics_digests: dict[str, str] = {}
    for label, opts in option_variants.items():
        compiled = compile_device_1d(spec_dev, intervals_per_layer=(4, 6, 4), **opts.compiler_kwargs_1d())
        compiled_digests[label] = compiled.identity
        id_inputs = p08_identity_inputs(spec_dev, build, options=opts)
        h_physics_digests[label] = physics_identity(id_inputs).sha256

    assert len(set(compiled_digests.values())) == len(option_variants)
    assert len(set(h_physics_digests.values())) == len(option_variants)

    # HistoricalDisable tracking when overriding v1 preset into prospective v2 models
    registry = build_p08_model_registry(docs)
    v1_preset = build_p08_preset(P08CompilerOptions(), docs)
    assert len(v1_preset.historical_disables) == 5
    topology = device_topology(spec_dev)
    param_sources = {
        Scope("layer", (layer.id,)): layer_parameter_source(layer)
        for layer in spec_dev.layers
        if layer.id in topology.electrical_layer_ids
    }
    ctx = CapabilityContext(1, "dc", "solarlab_assembled", "maxwell_boltzmann", temperature_K=300.0)
    layer_scope = Scope("layer", (spec_dev.layers[1].id,))
    override_v2 = {
        layer_scope: {
            P08_MODEL_VERSION_MAPPINGS[3].family: (
                ModelChoice(
                    P08_MODEL_VERSION_MAPPINGS[3].local_instance_id,
                    P08_MODEL_VERSION_MAPPINGS[3].v2_model,
                    parameter_schema="full_layer",
                ),
            )
        }
    }
    selected_override = select_preset(v1_preset, registry, topology, ctx, param_sources, overrides=override_v2)
    assert selected_override.reenabled_historical_mechanisms == ("prospective_thermionic_v2",)

    # D20 RS/CBO pre-P08 pinning rejects every V2 ModelRef unless explicitly opted in
    for v2_ref in P08_V2_MODEL_REFS:
        with pytest.raises(InterfaceCBOScanError, match="D20 model-pinning violation"):
            enforce_pre_p08_cbo_pinning(base, requested_models=(v2_ref,))
        admitted = enforce_pre_p08_cbo_pinning(
            base, requested_models=(v2_ref,), allow_prospective_p08_models=True
        )
        assert admitted == (v2_ref,)

    # Permanent pre-solve refusal of falsified interface_shared_occupancy across 1D, 2D & ionic 2D
    falsified_dev = _with_edits(
         prepare_confined_ion_device(),
        lambda d: d.setdefault("settings", {}).update(interface_shared_occupancy=True),
    )
    sp = JV2DInput(kind="jv_2d", lateral_length="0.4 um", Nx=2, Ny_per_layer=4, lateral_bc="neumann")
    with pytest.raises(CapabilityError, match="interface_shared_occupancy"):
        compile_device_1d(_with_edits(base, lambda d: d.setdefault("settings", {}).update(interface_shared_occupancy=True)))
    with pytest.raises(CapabilityError, match="interface_shared_occupancy"):
        compile_device_2d(_with_edits(base, lambda d: d.setdefault("settings", {}).update(interface_shared_occupancy=True)), sp)
    with pytest.raises(CapabilityError, match="interface_shared_occupancy"):
        compile_ionic_device_2d(
            falsified_dev,
            sp,
            ion_domain_layer="absorber",
            ion_domain_endpoints=("blocking", "blocking"),
        )

    # Fixed 5 meV contact thermodynamic gate & historical 6.024958093130e-3 eV SCAPS label remain enforced
    assert CONTACT_THERMODYNAMIC_TOLERANCE_EV == 5.0e-3
    vt_300 = 8.617333262145e-5 * 300.0
    nc_l, nv_l, chi_l, eg_l = 2.2e24, 1.8e25, 3.50, 2.05
    nc_r, nv_r, chi_r, eg_r = 2.2e24, 1.8e25, 4.00, 3.20
    p_l = 1.0e24
    n_l = nc_l * nv_l * math.exp(-eg_l / vt_300) / p_l
    n_r = 1.0e24
    p_r = nc_r * nv_r * math.exp(-eg_r / vt_300) / n_r
    wf_left = chi_l + eg_l - vt_300 * math.log(nv_l / p_l)
    wf_right = chi_r - vt_300 * math.log(n_r / nc_r)
    v_bi_compatible = wf_left - wf_right
    scaps_offset_assessment = assess_contact_thermodynamics(
        built_in_potential_mode="legacy_manual",
        left=ContactEndpoint(n_l, p_l, nc_l, nv_l, chi_l, eg_l),
        right=ContactEndpoint(n_r, p_r, nc_r, nv_r, chi_r, eg_r),
        V_bi_bc=v_bi_compatible + 6.024958093130195e-3,
        thermal_voltage_V=vt_300,
        semiconductor_potential_V=v_bi_compatible,
        has_band_reference=True,
    )
    assert scaps_offset_assessment.status == "inconsistent"
    assert scaps_offset_assessment.historical_label == "legacy_manual_v_bi_contact_offset"
    with pytest.raises(ContactThermodynamicError):
        require_contact_thermodynamic_certificate(scaps_offset_assessment)

    # Certificate non-reuse across v1 <-> v2 on 1D transient & small-signal impedance
    dev_v1 = compile_device_1d(spec_dev, intervals_per_layer=(4, 6, 4))
    dev_v2 = compile_device_1d(
        spec_dev,
        intervals_per_layer=(4, 6, 4),
        **P08CompilerOptions.all_prospective_v2().compiler_kwargs_1d(),
    )
    eq_v1 = solve_equilibrium(dev_v1)
    protocol = TransientProtocol(
        times_s=np.array([0.0, 1.0e-6]),
        voltage_V=np.array([0.0, 0.0]),
        generation_scale=np.array([0.0, 0.0]),
    )
    with pytest.raises(ValueError, match="steady point identity"):
        solve_transient(
            dev_v2,
            eq_v1,
            protocol,
            options=_transient_options(),
            initial_options=SteadyOptions(),
        )
    with pytest.raises(ValueError, match="steady point identity"):
        solve_impedance(dev_v2, eq_v1, [100.0])

    # Certificate non-reuse across v1 <-> v2 on ionic transient & ionic impedance
    ion_prep = prepare_confined_ion_device(diffusion_only=True)
    ion_v1 = compile_ionic_device_1d(
        ion_prep,
        intervals_per_layer=(4, 6, 4),
        ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"),
    )
    ion_v2 = compile_ionic_device_1d(
        ion_prep,
        intervals_per_layer=(4, 6, 4),
        ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"),
        dielectric_rule=LAYER_SERIES_CAPACITANCE_V2,
        geometry_name=PHYSICAL_HALF_ENDPOINT_V2,
        band_step_mode=HOLE_SEPARATE_INCREMENT_V1,
    )
    assert ion_v1.identity != ion_v2.identity
    ion_ctrl = ionic_steady_controls(ion_v1.ion_node_count)
    ion_ss_v1 = solve_ionic_steady_state(ion_v1, voltage_V=0.0, generation_scale=0.0, options=ion_ctrl)
    assert ion_ss_v1.accepted
    artifact_doc = {
        "summary": {
            "accepted": True,
            "compiled_identity": ion_v1.identity,
            "voltage_V": 0.0,
            "generation_scale": 0.0,
        },
        "result": ion_ss_v1,
    }
    artifact_text = json.dumps(json_value(artifact_doc), indent=2, allow_nan=False) + "\n"
    artifact_sha = hashlib.sha256(artifact_text.encode()).hexdigest()
    artifact_path = tmp_path / "ion_v1_initial.json"
    artifact_path.write_text(artifact_text)

    initial_v1 = load_ionic_steady_initial(ion_v1, artifact_path, artifact_sha, controls=ion_ctrl)
    with pytest.raises(ValueError, match="different compiled device"):
        load_ionic_steady_initial(ion_v2, artifact_path, artifact_sha, controls=ion_ctrl)
    with pytest.raises(ValueError, match="different compiled device"):
        solve_ionic_transient(
            ion_v2,
            initial_v1,
            protocol,
            options=IonicTransientOptions(_transient_options(), 1.0e6, 1.0),
        )
    with pytest.raises(ValueError, match="different compiled device"):
        solve_ionic_impedance(ion_v2, initial_v1, [10.0])


# ---------------------------------------------------------------------------
# (c) End-to-End 1D All-V2 Parity, Conservation, +3 V Stress & Refinement
# ---------------------------------------------------------------------------


def test_p08_05_compiled_1d_all_v2_parity_conservation_and_spatial_refinement() -> None:
    """End-to-end 1D all-V2: 0-ULP Assembled parity, FD Jacobian, BE conservation & grid refinement."""
    base = load_device()
    all_v2 = P08CompilerOptions.all_prospective_v2()

    # 1. Deduplicated all-V2 0-ULP parity, FD Jacobian, DC J-V, and BE transient conservation
    spec_dedup = _spectral_heterojunction_device(base, two_sided_trace=False)
    dev_dedup = compile_device_1d(spec_dedup, intervals_per_layer=(6, 10, 6), **all_v2.compiler_kwargs_1d())
    total_thickness = sum(layer.thickness for layer in spec_dedup.layers if layer.role != "substrate")
    assert abs(float(np.sum(dev_dedup.geometry.cell_measures_m)) - total_thickness) <= 1.0e-15 * total_thickness

    rng = np.random.default_rng(20261116)
    u_pert = np.r_[dev_dedup.reference_potential_V, np.zeros(2 * dev_dedup.node_count)]
    u_pert[: dev_dedup.node_count] += 0.03 * np.sin(np.linspace(0.0, np.pi, dev_dedup.node_count))
    u_pert[dev_dedup.node_count :] += rng.normal(0.0, 0.15, size=2 * dev_dedup.node_count)
    phi_l_dedup, phi_r_dedup = dev_dedup.contact_potentials(0.45)
    u_pert[0] = phi_l_dedup
    u_pert[dev_dedup.node_count - 1] = phi_r_dedup

    p_ref = DC1DProblem(dev_dedup, voltage_V=0.45, generation_scale=1.0)
    p_asm = AssembledDC1DProblem(dev_dedup, voltage_V=0.45, generation_scale=1.0)
    np.testing.assert_array_equal(p_ref.residual(u_pert), p_asm.residual(u_pert))
    np.testing.assert_array_equal(p_ref.fluxes(u_pert).electron.flux, p_asm.fluxes(u_pert).electron.flux)
    np.testing.assert_array_equal(p_ref.fluxes(u_pert).hole.flux, p_asm.fluxes(u_pert).hole.flux)
    np.testing.assert_array_equal(p_ref.jacobian(u_pert).toarray(), p_asm.jacobian(u_pert).toarray())

    # 4th-order central finite-difference check on AssembledDC1DProblem Jacobian
    j_exact = p_asm.jacobian(u_pert).toarray()
    j_fd = np.zeros_like(j_exact)
    eps = 1.0e-6
    for col in range(u_pert.size):
        step = np.zeros_like(u_pert)
        step[col] = eps
        r_p1 = p_asm.residual(u_pert + step)
        r_m1 = p_asm.residual(u_pert - step)
        r_p2 = p_asm.residual(u_pert + 2.0 * step)
        r_m2 = p_asm.residual(u_pert - 2.0 * step)
        j_fd[:, col] = (-r_p2 + 8.0 * r_p1 - 8.0 * r_m1 + r_m2) / (12.0 * eps)
    rel_jac_err = np.linalg.norm(j_exact - j_fd) / max(np.linalg.norm(j_exact), 1.0)
    assert rel_jac_err <= 1.0e-9

    # Dark equilibrium, illuminated J-V, and Backward Euler transient conservation
    steady_opts = SteadyOptions(
        continuity_tolerance_A_m2=1.0e-6,
        current_spread_tolerance_A_m2=1.0e-6,
        convergence_policy=POISSON_CONTACT_BALANCE_V1,
    )
    eq_dedup = solve_equilibrium(dev_dedup, options=steady_opts)
    assert float(np.max(np.abs(eq_dedup.contact_current_density_A_m2))) <= 1.0e-13

    jv_res = run_steady_jv(
        dev_dedup,
        np.linspace(0.0, 1.10, 23),
        generation_scale=1.0,
        options=steady_opts,
    )
    jsc_a_m2 = jv_res.metrics.short_circuit_current_A_m2
    voc_v = jv_res.metrics.open_circuit_voltage_V
    assert jsc_a_m2 is not None and jsc_a_m2 > 50.0
    assert voc_v is not None and 0.6 < voc_v < 1.15

    tr_times = np.concatenate(([0.0], np.geomspace(1.0e-9, 1.0e-6, 6)))
    tr_volts = np.linspace(0.0, 0.45, tr_times.size)
    tr_gen = np.concatenate(([0.0], np.ones(6)))
    tr_proto = TransientProtocol(times_s=tr_times, voltage_V=tr_volts, generation_scale=tr_gen)
    tr_opts = _transient_options(predictor="generation_inventory")
    tr_res = solve_transient(
        dev_dedup,
        eq_dedup,
        tr_proto,
        options=tr_opts,
        initial_options=steady_opts,
        method="backward_euler",
    )
    assert tr_res.completed and not tr_res.failed_attempts
    for attempt in tr_res.attempts:
        assert attempt.accepted and isinstance(attempt.accounting, dict)
        assert float(cast(float, attempt.accounting["continuity_defect_bound_A_m2"])) <= 1.0e-6
        assert abs(float(cast(float, attempt.accounting["global_charge_balance_A_m2"]))) <= 1.0e-6
        assert float(np.max(np.abs(np.asarray(attempt.accounting["maxwell_identity_error_A_m2"])))) <= 1.0e-9

    # 2. Two-sided trace all-V2 0-ULP parity and +3.0 V super-flatband forward injection
    spec_trace = _spectral_heterojunction_device(base, two_sided_trace=True)
    trace_v2_opts = P08CompilerOptions(
        dielectric_rule=LAYER_SERIES_CAPACITANCE_V2,
        band_step_mode=HOLE_SEPARATE_INCREMENT_V1,
        thermionic_mode=RECIPROCAL_PHYSICAL_THERMIONIC_V2,
        optical_model="beer_lambert_spectral",
    )
    dev_trace = compile_device_1d(
        spec_trace,
        intervals_per_layer=(6, 8, 6),
        interface_topology="two_sided_trace",
        **trace_v2_opts.compiler_kwargs_1d(),
    )
    u_tr = np.r_[dev_trace.reference_potential_V, np.zeros(2 * dev_trace.node_count)]
    u_tr[dev_trace.node_count :] += rng.normal(0.0, 0.1, size=2 * dev_trace.node_count)
    _, phi_r_tr = dev_trace.contact_potentials(0.5)
    u_tr[dev_trace.node_count - 1] = phi_r_tr
    pt_ref = DC1DProblem(dev_trace, voltage_V=0.5, generation_scale=1.0)
    pt_asm = AssembledDC1DProblem(dev_trace, voltage_V=0.5, generation_scale=1.0)
    np.testing.assert_array_equal(pt_ref.residual(u_tr), pt_asm.residual(u_tr))
    np.testing.assert_array_equal(pt_ref.jacobian(u_tr).toarray(), pt_asm.jacobian(u_tr).toarray())

    eq_trace = solve_equilibrium(dev_trace, options=steady_opts)
    assert float(np.max(np.abs(eq_trace.contact_current_density_A_m2))) <= 1.0e-13
    high_inj_opts = SteadyOptions(
        continuity_tolerance_A_m2=1.0e-3,
        current_spread_tolerance_A_m2=1.0e-3,
        max_bias_step_V=0.05,
        convergence_policy=POISSON_CONTACT_BALANCE_V1,
    )
    prev_state = solve_steady_state(dev_trace, voltage_V=0.0, generation_scale=1.0, options=high_inj_opts)
    for bias_v in np.linspace(0.05, 3.0, 60):
        prev_state = solve_steady_state(
            dev_trace,
            voltage_V=float(bias_v),
            generation_scale=1.0,
            initial_state=prev_state.state,
            options=high_inj_opts,
        )
        assert np.all(np.isfinite(prev_state.state.u))
        assert float(np.min(prev_state.state.electron_density_m3)) > 0.0
        assert float(np.min(prev_state.state.hole_density_m3)) > 0.0

    # 3. Spatial refinement convergence on compiled 1D all-V2 (N_per_layer = 4, 8, 16, 32)
    uniform_grid_dev = _with_edits(
        spec_dedup,
        lambda d: [g.update(alpha=1.0) for g in d.get("electrical_grid", [])],
    )
    j_by_n: list[float] = []
    for n_per in (4, 8, 16, 32):
        dev_n = compile_device_1d(
            uniform_grid_dev,
            intervals_per_layer=(n_per, n_per, n_per),
            **all_v2.compiler_kwargs_1d(),
        )
        ss_n = solve_steady_state(dev_n, voltage_V=0.65, generation_scale=1.0, options=steady_opts)
        j_by_n.append(float(ss_n.contact_current_density_A_m2[1]))
    err_4 = abs(j_by_n[0] - j_by_n[3])
    err_8 = abs(j_by_n[1] - j_by_n[3])
    err_16 = abs(j_by_n[2] - j_by_n[3])
    assert err_4 > err_8 > err_16 > 0.0
    empirical_order = math.log2(err_8 / err_16)
    assert empirical_order >= 1.7


# ---------------------------------------------------------------------------
# (d) Compiled 2D All-V2 Parity, Series Capacitor & 1D-vs-2D Column Consistency
# ---------------------------------------------------------------------------


def test_p08_05_compiled_2d_all_v2_parity_capacitor_and_column_consistency() -> None:
    """Compiled 2D all-V2 matches AssembledDC2DProblem 0-ULP, 1D column DC, and exact series capacitor."""
    base = load_device()
    spec_dedup = _spectral_heterojunction_device(base, two_sided_trace=False)
    all_v2 = P08CompilerOptions.all_prospective_v2()

    dev_1d = compile_device_1d(
        spec_dedup,
        intervals_per_layer=(5, 5, 5),
        **all_v2.compiler_kwargs_1d(),
    )
    sp = JV2DInput(kind="jv_2d", lateral_length="0.4 um", Nx=2, Ny_per_layer=5, lateral_bc="neumann")
    dev_2d = compile_device_2d(spec_dedup, sp, **all_v2.compiler_kwargs_2d())

    # 1. 0-ULP parity between DC2DProblem and AssembledDC2DProblem + FD Jacobian check
    rng = np.random.default_rng(20261117)
    u2 = dev_2d.extrude_coordinates(
        np.r_[dev_2d.column.reference_potential_V, np.zeros(2 * dev_2d.column.node_count)]
    ).copy()
    mask_int: NDArray[np.bool_] = np.ones(dev_2d.geometry.shape, dtype=bool)
    mask_int[0, :] = False
    mask_int[-1, :] = False
    u2[: dev_2d.node_count][mask_int.ravel()] += rng.normal(0.0, 0.02, size=int(np.sum(mask_int)))
    u2[dev_2d.node_count :] += rng.normal(0.0, 0.12, size=2 * dev_2d.node_count)
    ny, nx = dev_2d.geometry.shape
    top_nodes = np.arange(nx)
    bot_nodes = np.arange((ny - 1) * nx, ny * nx)
    _, phi_r_2d = dev_2d.column.contact_potentials(0.4)
    u2[bot_nodes] = phi_r_2d

    p2_ref = DC2DProblem(dev_2d, voltage_V=0.4, generation_scale=1.0)
    p2_asm = AssembledDC2DProblem(dev_2d, voltage_V=0.4, generation_scale=1.0)
    np.testing.assert_array_equal(p2_ref.residual(u2), p2_asm.residual(u2))
    np.testing.assert_array_equal(p2_ref.fluxes(u2).electron.flux, p2_asm.fluxes(u2).electron.flux)
    np.testing.assert_array_equal(p2_ref.fluxes(u2).hole.flux, p2_asm.fluxes(u2).hole.flux)
    np.testing.assert_array_equal(p2_ref.jacobian(u2).toarray(), p2_asm.jacobian(u2).toarray())

    j2_exact = p2_asm.jacobian(u2).toarray()
    j2_fd = np.zeros_like(j2_exact)
    eps = 1.0e-6
    for col in range(u2.size):
        step = np.zeros_like(u2)
        step[col] = eps
        j2_fd[:, col] = (p2_asm.residual(u2 + step) - p2_asm.residual(u2 - step)) / (2.0 * eps)
    assert np.linalg.norm(j2_exact - j2_fd) / max(np.linalg.norm(j2_exact), 1.0) <= 1.0e-7

    # 2. 1D-vs-2D uniform-column all-V2 steady-state agreement
    ss_1d = solve_steady_state(dev_1d, voltage_V=0.55, generation_scale=1.0)
    ss_2d = solve_steady_state_2d(
        dev_2d,
        voltage_V=0.55,
        generation_scale=1.0,
        initial_state=dev_2d.state(dev_2d.extrude_coordinates(ss_1d.state.u), voltage_V=0.55),
        options=Steady2DOptions(),
    )
    assert ss_2d.accepted and ss_2d.state is not None and ss_2d.diagnostics is not None
    j_2d = np.asarray(ss_2d.diagnostics["terminal_mean_current_density_A_m2"], dtype=np.float64)
    j_1d = np.asarray(ss_1d.contact_current_density_A_m2, dtype=np.float64)
    assert float(np.max(np.abs(j_2d - j_1d))) <= 1.0e-11 * max(float(np.max(np.abs(j_1d))), 1.0)
    phi_2d = ss_2d.state.potential_V
    np.testing.assert_allclose(phi_2d[:, 0], phi_2d[:, 1], rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(phi_2d[:, 0], ss_1d.state.potential_V, rtol=1.0e-11, atol=1.0e-11)

    # 3. Charge-free 3-layer compiled 2D series capacitor exactness under LAYER_SERIES_CAPACITANCE_V2
    inv_c_series = math.fsum(
        layer.thickness / (VACUUM_PERMITTIVITY_F_M * float(cast(float, dict(layer.parameters)["eps_r"])))
        for layer in spec_dedup.layers
        if layer.role != "substrate"
    )
    c_series_exact = 1.0 / inv_c_series
    # Solve pure 2D Laplace operator assembled from dev_2d.face_capacitance_F_m2 with 1 V drop
    pairs = dev_2d.geometry.contract_geometry.pairs
    cap_w = dev_2d.face_capacitance_F_m2 * dev_2d.geometry.contract_geometry.face_measures
    lap: NDArray[np.float64] = np.zeros((dev_2d.node_count, dev_2d.node_count), dtype=float)
    for (i_l, i_r), cw in zip(pairs, cap_w):
        lap[i_l, i_l] -= cw
        lap[i_l, i_r] += cw
        lap[i_r, i_l] += cw
        lap[i_r, i_r] -= cw
    rhs: NDArray[np.float64] = np.zeros(dev_2d.node_count, dtype=float)
    for idx in top_nodes:
        lap[idx, :] = 0.0
        lap[idx, idx] = 1.0
        rhs[idx] = 1.0
    for idx in bot_nodes:
        lap[idx, :] = 0.0
        lap[idx, idx] = 1.0
        rhs[idx] = 0.0
    phi_cap = np.linalg.solve(lap, rhs).reshape(ny, nx)
    dy0 = float(dev_2d.geometry.y_geometry.nodes_m[1] - dev_2d.geometry.y_geometry.nodes_m[0])
    eps0_layer = float(cast(float, dict(spec_dedup.layers[0].parameters)["eps_r"]))
    d_top = VACUUM_PERMITTIVITY_F_M * eps0_layer * (phi_cap[0, 0] - phi_cap[1, 0]) / dy0
    assert abs(d_top - c_series_exact) / c_series_exact <= 2.0e-14

    # 4. Compiled 2D ionic V2 0-ULP parity between Ionic2DProblem and AssembledIonic2DProblem
    ion_prep = prepare_confined_ion_device(diffusion_only=True)
    c_ion2d_v2 = compile_ionic_device_2d(
        ion_prep,
        sp,
        ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"),
        dielectric_rule=LAYER_SERIES_CAPACITANCE_V2,
        stack_geometry=PHYSICAL_HALF_ENDPOINT_V2,
        band_step_mode=HOLE_SEPARATE_INCREMENT_V1,
        thermionic_mode=RECIPROCAL_PHYSICAL_THERMIONIC_V2,
    )
    assert c_ion2d_v2.identity != PINNED_V1_IONIC_2D_SHA256
    u_ion2 = np.r_[
        c_ion2d_v2.electronic_component.extrude_coordinates(
            np.r_[
                c_ion2d_v2.electronic_component.column.reference_potential_V,
                np.zeros(2 * c_ion2d_v2.electronic_component.column.node_count),
            ]
        ),
        np.zeros(c_ion2d_v2.ion_node_count),
    ]
    u_ion2[c_ion2d_v2.node_count : 3 * c_ion2d_v2.node_count] += rng.normal(
        0.0, 0.08, size=2 * c_ion2d_v2.node_count
    )
    u_ion2[3 * c_ion2d_v2.node_count :] += rng.normal(0.0, 0.05, size=c_ion2d_v2.ion_node_count)
    _, phi_r_ion2 = c_ion2d_v2.electronic_component.column.contact_potentials(0.2)
    ny_i, nx_i = c_ion2d_v2.electronic_component.geometry.shape
    u_ion2[np.arange((ny_i - 1) * nx_i, ny_i * nx_i)] = phi_r_ion2
    pi2_ref = Ionic2DProblem(c_ion2d_v2, voltage_V=0.2, generation_scale=0.0)
    pi2_asm = AssembledIonic2DProblem(c_ion2d_v2, voltage_V=0.2, generation_scale=0.0)
    np.testing.assert_array_equal(pi2_ref.residual(u_ion2), pi2_asm.residual(u_ion2))
    np.testing.assert_allclose(pi2_ref.jacobian(u_ion2).toarray(), pi2_asm.jacobian(u_ion2).toarray(), rtol=1.0e-15, atol=0.0)
