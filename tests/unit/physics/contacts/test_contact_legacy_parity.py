"""Contacts: potential sources, reservoirs, exchange and thermodynamics (M class).

Every value is compared bitwise with the legacy engine (read-only import) on
shipped presets: the Poisson contact potential for each source (and the old
compatibility sentinel), junction polarity and the right boundary, the
doping/metal-floor reservoirs, the Robin current sign table (signed zero
included), the exchange-velocity resolution, the semiconductor contact
closures (MB/FD x fully/discrete) and the four-QFL thermodynamic assessment.
The device-to-input adaptation lives in this test, never in the product.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
import warnings

import numpy as np
import pytest

from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.models import device as legacy_device
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.mode import resolve_mode
from perovskite_sim.physics import contacts as legacy_contacts
from perovskite_sim.physics import temperature as legacy_temperature
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver import mol as legacy_mol

from solarlab.physics.contacts import (
    BuiltInPotentialSources, ContactClosureCapabilityError, ContactEndpoint, ContactMaterial,
    ContactPotentialBoundary, LegacyBandContact, assess_contact_thermodynamics,
    doping_equilibrium_densities, junction_polarity, legacy_band_fermi_level,
    legacy_selective_contact_current, metal_reservoir_floor, resolve_built_in_potential_mode,
    resolve_exchange_velocities, schottky_equilibrium_n, schottky_equilibrium_p,
    semiconductor_contact_state, semiconductor_work_function,
)

ROOT = Path(__file__).resolve().parents[4] / "perovskite-sim"
SCAPS = ("configs/scaps_mirror_v2.yaml", "tests/fixtures/configs/scaps_mirror.yaml",
         "tests/fixtures/configs/scaps_mirror_v2_robin_moderate.yaml",
         "tests/fixtures/configs/interface_charge_reference.yaml")
STANDARD = ("tests/fixtures/configs/nip_MAPbI3.yaml",
            "tests/fixtures/configs/ionmonger_benchmark.yaml",
            "tests/fixtures/configs/cigs_baseline.yaml",
            "tests/fixtures/configs/cSi_homojunction.yaml",
            "tests/fixtures/configs/tpv_physical_reference.yaml",
            "tests/fixtures/configs/solarscale_nip_band_aligned_iface.yaml",
            "tests/fixtures/configs/selective_contacts_demo.yaml")
FD_PRESETS = ("tests/fixtures/configs/csi_degenerate_pn_research.yaml",
              "tests/fixtures/configs/csi_incomplete_ionization_pn_research.yaml")


def load(path):
    return (load_scaps_yaml if path in SCAPS else load_device_from_yaml)(str(ROOT / path))


def same(a, b):
    return repr(a) == repr(b)


def material_arrays(stack, N=40, **kwargs):
    x = build_electrical_grid(stack, N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return legacy_mol.build_material_arrays(x, stack, **kwargs)


def nan_if_missing(value):
    return math.nan if value is None else float(value)


def contact_material(params, *, temperature, scaling):
    """The temperature resolution the legacy contact closures apply internally."""
    band_gap = (legacy_temperature.eg_at_T(params.Eg, temperature, params.varshni_alpha,
                                           params.varshni_beta) if scaling else float(params.Eg))
    dos_scale = (temperature / 300.0) ** 1.5 if scaling else 1.0
    return ContactMaterial(
        temperature_K=temperature, electron_affinity_eV=float(params.chi),
        band_gap_eV=band_gap,
        conduction_dos_m3=float(params.Nc300) * dos_scale if params.Nc300 else nan_if_missing(
            params.Nc300),
        valence_dos_m3=float(params.Nv300) * dos_scale if params.Nv300 else nan_if_missing(
            params.Nv300),
        donor_density_m3=float(params.N_D), acceptor_density_m3=float(params.N_A),
        carrier_statistics=params.carrier_statistics,
        dopant_ionization_model=params.dopant_ionization_model,
        donor_binding_energy_eV=params.donor_binding_energy_eV,
        acceptor_binding_energy_eV=params.acceptor_binding_energy_eV,
        donor_degeneracy=float(params.donor_degeneracy),
        acceptor_degeneracy=float(params.acceptor_degeneracy),
        band_gap_narrowing_model=params.band_gap_narrowing_model,
        bulk_trap_distribution=params.bulk_trap_distribution is not None,
        explicit_defects=charged_explicit_defects(params),
    )


def charged_explicit_defects(params):
    """The legacy dispatch: only charged, distributed or multivalent species reroute."""
    from perovskite_sim.models.multivalent_defects import MULTIVALENT_DEFECT_SCHEMA_VERSION

    multivalent = params.defect_schema_version == MULTIVALENT_DEFECT_SCHEMA_VERSION
    return multivalent or any(
        item.charge_transition in {"acceptor", "donor"} or item.distribution.kind != "single_level"
        for item in params.bulk_defects)


def potential_sources(stack):
    layers = legacy_device.electrical_layers(stack)
    sim_mode = resolve_mode(stack.mode)
    scaling = sim_mode.use_temperature_scaling
    temperature = float(stack.T) if scaling else 300.0
    historical_grading = getattr(stack, "band_grading", False)
    strict_grading = bool(stack.band_grading) and sim_mode.name != "legacy"
    left_h = legacy_device._edge_params(layers[0], "front", historical_grading)
    right_h = legacy_device._edge_params(layers[-1], "back", historical_grading)
    left_s = legacy_device._edge_params(layers[0], "front", strict_grading)
    right_s = legacy_device._edge_params(layers[-1], "back", strict_grading)

    def band(params):
        return LegacyBandContact(params.chi, params.Eg, params.N_D, params.N_A, params.ni)

    return BuiltInPotentialSources(
        mode=stack.built_in_potential_mode, manual_V_bi=stack.V_bi,
        flat_band_contacts=stack.flat_band_contacts,
        left_band=band(left_h), right_band=band(right_h),
        electrical_layer_bands=tuple((layer.params.chi, layer.params.Eg) for layer in layers
                                     if layer.params is not None),
        left_material=contact_material(left_s, temperature=temperature, scaling=scaling),
        right_material=contact_material(right_s, temperature=temperature, scaling=scaling),
        work_function_left_eV=stack.work_function_left_eV,
        work_function_right_eV=stack.work_function_right_eV,
    )


def outcome(call):
    try:
        return repr(call())
    except Exception as error:  # noqa: BLE001 - compare the legacy failure exactly
        return f"{type(error).__name__}: {error}"


def variants(stack):
    """The loaded stack plus every contact-potential source the legacy stack accepts."""
    yield stack
    changes = (
        dict(built_in_potential_mode=None, flat_band_contacts=True),
        dict(built_in_potential_mode=None, flat_band_contacts=False),
        dict(built_in_potential_mode="legacy_manual", V_bi=0.91),
        dict(built_in_potential_mode="metal_work_function", work_function_left_eV=5.1,
             work_function_right_eV=4.2),
        dict(built_in_potential_mode="semiconductor_work_function"),
        dict(built_in_potential_mode="semiconductor_work_function", T=250.0),
        dict(built_in_potential_mode="semiconductor_work_function", T=350.0),
    )
    for change in changes:
        try:   # e.g. FD layers already refuse every source but the semiconductor one
            variant = dataclasses.replace(stack, **change)
        except ValueError:
            continue
        yield variant


@pytest.mark.parametrize("path", SCAPS + STANDARD + FD_PRESETS)
def test_contact_potential_sources_match_the_legacy_resolution_bitwise(path):
    for stack in variants(load(path)):
        sources = potential_sources(stack)
        assert sources.resolved_mode == stack.resolved_built_in_potential_mode()
        assert outcome(sources.poisson_potential) == outcome(stack.poisson_built_in_potential)
        assert outcome(sources.operating_potential) == outcome(
            stack.operating_built_in_potential)
        assert outcome(sources.historical_estimate) == outcome(stack.compute_V_bi)
        if stack.built_in_potential_mode != "metal_work_function":
            assert outcome(sources.semiconductor_potential) == outcome(
                stack.compute_semiconductor_V_bi)


@pytest.mark.parametrize("path", SCAPS + STANDARD)
def test_poisson_boundary_and_junction_polarity_match_the_material_arrays(path):
    stack = load(path)
    mat = material_arrays(stack)
    sources = potential_sources(stack)
    boundary = ContactPotentialBoundary(sources.poisson_potential())
    assert same(boundary.V_bi_bc, mat.V_bi_bc)
    assert boundary.junction_polarity == mat.junction_polarity
    for V in (0.0, 0.3, -0.2, 1.05, 2.0):
        assert same(boundary.right(V), legacy_mol.poisson_right_boundary(mat, V))
    assert boundary.values(0.0)[0] == 0.0
    assert junction_polarity(-0.0) == 1.0 and junction_polarity(-1e-300) == -1.0


def test_historical_fermi_level_matches_the_legacy_helper():
    rng = np.random.default_rng(13)
    for _ in range(500):
        params = dataclasses.replace(
            load("tests/fixtures/configs/nip_MAPbI3.yaml").layers[0].params,
            chi=float(rng.uniform(2.0, 5.0)), Eg=float(rng.uniform(0.5, 3.0)),
            N_D=float(rng.choice([0.0, 10 ** rng.uniform(10, 26)])),
            N_A=float(rng.choice([0.0, 10 ** rng.uniform(10, 26)])),
            ni=float(10 ** rng.uniform(0, 18)))
        contact = LegacyBandContact(params.chi, params.Eg, params.N_D, params.N_A, params.ni)
        assert same(legacy_band_fermi_level(contact), legacy_device._fermi_level(params))


@pytest.mark.parametrize("path", SCAPS + STANDARD)
def test_reservoir_densities_match_the_material_arrays_bitwise(path):
    for flat_band_metal in (False, True):
        stack = dataclasses.replace(load(path), flat_band_metal_contacts=flat_band_metal,
                                    contact_phi_B_eV=0.12 if flat_band_metal else 0.0)
        mat = material_arrays(stack)
        n_L, p_L = doping_equilibrium_densities(mat.N_D[0], mat.N_A[0], mat.ni_sq[0])
        n_R, p_R = doping_equilibrium_densities(mat.N_D[-1], mat.N_A[-1], mat.ni_sq[-1])
        if flat_band_metal and resolve_mode(stack.mode).name != "legacy":
            layers = legacy_device.electrical_layers(stack)
            for side, params, node in (("right", layers[-1].params, -1),
                                       ("left", layers[0].params, 0)):
                floored = metal_reservoir_floor(
                    *(n_R, p_R) if side == "right" else (n_L, p_L),
                    intrinsic_density_squared_m6=float(mat.ni_sq[node]),
                    donor_density_m3=mat.N_D[node], acceptor_density_m3=mat.N_A[node],
                    conduction_dos_m3=params.Nc300, valence_dos_m3=params.Nv300,
                    barrier_eV=stack.contact_phi_B_eV, thermal_voltage_V=mat.V_T_device)
                if side == "right":
                    n_R, p_R = floored
                else:
                    n_L, p_L = floored
        assert same((n_L, p_L, n_R, p_R), (mat.n_L, mat.p_L, mat.n_R, mat.p_R))


def test_selective_contact_current_matches_the_legacy_sign_table_bitwise():
    rng = np.random.default_rng(19)
    for carrier in ("n", "p"):
        for side in ("left", "right"):
            for S in (0.0, 1e-3, 1.0e5, 3.7e7):
                for _ in range(50):
                    density = float(10 ** rng.uniform(-5, 25))
                    reservoir = float(10 ** rng.uniform(-5, 25))
                    new = legacy_selective_contact_current(density, reservoir, S,
                                                           carrier=carrier, side=side)
                    old = legacy_contacts.selective_contact_flux(density, reservoir, S,
                                                                 carrier=carrier, side=side)
                    assert same(new, old) and type(new) is type(old)
                array = 10 ** rng.uniform(0, 20, size=6)
                new = legacy_selective_contact_current(array, 1e10, S, carrier=carrier, side=side)
                old = legacy_contacts.selective_contact_flux(array, 1e10, S, carrier=carrier,
                                                             side=side)
                assert np.asarray(new).tobytes() == np.asarray(old).tobytes()
    # S = 0 keeps the historical signed zero.
    assert math.copysign(1.0, legacy_selective_contact_current(1.0, 2.0, 0.0, carrier="n",
                                                               side="left")) == -1.0
    for bad in ({"carrier": "x", "side": "left"}, {"carrier": "n", "side": "top"}):
        with pytest.raises(ValueError):
            legacy_selective_contact_current(1.0, 1.0, 1.0, **bad)


def test_schottky_densities_match_the_legacy_helpers_bitwise():
    for args in ((1.0e25, 0.6, 0.025852), (2.2e24, 0.0, 0.03), (8e25, 1.4, 0.0258)):
        assert same(schottky_equilibrium_n(*args), legacy_contacts.schottky_equilibrium_n(*args))
        assert same(schottky_equilibrium_p(*args), legacy_contacts.schottky_equilibrium_p(*args))


@pytest.mark.parametrize("path", SCAPS + STANDARD)
def test_exchange_velocity_resolution_matches_the_legacy_rule(path):
    base = load(path)
    for flat_band in (False, True):
        for mode in ("legacy", "fast", "full"):
            for velocities in ((None, None, None, None), (1e3, None, 0.0, 2e5)):
                stack = dataclasses.replace(base, flat_band_contacts=flat_band, mode=mode,
                                            S_n_left=velocities[0], S_p_left=velocities[1],
                                            S_n_right=velocities[2], S_p_right=velocities[3])
                assert resolve_exchange_velocities(
                    velocities, flat_band_contacts=stack.flat_band_contacts,
                    selective_contacts_enabled=resolve_mode(stack.mode).use_selective_contacts,
                ) == legacy_contacts.resolved_contact_velocities(stack)


def state_fields(state):
    return (state.statistics, state.electron_affinity_eV, state.band_gap_eV,
            state.work_function_eV, repr(state.neutrality), state.band_gap_narrowing_eV,
            state.conduction_band_shift_eV, state.valence_band_shift_eV)


@pytest.mark.parametrize("path", FD_PRESETS + SCAPS[:2] + ("tests/fixtures/configs/tpv_physical_reference.yaml",))
@pytest.mark.parametrize("temperature, scaling", [(300.0, False), (350.0, True), (250.0, True)])
def test_semiconductor_contact_closures_match_the_legacy_engine_bitwise(path, temperature,
                                                                         scaling):
    for layer in legacy_device.electrical_layers(load(path)):
        params = layer.params
        if not (params.Nc300 and params.Nv300 and params.chi > 0 and params.Eg > 0):
            continue
        material = contact_material(params, temperature=temperature, scaling=scaling)
        old = legacy_contacts.build_semiconductor_contact_state(
            params, temperature_K=temperature, use_temperature_scaling=scaling)
        new = semiconductor_contact_state(material)
        assert same(state_fields(new), state_fields(old))
        assert same(semiconductor_work_function(material), legacy_device._semiconductor_work_function(
            params, temperature, scaling))


def test_unowned_contact_closures_are_rejected_not_dropped():
    stack = load("tests/fixtures/configs/csi_incomplete_ionization_bgn_pn_research.yaml")
    params = legacy_device.electrical_layers(stack)[0].params
    assert params.band_gap_narrowing_model != "off"
    legacy_contacts.build_semiconductor_contact_state(params, temperature_K=300.0,
                                                      use_temperature_scaling=False)
    material = contact_material(params, temperature=300.0, scaling=False)
    with pytest.raises(ContactClosureCapabilityError, match="P04-11"):
        semiconductor_contact_state(material)
    with pytest.raises(ContactClosureCapabilityError, match="P04-11"):
        semiconductor_work_function(material)
    for flags in ({"bulk_trap_distribution": True}, {"explicit_defects": True}):
        traps = dataclasses.replace(contact_material(
            legacy_device.electrical_layers(load(FD_PRESETS[0]))[0].params,
            temperature=300.0, scaling=False), **flags)
        with pytest.raises(ContactClosureCapabilityError, match="P04-10"):
            semiconductor_contact_state(traps)
    with pytest.raises(ValueError, match="unknown built-in-potential mode"):
        resolve_built_in_potential_mode("flat_band", flat_band_contacts=False)


def assessment_inputs(stack, mat):
    sources = potential_sources(stack)
    try:
        semiconductor = sources.semiconductor_potential()
        if not np.isfinite(float(mat.V_bi_bc - semiconductor)):
            semiconductor = None
    except (AttributeError, TypeError, ValueError, OverflowError):
        semiconductor = None

    def endpoint(node, n, p):
        def pick(array):
            return None if array is None else float(np.asarray(array, dtype=float)[node])
        return ContactEndpoint(float(n), float(p), pick(mat.N_C_physical), pick(mat.N_V_physical),
                               pick(mat.chi_phys), pick(mat.Eg_phys))

    mode = stack.resolved_built_in_potential_mode()
    layers = legacy_device.electrical_layers(stack)
    return dict(
        built_in_potential_mode=mode, left=endpoint(0, mat.n_L, mat.p_L),
        right=endpoint(-1, mat.n_R, mat.p_R), V_bi_bc=float(mat.V_bi_bc),
        thermal_voltage_V=mat.V_T_device, statistics=mat.carrier_statistics,
        semiconductor_potential_V=semiconductor,
        metal_work_functions_eV=(stack.work_function_left_eV, stack.work_function_right_eV)
        if mode == "metal_work_function" else None,
        has_band_reference=any(layer.params is not None and (
            layer.params.chi != 0.0 or layer.params.Eg != 0.0) for layer in layers),
    )


def certificate_fields(value):
    return (value.status, value.built_in_potential_mode, value.tolerance_eV,
            value.fermi_level_span_eV, value.potential_mismatch_V,
            value.metal_work_function_mismatch_eV, value.contact_quasi_fermi_levels_eV,
            value.message)


@pytest.mark.parametrize("path", SCAPS + STANDARD)
def test_contact_thermodynamic_assessment_matches_the_legacy_certificate(path):
    assessed = 0
    for stack in variants(load(path)):
        try:
            mat = material_arrays(stack, 24)
        except ValueError:      # legacy cannot build this source (e.g. no endpoint DOS)
            continue
        assessed += 1
        old = legacy_contacts.assess_contact_thermodynamics(stack, mat)
        new = assess_contact_thermodynamics(**assessment_inputs(stack, mat))
        assert same(certificate_fields(new), certificate_fields(old))
    assert assessed >= 5
