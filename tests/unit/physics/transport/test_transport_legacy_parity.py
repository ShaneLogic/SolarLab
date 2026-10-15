"""Historical MB/FD drift-diffusion face quantities and the FD gate (M class).

Bitwise against the legacy engine (read-only import): the carrier face
diffusivity (LD-03 NaN kept), the MB face currents on DOS-folded transport
potentials (before the thermionic cap, which belongs to P04-09), the
generalized FD currents and their face statistics, and the FD branch of
``carrier_face_currents`` on the qualified research preset. The capability
gate reproduces the legacy accept/reject decision on that preset and its
disqualified variants. Acceptance: the restricted FD SG reduces to the MB SG
in the non-degenerate limit.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import warnings

import numpy as np
import pytest

from perovskite_sim.discretization.fe_operators import sg_fluxes_n, sg_fluxes_p
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics import continuity as legacy_continuity
from perovskite_sim.physics import degenerate_transport as legacy_degenerate
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver import mol as legacy_mol

from solarlab.discretization import legacy_sg_current_n, legacy_sg_current_p
from solarlab.mesh import RIGHT_LAYER_WINS
from solarlab.mesh.grid import layer_node_masks
from solarlab.physics.statistics import BandStructure, dos_fold_reference_layer, fold_dos_into_transport_bands
from solarlab.physics.transport import (
    TransportCapabilityError, TransportCapabilityRequest, generalized_carrier_face_statistics,
    legacy_carrier_face_currents, legacy_face_diffusivity, legacy_fermi_dirac_face_currents,
    legacy_generalized_sg_current_n, legacy_generalized_sg_current_p, require_carrier_transport,
)

ROOT = Path(__file__).resolve().parents[4] / "perovskite-sim"
FD_PRESET = ROOT / "tests/fixtures/configs/csi_degenerate_pn_research.yaml"
EPS = float(np.finfo(float).eps)
Q = 1.602176634e-19


def same(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def material(path, loader, N, **kwargs):
    stack = loader(str(path))
    x = build_electrical_grid(stack, N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return stack, x, legacy_mol.build_material_arrays(x, stack, **kwargs)


def bands_of(stack, x, mat):
    layers = electrical_layers(stack)
    masks = layer_node_masks(x, [layer.thickness for layer in layers], RIGHT_LAYER_WINS)
    return fold_dos_into_transport_bands(
        mat.chi_phys, mat.Eg_phys, masks,
        [(layer.params.Nc300, layer.params.Nv300) for layer in layers],
        reference_layer=dos_fold_reference_layer([layer.role for layer in layers]),
        thermal_voltage_V=mat.V_T_device)


def random_state(x, rng, V):
    phi = V * (x - x[0]) / (x[-1] - x[0]) + rng.normal(scale=0.05, size=x.size)
    n = 10.0 ** rng.uniform(-5.0, 25.0, size=x.size)
    p = 10.0 ** rng.uniform(-5.0, 25.0, size=x.size)
    return phi, n, p


@pytest.mark.parametrize("path", ["configs/scaps_mirror_v2.yaml",
                                  "tests/fixtures/configs/scaps_mirror.yaml"])
def test_face_diffusivity_and_mb_face_currents_match_the_legacy_engine_bitwise(path):
    stack, x, mat = material(ROOT / path, load_scaps_yaml, 60)
    layers = electrical_layers(stack)
    masks = layer_node_masks(x, [layer.thickness for layer in layers], RIGHT_LAYER_WINS)
    D_n, D_p = np.zeros(x.size), np.zeros(x.size)
    for layer, mask in zip(layers, masks):
        D_n[mask] = layer.params.mu_n * mat.V_T_device
        D_p[mask] = layer.params.mu_p * mat.V_T_device
    assert same(legacy_face_diffusivity(D_n), mat.D_n_face)
    assert same(legacy_face_diffusivity(D_p), mat.D_p_face)
    bands = bands_of(stack, x, mat)
    params = dict(mat.carrier_params)
    params.pop("interface_faces", None)        # no thermionic cap: P04-09
    rng = np.random.default_rng(7)
    dx = np.diff(x)
    for _ in range(40):
        phi, n, p = random_state(x, rng, mat.V_bi_bc)
        J_n, J_p = legacy_carrier_face_currents(x, phi, n, p, mat.D_n_face, mat.D_p_face,
                                                mat.V_T_device, bands)
        assert same(J_n, sg_fluxes_n(phi + mat.chi, n, dx, mat.D_n_face, mat.V_T_device))
        assert same(J_p, sg_fluxes_p(phi + mat.chi + mat.Eg, p, dx, mat.D_p_face,
                                     mat.V_T_device))
        old_n, old_p = legacy_continuity.carrier_face_currents(x, phi, n, p, params)
        assert same(J_n, old_n) and same(J_p, old_p)


def test_carrier_face_diffusivity_keeps_the_zero_zero_nan_ld03():
    with np.errstate(invalid="ignore"):
        new = legacy_face_diffusivity(np.array([0.0, 0.0, 2.0e-4, 2.0e-4, 0.0]))
        old_node = np.array([0.0, 0.0, 2.0e-4, 2.0e-4, 0.0])
        old = 2.0 * old_node[:-1] * old_node[1:] / (old_node[:-1] + old_node[1:])
    assert np.isnan(new[0]) and same(new, old)
    assert new[1] == 0.0 and new[3] == 0.0
    assert new[2] == pytest.approx(2.0e-4, rel=2e-16)   # 2ab/(a+b) of equal values, rounded


def test_generalized_fd_currents_and_face_statistics_match_the_legacy_engine_bitwise():
    rng = np.random.default_rng(23)
    for _ in range(60):
        size = int(rng.integers(2, 30))
        x = np.cumsum(np.concatenate(([0.0], rng.uniform(1e-9, 2e-8, size=size - 1))))
        potential = rng.normal(scale=0.3, size=size)
        dos = float(10.0 ** rng.uniform(24.0, 26.0))
        density = dos * 10.0 ** rng.uniform(-12.0, 2.5, size=size)
        if rng.random() < 0.3:
            density[1:] = density[0]                  # removable equal-state faces
        mobility = float(10.0 ** rng.uniform(-5.0, -2.0))
        for model in ("maxwell_boltzmann", "fermi_dirac"):
            new_face = generalized_carrier_face_statistics(density, dos, statistics=model)
            old_face = legacy_degenerate.generalized_carrier_face_statistics(
                density, dos, statistics=model)
            for name in ("reduced_fermi_level", "log_occupation", "diffusion_enhancement"):
                assert same(getattr(new_face, name), getattr(old_face, name))
            args = (potential, density, np.diff(x), mobility, 0.025852, dos)
            assert same(legacy_generalized_sg_current_n(*args, statistics=model),
                        legacy_degenerate.generalized_sg_fluxes_n(*args, statistics=model))
            assert same(legacy_generalized_sg_current_p(*args, statistics=model),
                        legacy_degenerate.generalized_sg_fluxes_p(*args, statistics=model))
    with pytest.raises(ValueError, match="finite positive"):
        generalized_carrier_face_statistics(np.array([1.0, 0.0]), 1e25, statistics="fermi_dirac")


def test_fd_face_currents_match_the_legacy_research_branch_bitwise():
    stack, x, mat = material(FD_PRESET, load_device_from_yaml, 40,
                             carrier_statistics_transport="research_recombination_off")
    bands = bands_of(stack, x, mat)
    params = mat.carrier_params
    assert params["carrier_statistics"] == "fermi_dirac"
    rng = np.random.default_rng(5)
    for _ in range(20):
        phi = rng.normal(scale=0.2, size=x.size)
        n = float(mat.N_C_physical[0]) * 10.0 ** rng.uniform(-8.0, 1.5, size=x.size)
        p = float(mat.N_V_physical[0]) * 10.0 ** rng.uniform(-8.0, 1.5, size=x.size)
        new = legacy_fermi_dirac_face_currents(
            x, phi, n, p, mat.D_n_face, mat.D_p_face, mat.V_T_device, bands,
            conduction_dos_m3=params["N_C"], valence_dos_m3=params["N_V"],
            recombination_closure=params["degenerate_recombination_model"])
        old = legacy_continuity.carrier_face_currents(x, phi, n, p, params)
        assert same(new[0], old[0]) and same(new[1], old[1])
    # FD is never auto-paired with the MB recombination law.
    with pytest.raises(TransportCapabilityError, match="explicit compatible recombination"):
        legacy_fermi_dirac_face_currents(
            x, phi, n, p, mat.D_n_face, mat.D_p_face, mat.V_T_device, bands,
            conduction_dos_m3=params["N_C"], valence_dos_m3=params["N_V"],
            recombination_closure="maxwell_boltzmann")
    old_params = dict(params, degenerate_recombination_model="maxwell_boltzmann")
    with pytest.raises(ValueError, match="explicit compatible recombination"):
        legacy_continuity.carrier_face_currents(x, phi, n, p, old_params)


def request_for(stack, *, recombination_closure, illuminated=None):
    layers = electrical_layers(stack)
    homogeneous = ("eps_r", "chi", "Eg", "Nc300", "Nv300", "bgn_reference_energy_eV",
                   "bgn_reference_density_m3", "bgn_log_shape", "bgn_conduction_band_fraction")
    reference = layers[0].params
    return TransportCapabilityRequest(
        carrier_statistics=tuple(layer.params.carrier_statistics for layer in layers),
        dopant_ionization=tuple(layer.params.dopant_ionization_model for layer in layers),
        band_gap_narrowing=tuple(layer.params.band_gap_narrowing_model for layer in layers),
        recombination_closure=recombination_closure,
        built_in_potential_mode=stack.built_in_potential_mode,
        inhomogeneous_fields=tuple(name for name in homogeneous if any(
            getattr(layer.params, name) != getattr(reference, name) for layer in layers[1:])),
        optical_generation=float(stack.Phi) != 0.0 if illuminated is None else illuminated,
        band_grading=bool(stack.band_grading),
        contact_floors=bool(stack.flat_band_contacts or stack.flat_band_metal_contacts),
        selective_contacts=any(value is not None for value in (
            stack.S_n_left, stack.S_p_left, stack.S_n_right, stack.S_p_right)),
    )


@pytest.mark.parametrize("change, reason", [
    ({}, None),
    ({"Phi": 1.0e21}, "optical generation"),
    ({"flat_band_metal_contacts": True}, "calibrated contact floors"),
    ({"S_n_left": 1.0e3}, "selective contacts"),
])
def test_capability_gate_reproduces_the_legacy_research_decision(change, reason):
    stack = dataclasses.replace(load_device_from_yaml(str(FD_PRESET)), **change)
    x = build_electrical_grid(stack, 30)
    request = request_for(stack, recombination_closure="off")
    if reason is None:
        legacy_mol.build_material_arrays(
            x, stack, carrier_statistics_transport="research_recombination_off")
        assert require_carrier_transport(request) == "fermi_dirac"
        return
    with pytest.raises(legacy_mol.BulkCarrierStatisticsCapabilityError) as legacy_error:
        legacy_mol.build_material_arrays(
            x, stack, carrier_statistics_transport="research_recombination_off")
    with pytest.raises(TransportCapabilityError) as new_error:
        require_carrier_transport(request)
    assert reason in str(legacy_error.value) and reason in str(new_error.value)
    assert str(new_error.value).startswith("research degenerate transport topology rejected")


def test_fd_requires_the_semiconductor_work_function_contact_potential():
    stack = load_device_from_yaml(str(FD_PRESET))
    # The legacy stack already refuses FD with any other contact-potential source.
    with pytest.raises(ValueError, match="semiconductor_work_function"):
        dataclasses.replace(stack, built_in_potential_mode="legacy_manual", V_bi=0.9)
    request = dataclasses.replace(request_for(stack, recombination_closure="off"),
                                  built_in_potential_mode="legacy_manual")
    with pytest.raises(TransportCapabilityError, match="non-thermodynamic contact potential"):
        require_carrier_transport(request)


def test_capability_gate_rejects_unqualified_statistics_sources():
    stack = load_device_from_yaml(str(FD_PRESET))
    x = build_electrical_grid(stack, 30)
    # Default path with FD layers: rejected by both (no MB recombination pairing).
    with pytest.raises(legacy_mol.BulkCarrierStatisticsCapabilityError):
        legacy_mol.build_material_arrays(x, stack)
    with pytest.raises(TransportCapabilityError, match="never paired with the MB"):
        require_carrier_transport(request_for(stack, recombination_closure="maxwell_boltzmann"))
    base = request_for(stack, recombination_closure="off")
    for change, message in (
        ({"carrier_statistics": ("blakemore", "blakemore")}, "Blakemore"),
        ({"carrier_statistics": ("fermi_dirac", "maxwell_boltzmann")}, "one carrier-statistics"),
        ({"inhomogeneous_fields": ("Nc300",)}, "non-homojunction fields=Nc300"),
        ({"mobile_ions": True}, "mobile ions"),
        ({"environment_overrides": ("SOLARLAB_DOS_BAND",)}, "environment overrides"),
        ({"dopant_ionization": ("fully_ionized",)}, "every electrical layer"),
        ({"recombination_closure": "srh_degenerate"}, "unknown bulk recombination"),
    ):
        with pytest.raises(TransportCapabilityError, match=message):
            require_carrier_transport(dataclasses.replace(base, **change))
    default = TransportCapabilityRequest(("maxwell_boltzmann",) * 3, ("fully_ionized",) * 3,
                                         ("off",) * 3, "maxwell_boltzmann", None,
                                         optical_generation=True, mobile_ions=True)
    assert require_carrier_transport(default) == "maxwell_boltzmann"


def test_fd_face_currents_use_physical_edges_not_the_transport_fold():
    """FD transport reads the physical edges; the fold belongs to MB transport only."""
    x = np.linspace(0.0, 2.0e-7, 9)
    chi = np.full(x.size, 4.05)
    gap = np.full(x.size, 1.12)
    folded = BandStructure(chi, gap, chi + 0.01, gap - 0.02, dos_folded=True, reference_layer=0)
    unfolded = BandStructure.without_dos_fold(chi, gap)
    phi = np.linspace(0.0, 0.3, x.size)
    n = np.geomspace(1e20, 1e25, x.size)
    p = np.geomspace(1e24, 1e19, x.size)
    D = np.full(x.size - 1, 3e-3)
    kwargs = dict(conduction_dos_m3=2.8e25, valence_dos_m3=1.0e25, recombination_closure="off")
    a = legacy_fermi_dirac_face_currents(x, phi, n, p, D, D, 0.025852, folded, **kwargs)
    b = legacy_fermi_dirac_face_currents(x, phi, n, p, D, D, 0.025852, unfolded, **kwargs)
    assert same(a[0], b[0]) and same(a[1], b[1])


def test_restricted_fd_sg_reduces_to_mb_sg_in_the_non_degenerate_limit():
    rng = np.random.default_rng(71)
    for _ in range(40):
        size = int(rng.integers(3, 20))
        x = np.cumsum(np.concatenate(([0.0], rng.uniform(1e-9, 2e-8, size=size - 1))))
        potential = rng.normal(scale=0.2, size=size)
        dos = 1.0e25
        V_T = 0.025852
        D = float(10 ** rng.uniform(-6, -3))
        dx = np.diff(x)
        for decades, rtol in ((-25.0, 4e-11), (-18.0, 4e-8)):
            density = dos * 10.0 ** (decades + rng.uniform(-0.5, 0.5, size=size))
            fd_n = legacy_generalized_sg_current_n(potential, density, dx, D / V_T, V_T, dos,
                                                   statistics="fermi_dirac")
            mb_n = legacy_sg_current_n(potential, density, dx, D, V_T)
            fd_p = legacy_generalized_sg_current_p(potential, density, dx, D / V_T, V_T, dos,
                                                   statistics="fermi_dirac")
            mb_p = legacy_sg_current_p(potential, density, dx, D, V_T)
            for fd, mb in ((fd_n, mb_n), (fd_p, mb_p)):
                scale = np.abs(mb) + Q * D / dx * density.max() * 1e-14
                assert np.all(np.abs(fd - mb) <= rtol * scale)
        # Below the table (eta <= -40) the FD face statistics are exactly MB.
        deep = dos * np.exp(-41.0 - rng.uniform(0.0, 3.0, size=size))
        fd = legacy_generalized_sg_current_n(potential, deep, dx, D / V_T, V_T, dos,
                                             statistics="fermi_dirac")
        mb = legacy_generalized_sg_current_n(potential, deep, dx, D / V_T, V_T, dos,
                                             statistics="maxwell_boltzmann")
        assert fd.tobytes() == mb.tobytes()
        assert np.all(np.abs(fd - legacy_sg_current_n(potential, deep, dx, D, V_T))
                      <= 8 * EPS * (np.abs(fd) + Q * D / dx * deep.max()))
