"""Physical band edges vs DOS-folded transport potentials (M class + LD-05).

M class: the folded transport affinity/gap equal the legacy
``build_material_arrays`` ``chi``/``Eg`` bitwise and the physical edges equal
its ``chi_phys``/``Eg_phys``, on DOS-carrying presets (with and without a
substrate prefix) and on DOS-free presets (no fold). The shared interface node
takes the right layer's correction (LD-05, kept explicitly).
"""

from __future__ import annotations

from pathlib import Path
import math
import warnings

import numpy as np
import pytest

from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import build_material_arrays

from solarlab.mesh import LEFT_LAYER_WINS, RIGHT_LAYER_WINS, multilayer_grid, node_owner
from solarlab.mesh.grid import layer_node_masks
from solarlab.physics.statistics import (
    BandStructure, LayerBandInput, dos_fold_reference_layer, fold_dos_into_transport_bands,
    layered_band_structure,
)

ROOT = Path(__file__).resolve().parents[4]
PRESETS = ROOT / "perovskite-sim"
SCAPS_PRESETS = ("configs/scaps_mirror_v2.yaml", "tests/fixtures/configs/scaps_mirror.yaml",
                 "tests/fixtures/configs/scaps_mirror_v2_robin_strong.yaml",
                 "tests/fixtures/configs/interface_charge_reference.yaml")
STANDARD_PRESETS = ("tests/fixtures/configs/nip_MAPbI3.yaml",
                    "tests/fixtures/configs/ionmonger_benchmark.yaml",
                    "tests/fixtures/configs/cigs_baseline.yaml",
                    "tests/fixtures/configs/tpv_physical_reference.yaml",
                    "tests/fixtures/configs/scaps_defect_s0_neutral.yaml")


def same(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def legacy_material(path, N):
    loader = load_scaps_yaml if path in SCAPS_PRESETS else load_device_from_yaml
    stack = loader(str(PRESETS / path))
    x = build_electrical_grid(stack, N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return stack, x, build_material_arrays(x, stack)


def fold_like_the_legacy_builder(stack, x, mat):
    layers = electrical_layers(stack)
    masks = layer_node_masks(x, [layer.thickness for layer in layers], RIGHT_LAYER_WINS)
    return fold_dos_into_transport_bands(
        mat.chi_phys, mat.Eg_phys, masks,
        [(layer.params.Nc300, layer.params.Nv300) for layer in layers],
        reference_layer=dos_fold_reference_layer([layer.role for layer in layers]),
        thermal_voltage_V=mat.V_T_device)


@pytest.mark.parametrize("N", [40, 97])
@pytest.mark.parametrize("path", SCAPS_PRESETS + STANDARD_PRESETS)
def test_dos_fold_matches_the_legacy_material_arrays_bitwise(path, N):
    stack, x, mat = legacy_material(path, N)
    bands = fold_like_the_legacy_builder(stack, x, mat)
    assert same(bands.transport_electron_affinity_eV, mat.chi)
    assert same(bands.transport_band_gap_eV, mat.Eg)
    assert same(bands.physical_electron_affinity_eV, mat.chi_phys)
    assert same(bands.physical_band_gap_eV, mat.Eg_phys)
    if path in SCAPS_PRESETS:   # DOS contrast: the transport arrays really moved
        assert bands.dos_folded and not same(mat.chi, mat.chi_phys)
    phi = np.linspace(0.0, 1.0, x.size)
    psi_n, psi_p = bands.transport_potentials(phi)
    assert same(psi_n, phi + mat.chi) and same(psi_p, phi + mat.chi + mat.Eg)


def test_physical_edges_and_transport_potentials_are_never_interchanged():
    stack, x, mat = legacy_material("configs/scaps_mirror_v2.yaml", 60)
    bands = fold_like_the_legacy_builder(stack, x, mat)
    phi = np.linspace(0.0, 1.3, x.size)
    conduction, valence = bands.physical_band_edges(phi)
    assert same(conduction, -phi - mat.chi_phys)
    assert same(valence, -phi - mat.chi_phys - mat.Eg_phys)
    # The fold is a transport term: chi_t + Eg_t shifts by -dV, never the edges.
    reference = dos_fold_reference_layer([layer.role for layer in electrical_layers(stack)])
    layers = electrical_layers(stack)
    owner = node_owner(x, [layer.thickness for layer in layers], RIGHT_LAYER_WINS)
    ref = layers[reference].params
    for node in range(x.size):
        params = layers[owner[node]].params
        dC = mat.V_T_device * math.log(params.Nc300 / ref.Nc300)
        assert bands.transport_electron_affinity_eV[node] == pytest.approx(
            mat.chi_phys[node] + dC, abs=1e-15)
    with pytest.raises(ValueError, match="reuse the physical edges"):
        BandStructure(mat.chi_phys, mat.Eg_phys, mat.chi, mat.Eg, dos_folded=False,
                      reference_layer=None)


def test_shared_interface_node_takes_the_right_layer_correction_ld05():
    thicknesses = (1.0e-7, 2.0e-7)
    x = multilayer_grid([(thicknesses[0], 6), (thicknesses[1], 8)])
    layers = [LayerBandInput(thicknesses[0], 3.6, 1.8, 4.0e25, 1.0e25, "HTL"),
              LayerBandInput(thicknesses[1], 3.9, 1.6, 1.0e25, 2.0e25, "absorber")]
    bands = layered_band_structure(x, layers, thermal_voltage_V=0.025852, fold_dos=True)
    shared = int(np.argmin(np.abs(x - thicknesses[0])))
    right_owner = node_owner(x, thicknesses, RIGHT_LAYER_WINS)
    left_owner = node_owner(x, thicknesses, LEFT_LAYER_WINS)
    assert right_owner[shared] == 1 and left_owner[shared] == 0
    # The absorber is the reference: its nodes, the shared node included, are unshifted.
    assert bands.transport_electron_affinity_eV[shared] == 3.9
    assert bands.physical_electron_affinity_eV[shared] == 3.9
    assert bands.transport_electron_affinity_eV[shared - 1] == pytest.approx(
        3.6 + 0.025852 * math.log(4.0), abs=1e-15)
    assert dos_fold_reference_layer([layer.role for layer in layers]) == 1


def test_fold_skips_missing_dos_and_needs_a_reference_with_dos():
    x = np.linspace(0.0, 3.0e-7, 7)
    masks = layer_node_masks(x, [1.0e-7, 1.0e-7, 1.0e-7], RIGHT_LAYER_WINS)
    chi = np.array([3.6, 3.6, 3.9, 3.9, 4.0, 4.0, 4.0])
    gap = np.array([1.8, 1.8, 1.6, 1.6, 2.0, 2.0, 2.0])
    skipped = fold_dos_into_transport_bands(
        chi, gap, masks, [(None, None), (1e25, 1e25), (2e25, 0.0)], reference_layer=1,
        thermal_voltage_V=0.025)
    assert skipped.dos_folded and same(skipped.transport_electron_affinity_eV, chi)
    no_reference = fold_dos_into_transport_bands(
        chi, gap, masks, [(2e25, 2e25), (None, 1e25), (2e25, 2e25)], reference_layer=1,
        thermal_voltage_V=0.025)
    assert not no_reference.dos_folded
    assert same(no_reference.transport_electron_affinity_eV, chi)
    assert dos_fold_reference_layer(["HTL", "ETL"]) == 0
    overlapping = [np.ones(7, dtype=bool), np.zeros(7, dtype=bool), np.ones(7, dtype=bool)]
    with pytest.raises(ValueError, match="partition"):
        fold_dos_into_transport_bands(chi, gap, overlapping, [(1e25, 1e25)] * 3,
                                      reference_layer=0, thermal_voltage_V=0.025)
    with pytest.raises(ValueError, match="reference_layer"):
        fold_dos_into_transport_bands(chi, gap, masks, [(1e25, 1e25)] * 3, reference_layer=3,
                                      thermal_voltage_V=0.025)
    with pytest.raises(ValueError, match="math domain|positive"):
        fold_dos_into_transport_bands(chi, gap, masks, [(1e25, 1e25), (1e25, 1e25),
                                                        (-1e25, 1e25)],
                                      reference_layer=0, thermal_voltage_V=0.025)
