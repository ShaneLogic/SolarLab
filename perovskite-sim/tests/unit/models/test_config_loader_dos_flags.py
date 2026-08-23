"""Standard-loader wiring for the validated physics flags (2026-06).

``dos_band_potentials`` and ``flat_band_contacts`` were only parseable from
SCAPS-shape YAMLs (scaps_compat/loader.py); a standard config exploring a new
device system could carry per-layer ``Nc300``/``Nv300`` (already parsed) but
had no declarative way to enable the DOS-corrected transport. These tests pin
the device-level keys on ``models/config_loader.py`` and that the DOS fold
actually activates end-to-end from a standard YAML.
"""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.models.config_loader import (
    load_device_from_yaml,
    material_params_from_dict,
)
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.physics.statistics import (
    DISCRETE_LEVEL,
    FERMI_DIRAC,
    FULLY_IONIZED,
    MAXWELL_BOLTZMANN,
)
from perovskite_sim.solver.mol import (
    BulkCarrierStatisticsCapabilityError,
    build_material_arrays,
)

_BASE = "configs/nip_MAPbI3.yaml"


def _write_cfg(tmp_path, dev_extra=None, dos_layers=False):
    cfg = yaml.safe_load(open(_BASE))
    if dev_extra:
        cfg["device"].update(dev_extra)
    if dos_layers:
        # DOS contrast vs the absorber so the fold has something to shift
        for layer, (nc, nv) in zip(cfg["layers"],
                                   [(2.0e24, 1.0e25), (8.0e24, 8.0e24), (5.0e25, 1.0e25)]):
            layer["Nc300"] = nc
            layer["Nv300"] = nv
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def _build(stack):
    layers = [Layer(thickness=L.thickness, N=10) for L in stack.layers]
    return multilayer_grid(layers), stack


def test_flags_default(tmp_path):
    # 2026-06: dos_band_potentials defaults ON (correct heterojunction
    # transport; a no-op without per-layer DOS data anyway). flat_band_contacts
    # stays default-off.
    stack = load_device_from_yaml(_write_cfg(tmp_path))
    assert stack.dos_band_potentials is True
    assert stack.flat_band_contacts is False


def test_flags_roundtrip_from_yaml(tmp_path):
    stack = load_device_from_yaml(_write_cfg(
        tmp_path, dev_extra={"dos_band_potentials": True,
                             "flat_band_contacts": True}))
    assert stack.dos_band_potentials is True
    assert stack.flat_band_contacts is True


def test_dos_fold_activates_from_standard_yaml(tmp_path):
    """End-to-end: standard YAML with per-layer Nc300/Nv300 + the flag folds
    the DOS terms into the cached chi arrays (transport-layer nodes shift,
    absorber reference nodes do not)."""
    off = load_device_from_yaml(_write_cfg(
        tmp_path, dev_extra={"dos_band_potentials": False}, dos_layers=True))
    on = load_device_from_yaml(_write_cfg(
        tmp_path, dev_extra={"dos_band_potentials": True}, dos_layers=True))
    x, _ = _build(off)
    chi_off = build_material_arrays(x, off).chi
    chi_on = build_material_arrays(x, on).chi
    assert not np.allclose(chi_off, chi_on), "DOS fold must shift chi"
    # absorber is the fold reference — its nodes stay put
    n3 = len(x) // 3
    np.testing.assert_allclose(chi_on[n3 + 2: 2 * n3 - 2],
                               chi_off[n3 + 2: 2 * n3 - 2])


def test_no_dos_data_is_bit_identical_even_with_flag(tmp_path):
    """Layers without Nc300/Nv300 are skipped by the fold — a legacy config
    with the flag on stays bit-identical."""
    off = load_device_from_yaml(_write_cfg(tmp_path))
    on = load_device_from_yaml(_write_cfg(
        tmp_path, dev_extra={"dos_band_potentials": True}))
    x, _ = _build(off)
    np.testing.assert_array_equal(build_material_arrays(x, off).chi,
                                  build_material_arrays(x, on).chi)


def test_layer_carrier_statistics_default_and_fd_opt_in_are_strict():
    cfg = yaml.safe_load(open(_BASE))
    layer = dict(cfg["layers"][1])
    assert (
        material_params_from_dict(layer).carrier_statistics
        == MAXWELL_BOLTZMANN
    )

    layer.update(
        carrier_statistics=" Fermi_Dirac ",
        Eg=1.6,
        Nc300=2.8e25,
        Nv300=1.04e25,
    )
    assert material_params_from_dict(layer).carrier_statistics == FERMI_DIRAC

    layer["carrier_statistics"] = "fd"
    with pytest.raises(ValueError, match="carrier statistics"):
        material_params_from_dict(layer)
    layer["carrier_statistics"] = 1
    with pytest.raises(ValueError, match="must be a string"):
        material_params_from_dict(layer)


def test_layer_dopant_ionization_default_and_discrete_opt_in_are_strict():
    cfg = yaml.safe_load(open(_BASE))
    layer = dict(cfg["layers"][1])
    baseline = material_params_from_dict(layer)
    assert baseline.dopant_ionization_model == FULLY_IONIZED
    assert baseline.donor_binding_energy_eV is None
    assert baseline.acceptor_binding_energy_eV is None

    layer.update(
        dopant_ionization_model=" Discrete_Level ",
        N_D=1.0e23,
        N_A=0.0,
        Eg=1.6,
        Nc300=2.8e25,
        Nv300=1.04e25,
        donor_binding_energy_eV=0.045,
    )
    parsed = material_params_from_dict(layer)
    assert parsed.dopant_ionization_model == DISCRETE_LEVEL
    assert parsed.donor_binding_energy_eV == 0.045
    assert parsed.donor_degeneracy == 2.0

    layer["dopant_ionization_model"] = "freeze_out"
    with pytest.raises(ValueError, match="dopant ionization model"):
        material_params_from_dict(layer)


def test_discrete_dopant_level_requires_active_species_binding_energy():
    cfg = yaml.safe_load(open(_BASE))
    layer = dict(cfg["layers"][1])
    layer.update(
        dopant_ionization_model=DISCRETE_LEVEL,
        N_D=1.0e23,
        N_A=0.0,
        Eg=1.6,
        Nc300=2.8e25,
        Nv300=1.04e25,
    )
    with pytest.raises(ValueError, match="active discrete donors"):
        material_params_from_dict(layer)


def test_default_material_assembly_rejects_incomplete_ionization():
    cfg = yaml.safe_load(open(_BASE))
    layer = dict(cfg["layers"][1])
    layer.update(
        dopant_ionization_model=DISCRETE_LEVEL,
        N_D=1.0e23,
        N_A=0.0,
        Eg=1.6,
        Nc300=2.8e25,
        Nv300=1.04e25,
        donor_binding_energy_eV=0.045,
    )
    params = material_params_from_dict(layer)
    stack = DeviceStack(
        layers=(LayerSpec("discrete", 100.0e-9, params, "absorber"),),
        Phi=0.0,
        built_in_potential_mode="semiconductor_work_function",
    )
    grid = multilayer_grid((Layer(100.0e-9, 8),))
    with pytest.raises(
        BulkCarrierStatisticsCapabilityError,
        match="incomplete-ionization transport closure is not enabled",
    ):
        build_material_arrays(grid, stack)
