"""Test support: legacy presets mapped onto the new optical binding.

Only tests import this module. Optical constants are read from the shipped
legacy CSV tables with the legacy reader's own ``np.loadtxt`` calls and
sampled by the NEW ``sample_nk_table``; the layer-to-spec rules (table,
constant index, ``sqrt(eps_r)`` fallback, substrate = optical-only) are the
new binding's. Graded CIGS slices take their n, k from the legacy optical
stack because composition binding is P04-11.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from solarlab.physics.generation import (
    OpticalLayerSpec,
    OpticalStackBinding,
    SpectralPhotonFlux,
    constant_index_optical_constants,
    sample_nk_table,
)

LEGACY_ROOT = Path(__file__).resolve().parents[4] / "perovskite-sim"
DATA = LEGACY_ROOT / "perovskite_sim" / "data"
CONFIGS = LEGACY_ROOT / "tests" / "fixtures" / "configs"
#: The wavelength window ``_compute_tmm_generation`` models.
WAVELENGTHS_NM = np.linspace(300.0, 1000.0, 200)


def config(name: str) -> str:
    return str(CONFIGS / name)


def load_stack(name: str):
    from perovskite_sim.models.config_loader import load_device_from_yaml

    return load_device_from_yaml(config(name))


def electrical_grid(stack, n_grid: int) -> np.ndarray:
    from perovskite_sim.discretization.grid import Layer, multilayer_grid
    from perovskite_sim.experiments.jv_sweep import _layer_node_counts
    from perovskite_sim.models.device import electrical_layers

    elec = electrical_layers(stack)
    return multilayer_grid([Layer(layer.thickness, n) for layer, n in zip(elec, _layer_node_counts(stack, n_grid))])


def am15g(wavelengths_nm: np.ndarray = WAVELENGTHS_NM) -> SpectralPhotonFlux:
    table = np.loadtxt(DATA / "am15g.csv", delimiter=",", comments="#")
    return SpectralPhotonFlux.from_table(table[:, 0], table[:, 1], wavelengths_nm, source_id="ASTM-G173-03-photon-flux")


def nk_table(material: str) -> np.ndarray:
    return np.loadtxt(DATA / "nk" / f"{material}.csv", delimiter=",", skiprows=1)


def binding_from_stack(stack, wavelengths_nm: np.ndarray = WAVELENGTHS_NM, **kwargs) -> OpticalStackBinding:
    specs = []
    for layer in stack.layers:
        params = layer.params
        if params.optical_material is not None:
            table = nk_table(params.optical_material)
            n, k = sample_nk_table(table[:, 0], table[:, 1], table[:, 2], wavelengths_nm,
                                   material=params.optical_material)
        else:
            n, k = constant_index_optical_constants(
                wavelengths_nm,
                absorption_coefficient_m1=params.alpha,
                refractive_index=params.n_optical,
                relative_permittivity=params.eps_r,
            )
        specs.append(OpticalLayerSpec(layer.name, layer.thickness, n, k, bool(params.incoherent),
                                      layer.role != "substrate"))
    return OpticalStackBinding(tuple(specs), am15g(wavelengths_nm), **kwargs)


def binding_from_legacy_optical_stack(stack, wavelengths_nm: np.ndarray = WAVELENGTHS_NM) -> OpticalStackBinding:
    from perovskite_sim.physics.optical_stack import build_device_optical_stack

    optical = build_device_optical_stack(stack, wavelengths_nm)
    specs = []
    for layer, part in zip(stack.layers, optical.physical_layer_slices):
        for tmm in optical.layers[part]:
            specs.append(OpticalLayerSpec(layer.name, tmm.d, tmm.n, tmm.k, tmm.incoherent, layer.role != "substrate"))
    return OpticalStackBinding(tuple(specs), am15g(wavelengths_nm))
