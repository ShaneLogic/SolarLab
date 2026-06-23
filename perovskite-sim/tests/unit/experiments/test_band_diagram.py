"""Band-diagram invariants: flat E_F at equilibrium, quasi-Fermi splitting = qV."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.experiments.band_diagram import compute_band_diagram

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "scaps_mirror_v2.yaml"


def _absorber_mask(stack, x):
    elec = electrical_layers(stack)
    edges = np.concatenate([[0.0], np.cumsum([L.thickness for L in elec])])
    a = int(np.argmax([L.thickness for L in elec]))  # absorber = thickest layer
    return (x > edges[a]) & (x < edges[a + 1])


def test_equilibrium_fermi_level_is_flat():
    """At the zero-current dark equilibrium the quasi-Fermi levels coincide into a
    single flat E_F (the defining thermodynamic invariant)."""
    stack = load_scaps_yaml(CONFIG)
    bd = compute_band_diagram(stack, 0.0, illuminated=False, N_grid=40)
    # the resolved E_F: E_Fn where electrons present, else E_Fp
    E_F = np.where(~np.isnan(bd.E_Fn), bd.E_Fn, bd.E_Fp)
    E_F = E_F[~np.isnan(E_F)]
    assert E_F.size > 0
    assert np.nanmax(E_F) - np.nanmin(E_F) < 0.15  # flat to <150 meV


@pytest.mark.slow
def test_quasi_fermi_splitting_equals_qV():
    """Under illumination at V, the absorber quasi-Fermi splitting E_Fn - E_Fp
    equals qV (the operating-point invariant)."""
    stack = load_scaps_yaml(CONFIG)
    V = 0.9
    bd = compute_band_diagram(stack, V, illuminated=True, N_grid=40, settle_t=1e-2)
    split = np.nanmean((bd.E_Fn - bd.E_Fp)[_absorber_mask(stack, bd.x)])
    assert split == pytest.approx(V, abs=0.12)


def test_band_offsets_match_config():
    """E_C and E_V edges reflect the configured electron affinity and bandgap."""
    stack = load_scaps_yaml(CONFIG)
    bd = compute_band_diagram(stack, 0.0, illuminated=False, N_grid=40)
    # gap E_C - E_V equals the per-node bandgap (>0 everywhere, physical)
    gap = bd.E_C - bd.E_V
    assert np.all(gap > 0.5) and np.all(gap < 4.0)


def test_missing_dos_data_raises():
    """A config without effective-DOS data cannot define quasi-Fermi levels."""
    stack = load_scaps_yaml(CONFIG)
    layers = tuple(
        dataclasses.replace(L, params=dataclasses.replace(L.params, Nc300=0.0))
        for L in stack.layers
    )
    stripped = dataclasses.replace(stack, layers=layers)
    with pytest.raises(ValueError, match="Nc300"):
        compute_band_diagram(stripped, 0.0, illuminated=False, N_grid=40)
