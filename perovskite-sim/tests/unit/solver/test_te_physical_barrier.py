"""Thermionic emission must cross the PHYSICAL band step, not the folded one.

With ``dos_band_potentials`` active, ``MaterialArrays.chi`` and ``.Eg`` are
transport potentials: they carry ``V_T ln(N_C/N_C_ref)`` so that the
Scharfetter-Gummel flux is correct under Boltzmann statistics. That shift is
a device for getting drift-diffusion right, not a real energy displacement —
the conduction and valence edges have not moved. Thermionic emission crosses
the real step, so feeding it the folded arrays puts the wrong number in a
Boltzmann exponent.

Measured on scaps_mirror_v2 at N_grid = 30, capped faces (9, 19):

    face   dE_c folded   dE_c physical   dE_v folded   dE_v physical
      9      -1.4568       -1.5400        +0.0968       +0.1800
     19      -0.2138       -0.1600        -0.4762       -0.5300

The hole barrier at face 9 is off by nearly a factor two. The fold can also
move a step across the 50 meV capping threshold in either direction, so the
face-selection test in ``build_material_arrays`` reads the physical edges
too.

WHY THIS IS INERT ON CURRENT DEFAULTS, and why that is not an argument
against fixing it: the legacy density-weighted bound makes ``|J_te|``
~1e28-1e35, so the cap essentially never binds (measured 0 binds in 80
checks across five shipped presets) and the barrier value is unused. The
moment ``te_physical_norm`` brings the bound down to a scale where it binds,
this number is in the exponent that sets it.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import _layer_node_counts
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import build_material_arrays

_CONFIG = "configs/scaps_mirror_v2.yaml"


def _mat(stack, N_grid=30):
    x = multilayer_grid([
        Layer(l.thickness, n) for l, n in
        zip(electrical_layers(stack), _layer_node_counts(stack, N_grid))
    ])
    return x, build_material_arrays(x, stack)


@pytest.fixture(scope="module")
def folded():
    """The parity configuration: DOS fold ON."""
    stack = load_scaps_yaml(_CONFIG)
    assert stack.dos_band_potentials, "this fixture needs the fold active"
    return _mat(stack)


def test_physical_bands_are_cached_and_differ_from_the_folded_ones(folded):
    """The whole point: with the fold on, the two must not be the same array."""
    _, mat = folded
    assert mat.chi_phys is not None and mat.Eg_phys is not None
    assert not np.allclose(mat.chi, mat.chi_phys), (
        "chi_phys equals chi with the fold active — the pre-fold copy is "
        "being taken at the wrong point"
    )


def test_te_barrier_uses_the_physical_valence_step(folded):
    """Pin the measured physical step at the HTL/PVK face.

    +0.180 eV is the configured band offset; +0.097 eV is what the folded
    arrays report there. A regression that silently reverts to the folded
    value shows up here as the wrong number, not as a subtle drift.
    """
    _, mat = folded
    f = mat.interface_faces[0]
    dEv_phys = float(
        (mat.chi_phys[f] + mat.Eg_phys[f])
        - (mat.chi_phys[f + 1] + mat.Eg_phys[f + 1])
    )
    dEv_folded = float(
        (mat.chi[f] + mat.Eg[f]) - (mat.chi[f + 1] + mat.Eg[f + 1])
    )
    assert dEv_phys == pytest.approx(0.180, abs=2e-3), (
        f"physical valence step is {dEv_phys:.4f} eV, expected the "
        "configured 0.180"
    )
    assert abs(dEv_phys - dEv_folded) > 0.05, (
        "folded and physical valence steps have converged; the measurement "
        "this test is built on needs re-deriving"
    )


def test_carrier_params_hands_the_physical_bands_to_the_cap(folded):
    """The RHS must receive them under a separate key.

    ``chi``/``Eg`` stay folded because the SG drift potentials need them;
    only the TE barrier switches. Both must be present and distinct.
    """
    _, mat = folded
    d = mat.carrier_params
    assert "chi_te" in d and "Eg_te" in d, (
        "the TE barrier keys are missing, so continuity falls back to the "
        "folded arrays"
    )
    assert np.array_equal(d["chi_te"], mat.chi_phys)
    assert np.array_equal(d["chi"], mat.chi)
    assert not np.allclose(d["chi"], d["chi_te"])


def test_fold_off_makes_the_two_identical(folded):
    """With no fold there is nothing to separate, by construction."""
    stack = dataclasses.replace(
        load_scaps_yaml(_CONFIG), dos_band_potentials=False,
    )
    _, mat = _mat(stack)
    np.testing.assert_array_equal(mat.chi, mat.chi_phys)
    np.testing.assert_array_equal(mat.Eg, mat.Eg_phys)


def test_face_selection_uses_physical_steps(folded):
    """The 50 meV threshold asks whether a real band step exists.

    Asserted structurally: every selected face must clear the threshold on
    the PHYSICAL edges. A face that only clears it after the fold would mean
    the selection is still reading transport potentials.
    """
    _, mat = folded
    assert mat.interface_faces, "no capped faces on this preset"
    for f in mat.interface_faces:
        dEc = abs(float(mat.chi_phys[f] - mat.chi_phys[f + 1]))
        dEv = abs(float(
            (mat.chi_phys[f] + mat.Eg_phys[f])
            - (mat.chi_phys[f + 1] + mat.Eg_phys[f + 1])
        ))
        assert max(dEc, dEv) > 0.05, (
            f"face {f} was selected but its physical steps are "
            f"dE_c={dEc:.4f}, dE_v={dEv:.4f} eV, both under the 50 meV "
            "threshold"
        )


def test_legacy_preset_without_dos_data_is_unaffected():
    """A config with no Nc300/Nv300 never folds, so nothing changes."""
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    _, mat = _mat(stack)
    np.testing.assert_array_equal(mat.chi, mat.chi_phys)
    np.testing.assert_array_equal(mat.Eg, mat.Eg_phys)
