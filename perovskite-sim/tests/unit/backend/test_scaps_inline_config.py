"""SCAPS-schema presets through the backend/UI path.

SCAPS presets (scaps_mirror*, robin variants) use a different layer schema
(``mu_n_cm2`` / ``N_C_cm3`` / ``E_g_eV`` / ``thickness_nm``, cm/eV units, ``ni``
computed from the DOS) that only ``scaps_compat.load_scaps_yaml`` can parse. The
backend ``get_config`` returned them raw and the frontend, which assumes the
standard schema, found none of its keys and submitted an all-zero device → a
``ZeroDivisionError`` in ``build_material_arrays`` (``_eq_n_p_layer``: net=0,
ni=0). ``get_config`` now converts SCAPS configs to the standard schema so they
load / edit / run through the inline-device path like any other preset.
"""
from __future__ import annotations

import yaml

import backend.main as bm

SCAPS_CFG = "configs/scaps_mirror_v2.yaml"
STD_CFG = "configs/nip_MAPbI3_tmm.yaml"


def test_detects_scaps_schema():
    assert bm._is_scaps_schema(yaml.safe_load(open(SCAPS_CFG))) is True


def test_standard_schema_is_not_scaps():
    assert bm._is_scaps_schema(yaml.safe_load(open(STD_CFG))) is False


def test_scaps_config_dict_has_real_standard_values():
    """Conversion yields editable standard fields with physical (non-zero)
    values — not the all-zero device that crashed the solver."""
    std = bm._config_dict_from_path(SCAPS_CFG)
    absb = next(l for l in std["layers"] if l["role"] == "absorber")
    assert absb["mu_n"] > 0.0
    assert absb["ni"] > 0.0
    assert (absb["N_A"] > 0.0) or (absb["N_D"] > 0.0)
    assert absb["optical_material"]  # TMM optics survive the round-trip


def test_standard_config_dict_passes_through_unchanged():
    """A standard preset is returned byte-identical (no SCAPS conversion)."""
    raw = yaml.safe_load(open(STD_CFG))
    assert bm._config_dict_from_path(STD_CFG) == raw


def test_scaps_inline_rebuild_runs_without_zerodivision():
    """The converted dict, submitted back inline (what the frontend does),
    rebuilds a real stack and reaches build_material_arrays without the
    ZeroDivisionError that the all-zero device triggered."""
    from perovskite_sim.experiments.steady_state import _grid_for
    from perovskite_sim.solver.mol import build_material_arrays

    std = bm._config_dict_from_path(SCAPS_CFG)
    stack = bm.stack_from_dict(std)
    build_material_arrays(_grid_for(stack, 40), stack)  # used to raise
