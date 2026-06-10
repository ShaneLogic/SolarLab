"""Interface-defect substrate-offset alignment (2026-06 regression fix).

``stack.interface_defects`` is FULL-layer-aligned (parallel to
``stack.interfaces``; the loader resolves targets against the full layer list),
but ``build_material_arrays`` historically indexed it with the ELECTRICAL
interface index — on substrate-prefixed stacks (scaps_mirror_v2's glass layer,
E10.1) the HTL/PVK interface silently fell back to the legacy path and the
PVK/ETL interface received the HTL/PVK defect object.

Fix: ``electrical_interface_defects()`` in models/device.py mirrors
``electrical_interfaces()`` (drops the substrate-prefix slots), and the build
consumes it.
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.models.device import (
    electrical_layers,
    electrical_interface_defects,
)
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.sweeps.device_parameter_sweep import srh_n1_p1_from_trap_depth

_V2 = "configs/scaps_mirror_v2.yaml"   # glass substrate prefix (4 layers)
_V1 = "configs/scaps_mirror.yaml"      # no substrate (3 layers)


def _build(stack):
    elec = electrical_layers(stack)
    x = multilayer_grid([Layer(thickness=L.thickness, N=30 // len(elec)) for L in elec])
    return x, build_material_arrays(x, stack)


def test_helper_drops_substrate_prefix_on_v2():
    stack = load_scaps_yaml(_V2)
    defects = electrical_interface_defects(stack)
    assert len(defects) == 2, "two electrical interfaces (HTL/PVK, PVK/ETL)"
    assert defects[0] is not None, "HTL/PVK defect must survive the offset"
    assert defects[1] is not None, "PVK/ETL defect must survive the offset"


def test_helper_identity_without_substrate():
    stack = load_scaps_yaml(_V1)
    assert electrical_interface_defects(stack) == tuple(stack.interface_defects)


def test_build_uses_etaware_n1_p1_at_htl_pvk():
    """With the offset fixed, the HTL/PVK interface gets E_t-derived n1/p1
    (E_t = 0.6 eV below CB of the absorber), not the legacy bulk values."""
    stack = load_scaps_yaml(_V2)
    x, mat = _build(stack)
    absorber = next(L.params for L in electrical_layers(stack) if L.role == "absorber")
    n1_exp, p1_exp = srh_n1_p1_from_trap_depth(
        absorber.ni, absorber.Eg, 0.6, reference="below_cb",
    )
    assert mat.interface_n1[0] == pytest.approx(n1_exp, rel=1e-9)
    assert mat.interface_p1[0] == pytest.approx(p1_exp, rel=1e-9)
    # PVK/ETL (same declared E_t) must match too — and must NOT be the
    # legacy per-node fallback.
    assert mat.interface_n1[1] == pytest.approx(n1_exp, rel=1e-9)


def test_build_cross_carrier_eval_nodes_active_at_htl_pvk():
    """Legacy fallback samples eval nodes at idx; the E_t-aware path samples
    idx+1 / idx-1 (cross-carrier)."""
    stack = load_scaps_yaml(_V2)
    x, mat = _build(stack)
    idx0 = mat.interface_nodes[0]
    assert mat.interface_eval_node_n[0] == idx0 + 1
    assert mat.interface_eval_node_p[0] == idx0 - 1
