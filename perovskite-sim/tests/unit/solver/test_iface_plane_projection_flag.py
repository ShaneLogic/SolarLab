"""Interface-plane projection as a first-class config flag (2026-06).

Promotes the ``SOLARLAB_IFACE_PROJ`` env-gate (E8 Boltzmann projection of the
cross-carrier interface-SRH eval densities onto the band-bending-suppressed
interface plane — the SCAPS Pauwels-Vanhoutte sampling) to a discoverable,
default-OFF ``DeviceStack.interface_plane_projection`` flag, carried onto
``MaterialArrays.iface_plane_projection`` and honoured (OR'd with the env var)
by ``_apply_interface_recombination``.

Default OFF → every existing config is bit-identical to the pre-flag path.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import yaml

from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import (
    build_material_arrays,
    _apply_interface_recombination,
)

_V2 = "configs/scaps_mirror_v2.yaml"


def _build(projection: bool | None):
    """Build (stack, x, mat) from scaps_mirror_v2 with the projection flag set.

    ``projection=None`` leaves the stack default (False) — used for the env path.
    """
    stack = load_scaps_yaml(_V2)
    if projection is not None:
        stack = dataclasses.replace(stack, interface_plane_projection=projection)
    elec = electrical_layers(stack)
    layers_grid = [Layer(thickness=L.thickness, N=30 // len(elec)) for L in elec]
    x = multilayer_grid(layers_grid)
    mat = build_material_arrays(x, stack)
    return stack, x, mat


def _interface_dn(stack, x, mat):
    """Synthetic illuminated densities + a φ gradient → interface dn vector."""
    N = len(x)
    n = np.full(N, 1.0e22)
    p = np.full(N, 1.0e22)
    phi = np.linspace(0.0, 1.0, N)  # non-zero gradient so the projection factor != 1
    dn = np.zeros(N)
    dp = np.zeros(N)
    _apply_interface_recombination(dn, dp, n, p, stack, mat, phi)
    return dn


# ----------------------------- defaults -----------------------------

def test_devicestack_default_projection_off():
    stack = load_scaps_yaml(_V2)
    assert stack.interface_plane_projection is False


def test_material_arrays_default_projection_off():
    _, _, mat = _build(False)
    assert mat.iface_plane_projection is False


# --------------------------- build plumbing -------------------------

def test_material_arrays_projection_from_stack_flag():
    _, _, mat = _build(True)
    assert mat.iface_plane_projection is True


def test_material_arrays_projection_from_env(monkeypatch):
    monkeypatch.setenv("SOLARLAB_IFACE_PROJ", "1")
    _, _, mat = _build(False)  # stack flag off, env on
    assert mat.iface_plane_projection is True


# ----------------------------- behaviour ----------------------------

def test_projection_on_changes_interface_recombination():
    """With the flag on, the interface SRH rate is computed from the
    interface-plane (projected) densities, so dn differs from the off path."""
    s_off, x, m_off = _build(False)
    s_on, _, m_on = _build(True)
    dn_off = _interface_dn(s_off, x, m_off)
    dn_on = _interface_dn(s_on, x, m_on)
    assert not np.allclose(dn_off, dn_on), "projection should change the interface rate"


def test_stack_flag_matches_env(monkeypatch):
    """Enabling via the stack flag is identical to enabling via the env var."""
    # env path (stack flag default off)
    monkeypatch.setenv("SOLARLAB_IFACE_PROJ", "1")
    s_env, x_env, m_env = _build(None)
    monkeypatch.delenv("SOLARLAB_IFACE_PROJ", raising=False)
    # flag path (env off)
    s_flag, x_flag, m_flag = _build(True)
    assert m_env.iface_plane_projection is True
    assert m_flag.iface_plane_projection is True
    np.testing.assert_allclose(
        _interface_dn(s_env, x_env, m_env),
        _interface_dn(s_flag, x_flag, m_flag),
    )


# --------------------------- config loaders -------------------------

def test_config_loader_reads_projection(tmp_path):
    cfg = yaml.safe_load(Path("configs/nip_MAPbI3.yaml").read_text())
    cfg.setdefault("device", {})["interface_plane_projection"] = True
    dst = tmp_path / "proj.yaml"
    dst.write_text(yaml.safe_dump(cfg))
    stack = load_device_from_yaml(str(dst))
    assert stack.interface_plane_projection is True


def test_config_loader_default_projection_off():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    assert stack.interface_plane_projection is False


def test_scaps_loader_reads_projection(tmp_path):
    cfg = yaml.safe_load(Path(_V2).read_text())
    cfg["device"]["interface_plane_projection"] = True
    dst = tmp_path / "scaps_proj.yaml"
    dst.write_text(yaml.safe_dump(cfg))
    stack = load_scaps_yaml(str(dst))
    assert stack.interface_plane_projection is True
