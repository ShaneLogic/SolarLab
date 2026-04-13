"""End-to-end regression for stacks with a role:substrate prefix.

Regression guard for 6c741bb: the three experiment entry points built their
grid from ``stack.layers`` (full list) instead of ``electrical_layers(stack)``,
and ``_apply_interface_recombination`` zipped ``stack.interfaces`` (full)
against ``mat.interface_nodes`` (filtered). Both bugs were silent until a
role:substrate layer was added to a stack. This test wraps an existing preset
with a glass substrate and runs a short J-V sweep end-to-end; before the fix
the masks landed on substrate nodes and the sweep returned unphysical metrics
or crashed outright.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import (
    LayerSpec, electrical_interfaces, electrical_layers,
)
from perovskite_sim.models.parameters import MaterialParams


def _glass_substrate_layer(thickness_m: float = 1.0e-3) -> LayerSpec:
    """A physically inert glass substrate.

    eps_r ≈ 2.25 (n ≈ 1.5), zero mobility / recombination / absorption so the
    layer can only matter to the electrical solver if something mistakenly
    routes grid nodes or masks into it.
    """
    glass_params = MaterialParams(
        eps_r=2.25,
        mu_n=0.0, mu_p=0.0,
        D_ion=0.0, P_lim=1e30, P0=0.0,
        ni=1.0,  # avoid log(0) in V_bi computation; harmless (chi=Eg=0)
        tau_n=1e-9, tau_p=1e-9,
        n1=1.0, p1=1.0, B_rad=0.0,
        C_n=0.0, C_p=0.0,
        alpha=0.0, N_A=0.0, N_D=0.0,
        optical_material=None,
        n_optical=1.5,
        incoherent=True,
    )
    return LayerSpec(
        name="glass",
        thickness=thickness_m,
        params=glass_params,
        role="substrate",
    )


def _prepend_substrate(stack):
    """Return a new DeviceStack with a glass substrate prepended.

    A dummy ``(0, 0)`` interface entry is prepended to ``stack.interfaces``
    so the ``len(interfaces) == len(layers) - 1`` invariant still holds.
    The ``electrical_interfaces`` helper is expected to drop this entry.
    """
    substrate = _glass_substrate_layer()
    new_layers = (substrate,) + tuple(stack.layers)
    new_interfaces = ((0.0, 0.0),) + tuple(stack.interfaces)
    return replace(stack, layers=new_layers, interfaces=new_interfaces)


def test_electrical_layers_filters_substrate_prefix():
    """``electrical_layers`` drops the substrate and keeps everything else."""
    real = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    wrapped = _prepend_substrate(real)
    elec = electrical_layers(wrapped)
    assert len(elec) == len(real.layers)
    assert all(l.role != "substrate" for l in elec)
    # And the surviving order is preserved.
    for a, b in zip(elec, real.layers):
        assert a.name == b.name


def test_electrical_interfaces_drops_substrate_prefix_entry():
    """``electrical_interfaces`` must line up with ``electrical_layers``."""
    real = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    wrapped = _prepend_substrate(real)
    ifaces = electrical_interfaces(wrapped)
    # One fewer than the number of electrical layers.
    assert len(ifaces) == len(electrical_layers(wrapped)) - 1
    # The dropped leading (0, 0) must not appear at index 0 unless the real
    # stack happened to start with a (0, 0) too.
    assert ifaces == tuple(real.interfaces)


def test_electrical_layers_rejects_mid_stack_substrate():
    """Mid-stack substrate must raise — the prefix invariant is enforced."""
    real = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    substrate = _glass_substrate_layer()
    # Insert after the first electrical layer (illegal placement).
    bad_layers = (real.layers[0], substrate) + tuple(real.layers[1:])
    bad_ifaces = (real.interfaces[0], (0.0, 0.0)) + tuple(real.interfaces[1:])
    bad_stack = replace(real, layers=bad_layers, interfaces=bad_ifaces)
    with pytest.raises(ValueError, match="contiguous prefix"):
        electrical_layers(bad_stack)


@pytest.mark.slow
def test_substrate_stack_run_jv_sweep_smoke():
    """End-to-end J-V sweep must run and produce physically sane metrics.

    This is the regression guard for 6c741bb. Before the fix the experiments
    built ``x`` from ``stack.layers`` so the grid covered the substrate too,
    and the interface-recombination loop zipped the wrong pairs. Either bug
    corrupts ``J_sc`` or crashes before returning.
    """
    real = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    wrapped = _prepend_substrate(real)
    result = run_jv_sweep(wrapped, n_points=7, N_grid=60)
    assert result.metrics_fwd.J_sc > 0.0
    assert result.metrics_fwd.V_oc > 0.0
