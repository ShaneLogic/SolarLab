"""Schema and capability boundaries for P4.3 bulk-trap charge."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.reproducibility import semantic_sha256
from perovskite_sim.solver.mol import (
    BULK_TRAP_CHARGE_RESEARCH_EQUILIBRIUM,
    BulkTrapChargeCapabilityError,
    StateVec,
    assemble_rhs,
    build_material_arrays,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/csi_gaussian_bulk_trap_pn_research.yaml"


def _stack():
    return load_device_from_yaml(str(CONFIG))


def _grid(intervals: int = 8):
    stack = _stack()
    return multilayer_grid(
        tuple(Layer(layer.thickness, intervals) for layer in stack.layers),
        alpha=3.0,
    )


def test_standard_yaml_roundtrips_explicit_si_distribution():
    stack = _stack()
    left = stack.layers[0].params.bulk_trap_distribution
    right = stack.layers[1].params.bulk_trap_distribution

    assert left is not None
    assert left == right
    assert left.distribution == "gaussian"
    assert left.total_density_m3 == pytest.approx(1.0e22)
    assert left.energy_sigma_eV == pytest.approx(0.08)
    assert left.charge_transition == "acceptor"


def test_active_distribution_is_semantically_content_addressed():
    active = _stack()
    inert = replace(
        active,
        layers=tuple(
            replace(
                layer,
                params=replace(layer.params, bulk_trap_distribution=None),
            )
            for layer in active.layers
        ),
    )

    assert semantic_sha256(active) != semantic_sha256(inert)


def test_bulk_trap_schema_requires_mb_fully_ionized_base_gap():
    params = _stack().layers[0].params
    with pytest.raises(ValueError, match="finite positive Eg"):
        replace(params, Eg=0.0)
    with pytest.raises(ValueError, match="maxwell_boltzmann"):
        replace(params, carrier_statistics="fermi_dirac")
    with pytest.raises(ValueError, match="fully_ionized"):
        replace(
            params,
            dopant_ionization_model="discrete_level",
            acceptor_binding_energy_eV=0.045,
        )
    with pytest.raises(ValueError, match="exclude band-gap narrowing"):
        replace(params, band_gap_narrowing_model="slotboom")


def test_device_requires_semiconductor_work_function_contact_contract():
    with pytest.raises(ValueError, match="bulk trap charge.*work_function"):
        replace(_stack(), built_in_potential_mode="legacy_manual")


def test_default_material_and_mol_paths_fail_closed():
    stack = _stack()
    grid = _grid()
    with pytest.raises(
        BulkTrapChargeCapabilityError,
        match="not enabled on the default solver path",
    ):
        build_material_arrays(grid, stack)

    material = build_material_arrays(
        grid,
        stack,
        bulk_trap_charge_closure=BULK_TRAP_CHARGE_RESEARCH_EQUILIBRIUM,
    )
    state = StateVec.pack(
        np.full(grid.size, material.n_L),
        np.full(grid.size, material.p_L),
        material.P_ion0.copy(),
    )
    with pytest.raises(
        BulkTrapChargeCapabilityError,
        match="production MoL does not include",
    ):
        assemble_rhs(0.0, state, grid, stack, material, illuminated=False)


def test_research_material_uses_trap_aware_contact_reservoirs():
    stack = _stack()
    material = build_material_arrays(
        _grid(),
        stack,
        bulk_trap_charge_closure=BULK_TRAP_CHARGE_RESEARCH_EQUILIBRIUM,
    )

    assert material.bulk_trap_distribution is not None
    assert material.bulk_trap_charge_closure == "research_equilibrium"
    assert material.n_L * material.p_L == pytest.approx(
        material.ni_sq[0],
        rel=2.0e-14,
    )
    assert material.n_R * material.p_R == pytest.approx(
        material.ni_sq[-1],
        rel=2.0e-14,
    )


def test_research_material_rejects_inconsistent_mass_action_inputs():
    stack = _stack()
    inconsistent = replace(
        stack,
        layers=tuple(
            replace(layer, params=replace(layer.params, ni=2.0 * layer.params.ni))
            for layer in stack.layers
        ),
    )

    with pytest.raises(
        BulkTrapChargeCapabilityError,
        match="same Maxwell-Boltzmann mass-action law",
    ):
        build_material_arrays(
            _grid(),
            inconsistent,
            bulk_trap_charge_closure=BULK_TRAP_CHARGE_RESEARCH_EQUILIBRIUM,
        )
