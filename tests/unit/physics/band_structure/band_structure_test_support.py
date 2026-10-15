"""Test support: legacy presets mapped onto the P04-11 parameter binding.

Only tests import this module. The mapping reads legacy ``MaterialParams``
attribute names and resolves the legacy tier switches exactly as
``solver/mol.build_material_arrays`` does; the product code never sees them.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np

from solarlab.physics.band_structure import (
    BandGapNarrowingSpec,
    DopingProfileSpec,
    GradingSpec,
    LayerParameterInput,
    TemperatureLawSpec,
    compile_layer_parameters,
)

LEGACY_ROOT = Path(__file__).resolve().parents[4] / "perovskite-sim"
CONFIGS = LEGACY_ROOT / "tests" / "fixtures" / "configs"


def load_stack(name: str, temperature_K: float | None = None):
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.scaps_compat.loader import load_scaps_yaml

    if name == "scaps_mirror_v2.yaml":
        stack = load_scaps_yaml(str(LEGACY_ROOT / "configs" / name))
    else:
        stack = load_device_from_yaml(str(CONFIGS / name))
    if temperature_K is not None:
        stack = dataclasses.replace(stack, T=temperature_K)
    return stack


def electrical_grid(stack, n_grid: int = 60) -> np.ndarray:
    from perovskite_sim.discretization.grid import Layer, multilayer_grid
    from perovskite_sim.experiments.jv_sweep import _layer_node_counts
    from perovskite_sim.models.device import electrical_layers

    elec = electrical_layers(stack)
    return multilayer_grid([Layer(layer.thickness, n) for layer, n in zip(elec, _layer_node_counts(stack, n_grid))])


def legacy_material_arrays(x, stack):
    from perovskite_sim.solver.mol import (
        DEGENERATE_TRANSPORT_RESEARCH_RECOMBINATION_OFF,
        build_material_arrays,
    )
    from perovskite_sim.models.device import electrical_layers

    research = any(
        layer.params.band_gap_narrowing_model != "off"
        or layer.params.carrier_statistics == "fermi_dirac"
        or layer.params.dopant_ionization_model == "discrete_level"
        for layer in electrical_layers(stack)
    )
    if research:
        return build_material_arrays(x, stack, carrier_statistics_transport=DEGENERATE_TRANSPORT_RESEARCH_RECOMBINATION_OFF)
    return build_material_arrays(x, stack)


def layer_input(layer) -> LayerParameterInput:
    p = layer.params
    return LayerParameterInput(
        name=layer.name,
        role=layer.role,
        thickness_m=layer.thickness,
        electron_affinity_eV=p.chi,
        band_gap_eV=p.Eg,
        intrinsic_density_m3=p.ni,
        srh_n1_m3=p.n1,
        srh_p1_m3=p.p1,
        electron_mobility_m2_V_s=p.mu_n,
        hole_mobility_m2_V_s=p.mu_p,
        radiative_coefficient_m3_s=p.B_rad,
        doping=DopingProfileSpec(
            p.N_A, p.N_D, p.N_A_bulk, p.N_D_bulk, p.doping_profile_shape, p.doping_decay_length, p.doping_edge
        ),
        thermal_velocity_m_s=p.v_th,
        ion_diffusivity_m2_s=p.D_ion,
        negative_ion_diffusivity_m2_s=p.D_ion_neg,
        conduction_dos_300_m3=p.Nc300,
        valence_dos_300_m3=p.Nv300,
        temperature_laws=TemperatureLawSpec(p.mu_T_gamma, p.E_a_ion, p.B_rad_T_gamma, p.varshni_alpha, p.varshni_beta),
        grading=GradingSpec(
            p.Eg_back, p.chi_back, p.grading_profile, p.grading_direction, p.grading_bowing, p.grading_char_length
        ),
        narrowing=BandGapNarrowingSpec(
            p.band_gap_narrowing_model,
            p.bgn_reference_energy_eV,
            p.bgn_reference_density_m3,
            p.bgn_log_shape,
            p.bgn_conduction_band_fraction,
        ),
    )


def tier_switches(stack) -> dict[str, bool]:
    """The legacy material build's resolved switches (mode gates and env overrides)."""

    from perovskite_sim.models.mode import resolve_mode

    mode = resolve_mode(getattr(stack, "mode", "full"))
    return {
        "temperature_scaling": bool(mode.use_temperature_scaling),
        "band_grading": bool(
            getattr(stack, "band_grading", False) or os.environ.get("SOLARLAB_BAND_GRADING") == "1"
        )
        and mode.name != "legacy",
        "dos_band_potentials": bool(
            getattr(stack, "dos_band_potentials", True) or os.environ.get("SOLARLAB_DOS_BAND") == "1"
        )
        and mode.name != "legacy",
        "dual_ions": bool(mode.use_dual_ions),
    }


def binding_for(stack, x):
    from perovskite_sim.models.device import electrical_layers

    return compile_layer_parameters(
        x, [layer_input(layer) for layer in electrical_layers(stack)], temperature_K=stack.T, **tier_switches(stack)
    )


def same(a, b) -> bool:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a.shape == b.shape and a.tobytes() == b.tobytes()
