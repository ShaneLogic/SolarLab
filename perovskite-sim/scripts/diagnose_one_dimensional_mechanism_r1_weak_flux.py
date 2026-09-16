"""Independent Decimal oracle for R1 weak ion flux; no replacement gate."""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import perovskite_sim
from perovskite_sim.constants import Q
from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import prepare_common_state, restore_common_state
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.ion_migration import ion_face_flux


def decimal_flux(phi, density, dx, diffusion, thermal_voltage, site_limit, *, precision):
    """Diffusion-only lattice-gas SG law at exact binary input values.

    Decimal arithmetic is independent of the NumPy Bernoulli implementation.
    This oracle excludes clipped populations and the legacy whole-flux steric
    model; it does not select or redefine any acceptance tolerance.
    """
    def number(x):
        return Decimal.from_float(float(x))

    with localcontext() as context:
        context.prec = precision
        one = Decimal(1)
        potentials = list(map(number, phi))
        populations = list(map(number, density))
        limits = list(map(number, site_limit))
        crowding = [-(one - p/limit).ln() for p, limit in zip(populations, limits)]
        vt = number(thermal_voltage)

        def bernoulli(x):
            return one if x == 0 else x/(x.exp()-one)

        result = []
        for i in range(len(dx)):
            x = (potentials[i+1]-potentials[i])/vt + crowding[i+1]-crowding[i]
            result.append(number(diffusion[i])/number(dx[i]) * (
                bernoulli(x)*populations[i]-bernoulli(-x)*populations[i+1]))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    root = Path(perovskite_sim.__file__).resolve().parents[1]
    stack = load_device_from_yaml(root / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
    binding = json.loads((root / "tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json").read_text())
    policy = r1_policy()
    prepared = prepare_common_state(stack, 16, binding, policy=policy)
    system, state = restore_common_state(prepared, stack, 16, binding, policy=policy)
    mat = system.material
    if not mat.ion_steric_diffusion_only or mat.has_dual_ions:
        raise ValueError("This oracle is restricted to the fixed R1 single-ion diffusion-only model")
    limits = np.broadcast_to(mat.P_lim_node, state.positive.shape)
    active = np.asarray(system.positive_nodes)
    if np.any(state.positive[active]/limits[active] >= .999):
        raise ValueError("Clipped populations are outside this oracle")
    # Inactive nodes have no mobile species and no flux. A finite positive
    # unused limit keeps their entropy expression mathematically defined.
    limits = np.where(limits > 0., limits, 1.)
    node = int(active[len(active)//2])
    cases = []
    for label, direction, increment in [
        ("prepared", 0, 0.), ("positive_signal", 1, 1e-8), ("negative_signal", -1, 1e-8),
        ("positive_weak", 1, 1e-12), ("negative_weak", -1, 1e-12),
        ("positive_ulp", 1, None), ("negative_ulp", -1, None),
    ]:
        phi = state.phi.copy()
        if increment is None:
            phi[node] = np.nextafter(phi[node], np.inf if direction > 0 else -np.inf)
        else:
            phi[node] += direction*increment
        arguments = (phi, state.positive, np.diff(system.grid), mat.D_ion_face,
                     mat.V_T_device, limits)
        coarse = decimal_flux(*arguments, precision=50)
        fine = decimal_flux(*arguments, precision=80)
        actual = ion_face_flux(phi, state.positive, np.diff(system.grid), mat.D_ion_face,
                               mat.V_T_device, mat.P_lim_face, steric_diffusion_only=True,
                               P_lim_node=mat.P_lim_node)
        error = np.array([float(Decimal.from_float(float(a))-b) for a,b in zip(actual, fine)])
        refinement = np.array([float(a-b) for a,b in zip(coarse, fine)])
        cases.append({"case": label, "potential_V": phi.tolist(),
                      "float_flux_m2_s": actual.tolist(),
                      "decimal_80_flux_m2_s": [str(x) for x in fine],
                      "maximum_absolute_flux_error_m2_s": float(np.max(np.abs(error))),
                      "maximum_absolute_current_error_A_m2": float(Q*np.max(np.abs(error))),
                      "oracle_50_to_80_maximum_flux_change_m2_s": float(np.max(np.abs(refinement))),
                      "affected_faces": [node-1, node],
                      "affected_float_flux_m2_s": actual[[node-1, node]].tolist(),
                      "affected_decimal_flux_m2_s": [str(fine[j]) for j in (node-1, node)],
                      "affected_absolute_current_error_A_m2": (Q*np.abs(error[[node-1, node]])).tolist(),
                      "changed_potential_V": float(phi[node]-state.phi[node])})
    report = {"schema": "R1WeakFluxOracleV1", "scope": "diagnostic_only_no_new_acceptance_criterion",
              "reference_precision_digits": [50, 80], "source_sha256": {
                  name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                  for name in ["perovskite_sim/physics/ion_migration.py", "perovskite_sim/discretization/fe_operators.py"]},
              "inputs": {"positive_m3": state.positive.tolist(), "grid_m": system.grid.tolist(),
                         "diffusion_m2_s": mat.D_ion_face.tolist(), "site_limit_m3": limits.tolist(),
                         "thermal_voltage_V": mat.V_T_device, "perturbed_node": node},
              "cases": cases,
              "interpretation_limit": "Exact represented inputs; a small absolute flux error alone does not certify trajectory or operator equivalence."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in cases:
        print(row["case"], row["maximum_absolute_current_error_A_m2"],
              row["oracle_50_to_80_maximum_flux_change_m2_s"], flush=True)


if __name__ == "__main__":
    main()
