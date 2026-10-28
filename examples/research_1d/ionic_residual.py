"""Evaluate a real single-ion F/J/storage case, without a legacy or time solve.

Run from the installed repository, with BLAS threads pinned to one:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 python -m examples.research_1d.ionic_residual
"""

from __future__ import annotations

import json
import math

import numpy as np

from examples.research_1d.steady import load_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.ionic_1d import Ionic1DProblem


def prepare_single_ion_device(*, diffusion_only: bool = False) -> PreparedDevice:
    """Declare a homogeneous ionic slab using the existing physical material catalog."""
    template = load_device()
    data = template.to_input().editing_data()
    layer = next(layer for layer in data["layers"] if layer["role"].lower() == "absorber")
    data["id"] = "single_positive_ion_slab"
    data["name"] = "Single positive ion slab"
    data["layers"] = [layer]
    data["interfaces"] = []
    data["electrical_grid"] = [entry for entry in data.get("electrical_grid", []) if entry["layer"] == layer["id"]]
    data["contacts"] = [dict(id=side, side=side, layer=layer["id"], S_n=None, S_p=None) for side in ("left", "right")]
    data["settings"]["ion_steric_diffusion_only"] = diffusion_only
    layer["parameters"].update(P0=1e24, P_lim=1e27, D_ion=1e-16, P0_neg=0.0, D_ion_neg=0.0, E_a_ion=0.58)
    return resolve_device(DeviceInput.model_validate(data), template.defaults, template.resources, sources=template.sources)


def main() -> None:
    prepared = prepare_single_ion_device(diffusion_only=True)
    compiled = compile_ionic_device_1d(prepared, intervals_per_layer=12)
    problem = Ionic1DProblem(compiled)
    count = compiled.node_count
    cold = compiled.initial_state()
    y = cold.u.copy()
    coordinate = (compiled.nodes_m - compiled.nodes_m[0]) / compiled.ion_geometry.extent_m
    y[:count] += 0.002 * np.sin(np.pi * coordinate)
    y[3 * count:] = 0.1 * np.cos(np.pi * coordinate)
    state = compiled.state(y)
    residual = problem.residual(y)
    jacobian = problem.jacobian(y)
    flux = problem.ion_fluxes(y)
    ion_balance = residual[3 * count:]
    roundoff_bound = 4 * np.finfo(float).eps * math.fsum(abs(flux).tolist())
    _, n, p, density = compiled.fields(y)
    rho = compiled.charge_ledger(n, p, density).space_charge_C_m3()
    print(json.dumps({
        "scope": "actual coupled F/J/storage evaluation; ionic steady/transient qualification remains open",
        "compiled_identity": compiled.identity,
        "original_prepared_identity": compiled.original_prepared_sha256,
        "electronic_component_identity": compiled.electronic_component.identity,
        "node_count": count,
        "domain_layer": compiled.domain_layer_id,
        "domain_bounds_m": compiled.domain_bounds_m,
        "ion_geometry": compiled.ion_geometry.name,
        "steric_model": compiled.steric_model,
        "reference_inventory_m2": compiled.reference_inventory_m2,
        "trial_inventory_m2": state.domain_inventory_m2,
        "ion_face_flux_m2_s": flux.tolist(),
        "ion_balance_sum_m2_s": math.fsum(ion_balance.tolist()),
        "ion_balance_roundoff_bound_m2_s": roundoff_bound,
        "max_charge_density_C_m3": float(np.max(abs(rho))),
        "max_poisson_residual_C_m2": float(np.max(abs(residual[1:count - 1]))),
        "jacobian_shape": jacobian.shape,
        "jacobian_nonzeros": jacobian.nnz,
        "ion_storage_diagonal_m2": problem.storage_jacobian(y).diagonal()[3 * count:].tolist(),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
