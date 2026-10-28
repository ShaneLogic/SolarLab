"""Evaluate the explicitly confined absorber ion domain in a real three-layer device.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 python -m examples.research_1d.ionic_multilayer
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


def prepare_confined_ion_device(*, diffusion_only: bool = True) -> PreparedDevice:
    """Original HTL/absorber/ETL with one explicitly populated mobile absorber."""
    template = load_device()
    data = template.to_input().editing_data()
    data["id"] = "confined_absorber_positive_ion"
    data["settings"]["ion_steric_diffusion_only"] = diffusion_only
    absorber = next(layer for layer in data["layers"] if layer["role"].lower() == "absorber")
    absorber["parameters"].update(P0=1e24, P_lim=1e27, D_ion=1e-16, E_a_ion=0.58)
    return resolve_device(DeviceInput.model_validate(data), template.defaults, template.resources, sources=template.sources)


def main() -> None:
    device = compile_ionic_device_1d(
        prepare_confined_ion_device(), intervals_per_layer=(12, 12, 12),
        ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"),
    )
    problem = Ionic1DProblem(device, voltage_V=0.002)
    state = device.initial_state(voltage_V=0.002)
    residual, J, K = problem.residual(state.u), problem.jacobian(state.u), problem.storage_jacobian(state.u)
    flux = problem.ion_fluxes(state.u)
    print(json.dumps({
        "scope": "actual confined-domain F/J/storage; no steady/transient qualification",
        "compiled_identity": device.identity, "original_prepared_identity": device.original_prepared_sha256,
        "electronic_component_identity": device.electronic_component.identity,
        "electrical_nodes": device.node_count, "ion_nodes": device.ion_node_count,
        "state_size": device.state_size, "domain_bounds_m": device.domain_bounds_m,
        "ion_node_indices": device.ion_node_indices.tolist(),
        "charge_weight_ratio": device.ion_charge_weight_ratio.tolist(),
        "reference_inventory_m2": device.reference_inventory_m2,
        "ion_balance_sum_m2_s": math.fsum(residual[3 * device.node_count:].tolist()),
        "ion_face_flux_m2_s": flux.tolist(),
        "outside_ion_flux_m2_s": problem.ion_fluxes_on_electronic_faces(state.u).tolist(),
        "jacobian_shape": J.shape, "jacobian_nonzeros": J.nnz,
        "ion_storage_diagonal_m2": K.diagonal()[3 * device.node_count:].tolist(),
        "all_finite": bool(np.all(np.isfinite(residual)) and np.all(np.isfinite(J.data))),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
