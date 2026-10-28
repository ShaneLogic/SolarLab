"""Actual two blocking absorber domains, without a nonlinear solve."""

from __future__ import annotations

from copy import deepcopy
import json
import math

import numpy as np

from examples.research_1d.ionic_multilayer import prepare_confined_ion_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_multiple_ionic_1d import compile_multiple_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.multiple_ionic_1d import MultipleIonic1DProblem

DOMAINS = (("absorber_left", ("blocking", "blocking")), ("absorber_right", ("blocking", "blocking")))
INTERVALS = (2, 3, 2, 3, 2)


def prepare_multiple_ionic_device(*, kind: str = "positive", right_immobile: bool = False) -> PreparedDevice:
    """HTL/absorber/gap/absorber/ETL; every modification is an explicit input."""
    if kind not in {"positive", "independent", "shared"}:
        raise ValueError("kind must be positive, independent or shared")
    prepared = prepare_confined_ion_device()
    data = prepared.to_input().editing_data()
    htl, absorber, etl = (deepcopy(layer) for layer in data["layers"])
    first, second, gap = deepcopy(absorber), deepcopy(absorber), deepcopy(htl)
    first.update(id="absorber_left", thickness=120e-9)
    second.update(id="absorber_right", thickness=180e-9)
    gap.update(id="inactive_gap", thickness=40e-9)
    first["parameters"].update(P0=1e24, P_lim=4e24, D_ion=1e-16)
    second["parameters"].update(P0=7e23, P_lim=3e24, D_ion=0.0 if right_immobile else 4e-17)
    if kind != "positive":
        first["parameters"].update(P0_neg=5e23, P_lim_neg=first["parameters"]["P_lim"] if kind == "shared" else 7e24, D_ion_neg=3e-17)
        second["parameters"].update(P0_neg=4e23, P_lim_neg=second["parameters"]["P_lim"] if kind == "shared" else 6e24,
                                    D_ion_neg=0.0 if right_immobile else 2e-17)
    data["id"] = "two_separated_blocking_ion_domains_" + kind
    data["settings"]["ion_steric_shared_site"] = kind == "shared"
    data["layers"] = [htl, first, gap, second, etl]
    grids = {entry["layer"]: entry for entry in data["electrical_grid"]}
    data["electrical_grid"] = [
        {**grids[original], "layer": layer["id"]}
        for layer, original in ((htl, htl["id"]), (first, absorber["id"]), (gap, htl["id"]),
                                (second, absorber["id"]), (etl, etl["id"]))
    ]
    return resolve_device(DeviceInput.model_validate(data), prepared.defaults, prepared.resources, sources=prepared.sources)


def main() -> None:
    prepared = prepare_multiple_ionic_device()
    device = compile_multiple_ionic_device_1d(prepared, intervals_per_layer=INTERVALS, ion_domains=DOMAINS)
    problem = MultipleIonic1DProblem(device, voltage_V=0.002)
    y = device.initial_state(voltage_V=0.002).u.copy()
    for domain in device.domains:
        for species, block in device.ion_species_slices[domain.domain_layer_id].items():
            y[block] = np.linspace(-0.12, 0.15 if species == "positive" else 0.08, domain.ion_node_count)
    state = device.state(y, voltage_V=0.002)
    F, J, K = problem.residual(y), problem.jacobian(y), problem.storage_jacobian(y)
    print(json.dumps({
        "compiled_identity": device.identity, "original_prepared_identity": device.original_prepared_sha256,
        "electrical_nodes": device.node_count, "state_size": device.state_size,
        "domains": [{"layer": domain.domain_layer_id, "electrical_node_indices": domain.ion_node_indices.tolist(),
                     "physical_weights_m": domain.ion_geometry.cell_measures_m.tolist(),
                     "site_contract": domain.ion_site_contract} for domain in device.domains],
        "reference_inventories_m2": device.reference_inventories_m2, "actual_inventories_m2": state.domain_inventories_m2,
        "original_balance_sums_m2_s": {name: {species: math.fsum(F[block].tolist()) for species, block in blocks.items()}
                                     for name, blocks in device.ion_species_slices.items()},
        "signed_ionic_current_A_m2": problem.ion_current_density_A_m2(y).tolist(),
        "jacobian_shape": J.shape, "jacobian_nonzeros": J.nnz, "storage_nonzeros": K.nnz,
        "physical_storage_m2": problem.physical_storage(y).tolist(),
        "all_finite": bool(np.all(np.isfinite(F)) and np.all(np.isfinite(J.data)) and np.all(np.isfinite(K.data))),
        "nonlinear_solves": 0, "qualification": False,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
