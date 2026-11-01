"""Exact declared-profile closure for zero-diffusivity ion populations.

Targets come from compiled original P0/P0_neg, never an initial guess. A
positive-D connected species keeps its global inventory; an exactly zero-D
species retains a physical constraint in every domain cell. No original
transport, charge or time-storage equation is changed by these helpers.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from scipy.sparse import csc_matrix

from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, Ionic1DProblem

__all__ = ["ion_mobility_partition", "declared_immobile_profiles", "immobile_profile_binding",
           "ionic_closure_residual", "ionic_closure_jacobian"]


def ion_mobility_partition(device: CompiledIonicDevice1D) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return connected mobile and exactly immobile species; no small-D threshold."""
    mobile, immobile = [], []
    for species in device.ion_species_names:
        faces = getattr(device, species + "_ion_diffusivity_faces_m2_s")
        if np.all(faces == 0.0):
            immobile.append(species)
        elif np.all(faces > 0.0):
            mobile.append(species)
        else:
            raise CapabilityError("ionic closure supports uniformly immobile or connected positive-D species; disconnected mixed faces require another scope")
    return tuple(mobile), tuple(immobile)


def declared_immobile_profiles(device: CompiledIonicDevice1D) -> dict[str, np.ndarray]:
    """Original physical reference arrays, preserving their source/background."""
    return {species: getattr(device, species + "_ion_reference_m3")
            for species in ion_mobility_partition(device)[1]}


def immobile_profile_binding(device: CompiledIonicDevice1D) -> dict[str, object] | None:
    """Conditional provenance; positive-D records retain their existing schema."""
    profiles = declared_immobile_profiles(device)
    if not profiles:
        return None
    payload: dict[str, object] = {
        "rule": "declared_per_cell_zero_diffusivity_profile_v1",
        "compiled_identity": device.identity,
        "domain_layer": device.domain_layer_id,
        "electrical_node_indices": device.ion_node_indices.tolist(),
        "cell_measures_m": device.ion_geometry.cell_measures_m.tolist(),
        "species": {species: {"source_parameter": "P0" if species == "positive" else "P0_neg",
                              "target_density_m3": values.tolist(),
                              "face_diffusivity_m2_s": getattr(device, species + "_ion_diffusivity_faces_m2_s").tolist(),
                              "background_policy": "original_compiled_reference_unchanged"}
                    for species, values in profiles.items()},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**payload, "profile_sha256": hashlib.sha256(encoded.encode()).hexdigest()}


def _row_node(problem: Ionic1DProblem, inventory_row_node: int) -> None:
    if type(inventory_row_node) is not int or not 0 <= inventory_row_node < problem.device.ion_node_count:
        raise ValueError("inventory_row_node must name an actual ion-domain cell")


def ionic_closure_residual(problem: Ionic1DProblem, y: np.ndarray, inventory_row_node: int) -> np.ndarray:
    """Executable original rows plus mobile/global or immobile/local closure."""
    _row_node(problem, inventory_row_node)
    device = problem.device
    result = problem.residual(y)
    profiles = declared_immobile_profiles(device)
    populations = device.fields_by_species(y, voltage_V=problem.voltage_V)[3]
    for species, density, block in zip(device.ion_species_names, populations, device.ion_species_slices):
        if species in profiles:
            result[block] = device.ion_geometry.cell_measures_m * (density - profiles[species])
        else:
            reference = getattr(device, species + "_ion_reference_m3")
            target = math.fsum((device.ion_geometry.cell_measures_m * reference).tolist())
            result[block.start + inventory_row_node] = math.fsum((device.ion_geometry.cell_measures_m * density).tolist()) - target
    return result


def ionic_closure_jacobian(problem: Ionic1DProblem, y: np.ndarray, inventory_row_node: int) -> csc_matrix:
    """Full analytic physical gradients, including both shared coordinate blocks."""
    _row_node(problem, inventory_row_node)
    device = problem.device
    matrix = problem.jacobian(y).tolil()
    storage = problem.storage_jacobian(y)
    profiles = declared_immobile_profiles(device)
    for species, block in zip(device.ion_species_names, device.ion_species_slices):
        if species in profiles:
            for row in range(block.start, block.stop):
                gradient = -storage[row, :].toarray()[0]
                columns = np.flatnonzero(gradient)
                matrix.rows[row], matrix.data[row] = columns.tolist(), gradient[columns].tolist()
        else:
            row = block.start + inventory_row_node
            gradient = -np.asarray(storage[block, :].sum(axis=0)).ravel()
            columns = np.flatnonzero(gradient)
            matrix.rows[row], matrix.data[row] = columns.tolist(), gradient[columns].tolist()
    return matrix.tocsc()
