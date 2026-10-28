"""Bind validated neutral single-level inventories to the existing DEF-1 law."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json

import numpy as np

from solarlab.device.resolved import PreparedDefect, PreparedLayer
from solarlab.equations.dc_1d import CapabilityError
from solarlab.physics.recombination.explicit_neutral import (
    NeutralDefectSpecies, NeutralExplicitDefectModel, compile_neutral_defect_species,
)


def bind_neutral_defects(
    layers: Sequence[PreparedLayer], masks: Sequence[np.ndarray], *,
    ni_sq_m6: np.ndarray, band_gap_eV: np.ndarray, thermal_voltage_V: float,
    conduction_dos_m3: np.ndarray, valence_dos_m3: np.ndarray,
) -> NeutralExplicitDefectModel | None:
    """Preserve declared species order, resolved documents and supplied ownership.

    Masks are the compiler's RIGHT_LAYER_WINS masks. Capture lifetimes and
    reference densities are evaluated only by the unchanged historical law.
    Its intrinsic-density/midgap convention preserves a declared energy above
    the valence band only when the physical conduction and valence DOS agree.
    """
    if len(layers) != len(masks):
        raise ValueError("one node ownership mask is required per electrical layer")
    species: list[NeutralDefectSpecies] = []
    documents: list[tuple[str, str]] = []
    for layer, mask in zip(layers, masks):
        if not layer.defects:
            continue
        where = f"layers.{layer.id}.bulk_defects"
        if layer.defect_model != "explicit_quasi_steady":
            raise CapabilityError(f"{where}: an explicit inventory requires explicit_quasi_steady")
        if any(not isinstance(item, PreparedDefect) for item in layer.defects):
            raise CapabilityError(f"{where}: multivalent defects are not bound")
        if np.any(conduction_dos_m3[mask] != valence_dos_m3[mask]):
            raise CapabilityError(
                f"{where}: neutral DEF-1 requires equal physical conduction and valence DOS "
                "on owned nodes to preserve the declared above-valence-band energy"
            )
        resolved = [item.to_mapping() for item in layer.defects]
        document = {"schema_version": layer.defect_schema_version,
                    "defect_model": layer.defect_model, "bulk_defects": resolved}
        digest = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
        documents.append((layer.id, digest))
        for item in resolved:
            identifier = f"{layer.id}/{item['id']}"
            distribution, kinetics = item["distribution"], item["kinetics"]
            if item["charge_transition"] != "neutral" or item["neutral_reference"] != "all_occupancies":
                raise CapabilityError(f"{where}.{item['id']}: charged defects are not bound")
            if distribution["kind"] != "single_level" or item.get("spatial_profile") is not None:
                raise CapabilityError(f"{where}.{item['id']}: distributed defects are not bound")
            if item["degeneracy"] != 1.0:
                raise CapabilityError(f"{where}.{item['id']}: neutral DEF-1 requires unit degeneracy")
            center = distribution["center_eV_above_vb"]
            if center < 0.0 or np.any(center > band_gap_eV[mask]):
                raise CapabilityError(f"{where}.{item['id']}: resolved level lies outside the physical gap")
            try:
                species.append(compile_neutral_defect_species(
                    identifier=identifier, document_sha256=digest, active_nodes=mask,
                    ni_sq_m6=ni_sq_m6, band_gap_eV=band_gap_eV, thermal_voltage_V=thermal_voltage_V,
                    total_density_m3=distribution["total_density_m3"], center_eV_above_vb=center,
                    sigma_n_m2=kinetics["sigma_n_m2"], sigma_p_m2=kinetics["sigma_p_m2"],
                    thermal_velocity_n_m_s=kinetics["thermal_velocity_n_m_s"],
                    thermal_velocity_p_m_s=kinetics["thermal_velocity_p_m_s"],
                    charge_transition=item["charge_transition"], distribution=distribution["kind"],
                    degeneracy=item["degeneracy"],
                ))
            except ValueError as error:
                raise CapabilityError(f"{where}.{item['id']}: {error}") from error
    if not species:
        return None
    owned = np.logical_or.reduce([item.active_nodes for item in species])
    return NeutralExplicitDefectModel(tuple(species), owned, tuple(documents))
