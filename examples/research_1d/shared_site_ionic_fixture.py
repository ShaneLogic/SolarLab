"""Explicit shared-site absorber fixture for bounded operator/driver evidence."""

from __future__ import annotations

from examples.research_1d.ionic_multilayer import prepare_confined_dual_ion_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice


def prepare_confined_shared_site_ion_device(
    *, site_capacity_m3: float = 2e24,
    positive_reference_m3: float = 1e24, negative_reference_m3: float = 6e23,
    positive_diffusivity_m2_s: float = 1e-16, negative_diffusivity_m2_s: float = 5e-17,
) -> PreparedDevice:
    """Same electronic layers and separate backgrounds, one common ion capacity.

    Default total reference occupancy is 0.8. The compiler requires explicit
    blocking absorber endpoints and checks both references and the vacancy.
    """
    template = prepare_confined_dual_ion_device(
        positive_diffusivity_m2_s=positive_diffusivity_m2_s,
        negative_diffusivity_m2_s=negative_diffusivity_m2_s,
    )
    data = template.to_input().editing_data()
    data["id"] = "confined_absorber_shared_site_ions"
    data["settings"]["ion_steric_shared_site"] = True
    absorber = next(layer for layer in data["layers"] if layer["role"].lower() == "absorber")
    absorber["parameters"].update(P0=positive_reference_m3, P0_neg=negative_reference_m3,
                                  P_lim=site_capacity_m3, P_lim_neg=site_capacity_m3)
    return resolve_device(DeviceInput.model_validate(data), template.defaults, template.resources, sources=template.sources)
