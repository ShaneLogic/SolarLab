"""Bulk carrier transport: historical MB/FD face currents and the FD capability gate.

The namespace exports the M-class (legacy bit-identical) transport. The
unified directed particle flux in stable electrochemical form (N class) is
:mod:`solarlab.physics.transport.electrochemical_flux`.
"""

from solarlab.physics.transport.drift_diffusion import (
    legacy_carrier_face_currents, legacy_face_diffusivity,
)
from solarlab.physics.transport.generalized import (
    BLAKEMORE, RECOMBINATION_OFF, GeneralizedCarrierFaceStatistics,
    TransportCapabilityError, TransportCapabilityRequest,
    generalized_carrier_face_statistics, legacy_fermi_dirac_face_currents,
    legacy_generalized_sg_current_n, legacy_generalized_sg_current_p,
    require_carrier_transport,
)

__all__ = [
    "BLAKEMORE", "GeneralizedCarrierFaceStatistics", "RECOMBINATION_OFF",
    "TransportCapabilityError", "TransportCapabilityRequest",
    "generalized_carrier_face_statistics", "legacy_carrier_face_currents",
    "legacy_face_diffusivity", "legacy_fermi_dirac_face_currents",
    "legacy_generalized_sg_current_n", "legacy_generalized_sg_current_p",
    "require_carrier_transport",
]
