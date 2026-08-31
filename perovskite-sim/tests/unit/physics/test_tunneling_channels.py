"""D8-E0 the four tunnelling channels, each on its own switch."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.models.tunneling_channels import (
    CHANNEL_NAMES,
    BandToBandTunnellingChannel,
    ContactTunnellingChannel,
    InterfaceDefectAssistedTunnellingChannel,
    IntrabandTunnellingChannel,
    TunnellingChannelDocument,
    TunnellingChannelSchemaError,
    tunnelling_channel_document_from_mapping,
)
from perovskite_sim.physics.tunneling_channels import (
    TunnellingChannelError,
    band_to_band_flux,
    conduction_band_eV,
    contact_tunnelling_flux,
    interface_defect_assisted_rate,
    intraband_flux,
    valence_band_eV,
)


THERMAL_V = 0.025852


def _tilted_bands(field_V_m: float = 1.0e7, points: int = 201):
    x = np.linspace(0.0, 20.0e-9, points)
    potential = -field_V_m * x
    affinity = np.full_like(x, 4.0)
    gap = np.full_like(x, 1.2)
    return (
        x,
        conduction_band_eV(potential, affinity),
        valence_band_eV(potential, affinity, gap),
    )


def _spike(points: int = 201, height_eV: float = 0.35):
    x = np.linspace(0.0, 10.0e-9, points)
    return x, -4.0 + height_eV * np.exp(-(((x - 5.0e-9) / 1.5e-9) ** 2))


def _contact_barrier(points: int = 201, height_eV: float = 0.5):
    x = np.linspace(0.0, 8.0e-9, points)
    return x, -4.0 + height_eV * (1.0 - x / x[-1])


def _trap_inputs():
    return dict(
        trap_energy_eV=-4.2,
        trap_density_m2=1.0e16,
        electron_capture_velocity_m_s=1.0e5,
        hole_capture_velocity_m_s=8.0e4,
        electron_density_m3=1.0e20,
        hole_density_m3=1.0e18,
        electron_reference_density_m3=1.0e17,
        hole_reference_density_m3=1.0e19,
    )


# --------------------------------------------------------------------------
# Zero net current at zero bias, per channel
# --------------------------------------------------------------------------


def test_band_to_band_is_exactly_zero_at_equal_fermi_levels():
    x, conduction, valence = _tilted_bands()
    channel = BandToBandTunnellingChannel(enabled=True, energy_quadrature_order=48)

    flux = band_to_band_flux(
        x,
        conduction,
        valence,
        channel,
        left_fermi_eV=-4.5,
        right_fermi_eV=-4.5,
        thermal_voltage_V=THERMAL_V,
    )

    assert flux.net_flux_m2_s == 0.0
    assert flux.forward_flux_m2_s == flux.reverse_flux_m2_s > 0.0


def test_intraband_is_exactly_zero_at_equal_fermi_levels():
    x, barrier = _spike()
    channel = IntrabandTunnellingChannel(enabled=True, energy_quadrature_order=48)

    flux = intraband_flux(
        x,
        barrier,
        channel,
        carrier="electron",
        left_fermi_eV=-4.0,
        right_fermi_eV=-4.0,
        thermal_voltage_V=THERMAL_V,
    )

    assert flux.net_flux_m2_s == 0.0


def test_contact_is_exactly_zero_when_the_metal_and_semiconductor_align():
    x, barrier = _contact_barrier()
    channel = ContactTunnellingChannel(
        enabled=True, barrier_height_eV=0.5, energy_quadrature_order=48
    )

    flux = contact_tunnelling_flux(
        x,
        barrier,
        channel,
        carrier="electron",
        metal_fermi_eV=-4.0,
        semiconductor_fermi_eV=-4.0,
        thermal_voltage_V=THERMAL_V,
    )

    assert flux.net_flux_m2_s == 0.0


def test_defect_assisted_net_rate_vanishes_at_its_stationary_occupancy():
    """Reported as an occupancy residual, not a bare rate.

    d(rate)/df is ~1e30 here, so representing the stationary occupancy in
    double precision alone perturbs the net rate by ~1e14. Comparing the rate
    against zero would therefore be measuring floating-point spacing; the
    scale-free statement is that the residual, expressed back as an occupancy
    offset, is at machine precision.
    """
    x, barrier = _spike()
    channel = InterfaceDefectAssistedTunnellingChannel(enabled=True)
    inputs = _trap_inputs()

    probe = interface_defect_assisted_rate(
        x, barrier, barrier - 1.2, channel, occupancy=0.5, **inputs
    )
    stationary = interface_defect_assisted_rate(
        x,
        barrier,
        barrier - 1.2,
        channel,
        occupancy=probe.equilibrium_occupancy,
        **inputs,
    )

    assert stationary.stationary_occupancy_residual < 1.0e-14
    assert probe.stationary_occupancy_residual > 1.0e-3
    assert 0.0 < probe.equilibrium_occupancy < 1.0


# --------------------------------------------------------------------------
# Each channel responds to bias, and only through its own switch
# --------------------------------------------------------------------------


def test_every_channel_carries_current_once_the_levels_split():
    x, conduction, valence = _tilted_bands()
    btb = band_to_band_flux(
        x,
        conduction,
        valence,
        BandToBandTunnellingChannel(enabled=True),
        left_fermi_eV=-4.4,
        right_fermi_eV=-4.6,
        thermal_voltage_V=THERMAL_V,
    )
    xs, spike = _spike()
    intra = intraband_flux(
        xs,
        spike,
        IntrabandTunnellingChannel(enabled=True),
        carrier="electron",
        left_fermi_eV=-3.95,
        right_fermi_eV=-4.05,
        thermal_voltage_V=THERMAL_V,
    )
    xc, barrier = _contact_barrier()
    contact = contact_tunnelling_flux(
        xc,
        barrier,
        ContactTunnellingChannel(enabled=True, barrier_height_eV=0.5),
        carrier="electron",
        metal_fermi_eV=-3.9,
        semiconductor_fermi_eV=-4.1,
        thermal_voltage_V=THERMAL_V,
    )

    for flux in (btb, intra, contact):
        assert flux.net_flux_m2_s > 0.0
        assert np.all(flux.transmission >= 0.0)
        assert np.all(flux.transmission <= 1.0)


@pytest.mark.parametrize("channel_name", CHANNEL_NAMES)
def test_a_disabled_channel_refuses_rather_than_returning_zero(channel_name):
    """A disabled channel must not be silently indistinguishable from a
    vanishing one; the exit condition needs per-channel switches to be real."""
    x, conduction, valence = _tilted_bands()
    xs, spike = _spike()
    xc, barrier = _contact_barrier()

    with pytest.raises(TunnellingChannelError, match="disabled"):
        if channel_name == "band_to_band":
            band_to_band_flux(
                x,
                conduction,
                valence,
                BandToBandTunnellingChannel(),
                left_fermi_eV=-4.4,
                right_fermi_eV=-4.6,
                thermal_voltage_V=THERMAL_V,
            )
        elif channel_name == "intraband":
            intraband_flux(
                xs,
                spike,
                IntrabandTunnellingChannel(),
                carrier="electron",
                left_fermi_eV=-3.9,
                right_fermi_eV=-4.1,
                thermal_voltage_V=THERMAL_V,
            )
        elif channel_name == "interface_defect_assisted":
            interface_defect_assisted_rate(
                xs,
                spike,
                spike - 1.2,
                InterfaceDefectAssistedTunnellingChannel(),
                occupancy=0.5,
                **_trap_inputs(),
            )
        else:
            contact_tunnelling_flux(
                xc,
                barrier,
                ContactTunnellingChannel(),
                carrier="electron",
                metal_fermi_eV=-3.9,
                semiconductor_fermi_eV=-4.1,
                thermal_voltage_V=THERMAL_V,
            )


def test_intraband_refuses_a_carrier_the_channel_was_not_configured_for():
    xs, spike = _spike()
    electrons_only = IntrabandTunnellingChannel(enabled=True, carrier="electron")

    with pytest.raises(TunnellingChannelError, match="configured for"):
        intraband_flux(
            xs,
            spike,
            electrons_only,
            carrier="hole",
            left_fermi_eV=-3.9,
            right_fermi_eV=-4.1,
            thermal_voltage_V=THERMAL_V,
        )


def test_channels_refuse_structures_that_cannot_support_them():
    x = np.linspace(0.0, 10.0e-9, 51)
    flat = np.full_like(x, -4.0)

    with pytest.raises(TunnellingChannelError, match="barrier spike"):
        intraband_flux(
            x,
            flat,
            IntrabandTunnellingChannel(enabled=True),
            carrier="electron",
            left_fermi_eV=-3.9,
            right_fermi_eV=-4.1,
            thermal_voltage_V=THERMAL_V,
        )
    with pytest.raises(TunnellingChannelError, match="positive gap"):
        band_to_band_flux(
            x,
            flat,
            flat,
            BandToBandTunnellingChannel(enabled=True),
            left_fermi_eV=-4.0,
            right_fermi_eV=-4.1,
            thermal_voltage_V=THERMAL_V,
        )


def test_band_to_band_flags_a_field_below_its_declared_minimum():
    x, conduction, valence = _tilted_bands(field_V_m=1.0e5)
    channel = BandToBandTunnellingChannel(enabled=True, minimum_field_V_m=1.0e6)

    flux = band_to_band_flux(
        x,
        conduction,
        valence,
        channel,
        left_fermi_eV=-4.4,
        right_fermi_eV=-4.6,
        thermal_voltage_V=THERMAL_V,
    )

    assert "field_below_channel_minimum" in flux.notes
    assert flux.valid is False


def test_transmission_falls_as_the_barrier_grows_in_every_channel():
    fluxes = []
    for height in (0.2, 0.4, 0.6):
        xs, spike = _spike(height_eV=height)
        fluxes.append(
            intraband_flux(
                xs,
                spike,
                IntrabandTunnellingChannel(enabled=True),
                carrier="electron",
                left_fermi_eV=-3.9,
                right_fermi_eV=-4.1,
                thermal_voltage_V=THERMAL_V,
            ).net_flux_m2_s
        )
    assert all(a > b for a, b in zip(fluxes, fluxes[1:])), fluxes


# --------------------------------------------------------------------------
# Canonical document
# --------------------------------------------------------------------------


def test_document_defaults_to_every_channel_disabled():
    document = TunnellingChannelDocument()

    assert document.enabled_channels == ()
    assert document.any_enabled is False
    assert len(document.sha256) == 64


def test_channels_are_independently_switchable_and_change_the_hash():
    baseline = TunnellingChannelDocument()
    seen = {baseline.sha256}

    for name, enabled in (
        ("band_to_band", BandToBandTunnellingChannel(enabled=True)),
        ("intraband", IntrabandTunnellingChannel(enabled=True)),
        (
            "interface_defect_assisted",
            InterfaceDefectAssistedTunnellingChannel(enabled=True),
        ),
        (
            "contact",
            ContactTunnellingChannel(enabled=True, barrier_height_eV=0.4),
        ),
    ):
        document = replace(baseline, **{name: enabled})
        assert document.enabled_channels == (name,)
        assert document.sha256 not in seen
        seen.add(document.sha256)


def test_document_round_trips_and_rejects_unknown_keys():
    document = TunnellingChannelDocument(
        band_to_band=BandToBandTunnellingChannel(enabled=True)
    )
    assert TunnellingChannelDocument.from_dict(document.to_dict()).sha256 == (
        document.sha256
    )

    payload = document.to_dict()
    payload["band_to_band"]["unexpected"] = 1.0
    with pytest.raises(TunnellingChannelSchemaError, match="schema mismatch"):
        TunnellingChannelDocument.from_dict(payload)


def test_contact_channel_requires_a_barrier_when_enabled():
    with pytest.raises(TunnellingChannelSchemaError, match="barrier_height_eV"):
        ContactTunnellingChannel(enabled=True, barrier_height_eV=0.0)


def test_defect_assisted_channel_cannot_waive_an_explicit_occupancy():
    with pytest.raises(
        TunnellingChannelSchemaError,
        match="algebraically eliminated",
    ):
        InterfaceDefectAssistedTunnellingChannel(
            enabled=True,
            requires_explicit_occupancy=False,
        )


def test_device_mapping_parses_yaml_style_scalars_and_defaults_missing_channels():
    document = tunnelling_channel_document_from_mapping(
        {
            "tunnelling_channels": {
                # unsigned exponent: YAML 1.1 hands this over as a string
                "band_to_band": {
                    "enabled": True,
                    "reduced_effective_mass_rel": 0.08,
                    "energy_quadrature_order": 16,
                    "minimum_field_V_m": "1.0e6",
                }
            }
        }
    )

    assert document is not None
    assert document.enabled_channels == ("band_to_band",)
    assert document.band_to_band.minimum_field_V_m == 1.0e6
    assert document.contact.enabled is False
    assert tunnelling_channel_document_from_mapping({}) is None
