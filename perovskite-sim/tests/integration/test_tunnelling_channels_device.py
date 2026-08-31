"""D8-E1: the WKB tunnelling family wired into the guarded QF/DC lane.

D8-E0 proved the four channels are correct in isolation. What these tests
cover is the wiring: that an enabled channel actually reaches the residual
(rather than producing a plausible diagnostic and no current), that it is
driven by the quasi-Fermi drop across its *own* barrier rather than by the
applied bias, that a disabled family is bit-identical, and that every route
other than the certified QF/DC lane fails closed.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    QuasiFermiSteadyStateError,
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.models.tunneling_channels import (
    BandToBandTunnellingChannel,
    ContactTunnellingChannel,
    InterfaceDefectAssistedTunnellingChannel,
    IntrabandTunnellingChannel,
    TunnellingChannelDocument,
    TunnellingChannelSchemaError,
)
from perovskite_sim.physics.temperature import thermal_voltage
from perovskite_sim.reproducibility import semantic_sha256
from perovskite_sim.physics.tunneling_channel_device import (
    TunnellingChannelCapabilityError,
    compile_tunnelling_channels,
)
from perovskite_sim.solver.mol import assemble_rhs, build_material_arrays


TEMPERATURE_K = 300.0
NC_M3 = 1.0e24
NV_M3 = 8.0e23


def _params(gap_eV: float, affinity_eV: float, **overrides) -> MaterialParams:
    intrinsic = math.sqrt(
        NC_M3 * NV_M3 * math.exp(-gap_eV / thermal_voltage(TEMPERATURE_K))
    )
    fields = dict(
        eps_r=20.0,
        mu_n=2.0e-3,
        mu_p=2.0e-3,
        D_ion=0.0,
        P_lim=1.0e30,
        P0=0.0,
        ni=intrinsic,
        tau_n=1.0e-6,
        tau_p=1.0e-6,
        n1=intrinsic,
        p1=intrinsic,
        B_rad=0.0,
        C_n=0.0,
        C_p=0.0,
        alpha=0.0,
        N_A=0.0,
        N_D=0.0,
        chi=affinity_eV,
        Eg=gap_eV,
        Nc300=NC_M3,
        Nv300=NV_M3,
    )
    fields.update(overrides)
    return MaterialParams(**fields)


# A p-n junction with a thin wide-gap interlayer, so the conduction band
# carries a real spike that several grid nodes resolve. A single-node offset
# would give the WKB integrator no width to work with and every transmission
# would be exactly 1 — a barrier the test could not distinguish from none.
_ABSORBER = _params(1.5, 4.0, N_A=1.0e22, alpha=4.0e5)
# The interlayer is offset in BOTH bands: chi 0.3 eV lower gives the
# electron spike, and chi + Eg 0.3 eV higher gives the hole spike. Offsetting
# only chi would leave E_V continuous, and the hole channel would then be
# refused for the honest reason that its barrier does not exist.
_SPIKE = _params(2.1, 3.7, N_D=1.0e21)
_ETL = _params(1.5, 4.0, N_D=1.0e22)


def _stack(
    document: TunnellingChannelDocument | None = None,
    *,
    photon_flux_m2_s: float = 1.0e21,
    spike: bool = True,
) -> DeviceStack:
    layers = (
        (
            LayerSpec("absorber", 150.0e-9, _ABSORBER, "absorber"),
            LayerSpec("spike", 12.0e-9, _SPIKE, "ETL"),
            LayerSpec("etl", 100.0e-9, _ETL, "ETL"),
        )
        if spike
        else (LayerSpec("absorber", 250.0e-9, _ABSORBER, "absorber"),)
    )
    return DeviceStack(
        layers=layers,
        V_bi=0.0,
        Phi=photon_flux_m2_s,
        interfaces=tuple((0.0, 0.0) for _ in layers[:-1]),
        mode="legacy",
        built_in_potential_mode="semiconductor_work_function",
        tunnelling_channels=document,
    )


def _grid(stack: DeviceStack) -> np.ndarray:
    counts = {150.0e-9: 12, 12.0e-9: 10, 100.0e-9: 12, 250.0e-9: 16}
    return multilayer_grid(
        [Layer(layer.thickness, counts[layer.thickness]) for layer in stack.layers]
    )


def _intraband(order: int = 24) -> TunnellingChannelDocument:
    return TunnellingChannelDocument(
        intraband=IntrabandTunnellingChannel(
            enabled=True, carrier="electron", energy_quadrature_order=order
        )
    )


def _solve(document=None, *, V_app=0.2, illuminated=False, spike=True):
    stack = _stack(document, spike=spike)
    return solve_quasi_fermi_steady_state(
        _grid(stack), stack, V_app=V_app, illuminated=illuminated
    )


# --------------------------------------------------------------------------
# The disabled family must not exist as far as the solver is concerned.
# --------------------------------------------------------------------------


def test_no_document_leaves_the_lane_untouched():
    result = _solve()

    assert result.certified is True
    assert result.tunnelling_channel_diagnostics is None


def test_an_all_disabled_document_is_bit_identical_to_no_document():
    """A document that enables nothing must compile away entirely.

    Not merely "small": a default-constructed document has to be the same
    object-level no-op as omitting it, or every shipped result silently
    depends on whether the key is present in the YAML.
    """
    without = _solve()
    with_inert = _solve(TunnellingChannelDocument())

    assert with_inert.tunnelling_channel_diagnostics is None
    assert np.array_equal(without.y, with_inert.y)
    assert with_inert.current_A_m2 == without.current_A_m2


def test_an_inert_family_does_not_move_a_configs_semantic_hash():
    """Adding an inert capability must not re-address every shipped config.

    Introducing the field moved the frozen semantic SHA-256 of every config in
    the tree, including ones that have nothing to do with tunnelling. The rule
    is on whether a channel is ENABLED rather than whether the key is present:
    an all-disabled document compiles away entirely and is bit-identical to
    omitting it, so hashing it differently would content-address a distinction
    the solver cannot make.
    """
    bare = semantic_sha256(_stack())
    inert = semantic_sha256(_stack(TunnellingChannelDocument()))
    active = semantic_sha256(_stack(_intraband()))

    assert inert == bare
    assert active != bare


def test_an_inert_document_compiles_to_none_rather_than_an_empty_evaluator():
    assert (
        compile_tunnelling_channels(
            TunnellingChannelDocument(), node_count=32, interface_nodes=(12,)
        )
        is None
    )
    assert (
        compile_tunnelling_channels(None, node_count=32, interface_nodes=(12,)) is None
    )


def test_shipped_presets_carry_no_tunnelling_channels():
    """The family is opt-in; nothing in the tree may switch it on implicitly."""
    stack = _stack()
    mat = build_material_arrays(_grid(stack), stack)

    assert stack.tunnelling_channels is None
    assert mat.tunnelling_channels is None


# --------------------------------------------------------------------------
# An enabled channel must reach the residual, not just the diagnostics.
# --------------------------------------------------------------------------


def test_an_enabled_channel_changes_the_certified_terminal_current():
    """This is the wiring test, and it has a specific defect in its sights.

    The interface plane zeroes its own face current so its reservoir transfer
    is not double-counted in the divergence. A tunnelling current injected
    before that zeroing is deleted: the diagnostics still report a perfectly
    good flux and the terminal current does not move at all. Asserting on the
    diagnostics alone would pass against that bug, so the assertion here is on
    the solved current.
    """
    without = _solve()
    with_channel = _solve(_intraband())
    diagnostics = with_channel.tunnelling_channel_diagnostics

    assert with_channel.certified is True
    assert diagnostics is not None
    assert diagnostics.channel_names == ("intraband_electron",)
    net = diagnostics.channel_net_flux_m2_s[0]
    assert net != 0.0
    assert with_channel.current_A_m2 != without.current_A_m2
    # The shift must be of the order the channel itself reports, not noise.
    shift = abs(with_channel.current_A_m2 - without.current_A_m2)
    assert shift <= abs(Q * net) * 10.0


def test_the_injected_face_current_matches_the_reported_flux():
    """Charge is not created between the channel and the face array."""
    result = _solve(_intraband())
    diagnostics = result.tunnelling_channel_diagnostics
    face_current = np.asarray(diagnostics.electron_face_current_A_m2)

    assert np.count_nonzero(face_current) == 1
    assert float(face_current.sum()) == pytest.approx(
        -Q * diagnostics.channel_net_flux_m2_s[0], rel=1.0e-12
    )
    assert np.all(np.asarray(diagnostics.hole_face_current_A_m2) == 0.0)


def test_equilibrium_net_flux_is_exactly_zero_through_the_device_wiring():
    """Reciprocity must survive the wiring, not just hold in the primitive."""
    result = _solve(_intraband(), V_app=0.0)
    diagnostics = result.tunnelling_channel_diagnostics

    assert diagnostics.channel_net_flux_m2_s[0] == 0.0
    assert np.all(np.asarray(diagnostics.electron_face_current_A_m2) == 0.0)


# --------------------------------------------------------------------------
# The channel is driven by its own barrier, not by the applied bias.
# --------------------------------------------------------------------------


def test_the_channel_is_driven_by_the_local_drop_not_the_contact_split():
    """A channel is one conduction path across ONE barrier.

    Its driving force is the quasi-Fermi drop across that barrier — the same
    drop the Scharfetter-Gummel flux on the same face sees, which is what
    makes the two additive rather than double-counted. Reading the contact
    levels instead would drive every interface channel with the full applied
    bias, inflating the flux by orders of magnitude and making it *grow* with
    bias. The measured behaviour is the opposite: raising the bias flattens
    the junction, the local drop shrinks, and so does the tunnelling flux.
    """
    low = _solve(_intraband(), V_app=0.2)
    high = _solve(_intraband(), V_app=0.5)

    low_flux = abs(low.tunnelling_channel_diagnostics.channel_net_flux_m2_s[0])
    high_flux = abs(high.tunnelling_channel_diagnostics.channel_net_flux_m2_s[0])

    assert low_flux > 0.0
    assert high_flux < low_flux
    # Contact-level driving would scale the occupation difference up with the
    # applied bias; an order-of-magnitude fall in the opposite direction is
    # not reachable that way.
    assert high_flux < 0.1 * low_flux


def test_states_far_below_both_quasi_fermi_levels_carry_no_net_flux():
    """Full on both sides means no net current, however opaque the barrier.

    Under strong illumination the electron quasi-Fermi level here sits about
    an eV above the spike, so every energy in the tunnelling window is
    occupied on both sides. A channel that reported current in that regime
    would be double-counting carriers the drift-diffusion flux already carries
    over the barrier.
    """
    result = _solve(_intraband(), V_app=0.2, illuminated=True)

    assert result.certified is True
    assert result.tunnelling_channel_diagnostics.channel_net_flux_m2_s[0] == 0.0


def test_the_transmission_audit_reports_the_opaque_end_of_the_window():
    """A diagnostic that is 1.0 by construction is not a measurement.

    Every energy window here runs up to the barrier top, so the maximum
    transmission is 1 and the corresponding action is 0 whatever the barrier
    does. The informative pair is at the other end, and it is what the
    channel's opacity has to be read from.
    """
    diagnostics = _solve(_intraband()).tunnelling_channel_diagnostics

    assert diagnostics.channel_maximum_transmission[0] == pytest.approx(1.0)
    least = diagnostics.channel_minimum_transmission[0]
    assert 0.0 < least < 1.0


def test_every_diagnostic_tuple_has_one_entry_per_channel():
    """A mismatched tuple would silently misattribute a flux to a channel."""
    document = TunnellingChannelDocument(
        intraband=IntrabandTunnellingChannel(
            enabled=True, carrier="both", energy_quadrature_order=16
        ),
        band_to_band=BandToBandTunnellingChannel(
            enabled=True, energy_quadrature_order=16
        ),
    )
    diagnostics = _solve(document).tunnelling_channel_diagnostics
    count = len(diagnostics.channel_names)

    assert count == 3
    for field in (
        diagnostics.channel_net_flux_m2_s,
        diagnostics.channel_maximum_transmission,
        diagnostics.channel_minimum_transmission,
        diagnostics.channel_valid,
        diagnostics.channel_notes,
    ):
        assert len(field) == count


# --------------------------------------------------------------------------
# Each channel switches independently at device level.
# --------------------------------------------------------------------------


def test_each_channel_reports_under_its_own_name_and_only_when_enabled():
    both = TunnellingChannelDocument(
        intraband=IntrabandTunnellingChannel(
            enabled=True, carrier="electron", energy_quadrature_order=16
        ),
        band_to_band=BandToBandTunnellingChannel(
            enabled=True, energy_quadrature_order=16
        ),
    )
    result = _solve(both)
    names = result.tunnelling_channel_diagnostics.channel_names

    assert set(names) == {"intraband_electron", "band_to_band"}
    assert len(names) == len(set(names))


def test_the_two_carrier_intraband_channel_reports_both_carriers():
    document = TunnellingChannelDocument(
        intraband=IntrabandTunnellingChannel(
            enabled=True, carrier="both", energy_quadrature_order=16
        )
    )
    result = _solve(document)
    diagnostics = result.tunnelling_channel_diagnostics

    assert diagnostics.channel_names == ("intraband_electron", "intraband_hole")
    assert len(diagnostics.channel_net_flux_m2_s) == 2


# --------------------------------------------------------------------------
# Everything outside the certified lane fails closed.
# --------------------------------------------------------------------------


def test_the_transient_rhs_refuses_a_stack_carrying_tunnelling_channels():
    """The channels are certified on the QF/DC lane only.

    The transient RHS would otherwise integrate them without any of the
    residual certification the lane provides, which is exactly the silent
    substitution this guard exists to prevent.
    """
    stack = _stack(_intraband())
    x = _grid(stack)
    mat = build_material_arrays(x, stack)

    assert mat.tunnelling_channels is not None
    state = np.concatenate(
        [
            np.full(x.size, 1.0e18),
            np.full(x.size, 1.0e18),
            np.zeros(x.size),
        ]
    )
    with pytest.raises(RuntimeError, match="tunnelling"):
        assemble_rhs(0.0, state, x, stack, mat, V_app=0.0)
    # A frozen potential must not open a side door: it excuses a missing
    # Poisson charge, never a missing current.
    with pytest.raises(RuntimeError, match="tunnelling"):
        assemble_rhs(0.0, state, x, stack, mat, V_app=0.0, phi_frozen=np.zeros(x.size))


def test_an_interface_bound_channel_needs_a_heterointerface():
    """A single-layer stack has no interface for the channel to sit on."""
    with pytest.raises(
        (TunnellingChannelCapabilityError, QuasiFermiSteadyStateError),
        match="interface",
    ):
        _solve(_intraband(), spike=False)


def test_the_defect_assisted_channel_refuses_a_lane_without_an_occupancy():
    """Its schema flag cannot be waived by the device layer.

    The default lane eliminates the interface occupancy algebraically, so
    there is no occupancy for this channel to bind to. Fabricating one would
    make the channel look supported everywhere.
    """
    document = TunnellingChannelDocument(
        interface_defect_assisted=InterfaceDefectAssistedTunnellingChannel(enabled=True)
    )
    with pytest.raises(
        (TunnellingChannelCapabilityError, QuasiFermiSteadyStateError),
        match="occupancy",
    ):
        _solve(document)


def test_a_zero_contact_barrier_is_refused_by_the_schema():
    """No barrier is not the same statement as a transparent barrier.

    This is caught at document construction rather than at the device, so an
    ohmic contact can never be described as a tunnelling channel with zero
    height and then quietly contribute nothing.
    """
    with pytest.raises(TunnellingChannelSchemaError, match="barrier_height_eV"):
        ContactTunnellingChannel(
            enabled=True,
            side="left",
            barrier_height_eV=0.0,
            energy_quadrature_order=16,
        )


def test_a_grid_too_coarse_to_hold_the_channel_fails_closed():
    stack = _stack(_intraband())
    with pytest.raises(TunnellingChannelCapabilityError, match="three electrical"):
        compile_tunnelling_channels(
            stack.tunnelling_channels, node_count=2, interface_nodes=(1,)
        )


def test_an_interface_face_outside_the_transport_faces_fails_closed():
    stack = _stack(_intraband())
    with pytest.raises(TunnellingChannelCapabilityError, match="outside"):
        compile_tunnelling_channels(
            stack.tunnelling_channels, node_count=8, interface_nodes=(8,)
        )


def test_the_document_identity_is_carried_through_to_the_diagnostics():
    """A result must say which channel document produced it."""
    document = _intraband()
    result = _solve(document)

    assert result.tunnelling_channel_diagnostics.identity_sha256 == document.sha256


def test_changing_a_channel_parameter_changes_the_document_identity():
    coarse = _intraband(order=16)
    fine = _intraband(order=32)

    assert coarse.sha256 != fine.sha256


def test_a_frozen_stack_replacement_keeps_the_channels_immutable():
    """The stack is frozen; the channels must not be a mutable back door."""
    document = _intraband()
    stack = _stack(document)
    stripped = replace(stack, tunnelling_channels=None)

    assert stack.tunnelling_channels is document
    assert stripped.tunnelling_channels is None
    diagnostics = _solve(document).tunnelling_channel_diagnostics
    assert np.asarray(diagnostics.electron_face_current_A_m2).flags.writeable is False
