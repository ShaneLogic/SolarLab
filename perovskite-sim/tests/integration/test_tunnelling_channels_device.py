"""Resolved electron-path wiring and explicitly unvalidated device couplings.

The old thick-spike input remains a retraction/control case. The new resolved
barrier certificate separately checks an observable terminal-current effect.
Local formulas for other channels do not establish their device injection.
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
    evaluate_tunnelling_channels,
    physical_quasi_fermi_levels_eV,
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


def test_retracted_fixture_does_not_retain_the_old_large_flux_claim():
    """Correct physical occupations must not preserve the withdrawn magnitude."""
    without = _solve()
    with_channel = _solve(_intraband())
    diagnostics = with_channel.tunnelling_channel_diagnostics

    assert with_channel.certified is True
    assert diagnostics is not None
    assert diagnostics.channel_names == ("intraband_electron",)
    net = diagnostics.channel_net_flux_m2_s[0]
    assert net != 0.0
    assert abs(Q * net) < 0.01 * abs(without.current_A_m2)


def test_nonlocal_injection_preserves_particle_number_and_its_current_profile():
    """A multi-cell transition has zero total particle source, not one face."""
    result = _solve(_intraband())
    diagnostics = result.tunnelling_channel_diagnostics
    face_current = np.asarray(diagnostics.electron_face_current_A_m2)

    assert np.count_nonzero(face_current) > 3
    transfer = np.diff(np.r_[0.0, face_current, 0.0]) / Q
    assert abs(float(np.sum(transfer))) < 1e-12 * float(np.sum(np.abs(transfer)))
    np.testing.assert_array_equal(face_current, -Q * diagnostics.intraband_path.particle_face_flux_m2_s)
    assert np.all(np.asarray(diagnostics.hole_face_current_A_m2) == 0.0)


def test_equilibrium_device_current_is_bounded_without_saturated_occupations():
    """A solved equilibrium has finite residuals; its occupations must be informative."""
    result = _solve(_intraband(), V_app=0.0)
    diagnostics = result.tunnelling_channel_diagnostics

    assert abs(Q * diagnostics.channel_net_flux_m2_s[0]) < 1e-8
    assert np.max(np.abs(diagnostics.electron_face_current_A_m2)) < 1e-8
    for occupation in (diagnostics.intraband_path.flux.left_occupation,
                       diagnostics.intraband_path.flux.right_occupation):
        assert np.all((occupation > 0.0) & (occupation < 1e-3))


# --------------------------------------------------------------------------
# The channel is driven by its own barrier, not by the applied bias.
# --------------------------------------------------------------------------


def test_the_channel_is_driven_by_the_local_drop_not_the_contact_split():
    """Changing unused contact levels cannot change an interior path."""
    stack = _stack(_intraband())
    grid = _grid(stack)
    material = build_material_arrays(grid, stack)
    state = _solve(_intraband())
    fn, fp = physical_quasi_fermi_levels_eV(
        potential_V=state.phi, affinity_eV=material.chi_phys, band_gap_eV=material.Eg_phys,
        electron_density_m3=state.y[:grid.size], hole_density_m3=state.y[grid.size:2 * grid.size],
        conduction_dos_m3=material.N_C_physical, valence_dos_m3=material.N_V_physical,
        thermal_voltage_V=material.V_T_device,
    )
    kwargs = dict(positions_m=grid, potential_V=state.phi, affinity_eV=material.chi_phys,
                  band_gap_eV=material.Eg_phys, electron_quasi_fermi_eV=fn,
                  hole_quasi_fermi_eV=fp, thermal_voltage_V=material.V_T_device)
    original = evaluate_tunnelling_channels(material.tunnelling_channels, **kwargs)
    moved = fn.copy()
    moved[0] += 0.5
    moved[-1] -= 0.4
    altered = evaluate_tunnelling_channels(material.tunnelling_channels,
                                           **dict(kwargs, electron_quasi_fermi_eV=moved))
    np.testing.assert_array_equal(altered.electron_face_current_A_m2, original.electron_face_current_A_m2)


def test_illuminated_state_retains_physical_nonsaturated_channel_occupations():
    result = _solve(_intraband(), V_app=0.2, illuminated=True)
    assert result.certified is True
    path = result.tunnelling_channel_diagnostics.intraband_path
    assert path.flux.net_flux_m2_s != 0.0
    assert np.max(path.flux.left_occupation) < 0.1
    assert np.max(path.flux.right_occupation) < 0.1


def test_the_transmission_audit_reports_the_opaque_end_of_the_window():
    """The open energy rule reports actual actions at both spectral ends."""
    diagnostics = _solve(_intraband()).tunnelling_channel_diagnostics

    least = diagnostics.channel_minimum_transmission[0]
    assert 0.0 < least < diagnostics.channel_maximum_transmission[0] < 1.0
    assert least == pytest.approx(np.exp(-2 * np.max(diagnostics.intraband_path.actions)), rel=1e-12)


def test_every_diagnostic_tuple_has_one_entry_per_channel():
    """A mismatched tuple would silently misattribute a flux to a channel."""
    diagnostics = _solve(_intraband()).tunnelling_channel_diagnostics
    count = len(diagnostics.channel_names)

    assert count == 1
    for field in (
        diagnostics.channel_net_flux_m2_s,
        diagnostics.channel_maximum_transmission,
        diagnostics.channel_minimum_transmission,
        diagnostics.channel_valid,
        diagnostics.channel_notes,
    ):
        assert len(field) == count


def test_band_to_band_requires_its_own_pair_injection_validation():
    document = TunnellingChannelDocument(
        band_to_band=BandToBandTunnellingChannel(
            enabled=True, energy_quadrature_order=16
        )
    )
    with pytest.raises(TunnellingChannelCapabilityError, match="conservative device validation"):
        _solve(document)


def test_the_contact_channel_needs_a_stack_that_has_a_contact_barrier():
    """The local contact primitive is not a completed device injection model."""
    document = TunnellingChannelDocument(
        contact=ContactTunnellingChannel(
            enabled=True,
            side="left",
            barrier_height_eV=0.3,
            energy_quadrature_order=16,
        )
    )
    with pytest.raises((QuasiFermiSteadyStateError, TunnellingChannelCapabilityError)):
        _solve(document)


# --------------------------------------------------------------------------
# Each channel switches independently at device level.
# --------------------------------------------------------------------------


def test_combining_an_unvalidated_channel_cannot_silently_run_the_electron_subset():
    both = TunnellingChannelDocument(
        intraband=IntrabandTunnellingChannel(
            enabled=True, carrier="electron", energy_quadrature_order=16
        ),
        band_to_band=BandToBandTunnellingChannel(
            enabled=True, energy_quadrature_order=16
        ),
    )
    with pytest.raises(TunnellingChannelCapabilityError, match="conservative device validation"):
        _solve(both)


@pytest.mark.parametrize("carrier", ["hole", "both"])
def test_hole_device_coupling_is_not_promoted_from_electron_evidence(carrier):
    document = TunnellingChannelDocument(
        intraband=IntrabandTunnellingChannel(
            enabled=True, carrier=carrier, energy_quadrature_order=24
        )
    )
    with pytest.raises(TunnellingChannelCapabilityError, match="conservative device validation"):
        _solve(document)


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
