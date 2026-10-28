"""CT stays outside AC until the port tangent is bound; no AC solves here."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
from examples.research_1d.steady import load_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import CapabilityError, DC1DProblem
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.equations.ionic_small_signal import build_ionic_small_signal
from solarlab.experiments import impedance
from solarlab.materials.source import SourceDocument
from solarlab.solvers.nonlinear.steady_1d import SteadyState1D


def with_ct(base: PreparedDevice, carrier: str) -> PreparedDevice:
    document = base.to_input().editing_data()
    absorber = next(layer for layer in document["layers"] if layer["id"] == "absorber")
    absorber["parameters"]["v_sat_" + carrier] = 1e4
    absorber["parameters"]["ct_beta_" + carrier] = 2.0
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    source = SourceDocument("field_transport_ac_guard", raw)
    return resolve_device(DeviceInput.model_validate(document), base.defaults, base.resources,
                          sources=(*base.sources, source))


def forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("unsupported CT reached AC validation, linearization or a linear solve")


@pytest.mark.parametrize("carrier", ["n", "p"])
def test_ion_free_ac_refuses_ct_before_point_validation_or_linear_solve(
    monkeypatch: pytest.MonkeyPatch, carrier: str,
) -> None:
    device = compile_device_1d(with_ct(load_device(), carrier), intervals_per_layer=(12, 12, 12))
    assert device.field_transport is not None
    monkeypatch.setattr(impedance, "_verify_point", forbidden)
    monkeypatch.setattr(impedance, "solve_frequency_systems", forbidden)
    monkeypatch.setattr(DC1DProblem, "jacobian", forbidden)
    monkeypatch.setattr(DC1DProblem, "storage_jacobian", forbidden)
    # Admission must refuse CT before asking for a solved operating point.
    with pytest.raises(CapabilityError, match="CT AC current tangents are not bound"):
        impedance.solve_impedance(device, cast(SteadyState1D, None), [0.0, 1.0])


@pytest.mark.parametrize("carrier", ["n", "p"])
def test_ionic_ac_refuses_ct_before_assembling_any_operator(
    monkeypatch: pytest.MonkeyPatch, carrier: str,
) -> None:
    device = compile_ionic_device_1d(with_ct(prepare_confined_shared_site_ion_device(), carrier),
        intervals_per_layer=(12, 12, 12), ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"))
    assert device.electronic_component.field_transport is not None
    state = device.initial_state()  # A physical reference, never a claimed steady result.
    monkeypatch.setattr(Ionic1DProblem, "jacobian", forbidden)
    monkeypatch.setattr(Ionic1DProblem, "storage_jacobian", forbidden)
    monkeypatch.setattr(DC1DProblem, "jacobian", forbidden)
    monkeypatch.setattr(impedance, "solve_frequency_systems", forbidden)
    with pytest.raises(CapabilityError, match="CT AC current tangents are not bound"):
        build_ionic_small_signal(device, state)


def test_no_ct_ion_free_keeps_original_missing_point_refusal() -> None:
    device = compile_device_1d(load_device(), intervals_per_layer=(12, 12, 12))
    assert device.field_transport is None
    with pytest.raises(ValueError, match="accepted same-device SteadyState1D"):
        impedance.solve_impedance(device, cast(SteadyState1D, None), [1.0])


def test_no_ct_ionic_keeps_original_operator_path_without_executing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = compile_ionic_device_1d(prepare_confined_shared_site_ion_device(),
        intervals_per_layer=(12, 12, 12), ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"))
    state = device.initial_state()
    assert device.electronic_component.field_transport is None
    class ReachedOriginalAssembly(Exception):
        pass
    def reached(*args: Any, **kwargs: Any) -> Any:
        raise ReachedOriginalAssembly
    monkeypatch.setattr(Ionic1DProblem, "jacobian", reached)
    with pytest.raises(ReachedOriginalAssembly):
        build_ionic_small_signal(device, state)
