"""The accepted CT steady binding is not an admitted CT J-V source binding."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NoReturn

import pytest

from examples.research_1d.steady import load_device
from examples.research_2d.dc_assembly import coordinates, spatial_input
from examples.research_2d.field_transport import prepare_case
from examples.research_2d.grains import spatial_input as grain_spatial_input
from solarlab.device.compile_2d import compile_device_2d
from solarlab.equations.dc_2d import CompiledDevice2D
from solarlab.experiments.jv import steady_2d as driver
from solarlab.solvers.nonlinear import steady_2d as solver


@pytest.fixture(params=(False, True), ids=("uniform", "neutral_grain"))
def ct_device(request: pytest.FixtureRequest) -> CompiledDevice2D:
    spatial = grain_spatial_input() if request.param else spatial_input()
    device = compile_device_2d(prepare_case(), spatial)
    assert device.field_transport is not None
    solver._scope(device)  # The reviewed single-point steady scope admits CT.
    return device


@pytest.mark.parametrize("entry", ("run", "continue", "initial"))
def test_active_field_transport_is_refused_before_jv_work(
    ct_device: CompiledDevice2D, entry: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("CT J-V reached source capture, diagnostics or a solver")

    for name in ("solve_steady_state_2d", "_sources", "_steady_sources", "_diagnostics"):
        monkeypatch.setattr(driver, name, forbidden)
    monkeypatch.setattr(solver, "solve_newton", forbidden)
    _, u = coordinates(ct_device, lateral=False)
    text = "scope-only unaccepted numerical fixture"
    digest = hashlib.sha256(text.encode()).hexdigest()
    initial = driver.SteadyJV2DInitial(ct_device.state(u, voltage_V=.002), 1.0, "scope-only", digest, text)
    journal = tmp_path / "not_started"
    with pytest.raises(ValueError, match=r"^unsupported_field_transport_2d_jv:"):
        if entry == "run":
            driver.run_steady_jv_2d(ct_device, initial, [.002], checkpoint_directory=journal)
        elif entry == "continue":
            driver.continue_steady_jv_2d(ct_device, tmp_path / "unread_prefix.json", digest,
                                       checkpoint_directory=journal)
        else:
            driver._admit(ct_device, text, digest, driver.SteadyJV2DOptions().steady)
    assert not journal.exists()


@pytest.mark.parametrize("grain", (False, True), ids=("uniform", "neutral_grain"))
def test_disabled_field_transport_keeps_existing_jv_scope(grain: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("scope check attempted a solve")

    monkeypatch.setattr(driver, "solve_steady_state_2d", forbidden)
    device = compile_device_2d(load_device(), grain_spatial_input() if grain else spatial_input())
    assert device.field_transport is None and device.column.field_transport is None
    driver._scope(device)
