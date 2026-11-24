"""Bounded HTTP routes for 1D iodine chemistry steady, impedance, and aging runs.

Keeps module import completely free of ``scipy``, ``solarlab.equations``,
``solarlab.solvers``, and ``solarlab.experiments`` so historical ``RunStore``
readback and registered job supervisors never load numerical solver modules at
application startup.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from starlette.concurrency import run_in_threadpool

router = APIRouter(prefix="/iodine", tags=["iodine"])

_MAX_HTTP_INTERVALS = 48
_MAX_HTTP_FREQS = 61
_MAX_HTTP_STAGES = 8
_MAX_HTTP_BDF_STEPS = 600


class IodineHttpRunRequest(BaseModel):
    """Bounded HTTP request selecting a packaged preset or inline declarative deck."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    preset: str | None = None
    deck: dict[str, Any] | None = None
    overrides: dict[str, Any] | None = None
    persist_to_store: bool = False
    run_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}$",
    )

    @model_validator(mode="after")
    def _validate_target(self) -> IodineHttpRunRequest:
        from solarlab.materials.iodine_presets import PACKAGED_IODINE_PRESET_NAMES

        if (self.preset is None) == (self.deck is None):
            raise ValueError("specify exactly one of 'preset' or 'deck'")
        if self.preset is not None and self.preset not in PACKAGED_IODINE_PRESET_NAMES:
            raise ValueError(
                f"unknown packaged iodine preset {self.preset!r}; "
                f"available: {list(PACKAGED_IODINE_PRESET_NAMES)}"
            )
        return self


def _resolve_bounded_http_deck(body: IodineHttpRunRequest) -> tuple[Any, Any]:
    from solarlab.experiments.iodine_declarative import (
        IodineDeckInput,
        IodineOverridesInput,
        apply_iodine_deck_overrides,
        load_iodine_deck,
    )

    if body.preset is not None:
        base_deck, src = load_iodine_deck(body.preset)
    else:
        assert body.deck is not None
        base_deck, src = load_iodine_deck(body.deck)
    parsed_overrides = (
        None
        if body.overrides is None
        else IodineOverridesInput.model_validate(body.overrides)
    )
    effective = apply_iodine_deck_overrides(base_deck, parsed_overrides)

    intervals = effective.electronic_stack.intervals_per_layer
    if isinstance(intervals, int):
        if intervals > _MAX_HTTP_INTERVALS:
            raise ValueError(
                f"HTTP intervals_per_layer ({intervals}) exceeds {_MAX_HTTP_INTERVALS}"
            )
    else:
        if len(intervals) > 6 or any(n > _MAX_HTTP_INTERVALS for n in intervals):
            raise ValueError(
                f"HTTP intervals_per_layer entries must not exceed {_MAX_HTTP_INTERVALS}"
            )
    if effective.impedance.n_frequencies > _MAX_HTTP_FREQS:
        raise ValueError(
            f"HTTP n_frequencies ({effective.impedance.n_frequencies}) exceeds {_MAX_HTTP_FREQS}"
        )
    if effective.aging is not None:
        if len(effective.aging.stages) > _MAX_HTTP_STAGES:
            raise ValueError(
                f"HTTP aging stages ({len(effective.aging.stages)}) exceed {_MAX_HTTP_STAGES}"
            )
        if effective.aging.bdf.max_steps > _MAX_HTTP_BDF_STEPS:
            data = effective.model_dump(mode="python")
            data["aging"]["bdf"]["max_steps"] = _MAX_HTTP_BDF_STEPS
            effective = IodineDeckInput.model_validate(data)
    return effective, src


@router.get("/presets")
def get_iodine_presets() -> dict[str, Any]:
    from solarlab.materials.iodine_presets import list_packaged_iodine_preset_summaries

    return {
        "schema": "solarlab.iodine-presets.v1",
        "presets": list_packaged_iodine_preset_summaries(),
        "scientific_acceptance": "not_evaluated",
        "scientific_qualification": False,
    }


@router.get("/schema")
def get_iodine_schema() -> dict[str, Any]:
    from solarlab.experiments.iodine_declarative import export_iodine_deck_schema

    return export_iodine_deck_schema()


@router.post("/steady")
async def post_iodine_steady(body: IodineHttpRunRequest) -> dict[str, Any]:
    from solarlab.contracts import ContractError
    from solarlab.equations.dc_1d import CapabilityError
    from solarlab.experiments.iodine_declarative import execute_iodine_steady_deck
    from solarlab.solvers.nonlinear.steady_1d import SteadyStateError

    try:
        deck, src = _resolve_bounded_http_deck(body)
        return await run_in_threadpool(execute_iodine_steady_deck, deck, source=src)
    except (
        CapabilityError,
        ContractError,
        SteadyStateError,
        ValidationError,
        ValueError,
    ) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/impedance")
async def post_iodine_impedance(body: IodineHttpRunRequest) -> dict[str, Any]:
    from solarlab.contracts import ContractError
    from solarlab.equations.dc_1d import CapabilityError
    from solarlab.experiments.iodine_declarative import execute_iodine_impedance_deck
    from solarlab.solvers.nonlinear.steady_1d import SteadyStateError

    try:
        deck, src = _resolve_bounded_http_deck(body)
        return await run_in_threadpool(execute_iodine_impedance_deck, deck, source=src)
    except (
        CapabilityError,
        ContractError,
        SteadyStateError,
        ValidationError,
        ValueError,
    ) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/aging")
async def post_iodine_aging(
    request: Request,
    body: IodineHttpRunRequest,
) -> dict[str, Any]:
    from solarlab.contracts import ContractError
    from solarlab.equations.dc_1d import CapabilityError
    from solarlab.experiments.iodine_aging import IodineAgingError
    from solarlab.experiments.iodine_declarative import execute_iodine_aging_deck
    from solarlab.solvers.nonlinear.steady_1d import SteadyStateError

    store_dir = None
    if body.persist_to_store:
        job_service = request.app.state.job_service
        if job_service is None:
            raise HTTPException(
                status_code=503,
                detail="persisting runs requires an active server job_service writer",
            )
        store_dir = job_service.supervisor.store.root

    try:
        deck, src = _resolve_bounded_http_deck(body)
        return await run_in_threadpool(
            execute_iodine_aging_deck,
            deck,
            source=src,
            store_dir=store_dir,
            run_id=body.run_id,
        )
    except (
        CapabilityError,
        ContractError,
        IodineAgingError,
        SteadyStateError,
        ValidationError,
        ValueError,
    ) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
