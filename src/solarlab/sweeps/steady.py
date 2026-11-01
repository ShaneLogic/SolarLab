"""Serial new-core DC sweeps of the existing prepared device declarations.

Every point is resolved and compiled cold. No state, factorization or result
is reused across parameter changes; voltage continuation stays within a point.
Point failures remain explicit and do not discard other completed points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from numpy.typing import ArrayLike

from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.jv.steady import SteadyJVResult, run_steady_jv
from solarlab.sweeps.preparation import PreparedSweep, prepare_point

if TYPE_CHECKING:
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions


@dataclass(frozen=True)
class SteadySweepPoint:
    point_id: str
    coordinates: dict[str, Any]
    status: str
    result: SteadyJVResult | None
    error: str | None


def run_prepared_steady_sweep(
    sweep: PreparedSweep, voltage_V: ArrayLike, *,
    intervals_per_layer: int = 24, generation_scale: float = 1.0,
    incident_power_W_m2: float | None = None, options: SteadyOptions | None = None,
) -> tuple[SteadySweepPoint, ...]:
    """Execute device-only sweep points with the real 1D compiler and solver.

    A PreparedJVExperiment carries an independent timed protocol. It is not
    silently converted to this explicitly steady protocol.
    """
    from solarlab.device.compile_1d import compile_device_1d
    from solarlab.solvers.nonlinear.steady_1d import SteadyStateError

    if not isinstance(sweep, PreparedSweep):
        raise TypeError("expected PreparedSweep")
    if sweep.to_input().base.schema_version != "solarlab.device-preparation.v1":
        raise ValueError("steady sweep requires a device base and explicit steady voltages")
    output = []
    for point in sweep.to_mapping()["points"]:
        result, error = None, None
        try:
            if point["applied_input"] is None or point["status"] == "unresolved":
                raise ValueError(f"unresolved sweep point: {point.get('field_errors', [])}")
            prepared = prepare_point(point["applied_input"], sweep.defaults, sweep.resources, sweep.sources)
            if not isinstance(prepared, PreparedDevice):
                raise ValueError("steady sweep point must resolve to PreparedDevice")
            compiled = compile_device_1d(prepared, intervals_per_layer=intervals_per_layer)
            result = run_steady_jv(compiled, voltage_V, generation_scale=generation_scale,
                                   incident_power_W_m2=incident_power_W_m2, options=options)
        except (ValueError, SteadyStateError) as failure:
            error = f"{type(failure).__name__}: {failure}"
        output.append(SteadySweepPoint(point["id"], point["normalized_coordinates"],
                                        "completed" if result is not None else "failed", result, error))
    return tuple(output)
