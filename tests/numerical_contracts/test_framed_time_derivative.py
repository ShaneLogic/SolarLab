"""Explicit time partials of an affine moving coordinate system."""

from decimal import Decimal, localcontext
from fractions import Fraction
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from scripts.benchmarks.contract_prototype import (
    ContractError, SparseLinearization, SparseStructure,
)
from scripts.benchmarks.coupled_device_prototype import (
    AffineVoltageMap, SegmentAffineFrame, VoltageLiftAdapter,
)
from scripts.benchmarks.precision_prototype import PrimitiveExpansion


def frame(velocity):
    return SegmentAffineFrame("a"*64, "b"*64, 1, "c"*64, "d"*64,
                              0.0, np.zeros(velocity.shape), velocity)


def linearized(matrix, velocity, *, columns=None, rows=None, physical_time=None):
    """Exercise the actual transformation with independent analytic partials.

    This component fixture supplies a known linear physical equation. It does
    not construct a device or claim to validate the adapter's constructor.
    """
    matrix = csc_matrix(matrix, dtype=float)
    n = matrix.shape[0]
    structure = SparseStructure(matrix.shape, matrix.indices, matrix.indptr,
                                "analytic-linear-equation", "analytic-layout")
    mass = structure.filled(np.full(matrix.nnz, 5.0))
    physical = SparseLinearization(matrix, mass, np.zeros((n, 2)),
        np.zeros((n, 2)), np.zeros(n) if physical_time is None else physical_time,
        structure, "analytic-physical-partials")
    mapping = SimpleNamespace(columns=np.ones(n) if columns is None else np.asarray(columns),
        rows=np.ones(n) if rows is None else np.asarray(rows), lift=np.zeros((n, 2)),
        frame=None if velocity is None else frame(velocity))
    context = SimpleNamespace(mapping=mapping, source_identity="analytic-coordinate-map",
        problem=SimpleNamespace(linearize=lambda *_: physical),
        _point_rate=lambda *_: (None, None, np.zeros(n)))
    result = VoltageLiftAdapter.linearize(context, 0.0, np.zeros(n), np.zeros(n),
                                          np.zeros(2), np.zeros(2))
    return result


@pytest.mark.parametrize("velocity,expected", [(None, 3.5), (0.0, 3.5), (0.25, 14.0), (-0.25, -7.0)])
def test_affine_clock_partial(velocity, expected):
    # F=3*x+5*xdot+t/2, x=2*(q0+(t-t0)*V0+z), row scale=7.
    rate = None if velocity is None else PrimitiveExpansion.from_value([velocity])
    result = linearized([[3.0]], rate, columns=[2.0], rows=[7.0], physical_time=[0.5])
    assert result.time[0] == expected
    assert result.y[0, 0] == 42.0
    assert result.ydot[0, 0] == 70.0
    np.testing.assert_array_equal(result.inputs, np.zeros((1, 2)))
    np.testing.assert_array_equal(result.input_rate, np.zeros((1, 2)))


def test_frame_velocity_low_word_survives_tangent_cancellation():
    velocity = PrimitiveExpansion.from_value([1.0, 1.0]).add(
        PrimitiveExpansion.from_value([2.0**-70, 0.0]))
    result = linearized([[1.0, -1.0], [0.0, 1.0]], velocity)
    assert result.time[0] == 2.0**-70
    assert result.time[1] == 1.0


def test_scale_precedes_time_partial_projection():
    smallest = float.fromhex("0x0.0000000000001p-1022")
    result = linearized([[smallest]], PrimitiveExpansion.from_value([0.5]), rows=[2.0])
    assert result.time[0] == smallest


def test_finite_time_cancellation_does_not_evaluate_overflowing_fallback():
    with np.errstate(over="raise", invalid="raise"):
        result = linearized([[-2.5e307]], PrimitiveExpansion.from_value([4.0]),
                            rows=[2.0], physical_time=[1e308])
    assert result.time[0] == 0.0


@pytest.mark.parametrize("coefficient,velocity", [
    (float.fromhex("0x0.0000000000001p-1022"), 0.5), (1e308, 4.0),
])
def test_unrepresentable_time_partial_is_explicit(coefficient, velocity):
    with pytest.raises(ContractError, match="segment_frame_time_partial_unrepresentable"):
        linearized([[coefficient]], PrimitiveExpansion.from_value([velocity]))


def _decimal_fraction(value):
    value = Fraction(value)
    return Decimal(value.numerator)/Decimal(value.denominator)


@pytest.mark.slow
def test_current_device_frame_time_partial_ladder(request):
    """Use the original case/gate inputs and independent Decimal equations.

    REAL_DEVICE_PLAN is required, as for the existing coupled-device suite.
    This is a fixed-state scientific check and never starts a trajectory.
    """
    path = Path(__file__).with_name("test_coupled_device.py")
    spec = importlib.util.spec_from_file_location("framed_time_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    gates = reference.GATES
    records = []
    for case in gates["cases"]:
        for family in gates["points"]:
            model, base, _, q0, inputs = reference.lift_fixture(case, family)
            n = model.layout.size
            velocity = .03*np.sin(np.arange(n))
            if "f" in model.layout.offsets:
                offset = model.layout.offsets["f"]
                velocity[offset.start] = velocity[offset.stop-1] = 0.0
            base = AffineVoltageMap(model, mapped_input_profile="frame-input-expansion12-v1")
            origin = SegmentAffineFrame(base.identity, "a"*64, 1, model.reference.identity,
                                         "b"*64, 0.0, q0, velocity)
            mapping = AffineVoltageMap(model, origin, "frame-input-expansion12-v1")
            adapter = VoltageLiftAdapter(mapping)
            z, zdot = np.zeros(n), .01*np.cos(np.arange(n))
            adot = np.array([.1, .3*model.definition.photon_reference])
            actual = adapter.linearize(0.0, z, zdot, inputs, adot).time
            primitive = mapping.physical_primitive(z, inputs, time=0.0)
            physical_rate = mapping.physical_rate(zdot, adot)
            independent = {}
            for precision in (80, 100):
                with localcontext() as context:
                    context.prec = precision
                    y = reference.affine_exact_input(model, primitive)
                    v = [_decimal_fraction(sum((Fraction(float(w[i])) for w in physical_rate.words), Fraction(0)))
                         for i in range(n)]
                    drift = [_decimal_fraction(Fraction(float(mapping.columns[i]))*sum(
                        (Fraction(float(w[i])) for w in origin.v0.words), Fraction(0))) for i in range(n)]
                    h = Decimal("1e-22")
                    a = list(map(reference.dec, inputs))
                    plus = reference.decimal_kernel(model, [x+h*d for x, d in zip(y, drift)], a, v, precision)["F"]
                    minus = reference.decimal_kernel(model, [x-h*d for x, d in zip(y, drift)], a, v, precision)["F"]
                    independent[precision] = [(p-m)/(2*h) for p, m in zip(plus, minus)]
            comparison = reference.compare(actual/model.Drow, independent[100], model.Drow,
                gates["jacobian_scaled_atol"], gates["jacobian_scaled_rtol"])
            with localcontext() as context:
                context.prec = 100
                difference = [a-b for a, b in zip(independent[80], independent[100])]
            uncertainty = reference.compare(np.zeros(n), difference, model.Drow,
                gates["jacobian_scaled_atol"]*gates["reference_fraction_of_gate"], 0)
            errors = []
            for epsilon in gates["fd_epsilon_ladder"]:
                plus = adapter.residual(epsilon, z, zdot, inputs, adot)
                minus = adapter.residual(-epsilon, z, zdot, inputs, adot)
                difference = (plus-minus)/(2*epsilon)
                errors.append(float(np.max(np.abs(difference-actual))/max(np.max(np.abs(actual)), 1e-30)))
            hits = np.asarray(errors) <= gates["fd_plateau_relative_limit"]
            width = gates["fd_required_adjacent_points"]
            plateau = any(np.all(hits[i:i+width]) for i in range(len(hits)-width+1))
            records.append({"case": case, "point": family, "comparison": comparison,
                "reference_uncertainty": uncertainty, "fd_errors": errors, "fd_plateau": bool(plateau)})
    reference.log_case(request, {"family": "framed_time_partial", "records": records,
        "epsilon_ladder": gates["fd_epsilon_ladder"], "native_calls": 0},
        {"all_independent_partials": all(r["comparison"]["passed"] for r in records),
         "reference_precision_share": all(r["reference_uncertainty"]["passed"] for r in records),
         "all_fd_plateaus": all(r["fd_plateau"] for r in records)})
