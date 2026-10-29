"""Assembler-owned F/J/storage, stable increments and the BDF sign contract."""

import numpy as np
import pytest
from scipy.sparse import csc_matrix
from threadpoolctl import threadpool_limits

from examples.research_1d.steady import load_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.solvers.time.assembled_storage import AssembledStorage1D
from solarlab.solvers.time.backward_euler import BackwardEuler1DStep, storage_mask
from solarlab.solvers.time.bdf import BDFStep, HistoryPoint
from solarlab.solvers.time.bdf_1d import ElectronicStorage1D


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def device():
    return compile_device_1d(load_device(), intervals_per_layer=12)


def bits(reference, candidate):
    assert candidate.shape == reference.shape
    assert candidate.dtype == reference.dtype
    assert candidate.tobytes() == reference.tobytes()


def matrix_bits(reference, candidate):
    reference, candidate = csc_matrix(reference), csc_matrix(candidate)
    reference.sort_indices()
    candidate.sort_indices()
    for field in ("indptr", "indices", "data"):
        bits(getattr(reference, field), getattr(candidate, field))


def coordinates(device, voltage):
    count = device.node_count
    old = np.r_[device.reference_potential_V, np.zeros(2 * count)]
    new = device.rebase_coordinates(old, 0.0, voltage)
    new[count + 1:2 * count - 1] += .001
    new[2 * count + 1:-1] -= .002
    return old, new


@pytest.mark.parametrize("voltage,generation", [(0.0, 0.0), (0.0, 1.0), (.002, 0.0), (.002, 1.0)])
def test_assembled_protocol_values_are_bitwise_electronic_values(device, voltage, generation):
    _, u = coordinates(device, voltage)
    original = ElectronicStorage1D(DC1DProblem(device, voltage, generation))
    assembled = AssembledStorage1D(AssembledDC1DProblem(device, voltage, generation))
    bits(original.residual(u), assembled.residual(u))
    matrix_bits(original.jacobian(u), assembled.jacobian(u))
    bits(original.storage(u), assembled.storage(u))
    matrix_bits(original.storage_tangent(u), assembled.storage_tangent(u))
    matrix_bits(-assembled.storage_tangent(u), assembled.mass_matrix(u))
    bits(original.physical_storage(u), assembled.physical_storage(u))


@pytest.mark.parametrize("voltage", [0.0, .002])
def test_masked_physical_secants_and_assembled_BDF_rows_are_bitwise(device, voltage):
    old_u, new_u = coordinates(device, voltage)
    original_old = ElectronicStorage1D(DC1DProblem(device, 0.0, 0.0))
    original_new = ElectronicStorage1D(DC1DProblem(device, voltage, 1.0))
    assembled_old = AssembledStorage1D(AssembledDC1DProblem(device, 0.0, 0.0))
    assembled_new = AssembledStorage1D(assembled_old.problem.at(voltage, 1.0))
    bits(original_new.physical_increment(new_u, original_old, old_u),
         assembled_new.physical_increment(new_u, assembled_old, old_u))
    history = HistoryPoint(0.0, old_u, assembled_old.storage(old_u), assembled_old)
    reference_history = HistoryPoint(0.0, old_u, original_old.storage(old_u), original_old)
    for h in (1e-7, 2e-7):
        step = BDFStep(assembled_new, (history,), h, 1)
        reference_step = BDFStep(original_new, (reference_history,), h, 1)
        be = BackwardEuler1DStep(original_new.problem, device.state(old_u, voltage_V=0.0), h)
        bits(reference_step.residual(new_u), step.residual(new_u))
        matrix_bits(be.jacobian(new_u), step.jacobian(new_u))
    bits(original_new.storage_increment(new_u, original_old, old_u),
         assembled_new.storage_increment(new_u, assembled_old, old_u))


def test_stable_secant_resolves_motion_below_rounded_inventory_precision(device):
    count = device.node_count
    u = np.r_[device.reference_potential_V, np.zeros(2 * count)]
    new_u = u.copy()
    new_u[count + 1] = 1e-17
    assembled = AssembledStorage1D(AssembledDC1DProblem(device))
    change = assembled.storage_increment(new_u, assembled, u)
    rounded = assembled.storage(new_u) - assembled.storage(u)
    assert change[count + 1] != 0.0 and rounded[count + 1] == 0.0
    mask = storage_mask(device).ravel()
    assert not np.any(change[count:][~mask])


def test_history_must_keep_its_assembled_binding(device):
    assembled = AssembledStorage1D(AssembledDC1DProblem(device))
    u = np.r_[device.reference_potential_V, np.zeros(2 * device.node_count)]
    with pytest.raises(TypeError, match="same adapter"):
        assembled.storage_increment(u, ElectronicStorage1D(DC1DProblem(device)), u)
    different = compile_device_1d(load_device(), intervals_per_layer=4)
    with pytest.raises(ValueError, match="same bound system"):
        assembled.storage_increment(u, AssembledStorage1D(AssembledDC1DProblem(different)), u)
    with pytest.raises(TypeError, match="AssembledDC1DProblem"):
        AssembledStorage1D(DC1DProblem(device))


def test_adapter_never_evaluates_the_legacy_equation_class(device, monkeypatch):
    _, u = coordinates(device, .002)
    assembled = AssembledStorage1D(AssembledDC1DProblem(device, .002, 1.0))
    def forbidden(*args, **kwargs):
        raise AssertionError("assembled StorageProblem must evaluate the assembler")
    for name in ("residual", "jacobian", "storage_jacobian"):
        monkeypatch.setattr(DC1DProblem, name, forbidden)
    assembled.residual(u)
    assembled.jacobian(u)
    assembled.storage(u)
    assembled.storage_tangent(u)
    assembled.mass_matrix(u)
    assembled.storage_increment(u, assembled, u)


def test_public_exchange_mapping_preserves_bound_channel_law_and_refusals(device):
    prepared = load_device()
    document = prepared.to_input().editing_data()
    document["contacts"] = [dict(id="left", side="left", layer="htl", S_n=0.0, S_p=200.0),
                            dict(id="right", side="right", layer="etl", S_n=200.0, S_p=0.0)]
    prepared = resolve_device(DeviceInput.model_validate(document), prepared.defaults,
                              prepared.resources, sources=prepared.sources)
    bound = compile_device_1d(prepared, intervals_per_layer=4)
    original = DC1DProblem(bound, .002, 1.0)
    assembled = AssembledStorage1D(AssembledDC1DProblem(bound, .002, 1.0))
    _, u = coordinates(bound, .002)
    for carrier in ("n", "p"):
        for side, node in (("left", 0), ("right", bound.node_count - 1)):
            channel = bound.channel(carrier, side)
            expected = np.array(original._contact_exchange(u, channel, node))
            bits(expected, np.array(assembled.contact_exchange(u, channel, node)))
            bits(expected, np.array(assembled._contact_exchange(u, channel, node)))
            with pytest.raises(ValueError, match="no bound exchange"):
                assembled.contact_exchange(u, channel, bound.node_count - 1 - node)
    with pytest.raises(ValueError, match="coordinate vector"):
        assembled.contact_exchange(u[:-1], bound.channel("n", "left"), 0)
    fixed = AssembledStorage1D(AssembledDC1DProblem(device))
    fixed_u = np.r_[device.reference_potential_V, np.zeros(2 * device.node_count)]
    with pytest.raises(ValueError, match="no bound exchange"):
        fixed.contact_exchange(fixed_u, device.channel("n", "left"), 0)
