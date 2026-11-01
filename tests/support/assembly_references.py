"""Live reference factories for later M-class assembly migrations.

These are finite mathematical probes, not accepted steady/transient/AC results.
No nonlinear or frequency solve is called. Steppers retain their actual
physical-inventory secants and expose R_step/J_step beside canonical F/J.
Small-signal cases retain their actual constrained system, drives and masks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np

from examples.research_1d.field_transport import prepare_case as prepare_ct_1d
from examples.research_1d.immobile_ions import prepare_immobile_ion_device
from examples.research_1d.ionic_multilayer import prepare_confined_dual_ion_device, prepare_confined_ion_device
from examples.research_1d.ionic_residual import prepare_single_ion_device
from examples.research_1d.multiple_ionic_domains import DOMAINS, INTERVALS, prepare_multiple_ionic_device
from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
from examples.research_1d.steady import load_device
from examples.research_2d.contacts import contact_device
from examples.research_2d.dc_assembly import spatial_input
from examples.research_2d.field_transport import prepare_case as prepare_ct_2d
from examples.research_2d.grains import spatial_input as grain_input
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_2d import compile_device_2d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.compile_multiple_ionic_1d import compile_multiple_ionic_device_1d
from solarlab.equations.dc_1d import CompiledDevice1D, DC1DProblem
from solarlab.equations.dc_2d import CompiledDevice2D, DC2DProblem
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, Ionic1DProblem
from solarlab.equations.ionic_small_signal import IonicSmallSignal1D
from solarlab.equations.multiple_ionic_1d import CompiledMultipleIonicDevice1D, MultipleIonic1DProblem
from solarlab.equations.multiple_ionic_small_signal import MultipleIonicSmallSignal1D
from solarlab.solvers.nonlinear.ionic_1d import ConstrainedIonic1DProblem
from solarlab.solvers.nonlinear.multiple_ionic_1d import ConstrainedMultipleIonic1DProblem
from solarlab.solvers.time.backward_euler import BackwardEuler1DStep, storage_mask, inventory_m2 as carrier_inventory
from solarlab.solvers.time.ionic_backward_euler import IonicBackwardEuler1DStep, inventory_m2 as ion_inventory
from solarlab.solvers.time.multiple_ionic_backward_euler import MultipleIonicBackwardEuler1DStep
from .assembly_parity import Block, Extraction

ROOT = Path(__file__).resolve().parents[2]
STATE_NAMES = ("equilibrium_reference", "forward_bias", "illuminated", "seeded_perturbation")
Device: TypeAlias = CompiledDevice1D | CompiledDevice2D | CompiledIonicDevice1D | CompiledMultipleIonicDevice1D


def _frozen(value: Any) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Probe:
    name: str
    voltage_V: float
    generation_scale: float
    u: np.ndarray
    udot: np.ndarray
    alpha: float
    previous_u: np.ndarray
    previous_voltage_V: float
    dt_s: float
    frequency_Hz: float
    coordinate_form: str
    ac_perturbation: np.ndarray

    def __post_init__(self) -> None:
        for name in ("u", "udot", "previous_u", "ac_perturbation"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))


@dataclass(frozen=True)
class ReferenceCase:
    name: str
    family: str
    variant: str
    target_class: str
    device: Device
    blocks: tuple[Block, ...]
    states: tuple[Probe, ...]
    fixture_sources: tuple[str, ...]

    def extract(self, state: Probe | str) -> Extraction:
        selected = next(p for p in self.states if p.name == state) if isinstance(state, str) else state
        if not any(selected is probe for probe in self.states):
            raise ValueError("the probe must belong to this independently built reference case")
        return extract(self, selected)


@dataclass(frozen=True)
class ReferenceFactory:
    name: str
    family: str
    variant: str
    target_class: str

    def __call__(self) -> ReferenceCase:
        device, fixtures = _device(self.family, self.variant)
        blocks = _blocks(device)
        return ReferenceCase(self.name, self.family, self.variant, self.target_class,
                             device, blocks, _probes(device, self.family), fixtures)


def _device(family: str, variant: str) -> tuple[Device, tuple[str, ...]]:
    if family == "dc2d":
        ct, contacts, grain = variant in {"ct", "composed"}, variant in {"contacts", "composed"}, variant in {"grain", "composed"}
        prepared = prepare_ct_2d() if ct else load_device()
        if contacts:
            prepared = contact_device(prepared=prepared)
        sources = ["examples/research_1d/steady.py", "examples/research_2d/dc_assembly.py"]
        sources.extend("examples/research_2d/" + name + ".py" for active, name in
                       ((ct, "field_transport"), (contacts, "contacts"), (grain, "grains")) if active)
        return compile_device_2d(prepared, grain_input() if grain else spatial_input()), tuple(sources)
    if family in {"electronic", "be"}:
        prepared = prepare_ct_1d() if variant == "ct" else load_device()
        if variant == "contacts":
            prepared = contact_device(prepared=prepared)
        sources = ["examples/research_1d/steady.py"]
        if variant == "ct":
            sources.append("examples/research_1d/field_transport.py")
        if variant == "contacts":
            sources.append("examples/research_2d/contacts.py")
        return compile_device_1d(prepared, intervals_per_layer=(4, 4, 4)), tuple(sources)
    if family.startswith("multiple"):
        prepared = prepare_multiple_ionic_device(kind="positive" if variant == "immobile" else variant,
                                                  right_immobile=variant == "immobile")
        return compile_multiple_ionic_device_1d(prepared, intervals_per_layer=INTERVALS, ion_domains=DOMAINS), (
            "examples/research_1d/multiple_ionic_domains.py", "examples/research_1d/ionic_multilayer.py")
    sources = ["examples/research_1d/ionic_multilayer.py"]
    if variant in {"single", "immobile_single"}:
        prepared = prepare_single_ion_device(diffusion_only=True) if variant == "single" else prepare_immobile_ion_device("single")
        sources += ["examples/research_1d/ionic_residual.py", "examples/research_1d/immobile_ions.py"]
        return compile_ionic_device_1d(prepared, intervals_per_layer=12), tuple(sources)
    if variant == "confined":
        prepared = prepare_confined_ion_device()
    elif variant == "dual":
        prepared = prepare_confined_dual_ion_device()
    elif variant == "shared":
        prepared = prepare_confined_shared_site_ion_device()
        sources.append("examples/research_1d/shared_site_ionic_fixture.py")
    elif variant == "immobile":
        prepared = prepare_immobile_ion_device("shared_mixed")
        sources += ["examples/research_1d/immobile_ions.py", "examples/research_1d/shared_site_ionic_fixture.py"]
    else:
        raise ValueError("unknown current reference variant: " + variant)
    return compile_ionic_device_1d(prepared, intervals_per_layer=(4, 4, 4),
        ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking")), tuple(sources)


def _blocks(device: Device) -> tuple[Block, ...]:
    n = device.node_count
    blocks = [Block("phi", 0, n), Block("n", n, 2 * n), Block("p", 2 * n, 3 * n)]
    if isinstance(device, CompiledIonicDevice1D):
        blocks.extend(Block(f"ion:{device.domain_layer_id}:{name}", part.start, part.stop)
                      for name, part in zip(device.ion_species_names, device.ion_species_slices))
    elif isinstance(device, CompiledMultipleIonicDevice1D):
        blocks.extend(Block(f"ion:{domain}:{name}", part.start, part.stop)
                      for domain, species in device.ion_species_slices.items() for name, part in species.items())
    return tuple(blocks)


def _coordinates(device: Device, voltage: float) -> np.ndarray:
    if isinstance(device, (CompiledIonicDevice1D, CompiledMultipleIonicDevice1D)):
        return device.initial_state(voltage_V=voltage).u.copy()
    column = device.column if isinstance(device, CompiledDevice2D) else device
    reference = np.r_[column.reference_potential_V, np.zeros(2 * column.node_count)]
    values = column.rebase_coordinates(reference, 0.0, voltage)
    return device.extrude_coordinates(values) if isinstance(device, CompiledDevice2D) else values


def _probes(device: Device, family: str) -> tuple[Probe, ...]:
    previous = _coordinates(device, 0.0)
    size, n = previous.size, device.node_count
    generator = np.random.default_rng(471903)
    direction = np.r_[3e-5 * generator.normal(size=n), .012 * generator.normal(size=size - n)]
    ac = (generator.normal(size=size) + 1j * generator.normal(size=size)) * 1e-4
    result = []
    for index, name in enumerate(STATE_NAMES):
        voltage, generation = (0.0 if index == 0 else .002), (0.0 if index < 2 else 1.0)
        u = _coordinates(device, voltage)
        derivative = np.zeros(size)
        if index == 3:
            u += direction
            derivative = direction * .1
        result.append(Probe(name, voltage, generation, u, derivative, 3.0 if index == 3 else 0.0,
            previous, 0.0, 1e-8 if family == "be" else 1e-3, (0.0, 1e3, 1e4, 37.0)[index],
            "moving_boundary_lift_v1" if index % 2 == 0 else "fixed_reference_v1", ac))
    return tuple(result)


def _problem(device: Device, state: Probe) -> Any:
    cls: Any = (DC2DProblem if isinstance(device, CompiledDevice2D) else
                MultipleIonic1DProblem if isinstance(device, CompiledMultipleIonicDevice1D) else
                Ionic1DProblem if isinstance(device, CompiledIonicDevice1D) else DC1DProblem)
    return cls(device, state.voltage_V, state.generation_scale)


def _physical_storage(problem: Any, u: np.ndarray) -> np.ndarray:
    """Reuse existing physical inventory readers and the actual algebraic mask."""
    device, n = problem.device, problem.device.node_count
    if isinstance(problem, MultipleIonic1DProblem):
        return np.asarray(problem.physical_storage(u))
    state = device.state(u, voltage_V=problem.voltage_V)
    if isinstance(problem, Ionic1DProblem):
        carrier, _ = ion_inventory(device, state)
        component = device.electronic_component
        ions = [ion_inventory(device, state, species=name)[1] for name in device.ion_species_names]
    else:
        carrier, component, ions = carrier_inventory(device, state), device, []
    carrier = np.where(storage_mask(component), carrier, 0.0)
    return np.concatenate((np.zeros(n), carrier.ravel(), *ions))


def _boundary_masks(device: Device, blocks: tuple[Block, ...]) -> dict[str, np.ndarray]:
    size, n = blocks[-1].stop, device.node_count
    potential: np.ndarray
    carrier: np.ndarray
    ion: np.ndarray
    potential, carrier, ion = (np.zeros(size, dtype=bool) for _ in range(3))
    if isinstance(device, CompiledDevice2D):
        potential[device.geometry.contact_nodes] = True
        carrier[n:3 * n] = device.fixed_carrier_mask.ravel()
    else:
        potential[[0, n - 1]] = True
        component = device if isinstance(device, CompiledDevice1D) else device.electronic_component
        for channel in component.channels:
            if channel.kind == "fixed_reservoir":
                offset = n if channel.carrier == "n" else 2 * n
                carrier[offset + (0 if channel.side == "left" else n - 1)] = True
    for block in blocks[3:]:
        ion[[block.start, block.stop - 1]] = True
    return {"potential_fixed": potential, "carrier_fixed": carrier, "ion_boundary": ion}


def _mark_rows(value: Any, mask: np.ndarray) -> None:
    if isinstance(value, Mapping):
        for part in value.values():
            _mark_rows(part, mask)
    elif isinstance(value, (tuple, list)):
        for part in value:
            _mark_rows(part, mask)
    else:
        mask[int(value)] = True


def extract(case: ReferenceCase, state: Probe) -> Extraction:
    problem, device = _problem(case.device, state), case.device
    masks = _boundary_masks(device, case.blocks)
    extra: dict[str, Any] = {}
    storage = mass = tangent = None
    target: Any = problem
    if case.family in {"ionic_constrained", "multiple_constrained"}:
        cls = ConstrainedMultipleIonic1DProblem if case.family == "multiple_constrained" else ConstrainedIonic1DProblem
        target = cls(problem, 1)
        masks["inventory_constraint"] = np.zeros(state.u.size, dtype=bool)
        masks["immobile_profile_constraint"] = np.zeros(state.u.size, dtype=bool)
        _mark_rows(target.constraint_rows, masks["inventory_constraint"])
        _mark_rows(target.profile_constraint_rows, masks["immobile_profile_constraint"])
        masks["inventory_constraint"] &= ~masks["immobile_profile_constraint"]
        F, J = target.residual(state.u), target.jacobian(state.u)
    else:
        F, J = problem.residual(state.u), problem.jacobian(state.u)
        if not isinstance(problem, DC2DProblem):
            storage, mass = _physical_storage(problem, state.u), problem.storage_jacobian(state.u)
            tangent = -mass
            extra["R_implicit"] = problem.residual(state.u, state.udot)
            extra["J_implicit"] = problem.jacobian(state.u, state.udot, alpha=state.alpha)
    if case.family in {"be", "ionic_be", "multiple_be"}:
        cls = {"be": BackwardEuler1DStep, "ionic_be": IonicBackwardEuler1DStep,
               "multiple_be": MultipleIonicBackwardEuler1DStep}[case.family]
        target = cls(problem, device.state(state.previous_u, voltage_V=state.previous_voltage_V), state.dt_s)
        extra.update(R_step=target.residual(state.u), J_step=target.jacobian(state.u))
    if case.family in {"ionic_ac", "multiple_ac"}:
        operator: Any = IonicSmallSignal1D if case.family == "ionic_ac" else MultipleIonicSmallSignal1D
        target = operator(problem, device.state(state.u, voltage_V=state.voltage_V), state.coordinate_form)
        mode = "frozen" if case.family == "ionic_ac" and state.name == "seeded_perturbation" else "mobile"
        system = target.system(state.frequency_Hz, voltage_phasor_V=complex(.01, .002), ion_mode=mode)
        omega = 2 * np.pi * state.frequency_Hz
        matrix = system.jacobian + 1j * omega * system.storage_jacobian
        rhs = system.rhs + 1j * omega * system.rhs_rate
        known: np.ndarray = np.zeros(state.u.size, dtype=complex)
        masks["ac_fixed"] = np.zeros(state.u.size, dtype=bool)
        for row, value in system.fixed_coordinates.items():
            known[row], masks["ac_fixed"][row] = value, True
        masks["ac_original_enforced"] = np.zeros(state.u.size, dtype=bool)
        masks["ac_original_enforced"][list(system.original_rows_enforced)] = True
        masks["ac_inventory_constraint"] = np.zeros(state.u.size, dtype=bool)
        masks["ac_immobile_constraint"] = np.zeros(state.u.size, dtype=bool)
        _mark_rows(system.constraint_rows, masks["ac_inventory_constraint"])
        _mark_rows(getattr(system, "immobile_constraint_rows", {}), masks["ac_immobile_constraint"])
        extra.update(J_ac=system.jacobian, M_ac=system.storage_jacobian, A_ac=matrix,
            R_ac=matrix @ state.ac_perturbation - rhs, rhs_ac=system.rhs, rhs_rate_ac=system.rhs_rate,
            fixed_ac=known, lift_ac=system.coordinate_lift_per_V,
            original_R_ac=target.original_residual(state.ac_perturbation, state.frequency_Hz,
                                                   voltage_phasor_V=complex(.01, .002)))
        # Capture the actual class-owned linearization, not a re-evaluation
        # that could hide a future AC migration error.
        J, mass, tangent = target.jacobian, target.storage_jacobian, -target.storage_jacobian
    module = sys.modules[type(target).__module__]
    if module.__file__ is None:
        raise ValueError("reference classes require actual source files")
    source = Path(module.__file__).resolve()
    if not source.is_relative_to(ROOT / "src"):
        raise ValueError("reference production class was imported outside this checkout")
    provenance = {"factory": case.name, "state": state.name, "compiled_identity": device.identity,
        "target_class": type(target).__module__ + "." + type(target).__name__,
        "target_source": str(source.relative_to(ROOT)), "target_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "state_sha256": hashlib.sha256(state.u.tobytes()).hexdigest(),
        "interpretation": "mathematical reference input; no nonlinear solve or accepted-state claim"}
    provenance.update({"fixture:" + name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                       for name in case.fixture_sources})
    return Extraction(F, J, case.blocks, storage, tangent, mass, extra, masks, provenance)


def _registry() -> Mapping[str, ReferenceFactory]:
    records: list[ReferenceFactory] = []
    groups = (
        ("ionic", "Ionic1DProblem", ("single", "confined", "dual", "shared", "immobile", "immobile_single")),
        ("ionic_constrained", "ConstrainedIonic1DProblem", ("confined", "dual", "shared", "immobile", "immobile_single")),
        ("multiple", "MultipleIonic1DProblem", ("positive", "independent", "shared", "immobile")),
        ("multiple_constrained", "ConstrainedMultipleIonic1DProblem", ("positive", "independent", "shared", "immobile")),
        ("electronic", "DC1DProblem[field_transport]", ("ct",)),
        ("dc2d", "DC2DProblem", ("fixed", "contacts", "ct", "grain", "composed")),
        ("ionic_ac", "IonicSmallSignal1D", ("confined", "shared", "immobile")),
        ("multiple_ac", "MultipleIonicSmallSignal1D", ("positive",)),
        ("be", "BackwardEuler1DStep", ("fixed", "contacts", "ct")),
        ("ionic_be", "IonicBackwardEuler1DStep", ("confined", "dual", "shared", "immobile")),
        ("multiple_be", "MultipleIonicBackwardEuler1DStep", ("positive", "independent", "shared", "immobile")),
    )
    for family, target, variants in groups:
        records.extend(ReferenceFactory(family + ":" + variant, family, variant, target) for variant in variants)
    return MappingProxyType({record.name: record for record in records})


REFERENCE_FACTORIES = _registry()
