# AIMC Integration

This branch is the continuation point for the proposed microscopic aging
model on top of SolarLab's current new-core development baseline. It has no
implemented target VMC or atomistic-to-device coupling engine yet.

## Existing objective

The retained research objective is MAPbI3/contact aging driven by ion or
vacancy motion and interface changes, with a real electronic ground-state
VMC calculation after each accepted position update and feedback to device
observables. A reported GPU allocation is an available resource claim, not
a measured throughput or accuracy result.

The six completed research runs cover literature, feasibility, charge/event
coupling, dimensional requirements and analytic cost checks. They did not
run target VMC, MD/kMC, coupled aging trajectories or target GPU benchmarks.

The canonical research material remains in this checkout's `work/ignis/`
directory. Its checked inventory is recorded in the SolarLab archive under
`test/ArchitectureRefactor/WorkspaceConsolidationV1/RetainedResearchAssets.json`
and `EvidenceManifest.json`. The assessment, literature ledger, papers,
analytic scripts and original run records are research assets outside Git;
they are not additional production capabilities. Preserve and archive them
before any future removal of this research directory.

## Interfaces to implement

| Boundary | Required information | Acceptance concern |
| --- | --- | --- |
| Device to microscopic model | Local field/potential and energy reference, carrier reservoirs, temperature, contact and protocol state | Explicit reference and feedback time |
| Nuclear/defect state to electronic solver | Actual nuclei, species, geometry, electron number/charge sector and model identity | No stale geometry, charge sector or numerical state |
| Electronic results to event model | Comparable energies and uncertainties; additional barrier, prefactor and capture/emission information where required | A ground-state energy alone does not define a physical event rate |
| Event state to device | Position, defect/charge state, spatial weights, inventory and reaction/charge source terms | Conserved mappings and no double counting |
| Coupled result to RunStore | Physical time, method/model/source identity, uncertainty, cost and original failures | Restart and solver-free readback preserve the actual run |

These are requirements for a future implementation, not names of existing
Python classes or completed APIs.

## Development order

1. Choose a specific contact, defect and charge sectors. Freeze the boundary
   and energy-reference assumptions, event definition, accuracy targets and
   budget before selecting an implementation.
2. Qualify the electronic calculation on a fixed small configuration,
   including the intended heavy-element treatment, boundary conditions,
   repeatability and measured cold/warm-start cost.
3. Establish comparable states and validated event rates. kMC is a research
   proposal, not an already adopted replacement for the user's update rule.
4. Implement conservative mapping to a bounded 1D device readout first if
   that dimensional reduction is justified. A local three-dimensional
   electronic model does not by itself qualify a one-dimensional device
   reduction or require a fully three-dimensional device solver.
5. Add feedback, trajectory persistence and recovery, then perform
   refinement and mechanism-specific validation before broadening scope.

Use the existing [development guide](DevelopmentGuide.md) and
`SystemLayout`/`Term`/`Assembler` extension path. Keep the electronic backend
optional and out of ordinary core imports. Do not advance the same ionic
population both in an external event model and in an internal continuum
time step without an explicit coupled formulation.

## Boundaries of current evidence

The contact structure, defect sectors, event model and tolerances remain
unfrozen. The existing legacy frozen-ion snapshot and dynamic-trap contracts
are reference material, not proof that the new core already accepts atomistic
VMC outputs. Full predictive aging and lifetime claims require evidence
beyond a successful electronic calculation or a fitted J-V curve.

Keep shared infrastructure changes on `devel`; land model-specific changes
here in small, tested commits. The first implementation milestone should
have a declared physical input/output contract and a reproducible bounded
case, rather than a nominal end-to-end capability flag.
