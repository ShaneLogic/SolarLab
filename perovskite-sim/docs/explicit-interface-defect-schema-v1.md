# Explicit Interface-Defect Input Contract v1

Status: D4-E1 canonical microscopic identity and charged-research wiring. The
standard and SCAPS-shaped config adapters populate the document, and every
charged steady-state research entry point now requires it. Production
interface-charge routes remain parked pending D4-E2 recertification and the
D4-E3 production decision.

## Purpose

Historically, SolarLab reduced each SCAPS interface triplet
`sigma * v_th * N_t` to a surface-recombination velocity and retained only
`E_t`, `N_t_cm2`, and calibration factors on `InterfaceDefect`. That was enough
for recombination-only compatibility, but it could not prove that trap
occupancy, recombination, and sheet charge belonged to the same microscopic
population.

D4-E0 added an immutable `InterfaceDefectDocument` in canonical SI units. The
legacy `DeviceStack.interfaces` tuple remains present, so default solver
arithmetic is unchanged. D4-E1 makes the charged QF dark-reference, backend,
refinement, and stress paths require the document and verify the resolved
compatibility velocity before they can run.

## Canonical SI document

```yaml
schema_version: solarlab-explicit-interface-defects-v1
energy_reference: below_reference_conduction_band
reference_selection: absorber_else_lower_gap
density_normalization: integrated_areal_total
trap_depth_eV: 0.55
total_density_m2: 1.0e17
kinetics:
  sigma_n_m2: 3.0e-24
  sigma_p_m2: 5.0e-24
  thermal_velocity_n_m_s: 1.0e5
  thermal_velocity_p_m_s: 1.0e5
charge_convention: equilibrium_referenced_electron_occupancy
degeneracy: 1.0
```

The v1 document describes one single-energy, energy-integrated interface trap
population. `total_density_m2` is an areal inventory, never a volume density or
an energy peak density. Cross sections may be zero to represent a capture-off
limit; density, thermal velocities, and degeneracy must be positive.

The reference side follows the existing deterministic SolarLab rule: use the
absorber if exactly one adjacent layer is an absorber, otherwise use the
lower-band-gap side. `trap_depth_eV` is measured downward from that side's
conduction band. A later interface-distribution schema must use a new version;
v1 does not reinterpret Gaussian metadata as a single level.

## SCAPS-cgs adapter

Existing flat inputs remain accepted:

```yaml
sigma_n_cm2: 3.0e-20
sigma_p_cm2: 5.0e-20
v_th_cm_s: 1.0e7
N_t_cm2: 1.0e13
E_t_eV_below_cb: 0.55
```

They convert as

```text
sigma[m2] = sigma[cm2] * 1e-4
v_th[m/s] = v_th[cm/s] * 1e-2
N_t[m-2]  = N_t[cm-2] * 1e4
S[m/s]    = sigma[m2] * v_th[m/s] * N_t[m-2]
```

Decimal scaling is applied before binary-float conversion. The backend now
serializes a populated microscopic document back to its true cgs fields. It no
longer invents `v_th=1e7 cm/s` and back-calculates `sigma` when the original
kinetics are available.

The standard flat adapter requires exactly the five physical fields plus the
two optional historical calibration factors. Unknown or missing fields fail
closed. The SCAPS-shaped loader retains its own strict distribution metadata;
only a resolved `single` level below the selected conduction-band reference is
promoted to v1. Gaussian or above-valence-band entries keep their existing
compatibility behaviour but do not claim a v1 microscopic identity.

## Charge convention

The v1 convention is the equilibrium-referenced electron-occupancy increment

```text
Delta sigma = -q N_t (f - f_eq).
```

This increment has the same sign for donor-like and acceptor-like one-electron
traps. The document does not claim an absolute trap charge, fixed countercharge,
or whole-device charge-neutrality convention. Those require a separate schema
and physical reference.

## D4-E1 execution contract

`require_uncalibrated_microscopic_interface_defects()` applies one shared,
fail-closed contract to every electrical interface in a charged research
stack. It requires:

- exactly one v1 canonical document per electrical interface;
- `calibration_factor == iface_state_calibration_factor == 1`;
- `degeneracy == 1`, because the v1 occupancy closure does not consume a
  separate degeneracy term;
- a trap depth inside the selected reference-side band gap; and
- exact identity between the compatibility SRV pair and
  `sigma * v_th * N_t` reconstructed from the document.

The charged QF adapter then rebuilds its compatibility SRV tuple from the
validated documents. `N_t` for sheet charge is taken directly from
`total_density_m2`, never from the duplicate cgs compatibility field. The dark
reference stores each document SHA-256 and capture-velocity pair; its content
hash additionally binds the trap densities, interface transmission, and dark
state. A modified density, SRV, document hash, transmission, grid, stack, or
dark-state array fails before a charged solve.

## Compatibility and execution boundary

- `InterfaceDefect(E_t_eV=...)` remains valid for legacy programmatic and
  recombination-only callers; its `microscopic_document` is `None`.
- Existing flat standard and single-level SCAPS inputs acquire a document while
  preserving the same resolved `DeviceStack.interfaces` velocities.
- Default charge-off material/RHS/J-V paths do not consume the new document.
- A mismatch between duplicated `E_t`/`N_t` compatibility fields and the
  canonical document raises at construction.
- Content identity is the SHA-256 of strict canonical JSON. Unknown nested
  fields are rejected rather than dropped.
- The research backend returns the document hashes and reconstructed capture
  velocities as evidence; the charged refinement executors gate
  `microscopic_defect_contract_verified == 1`.
- D4-E1 does not unlock production interface charge, J-V, transient, AC, 2D,
  tunnelling, multivalent, or metastable execution. D4-E2 must mint complete
  certificates under the v2 protocol/executor identity first.
