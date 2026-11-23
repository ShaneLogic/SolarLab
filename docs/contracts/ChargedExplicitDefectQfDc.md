# Charged Explicit Bulk Defects on the QF/DC Path

Status: DEF-3 public Python execution contract. The opt-in 1D quasi-Fermi DC
path couples monovalent bulk-defect occupancy, recombination, space charge,
contacts, and Poisson from one canonical model. It is not yet a DEF-4
grid/tolerance or SCAPS-reference certificate.

## 1. Activation and compatibility

A charged species still fails closed in the default material builder. The
only enabled device path is the residual-certified quasi-Fermi steady solver:

```python
result = solve_quasi_fermi_steady_state(
    grid,
    stack,
    V_app=0.0,
    illuminated=False,
)
```

The QF entry detects acceptor/donor species and internally requests
`explicit_defect_charge_closure="qf_dc"`. A caller constructing a material
cache directly must pass that same explicit token. The historical
`effective_lifetime` path and the DEF-1 neutral transient path do not activate
this model and retain their prior numerical route.

Charged QF/DC additionally requires:

- `built_in_potential_mode: semiconductor_work_function`;
- Maxwell-Boltzmann carrier statistics and fully ionized dopants;
- single-level, integrated-density species with unit degeneracy;
- finite positive, spatially uniform `Eg`, `Nc300`, and `Nv300` in each
  explicit layer;
- `ni^2 = Nc*Nv*exp(-Eg/V_T)` within the fixed compiler gate;
- no mobile ions, selective contacts, field mobility, photon recycling, or
  dynamic/interface-plane trap state vector.

Ordinary density-form MoL/transient, split-step, impedance, 2D, graded explicit
layers, energy distributions, and dynamic occupancy remain fail closed.

## 2. One closure, three consumers

For every active species and node, `evaluate_monovalent_bulk_defects` computes
one quasi-steady occupancy `f`, then derives both

```text
R_i       = Nt_i * cn_i * cp_i * (n*p - ni^2) / D_i
rho_i     = -q*Nt_i*f_i          acceptor
rho_i     = +q*Nt_i*(1-f_i)      donor
D_i       = cn_i*(n+n1_i) + cp_i*(p+p1_i)
```

The same compiled `MonovalentBulkDefectModel` is consumed by:

1. carrier continuity, through exact `sum_i R_i`;
2. Poisson, through `sum_i rho_i`;
3. contact neutrality/work functions, through the same species inventory and
   charge reference.

The solver does not clip occupancy. A returned certificate requires finite
diagnostics, `0 <= f <= 1`, and a strictly positive kinetic denominator on
every active species/node.

## 3. Contact and Poisson closure

Each semiconductor contact solves

```text
p - n + ND - NA + sum_i(Ncharge_i) = 0
n*p = ni^2
```

with the same monovalent closure used in the interior. The derived carrier
reservoirs and semiconductor work functions define the Poisson boundary drop.
The QF API then requires a `ContactThermodynamicCertificate`; an inconsistent
reservoir/drop pair raises `QuasiFermiSteadyStateError` before Newton starts.

At fixed electron and hole quasi-Fermi levels, the eliminated Poisson solve
uses the analytic diagonal contribution

```text
d rho_total / d phi = -q*(n+p)/V_T + sum_i(d rho_i / d phi)|QF
```

inside its tridiagonal Newton matrix. This tangent is tested against an
independent centered difference. Recombination exposes exact `dR/dn` and
`dR/dp`; the outer QF transport Newton continues to use the established
finite-difference Jacobian. DEF-3 therefore claims analytic local and
eliminated-Poisson tangents, not a wholly analytic outer transport Jacobian.

## 4. Heterojunction composition

Charged bulk defects can coexist with the guarded heterojunction interface
boundary closure. Bulk defect charge and carrier/dopant charge enter the same
Poisson residual on both the ordinary and interface-QSS branches. Interface
transport and any local QSS recombination are solved independently from the
bulk occupancy but share the converged carrier and electrostatic state.

The legacy density-form interface basin predictor cannot evaluate charged
bulk defects and is disabled for this combination. Charged devices instead
use the QF dark/light continuation and must pass the normal residual and
contact gates. This does not enable charged interface defects; their
two-sided sheet-charge model remains a later roadmap checkpoint.

## 5. Public result evidence

Every charged `QuasiFermiSteadyStateResult` carries:

- `bulk_defect_diagnostics.model_identity_sha256`;
- ordered species identifiers and charge transitions;
- per-species active-node mask, occupancy, occupied density, charge,
  recombination, and carrier derivatives;
- total charge, recombination, fixed-QF charge tangent, and extrema;
- `contact_thermodynamic_status` and `contact_fermi_level_span_eV`.

`solve_quasi_fermi_jv_sweep` retains those diagnostics at every voltage point,
and its aggregate `certified` property is true only when every retained point
is certified.

## 6. DEF-3 verification boundary

Focused tests cover acceptor/donor charge signs, contact compensation,
mass-action, multi-species aggregation, document/cache identity, exact
recombination dispatch, fixed-QF Poisson tangent, dark equilibrium, finite
light/bias states, heterojunction coexistence, public J-V propagation, and
fail-closed unsupported routes.

This establishes a 1D monovalent QF/DC execution contract. It does not yet
establish spatial-grid/tolerance convergence, direct SCAPS S0-S2 agreement,
the eleven sweeps in `SolarLabVerifyFormal260702.pdf`, AC/transient defect
response, energy distributions, or charged interface-defect physics. Those
claims require DEF-4 and later checkpoints.

## 7. Checkpoint verification record

The DEF-3 implementation was checked on 2026-08-28 with pinned single-thread
BLAS settings:

```text
charged QF/DC integration
=> 12 passed

focused contact/closure/recombination/QF/impedance/schema group
=> 151 passed, 1 deselected

full Python repository
=> 2881 passed, 2 skipped, 264 deselected, 12 existing warnings
```

Ruff passed in full on the new/clean files and with the repository's critical
selector on legacy `device.py` and `mol.py`; targeted `compileall` and
`git diff --check` passed. The warnings are the pre-existing NumPy `trapz`
deprecations.

For a direct inactive-path comparison, the current tree and the DEF-2 commit
`75a92f5` independently evaluated `configs/cigs_baseline.yaml` on the same
61-node dark-equilibrium grid. The physical RHS arrays were byte-identical and
both produced SHA-256
`fe5c9ff49ee10f47c35fc231e274042833ed22d93d7bf796955d43b67c315844`.
