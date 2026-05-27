# Phase E3 Sprint 6 — interface-plane state RFC

**Status:** design (no code change yet)
**Branch:** `e3-interface-plane-state` (off main)
**Predecessors:** four failed E2 prototype branches (BBD, thin-shell,
single-sided PV, full PV) — see findings docs.

## Goal

Close the SolarLab ↔ SCAPS parity gap by adding **dedicated interface-
plane state variables** to the solver. Each heterointerface k carries
4 NEW unknowns (n_1s, p_1s, n_2s, p_2s) coupled to bulk grid nodes via
thermionic-emission flux and Pauwels-Vanhoutte interface SRH.

This is the architectural change that four E2 prototype attempts have
shown is structurally necessary. SCAPS does this internally.

## State vector layout

### Current (3 N or 4 N grid-node densities)

```
y = [n[0], n[1], ..., n[N-1],
     p[0], ..., p[N-1],
     P[0], ..., P[N-1]
    {, P_neg[0], ..., P_neg[N-1]}]      # if dual ions
```

### After E3 (adds 4·N_iface_state at end)

```
y = [ ...bulk grid nodes (as above)...,
      n_1s[0], p_1s[0], n_2s[0], p_2s[0],   # interface 0
      n_1s[1], p_1s[1], n_2s[1], p_2s[1],   # interface 1
      ...                                    # one block per active iface
    ]
```

Block at the end so legacy `StateVec.unpack` slicing stays bit-
identical for the bulk portion. Interface block only present when
`MaterialArrays.has_interface_plane_state` is True.

### Convention per block (k-th interface)

For each heterointerface k:
- `n_1s` = electron density at interface plane, ETL (right / "1") side
- `p_1s` = hole density at interface plane, ETL side
- `n_2s` = electron density at interface plane, PVK (left / "2") side
- `p_2s` = hole density at interface plane, PVK side

Subscripts match Pauwels-Vanhoutte 1978 paper (translated to n+p
convention: paper sem1 = heavy-doped = ETL, sem2 = light-doped = PVK).

## Activation gate

Sprint 6: env-var prototype `SOLARLAB_INTERFACE_PLANE_STATE=1`.
Sprint 8 promotion: `SimulationMode.use_interface_plane_state` field.

LEGACY tier: forced off (IonMonger reproducibility).
FAST tier: forced off (cheaper).
FULL tier: ON by default IF stack has heterointerfaces with defects.

## Initial condition (dark equilibrium)

Per interface k, at thermal equilibrium (V_app = 0, no illumination):

```
V_2_eq = partition_2_left[k] * V_bi_eff       # PVK side band-bending
V_1_eq = (1 - partition_2_left[k]) * V_bi_eff  # ETL side band-bending
```

Interface-plane densities by Boltzmann from EQUILIBRIUM bulk (not photo-
injected) densities:

```
n_1s_eq = n_R_eq * exp(-V_1_eq / V_T)
p_1s_eq = p_R_eq * exp(+V_1_eq / V_T)
n_2s_eq = n_L_eq * exp(+V_2_eq / V_T)
p_2s_eq = p_L_eq * exp(-V_2_eq / V_T)
```

where `n_R_eq, p_R_eq, n_L_eq, p_L_eq` are the equilibrium bulk
densities derived from doping config (already cached on `MaterialArrays`).

**Key difference from E2 prototypes:** the interface-plane STATE is
initialised at equilibrium values, then **evolved by the time
integrator**. Under photo-injection / forward bias, the state finds
its own value via TE flux in + SRH out. No exp factor applied at
runtime to bulk densities.

## RHS contributions

Per interface k, the 4 ODEs:

```
dn_1s/dt = + J_n_TE_to_ETL[k] / dx_iface       (TE pumping from ETL bulk)
           - R_s1_with_n_1s[k] / dx_iface     (SRH sink: n_1s + p_2s pair)

dp_1s/dt = + J_p_TE_to_ETL[k] / dx_iface       (TE pumping ETL hole)
           - R_s2_with_p_1s[k] / dx_iface     (SRH sink: n_2s + p_1s pair)

dn_2s/dt = + J_n_TE_to_PVK[k] / dx_iface
           - R_s2_with_n_2s[k] / dx_iface

dp_2s/dt = + J_p_TE_to_PVK[k] / dx_iface
           - R_s1_with_p_2s[k] / dx_iface
```

Where:
- `J_n_TE_to_*` is the TE flux from the bulk side into the interface
  plane (paper eq 14a/b with sign convention).
- `dx_iface` is the effective thickness of the interface plane (small,
  e.g. 1 nm — the SCAPS shell width convention).
- `R_s1`, `R_s2` are paper eqs 12, 13 evaluated on state-vec densities.

### TE flux primitive (NEW `physics/interface_plane.py`)

Implements paper eq 14a/b for each carrier × side combination:

```python
def te_flux_n_etl(n_bulk_R, n_1s, V_T, v_t_etl, delta_E_c):
    """TE electron flux from ETL bulk into interface-plane n_1s state.

    Paper eq 14a (ΔE_c > 0): j_n = (n_R - n_1s) · v_t · exp(-ΔE_c/V_T)
    Paper eq 14b (ΔE_c < 0): j_n = (n_R - n_1s) · v_t
    """
    ...
```

Returns net flux into the interface-plane state. Positive when bulk
> interface (carrier flows into interface plane).

Bulk-side SG flux at the heterointerface is unchanged — it still flows
between bulk nodes, but now feeds the interface plane via the new
TE flux instead of crossing directly to the other layer.

## Interface SRH on state-vec densities

Rewritten `_apply_interface_recombination`:

```python
R_s1 = interface_recombination(n_1s, p_2s, ni_eff_sq, n_1, p_1, v_n, v_p)
R_s2 = interface_recombination(n_2s, p_1s, ni_eff_sq, n_1, p_1, v_n, v_p)
# Apply sinks to the four interface-plane state ODEs:
dn_1s_dt -= R_s1                  # n_1s captured into trap
dp_2s_dt -= R_s1                  # p_2s captured into trap (paired)
dn_2s_dt -= R_s2
dp_1s_dt -= R_s2
# No bulk dn[idx] / dp[idx] sink — the TE flux feeds the rate out of
# bulk; the bulk continuity already accounts for it.
```

This is the SCAPS-faithful coupling: rate consumes interface-plane
state, TE flux refills it from bulk.

## Bulk continuity at heterointerface — modified

Current SG flux at the heterointerface face computes a single flux
between idx and idx+1 (Slotboom with chi step). With interface-plane
state:

```
J_n_face_at_iface = J_n_TE_to_etl_iface   # bulk → interface plane
J_p_face_at_iface = J_p_TE_to_pvk_iface
```

The bulk continuity equation at idx and idx+1 then uses these TE
fluxes as the boundary condition for the heterointerface face, NOT
the SG flux across the chi step.

This is the key Q-Fermi STEP allowance: bulk node idx-1 and the
interface plane share Q-Fermi (within PVK), but interface plane to
bulk node idx+1 has the TE-controlled step (the heart of SCAPS' model).

## Files touched (sprint-by-sprint)

| File | Sprint 6 | Sprint 7 | Sprint 8 |
|---|---|---|---|
| `solver/mol.py` | StateVec extension | `_apply_interface_recombination` rewrite, dispatch | Tier-gate promotion |
| `physics/interface_plane.py` | NEW (~150 LoC) — TE flux primitive | refinement | — |
| `physics/continuity.py` | — | heterointerface face routes through TE | — |
| `models/device.py` | — | — | `DeviceStack.use_interface_plane_state` field |
| `models/mode.py` | — | — | `SimulationMode.use_interface_plane_state` flag |
| `configs/scaps_mirror.yaml` | — | — | `interface_plane_state: true` |
| `backend/main.py:stack_from_dict` | — | — | Round-trip field |
| `tests/integration/` | RED tests (legacy bit-identity + new dims) | full PV behaviour pinning | regression re-run |

## Tests (Sprint 6)

| Test | Pinned behaviour |
|---|---|
| `test_e3_statevec_bit_identical_without_iface_state` | y_eq dimension = 3N when flag off (legacy) |
| `test_e3_statevec_dimension_when_active` | y_eq dimension = 3N + 4*N_iface_state when flag on |
| `test_e3_dark_eq_n_1s_matches_boltzmann` | n_1s_eq computed correctly from n_R_eq + V_1_eq |
| `test_e3_dark_eq_n_2s_matches_boltzmann` | Same for n_2s |
| `test_e3_dark_eq_partition_heavy_doping_limit` | Partition → unity on heavy-doped side |
| `test_e3_voc_moves_with_env_active` | V_oc shifts when env=1 (proves wiring) |
| `test_e3_finite_jv_under_activation` | No NaN/Inf in JV |

## Risk mitigations

| Risk | Mitigation |
|---|---|
| Newton stability with new state vars | Bisection budget (already 10) + BDF fallback (already in place) handle this |
| Initial-condition convergence | Boltzmann projection from cached n_eq, p_eq gives sensible starting point |
| Per-interface state could explode if TE flux + SRH rates mismatch | Defensive clamps on state-vec densities to [0, ni_eff² / floor] |
| LU cache invalidation | Cache rebuild once at build_material_arrays — same as current |
| 4-8 weeks is optimistic | Sprint 6 → Sprint 8 = 6 weeks budget; if Sprint 6 alone takes 4 wks, replan with user |

## Pass criteria (Sprint 7 Day 7-10 gate)

| Criterion | Required | Source |
|---|---|---|
| ETL doping range | ≤ 200 mV | RFC primary |
| CBO closure | ≥ 80 % | RFC primary |
| Base V_oc | ∈ [1.05, 1.25] V | RFC envelope |
| Interface defect N_t | ≥ 200 mV (NOT collapsed) | survives the failure mode that killed single-sided PV |
| All 23 SCAPS-subset tests | PASS | regression |
| Full slow regression (TMM baselines) | PASS | architectural change must not break TMM |
| PVK doping direction | match SCAPS | bonus |

If gate fails on Newton stability: bisection-budget tuning, FAST-tier
fallback, or escalate.

If gate fails on ETL doping: investigate whether TE flux v_t convention
matches SCAPS' "smallest v_th of two neighbouring layers" rule
(Manual §3.8).

## Commit discipline

Each day = atomic commit per superpowers TDD:
- RED test first
- GREEN minimal implementation
- REFACTOR after gate passes
- Conventional commit body with `Constraint:` / `Rejected:` /
  `Confidence:` / `Directive:` trailers (per `.claude/CLAUDE.md`).

Branch `e3-interface-plane-state` stays local until Sprint 7 validation
gate passes. NO push to origin/main before gate.

**Related:** [[project-scaps-validation-parked]], all four E2 gate
findings docs, Pauwels-Vanhoutte 1978 formula extraction
`2026-05-27-e2-pauwels-vanhoutte-formula.md`.
