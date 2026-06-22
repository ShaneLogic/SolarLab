# Scope: Gummel decoupled-Newton fallback for the steady-state driver

**Date:** 2026-06-22  **Status:** SCOPE + SCAFFOLD BUILT (M1+M2); M3/M4 OPEN
**Owner area:** `perovskite_sim/experiments/steady_state.py`, `physics/continuity.py`, `physics/interface_plane.py`

## 0. Empirical update (2026-06-22) — scaffold built, basic Gummel STALLS

The M1+M2 scaffold is implemented (dormant, default-bit-identical):
- `assemble_rhs(phi_frozen=...)` + `_residual_fn(phi_frozen=...)` — the
  frozen-phi carrier residual seam (`solver/mol.py`, `steady_state.py`).
- `_gummel_point` — the decoupled outer loop (carrier Newton at frozen phi +
  `_qfl_poisson_relax` Poisson half + ln-space under-relaxation). Env-gated
  `SOLARLAB_SS_GUMMEL`; NOT wired into `run_jv_sweep_ss`/`solve_voc_ss` yet.

**Measured on the deep-CBO point (delta E_C = -1.0, where the coupled Newton
goes singular):** the loop reduces the seed residual by ~13 orders
(4e13 -> O(1-1e9)) but then **STALLS at a fixed point above the certification
guard** — res ~3.6 with iface_states, ~2e9 without, and **identical at
carrier_max_it 20 and 40**, i.e. a genuine *decoupled-Gummel stall*, not slow
convergence. **This confirms the Section 9 primary risk: the basic decoupled
scheme stalls at strong coupling.** So M1+M2 alone do NOT close the deep-CBO
gaps. Closing them is the **M4 acceleration** work (the schedule risk):
Anderson acceleration, a coupled-Newton polish that survives the singular
Jacobian, or a transient-assisted Gummel. Until then the deep-CBO SS points
stay documented as gapped (the transient covers that regime; SS ~ transient
there). The `phi_frozen` seam is the reusable, bit-identical foundation M4
builds on.

### M4 result (2026-06-22) — Anderson tried; deep-CBO confirmed INFEASIBLE *and* UNNECESSARY. CLOSED.

M4 was attempted and the leg is now **closed as won't-fix**. Two findings settle it:

**(1) Anderson acceleration does not crack it.** Anderson(m=5) was prototyped on
the Gummel fixed-point map G(z) = [carrier-Newton-at-frozen-phi -> qfl-Poisson-
relax] (Type-II, f-difference least-squares mixing). Measured on four deep
points (delta E_C = -1.0 / -0.5 at V in {0, 0.3, 0.7}, with and without
iface_states): **STALL at res 3.0e9-6.5e9 after 120 iterations** — the same
plateau as plain Gummel. The carrier half alone reduces 5 orders (1.30e12 ->
4.76e7 at it0) but the Poisson re-coupling pushes it back up and the outer map
has a contraction factor >= 1 at this coupling strength; the Anderson
extrapolation cannot fix a map whose fixed-point iteration diverges.

**(2) Gummel is strictly WORSE than the shipped coupled Newton here, and the
shipped driver already fails FAST.** Best peak-density-relative residual
(tol = 1e-6) at delta E_C = -1.0, V = 0.3:

| solver | best residual | wall |
|---|---|---|
| coupled SS Newton (shipped `solve_steady_state`) | **4.2e4** | raises in **0.6 s** |
| decoupled Gummel (M1+M2 scaffold) | 1e9 | stall |
| Anderson-accelerated Gummel (M4) | 3e9 | stall |

All three are 10-15 orders above tol. The coupled Newton gets *closest* (4.2e4)
and, critically, **raises a clean `SteadyStateError` in 0.6 s** ("line search
stalled ... transient assists exhausted") — it does **not** hang. So the deep-CBO
SS root is genuinely pathological for *any* algebraic Newton (coupled or
decoupled); the stiff modes only damp under a **long** transient (~8 s), which is
exactly why the bounded SS transient-assist exhausts. A Gummel fallback adds a
worse-conditioned path, not a better one.

**Why this is also UNNECESSARY.** CBO is a band-offset effect, not an
interface-limited one — **SS approx transient** over the whole CBO sweep (verified
in the flat-band region where both converge). The deep-CBO points therefore add
**zero** scientific value as SS points: the transient curves already cover that
regime in every figure, and the full-FOM figures correctly carry transient-only
there (the shipped driver's fast 0.6 s raise means the figure pipeline skips the
point cleanly, no grind).

**Disposition.** M4 is closed won't-fix. The `_gummel_point` scaffold and the
`phi_frozen` seam stay in-tree, **dormant + bit-identical** (env-gated
`SOLARLAB_SS_GUMMEL`, not wired into any sweep), as the documented foundation if a
*genuinely new* approach appears (pseudo-transient continuation with an unbounded
dt ramp == the transient, so PTC reduces to "just run the transient"). The
shipped solution for the deep tail is the existing transient driver, which is
correct and fast. The net deliverable of this campaign is the **interface-channel
calibration + Newton/LU reuse + `stop_after_voc`** (the operative regime — fast,
accurate, complete on all 10 flat-band-operative sweeps); the deep-CBO tail is a
documented engine boundary, not a gap to be closed with more solver machinery.

## 1. Motivation

The steady-state interface-states driver (`run_jv_sweep_ss(iface_states=True)`,
`solve_voc_ss`) is the configuration that recovers the SCAPS interface-doping
directions (Nd_ETL rising, etc.). It has one structural weakness: the coupled
damped-Newton (`solve_steady_state`) **fails to converge in collapsed-junction
and extreme-band-offset regimes**, where it falls back to the certified
transient point-fallback, which then grinds for minutes per point or fails.

Two concrete symptoms motivated this scope:

1. **Deep-CBO SS gaps (the trigger).** On the ETL/PVK conduction-band-offset
   sweep, the SS series converges only over the flat-band region
   (delta E_C >= -0.16 eV). The deep-offset points (delta E_C <= -0.2 eV,
   where V_oc collapses to ~0.33-0.83 V) **do not converge in SS** even at a
   300 s cap — yet the **transient solves them in ~8 s** and finds a real
   J=0 crossing. So a crossing *exists*; the SS solver simply cannot reach it.
2. **The `Nd_ETL <= 1e10` xfail** (`tests/unit/experiments/test_steady_state.py`
   near-insulating gate) — see the critical caveat in Section 7; this is a
   *different* problem and is **not** fully solved by Gummel.

The de-spike + interface-channel calibration work (2026-06) made the operative
regime fast and accurate; this scope is about the *hard tail* the coupled
Newton can't reach.

## 2. Root cause

Every `assemble_rhs` call solves Poisson globally
(`solver/mol.py:1693`), so `phi[i]` depends on `rho` at all nodes, and the
Scharfetter-Gummel carrier fluxes depend on `phi`. The carrier Jacobian
therefore acquires a **dense, phi-mediated tail** (measured: ~13.5 % of the
Jacobian Frobenius weight outside a 2-node band). In the near-insulating /
collapsed-junction regime the dielectric-relaxation mode makes that coupling
extreme, and the dense FD Newton goes **singular** (the existing ridge fallback
+ transient assists are the symptom-management). The full-thermal-velocity
interface-plane-state block (`iface_state_v_th = 1e5`) adds a second stiff
mode; today both share **one global dense Jacobian**, so they compound.

Gummel decoupling breaks the dense phi-coupling: solve the carrier-transport
equations at *frozen* phi (block-tridiagonal, well-conditioned), then relax phi
at *frozen* quasi-Fermi levels (the analytic, unconditionally well-posed
dielectric step), and iterate. The two stiff modes are then handled by the
operator each is well-conditioned for, instead of one ill-conditioned Newton.

## 3. The design

A Gummel **point solver** that replaces a single failed `solve_steady_state`
point — used as an *opt-in fallback before* the transient fallback:

```
y = ensure_iface_block(y_seed, mat)
y = _qfl_poisson_relax(x, mat, y, V_app)        # Poisson half (EXISTS)
for it in range(max_outer):
    # --- carrier half: continuity at FROZEN phi ---
    phi = solve_poisson_once(y)                  # one solve, then held fixed
    y = carrier_solve_frozen_phi(x, stack, mat, y, V_app, phi)   # NEW (~80-150 lines)
    y = underrelax_ln(y_old, y, omega)           # ln-space, positivity-free
    # --- interface-plane states: local per-interface subsystems ---
    if mat.N_iface_state: y = solve_iface_states_local(x, mat, y, phi)   # NEW (reuse solve_plane_densities)
    # --- Poisson half (EXISTS) ---
    y = _qfl_poisson_relax(x, mat, y, V_app)
    # --- convergence on the SAME coupled residual solve_steady_state uses ---
    res = peak_density_scaled_residual(assemble_rhs(0, y, ...))
    if res < tol: break
    if res rose: omega = max(0.1, 0.5*omega)     # adaptive damping (load-bearing)
if res > res_guard: raise SteadyStateError(...)   # fail-loud, same contract
return y, res
```

Three pieces, two of which already exist:

| Piece | Status | Source |
|---|---|---|
| Poisson half (nonlinear Poisson at frozen QFL, banded, damped) | **EXISTS** | `_qfl_poisson_relax` `steady_state.py:451-512` |
| Carrier half (continuity at frozen phi, block-tridiagonal banded solve) | **NEW** (~80-150 lines) | mirrors the Poisson half; `carrier_continuity_rhs` is already phi-pure |
| Interface-plane states as local per-interface subsystems | **NEW** (small) | template `solve_plane_densities` `interface_plane.py:385-433` |

## 4. Why it is tractable (verified from source)

- **`carrier_continuity_rhs` is already phi-pure** — it takes `phi` as an
  argument and never solves Poisson (`continuity.py:101`, reads phi only at
  `132-139`). A frozen-phi carrier residual is a direct call (zero edits) or a
  `phi=` kwarg on `assemble_rhs` (~5-10 lines wrapping `mol.py:1670-1695`).
- **The carrier Jacobian at frozen phi is block-tridiagonal** — SG flux is
  nearest-neighbour (`fe_operators.py:58-68`), divergence couples node i to
  i+-1 only (`continuity.py:244`), recombination is node-local
  (`recombination.py:47`). So it solves with `scipy.linalg.solve_banded`,
  exactly as the Poisson half already does (`steady_state.py:496`).
- **The interface-plane state block is block-diagonal per interface** — each
  contributor writes only into its own `base=4k` slot
  (`interface_plane.py:213,473,577`; `mol.py:1885`), no inter-interface
  coupling. With phi + carriers frozen, each interface is a closed 4-unknown
  system solvable with the existing damped log-Newton
  (`solve_plane_densities`, `interface_plane.py:385-433`). The only bulk
  back-coupling is the eval-node drain (`mol.py:1907-1910`), folded as a lagged
  source into the next carrier sweep. (Today this block lives in the *global*
  dense Jacobian, `steady_state.py:298,344-353` — decoupling it is itself a win.)

## 5. Implementation plan (milestones)

| M | Deliverable | Est. |
|---|---|---|
| **M1** | Frozen-phi carrier-continuity residual + banded carrier Jacobian (`carrier_solve_frozen_phi`); unit test vs `assemble_rhs` carrier rows at a converged state (must match). | 1 d |
| **M2** | Gummel outer loop (`_gummel_point_fallback`) alternating M1 + `_qfl_poisson_relax`, adaptive ln-space under-relaxation, certify-or-raise on the peak-density residual. Wire as opt-in fallback **before** `_transient_point_fallback` at the two `except SteadyStateError` sites (`run_jv_sweep_ss:602-610`, `solve_voc_ss:657-661`); gate `SOLARLAB_SS_GUMMEL` (default OFF) + `gummel_fallback=False` kwarg. | 1 d |
| **M3** | Interface-plane states as local per-interface subsystems inside the loop (reuse `solve_plane_densities`); validate on the deep-CBO points that currently gap (delta E_C <= -0.2). | 1-1.5 d |
| **M4** | Deep-regime convergence hardening (damping schedule, omega floor, semi-implicit bulk drain) + the test surface in Section 8 + docs. | 1.5-2 d |

**Core deliverable (M1-M4): converge the deep-CBO / collapsed-but-bracketed SS
points -> ~4-6 days.** This fills the CBO SS gaps (the user's ask).

## 6. Integration & safety (bit-identical default)

- The Gummel helper is reached **only after the coupled Newton already raised
  `SteadyStateError`** — every point that converges today never enters it. Even
  with the flag on, converging points stay **bit-identical**; only the
  currently-*failing* regime changes behaviour. Zero edits to the Newton loop.
- Default OFF via the established opt-in pattern (cf. `SOLARLAB_SS_JAC_REUSE`,
  `te_softness`, `iface_states`). On Gummel failure, **fall through** to
  `_transient_point_fallback` so existing behaviour is strictly preserved.
- Fail-loud contract kept: certify on the same peak-density residual scale
  (`steady_state.py:546-553`), raise on failure — no silent fallback.

## 7. CRITICAL caveat — Gummel does NOT flip the `Nd_ETL <= 1e10` xfail

The near-insulating xfail (`test_steady_state.py` low-doping gate) is **a
physics/boundary-condition problem, not a convergence problem**. The voltage
walk already *completes* there and finds **no J=0 crossing below 1.6 V** — both
the SS driver and the transient flat-band result agree there is no crossing
(the model's crossing sits near 1.29 V, above the detailed-balance ceiling).
SCAPS reports V_oc ~ 1.10 V there only via its near-insulating **contact
convention** (flat-band finite-S contact), which SolarLab does not apply by
default.

Therefore:
- Gummel gives **convergence** in the near-insulating regime (kills the
  dielectric-relaxation stiffness the coupled Newton chokes on) — necessary but
  **not sufficient**.
- Flipping the xfail to xpass *additionally* requires a **near-insulating
  contact boundary condition** (the existing `flat_band_contacts` path is the
  candidate seam) so that a physical crossing exists to be found.
- That combined effort is **~10-18 days with a go/no-go** after the first
  crossing attempt, and is **out of scope** for the core deliverable. The CBO
  gaps (Section 1.1) do *not* depend on it — they have a real crossing already.

## 8. Acceptance criteria / test surface

Core (M1-M4):
1. **Bit-identical default** — full unit + the SS parity suite
   (`test_ss_jv_matches_frozen_ion_transient`, `test_direct_voc_consistent_with_jv`)
   unchanged with the flag OFF, and unchanged for every point that converges
   today with the flag ON.
2. **M1 correctness** — `carrier_solve_frozen_phi` reproduces `assemble_rhs`
   carrier rows at a converged state to tolerance.
3. **Deep-CBO convergence (the headline)** — `run_jv_sweep_ss(gummel_fallback=True)`
   converges the delta E_C <= -0.2 eV CBO points that currently gap, with
   FOM (V_oc/J_sc/FF/PCE) physical and the SS curve overlaying the transient
   there (CBO is band-offset-limited).
4. **Fail-loud preserved** — a genuinely unsolvable point still raises
   `SteadyStateError`.

Stretch (separate effort, Section 7):
5. `Nd_ETL <= 1e10` flips to xpass *only* in combination with the contact-BC
   change — tracked as its own task with a go/no-go.

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Gummel is only linearly convergent; stalls/oscillates at high injection (Voc knee, large n*p) | Adaptive ln-space under-relaxation (omega start 1, halve on residual rise, floor ~0.1); fall through to the transient fallback on stall. |
| n,p form a 2x2 node block (recombination cross-term) | Iterate the node-local n-p coupling, or use a 2-field banded solve. |
| Interface bulk-drain (v_th=1e5) is large + lagged -> oscillation | Semi-implicit / under-relaxed drain; keep iface-states local-solve inside the outer loop. |
| Frozen-phi supply exponential clamped at 30 -> deep CBO can sit on the clamp and lose sensitivity | Re-project phi each outer sweep (not once); validate CBO sensitivity is retained. |
| Partial (non-converged) Gummel sweeps would shift validated FOM | Certify-or-raise on the coupled residual; never return a non-converged point. |
| OneDrive sync can overwrite an edited solver file before `git add` | Stage promptly (known repo gotcha). |

## 10. Scope boundaries

- **In:** a Gummel point-solver fallback that converges collapsed-junction /
  deep-offset SS points where the coupled Newton fails (fills the CBO SS gaps);
  opt-in, default-off, bit-identical; interface-states handled as local
  subsystems.
- **Out:** the near-insulating contact-convention / `Nd_ETL <= 1e10` physics
  fix (separate task, Section 7); making Gummel the *default* SS path
  (it is a fallback only); any change to the coupled-Newton hot loop.
