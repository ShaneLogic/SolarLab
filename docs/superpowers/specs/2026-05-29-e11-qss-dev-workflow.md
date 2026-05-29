# Phase E11 — risk-managed QSS interface-plane solver development workflow

**Target (R1):** replace the bulk-interior cross-carrier interface SRH + NOGEN
clamp with a **physically-correct interface-plane carrier model**, via a
**quasi-steady-state (QSS) algebraic reduction** of the interface-plane state
(`diface_state = 0` solved per-RHS → no stiff extra ODE DOF). This is the single
root cause behind base V_oc (135 mV heterojunction drop), Nd_ETL direction, and
Nt_C/V_PVK masking.

**Why it's risky:** 7 prior prototypes died here (Newton hang/stall/stiffness).
The interface SRH drives the working 7/10 (CBO 86%, PVK/ETL 74%, J_sc), so a
naive change regresses them.

## Risk controls (mandatory, every iteration)

1. **Env-gate default-OFF** (`SOLARLAB_IFACE_QSS=1`). Production path bit-identical
   until the final promote. Verify off-path identity every commit.
2. **Golden-master regression** (`scripts/qss_golden_master.py`): a pinned JSON
   baseline of the current 7/10 + J_sc + base V_oc + physics-gate booleans.
   `--check` re-runs and diffs vs tolerance. Run before every commit.
3. **Physics gate** (automated): J_sc≤SQ, V_oc≤V_bi, R_interface≥0 OR
   detailed-balance-zero at dark eq, energy conservation, monotone sweep dirs.
4. **Analytical reference limits** (unit tests, written FIRST/TDD):
   - dark equilibrium → R_interface = 0 (machine precision).
   - flat-band interface (ΔE_C=ΔE_V=0) → QSS reduces to standard 2-carrier SRH.
   - QSS interface-plane density = bulk × exp(−ΔE/V_T) (Boltzmann projection).
   - monotone: V_oc decreases as interface N_t increases.
5. **Bounded solves**: every `run_transient`/sweep carries `max_nfev` + a wall
   timeout. No unbounded Newton.
6. **Worktree isolation**: develop in `.worktrees/qss/` (gitignored); merge to
   main only after Phase 4 passes.
7. **Known-good fallback**: E1.5 + NOGEN clamp remains the shipping default.

## Phased plan (each phase = a gate + an independent revert point)

### Phase 0 — risk infrastructure (no solver change) ✅ low risk
- Build `scripts/qss_golden_master.py`: capture current baseline JSON; `--check`
  compares. Capture the current 7/10 + J_sc + base V_oc as `qss_baseline.json`.
- **Gate:** baseline captured; `--check` passes on unchanged code.

### Phase 1 — QSS math offline (standalone numpy, no solver) ✅ low risk
- Probe: for one interface, solve `te_flux_in + srh_sink = 0` algebraically for
  the interface-plane densities (n_1s,p_1s,n_2s,p_2s) given bulk n,p + offsets.
- **Gate:** reproduces all 4 analytical reference limits (above). Pure math, no
  Newton — catches formulation bugs before any solver risk.

### Phase 2 — single interface, single voltage (env-gated) ⚠️ medium
- Wire QSS for PVK/ETL only (HTL/PVK on legacy path). Apply the QSS interface-
  plane density into the existing algebraic SRH sink (NOT a new ODE DOF).
- **Gate:** base J-V converges (bounded nfev); physics gate; off-path identical;
  PVK/ETL sweep not regressed vs golden master.

### Phase 3 — both interfaces + full sweep ⚠️ medium
- Extend to HTL/PVK. **Gate:** HTL/PVK inert WITHOUT the clamp (matches SCAPS);
  golden-master 7/10 preserved; Nd_ETL / Nt_C_PVK improved; physics gate.

### Phase 4 — full validation
- Golden master + absolute scorecard + figures. **Gate:** ≥7/10 preserved AND
  ≥1 of {Nd_ETL, Nt_C/V_PVK, base V_oc} improved, all physical.

### Phase 5 — promote
- Merge worktree → main; flip default (or keep opt-in per partner call); update
  test guards, figures, report. Commit + push.

## Stop / fallback criteria
- If a phase's gate fails after 3 bounded attempts → revert that phase, keep the
  fallback default, document the failure mode (append to E8 failed-prototype log).
- Never ship a state that fails the physics gate or regresses the golden master.

## Key design choice (de-risks vs the 7 failures)
Prior prototypes added the interface-plane state as **ODE DOF** (stiff/unstable).
QSS makes it **algebraic** (solve `diface_state=0` each RHS for the plane
densities, feed into the existing SRH sink). No new DOF → no added stiffness, no
Jacobian-feedback instability. This is the documented intended path
("Sprint 7 will revisit with QSS algebraic reduction").

## Iteration log
- (E11.0) … see commits.
