# Phase E11 — risk-managed QSS interface-plane solver development workflow

**Target (R1):** replace the bulk-interior cross-carrier interface SRH + NOGEN
clamp with a **physically-correct interface-plane carrier model**, via a
**quasi-steady-state (QSS) algebraic reduction** of the interface-plane state
(`diface_state = 0` solved per-RHS → no stiff extra ODE DOF). This is the single
root cause behind base V<sub>oc</sub> (135 mV heterojunction drop), Nd_ETL direction, and
Nt_C/V_PVK masking.

**Why it's risky:** 7 prior prototypes died here (Newton hang/stall/stiffness).
The interface SRH drives the working 7/10 (CBO 86%, PVK/ETL 74%, J<sub>sc</sub>), so a
naive change regresses them.

## Risk controls (mandatory, every iteration)

1. **Env-gate default-OFF** (`SOLARLAB_IFACE_QSS=1`). Production path bit-identical
   until the final promote. Verify off-path identity every commit.
2. **Golden-master regression** (`scripts/qss_golden_master.py`): a pinned JSON
   baseline of the current 7/10 + J<sub>sc</sub> + base V<sub>oc</sub> + physics-gate booleans.
   `--check` re-runs and diffs vs tolerance. Run before every commit.
3. **Physics gate** (automated): J<sub>sc</sub>≤SQ, V<sub>oc</sub>≤V<sub>bi</sub>, R_interface≥0 OR
   detailed-balance-zero at dark eq, energy conservation, monotone sweep dirs.
4. **Analytical reference limits** (unit tests, written FIRST/TDD):
   - dark equilibrium → R_interface = 0 (machine precision).
   - flat-band interface (ΔE<sub>C</sub>=ΔE<sub>V</sub>=0) → QSS reduces to standard 2-carrier SRH.
   - QSS interface-plane density = bulk × exp(−ΔE/V<sub>T</sub>) (Boltzmann projection).
   - monotone: V<sub>oc</sub> decreases as interface N<sub>t</sub> increases.
5. **Bounded solves**: every `run_transient`/sweep carries `max_nfev` + a wall
   timeout. No unbounded Newton.
6. **Worktree isolation**: develop in `.worktrees/qss/` (gitignored); merge to
   main only after Phase 4 passes.
7. **Known-good fallback**: E1.5 + NOGEN clamp remains the shipping default.

## Phased plan (each phase = a gate + an independent revert point)

### Phase 0 — risk infrastructure (no solver change) ✅ low risk
- Build `scripts/qss_golden_master.py`: capture current baseline JSON; `--check`
  compares. Capture the current 7/10 + J<sub>sc</sub> + base V<sub>oc</sub> as `qss_baseline.json`.
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
  ≥1 of {Nd_ETL, Nt_C/V_PVK, base V<sub>oc</sub>} improved, all physical.

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

- **E11.0 (commit aa4a543)** — golden-master guard + baseline (9 probes). ✅
- **E11.1 (commit 3042cae)** — QSS math validated offline; all 5 analytical
  limits pass; QSS R = 3e-5 × bulk-interior over-count. ✅
- **E11.2 (commit 69bfd51)** — QSS wired into solver (`SOLARLAB_IFACE_QSS=1`,
  default OFF). **Converges, stable at base (6 s, no Newton hang — the
  algebraic-not-ODE de-risk works).** Off-path bit-identical. HTL/PVK flat at
  1.050 for 1e9-1e12 (matches SCAPS, no clamp). ⚠️ partial.
  - **Open (Phase 3):** (a) extreme N<sub>t</sub> (1e15) slow/stalls — stiffness at strong
    SRV; (b) base V<sub>oc</sub> 1.073→1.057 (QSS adds the small HTL/PVK recombination the
    clamp had zeroed); (c) Nd_ETL / Nt_C_PVK not yet validated (batch run timed
    out on the slow point). All three reduce to ONE root cause: the 1-cell φ
    projection under-depletes the interface plane (misses the band-offset / full
    junction bending). Phase-3 fix = full-band-bending (V₁/V₂ partition +
    χ-step) projection + a max_nfev bound so strong-SRV points fail-fast.
- **COUNTERMEASURE VERIFIED** — `qss_golden_master.py --check` (QSS off): all 9
  probes 0.0 mV drift, physics gate passed → **the current best version is
  provably unaffected** by all QSS work. env-gate OFF default + golden master
  guarantee this regardless of QSS outcome.

### E11.3 — Phase 3 gap validation: QSS does NOT solve the 3 gaps (decisive)
Ran the 3 target sweeps under QSS (bulk/doping sweeps are stable; no extreme-SRV
stiffness):
- **Nd_ETL**: 1.090→1.039 (dip @1e16) →1.079 — STILL V-shaped, not monotonic.
  Identical dip to production. The dip is **contact/V<sub>bi</sub> physics at low N<sub>D</sub>**,
  NOT interface sampling → QSS cannot fix it.
- **Nt_C_PVK**: flat 1.050 across 1e9-1e15 (0.1 mV). STILL masked. The interface
  V<sub>oc</sub> ceiling (now 1.050, even lower than production 1.073) is far below the
  bulk-N<sub>t</sub>-sensitive regime (~1.29 V). Unmasking needs a HIGHER ceiling = LESS
  total recombination, which conflicts with the interface SRH the CBO/PVK-ETL
  trends need (the R3 cascade/ceiling wall). QSS lowers the ceiling → WORSE.
- **HTL/PVK**: QSS gives physical interface recombination STRONGER than SCAPS's
  near-inert interface → SolarLab-vs-SCAPS interface-model divergence.

**Hypothesis "all 3 gaps = R1 (interface-plane sampling)" is FALSIFIED.** QSS
correctly fixes interface *sampling* (physical, no clamp, no spurious
generation) but the 3 gaps live elsewhere: Nd_ETL=contact/V<sub>bi</sub>,
Nt_C_PVK=recombination cascade/V<sub>oc</sub> ceiling (R3, base-V<sub>oc</sub>-linked),
HTL/PVK=SCAPS interface-model divergence. QSS also lowers base V<sub>oc</sub> (more
physical, further from SCAPS). **Decision: do NOT promote QSS** (more rigorous
but does not improve SCAPS match; would regress base V<sub>oc</sub>). Keep env-gated as a
documented physically-rigorous alternative interface model.

### Verdict (this session)
QSS Phases 0-2 complete: math-validated, Newton-stable at base, HTL/PVK flat
(promising). It does NOT yet cleanly solve all 3 gaps (base V<sub>oc</sub> regresses,
high-N<sub>t</sub> stiff). Phase 3 (full-band-bending projection + stiffness hardening) is
the remaining multi-day work. The countermeasure is in place and verified, so
this is a safe checkpoint — the current best ships unchanged; QSS stays opt-in
until Phase 3-5 land.
