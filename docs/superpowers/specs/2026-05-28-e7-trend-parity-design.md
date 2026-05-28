# Phase E7 — Trend-parity closure design

**Status:** spec superseded by Day 1 spike findings (2026-05-28). See
[Phase E7 spike report](2026-05-28-e7-spike-report.md) for revised scope.
**Date:** 2026-05-28
**Branch (proposed):** `e7-trend-parity`
**Predecessor:** Phase E6.5 (V_max probe, 2026-05-28)
**Decision gate prior:** `docs/superpowers/specs/2026-05-28-e6-decision-gate.md`

## Post-spike scope (2026-05-28)

After Day 1 spike Probes A / B / C, the original three-phase plan
collapsed to a one-phase ship:

- **Y3 dropped** — Probe A confirmed PVK doping direction matches SCAPS
  under v2 automatically (no code needed).
- **Y1 simplified** — Probe B showed multi-defect collapse is exact;
  Y1 = YAML-only PVK/ETL SRV tune (~0.5 day) rather than multi-defect
  solver hook (2-3 days).
- **Y2 parked** — Probe C showed Robin BC family cannot break the
  bulk-limited V_oc ceiling on Nd_ETL high-N_d regime. Root cause is
  interface SRH formulation, not contact BC; closure requires either
  the parked SG-face-density refactor (multi-week, falsified prototype
  family) or partner SCAPS contact-model spec (unavailable). User chose
  to park Nd_ETL gap and ship Y1 only.

The sections below are kept as-is for historical context, but **only the
Y1 Branch B1 (YAML SRV tune) path is being executed**. Y2 architectural
hypotheses, Y3 PVK doping ladder, and the multi-defect solver hook
discussion are now archived.

## Context

Phase E6.4 falsified the parked Newton-Krylov / SG-face-density diagnosis
and re-baselined SCAPS parity against the corrected `scaps_mirror_v2.yaml`
4-defect inventory. Three sweep gaps remain:

| Sweep | Pre-E7 closure | Issue |
|---|---|---|
| Nd_ETL (ETL doping) | 30% UNDER (working regime) + unphysical 2.1 V branch at low Nd | contact BC pinning |
| Nt_C_PVK (PVK bulk N_t) | 0.2% | PVK/ETL interface SRH 250× dominates bulk SRH; multi-defect loader collapse may mask |
| Na_PVK (PVK doping) | direction unknown under v2 (was reversed under v1) | Probe-dependent |

User clarification (2026-05-28): parity bar is **trend fidelity** (sweep
direction + relative magnitude), not absolute V_oc / J_sc / FF / PCE.
Base J-V absolutes within ±10% envelope are acceptable.

## Goals

1. Close `Nd_ETL` trend closure to ≥ 70% AND eliminate the V_oc > E_g/q unphysical branch.
2. Close `Nt_C_PVK` trend closure to ≥ 50%.
3. Match `Na_PVK` direction (V_oc derivative sign matches SCAPS) + closure ≥ 50%.
4. Preserve current CHI_ETL (CBO, 83%) and Nt_PVK_ETL (interface, 109%) closures within ±5 pp.

## Non-goals

- Base V_oc absolute < 50 mV (per E1.14, needs SCAPS source bisection).
- J_sc TMM Fresnel +27% gap (optics model, not parameter parity).
- CBO spike-side plateau 150 mV residual (direction already passes).
- Newton-Krylov / QSS / SG-face-density refactor (falsified by E6.4).
- Boltzmann-degenerate Fermi-Dirac stats (multi-week; only as Y3 hypothesis test, not implementation).
- 2D / tandem stack SCAPS validation.

## Architecture overview

Three architectural levers map one-to-one onto the three trend gaps.
Day 1 diagnostic spike confirms root cause for each before any commit.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Day 1 — Diagnostic Spike (no commits)                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │  A: PVK     │    │  B: bulk    │    │  C: Robin BC dry-run    │  │
│  │  doping v2  │    │  SRH solver │    │  with v2 + Φ_b-derived  │  │
│  │  re-test    │    │  probe      │    │  S_n / S_p              │  │
│  └──────┬──────┘    └──────┬──────┘    └────────────┬────────────┘  │
└─────────┼──────────────────┼─────────────────────────┼──────────────┘
          │                  │                         │
          ▼                  ▼                         ▼
   ┌────────────┐     ┌────────────┐         ┌─────────────────┐
   │ Phase Y3   │     │ Phase Y1   │         │ Phase Y2        │
   │ (cond.)    │     │ bulk SRH   │         │ contact BC      │
   │ PVK doping │     │ multi-     │         │ Dirichlet →     │
   │ direction  │     │ defect hook│         │ Robin / Φ_b     │
   │ fix        │     │            │         │ + two-preset    │
   │            │     │            │         │ split fallback  │
   └────────────┘     └────────────┘         └─────────────────┘
```

**Sequencing**: Y1 (cheapest, lowest risk) → Y2 (highest impact, highest
risk, has fallback) → Y3 (conditional, may be auto-resolved by Y1+Y2).

## Day 1 — Diagnostic spike

No code commits. Three probes run in parallel. Output:
`docs/superpowers/specs/2026-05-28-e7-spike-report.md`.

### Probe A — PVK doping under v2 (~30 min)

**Question**: Does v2 (4-defect inventory + Gaussian PVK/ETL) flip the
PVK doping V_oc direction to match SCAPS?

**Method**:
```bash
python scripts/run_scaps_v2_regression.py --sheets Na_PVK --v-max 1.6
```

**Exit criteria**:
- Direction matches SCAPS → Y3 skipped, v2 closes the gap automatically.
- Direction still reversed → full Y3 scoped.
- Both bracket-fail → widen `--v-max` first.

### Probe B — bulk SRH multi-defect collapse audit (~2 hr)

**Question**: Does the E6.3 loader's parallel-SRH collapse (loader.py
`_combine_bulk_defects`) correctly model SCAPS' parallel defect
summation, or does inverse-τ-weighted n1/p1 smearing mask the bulk
N_t sweep?

**Method** (no code change):
1. Load `scaps_mirror_v2.yaml`, dump per-defect (σ, v_th, N_t, n1, p1) for
   bulk_defects[0] (PVK-CB) + bulk_defects[1] (PVK-VB).
2. Compute true two-defect R_SRH(n, p) at three (n, p) sample points
   (low / mid / high injection) by summing per-defect R_i.
3. Compute collapsed R_SRH(n, p) using `_combine_bulk_defects` outputs.
4. Ratio R_true / R_collapsed at each sample point.

**Exit criteria**:
- Ratio ∈ [0.9, 1.1] across injection range → collapse fine. Y1 = YAML-only PVK/ETL SRV tune.
- Ratio ≥ 2× divergence anywhere → collapse wrong. Y1 = true multi-defect solver hook.
- Mixed → both levers in Y1.

### Probe C — Robin contact dry-run (~2 hr)

**Question**: Does activating Robin BC on `scaps_mirror_v2.yaml` kill the
Nd_ETL low-Nd 2.1 V unphysical branch without regressing CBO / interface
closures?

**Method**:
1. Derive `S_n_right`, `S_p_right` (ETL contact), `S_n_left`, `S_p_left`
   (HTL contact) from SCAPS Φ_b values in partner xlsx:
   `S_majority = v_th`, `S_minority = v_th · exp(−Φ_b / V_T)`.
2. Write `configs/scaps_mirror_v2_robin.yaml` mirroring v2 with S values
   populated. Loader path: `scaps_compat/loader.py` (already parses S
   per Phase E1.10).
3. Run regression:
   ```bash
   python scripts/run_scaps_v2_regression.py \
       --config configs/scaps_mirror_v2_robin.yaml \
       --out-dir outputs/scaps_e7_probe_c
   ```

**Exit decision matrix**:

| Outcome | Action |
|---|---|
| All 4 sweeps ≥ E6.4 closure; Nd_ETL low-Nd points physical | Y2 = Robin as default. No two-preset split. |
| Nd_ETL improves; CBO or interface regresses ≥ 5 pp | Y2 = two-preset split. |
| Nd_ETL 2.1 V branch survives | Y2 = re-scope (Φ_b derivation or deeper contact model). |

### Spike exit gate

End of Day 1: probe reports committed (data files only, no source
changes). Decision matrix locks Y1/Y2/Y3 scope for Days 2+.

## Phase Y1 — Bulk SRH multi-defect hook

**Activation**: only if Probe B shows collapse ratio ≥ 2×. Otherwise
Y1 = YAML-only tune.

**Estimated time**: 2-3 days code + 1 day tests.

### Branch B1 — collapse fine (probe ratio ≈ 1)

YAML-only. Lower `interfaces[PVK/ETL].calibration_factor` from `1.0e-4`
until PVK/ETL areal SRH drops below bulk SRH in working regime.
Iteratively tune until Nt_C_PVK V_oc range matches SCAPS 39 mV ± 50%.

**Mitigation for CBO/interface regression risk**: per-sweep
`calibration_factor` override map in `scripts/run_scaps_v2_regression.py`
(script-level, not solver-level).

### Branch B2 — collapse wrong (probe ratio ≥ 2×)

True multi-defect SRH solver hook:

1. `physics/recombination.py` — add `srh_recombination_multi(n, p, ni_sq, defects)`
   summing per-defect R_i. Keep scalar `srh_recombination()` unchanged.
2. `models/device.py` — add `DeviceStack.bulk_defects_list: tuple[SRHDefect, ...] | None`
   alongside existing scalar `tau_n`, `tau_p`, `n1`, `p1`.
3. `physics/continuity.py` — switch SRH evaluation site: if
   `mat.bulk_defects_list` non-empty → call multi-defect; else fall back
   to scalar (preserves v1 single-defect behaviour).
4. `scaps_compat/loader.py` — populate `bulk_defects_list` from plural
   v2 `bulk_defects:` block (still computes scalar fallback for
   backwards compat).

**Risks**:
- Performance: N=2 defects → ≤10% RHS-step slowdown. Acceptable.
- Newton stability: per-defect (n1, p1) keeps R continuous in (n, p);
  Jacobian well-defined.
- v1 single-defect configs: untouched by the fallback path.

### Branch B3 — mixed

Apply B2 (architectural) first, B1 (parameter tune) second.

### Exit criteria for Y1

- `Nt_C_PVK` closure: 0.2% → ≥ 50% (≥ 20 mV V_oc range vs SCAPS 39 mV).
- CBO + interface_N_t closures stay within ±5 pp of E6.4 baseline.
- All existing SCAPS-subset tests still pass (142+).

### New tests for Y1

- `tests/unit/physics/test_srh_recombination_multi.py` (4-6 tests):
  per-defect summation vs scalar collapse divergence cases.
- `tests/integration/test_e7_y1_bulk_nt_closure.py` (1 test): Nt_C_PVK
  closure floor.

## Phase Y2 — Contact BC overhaul

**Activation**: Probe C confirms Robin kills the 2.1 V branch. If not,
re-scope (out of this spec).

**Estimated time**: 2-4 days.

### Branch C1 — clean Robin win

YAML-only change to `configs/scaps_mirror_v2.yaml`:
```yaml
layers:
  - name: HTL
    s_n_left: <derived>
    s_p_left: <derived>
  - name: ETL
    s_n_right: <derived>
    s_p_right: <derived>
```

S derivation, with per-carrier-per-side mapping spelled out:

| Side / contact | Majority carrier | S value | Minority carrier | S value |
|---|---|---|---|---|
| HTL (left, p-type) | holes | `S_p_left = v_th` | electrons | `S_n_left = v_th · exp(−Φ_b_HTL / V_T)` |
| ETL (right, n-type) | electrons | `S_n_right = v_th` | holes | `S_p_right = v_th · exp(−Φ_b_ETL / V_T)` |

Formula rationale: Schottky-Bethe — majority carrier sees no barrier so
emission velocity is `v_th`; minority sees Φ_b suppression.

Φ_b_HTL and Φ_b_ETL from SCAPS partner xlsx contact sheet. Both ETL/HTL
are ohmic in SCAPS PDF (Φ_b ≈ 0.1–0.3 eV); exact values pulled from xlsx
during Probe C.

**Why E1.11 failed under v1, may work under v2**: v1 had σ=1e-15 + cf=1e-4
4-order σ error. Interface SRH magnitude was wrong; Robin × mis-calibrated
interface produced unpredictable branches. v2 fixes σ=1e-19 + Gaussian
first → Robin retry on physically correct base.

### Branch C2 — two-preset split

Two YAMLs:
- `scaps_mirror_v2.yaml` — Dirichlet (current). Used for CBO + interface_N_t sweeps.
- `scaps_mirror_v2_robin.yaml` — Robin. Used for Nd_ETL + Na_PVK sweeps.

Script change: `scripts/run_scaps_v2_regression.py` gets
`_CONFIG_PER_SHEET` override map.

**Maintenance burden**: two configs drift over time. For now pure
side-by-side (≈ 80-line duplication). Add YAML `extends:` support only
if configs diverge further.

### Branch C3 — Robin insufficient

Re-scope Y2. Park Nd_ETL gap as `requires deeper contact-model rework`.
Continue with Y1 + Y3. Hypotheses for separate spec:
- Φ_b values wrong → partner SCAPS contact-model spec needed.
- Newton initial guess / V_max ramping protocol.
- Boltzmann-degenerate stats at N_D_ETL = 1e18 (Y4 future work).

### Exit criteria for Y2

- `Nd_ETL` working-regime closure: 30% UNDER → ≥ 70%.
- `Nd_ETL` V_oc ≤ E_g/q = 1.53 V across full sweep (no unphysical branch).
- Other sweep closures within ±5 pp of E6.4 baseline.

### New tests for Y2

- `tests/integration/test_e7_y2_nd_etl_closure.py`: closure floor + V_oc upper bound.
- `tests/integration/test_e7_y2_robin_no_regression.py`: CBO + interface stay ≥ baseline.
- If two-preset split: `tests/unit/scripts/test_per_sheet_config_map.py`.

## Phase Y3 — PVK doping direction (conditional)

**Activation**: Probe A shows direction still reversed under v2.

**Estimated time**: 1-2 days.

### Y3 branches (probe A outcome → action)

| Probe A | Y3 | Cost |
|---|---|---|
| Direction matches | skip | 0 |
| Direction reversed | full Y3 | 1-2 days |
| Direction match, magnitude > 2× off | parameter tune only | 0.5 day |

### Full Y3 hypothesis ladder (cheapest first)

1. **HTL/PVK defect σ bump** (~30 min). Re-test with σ=1e-15 (vs PDF
   1e-19). If flips → SCAPS interpretation of HTL/PVK σ differs from PDF.
   E1.15 ruled this out under v1, but v1 missed PVK-VB defect — re-test
   under v2 cheap.
2. **Y2 interaction** (~30 min, if Y2 shipped). Re-run under
   `scaps_mirror_v2_robin.yaml`. Robin × correct defect inventory may flip.
3. **Boltzmann-degenerate audit** (~4 hr, no code). Hand-compute
   (n_eq, p_eq) under degenerate vs non-degenerate stats at
   N_D ∈ {1e16, 1e17, 1e18, 1e19}. Confirm whether degenerate stats
   predict opposite V_oc direction.
4. **HTL Φ_b BC** (~4 hr, conditional on Y2). HTL Dirichlet pins
   p_L = N_A_PVK; SCAPS Φ_b workfunction decouples from bulk N_A.
   Re-test under Robin HTL.
5. **Fallback**: park as Y4 / future work. Document required either
   Fermi-Dirac stats OR partner SCAPS contact-model spec.

### Exit criteria for Y3

- `Na_PVK` V_oc derivative sign matches SCAPS across ≥ 80% of sweep range.
- Closure ≥ 50% (≥ 17 mV V_oc range vs SCAPS 34 mV).
- Other sweep closures within ±5 pp of pre-Y3 baseline.

### Out-of-scope under Y3

- Full Fermi-Dirac degenerate stats implementation (multi-week,
  separate spec if needed).
- Φ_b ohmic-equivalent contact BC beyond what Y2 ships.

## Tests + regression protection

Single new aggregate file: `tests/integration/test_e7_regression_floors.py`.

```python
EXPECTED_CLOSURE = {
    "CHI_ETL": (0.75, 1.20),       # min closure, max closure
    "Nt_PVK_ETL": (0.80, 1.50),
    "Nd_ETL": (0.50, 1.50),
    "Nt_C_PVK": (0.30, 1.50),
    "Na_PVK": (0.50, 1.50),
}
EXPECTED_VOC_BOUND = 1.53          # E_g/q for MAPbI3, V
```

Runs full v2 regression, parses `summary.json`, asserts per-sweep closure
∈ bounds AND no V_oc > bound. `@pytest.mark.slow`, ~7.5 min wall time.

### Test-count target

Pre-E7: 142+ SCAPS-subset tests pass. Post-E7: 150+.

## Overall exit criteria

| Sweep | Pre-E7 | E7 target | Hard floor |
|---|---|---|---|
| CHI_ETL | 83% | ≥ 80% | ≥ 75% |
| Nt_PVK_ETL | 109% | ≥ 90% | ≥ 80% |
| Nd_ETL | 30% UNDER + unphysical branch | ≥ 70% + V_oc ≤ 1.53 V | ≥ 50% + branch gone |
| Nt_C_PVK | 0.2% | ≥ 50% | ≥ 30% |
| Na_PVK | unknown v2 | direction matches + ≥ 50% | direction matches |

## Stop conditions

E7 stops when **any** of:
- All 5 sweeps meet target → ship + partner review.
- 2 of 3 trend gaps closed + Y3 falsified to need Fermi-Dirac → ship partial, park Y4.
- Any phase regresses ≥ 5 pp on non-target sweep AND fallback fails → revert + park.
- Spike Day 1 reveals all 3 probes need work > 1 week → re-scope spec entirely.

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Robin retry regresses CBO/interface | medium | high | Two-preset split (user-confirmed) |
| Multi-defect SRH perf > 10% | low | medium | Profile; defer N > 2 |
| Probe A: direction needs Fermi-Dirac | medium | high | Park Y3, ship Y1+Y2 |
| Φ_b xlsx values wrong/missing | medium | medium | Sensitivity sweep; request partner clarification |
| OneDrive sync corrupts outputs | low | low | Stage promptly per `project_onedrive_sync_resurrects_files.md` |

## Reproducibility

End-state reproducer (post-E7):
```bash
cd perovskite-sim && git checkout main && git pull
python scripts/run_scaps_v2_regression.py --out-dir outputs/scaps_validation_e7_final
pytest tests/integration/test_e7_regression_floors.py -m slow
```

Per-phase output dirs:
- Spike: `outputs/scaps_e7_probe_{a,b,c}/`
- Y1: `outputs/scaps_validation_e7_y1/`
- Y2: `outputs/scaps_validation_e7_y2/`
- Y3: `outputs/scaps_validation_e7_y3/`
- Final: `outputs/scaps_validation_e7_final/`

## Related artefacts

- `docs/scaps_validation_report.md` — running parity status; updated per phase.
- `docs/superpowers/specs/2026-05-28-e6-decision-gate.md` — E6.4 decision gate.
- `docs/superpowers/specs/2026-05-28-e6.5-vmax-low-nd.md` — E6.5 V_max probe (falsified V_max-only fix).
- `docs/superpowers/specs/2026-05-26-e1.14-base-voc-audit.md` — E1.14 base V_oc audit (74 mV structural, out-of-scope under trend bar).
- `docs/superpowers/specs/2026-05-27-e1.15-pvk-doping-direction.md` — E1.15 PVK doping (v1 baseline; superseded by Probe A).
- `tests/integration/scaps_reference.json` — partner ground truth (xlsx + PDF parsed).
- `configs/scaps_mirror_v2.yaml` — current SCAPS parity baseline.
- `scripts/run_scaps_v2_regression.py` — reproducer.

## What NOT to retry

Per E6.4 decision gate, do not reopen without explicit partner authorisation:

- Newton-Krylov reformulation with iface-plane state as DAE block.
- QSS reduction to Pauwels-Vanhoutte algebraic constraint.
- SG-flux-consistent face-density extraction in `physics/continuity.py`.
- Thin-shell volumetric SRH on absorber/ETL interface.
- Boltzmann-degenerate carrier statistics implementation (only as Y3 hypothesis test).
- Φ_b ohmic-equivalent contact BC beyond Robin (only as Y3 hypothesis test).

These remain archived as `failed-prototype/*` tags.
