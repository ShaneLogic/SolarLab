# Phase E2 design RFC — band-bending-depletion interface face density

**Status:** design RFC (Sprint 1 Day 4-5 deliverable, no code change yet)
**Author:** Phase E2a investigation
**Branch:** `e2a-scaps-source-audit` (RFC); E2 implementation lands on `e2-bbd-face-density` (cut off main when this RFC is approved)
**Predecessors:** Sprint 1 Days 1-3.5 (audit + probe). See related docs.

## Goal

Close the three SCAPS-parity gaps blocked on interface face-density convention:
- **ETL doping over-sensitivity** (SolarLab 1075 mV vs SCAPS 137 mV → 8× over)
- **Bulk N_t mismatch** (SolarLab 0 mV vs SCAPS 39 mV — masked by interface SRH)
- **(partial) PVK doping direction** (HTL/PVK side, structural)

Hypothesis (Day 1 audit + Day 3 probe + Day 3.5 N_D sweep):
**SCAPS samples interface-plane densities with a band-bending depletion
factor `exp(-Δφ/V_T)` applied within the bulk side of each layer**,
NOT bulk-interior densities (E1.5) and NOT bulk × χ-step (E1.6 v2).

Day 3.5 probe shows BBD np sensitivity to N_D_ETL is 2.8× LOWER than
E1.5 — right direction, partial magnitude. Need solver-wired prototype
to confirm magnitude in the real SRH path (asymptotic vs. low-injection).

## Decision: prototype-first, then ship-or-pivot

**Sprint 2 Day 1 (1 day):** Solver-wired BBD prototype gated by env var.
**Sprint 2 Day 2-3:** Pass/fail decision based on validation pass.
**Sprint 2 Days 4-10:** Either ship BBD (pass) or pivot to thin-shell
volumetric SRH (fail).

This prototype-first sequencing limits research-grade risk: if the BBD
formula is wrong, we burn 1 day not 2 weeks.

## Architecture

### Phase E2.1 — Prototype (Sprint 2 Day 1)

**File touches:** `perovskite_sim/solver/mol.py` only.

**Change to `_apply_interface_recombination` signature:**

```python
def _apply_interface_recombination(
    dn, dp, n, p, stack, mat,
    *,
    phi: np.ndarray | None = None,  # NEW — for BBD path
) -> None:
```

**Inside the loop, when env var `SOLARLAB_BBD_FACE=1`:**

```python
if os.getenv("SOLARLAB_BBD_FACE") == "1" and phi is not None:
    n_face = float(n[eval_n_idx]) * math.exp(
        (phi[idx] - phi[eval_n_idx]) / V_T
    )
    p_face = float(p[eval_p_idx]) * math.exp(
        -(phi[idx] - phi[eval_p_idx]) / V_T
    )
else:
    n_face = float(n[eval_n_idx])  # E1.5 cross-carrier (legacy)
    p_face = float(p[eval_p_idx])
R_s = interface_recombination(n_face, p_face, ni_sq_eff, ...)
```

**Caller update in `assemble_rhs`** (line ~1131):

```python
_apply_interface_recombination(dn, dp, n, p, stack, mat, phi=phi)
```

`phi` is already in scope at that point (line ~1090 from the Poisson solve).

**LoC delta:** ~15 lines + 1 import. No new data classes.
**Test surface:** 2 RED tests pinning legacy bit-identity (env unset)
and BBD-active prototype path (env set, deterministic).
**Risk:** narrow — gated by env var, default unchanged.

### Phase E2.2 — Promotion to data-model flag (Sprint 2 Day 4-5, IF prototype passes)

If E2.1 passes validation, promote env var to a proper field:

```python
# perovskite_sim/models/device.py
@dataclass(frozen=True)
class InterfaceDefect:
    ...
    face_density_model: Literal["cross_carrier", "bbd"] = "cross_carrier"
```

Plumb through `scaps_compat/loader.py` (parse `face_density_model: bbd`
from YAML), `backend/main.py:stack_from_dict` (parse from inline device
JSON), `MaterialArrays.interface_face_density_model: tuple[str, ...]`,
and `_apply_interface_recombination` reads the per-interface model
from `mat.interface_face_density_model[k]` rather than env var.

scaps_mirror.yaml updates the single PVK/ETL interface to
`face_density_model: bbd`. Legacy stacks (no field set) default to
`cross_carrier` for full bit-identity.

**LoC delta:** ~80 lines (loader + dataclass + plumbing + 6 tests).
**Test surface:** roundtrip tests, legacy bit-identity, BBD activation,
per-interface mix (one BBD + one cross-carrier).

### Phase E2.3 — calibration_factor consolidation (Sprint 2 Day 6-7)

With BBD active, the empirical `calibration_factor: 1e-4` in
scaps_mirror.yaml is partially absorbed by the physical depletion
factor. Re-fit `calibration_factor` per-interface so it absorbs only
the residual SCAPS-specific factors (Pauwels-Vanhoutte form, v_th
convention, etc.). Target: cf ∈ [0.1, 10] (close to 1.0 — confirms
BBD captured the bulk of the gap).

If cf still needs to be < 1e-2, the BBD hypothesis is incomplete and
we need to investigate the remaining factor (likely Pauwels-Vanhoutte
form requires the original paper).

### Phase E2.4 — Validation + report (Sprint 2 Day 8-10)

Re-run `python scripts/run_scaps_validation.py` with
`--out-dir outputs/scaps_validation_e2`. Update
`docs/scaps_validation_report.md` with the BBD parity numbers. Cut
Phase E2 commit, merge to main with `--ff-only`, push.

If pass, the partner report shows:
- CBO closure ≥ 80 % ✓
- PVK/ETL interface defect closure ≥ 74 % ✓
- ETL doping range ≤ 200 mV (was 1075) ✓✓
- bulk N_t mismatch unmasked, sensitivity matches SCAPS ✓✓
- (partial) PVK doping direction improved

## Pass / fail criteria (Sprint 2 Day 3 gate)

### PASS (proceed to E2.2-E2.4)
- ETL doping V_oc range closes to ≤ 200 mV (target SCAPS 137 mV ±50%)
- CBO closure stays ≥ 80 % (was 85 % at ba10b10)
- Base V_oc stays within [1.05, 1.25] V envelope
- No new test failures in the existing 125+ SCAPS-subset tests

### FAIL (pivot to thin-shell volumetric SRH)
- ETL doping range still > 300 mV (BBD insufficient)
- OR CBO closure regresses below 70 %
- OR base V_oc moves outside [1.05, 1.25] V envelope
- OR > 5 existing tests break

### Fallback path: thin-shell volumetric SRH

If BBD fails the gate, the alternative is to treat interface SRH as a
volumetric source over a thin shell (~1 nm) at the interface node,
with R = N_t_volumetric · σ · v_th · ... evaluated using the
solver's own n[idx], p[idx] at the interface node (NOT eval_n_idx /
eval_p_idx). The shell width and N_t conversion are calibrated to
SCAPS' areal N_t. This path is more invasive (changes the source
location from idx → shell mask) and has higher Newton risk near the
heterointerface, hence is the fallback rather than the primary.

Expected effort if needed: ~2 weeks for the shell mask + tests.

## Risk analysis

| Risk | Likelihood | Mitigation |
|---|---|---|
| BBD formula is wrong (Pauwels-Vanhoutte has different form) | medium | Prototype-first gates at 1 day, not 2 weeks |
| BBD breaks low-N_D Newton convergence | medium | Phase E1.9 `v_max_max_attempts` already mitigates; test with SOLARLAB_BBD_FACE=1 on the low-N_D end of the SCAPS sweep |
| BBD changes CBO sweep parity (was 85% closure at ba10b10) | low | Probe shows CBO direction preserved; validation gate catches regression |
| `phi` array not in scope in all `_apply_interface_recombination` callers | low | Single caller (assemble_rhs line 1131); easy to verify |
| Env-var path breaks frozen-dataclass conventions | low | Sprint 2.1 is intentional prototype; Sprint 2.2 promotes to proper field if prototype passes |
| Per-interface mix (one BBD + one cross-carrier) needed for HTL/PVK | low | Sprint 2.2 plumbs as per-interface field, supports mix by design |

## Convention checklist (per `.claude/CLAUDE.md` commit_protocol)

- Atomic commits with `Constraint:` / `Rejected:` / `Confidence:`
  trailers per phase.
- Branch `e2-bbd-face-density`. Cut from main when this RFC approved.
  Merge with `--ff-only`, delete after merge.
- TDD: each E2 commit ships RED → GREEN → REFACTOR with at least one
  test pinning the new behaviour.
- Frontend round-trip: Sprint 2.2 must include a
  `tests/integration/backend/test_stack_from_dict_face_density_model.py`
  test mirroring the Phase E1.7 pattern. CLAUDE.md flags silent
  placebo bugs as recurring class.
- Validation script naming: `outputs/scaps_validation_e2/` for Sprint 2.4
  partner-readable history.

## Next action (when this RFC is approved by user)

Cut `e2-bbd-face-density` branch off main. Write the first RED test
`tests/integration/test_e2_bbd_face_density_prototype.py` pinning the
env-var behaviour. GREEN implementation in `solver/mol.py` per E2.1
above. Single commit `feat(e2.1): prototype BBD face-density gated by
SOLARLAB_BBD_FACE`.

## Open question for user

This RFC commits the next 1 day (prototype) + conditional 1-2 weeks
(ship-or-pivot). Confirm partner is OK with prototype-first sequencing,
or prefer to wait on ref [13] Pauwels-Vanhoutte 1978 acquisition before
any code change (would defer Sprint 2 start by 1-2 weeks for library
ILL).

**Related memories:** [[project-scaps-validation-parked]],
[[feedback-karpathy-skill-for-optimization]] (karpathy guidelines before
solver edits), [[feedback-docs-in-sync]] (update README + CLAUDE.md
alongside new physics flag).
