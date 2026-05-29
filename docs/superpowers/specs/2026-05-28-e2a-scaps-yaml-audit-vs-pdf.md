# Phase E2a Sprint 1 Day 4 — scaps_mirror.yaml vs SCAPS PDF audit

**Status:** investigation complete, ready for action
**Inputs:**
- `~/Desktop/1D-SCAPS 模拟.pdf` (21 p, partner SCAPS test report)
- `~/Desktop/1R-Parameters.xlsx` (12 worksheets, raw sweep data)
- `configs/scaps_mirror.yaml` (current SolarLab mirror)

## Audit summary

| Block | Matches PDF? | Notes |
|---|---|---|
| **Device-level** (V<sub>bi</sub>, Phi, mode) | partial | V<sub>bi</sub>=1.30 V is SolarLab-derived; SCAPS does not declare V<sub>bi</sub> explicitly |
| **HTL layer** (11 fields) | ✓ ALL MATCH | thickness, Eg, chi, eps, mu, Nc/Nv, Nd/Na, v<sub>th</sub> identical |
| **PVK layer** (14 fields) | ✓ ALL MATCH | inc. B<sub>rad</sub>=1e-12, C<sub>n</sub>/p=2.3e-29 |
| **ETL layer** (11 fields) | ✓ ALL MATCH | inc. asymmetric mu_n=1e-2 / mu_p=1e-5 |
| **Bulk defect** (PVK only) | partial | YAML carries 1 (PVK-CB Single Et=0.1). PDF declares 4. |
| **Interface defect** (PVK/ETL) | ✗ MISMATCH | σ off 4×, units off, distribution off, calibration_factor hides gap |

## Critical defect-block mismatches

### PDF declares 4 defects (all `cm⁻³`, all Neutral); YAML declares 2

| # | PDF defect | σ_n=σ_p (cm²) | Distribution | Et (eV) | N<sub>t</sub> (cm⁻³) | YAML status |
|---|---|---|---|---|---|---|
| 1 | HTL/Perovskite | 1e-19 | Single | 0.6 | 1e12 | ✗ MISSING |
| 2 | Perovskite-CB | 1e-15 | Single | 0.1 | 1e12 | ✓ present as PVK `bulk_defect` |
| 3 | Perovskite-VB | 1e-15 | Single | 0.1 above VB | 1e12 | ✗ MISSING |
| 4 | Perovskite/ETL | 1e-19 | **Gaussian** (E_char=0.1) | 0.6 | N_peak=5.64e8, N_total=1e12 | ✗ MAPPED INCORRECTLY |

Defect 4 in YAML is encoded as an `interfaces:` entry with `N_t_cm2=1e12` (areal cm⁻²) + `sigma=1e-15` (4 orders off) + Single (wrong distribution) + `calibration_factor=1e-4` (absorbs all errors).

### Sweep notes from PDF for defect 4

> "若 SCAPS 输入 peak defect density 而不是 total defect density, 保持 characteristic energy 为 0.1 eV, 按比例缩放 N_peak = 5.64×10⁸ × N<sub>t</sub> / (1×10¹²)"

→ SCAPS Gaussian peak scales linearly with the sweep variable while holding E_char fixed.

## SCAPS reference sweep ranges (from PDF + xlsx)

| Sweep | V<sub>oc</sub> range | Notes |
|---|---|---|
| CBO (ΔE<sub>C</sub> from -1.0 to +0.7) | **918 mV** | 0.332 → 1.250, base at ΔE<sub>C</sub>=-0.16 → 1.1676 V |
| ETL donor doping (1e10 to 1e20) | **137 mV** | 1.1002 → 1.2373, base at 1e18 → 1.1676 V |
| PVK donor doping (1e8 to 1e18) | 34 mV | base 1e14 → 1.1676 V; only 1e16/1e18 deviate |
| HTL/PVK defect N<sub>t</sub> (1e9 to 1e15) | **0.5 mV** | essentially flat — interface 1 is inactive |
| PVK-CB defect N<sub>t</sub> (1e9 to 1e15) | 39 mV | base 1e12 → 1.1676; 1e15 → 1.129 |
| PVK-VB defect N<sub>t</sub> (1e9 to 1e15) | 11 mV | base 1e12 → 1.1676; 1e15 → 1.157 |
| PVK/ETL defect N<sub>t</sub> (1e9 to 1e15) | **282 mV** | dominant interface; base 1e12 → 1.1676; 1e15 → 0.968 |
| HTL/PVK defect E<sub>t</sub> (0.01 to 0.6) | <0.1 mV | flat |
| PVK-CB defect E<sub>t</sub> (0.01 to 0.6) | 0.4 mV | nearly flat |

Base SCAPS perf: V<sub>oc</sub>=1.1676 V, J<sub>sc</sub>=26.28 mA/cm², FF=86.99%, η=26.69%.

## Implications

1. **Base layer params match perfectly.** The 99 mV V<sub>oc</sub> gap (SolarLab 1.069 vs SCAPS 1.1676) does NOT come from layer band/mobility/doping mistuning.
2. **Defect inventory is the smoking gun.** YAML declares 2 of 4 defects. The dominant interface defect (PVK/ETL Gaussian → SCAPS V<sub>oc</sub> range 282 mV) is encoded with wrong σ (4 orders), wrong units (cm⁻² vs cm⁻³), wrong distribution (Single vs Gaussian), wrong Et below CB position, and a 1e-4 calibration factor hiding all of the above.
3. **ETL doping gap (1075 vs 137 mV) is partly explained by defect-block mismatch.** The cross-carrier P-V SRH in `_apply_interface_recombination` reads bulk-interior carrier densities; this couples linearly with N_D_ETL when the defect's σ·N<sub>t</sub> product is set 4 orders too large via the calibration trick. Correcting σ + N<sub>t</sub> units + distribution may narrow the gap WITHOUT touching the solver.
4. **PVK-VB defect is missing.** Adds a recombination channel SCAPS uses that SolarLab does not, contributing to V<sub>oc</sub> shortfall.
5. **HTL/PVK defect is missing but inactive in SCAPS** (V<sub>oc</sub> sweep flat). Adding it costs nothing structurally; matches partner model completeness.

## Recommended action sequence

### Step 1 (1 hr): Stage 1 fix — match defect inventory
- Add `bulk_defect` for PVK-VB (σ=1e-15, Single, Et=0.1 above VB, N<sub>t</sub>=1e12) to PVK layer
- Add HTL/PVK interface defect (σ=1e-19, Single, Et=0.6, N<sub>t</sub>=1e12 cm⁻³ — note volumetric)
- Remove current `interfaces:` PVK/ETL block with calibration_factor=1e-4
- Add new PVK/ETL Gaussian defect block (σ=1e-19, Gaussian, Et=0.6, E_char=0.1, N_peak=5.64e8 cm⁻³)
- ⚠ Requires loader extension if Gaussian distribution + bulk-vs-interface encoding not supported by `scaps_compat.loader`

### Step 2 (half-day): Extract xlsx → CSV
- 12 worksheets → 12 CSVs in `tests/fixtures/scaps_reference/`
- Source of truth for validation script + future regression tests

### Step 3 (half-day): Loader audit
- Check `scaps_compat.defects.py` accepts Gaussian distribution
- Check whether defect 4 must be expressed as bulk (PVK layer w/ Gaussian profile) or as interface (cm⁻² area density) in SolarLab's model
- Note: SCAPS treats it as bulk (cm⁻³) — SolarLab may need bulk-with-Gaussian-energy-profile not interface-SRH

### Step 4 (day): Rerun validation script
- Compare new SolarLab base V<sub>oc</sub> vs SCAPS 1.1676 V
- Compare ETL doping range — does correcting σ + units narrow 1075 → 137 mV?
- Compare PVK/ETL N<sub>t</sub> sweep — does it hit 282 mV without calibration_factor?

### Step 5 (decision gate):
- If Step 4 closes ≥80% of gaps → architectural refactor NOT needed → ship corrected YAML + PDF report
- If Step 4 still 8× over-sensitive on ETL doping → architectural limit confirmed regardless of config; Newton-Krylov path

## DO NOT

- Keep `calibration_factor=1e-4` "for backwards compatibility" — it hides the unit error
- Add HTL/PVK or PVK-VB defects via solver-level interface SRH — they are bulk defects in SCAPS
- Re-run any `failed-prototype/*` discretisation refactor before completing Step 4

## Related

- `docs/superpowers/specs/2026-05-27-e2a-scaps-source-audit.md` — Manual PDF source audit (parent spec)
- `docs/superpowers/specs/2026-05-26-e1.6-sg-face-density-spec.md` — calibration_factor origin
- `~/.claude/projects/.../memory/project_scaps_validation_parked.md` — parked-memory with 7-prototype failure log
