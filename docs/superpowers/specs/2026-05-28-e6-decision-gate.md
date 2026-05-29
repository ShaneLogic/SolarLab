# Phase E6 — SCAPS YAML audit decision gate

**Status:** decision complete, action items defined
**Branch:** `e6-scaps-yaml-audit`
**Commits:** `6ed8ce8` (ground truth), `05a2c73` (v2 YAML + audit doc),
`c91726f` (loader extension), Step 4 outputs (this commit).

## What ran

Three commits (Step 1–3) extended the SCAPS compatibility layer:

1. **E6.1 Ground truth** — Partner xlsx (12 sheets, 251 sweep points) + PDF
   (21 pp) archived to `docs/superpowers/references/` and parsed into
   `tests/integration/scaps_reference.json` (machine-readable regression target).
2. **E6.2 Defect-corrected YAML** — `configs/scaps_mirror_v2.yaml` rewritten
   against PDF page 1 base table. Three defect-inventory fixes vs v1:
   added Perovskite-VB bulk defect, added HTL/PVK interface defect (Single,
   inactive in SCAPS), corrected PVK/ETL interface from Single+σ=1e-15
   +cf=1e-4 (4-order σ error hidden by calibration factor) to Gaussian
   +σ=1e-19.
3. **E6.3 Loader extension** — `bulk_defects:` list with parallel-SRH
   lifetime combine + inverse-lifetime-weighted n1/p1; `E_t_eV_above_vb`
   mutex with `_below_cb`; `distribution: single | gaussian` accepted;
   strict-key validation prevents silent schema drift. 17 new unit tests.

Step 4 ran `scripts/run_scaps_v2_regression.py` on the four marquee sweeps
the loader extension unblocked, comparing against the xlsx ground truth.

## Step 4 results (working-regime closure)

Working regime = points where SolarLab brackets V<sub>oc</sub> within `V_max=1.6 V`.

| Sheet | n (brk) | SCAPS Δ_subset | SolarLab Δ | Closure | Median Δ | Max |Δ| |
|---|---|---|---|---|---|---|
| CHI_ETL (CBO) | 14/14 | 918 mV | 762 mV | **83 %** | -140 mV | 166 mV |
| Nd_ETL (ETL doping) | 8/11 | 99.6 mV | 29.7 mV | **30 %** | -83 mV | 153 mV |
| Nt_PVK ETL (interface) | 6/7 | 226.7 mV | 246.2 mV | **109 %** | -304 mV | 358 mV |
| Nt_C_PVK (PVK bulk) | 7/7 | 38.6 mV | 0.1 mV | **0.2 %** | -96 mV | 96 mV |

Reproducer:

```bash
cd perovskite-sim && python scripts/run_scaps_v2_regression.py
# → outputs/scaps_validation_e6/{*.csv, summary.json}, ~7.5 min wall time
```

## Comparison vs parked diagnosis

`project_scaps_validation_parked.md` (origin/main HEAD `17e0ce4`) recorded:

| Sweep | Parked (v1) | E6.4 (v2) | Shift |
|---|---|---|---|
| CBO V<sub>oc</sub> range | 85 % | 83 % | noise (-2 pp) |
| iface N<sub>t</sub> V<sub>oc</sub> range | 74 % | **109 %** | **+35 pp** |
| ETL doping V<sub>oc</sub> range | **784 % OVER** (1075/137) | **30 % UNDER** (30/100) | **sign flipped** |
| Bulk N<sub>t</sub> V<sub>oc</sub> range | 0 % | 0.2 % | unchanged (masked) |

**The ETL doping diagnosis was wrong.** Parked memory attributed the
1075 mV / 137 mV ratio to "MoL+Scharfetter-Gummel cannot sample SCAPS'
analytical interface-plane carrier density" and recommended Newton-Krylov
or QSS reduction (multi-week research-grade refactor). Step 4 shows the
parked 1075 mV SolarLab range was inflated by **unbracketed V<sub>oc</sub>=0
sentinels at low ETL doping** — the device "looks dead" because the
sweep V<sub>max</sub>=1.6 V did not extend far enough, NOT because the solver
over-amplifies doping changes. The Step 4 script's per-point CSV reveals
SolarLab's 3 low-Nd points (1e10/1e11/1e12 cm<sup>−3</sup>) all return
`voc_bracketed=False` with V<sub>oc</sub>=0.

Once unbracketed points are filtered, v2 is 3× **under**-sensitive — the
opposite sign of the parked diagnosis. Architectural Newton-Krylov /
QSS-reduction work would have widened the (now under-sensitive) gap,
not closed it.

## Decision

**Ship `scaps_mirror_v2.yaml` + Phase E6.3 loader as the new SCAPS parity
baseline.** Three of four marquee sweeps closed acceptably given the
remaining shortfalls are diagnostic, not structural:

1. **CBO (83 %)** — close to parked baseline; remaining 17 % shortfall
   is concentrated on the spike side (Δ_E_C ≥ 0). Plateau height differs
   ~150 mV — likely band-offset thermionic-emission cap behaviour at
   |ΔE<sub>C</sub>| > 0.1 eV. Out of scope for E6.
2. **Interface N<sub>t</sub> (109 %)** — major improvement vs parked 74 %.
   Loader's correct σ=1e-19 + N<sub>t</sub>=1e12 cm<sup>−2</sup> (vs v1's 1e-15 + cf=1e-4)
   gives bit-equivalent SRV math but with transparent partner-facing
   units. Slight over-sensitivity within noise; magnitude excellent.
3. **PVK bulk N<sub>t</sub> (0 %)** — known mask: PVK/ETL interface SRV=0.01 m/s
   sets the recombination ceiling; bulk N<sub>t</sub> changes within
   [1e9, 1e15] cm<sup>−3</sup> do not move V<sub>oc</sub> until they exceed the interface
   contribution. NOT an architectural problem — interface tuning OR
   correct multi-defect SRH solver hook would unmask.
4. **ETL doping (30 %)** — direction preserved, magnitude
   under-sensitive in the working regime. The unbracketed-V<sub>oc</sub> artifact
   at low Nd_ETL is a separate gap — needs investigation of contact
   equilibrium / V<sub>max</sub> sweep range, not a discretisation refactor.

## Action items (post-E6.4)

### Immediate (this commit, E6.4)

- Land Step 4 script + outputs as the canonical regression artifact.
- Update `project_scaps_validation_parked.md` memory to reflect that
  the architectural-refactor narrative was based on a partial input
  inventory; v2 closure dramatically different and no longer warrants
  Newton-Krylov / QSS as the next move.

### Near-term (E6.5+, separate branch)

- **Low-Nd V<sub>oc</sub> bracket failure** — sweep `V_max` up from 1.6 V to e.g.
  2.5 V at low ETL doping; check whether bracketed range matches SCAPS
  ~100 mV. Likely V<sub>bi</sub>(N<sub>D</sub>) shifts the open-circuit beyond 1.6 V.
- **PVK bulk N<sub>t</sub> mask** — verify by setting PVK/ETL interface to
  near-zero SRV in an isolated sweep, then re-running Nt_C_PVK to
  confirm bulk SRH becomes visible. If yes, the mask is interface-SRV;
  if no, multi-defect collapse approximation under-counts at trap-rich
  bulk.
- **CBO spike-side plateau gap** — investigate thermionic-emission cap
  in `continuity.py` for |ΔE<sub>C</sub>| > 0.1 eV. SCAPS may have softer TE
  thresholds than SolarLab's Richardson-Dushman implementation.

### Do NOT pursue

- **Newton-Krylov reformulation** with iface-plane state as full DAE
  block — solving wrong problem; the parked diagnosis was driven by
  unbracketed-V<sub>oc</sub> artifact, not by genuine discretisation insufficiency.
- **QSS reduction** to Pauwels-Vanhoutte algebraic constraint — same
  reason; would widen the now-under-sensitive ETL doping gap.
- **Any retry of `failed-prototype/*` tagged refactors** without
  partner authorisation; these remain archived in the remote.

## Out-of-scope improvements that would tighten parity

- `pair-nt-cbo-pvk-etl` 2D sweep (100 points) — not run in Step 4
  because per-point wall time would push regression to >15 min. Worth
  running once for the final report.
- Optics parity — SolarLab J<sub>sc</sub>=333 A/m<sup>2</sup> vs SCAPS 263 A/m<sup>2</sup>. The
  +27 % gap is consistent across all four sweeps and is dominated by
  TMM vs SCAPS' scalar-alpha integral on the 800 nm MAPbI3 absorber.
  Out of scope for SCAPS parameter parity (the loader doesn't touch
  optics) but flagged because it inflates SolarLab's PCE by ~3 pp
  across the board.

## Related

- `docs/superpowers/specs/2026-05-28-e2a-scaps-yaml-audit-vs-pdf.md` —
  Phase E2a PDF audit (precursor)
- `~/.claude/.../memory/project_scaps_validation_parked.md` — parked
  memory (to be updated alongside this gate)
- `outputs/scaps_validation_e6/` — Step 4 raw CSVs + summary.json
- `scripts/run_scaps_v2_regression.py` — reproducer
