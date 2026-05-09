# JMI Conference Presentation — Design Spec

**Author:** Xuan-Yan Chen (Shane)
**Created:** 2026-05-10
**Status:** Draft, awaiting user review

---

## 1. Purpose

Design a 25-minute oral presentation for the JMI conference that combines two
parallel research threads into one coherent narrative:

1. **AI + DFT photovoltaic material screening** — small-sample machine
   learning surrogate for HSE06 properties, used to filter a materials
   database for candidates that achieve both high efficiency and high
   stability.
2. **SolarLab device-simulation platform** — 1D + 2D drift-diffusion
   simulator with TMM optics, mobile-ion transport, and selective Robin
   contacts; positioned as the bridge between predicted material
   properties and predicted device-scale photovoltaic performance.

The talk closes with a forward-looking experimental synthesis collaboration.

## 2. Scope

In scope:

- Slide outline (titles, content, figure-per-slide).
- Visual design system (palette, typography, slide template, dividers).
- Hero-figure mockups for the three highest-stakes slides.
- Risk register listing data still missing before the deck is talk-ready.

Out of scope:

- The PowerPoint file itself (built in a later implementation plan).
- Speaker notes wording.
- Running new ML / DFT / device-sim jobs to generate missing data.

## 3. Format

- **Length:** 25 minutes (≈20 min talk + 5 min Q&A).
- **Slot:** oral talk, JMI conference.
- **Audience:** assumed mixed materials-informatics — both
  DFT/experimental and ML/AI literate. Assume both sides primer-free.
  Listed in §10 as an open item to confirm.
- **Language:** slides English, speech English.
- **Deck format:** Microsoft PowerPoint (.pptx).

## 4. Narrative arc — Funnel (approach B)

Six sections, ordered for payoff. The device-simulation software is
introduced *after* candidate properties are in hand, so its capabilities
land right before the simulated J-V on a top candidate.

1. **Problem** — efficiency vs stability tradeoff. CIGS stable but
   inefficient; perovskite efficient but unstable.
2. **Strategy** — template-guided database screening on perovskite and
   chalcogenide skeletons, optimising both metrics.
3. **ML + features** — HSE06 is too costly to apply at database scale;
   small-sample ML surrogate trained on a curated feature set.
4. **Candidates + DFT stability** — top-N screened structures, validated
   by full DFT (formation energy, decomposition energy, phonon DOS).
5. **Bridge — device simulation** — SolarLab capabilities, improvements
   over SCAPS, 1D vs 2D fidelity.
6. **Payoff** — predicted device-scale J-V on a top candidate (or
   roadmap if not yet run); experimental-synthesis collaboration; close.

## 5. Slide outline (22 slides, ~50 s/slide for 20 min talk)

```
COVER (1)         s1   Title, authors, affiliation. Type only — no JMI logo.
                  ─────────────────────────────────────────────
01 PROBLEM (3)    s2   Efficiency–stability landscape (target box upper-right)
                  s3   Why neither alone works (PCE timeline + degradation)
                  s4   Roadmap (six-step funnel diagram)
                  ─────────────────────────────────────────────
02 STRATEGY (2)   s5   Template skeletons (perovskite ABX₃ + chalcogenide)
                  s6   Pipeline schematic (DB → ML → DFT → device → expt)
                  ─────────────────────────────────────────────
03 ML+FEATURES(4) s7   HSE06 wall-time bar chart (PBE / HSE06 / GW)
                  s8   Feature set (composition / structural / electronic)
                  s9   Small-sample model (architecture, n_train, CV scores)
                  s10  Parity plot vs HSE06 hold-out (Eg, μ, m*, Eb)
                  ─────────────────────────────────────────────
04 CANDIDATES (4) s11  Top-N table (Eg, μₑ, μₕ, m*, Eb, formation E)
                  s12  DFT stability (decomposition E + phonon DOS)
                  s13  Hit #1 — structure + property card
                  s14  Hit #2 — structure + delta radar vs perovskite
                  ─────────────────────────────────────────────
05 BRIDGE (4)     s15  Why device simulation (DFT params → J-V)
                  s16  SolarLab software stack (TMM, ions, Robin, solver)
                  s17  Capability table vs SCAPS-1D
                  s18  1D vs 2D fidelity hero (J-V + grain-recomb spatial map)
                  ─────────────────────────────────────────────
06 PAYOFF (3)     s19  Predicted J-V on top candidate (or roadmap)
                  s20  Outlook + experimental collaboration
                  s21  Summary (three takeaway cards)
                  ─────────────────────────────────────────────
END (1)           s22  Acknowledgments + references + Q&A
```

Trim notes:

- Solver-tech detail (Radau + bisection-in-time + Phase 3.1b fallback)
  is folded into s16 as a single bullet, not its own slide.
- Acknowledgments + references combined onto s22 for time budget.

## 6. Figure-per-slide table

| # | Slide | Figure | Source / Plan |
|---|---|---|---|
| s1 | Cover | Type only | Custom |
| s2 | Eff-vs-stability | Scatter PCE vs T₈₀, target box upper-right | Lit data |
| s3 | Why neither | (a) PCE timeline by tech, (b) degradation curves | NREL + lit |
| s4 | Roadmap | 6-step funnel diagram | Custom SVG |
| s5 | Templates | Perovskite + chalcogenide structure thumbnails | VESTA |
| s6 | Pipeline | Boxed flowchart | Custom SVG |
| s7 | HSE06 cost | Wall-time bar chart | DFT logs |
| s8 | Features | Pictographic 3-cluster | Custom |
| s9 | ML model | Architecture box + CV-score table | ML run |
| s10 | Validation | Parity plot vs HSE06 | ML run |
| s11 | Top-N | Compact ranked table | Screening run |
| s12 | Stability | Decomposition-E bar + phonon DOS | DFT runs |
| s13 | Hit #1 | Structure + property card | VESTA + table |
| s14 | Hit #2 | Structure + radar vs perovskite | VESTA + custom |
| s15 | Why sim | DFT-params → J-V diagram | Custom SVG |
| s16 | Stack | Software-stack diagram | repo / docs |
| s17 | vs SCAPS | Capability checklist table | Lit + repo |
| s18 | 1D vs 2D | J-V hero + 2D recomb map inset | `voc_grain_sweep.yaml` |
| s19 | Predicted J-V | J-V on top candidate w/ PCE annotation | New sim run |
| s20 | Outlook | Sim → synthesis arrow diagram | Custom |
| s21 | Summary | 3 takeaway cards | Type only |
| s22 | End | Acks + bibliography | Type only |

Hero figures (highest visual investment): **s2, s4, s6, s17, s18**.
Mockups for s2, s17, s18 approved 2026-05-10 in brainstorming session.

## 7. Visual design system

### 7.1 Palette — P1 graphite + crimson

| Role | Hex |
|---|---|
| Ink (titles, axes) | `#1c2533` |
| Body text | `#475569` |
| Hairline / disabled | `#cbd5e1` |
| Background tint | `#f5f7fa` |
| Accent (highlights, brand mark) | `#c0392b` |

Single warm accent used sparingly: section number on dividers, brand
mark in footer, one-data-series highlight on plots. Everything else
graphite.

### 7.2 Typography — Arial only

- Headings: Arial Bold.
- Body: Arial Regular.
- Equations, numbers, code fragments: Arial Regular (no monospaced
  fallback).
- Hierarchy by **weight + size only**, not family.
- Sizes (16:9 deck): title 28pt, subtitle 16pt, body 14pt,
  caption 11pt, footer 10pt.

### 7.3 Subscripts and superscripts

This is a hard rule, not a preference. **Never** type `V_oc`, `J_0`,
`m^2`, `J_sc^2` literally on a slide. Use one of:

- PowerPoint native subscript / superscript formatting (Format → Font
  → Subscript / Superscript, or `Ctrl+=` / `Ctrl+Shift+=`).
- Real Unicode characters where coverage exists (`V₀`, `J²`, `m*`).

The same rule applies to plot labels, table headers, and the deck
filename. Reference: project memory
`feedback_physics_label_typography.md`.

### 7.4 Standard slide layout

Three-row grid (16:9):

- **Header** — section number + uppercase title (top-left, letter-
  spaced, body grey `#475569`), slide counter `n / 22` (top-right,
  hairline `#cbd5e1`).
- **Body** — slide title (Arial Bold 28pt) + subtitle (Arial Regular
  16pt) + content. Default split: figure left (60%) + bullets right
  (40%).
- **Footer** — author short cite (left, 10pt) + crimson `SolarLab`
  brand mark (right, 10pt, letter-spaced).

No JMI logo on any slide.

### 7.5 Section dividers

Full-bleed numeral. Big crimson `#c0392b` numeral `01` (~250pt) +
uppercase section title (`PROBLEM`, ~36pt, body grey `#475569`) on
ink background `#1c2533`. One per section = 6 dividers total, already
counted in slide budget.

### 7.6 Plotting

Re-use SolarLab `plot-theme.ts` Nature-style mode for any device-sim
figures (s17 reference, s18, s19). Same Arial substitution applied
to figure text.

## 8. Time budget

- 22 content slides @ ~50 s ≈ 18 min spoken.
- ~2 min buffer for transitions and demo callouts.
- 5 min Q&A.

## 9. Risk register — data gaps before talk-ready

| # | Gap | Action | Blocker? |
|---|---|---|---|
| R1 | s7 HSE06 wall-time numbers | Pull from DFT logs | No, easy |
| R2 | s9–s10 ML metrics + parity-plot data | Export trained-model metrics + hold-out predictions | No |
| R3 | s11 candidate top-N table | Finalise ranked list | Soft yes — anchors §4 |
| R4 | s12 phonon DOS + decomposition-E for top hits | Run DFT on top 2-3 candidates | Yes for §4 |
| R5 | s18 1D vs 2D matched-stack run | Run on `voc_grain_sweep.yaml` (this repo) | Yes for §5 |
| R6 | s19 predicted J-V on top candidate via SolarLab sim | Feed top-hit DFT params → `run_jv_sweep_2d` | Yes for §6 payoff |

R5 and R6 can be planned and executed inside this repo; the rest
require user-side data exports from the AI+DFT project tree.

## 10. Open items

- **Talk date / deadline.** TBD. Spec is timeless; implementation plan
  will need a date to schedule data-gap fills.
- **Audience composition.** Assumed mixed materials-informatics in §3.
  Confirm with conference programme or organisers before drafting
  speaker notes — primer depth depends on this.
- **Final wording for s21 summary takeaways.** Will be drafted during
  implementation pass once payoff slide (s19) data lands.
- **Top-candidate names.** Drafted with placeholder identifiers; real
  names slot in once R3 / R4 close.

## 11. References

- Brainstorming session 2026-05-10, mockups under
  `.superpowers/brainstorm/50880-1778341121/content/`.
- Project memory: `feedback_physics_label_typography.md`.
- Plot theme: `perovskite-sim/frontend/src/plot-theme.ts`
  (Nature-style mode shipped May 2026).
- 2D solver: `perovskite-sim/perovskite_sim/twod/solver_2d.py`
  (Robin selective contacts, validated against 1D within 6×10⁻⁷ V V₀C).
