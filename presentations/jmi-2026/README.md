# JMI 2026 Deck

Source for the 25-min JMI talk. Builds `output/jmi-2026.pptx` end-to-end.

## Install

    cd presentations/jmi-2026
    python -m venv .venv && source .venv/bin/activate
    pip install -e .
    pip install -e ../../perovskite-sim    # for R5/R6 sim runs

## Regenerate data (R5 + R6 from spec § 9 risk register)

    python runs/r5_1d_vs_2d.py
    python runs/r6_candidate_jv.py

## Build deck

    python build_deck.py

Output: `output/jmi-2026.pptx`. Open in PowerPoint to review.

## Edit copy

All slide text lives in `copy.yaml`. Use `[sub]oc[/sub]` for subscripts
and `[sup]2[/sup]` for superscripts — the build script converts these
to real PPT subscript/superscript runs. Never type `_` or `^` literally.

## Audit typography

After every build, verify no `_` / `^` leaked and that sub/sup runs
carry the OOXML `baseline` attribute:

    python audit_typography.py

Pass: `PASS: no _ or ^ literals; sub/sup runs have baseline attr.`

## Data gaps before talk-ready

| Slide | Gap | Source |
|---|---|---|
| s7 | HSE06 wall-time numbers | DFT logs (R1) |
| s9 – s10 | ML model metrics + parity-plot data | ML run export (R2) |
| s11 | Top-N candidate table | Screening output (R3) |
| s12 | Decomposition E + phonon DOS | DFT runs (R4) |
| s13 – s14 | Real candidate names + parameter cards | Screening output (R3) |
| s18 | Auto-generated from `runs/r5_1d_vs_2d.py` | filled (legacy preset) |
| s19 | Auto-generated from `runs/r6_candidate_jv.py` | placeholder candidate |

When R1 – R4 data lands: drop new figures into `figures/output/` and
update the relevant slide entries in `copy.yaml`. Then re-run
`python build_deck.py` and `python audit_typography.py`.

### Caveats on the current data

* **R5 (s18) used the legacy Beer-Lambert preset** (`nip_MAPbI3.yaml` /
  `nip_MAPbI3_singleGB.yaml`) whose Φ inflates J_sc ~1.45× above the
  true AM1.5G photon flux. Real PCE for the MAPbI3 baseline ≈ 20 %,
  not the ~29 % the run reports. The 50 mV V_oc drop between 1D and
  2D is the real GB-recombination signal and is unaffected.
* **R6 (s19) is a synthetic placeholder.** The legacy preset
  decouples the absorber Eg field from optical absorption (Beer-
  Lambert uses fixed α and Φ), so a literal Eg shift would not move
  the device J-V. R6 ships with `placeholder: true` in the JSON and
  the figure is annotated accordingly. Replace with a real 2D-TMM run
  on a candidate-derived parameter set once R3 + R4 close.
