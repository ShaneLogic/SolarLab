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
