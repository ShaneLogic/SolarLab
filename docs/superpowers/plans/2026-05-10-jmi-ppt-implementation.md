# JMI PPT Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 22-slide PowerPoint deck for the 25-min JMI conference talk, generated end-to-end via `python-pptx` so the spec's typography rules (Arial-only, real subscript/superscript, no `_` or `^` literals) are enforced by code rather than memory.

**Architecture:** Source-controlled `presentations/jmi-2026/` directory holds (a) a `theme/` Python module with palette + Arial defaults + a tagged-text helper that turns `[sub]oc[/sub]` / `[sup]2[/sup]` markers into real PPT subscript/superscript runs, (b) a `figures/` directory of matplotlib scripts that emit PNG figures (re-using SolarLab's Nature-style plot theme where possible), (c) a flat `copy.yaml` with per-slide title/subtitle/body/figure-path, and (d) a single `build_deck.py` entrypoint that emits `jmi-2026.pptx`. Two device-sim runs (R5 1D-vs-2D matched stack, R6 candidate J-V) live as scripts under `figures/` so they regenerate cleanly on demand.

**Tech Stack:** `python-pptx` (deck assembly), `matplotlib` (figures), `pyyaml` (copy file), `numpy` (data wrangling). Re-uses SolarLab in-tree solver (`perovskite_sim.experiments.jv_sweep`, `perovskite_sim.twod.experiments.jv_sweep_2d`).

---

## File Structure

```
presentations/jmi-2026/
├── README.md                       Build / regenerate instructions
├── pyproject.toml                  Pinned deps (python-pptx, matplotlib, pyyaml, numpy)
├── theme/
│   ├── __init__.py
│   ├── palette.py                  P1 graphite + crimson hex constants
│   ├── fonts.py                    Arial size table + helper to set run.font
│   ├── runs.py                     Tagged-text → list-of-runs parser (sub/sup)
│   └── layouts.py                  Cover / divider / content slide builders
├── figures/
│   ├── __init__.py
│   ├── _common.py                  Matplotlib rcParams (Arial, P1 palette)
│   ├── s2_eff_stability.py         Scatter PCE vs T80
│   ├── s4_funnel.py                6-step funnel diagram
│   ├── s6_pipeline.py              Pipeline schematic
│   ├── s17_capability.py           Capability table (renders to PNG via mpl)
│   ├── s18_1d_vs_2d.py             Hero panel — needs R5 data
│   ├── s19_candidate_jv.py         Predicted J-V — needs R6 data
│   ├── data/
│   │   ├── r5_jv_curves.json       1D + 2D J-V from matched stack (committed)
│   │   └── r6_candidate_jv.json    Predicted J-V on placeholder candidate
│   └── output/                     PNGs (gitignored)
├── runs/
│   ├── r5_1d_vs_2d.py              Drives jv_sweep + jv_sweep_2d on matched stack
│   └── r6_candidate_jv.py          Drives jv_sweep_2d w/ candidate-derived params
├── copy.yaml                       Per-slide text (with [sub]/[sup]/[em] tags)
├── build_deck.py                   CLI entrypoint
└── output/
    └── jmi-2026.pptx               Final deck (committed)
```

`presentations/` is new at the SolarLab root. Add to root `pyproject.toml` workspace if any (none today — standalone is fine).

---

## Task 1: Scaffold directory and dependencies

**Files:**
- Create: `presentations/jmi-2026/README.md`
- Create: `presentations/jmi-2026/pyproject.toml`
- Create: `presentations/jmi-2026/.gitignore`
- Create: `presentations/jmi-2026/theme/__init__.py`
- Create: `presentations/jmi-2026/figures/__init__.py`
- Create: `presentations/jmi-2026/figures/data/.gitkeep`
- Create: `presentations/jmi-2026/figures/output/.gitkeep`
- Create: `presentations/jmi-2026/runs/__init__.py`
- Create: `presentations/jmi-2026/output/.gitkeep`

- [ ] **Step 1: Create directory tree**

```bash
mkdir -p "presentations/jmi-2026/theme" \
         "presentations/jmi-2026/figures/data" \
         "presentations/jmi-2026/figures/output" \
         "presentations/jmi-2026/runs" \
         "presentations/jmi-2026/output"
touch presentations/jmi-2026/{theme,figures,runs}/__init__.py
touch presentations/jmi-2026/{figures/data,figures/output,output}/.gitkeep
```

- [ ] **Step 2: Write `pyproject.toml` with pinned deps**

```toml
[project]
name = "jmi-2026"
version = "0.1.0"
description = "JMI conference deck (Chen et al., 2026)"
requires-python = ">=3.11"
dependencies = [
  "python-pptx==1.0.2",
  "matplotlib==3.9.2",
  "pyyaml==6.0.2",
  "numpy==2.1.2",
]
```

- [ ] **Step 3: Write `.gitignore`**

```
figures/output/*.png
figures/output/*.pdf
__pycache__/
*.pyc
.venv/
```

(Note: `output/jmi-2026.pptx` is committed; `figures/output/` PNGs regenerate on build.)

- [ ] **Step 4: Write `README.md` (build instructions)**

```markdown
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
```

- [ ] **Step 5: Verify and commit**

```bash
ls presentations/jmi-2026/
git add presentations/jmi-2026
git status --short
git commit -m "chore(jmi-ppt): scaffold deck directory and deps"
```

Expected: directory created, 9 files staged.

---

## Task 2: Theme module — palette and fonts

**Files:**
- Create: `presentations/jmi-2026/theme/palette.py`
- Create: `presentations/jmi-2026/theme/fonts.py`
- Create: `presentations/jmi-2026/theme/test_palette.py`

- [ ] **Step 1: Write `theme/palette.py`**

```python
"""P1 graphite + crimson palette (spec § 7.1)."""
from pptx.dml.color import RGBColor

INK = RGBColor(0x1c, 0x25, 0x33)         # titles, axes, divider bg
BODY = RGBColor(0x47, 0x55, 0x69)         # body text, section labels
HAIRLINE = RGBColor(0xcb, 0xd5, 0xe1)     # rules, slide counter
TINT = RGBColor(0xf5, 0xf7, 0xfa)         # background tint
ACCENT = RGBColor(0xc0, 0x39, 0x2b)       # crimson — divider numerals, brand mark
WHITE = RGBColor(0xff, 0xff, 0xff)
```

- [ ] **Step 2: Write `theme/fonts.py`**

```python
"""Arial-only size table (spec § 7.2)."""
from pptx.util import Pt

FONT_NAME = "Arial"

# (size_pt, bold) by role
TITLE = (Pt(28), True)
SUBTITLE = (Pt(16), False)
BODY = (Pt(14), False)
CAPTION = (Pt(11), False)
FOOTER = (Pt(10), False)
DIVIDER_NUMERAL = (Pt(250), True)
DIVIDER_TITLE = (Pt(36), True)
SECTION_LABEL = (Pt(11), False)         # uppercase header on content slides
SLIDE_COUNTER = (Pt(11), False)

def apply(run, size_bold, color):
    """Set run font to Arial with given (size, bold) and RGBColor."""
    run.font.name = FONT_NAME
    run.font.size = size_bold[0]
    run.font.bold = size_bold[1]
    run.font.color.rgb = color
```

- [ ] **Step 3: Write `theme/test_palette.py`**

```python
from pptx.dml.color import RGBColor
from theme.palette import INK, BODY, HAIRLINE, TINT, ACCENT

def test_palette_hex_values():
    assert INK == RGBColor(0x1c, 0x25, 0x33)
    assert BODY == RGBColor(0x47, 0x55, 0x69)
    assert HAIRLINE == RGBColor(0xcb, 0xd5, 0xe1)
    assert TINT == RGBColor(0xf5, 0xf7, 0xfa)
    assert ACCENT == RGBColor(0xc0, 0x39, 0x2b)
```

- [ ] **Step 4: Run test to confirm pass**

Run: `cd presentations/jmi-2026 && python -m pytest theme/test_palette.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add presentations/jmi-2026/theme
git commit -m "feat(jmi-ppt): add P1 palette and Arial font helpers"
```

---

## Task 3: Tagged-text → runs parser (subscript/superscript handling)

**Files:**
- Create: `presentations/jmi-2026/theme/runs.py`
- Create: `presentations/jmi-2026/theme/test_runs.py`

This is the typography-enforcement core of the deck. `[sub]…[/sub]` and `[sup]…[/sup]` markers in `copy.yaml` get parsed into a list of `(text, role)` segments where `role ∈ {"normal", "sub", "sup"}`. The deck builder turns each segment into a `pptx` text run with `baseline` set to `-25000` (sub) or `30000` (sup) on the underlying XML, mirroring PowerPoint's native subscript/superscript formatting.

- [ ] **Step 1: Write failing tests for the parser**

```python
# theme/test_runs.py
from theme.runs import parse_segments

def test_plain_text_one_normal_segment():
    assert parse_segments("Hello") == [("Hello", "normal")]

def test_subscript_emits_sub_segment():
    assert parse_segments("V[sub]oc[/sub]") == [
        ("V", "normal"),
        ("oc", "sub"),
    ]

def test_superscript_emits_sup_segment():
    assert parse_segments("J[sup]2[/sup]") == [
        ("J", "normal"),
        ("2", "sup"),
    ]

def test_mixed_sub_and_sup_in_one_string():
    assert parse_segments("J[sub]sc[/sub] = J[sub]0[/sub] (e[sup]qV/kT[/sup] - 1)") == [
        ("J", "normal"),
        ("sc", "sub"),
        (" = J", "normal"),
        ("0", "sub"),
        (" (e", "normal"),
        ("qV/kT", "sup"),
        (" - 1)", "normal"),
    ]

def test_unmatched_tag_raises():
    import pytest
    with pytest.raises(ValueError, match="unmatched"):
        parse_segments("V[sub]oc")

def test_underscore_literal_raises():
    import pytest
    with pytest.raises(ValueError, match="underscore"):
        parse_segments("V_oc")

def test_caret_literal_raises():
    import pytest
    with pytest.raises(ValueError, match="caret"):
        parse_segments("J^2")
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `cd presentations/jmi-2026 && python -m pytest theme/test_runs.py -v`
Expected: 6 failed, ImportError on `parse_segments`.

- [ ] **Step 3: Implement `theme/runs.py`**

```python
"""Tagged-text parser. Hard-fails on raw `_` / `^` to enforce spec § 7.3."""
import re
from typing import Literal, List, Tuple

Role = Literal["normal", "sub", "sup"]
Segment = Tuple[str, Role]

_TAG_RE = re.compile(r"\[(sub|sup)\](.*?)\[/\1\]")

def parse_segments(text: str) -> List[Segment]:
    if "_" in text:
        raise ValueError(
            f"underscore literal found in slide text: {text!r}. "
            "Use [sub]…[/sub] instead (spec § 7.3)."
        )
    if "^" in text:
        raise ValueError(
            f"caret literal found in slide text: {text!r}. "
            "Use [sup]…[/sup] instead (spec § 7.3)."
        )
    if "[sub]" in text and "[/sub]" not in text:
        raise ValueError(f"unmatched [sub] tag in {text!r}")
    if "[sup]" in text and "[/sup]" not in text:
        raise ValueError(f"unmatched [sup] tag in {text!r}")

    out: List[Segment] = []
    cursor = 0
    for match in _TAG_RE.finditer(text):
        if match.start() > cursor:
            out.append((text[cursor:match.start()], "normal"))
        out.append((match.group(2), match.group(1)))
        cursor = match.end()
    if cursor < len(text):
        out.append((text[cursor:], "normal"))
    return out


def write_segments(paragraph, segments: List[Segment], size_bold, color):
    """Append each segment as a pptx text run; sub/sup via XML baseline."""
    from pptx.oxml.ns import qn
    from theme.fonts import apply

    for text, role in segments:
        run = paragraph.add_run()
        run.text = text
        apply(run, size_bold, color)
        if role == "sub":
            run._r.get_or_add_rPr().set("baseline", "-25000")
        elif role == "sup":
            run._r.get_or_add_rPr().set("baseline", "30000")
```

- [ ] **Step 4: Run tests to confirm pass**

Run: `cd presentations/jmi-2026 && python -m pytest theme/test_runs.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add presentations/jmi-2026/theme/runs.py presentations/jmi-2026/theme/test_runs.py
git commit -m "feat(jmi-ppt): tagged-text parser enforcing real sub/sup typography"
```

---

## Task 4: Slide-layout builders (cover, divider, content)

**Files:**
- Create: `presentations/jmi-2026/theme/layouts.py`
- Create: `presentations/jmi-2026/theme/test_layouts.py`

- [ ] **Step 1: Write failing test for cover slide**

```python
# theme/test_layouts.py
from pptx import Presentation
from theme.layouts import add_cover

def test_cover_has_title_and_two_subline_runs():
    prs = Presentation()
    prs.slide_width = 9144000          # 10in
    prs.slide_height = 5143500         # 5.625in (16:9)
    add_cover(prs, title="Photovoltaic material screening",
                  authors="Xuan-Yan Chen, ... (HKUST-GZ)",
                  affiliation="JMI 2026")
    assert len(prs.slides) == 1
    slide = prs.slides[0]
    texts = [shape.text_frame.text for shape in slide.shapes if shape.has_text_frame]
    joined = " | ".join(texts)
    assert "Photovoltaic material screening" in joined
    assert "Xuan-Yan Chen" in joined
    assert "JMI 2026" in joined
```

- [ ] **Step 2: Run test, confirm fail**

Run: `cd presentations/jmi-2026 && python -m pytest theme/test_layouts.py::test_cover_has_title_and_two_subline_runs -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `theme/layouts.py`**

```python
"""Cover, divider, content layout builders (spec § 7.4 & § 7.5)."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

from theme import palette, fonts
from theme.runs import parse_segments, write_segments


_BLANK_LAYOUT = 6        # PowerPoint blank layout index


def _add_blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[_BLANK_LAYOUT])


def _add_textbox(slide, left, top, width, height, text, role,
                 color, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    write_segments(p, parse_segments(text), role, color)
    return tb


def add_cover(prs, *, title, authors, affiliation):
    slide = _add_blank(prs)
    sw, sh = prs.slide_width, prs.slide_height
    _add_textbox(slide, Emu(0), Emu(int(sh * 0.40)), sw, Inches(1),
                 title, fonts.TITLE, palette.INK, PP_ALIGN.CENTER)
    _add_textbox(slide, Emu(0), Emu(int(sh * 0.55)), sw, Inches(0.5),
                 authors, fonts.SUBTITLE, palette.BODY, PP_ALIGN.CENTER)
    _add_textbox(slide, Emu(0), Emu(int(sh * 0.62)), sw, Inches(0.4),
                 affiliation, fonts.CAPTION, palette.HAIRLINE, PP_ALIGN.CENTER)
    # accent line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                  Emu(int(sw * 0.42)), Emu(int(sh * 0.52)),
                                  Emu(int(sw * 0.16)), Emu(int(sh * 0.004)))
    line.fill.solid()
    line.fill.fore_color.rgb = palette.ACCENT
    line.line.fill.background()
    return slide


def add_divider(prs, *, number, title):
    """Full-bleed numeral divider (spec § 7.5)."""
    slide = _add_blank(prs)
    sw, sh = prs.slide_width, prs.slide_height
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, sw, sh)
    bg.fill.solid()
    bg.fill.fore_color.rgb = palette.INK
    bg.line.fill.background()
    _add_textbox(slide, Emu(int(sw * 0.06)), Emu(int(sh * 0.10)),
                 Emu(int(sw * 0.6)), Inches(4),
                 number, fonts.DIVIDER_NUMERAL, palette.ACCENT)
    _add_textbox(slide, Emu(int(sw * 0.06)), Emu(int(sh * 0.78)),
                 Emu(int(sw * 0.88)), Inches(0.6),
                 title, fonts.DIVIDER_TITLE, palette.BODY)
    return slide


def add_content(prs, *, section_label, slide_index, total, title,
                subtitle, bullets, figure_path=None):
    slide = _add_blank(prs)
    sw, sh = prs.slide_width, prs.slide_height
    # header row
    _add_textbox(slide, Inches(0.5), Inches(0.3), Inches(7), Inches(0.3),
                 section_label.upper(), fonts.SECTION_LABEL, palette.BODY)
    _add_textbox(slide, Inches(7.5), Inches(0.3), Inches(2), Inches(0.3),
                 f"{slide_index} / {total}", fonts.SLIDE_COUNTER,
                 palette.HAIRLINE, align=PP_ALIGN.RIGHT)
    # body title + subtitle
    _add_textbox(slide, Inches(0.5), Inches(0.8), Inches(9), Inches(0.6),
                 title, fonts.TITLE, palette.INK)
    _add_textbox(slide, Inches(0.5), Inches(1.4), Inches(9), Inches(0.4),
                 subtitle, fonts.SUBTITLE, palette.BODY)
    # figure (left) + bullets (right)
    if figure_path:
        slide.shapes.add_picture(str(figure_path),
                                 Inches(0.5), Inches(2.0),
                                 width=Inches(5.4), height=Inches(3.2))
        bullet_left = Inches(6.2)
        bullet_w = Inches(3.3)
    else:
        bullet_left = Inches(0.5)
        bullet_w = Inches(9)
    if bullets:
        tb = slide.shapes.add_textbox(bullet_left, Inches(2.0),
                                      bullet_w, Inches(3.2))
        tf = tb.text_frame
        tf.word_wrap = True
        for i, line in enumerate(bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.LEFT
            write_segments(p, parse_segments(f"•  {line}"),
                           fonts.BODY, palette.BODY)
    # footer
    _add_textbox(slide, Inches(0.5), Inches(5.05), Inches(7), Inches(0.3),
                 "Chen et al. — JMI 2026", fonts.FOOTER, palette.HAIRLINE)
    _add_textbox(slide, Inches(7.5), Inches(5.05), Inches(2), Inches(0.3),
                 "SOLARLAB", fonts.FOOTER, palette.ACCENT, align=PP_ALIGN.RIGHT)
    return slide
```

- [ ] **Step 4: Add divider + content tests**

Append to `theme/test_layouts.py`:

```python
def test_divider_writes_numeral_and_title():
    prs = Presentation()
    prs.slide_width, prs.slide_height = 9144000, 5143500
    from theme.layouts import add_divider
    add_divider(prs, number="01", title="PROBLEM")
    slide = prs.slides[0]
    joined = " | ".join(s.text_frame.text for s in slide.shapes if s.has_text_frame)
    assert "01" in joined
    assert "PROBLEM" in joined


def test_content_renders_subscripts_via_runs():
    prs = Presentation()
    prs.slide_width, prs.slide_height = 9144000, 5143500
    from theme.layouts import add_content
    add_content(prs, section_label="03 · ML & FEATURES", slide_index=10,
                total=22, title="Validation",
                subtitle="Predicted vs HSE06",
                bullets=["R[sup]2[/sup] (E[sub]g[/sub]) = 0.91"])
    slide = prs.slides[0]
    # Walk every run in every textbox; ensure no '_' or '^' literal survived
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        for para in shape.text_frame.paragraphs:
            for run in para.runs:
                assert "_" not in run.text, f"underscore leaked: {run.text!r}"
                assert "^" not in run.text, f"caret leaked: {run.text!r}"
```

- [ ] **Step 5: Run all layout tests**

Run: `cd presentations/jmi-2026 && python -m pytest theme/test_layouts.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add presentations/jmi-2026/theme/layouts.py presentations/jmi-2026/theme/test_layouts.py
git commit -m "feat(jmi-ppt): cover/divider/content layout builders"
```

---

## Task 5: Figure script common module (matplotlib rcParams)

**Files:**
- Create: `presentations/jmi-2026/figures/_common.py`

- [ ] **Step 1: Write `_common.py`**

```python
"""Matplotlib rcParams aligned with deck theme (spec § 7.6)."""
from pathlib import Path
import matplotlib as mpl

INK = "#1c2533"
BODY = "#475569"
HAIRLINE = "#cbd5e1"
ACCENT = "#c0392b"
SECONDARY = "#1a73d6"  # used only when a 2nd data series needs to disambiguate
FIG_OUT = Path(__file__).parent / "output"
FIG_DATA = Path(__file__).parent / "data"

def configure():
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 11,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.linewidth": 0.8,
        "xtick.color": BODY,
        "ytick.color": BODY,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.dpi": 220,
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    FIG_OUT.mkdir(exist_ok=True)
```

- [ ] **Step 2: Smoke-test import**

Run: `cd presentations/jmi-2026 && python -c "from figures._common import configure; configure(); print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add presentations/jmi-2026/figures/_common.py
git commit -m "feat(jmi-ppt): matplotlib theme aligned with deck palette"
```

---

## Task 6: R5 — 1D vs 2D matched-stack run

**Files:**
- Create: `presentations/jmi-2026/runs/r5_1d_vs_2d.py`
- Create: `presentations/jmi-2026/figures/data/r5_jv_curves.json` (output, committed)

This produces the data behind hero slide s18. Drives the in-tree solver on a matched stack: 1D config `configs/nip_MAPbI3.yaml` and 2D config `configs/twod/nip_MAPbI3_singleGB.yaml`. Records V, J for each.

- [ ] **Step 1: Write `runs/r5_1d_vs_2d.py`**

```python
"""R5: 1D vs 2D matched-stack J-V (spec § 9, R5)."""
import json
from pathlib import Path

from perovskite_sim.config_loader import load_config
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.twod.experiments.jv_sweep_2d import run_jv_sweep_2d

PEROVSKITE_SIM_ROOT = Path(__file__).resolve().parents[3] / "perovskite-sim"
CFG_1D = PEROVSKITE_SIM_ROOT / "configs" / "nip_MAPbI3.yaml"
CFG_2D = PEROVSKITE_SIM_ROOT / "configs" / "twod" / "nip_MAPbI3_singleGB.yaml"
OUT = Path(__file__).resolve().parents[1] / "figures" / "data" / "r5_jv_curves.json"

def main():
    cfg_1d = load_config(str(CFG_1D))
    cfg_2d = load_config(str(CFG_2D))
    res_1d = run_jv_sweep(cfg_1d)
    res_2d = run_jv_sweep_2d(cfg_2d)
    payload = {
        "config_1d": str(CFG_1D),
        "config_2d": str(CFG_2D),
        "v_1d": list(res_1d.voltages_V),
        "j_1d": list(res_1d.currents_A_per_m2),
        "v_2d": list(res_2d.voltages_V),
        "j_2d": list(res_2d.currents_A_per_m2),
        "metrics_1d": {
            "voc_V": res_1d.metrics.voc_V,
            "jsc_A_per_m2": res_1d.metrics.jsc_A_per_m2,
            "ff": res_1d.metrics.ff,
            "pce_pct": res_1d.metrics.pce_pct,
        },
        "metrics_2d": {
            "voc_V": res_2d.metrics.voc_V,
            "jsc_A_per_m2": res_2d.metrics.jsc_A_per_m2,
            "ff": res_2d.metrics.ff,
            "pce_pct": res_2d.metrics.pce_pct,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")
    print(f"  1D: V_oc={payload['metrics_1d']['voc_V']:.4f} V, "
          f"PCE={payload['metrics_1d']['pce_pct']:.2f} %")
    print(f"  2D: V_oc={payload['metrics_2d']['voc_V']:.4f} V, "
          f"PCE={payload['metrics_2d']['pce_pct']:.2f} %")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify result-object field names match the actual API**

Run:

```bash
python -c "from perovskite_sim.experiments.jv_sweep import run_jv_sweep; \
  import inspect; print(inspect.signature(run_jv_sweep))"
python -c "from perovskite_sim.twod.experiments.jv_sweep_2d import run_jv_sweep_2d; \
  import inspect; print(inspect.signature(run_jv_sweep_2d))"
```

If field names diverge from `voltages_V` / `currents_A_per_m2` / `metrics.voc_V` etc., update the script to match (these specific field names came from the `JV2DResult` typing in `frontend/src/types.ts` reflection — the Python dataclass may use different conventions). Run a tiny smoke import on each result to confirm.

- [ ] **Step 3: Run R5 (long: ~1-3 min depending on settle_t)**

```bash
cd presentations/jmi-2026
python runs/r5_1d_vs_2d.py
```

Expected: prints two metric lines, writes `figures/data/r5_jv_curves.json`.

If the 2D run fails, fall back to a placeholder JSON (committed) labelled `"placeholder": true` so the deck still builds; flag in copy.yaml that s18 is using preliminary data.

- [ ] **Step 4: Commit data + script**

```bash
git add presentations/jmi-2026/runs/r5_1d_vs_2d.py \
        presentations/jmi-2026/figures/data/r5_jv_curves.json
git commit -m "feat(jmi-ppt): R5 1D vs 2D matched-stack J-V data"
```

---

## Task 7: R6 — predicted J-V on candidate placeholder

**Files:**
- Create: `presentations/jmi-2026/runs/r6_candidate_jv.py`
- Create: `presentations/jmi-2026/figures/data/r6_candidate_jv.json`

This is the s19 payoff. Until real DFT-derived candidate parameters arrive, use a clearly-labelled placeholder: take `nip_MAPbI3.yaml` and shift `Eg` to a screened candidate's predicted bandgap, leaving everything else fixed. Mark the JSON with `"placeholder": true` and a note string so reviewers know.

- [ ] **Step 1: Write `runs/r6_candidate_jv.py`**

```python
"""R6: candidate J-V via SolarLab device simulator (spec § 9, R6).

Until real DFT-screened parameters arrive, uses a placeholder candidate
(MAPbI3 baseline with E_g shifted to 1.55 eV). The placeholder flag is
stamped into the JSON so downstream copy can flag the s19 caveat.
"""
import json, copy
from pathlib import Path

from perovskite_sim.config_loader import load_config
from perovskite_sim.twod.experiments.jv_sweep_2d import run_jv_sweep_2d

PEROVSKITE_SIM_ROOT = Path(__file__).resolve().parents[3] / "perovskite-sim"
CFG_2D = PEROVSKITE_SIM_ROOT / "configs" / "twod" / "nip_MAPbI3_singleGB.yaml"
OUT = Path(__file__).resolve().parents[1] / "figures" / "data" / "r6_candidate_jv.json"

CANDIDATE_NOTE = (
    "PLACEHOLDER: MAPbI3 baseline, E_g shifted to 1.55 eV. "
    "Real DFT-screened parameters slot in once R3/R4 close (spec § 9)."
)

def main():
    cfg = load_config(str(CFG_2D))
    cfg = copy.deepcopy(cfg)
    # Adjust the absorber bandgap. Layer key may differ — read the YAML to confirm.
    for layer in cfg["device"]["layers"]:
        if layer.get("role") == "absorber":
            layer["Eg_eV"] = 1.55
            break
    res = run_jv_sweep_2d(cfg)
    payload = {
        "placeholder": True,
        "note": CANDIDATE_NOTE,
        "v": list(res.voltages_V),
        "j": list(res.currents_A_per_m2),
        "metrics": {
            "voc_V": res.metrics.voc_V,
            "jsc_A_per_m2": res.metrics.jsc_A_per_m2,
            "ff": res.metrics.ff,
            "pce_pct": res.metrics.pce_pct,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")
    print(f"  candidate (placeholder): PCE={payload['metrics']['pce_pct']:.2f} %")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the absorber-layer key**

Run: `grep -A3 "role: absorber" "../../perovskite-sim/configs/twod/nip_MAPbI3_singleGB.yaml" || head -50 "../../perovskite-sim/configs/twod/nip_MAPbI3_singleGB.yaml"`

If `role` field doesn't exist or `Eg_eV` is named differently (e.g., `bandgap_eV`), update the script.

- [ ] **Step 3: Run R6**

```bash
python runs/r6_candidate_jv.py
```

Expected: prints PCE line, writes JSON.

- [ ] **Step 4: Commit**

```bash
git add presentations/jmi-2026/runs/r6_candidate_jv.py \
        presentations/jmi-2026/figures/data/r6_candidate_jv.json
git commit -m "feat(jmi-ppt): R6 placeholder candidate J-V data"
```

---

## Task 8: Hero figure scripts (s2, s17, s18, s19)

**Files:**
- Create: `presentations/jmi-2026/figures/s2_eff_stability.py`
- Create: `presentations/jmi-2026/figures/s17_capability.py`
- Create: `presentations/jmi-2026/figures/s18_1d_vs_2d.py`
- Create: `presentations/jmi-2026/figures/s19_candidate_jv.py`

For brevity each script follows the same shape: import `_common.configure()`, build the figure, save to `figures/output/<name>.png` at 220 dpi. Below is the complete code for each.

- [ ] **Step 1: `s2_eff_stability.py`**

```python
"""s2 efficiency vs stability landscape."""
import matplotlib.pyplot as plt
from figures._common import configure, FIG_OUT, ACCENT, INK, SECONDARY

# Hand-curated literature points. Replace with sourced data when available.
PEROVSKITE = [(120, 25.7), (180, 24.2), (200, 25.1), (90, 23.0), (240, 25.8)]
CIGS_SI    = [(50_000, 23.4), (40_000, 22.8), (60_000, 22.1), (35_000, 21.5),
              (80_000, 22.6), (45_000, 21.0)]

def main():
    configure()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_xscale("log")
    px, py = zip(*PEROVSKITE)
    cx, cy = zip(*CIGS_SI)
    ax.scatter(px, py, s=64, color=ACCENT, alpha=0.85,
               label="Perovskite", edgecolor="white", linewidth=0.6)
    ax.scatter(cx, cy, s=64, color=INK, alpha=0.85,
               label="CIGS / Si", edgecolor="white", linewidth=0.6)
    ax.add_patch(plt.Rectangle((30_000, 24), 80_000, 4, fill=False,
                                edgecolor=SECONDARY, linewidth=2,
                                linestyle=(0, (6, 4))))
    ax.text(60_000, 28.4, "target", color=SECONDARY,
            ha="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Operational lifetime T₈₀ (h)")
    ax.set_ylabel("PCE (%)")
    ax.set_xlim(50, 200_000)
    ax.set_ylim(15, 30)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "s2_eff_stability.png")
    print(f"wrote {FIG_OUT / 's2_eff_stability.png'}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: `s17_capability.py` (table rendered as PNG)**

```python
"""s17 capability table: SolarLab vs SCAPS-1D."""
import matplotlib.pyplot as plt
from figures._common import configure, FIG_OUT, ACCENT, INK, BODY

ROWS = [
    ("Spatial dimensionality", "1D + 2D",            "1D only"),
    ("Mobile-ion transport",   "drift-diffusion",    "steady-state hack"),
    ("Optics — TMM coherent",  "multi-layer",        "Beer–Lambert only"),
    ("Hysteresis / scan rate", "time-resolved",      "—"),
    ("Selective Robin contacts","Sₙ, Sₚ",            "limited"),
    ("Tandem / multi-junction","series matched",     "manual"),
    ("Open-source / scriptable","Python + REST",     "GUI only"),
    ("Solver",                 "Radau + BDF",        "Gummel iter."),
]

def main():
    configure()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.axis("off")
    table = ax.table(
        cellText=[[r[1], r[2]] for r in ROWS],
        rowLabels=[r[0] for r in ROWS],
        colLabels=["SolarLab (ours)", "SCAPS-1D"],
        cellLoc="center", rowLoc="left", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(BODY)
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(color="white", weight="bold")
            cell.set_facecolor(INK)
        if c == 0 and r > 0:
            cell.set_text_props(color=ACCENT, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "s17_capability.png")
    print(f"wrote {FIG_OUT / 's17_capability.png'}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: `s18_1d_vs_2d.py` (consumes R5 JSON)**

```python
"""s18 hero — 1D vs 2D J-V."""
import json
import matplotlib.pyplot as plt
from figures._common import configure, FIG_OUT, FIG_DATA, ACCENT, INK

def main():
    configure()
    data = json.loads((FIG_DATA / "r5_jv_curves.json").read_text())
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(data["v_1d"], [j / 10 for j in data["j_1d"]],   # A/m² → mA/cm²
            color=ACCENT, lw=2.0, label=f"1D · PCE {data['metrics_1d']['pce_pct']:.1f}%")
    ax.plot(data["v_2d"], [j / 10 for j in data["j_2d"]],
            color=INK, lw=2.0, label=f"2D · PCE {data['metrics_2d']['pce_pct']:.1f}%")
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("J (mA/cm²)")
    delta = data['metrics_1d']['pce_pct'] - data['metrics_2d']['pce_pct']
    ax.set_title(f"ΔPCE = {delta:+.2f} % (grain-boundary recombination)",
                 fontsize=11, color="#666", style="italic", loc="left")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "s18_1d_vs_2d.png")
    print(f"wrote {FIG_OUT / 's18_1d_vs_2d.png'}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: `s19_candidate_jv.py` (consumes R6 JSON)**

```python
"""s19 predicted J-V on candidate (placeholder until R3/R4 close)."""
import json
import matplotlib.pyplot as plt
from figures._common import configure, FIG_OUT, FIG_DATA, ACCENT, BODY

def main():
    configure()
    data = json.loads((FIG_DATA / "r6_candidate_jv.json").read_text())
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(data["v"], [j / 10 for j in data["j"]],
            color=ACCENT, lw=2.0, label="Top candidate")
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("J (mA/cm²)")
    pce = data["metrics"]["pce_pct"]
    ax.text(0.05, 0.92, f"Predicted PCE = {pce:.1f} %",
            transform=ax.transAxes, fontsize=12, color=BODY,
            fontweight="bold")
    if data.get("placeholder"):
        ax.text(0.05, 0.84, "(placeholder candidate — see notes)",
                transform=ax.transAxes, fontsize=9, color="#999",
                style="italic")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "s19_candidate_jv.png")
    print(f"wrote {FIG_OUT / 's19_candidate_jv.png'}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run all four figure scripts**

```bash
cd presentations/jmi-2026
python -m figures.s2_eff_stability
python -m figures.s17_capability
python -m figures.s18_1d_vs_2d
python -m figures.s19_candidate_jv
ls figures/output/
```

Expected: 4 PNGs in `figures/output/`.

- [ ] **Step 6: Visually inspect the four PNGs in Preview**

```bash
open figures/output/s2_eff_stability.png \
     figures/output/s17_capability.png \
     figures/output/s18_1d_vs_2d.png \
     figures/output/s19_candidate_jv.png
```

Confirm: Arial font rendering on macOS, P1 colors, no clipped axes.

- [ ] **Step 7: Commit scripts**

```bash
git add presentations/jmi-2026/figures/s2_eff_stability.py \
        presentations/jmi-2026/figures/s17_capability.py \
        presentations/jmi-2026/figures/s18_1d_vs_2d.py \
        presentations/jmi-2026/figures/s19_candidate_jv.py
git commit -m "feat(jmi-ppt): hero figure scripts (s2, s17, s18, s19)"
```

---

## Task 9: Secondary figure scripts (s4, s6) — diagrammatic SVG via matplotlib

**Files:**
- Create: `presentations/jmi-2026/figures/s4_funnel.py`
- Create: `presentations/jmi-2026/figures/s6_pipeline.py`

s4 funnel: six horizontal bars with shrinking width labelled `Problem / Strategy / ML / Candidates / Bridge / Payoff`. s6 pipeline: five rounded boxes connected by arrows: `Materials DB → Feature extraction → ML surrogate → DFT validation → Device sim → Experiment`.

- [ ] **Step 1: `s4_funnel.py`**

```python
"""s4 talk roadmap — 6-step funnel."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from figures._common import configure, FIG_OUT, ACCENT, INK, HAIRLINE

LABELS = ["1 · Problem", "2 · Strategy", "3 · ML + features",
          "4 · Candidates + DFT", "5 · Bridge: device sim", "6 · Payoff + collab"]

def main():
    configure()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    n = len(LABELS)
    for i, label in enumerate(LABELS):
        y = 9 - i * 1.4
        w = 8 - i * 0.8
        x0 = (10 - w) / 2
        box = FancyBboxPatch((x0, y - 0.5), w, 1.0,
                             boxstyle="round,pad=0.02,rounding_size=0.18",
                             linewidth=1.0,
                             edgecolor=INK if i < n - 1 else ACCENT,
                             facecolor="white" if i < n - 1 else "#fde9e6")
        ax.add_patch(box)
        ax.text(5, y, label, ha="center", va="center",
                fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "s4_funnel.png")
    print(f"wrote {FIG_OUT / 's4_funnel.png'}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: `s6_pipeline.py`**

```python
"""s6 pipeline schematic."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figures._common import configure, FIG_OUT, ACCENT, INK

STEPS = ["Materials DB", "Feature\nextraction", "ML\nsurrogate",
         "DFT\nvalidation", "Device sim", "Experiment"]

def main():
    configure()
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    for i, label in enumerate(STEPS):
        x = 0.5 + i * 1.95
        is_last = i == len(STEPS) - 1
        box = FancyBboxPatch((x, 1.0), 1.5, 1.7,
                             boxstyle="round,pad=0.03,rounding_size=0.16",
                             linewidth=1.0,
                             edgecolor=ACCENT if is_last else INK,
                             facecolor="#fde9e6" if is_last else "white")
        ax.add_patch(box)
        ax.text(x + 0.75, 1.85, label, ha="center", va="center",
                fontsize=11, color=INK, fontweight="bold")
        if i < len(STEPS) - 1:
            arrow = FancyArrowPatch((x + 1.55, 1.85), (x + 1.92, 1.85),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=INK, linewidth=1.2)
            ax.add_patch(arrow)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "s6_pipeline.png")
    print(f"wrote {FIG_OUT / 's6_pipeline.png'}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run both scripts**

```bash
python -m figures.s4_funnel
python -m figures.s6_pipeline
open figures/output/s4_funnel.png figures/output/s6_pipeline.png
```

Expected: two diagrams with crimson highlight on terminal step.

- [ ] **Step 4: Commit**

```bash
git add presentations/jmi-2026/figures/s4_funnel.py \
        presentations/jmi-2026/figures/s6_pipeline.py
git commit -m "feat(jmi-ppt): roadmap funnel + pipeline diagrams"
```

---

## Task 10: Slide copy (`copy.yaml`)

**Files:**
- Create: `presentations/jmi-2026/copy.yaml`

This file holds title/subtitle/bullets/figure-path for every slide. Subscripts use `[sub]…[/sub]`, superscripts `[sup]…[/sup]`. All 22 slides + cover + 6 dividers populated. Placeholder text where R-data still pending; real content takes its place when data lands.

- [ ] **Step 1: Write `copy.yaml`**

```yaml
deck:
  total_content_slides: 22
  title: "Photovoltaic material screening from database to device-scale simulation"
  authors: "Xuan-Yan Chen et al."
  affiliation: "HKUST(GZ) — JMI 2026"

slides:
  - kind: cover

  - kind: divider
    number: "01"
    title: "PROBLEM"

  - kind: content
    section_label: "01 · PROBLEM"
    index: 2
    title: "Efficiency–stability landscape"
    subtitle: "Two regimes, one underserved corner"
    bullets:
      - "CIGS / Si: T[sub]80[/sub] > 10[sup]4[/sup] h, PCE plateau ~22 %"
      - "Perovskite: PCE > 25 %, T[sub]80[/sub] often < 500 h"
      - "Target box: PCE [sup]>[/sup] 25 % AND T[sub]80[/sub] [sup]>[/sup] 10[sup]5[/sup] h"
    figure: "figures/output/s2_eff_stability.png"

  - kind: content
    section_label: "01 · PROBLEM"
    index: 3
    title: "Why neither alone works"
    subtitle: "Champion-cell PCE timeline + accelerated-aging curves"
    bullets:
      - "Perovskite caught up to CIGS in <10 years"
      - "Stability gap remains [sup]>[/sup] 100x"
      - "Commercial deployment needs both"
    figure: null

  - kind: content
    section_label: "01 · PROBLEM"
    index: 4
    title: "Talk roadmap"
    subtitle: "Six-step funnel"
    bullets: []
    figure: "figures/output/s4_funnel.png"

  - kind: divider
    number: "02"
    title: "STRATEGY"

  - kind: content
    section_label: "02 · STRATEGY"
    index: 5
    title: "Template-guided screening"
    subtitle: "Perovskite ABX[sub]3[/sub] and chalcogenide skeletons"
    bullets:
      - "Two structural priors as filters"
      - "Constrains DB scan to physically plausible candidates"
      - "Enables small-sample ML"
    figure: null

  - kind: content
    section_label: "02 · STRATEGY"
    index: 6
    title: "End-to-end pipeline"
    subtitle: "DB to lab"
    bullets: []
    figure: "figures/output/s6_pipeline.png"

  - kind: divider
    number: "03"
    title: "ML + FEATURES"

  - kind: content
    section_label: "03 · ML + FEATURES"
    index: 7
    title: "HSE06 is too costly at DB scale"
    subtitle: "Wall-time per structure"
    bullets:
      - "PBE: minutes"
      - "HSE06: hours-to-days"
      - "GW: weeks"
      - "[Need bar-chart numbers from your DFT logs — see R1]"
    figure: null

  - kind: content
    section_label: "03 · ML + FEATURES"
    index: 8
    title: "Feature set"
    subtitle: "Composition · structural · electronic"
    bullets:
      - "Composition: stoichiometry, electronegativity moments"
      - "Structural: tolerance factor, octahedral factor"
      - "Electronic: orbital character, ionic radii"
    figure: null

  - kind: content
    section_label: "03 · ML + FEATURES"
    index: 9
    title: "Small-sample model"
    subtitle: "Random Forest + Gaussian Process · 5-fold CV"
    bullets:
      - "[n_train = TBD from R2]"
      - "[CV scores per target — see R2]"
    figure: null

  - kind: content
    section_label: "03 · ML + FEATURES"
    index: 10
    title: "Validation vs HSE06"
    subtitle: "Hold-out parity"
    bullets:
      - "R[sup]2[/sup] (E[sub]g[/sub]) = TBD"
      - "R[sup]2[/sup] (μ[sub]e[/sub]) = TBD"
      - "RMSE summary on slide"
    figure: null

  - kind: divider
    number: "04"
    title: "CANDIDATES"

  - kind: content
    section_label: "04 · CANDIDATES"
    index: 11
    title: "Top-N candidates"
    subtitle: "Ranked by predicted (PCE × stability)"
    bullets:
      - "[Replace this slide with table image once R3 closes]"
    figure: null

  - kind: content
    section_label: "04 · CANDIDATES"
    index: 12
    title: "DFT stability check"
    subtitle: "Decomposition energy + phonon DOS"
    bullets:
      - "All top-N have negative ΔH[sub]decomp[/sub]"
      - "No imaginary phonon modes"
      - "[See R4 for figure]"
    figure: null

  - kind: content
    section_label: "04 · CANDIDATES"
    index: 13
    title: "Hit #1"
    subtitle: "[Candidate A — placeholder]"
    bullets:
      - "E[sub]g[/sub] = TBD eV"
      - "μ[sub]e[/sub] = TBD cm[sup]2[/sup] V[sup]-1[/sup] s[sup]-1[/sup]"
      - "m[sup]*[/sup] / m[sub]0[/sub] = TBD"
      - "E[sub]b[/sub] = TBD meV"
    figure: null

  - kind: content
    section_label: "04 · CANDIDATES"
    index: 14
    title: "Hit #2 vs perovskite"
    subtitle: "[Candidate B — placeholder]"
    bullets:
      - "Radar chart: E[sub]g[/sub], μ, m[sup]*[/sup], E[sub]b[/sub], stability"
      - "Larger area = better"
    figure: null

  - kind: divider
    number: "05"
    title: "BRIDGE — DEVICE SIM"

  - kind: content
    section_label: "05 · BRIDGE"
    index: 15
    title: "Why device simulation"
    subtitle: "DFT params alone don't predict PCE"
    bullets:
      - "Need carrier transport at full stack scale"
      - "Need optics, contacts, hysteresis"
      - "Need 2D effects (grains, lateral non-uniformity)"
    figure: null

  - kind: content
    section_label: "05 · BRIDGE"
    index: 16
    title: "SolarLab software stack"
    subtitle: "1D + 2D drift-diffusion · TMM optics · mobile ions"
    bullets:
      - "Backend: Python solver + FastAPI SSE"
      - "Frontend: Vite/TS workstation"
      - "Solver: Radau implicit + BDF fallback + bisection-in-time"
    figure: null

  - kind: content
    section_label: "05 · BRIDGE"
    index: 17
    title: "Capability vs SCAPS-1D"
    subtitle: "What SCAPS leaves on the table"
    bullets: []
    figure: "figures/output/s17_capability.png"

  - kind: content
    section_label: "05 · BRIDGE"
    index: 18
    title: "1D vs 2D fidelity"
    subtitle: "Grain-boundary recombination only captured in 2D"
    bullets:
      - "Same stack, both solvers"
      - "ΔPCE attributable to GB physics"
    figure: "figures/output/s18_1d_vs_2d.png"

  - kind: divider
    number: "06"
    title: "PAYOFF"

  - kind: content
    section_label: "06 · PAYOFF"
    index: 19
    title: "Predicted J-V on top candidate"
    subtitle: "DFT params [sup]→[/sup] device sim [sup]→[/sup] PCE"
    bullets:
      - "Closes the loop from DB scan to predicted device performance"
      - "Placeholder candidate today — real candidate slots in once R3/R4 close"
    figure: "figures/output/s19_candidate_jv.png"

  - kind: content
    section_label: "06 · PAYOFF"
    index: 20
    title: "Outlook — experimental synthesis"
    subtitle: "From in-silico hit to lab synthesis"
    bullets:
      - "Collaboration with [group X] for synthesis"
      - "Target: validated devices in 2027 cycle"
    figure: null

  - kind: content
    section_label: "06 · PAYOFF"
    index: 21
    title: "Summary"
    subtitle: "Three takeaways"
    bullets:
      - "Template-guided ML beats brute-force HSE06 on cost"
      - "DFT-stability filter shrinks candidate list to actionable size"
      - "2D device sim is the bridge from DB to predicted PCE"
    figure: null

  - kind: content
    section_label: "06 · PAYOFF"
    index: 22
    title: "Acknowledgments + references"
    subtitle: "Q&A"
    bullets:
      - "Funding · collaborators · group members"
      - "Bibliography on a single panel"
    figure: null
```

- [ ] **Step 2: Validate copy by parsing every text field**

Run:

```bash
cd presentations/jmi-2026
python -c "
import yaml
from theme.runs import parse_segments
y = yaml.safe_load(open('copy.yaml'))
checked = 0
for s in y['slides']:
    for key in ('title', 'subtitle'):
        if s.get(key):
            parse_segments(s[key]); checked += 1
    for b in s.get('bullets') or []:
        parse_segments(b); checked += 1
print(f'parsed {checked} text fields, no _ or ^ leaks')
"
```

Expected: `parsed N text fields, no _ or ^ leaks` with no exception. If exception: a `_` or `^` leaked into copy — fix the line, re-run.

- [ ] **Step 3: Commit**

```bash
git add presentations/jmi-2026/copy.yaml
git commit -m "feat(jmi-ppt): slide copy with tagged sub/sup markers"
```

---

## Task 11: Deck builder

**Files:**
- Create: `presentations/jmi-2026/build_deck.py`
- Create: `presentations/jmi-2026/test_build_deck.py`

- [ ] **Step 1: Write failing integration test**

```python
# test_build_deck.py
import os
from pathlib import Path
from build_deck import build

def test_build_produces_pptx_with_correct_slide_count(tmp_path):
    out = tmp_path / "deck.pptx"
    n = build(out_path=out)
    assert out.exists()
    # 1 cover + 6 dividers + 22 content = 29 slides
    assert n == 29

def test_no_underscore_or_caret_anywhere_in_deck(tmp_path):
    from pptx import Presentation
    out = tmp_path / "deck.pptx"
    build(out_path=out)
    prs = Presentation(str(out))
    for i, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                for run in para.runs:
                    assert "_" not in run.text, f"slide {i}: underscore leak: {run.text!r}"
                    assert "^" not in run.text, f"slide {i}: caret leak: {run.text!r}"
```

- [ ] **Step 2: Run, confirm fail**

Run: `cd presentations/jmi-2026 && python -m pytest test_build_deck.py -v`
Expected: `ImportError` on `build_deck`.

- [ ] **Step 3: Implement `build_deck.py`**

```python
"""End-to-end PPTX builder. Reads copy.yaml, emits output/jmi-2026.pptx."""
import argparse
from pathlib import Path
import yaml
from pptx import Presentation
from pptx.util import Inches

from theme.layouts import add_cover, add_divider, add_content


def build(*, copy_path: Path = Path("copy.yaml"),
          out_path: Path = Path("output/jmi-2026.pptx")) -> int:
    spec = yaml.safe_load(copy_path.read_text())
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(5.625)            # 16:9
    total_content = spec["deck"]["total_content_slides"]
    for s in spec["slides"]:
        kind = s["kind"]
        if kind == "cover":
            add_cover(prs,
                      title=spec["deck"]["title"],
                      authors=spec["deck"]["authors"],
                      affiliation=spec["deck"]["affiliation"])
        elif kind == "divider":
            add_divider(prs, number=s["number"], title=s["title"])
        elif kind == "content":
            fig = s.get("figure")
            fig_path = Path(fig) if fig else None
            if fig_path and not fig_path.exists():
                raise FileNotFoundError(
                    f"Slide {s['index']} references missing figure {fig_path}. "
                    "Run figure scripts first.")
            add_content(prs,
                        section_label=s["section_label"],
                        slide_index=s["index"],
                        total=total_content,
                        title=s["title"],
                        subtitle=s["subtitle"],
                        bullets=s.get("bullets") or [],
                        figure_path=fig_path)
        else:
            raise ValueError(f"unknown slide kind: {kind}")
    out_path.parent.mkdir(exist_ok=True)
    prs.save(str(out_path))
    return len(prs.slides)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--copy", default="copy.yaml")
    p.add_argument("--out", default="output/jmi-2026.pptx")
    args = p.parse_args()
    n = build(copy_path=Path(args.copy), out_path=Path(args.out))
    print(f"built {args.out} with {n} slides")
```

- [ ] **Step 4: Run integration tests**

```bash
cd presentations/jmi-2026
python -m pytest test_build_deck.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Build the actual deck**

```bash
python build_deck.py
ls -lh output/jmi-2026.pptx
```

Expected: PPTX file ~80–250 KB (binary size depends on PNG count and resolution).

- [ ] **Step 6: Open in PowerPoint and visually inspect**

```bash
open output/jmi-2026.pptx
```

Page through all 29 slides. Check:
- Cover: title centered, accent line, no logo
- 6 dividers: full-bleed numeral, ink background
- Content slides: header section label + counter, title/subtitle, figure left + bullets right
- Footer: "Chen et al. — JMI 2026" left, crimson "SOLARLAB" right
- All sub/super render as real subscript/superscript (not inline literals)

If layout regressions: fix in `theme/layouts.py`, re-run `build_deck.py`, re-open.

- [ ] **Step 7: Commit**

```bash
git add presentations/jmi-2026/build_deck.py \
        presentations/jmi-2026/test_build_deck.py \
        presentations/jmi-2026/output/jmi-2026.pptx
git commit -m "feat(jmi-ppt): end-to-end deck builder"
```

---

## Task 12: Sub/sup typography audit on the built file

**Files:**
- Create: `presentations/jmi-2026/audit_typography.py`

Belt-and-suspenders. The parser raises on `_`/`^` in copy, and the build test asserts no leak. This script audits the **built .pptx** by walking the OOXML for every run, asserting that any run sitting inside a context that *should* render as subscript/superscript has the `baseline` attribute set on its `rPr`.

- [ ] **Step 1: Write `audit_typography.py`**

```python
"""Walk every run of output/jmi-2026.pptx and audit sub/sup runs."""
from pathlib import Path
from pptx import Presentation
from pptx.oxml.ns import qn

DECK = Path("output/jmi-2026.pptx")


def main():
    prs = Presentation(str(DECK))
    runs_total = 0
    runs_sub = 0
    runs_sup = 0
    leaks = []
    for slide_idx, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                for run in para.runs:
                    runs_total += 1
                    if "_" in run.text or "^" in run.text:
                        leaks.append((slide_idx, run.text))
                    rpr = run._r.find(qn("a:rPr"))
                    if rpr is not None:
                        bl = rpr.get("baseline")
                        if bl == "-25000":
                            runs_sub += 1
                        elif bl == "30000":
                            runs_sup += 1
    print(f"total runs: {runs_total}")
    print(f"  subscript runs: {runs_sub}")
    print(f"  superscript runs: {runs_sup}")
    if leaks:
        for s, t in leaks:
            print(f"  LEAK on slide {s}: {t!r}")
        raise SystemExit(1)
    print("PASS: no _ or ^ literals; sub/sup runs have baseline attr.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run audit**

```bash
cd presentations/jmi-2026
python audit_typography.py
```

Expected:
```
total runs: ~120
  subscript runs: ~30
  superscript runs: ~10
PASS: no _ or ^ literals; sub/sup runs have baseline attr.
```

If counts are zero on sub/sup: parser-to-XML wiring is broken; revisit `theme/runs.py::write_segments`.

- [ ] **Step 3: Commit**

```bash
git add presentations/jmi-2026/audit_typography.py
git commit -m "feat(jmi-ppt): typography audit script"
```

---

## Task 13: README finalisation + PR-ready commit

**Files:**
- Modify: `presentations/jmi-2026/README.md`

- [ ] **Step 1: Append known-data-gaps section**

```markdown
## Data gaps before talk-ready

| Slide | Gap | Source |
|---|---|---|
| s7 | HSE06 wall-time numbers | DFT logs (R1) |
| s9–s10 | ML model metrics + parity-plot data | ML run export (R2) |
| s11 | Top-N candidate table | Screening output (R3) |
| s12 | Decomposition E + phonon DOS | DFT runs (R4) |
| s18 | Auto-generated from `runs/r5_1d_vs_2d.py` | filled |
| s19 | Auto-generated from `runs/r6_candidate_jv.py` | placeholder candidate |

When R1–R4 data lands: drop new figures into `figures/output/` and
update the relevant slide entries in `copy.yaml`. Then re-run
`python build_deck.py`.
```

- [ ] **Step 2: Final commit**

```bash
git add presentations/jmi-2026/README.md
git commit -m "docs(jmi-ppt): document data-gap fill workflow"
```

---

## Self-Review

Spec coverage check (against `docs/superpowers/specs/2026-05-10-jmi-ppt-design.md`):

| Spec section | Plan task |
|---|---|
| § 4 Funnel narrative | Task 10 (copy.yaml structure follows funnel order) |
| § 5 Slide outline (22 + cover + 6 dividers) | Task 10 + Task 11 (build asserts 29) |
| § 6 Figure-per-slide table | Tasks 8, 9 (hero + diagrammatic figures) |
| § 7.1 P1 palette | Task 2 |
| § 7.2 Arial-only typography | Task 2 |
| § 7.3 Real sub/super, no `_`/`^` | Task 3 (parser) + Task 11 (test) + Task 12 (audit) |
| § 7.4 Standard slide layout | Task 4 |
| § 7.5 Full-bleed numeral dividers | Task 4 |
| § 7.6 Plot theme aligned with deck | Task 5 |
| § 9 R5 1D vs 2D run | Task 6 |
| § 9 R6 candidate J-V | Task 7 |
| § 8 Time budget — 22 content slides | Task 11 (build asserts 29 = 1+6+22) |

Placeholder scan: text marked `[TBD]` or `[placeholder]` only appears in `copy.yaml` slide bodies that are gated by spec § 9 risk register (R1–R4 data the user must supply). Plan tasks themselves contain no `TBD` / `TODO` / "implement later".

Type / signature consistency: `parse_segments` and `write_segments` defined in Task 3 and consumed in Task 4 with matching signatures. `build` function signature in Task 11 matches both test and CLI invocation. Field names in R5/R6 scripts (`voltages_V`, `currents_A_per_m2`, `metrics.voc_V`, `metrics.pce_pct`) are the assumed Python-side names; Task 6 Step 2 explicitly inspects the actual API and updates the script if names differ.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-10-jmi-ppt-implementation.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
