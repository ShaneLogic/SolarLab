# Paper-Reproduction Validation Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current trend-heavy validation figure with a paper-reproduction comparison figure that clearly shows paper values, our simulated values, deltas, and model-mismatch causes for Courtier 2019 / IonMonger and Calado 2016 / Driftfusion.

**Architecture:** Keep the work centered in the existing validation script and validation tests. Add small data structures inside `scripts/plot_paper_validation.py` to separate simulation collection, paper-reference metrics, comparison calculations, and plotting. Keep the figure honest: exact reproduction is not claimed when the solver physics differs from the paper model.

**Tech Stack:** Python, dataclasses, NumPy, SciPy `linregress`, Matplotlib Agg backend, pytest validation tests, existing `perovskite_sim` J-V sweep API.

---

## File Structure

- Modify: `scripts/plot_paper_validation.py`
  - Responsibility: collect paper-like simulation runs, compute comparison metrics, and render the final Arial paper-comparison figure.
  - Keep in one file because this is a standalone script, but split logic into focused helpers.
- Modify: `perovskite-sim/tests/validation/test_paper_configurations.py`
  - Responsibility: keep validation assertions and wording aligned with the paper-comparison interpretation.
- Modify: `perovskite-sim/tests/validation/test_physical_trends.py`
  - Responsibility: fix the natural-log vs decade slope wording if the existing trend test remains.
- Generated output: `perovskite-sim/paper_validation.png`
  - Responsibility: final 300 dpi figure for visual inspection.
- Optional generated output: `perovskite-sim/paper_validation.pdf`
  - Responsibility: vector export for manuscript/slides if Matplotlib PDF export works with Arial fallback.

---

### Task 1: Add explicit paper-reference comparison tests

**Files:**
- Modify: `perovskite-sim/tests/validation/test_paper_configurations.py`

- [ ] **Step 1: Add paper reference constants near the imports**

Add this after `pytestmark = pytest.mark.validation`:

```python
IONMONGER_PAPER = {
    "source": "Courtier 2019 set (b)",
    "V_oc": 1.07,
    "J_sc": 220.0,
    "FF_min": 0.70,
    "FF_max": 0.80,
}

DRIFTFUSION_PAPER = {
    "source": "Calado 2016 spiro/MAPbI3/TiO2",
    "V_oc_min": 1.00,
    "V_oc_max": 1.10,
    "J_sc": 220.0,
}
```

- [ ] **Step 2: Add a failing test for IonMonger comparison facts**

Add this after `test_ionmonger_ff_in_paper_range`:

```python
def test_ionmonger_paper_comparison_quantifies_known_voc_offset(
    ionmonger_result: JVResult,
) -> None:
    """Paper-comparison figure must treat high V_oc as a known model mismatch."""
    paper_voc = IONMONGER_PAPER["V_oc"]
    ours_voc = ionmonger_result.metrics_rev.V_oc
    delta_voc = ours_voc - paper_voc

    assert 0.08 <= delta_voc <= 0.18, (
        f"IonMonger V_oc offset must be explicitly reported as a model mismatch: "
        f"paper={paper_voc:.3f} V, ours={ours_voc:.3f} V, "
        f"delta={delta_voc * 1e3:.0f} mV"
    )
```

- [ ] **Step 3: Add a failing test for Driftfusion comparison facts**

Add this after `test_driftfusion_voc_in_expected_range`:

```python
def test_driftfusion_paper_comparison_quantifies_low_voc_offset(
    driftfusion_result: JVResult,
) -> None:
    """Paper-comparison figure must treat low V_oc as a known model mismatch."""
    paper_mid_voc = 0.5 * (
        DRIFTFUSION_PAPER["V_oc_min"] + DRIFTFUSION_PAPER["V_oc_max"]
    )
    ours_voc = driftfusion_result.metrics_rev.V_oc
    delta_voc = ours_voc - paper_mid_voc

    assert -0.50 <= delta_voc <= -0.25, (
        f"Driftfusion V_oc offset must be explicitly reported as a model mismatch: "
        f"paper_mid={paper_mid_voc:.3f} V, ours={ours_voc:.3f} V, "
        f"delta={delta_voc * 1e3:.0f} mV"
    )
```

- [ ] **Step 4: Run the new targeted tests and verify failure or pass-with-current-code**

Run from `perovskite-sim/`:

```bash
pytest tests/validation/test_paper_configurations.py::test_ionmonger_paper_comparison_quantifies_known_voc_offset tests/validation/test_paper_configurations.py::test_driftfusion_paper_comparison_quantifies_low_voc_offset -v
```

Expected: PASS if current simulated values still match the known offsets. If either fails, stop and inspect the actual metric values before editing figure code.

- [ ] **Step 5: Commit test baseline**

```bash
git add tests/validation/test_paper_configurations.py
git commit -m "test(validation): pin paper comparison offsets"
```

---

### Task 2: Fix natural-log slope wording in validation tests

**Files:**
- Modify: `perovskite-sim/tests/validation/test_paper_configurations.py`
- Modify: `perovskite-sim/tests/validation/test_physical_trends.py`

- [ ] **Step 1: Fix wording in `test_paper_configurations.py`**

In `test_driftfusion_illumination_slope_physical`, replace:

```python
"""dV_oc / d(ln Phi) must be in [25, 80] mV/decade.

The ideality-factor-controlled slope should be n_id · kT/q with
n_id ∈ [1.0, 3.1] for a device with both SRH and radiative recombination.
"""
```

with:

```python
"""dV_oc / d(ln Phi) must be in [25, 80] mV per natural-log unit.

The ideality-factor-controlled slope should be n_id · kT/q with
n_id ∈ [1.0, 3.1] for a device with both SRH and radiative recombination.
"""
```

And replace the assertion message fragment:

```python
f"dV_oc / d(ln Phi) = {slope_mv:.1f} mV/dec outside [25, 80] — "
```

with:

```python
f"dV_oc / d(ln Phi) = {slope_mv:.1f} mV/ln-unit outside [25, 80] — "
```

- [ ] **Step 2: Fix wording in `test_physical_trends.py`**

In `test_voc_vs_illumination`, replace this comment and calculation block:

```python
slope_mv_per_decade = slope * 1000  # V/decade → mV/decade

assert r_value > 0.95, (
    f"Suns-V_oc slope fit correlation r={r_value:.3f} too weak"
)
assert 20 <= slope_mv_per_decade <= 70, (
    f"Suns-V_oc slope {slope_mv_per_decade:.1f} mV/decade outside [20, 70]"
)
```

with:

```python
slope_mv_per_ln = slope * 1000

assert r_value > 0.95, (
    f"Suns-V_oc slope fit correlation r={r_value:.3f} too weak"
)
assert 20 <= slope_mv_per_ln <= 70, (
    f"Suns-V_oc slope {slope_mv_per_ln:.1f} mV/ln-unit outside [20, 70]"
)
```

- [ ] **Step 3: Run slope-related tests**

Run from `perovskite-sim/`:

```bash
pytest tests/validation/test_paper_configurations.py::test_driftfusion_illumination_slope_physical tests/validation/test_physical_trends.py::test_voc_vs_illumination -v
```

Expected: PASS. These are wording/variable-name fixes, not physics changes.

- [ ] **Step 4: Commit wording fix**

```bash
git add tests/validation/test_paper_configurations.py tests/validation/test_physical_trends.py
git commit -m "fix(validation): label suns-voc slope as natural-log units"
```

---

### Task 3: Refactor paper validation script into comparison data helpers

**Files:**
- Modify: `scripts/plot_paper_validation.py`

- [ ] **Step 1: Replace unused imports**

Remove these unused imports:

```python
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.constants import K_B, Q
```

Keep:

```python
from dataclasses import dataclass, replace
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.experiments.jv_sweep import run_jv_sweep, JVResult
```

- [ ] **Step 2: Add metric dataclasses after `matplotlib.use("Agg")`**

```python
@dataclass(frozen=True)
class PaperMetric:
    label: str
    paper: float | tuple[float, float]
    ours: float
    unit: str
    status: str
    cause: str


@dataclass(frozen=True)
class ComparisonBundle:
    ionmonger_bl: JVResult
    ionmonger_bl_vbi086: JVResult
    ionmonger_tmm_vbi110: JVResult
    driftfusion_flat: JVResult
    driftfusion_suns: list[tuple[float, float]]
    ionmonger_hi: list[tuple[float, float]]
    ionmonger_metrics: tuple[PaperMetric, ...]
    driftfusion_metrics: tuple[PaperMetric, ...]
```

- [ ] **Step 3: Add paper constants after dataclasses**

```python
IONMONGER_PAPER_VOC = 1.07
IONMONGER_PAPER_JSC = 22.0
IONMONGER_PAPER_FF = (0.70, 0.80)

DRIFTFUSION_PAPER_VOC = (1.00, 1.10)
DRIFTFUSION_PAPER_JSC = 22.0
DRIFTFUSION_PAPER_FF = (0.50, 0.80)
```

- [ ] **Step 4: Add formatting helpers after `_run_jv`**

```python
def _metric_delta(paper: float | tuple[float, float], ours: float) -> float:
    if isinstance(paper, tuple):
        return ours - 0.5 * (paper[0] + paper[1])
    return ours - paper


def _format_paper_value(value: float | tuple[float, float], unit: str) -> str:
    if isinstance(value, tuple):
        return f"{value[0]:.2f}-{value[1]:.2f} {unit}"
    return f"{value:.2f} {unit}"


def _format_delta(value: float, unit: str) -> str:
    if unit == "V":
        return f"{value * 1e3:+.0f} mV"
    if unit == "mA cm$^{-2}$":
        return f"{value:+.1f} mA cm$^{-2}$"
    return f"{value:+.3f}"
```

- [ ] **Step 5: Move existing data collection into `collect_comparison_data()`**

Replace the top-level data-collection statements with this helper:

```python
def _load_driftfusion_flatband() -> DeviceStack:
    df_stack = load_device_from_yaml("configs/driftfusion_benchmark.yaml")
    df_layers = []
    for layer in df_stack.layers:
        if layer.params is not None:
            df_layers.append(replace(layer, params=replace(layer.params, chi=0.0, Eg=0.0)))
        else:
            df_layers.append(layer)
    return replace(df_stack, layers=tuple(df_layers), mode="legacy")


def collect_comparison_data() -> ComparisonBundle:
    print("Loading IonMonger (Courtier 2019) BL ...")
    im_bl_stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    im_bl_result = _run_jv(im_bl_stack)

    print("Loading IonMonger BL (V_bi = 0.86)...")
    im_bl_vbi086_result = _run_jv(replace(im_bl_stack, V_bi=0.86))

    print("Loading IonMonger TMM (V_bi = 1.10, matched to BL)...")
    im_tmm_stack = load_device_from_yaml("configs/ionmonger_benchmark_tmm.yaml")
    im_tmm_vbi110_result = _run_jv(replace(im_tmm_stack, V_bi=1.10))

    print("Loading Driftfusion (Calado 2016) flat-band...")
    df_flat = _load_driftfusion_flatband()
    df_result = _run_jv(df_flat)

    print("Driftfusion V_oc vs illumination...")
    sun_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
    df_suns = []
    for s in sun_levels:
        res = _run_jv(replace(df_flat, Phi=df_flat.Phi * s))
        df_suns.append((s, res.metrics_rev.V_oc))

    print("IonMonger scan-rate hysteresis...")
    scan_rates = [0.1, 1.0, 10.0, 50.0]
    im_hi = []
    for v_rate in scan_rates:
        res = run_jv_sweep(im_bl_stack, N_grid=40, n_points=15, v_rate=v_rate, V_max=1.5)
        im_hi.append((v_rate, float(res.hysteresis_index)))

    im = im_bl_result.metrics_rev
    df = df_result.metrics_rev

    ionmonger_metrics = (
        PaperMetric(
            "V$_{oc}$",
            IONMONGER_PAPER_VOC,
            im.V_oc,
            "V",
            "high",
            "Band-offset heterostack raises QFL splitting vs IonMonger flat-interface model.",
        ),
        PaperMetric(
            "J$_{sc}$",
            IONMONGER_PAPER_JSC,
            im.J_sc / 10,
            "mA cm$^{-2}$",
            "ok",
            "Beer-Lambert photocurrent matches the paper-scale absorption envelope.",
        ),
        PaperMetric(
            "FF",
            IONMONGER_PAPER_FF,
            im.FF,
            "",
            "ok",
            "SRH-limited fill factor remains inside the reported paper range.",
        ),
    )

    driftfusion_metrics = (
        PaperMetric(
            "V$_{oc}$",
            DRIFTFUSION_PAPER_VOC,
            df.V_oc,
            "V",
            "low",
            "Flat-band legacy run is functional but differs in contact and recombination treatment.",
        ),
        PaperMetric(
            "J$_{sc}$",
            DRIFTFUSION_PAPER_JSC,
            df.J_sc / 10,
            "mA cm$^{-2}$",
            "ok",
            "Photocurrent is close to the shared Beer-Lambert absorption limit.",
        ),
        PaperMetric(
            "FF",
            DRIFTFUSION_PAPER_FF,
            df.FF,
            "",
            "ok",
            "Transport-limited fill factor remains in the broad expected range.",
        ),
    )

    return ComparisonBundle(
        im_bl_result,
        im_bl_vbi086_result,
        im_tmm_vbi110_result,
        df_result,
        df_suns,
        im_hi,
        ionmonger_metrics,
        driftfusion_metrics,
    )
```

- [ ] **Step 6: Run script smoke test**

Run from `perovskite-sim/`:

```bash
python ../scripts/plot_paper_validation.py
```

Expected before plotting rewrite: script may fail because plotting still references removed globals. If so, continue to Task 4. If it passes, it should still write `paper_validation.png`.

---

### Task 4: Render the approved paper-comparison figure

**Files:**
- Modify: `scripts/plot_paper_validation.py`

- [ ] **Step 1: Add Matplotlib style helper after formatting helpers**

```python
def _apply_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 18,
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.8,
    })
```

- [ ] **Step 2: Add J-V plotting helper**

```python
def _plot_jv_panel(ax, result: JVResult, title: str, color: str, note: str) -> None:
    V = np.asarray(result.V_rev)
    J = np.asarray(result.J_rev) / 10
    ax.plot(V, J, color=color)
    ax.axhline(0, color="0.55", ls=":", lw=0.8)
    ax.axvline(0, color="0.55", ls=":", lw=0.8)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current density (mA cm$^{-2}$)")
    ax.text(
        0.03,
        0.05,
        note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.85"},
    )
```

- [ ] **Step 3: Add metric table helper**

```python
def _draw_metric_table(ax, metrics: tuple[PaperMetric, ...], title: str) -> None:
    ax.axis("off")
    ax.set_title(title, loc="left", fontweight="bold")
    rows = []
    colors = []
    for metric in metrics:
        delta = _metric_delta(metric.paper, metric.ours)
        rows.append([
            metric.label,
            _format_paper_value(metric.paper, metric.unit),
            f"{metric.ours:.2f} {metric.unit}".strip(),
            _format_delta(delta, metric.unit),
        ])
        colors.append("#b91c1c" if metric.status in {"high", "low"} else "#047857")

    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Paper", "Ours", "Delta"],
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.36, 1.0, 0.56],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.82")
        if row == 0:
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(fontweight="bold")
        elif col == 3:
            cell.set_text_props(color=colors[row - 1], fontweight="bold")

    y = 0.26
    for metric in metrics:
        ax.text(0.0, y, f"• {metric.cause}", transform=ax.transAxes, fontsize=9, va="top")
        y -= 0.095
```

- [ ] **Step 4: Add compact auxiliary panels**

```python
def _draw_optics_panel(ax, bundle: ComparisonBundle) -> None:
    mb = bundle.ionmonger_bl.metrics_rev
    mt = bundle.ionmonger_tmm_vbi110.metrics_rev
    labels = ["V$_{oc}$", "J$_{sc}$"]
    values = [abs(mt.V_oc - mb.V_oc) * 1e3, mt.J_sc / mb.J_sc]
    ax.bar(labels, values, color=["#2563eb", "#16a34a"])
    ax.set_title("IonMonger TMM check, V$_{bi}$ fixed", loc="left", fontweight="bold")
    ax.set_ylabel("mV delta / ratio")
    ax.text(
        0.02,
        0.92,
        "Reported separately from paper reproduction because TMM adds optical-stack assumptions.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )


def _draw_suns_panel(ax, bundle: ComparisonBundle) -> None:
    suns = np.asarray([s for s, _ in bundle.driftfusion_suns])
    voc = np.asarray([v for _, v in bundle.driftfusion_suns])
    slope, _, r_val, _, _ = linregress(np.log(suns), voc)
    slope_mv_ln = slope * 1000
    ax.semilogx(suns, voc * 1000, "o-", color="#2563eb")
    ax.set_title("Driftfusion Suns-V$_{oc}$ sanity", loc="left", fontweight="bold")
    ax.set_xlabel("Illumination (suns)")
    ax.set_ylabel("V$_{oc}$ (mV)")
    ax.text(
        0.03,
        0.08,
        f"slope={slope_mv_ln:.1f} mV/ln-unit, n$_{{id}}$≈{slope_mv_ln / 25.85:.1f}, r={r_val:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.85"},
    )
```

- [ ] **Step 5: Replace plotting block with `render_figure()` and `main()`**

Replace all top-level plotting code from `print("Plotting...")` to end of file with:

```python
def render_figure(bundle: ComparisonBundle) -> None:
    print("Plotting...")
    _apply_style()
    fig = plt.figure(figsize=(15.5, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 1.05])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    im = bundle.ionmonger_bl.metrics_rev
    df = bundle.driftfusion_flat.metrics_rev

    _plot_jv_panel(
        ax1,
        bundle.ionmonger_bl,
        "A  Courtier 2019 / IonMonger paper-like run",
        "#111827",
        f"Ours: V$_{{oc}}$={im.V_oc:.3f} V, J$_{{sc}}$={im.J_sc / 10:.1f} mA cm$^{{-2}}$, FF={im.FF:.3f}",
    )
    _plot_jv_panel(
        ax2,
        bundle.driftfusion_flat,
        "B  Calado 2016 / Driftfusion flat-band run",
        "#1d4ed8",
        f"Ours: V$_{{oc}}$={df.V_oc:.3f} V, J$_{{sc}}$={df.J_sc / 10:.1f} mA cm$^{{-2}}$, FF={df.FF:.3f}",
    )
    _draw_metric_table(ax3, bundle.ionmonger_metrics + bundle.driftfusion_metrics, "C  Paper vs ours metrics")
    _draw_metric_table(ax4, bundle.ionmonger_metrics, "D  IonMonger mismatch causes")
    _draw_optics_panel(ax5, bundle)
    _draw_suns_panel(ax6, bundle)

    fig.suptitle(
        "Paper-model validation: reported values vs our drift-diffusion solver",
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "Paper references are comparison targets, not exact reproduction claims. Deviations are labeled by known model differences.",
        fontsize=9,
        color="0.35",
    )
    fig.savefig("paper_validation.png", dpi=300, bbox_inches="tight")
    fig.savefig("paper_validation.pdf", bbox_inches="tight")
    print("Saved: paper_validation.png")
    print("Saved: paper_validation.pdf")


def main() -> None:
    render_figure(collect_comparison_data())


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the script**

Run from `perovskite-sim/`:

```bash
python ../scripts/plot_paper_validation.py
```

Expected output:

```text
Loading IonMonger (Courtier 2019) BL ...
Loading IonMonger BL (V_bi = 0.86)...
Loading IonMonger TMM (V_bi = 1.10, matched to BL)...
Loading Driftfusion (Calado 2016) flat-band...
Driftfusion V_oc vs illumination...
IonMonger scan-rate hysteresis...
Plotting...
Saved: paper_validation.png
Saved: paper_validation.pdf
```

- [ ] **Step 7: Commit script refactor and figure generation**

```bash
git add ../scripts/plot_paper_validation.py paper_validation.png paper_validation.pdf
git commit -m "feat(validation): render paper comparison figure"
```

---

### Task 5: Verify visual clarity and generated artifacts

**Files:**
- Generated: `perovskite-sim/paper_validation.png`
- Generated: `perovskite-sim/paper_validation.pdf`

- [ ] **Step 1: Inspect figure dimensions and file outputs**

Run from repo root:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image
for path in [Path('perovskite-sim/paper_validation.png')]:
    im = Image.open(path)
    print(path, im.size, im.mode)
print('pdf_exists', Path('perovskite-sim/paper_validation.pdf').exists())
PY
```

Expected: PNG exists, dimensions are large enough for 300 dpi export, PDF exists.

- [ ] **Step 2: Inspect the PNG visually**

Use Read on:

```text
/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab/perovskite-sim/paper_validation.png
```

Expected: Arial-like text, no overlapping labels, readable metric table, panel titles not clipped.

- [ ] **Step 3: If Arial fallback warnings appear, confirm fallback is acceptable**

If Matplotlib logs `findfont: Font family 'Arial' not found`, keep the code as-is because it declares Arial first and falls back to Helvetica/DejaVu Sans. Do not add a bundled font file.

- [ ] **Step 4: Run validation tests**

Run from `perovskite-sim/`:

```bash
pytest tests/validation/test_paper_configurations.py tests/validation/test_physical_trends.py::test_voc_vs_illumination -v
```

Expected: PASS.

- [ ] **Step 5: Commit regenerated artifacts if Task 4 did not already commit**

```bash
git add paper_validation.png paper_validation.pdf
 git commit -m "chore(validation): regenerate paper comparison artifacts"
```

Skip this commit if Task 4 already committed both artifacts.

---

### Task 6: Independent review and final verification

**Files:**
- Review changed files from Tasks 1-5.

- [ ] **Step 1: Run code review agent**

Use the `everything-claude-code:code-reviewer` agent with this prompt:

```text
Review the paper-validation comparison changes. Focus on whether the figure honestly distinguishes paper targets from our solver output, whether metric units are correct, whether Matplotlib styling is readable, and whether tests pin the known mismatch without overclaiming exact reproduction. Report only critical/high/medium issues.
```

Expected: no critical/high issues. Fix any critical/high issue before continuing.

- [ ] **Step 2: Run Python reviewer if Python logic changed materially**

Use the `everything-claude-code:python-reviewer` agent with this prompt:

```text
Review scripts/plot_paper_validation.py and validation test edits for Python correctness, immutability, small helper boundaries, and unused imports. Report actionable issues only.
```

Expected: no critical/high issues. Fix any critical/high issue before continuing.

- [ ] **Step 3: Run final script and tests**

Run from `perovskite-sim/`:

```bash
python ../scripts/plot_paper_validation.py && pytest tests/validation/test_paper_configurations.py tests/validation/test_physical_trends.py::test_voc_vs_illumination -v
```

Expected: script saves both artifacts and tests pass.

- [ ] **Step 4: Check git status**

Run from repo root:

```bash
git status --short
```

Expected: only intended files changed or a clean working tree if all commits were made.

---

## Self-Review

**Spec coverage:**
- Paper-reproduction comparison: Tasks 1, 3, 4.
- Known model mismatches shown explicitly: Tasks 1 and 4.
- Arial and readable sizing: Task 4 and Task 5.
- Natural-log slope bug: Task 2 and Task 4.
- Verification and review: Tasks 5 and 6.

**Placeholder scan:** No TBD/TODO placeholders remain. Each code-changing step includes exact code or exact replacement text.

**Type consistency:** `PaperMetric` and `ComparisonBundle` are defined before use. Helper names are consistent across Tasks 3 and 4. All script paths are relative to the established repo layout.
