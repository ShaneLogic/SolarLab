"""Generate a paper-reproduction validation figure.

Run from perovskite-sim/:  python ../scripts/plot_paper_validation.py
Outputs: paper_validation.png and paper_validation.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.stats import linregress

from perovskite_sim.experiments.jv_sweep import JVResult, run_jv_sweep
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack


@dataclass(frozen=True)
class IonMongerPaperReference:
    """Numeric paper targets from Courtier 2019 set (b)."""

    source: str
    V_oc: float
    J_sc: float
    FF_min: float
    FF_max: float


@dataclass(frozen=True)
class DriftfusionPaperReference:
    """Numeric paper targets from Calado 2016."""

    source: str
    V_oc_min: float
    V_oc_max: float
    J_sc: float


IONMONGER_PAPER = IonMongerPaperReference(
    source="Courtier 2019 set (b)",
    V_oc=1.07,
    J_sc=220.0,
    FF_min=0.70,
    FF_max=0.80,
)
DRIFTFUSION_PAPER = DriftfusionPaperReference(
    source="Calado 2016 spiro/MAPbI3/TiO2",
    V_oc_min=1.00,
    V_oc_max=1.10,
    J_sc=220.0,
)
EXPORT_DPI = 300
BASELINE_LABEL = "SolarLab baseline model"
CALIBRATED_LABEL = "literature-aligned model"
BENCHMARK_LABEL = "published benchmark"
ARIAL_FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
ARIAL_BOLD_FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


@dataclass(frozen=True)
class PaperMetric:
    """One paper-vs-simulation metric row for the summary table."""

    device: str
    metric: str
    metric_label: str
    paper_value: float | tuple[float, float]
    ours: float
    units: str
    note: str
    delta_note: str = ""


@dataclass(frozen=True)
class ComparisonBundle:
    """All simulation data needed to render the validation figure."""

    ionmonger_default: JVResult
    ionmonger_repro: JVResult
    ionmonger_bl_vbi086: JVResult
    ionmonger_tmm_vbi110: JVResult
    driftfusion_default: JVResult
    driftfusion_repro: JVResult
    driftfusion_suns: tuple[float, ...]
    driftfusion_suns_voc: tuple[float, ...]
    metrics: tuple[PaperMetric, ...]


def _run_jv(stack: DeviceStack, *, n_grid: int = 60, n_points: int = 20) -> JVResult:
    return run_jv_sweep(
        stack,
        N_grid=n_grid,
        n_points=n_points,
        v_rate=5.0,
        V_max=1.5,
    )


def _load_driftfusion_flatband() -> DeviceStack:
    """Load Driftfusion in the flat-band LEGACY form used for paper comparison."""
    stack = load_device_from_yaml("configs/driftfusion_benchmark.yaml")
    layers = tuple(
        (
            replace(layer, params=replace(layer.params, chi=0.0, Eg=0.0))
            if layer.params is not None
            else layer
        )
        for layer in stack.layers
    )
    return replace(stack, layers=layers, mode="legacy")


def _metric_delta(metric: PaperMetric) -> float:
    """Return ours-minus-paper delta, using midpoint for paper ranges."""
    if isinstance(metric.paper_value, tuple):
        paper = 0.5 * (metric.paper_value[0] + metric.paper_value[1])
    else:
        paper = metric.paper_value
    return metric.ours - paper


def _format_paper_value(value: float | tuple[float, float], units: str) -> str:
    """Format a scalar or range paper value for the comparison table."""
    suffix = f" {units}" if units else ""
    if isinstance(value, tuple):
        return f"{value[0]:.2f}–{value[1]:.2f}{suffix}"
    if abs(value) >= 10:
        return f"{value:.1f}{suffix}"
    return f"{value:.2f}{suffix}"


def _format_delta(delta: float, units: str) -> str:
    """Format ours-minus-paper delta with an explicit sign."""
    suffix = f" {units}" if units else ""
    if units == "V":
        return f"{delta:+.3f}{suffix}"
    if units == "A/m²":
        return f"{delta:+.1f}{suffix}"
    return f"{delta:+.3f}{suffix}"


def collect_comparison_data() -> ComparisonBundle:
    """Run the paper-reproduction simulations and package plotting inputs."""
    print("Loading IonMonger default physical benchmark ...")
    ionmonger_stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    ionmonger_default = _run_jv(ionmonger_stack)

    print("Loading IonMonger Courtier 2019 reproduction preset ...")
    ionmonger_repro_stack = load_device_from_yaml(
        "configs/ionmonger_courtier2019_repro.yaml"
    )
    ionmonger_repro = _run_jv(ionmonger_repro_stack)

    print("Loading IonMonger Beer-Lambert with V_bi = 0.86 V ...")
    ionmonger_bl_vbi086 = _run_jv(replace(ionmonger_stack, V_bi=0.86))

    print("Loading IonMonger TMM with V_bi = 1.10 V ...")
    ionmonger_tmm_stack = load_device_from_yaml("configs/ionmonger_benchmark_tmm.yaml")
    ionmonger_tmm_vbi110 = _run_jv(replace(ionmonger_tmm_stack, V_bi=1.10))

    print("Loading Driftfusion default flat-band comparison ...")
    driftfusion_stack = _load_driftfusion_flatband()
    driftfusion_default = _run_jv(driftfusion_stack)

    print("Loading Driftfusion Calado 2016 reproduction preset ...")
    driftfusion_repro_stack = load_device_from_yaml(
        "configs/driftfusion_calado2016_repro.yaml"
    )
    driftfusion_repro = _run_jv(driftfusion_repro_stack)

    print("Running Driftfusion Suns-V_oc sanity sweep ...")
    suns = (0.1, 0.5, 1.0, 2.0, 5.0)
    suns_voc = tuple(
        _run_jv(
            replace(driftfusion_repro_stack, Phi=driftfusion_repro_stack.Phi * sun)
        ).metrics_rev.V_oc
        for sun in suns
    )

    im = ionmonger_repro.metrics_rev
    df = driftfusion_repro.metrics_rev
    metrics = (
        PaperMetric(
            device="IonMonger",
            metric="V_oc",
            metric_label="V$_{oc}$",
            paper_value=IONMONGER_PAPER.V_oc,
            ours=im.V_oc,
            units="V",
            note="Legacy convention, V$_{bi}$=0.98 V",
        ),
        PaperMetric(
            device="IonMonger",
            metric="J_sc",
            metric_label="J$_{sc}$",
            paper_value=IONMONGER_PAPER.J_sc,
            ours=im.J_sc,
            units="A/m²",
            note="Beer-Lambert\ngeneration envelope",
        ),
        PaperMetric(
            device="IonMonger",
            metric="FF",
            metric_label="FF",
            paper_value=(IONMONGER_PAPER.FF_min, IONMONGER_PAPER.FF_max),
            ours=im.FF,
            units="",
            note="SRH-limited set (b) range",
            delta_note="vs midpoint",
        ),
        PaperMetric(
            device="Driftfusion",
            metric="V_oc",
            metric_label="V$_{oc}$",
            paper_value=(DRIFTFUSION_PAPER.V_oc_min, DRIFTFUSION_PAPER.V_oc_max),
            ours=df.V_oc,
            units="V",
            note="Flat-band convention, V$_{bi}$=1.42 V",
            delta_note="vs midpoint",
        ),
        PaperMetric(
            device="Driftfusion",
            metric="J_sc",
            metric_label="J$_{sc}$",
            paper_value=DRIFTFUSION_PAPER.J_sc,
            ours=df.J_sc,
            units="A/m²",
            note="Correct optical\norder of magnitude",
        ),
    )
    return ComparisonBundle(
        ionmonger_default=ionmonger_default,
        ionmonger_repro=ionmonger_repro,
        ionmonger_bl_vbi086=ionmonger_bl_vbi086,
        ionmonger_tmm_vbi110=ionmonger_tmm_vbi110,
        driftfusion_default=driftfusion_default,
        driftfusion_repro=driftfusion_repro,
        driftfusion_suns=suns,
        driftfusion_suns_voc=suns_voc,
        metrics=metrics,
    )


def _apply_style() -> None:
    """Apply publication-oriented Matplotlib styling."""
    font_manager.fontManager.addfont(ARIAL_FONT)
    font_manager.fontManager.addfont(ARIAL_BOLD_FONT)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 11.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "legend.fontsize": 10,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "figure.dpi": 120,
            "savefig.dpi": EXPORT_DPI,
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.default": "regular",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _arial_font(size: float | None = None, *, bold: bool = False) -> font_manager.FontProperties:
    """Return an explicit Arial font so legends do not fall back silently."""
    return font_manager.FontProperties(fname=ARIAL_BOLD_FONT if bold else ARIAL_FONT, size=size)


def _current_density_ma_cm2(result: JVResult) -> np.ndarray:
    return np.asarray(result.J_rev) / 10.0


def _add_panel_note(ax: plt.Axes, text: str) -> None:
    """Place a compact explanation below an axes."""
    ax.text(
        0.0,
        -0.22,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.6,
        color="0.25",
        linespacing=1.25,
        clip_on=False,
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    """Add a small panel letter in axes coordinates."""
    ax.text(
        -0.10,
        1.05,
        label,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=14,
        fontweight="bold",
    )


def _plot_jv_panel(
    ax: plt.Axes,
    repro: JVResult,
    *,
    title: str,
    color: str,
    default: JVResult | None = None,
    paper_voc: float | tuple[float, float] | None = None,
    operational: bool = False,
) -> None:
    """Draw one J-V panel with metrics and optional paper V_oc marker."""
    metrics = repro.metrics_rev
    if default is not None:
        ax.plot(
            np.asarray(default.V_rev),
            _current_density_ma_cm2(default),
            color="0.55",
            lw=1.1,
            ls=":",
            label=BASELINE_LABEL,
        )
    ax.plot(
        np.asarray(repro.V_rev),
        _current_density_ma_cm2(repro),
        color=color,
        lw=1.7,
        label=CALIBRATED_LABEL,
    )
    ax.axhline(0, color="0.55", ls=":", lw=0.8)
    ax.axvline(0, color="0.55", ls=":", lw=0.8)
    if paper_voc is not None:
        if isinstance(paper_voc, tuple):
            ax.axvspan(
                paper_voc[0],
                paper_voc[1],
                color="0.82",
                alpha=0.35,
                label=f"{BENCHMARK_LABEL} V$_{{oc}}$ range",
            )
        else:
            ax.axvline(
                paper_voc,
                color="0.35",
                ls="--",
                lw=1.0,
                label=BENCHMARK_LABEL,
            )
    ax.legend(
        loc="upper left",
        prop=_arial_font(10),
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="0.82",
        handlelength=2.3,
        borderpad=0.35,
        labelspacing=0.32,
    )
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current density (mA/cm$^2$)")
    ax.set_title(title, pad=12)
    if operational:
        ax.set_xlim(-0.02, 1.2)
        ax.set_ylim(-2.0, 28.0)
    ax.text(
        0.03,
        0.05,
        f"V$_{{oc}}$={metrics.V_oc:.3f} V\n"
        f"J$_{{sc}}$={metrics.J_sc / 10:.1f} mA/cm$^2$\n"
        f"FF={metrics.FF:.3f}",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.8"},
    )


def _draw_metric_table(ax: plt.Axes, metrics: tuple[PaperMetric, ...]) -> None:
    """Draw the paper-vs-ours comparison table."""
    ax.axis("off")
    columns = (
        "Device",
        "Metric",
        "Published\nbenchmark",
        "Literature-\naligned model",
        "Delta",
        "Interpretation",
    )
    rows = [
        (
            metric.device,
            metric.metric_label,
            _format_paper_value(metric.paper_value, metric.units),
            _format_paper_value(metric.ours, metric.units),
            (
                f"{_format_delta(_metric_delta(metric), metric.units)}\n"
                f"{metric.delta_note}"
                if metric.delta_note
                else _format_delta(_metric_delta(metric), metric.units)
            ),
            metric.note,
        )
        for metric in metrics
    ]
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=(0.13, 0.10, 0.14, 0.15, 0.13, 0.35),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    table.scale(1.0, 2.05)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("0.82")
        if row == 0:
            cell.set_facecolor("#e9eef6")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f8fa")
    ax.set_title("Published benchmark vs literature-aligned model", pad=12)


def _draw_optics_panel(ax: plt.Axes, bundle: ComparisonBundle) -> None:
    """Draw Beer-Lambert vs TMM comparison at fixed V_bi."""
    bl = bundle.ionmonger_default
    tmm = bundle.ionmonger_tmm_vbi110
    ax.plot(
        np.asarray(bl.V_rev),
        _current_density_ma_cm2(bl),
        color="#1f1f1f",
        lw=1.5,
        label="Beer-Lambert baseline",
    )
    ax.plot(
        np.asarray(tmm.V_rev),
        _current_density_ma_cm2(tmm),
        color="#2ca02c",
        lw=1.5,
        ls="--",
        label="TMM optics model",
    )
    ax.axhline(0, color="0.55", ls=":", lw=0.8)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current density (mA/cm$^2$)")
    ax.set_title("TMM optics separated from V$_{bi}$", pad=12)
    ax.set_xlim(-0.02, 1.50)
    ax.set_ylim(-2.0, 24.0)
    ax.legend(
        frameon=True,
        prop=_arial_font(10),
        framealpha=0.94,
        facecolor="white",
        edgecolor="0.82",
        loc="lower left",
        handlelength=2.3,
        borderpad=0.35,
    )
    delta_voc = tmm.metrics_rev.V_oc - bl.metrics_rev.V_oc
    jsc_ratio = tmm.metrics_rev.J_sc / bl.metrics_rev.J_sc
    ax.text(
        0.98,
        0.05,
        "Fixed V$_{{bi}}$ = 1.10 V\n"
        f"ΔV$_{{oc}}$={delta_voc * 1e3:+.0f} mV\n"
        f"J$_{{sc}}$ ratio={jsc_ratio:.3f}",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=10.2,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.8"},
    )


def _draw_suns_panel(ax: plt.Axes, bundle: ComparisonBundle) -> None:
    """Draw Driftfusion Suns-V_oc sanity trend."""
    suns = np.asarray(bundle.driftfusion_suns)
    voc = np.asarray(bundle.driftfusion_suns_voc)
    slope, _, r_value, _, _ = linregress(np.log(suns), voc)
    slope_mv_per_ln = slope * 1000.0
    ax.semilogx(suns, voc * 1000.0, marker="o", color="#1f77b4", lw=1.6)
    ax.set_xlabel("Illumination (suns)")
    ax.set_ylabel("V$_{oc}$ (mV)")
    ax.set_title("Suns-V$_{oc}$ sanity: logarithmic light response", pad=12)
    ax.grid(True, which="both", alpha=0.22)
    ax.set_xticks(suns)
    ax.set_xticklabels(("0.1", "0.5", "1", "2", "5"))
    ax.text(
        0.04,
        0.96,
        f"dV$_{{oc}}$/d ln(Φ/Φ$_{{1sun}}$) = {slope_mv_per_ln:.1f} mV\n"
        f"r = {r_value:.3f}\n"
        f"n$_{{id}}$ ≈ {slope_mv_per_ln / 25.85:.1f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.8"},
    )


def _draw_device_configurations(ax: plt.Axes) -> None:
    """Draw the exact device stacks used by the reproduction presets."""
    ax.axis("off")
    ax.set_title("Device configurations tested", pad=12)
    stacks = (
        (
            "IonMonger / Courtier set (b)",
            "legacy, V$_{bi}$=0.98 V",
            ("spiro HTL\n200 nm", "MAPbI$_3$\n400 nm", "TiO$_2$ ETL\n100 nm"),
            ("#c7a4d8", "#d9c45f", "#8fb6d9"),
            ("χ/E$_g$ retained", "SRH-limited", "mobile ions"),
        ),
        (
            "Driftfusion / Calado",
            "legacy flat-band, V$_{bi}$=1.42 V",
            ("spiro HTL\n200 nm", "MAPbI$_3$\n400 nm", "TiO$_2$ ETL\n100 nm"),
            ("#c7a4d8", "#d9c45f", "#8fb6d9"),
            ("χ=E$_g$=0", "radiative+SRH", "mobile ions"),
        ),
    )
    y_positions = (0.74, 0.30)
    widths = (0.21, 0.34, 0.21)
    x0 = 0.11
    for (title, subtitle, layers, colors, tags), y in zip(stacks, y_positions):
        ax.text(0.02, y + 0.15, title, transform=ax.transAxes, weight="bold", fontsize=11.0)
        ax.text(0.02, y + 0.085, subtitle, transform=ax.transAxes, fontsize=9.8, color="0.25")
        x = x0
        for label, color, width in zip(layers, colors, widths):
            rect = plt.Rectangle(
                (x, y - 0.075),
                width,
                0.15,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="0.55",
                lw=0.8,
            )
            ax.add_patch(rect)
            ax.text(
                x + width / 2,
                y,
                label,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9.6,
            )
            x += width
        ax.text(
            0.08,
            y - 0.145,
            " | ".join(tags),
            transform=ax.transAxes,
            fontsize=9.4,
            color="0.25",
        )
    ax.text(
        0.02,
        -0.03,
        "Both literature-aligned cases keep the same p-i-n layer order.\n"
        "Validation changes physics conventions and literature-aligned V$_{bi}$, "
        "not the stack sequence.",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9.4,
        color="0.25",
        linespacing=1.20,
    )


def _draw_mismatch_causes(ax: plt.Axes, bundle: ComparisonBundle) -> None:
    """Summarize the approved mismatch-cause narrative."""
    ax.axis("off")
    im_default_delta = bundle.ionmonger_default.metrics_rev.V_oc - IONMONGER_PAPER.V_oc
    im_repro_delta = bundle.ionmonger_repro.metrics_rev.V_oc - IONMONGER_PAPER.V_oc
    df_mid = 0.5 * (DRIFTFUSION_PAPER.V_oc_min + DRIFTFUSION_PAPER.V_oc_max)
    df_default_delta = bundle.driftfusion_default.metrics_rev.V_oc - df_mid
    df_repro_delta = bundle.driftfusion_repro.metrics_rev.V_oc - df_mid
    vbi_delta = (
        bundle.ionmonger_bl_vbi086.metrics_rev.V_oc
        - bundle.ionmonger_default.metrics_rev.V_oc
    )
    im_vbi086_delta = bundle.ionmonger_bl_vbi086.metrics_rev.V_oc - IONMONGER_PAPER.V_oc
    guide = (
        "A IonMonger J-V: SolarLab baseline model vs Courtier-aligned model; "
        "the dashed line is the published V$_{oc}$ benchmark.\n"
        "B Driftfusion J-V: SolarLab baseline model vs Calado-aligned model; "
        "the shaded band is the published V$_{oc}$ benchmark interval.\n"
        "C Metrics table: published benchmarks, literature-aligned model values, and deltas "
        "(range deltas use the midpoint).\n"
        "D Device configurations: the exact p-i-n stacks and literature-aligned conventions "
        "used by the two aligned models.\n"
        "E Optics check: Beer-Lambert and TMM are compared at fixed V$_{bi}$ to "
        "separate optical effects from voltage calibration.\n"
        "F Suns-V$_{oc}$ sanity: the literature-aligned Driftfusion model retains a "
        "physical logarithmic illumination response."
    )
    causes = (
        f"IonMonger baseline V$_{{oc}}$ is {im_default_delta * 1e3:+.0f} mV vs the "
        f"published benchmark; the literature-aligned model is {im_repro_delta * 1e3:+.0f} mV. "
        f"Lowering BL V$_{{bi}}$ from 1.10 V to 0.86 V changes V$_{{oc}}$ by "
        f"{vbi_delta * 1e3:+.0f} mV ({im_vbi086_delta * 1e3:+.0f} mV vs published). "
        f"Driftfusion baseline V$_{{oc}}$ is {df_default_delta * 1e3:+.0f} mV vs "
        f"the Calado midpoint; the literature-aligned model is {df_repro_delta * 1e3:+.0f} mV. "
        "J$_{sc}$ remains near the Beer-Lambert paper scale, so V$_{oc}$ conventions "
        "dominate the discrepancy."
    )
    ax.set_title("Panel guide and main comparison takeaway", pad=12)
    ax.text(
        0.02,
        0.86,
        guide,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.8,
        linespacing=1.28,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f7f8fa", "edgecolor": "0.82"},
    )
    ax.text(
        0.59,
        0.86,
        "\n".join(
            (
                "Main comparison takeaway:",
                f"• IonMonger baseline V$_{{oc}}$: {im_default_delta * 1e3:+.0f} mV vs published;",
                f"  literature-aligned model: {im_repro_delta * 1e3:+.0f} mV.",
                f"• Lowering BL V$_{{bi}}$ from 1.10 V to 0.86 V changes",
                f"  V$_{{oc}}$ by {vbi_delta * 1e3:+.0f} mV "
                f"({im_vbi086_delta * 1e3:+.0f} mV vs published).",
                f"• Driftfusion baseline V$_{{oc}}$: {df_default_delta * 1e3:+.0f} mV vs Calado midpoint;",
                f"  literature-aligned model: {df_repro_delta * 1e3:+.0f} mV.",
                "• J$_{sc}$ remains near the Beer-Lambert paper scale;",
                "  V$_{oc}$ conventions dominate the discrepancy.",
            )
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.8,
        linespacing=1.28,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "0.82"},
    )


def render_figure(bundle: ComparisonBundle) -> plt.Figure:
    """Render the approved paper-validation mockup concept."""
    _apply_style()
    fig = plt.figure(figsize=(21.0, 13.2))
    mosaic = [
        ["im_jv", "df_jv", "table"],
        ["devices", "optics", "suns"],
        ["causes", "causes", "causes"],
    ]
    axes = fig.subplot_mosaic(
        mosaic,
        gridspec_kw={
            "width_ratios": [1.10, 1.10, 1.55],
            "height_ratios": [1.00, 0.94, 0.60],
            "hspace": 0.50,
            "wspace": 0.40,
        },
    )

    _panel_label(axes["im_jv"], "A")
    _plot_jv_panel(
        axes["im_jv"],
        bundle.ionmonger_repro,
        default=bundle.ionmonger_default,
        title="IonMonger: literature-aligned model vs baseline",
        color="#1f1f1f",
        paper_voc=IONMONGER_PAPER.V_oc,
        operational=True,
    )
    _panel_label(axes["df_jv"], "B")
    _plot_jv_panel(
        axes["df_jv"],
        bundle.driftfusion_repro,
        default=bundle.driftfusion_default,
        title="Driftfusion: literature-aligned model vs baseline",
        color="#1f77b4",
        paper_voc=(DRIFTFUSION_PAPER.V_oc_min, DRIFTFUSION_PAPER.V_oc_max),
        operational=True,
    )
    _panel_label(axes["table"], "C")
    _draw_metric_table(axes["table"], bundle.metrics)
    _panel_label(axes["devices"], "D")
    _draw_device_configurations(axes["devices"])
    _panel_label(axes["optics"], "E")
    _draw_optics_panel(axes["optics"], bundle)
    _panel_label(axes["suns"], "F")
    _draw_suns_panel(axes["suns"], bundle)
    _draw_mismatch_causes(axes["causes"], bundle)

    fig.suptitle(
        "SolarLab paper-reproduction validation: literature-aligned models vs baseline",
        fontsize=18,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.905, bottom=0.065)
    return fig


def main() -> None:
    bundle = collect_comparison_data()
    fig = render_figure(bundle)
    for path in (Path("paper_validation.png"), Path("paper_validation.pdf")):
        fig.savefig(path, bbox_inches="tight", dpi=EXPORT_DPI)
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
