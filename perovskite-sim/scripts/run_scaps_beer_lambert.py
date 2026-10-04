"""Rebuild the eleven-sweep report through the ordinary frontend J-V route.

Run from perovskite-sim with BLAS threads pinned to one. --prepare-reference
extracts the read-only partner workbook; normal runs never modify it.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perovskite_sim.validation.scaps_beer_lambert import EXTRA_SWEEP_POINTS, REFERENCE_PATH, SWEEPS, reference_catalog


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare_reference():
    import openpyxl

    source = ROOT / "reproducibility" / "ScapsParams.xlsx"
    book = openpyxl.load_workbook(source, data_only=True, read_only=True)
    sweeps = {}
    try:
        for sheet, (title, unit, _, _) in SWEEPS.items():
            rows = iter(book[sheet].values)
            header = next(rows)
            points = []
            for number, row in enumerate(rows, 2):
                values = dict(zip(header, row))
                if values.get("scan_value") is None:
                    continue
                metrics = {
                    key: float(values[column]) if values.get(column) is not None else None
                    for key, column in (
                        ("V_oc", "Voc_V"), ("J_sc", "Jsc_mA_cm2"),
                        ("FF", "FF_percent"), ("PCE", "PCE_percent"),
                    )
                }
                points.append({"x": float(values["scan_value"]), "row": number, "reference": metrics})
            sweeps[sheet] = {"title": title, "unit": unit, "points": points}
    finally:
        book.close()
    write_json(REFERENCE_PATH, {
        "schema_version": 1, "study": "scaps_beer_lambert_v1",
        "source": "reproducibility/ScapsParams.xlsx",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "interface_density_note": "Workbook labels cm^-3; historical study and interface parser use areal cm^-2. No volumetric conversion is inferred.",
        "sweeps": sweeps,
    })
    print(f"Extracted {sum(len(s['points']) for s in sweeps.values())} points in {len(sweeps)} sweeps", flush=True)


def run_worker(request_path, result_path):
    from backend.main import _run_jv_dispatch, stack_from_dict, to_serializable
    from perovskite_sim.experiments.jv_sweep import build_electrical_grid
    from perovskite_sim.solver.mol import build_material_arrays
    from perovskite_sim.physics.generation import dual_cell_integral, beer_lambert_generation

    request = json.loads(request_path.read_text())
    started = time.monotonic()
    try:
        stack = stack_from_dict(request["device"])
        result = _run_jv_dispatch(stack, **request["params"])
        grid = build_electrical_grid(stack, request["params"]["N_grid"])
        mat = build_material_arrays(grid, stack)
        generation = mat.G_optical
        if generation is None:
            generation = beer_lambert_generation(grid, mat.alpha, stack.Phi)
        payload = {
            "status": "succeeded" if result.metrics_fwd.voc_bracketed else "voc_not_bracketed",
            "result": to_serializable(result),
            "generation_budget_A_m2": 1.602176634e-19 * dual_cell_integral(grid, generation),
            "grid_nodes": len(grid),
            "optical_model": stack.optical_model,
        }
    except Exception as exc:
        payload = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    payload.update(elapsed_s=time.monotonic() - started, request_sha256=digest(request))
    write_json(result_path, payload)


def source_fingerprint(root=ROOT):
    files = sorted((root / "perovskite_sim").rglob("*.py"))
    files += [root / "backend" / "main.py", root / "scripts" / Path(__file__).name,
              root / REFERENCE_PATH.relative_to(ROOT)]
    return digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def collect(out, *, timeout, only, workers):
    from backend.main import _config_dict_from_path, scaps_beer_lambert_reference

    catalog = reference_catalog()
    base = _config_dict_from_path(str(ROOT / "configs" / "scaps_mirror_v2.yaml"))
    params = {**base["simulation_hints"]["jv_sweep"], "illuminated": True, "solver": "transient"}
    manifest_path = out / "Manifest.json"
    import numpy, scipy
    contract = {
        "source_fingerprint": source_fingerprint(), "reference": catalog, "baseline": base, "params": params,
        "environment": {"python": sys.version, "numpy": numpy.__version__, "scipy": scipy.__version__},
    }
    if manifest_path.exists() and json.loads(manifest_path.read_text())["contract"] != contract:
        raise ValueError("Existing report has a different code/input contract; choose a new output directory")
    if not manifest_path.exists():
        write_json(manifest_path, {"created_utc": datetime.now(timezone.utc).isoformat(), "contract": contract})
    snapshot = out / "Source"
    if not snapshot.exists():
        for folder in ("perovskite_sim", "backend", "configs", "docs", "reproducibility"):
            shutil.copytree(ROOT / folder, snapshot / folder, ignore=shutil.ignore_patterns("__pycache__"))
        (snapshot / "scripts").mkdir()
        shutil.copy2(Path(__file__).resolve(), snapshot / "scripts" / Path(__file__).name)
    if source_fingerprint(snapshot) != contract["source_fingerprint"]:
        raise ValueError("Source snapshot does not match the run contract")
    cases = [("Baseline", None, None, base)]
    for sheet, entry in catalog["sweeps"].items():
        if only and sheet not in only:
            continue
        for index, point in enumerate(entry["points"]):
            slug = sheet.replace(" ", "_")
            cfg = scaps_beer_lambert_reference(sheet, point["x"])["config"]
            cases.append((f"{slug}/Point{index + 1:02d}", sheet, point, cfg))
    def execute_case(indexed):
        index, (case, sheet, point, cfg) = indexed
        request = {"device": cfg, "params": {**params, **cfg["simulation_hints"]["jv_sweep"]}}
        folder = out / "Points" / case
        request_path, result_path = folder / "Request.json", folder / "Result.json"
        write_json(request_path, request)
        if not result_path.exists():
            print(f"[{index}/{len(cases)}] {case} {point['x'] if point else ''}", flush=True)
            try:
                completed = subprocess.run(
                    [sys.executable, str(snapshot / "scripts" / Path(__file__).name), "--worker", str(request_path), str(result_path)],
                    capture_output=True, text=True, timeout=timeout,
                )
                if not result_path.exists():
                    write_json(result_path, {"status": "failed", "error": completed.stderr[-4000:], "request_sha256": digest(request)})
            except subprocess.TimeoutExpired:
                write_json(result_path, {"status": "timeout", "error": f"Exceeded {timeout} seconds", "request_sha256": digest(request)})
        data = json.loads(result_path.read_text())
        if data["request_sha256"] != digest(request):
            raise ValueError(f"Stale result: {case}")
        print(f"  {data['status']} {data.get('elapsed_s', 0):.1f}s", flush=True)
        extension = {"origin": "solarlab_extension", "scope": "high_doping_extrapolation"} if point and point.get("origin") == "solarlab_extension" else {}
        return {"case": case, "sweep": sheet, "x": point["x"] if point else None, "reference": point["reference"] if point else None, **extension, **data}

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(execute_case, enumerate(cases, 1)):
            records.append(record)
            write_json(out / "Results.json", records)
    return records


def extend_pvk(out, *, timeout):
    """Append only the two requested points using the original frozen solver."""
    from backend.main import _config_dict_from_path, scaps_beer_lambert_reference

    out = out.resolve()
    manifest = json.loads((out / "Manifest.json").read_text())
    snapshot = out / "Source"
    if source_fingerprint(snapshot) != manifest["contract"]["source_fingerprint"]:
        raise ValueError("The original computational source snapshot has changed")
    base = _config_dict_from_path(str(ROOT / "configs/scaps_mirror_v2.yaml"))
    if base != manifest["contract"]["baseline"]:
        raise ValueError("Current frontend baseline differs from the recorded report")
    records = json.loads((out / "Results.json").read_text())
    before_hash = hashlib.sha256((out / "Results.json").read_bytes()).hexdigest()
    existing = {(row["sweep"], row["x"]): row for row in records}
    extension_records = []
    source_count = sum(row["sweep"] == "Nd_PVK" and row.get("origin") != "solarlab_extension" for row in records)
    for index, value in enumerate(EXTRA_SWEEP_POINTS["Nd_PVK"], source_count + 1):
        case = f"Nd_PVK/Point{index:02d}"
        cfg = scaps_beer_lambert_reference("Nd_PVK", value)["config"]
        request = {"device": cfg, "params": {**manifest["contract"]["params"], **cfg["simulation_hints"]["jv_sweep"]}}
        request_path, result_path = out / "Points" / case / "Request.json", out / "Points" / case / "Result.json"
        refined = False
        if request_path.exists() and result_path.exists():
            old_request = json.loads(request_path.read_text())
            if old_request != request:
                old_grid = old_request["params"]["N_grid"]
                old_physics = {k: v for k, v in old_request["device"].items() if k != "simulation_hints"}
                new_physics = {k: v for k, v in request["device"].items() if k != "simulation_hints"}
                if old_physics != new_physics or {**old_request["params"], "N_grid": request["params"]["N_grid"]} != request["params"] or request["params"]["N_grid"] <= old_grid:
                    raise ValueError("Only explicit grid refinement can replace an extension attempt")
                attempts = request_path.parent / "Attempts" / f"NGrid{old_grid}"
                attempts.mkdir(parents=True, exist_ok=True)
                for source in (request_path, result_path):
                    destination = attempts / source.name
                    if destination.exists() and destination.read_bytes() != source.read_bytes():
                        raise ValueError("Existing archived extension attempt differs")
                    if not destination.exists():
                        shutil.copy2(source, destination)
                refined = True
        write_json(request_path, request)
        if not result_path.exists() or refined:
            print(f"PVK extension: {value:g} cm^-3, N_grid={request['params']['N_grid']}", flush=True)
            try:
                completed = subprocess.run(
                    [sys.executable, str(snapshot / "scripts" / Path(__file__).name), "--worker", str(request_path), str(result_path)],
                    capture_output=True, text=True, timeout=timeout,
                )
                if not result_path.exists() or json.loads(result_path.read_text()).get("request_sha256") != digest(request):
                    write_json(result_path, {"status": "failed", "error": completed.stderr[-4000:], "request_sha256": digest(request)})
            except subprocess.TimeoutExpired:
                write_json(result_path, {"status": "timeout", "error": f"Exceeded {timeout} seconds", "request_sha256": digest(request)})
        result = json.loads(result_path.read_text())
        if result["request_sha256"] != digest(request):
            raise ValueError(f"Stale extension request: {case}")
        row = {"case": case, "sweep": "Nd_PVK", "x": value,
               "origin": "solarlab_extension", "scope": "high_doping_extrapolation",
               "reference": {key: None for key in ("V_oc", "J_sc", "FF", "PCE")}, **result}
        if ("Nd_PVK", value) not in existing:
            records.append(row)
        elif existing[("Nd_PVK", value)] != row:
            if not refined:
                raise ValueError(f"Existing extension result differs: {case}")
            records[records.index(existing[("Nd_PVK", value)])] = row
        extension_records.append(row)
        print(f"  {result['status']}: {result.get('result', {}).get('metrics_fwd')}", flush=True)
    extension_path = out / "PvkExtensionManifestV1.json"
    if not extension_path.exists():
        extension_manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "original_results_sha256": before_hash,
            "source_fingerprint": manifest["contract"]["source_fingerprint"],
            "values_cm3": list(EXTRA_SWEEP_POINTS["Nd_PVK"]),
            "scope": "High-doping extrapolation of the unchanged Maxwell-Boltzmann model; no additional SCAPS data or convergence certificate.",
        }
    else:
        extension_manifest = json.loads(extension_path.read_text())
    extension_manifest["requests"] = [r["request_sha256"] for r in extension_records]
    extension_manifest["N_grid"] = 90
    extension_manifest["prior_attempts"] = [
        "Points/Nd_PVK/Point07/Attempts/NGrid30",
        "Points/Nd_PVK/Point08/Attempts/NGrid30",
    ]
    write_json(extension_path, extension_manifest)
    write_json(out / "Results.json", records)
    return records


def build_report(out, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    import numpy as np
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle

    catalog = reference_catalog()
    if len(records) != 1 + sum(len(s["points"]) for s in catalog["sweeps"].values()):
        raise ValueError("The integrated report requires all eleven sweep families and the baseline")
    manifest = json.loads((out / "Manifest.json").read_text())
    styles = getSampleStyleSheet()
    for name in ("Normal", "BodyText", "Heading1", "Heading2", "Title"):
        styles[name].fontName = "Times-Roman" if name in ("Normal", "BodyText") else "Times-Bold"
    styles["Normal"].fontSize, styles["Normal"].leading = 10.5, 14
    styles["Title"].fontSize, styles["Title"].leading = 18, 22
    styles["Title"].alignment = TA_LEFT
    flow = []
    def para(text, style="Normal"):
        flow.extend([Paragraph(text, styles[style]), Spacer(1, 8)])
    def table(rows, widths=None):
        wrapped = [[Paragraph(str(cell), styles["Normal"]) for cell in row] for row in rows]
        item = Table(wrapped, colWidths=widths, hAlign="LEFT", repeatRows=1)
        item.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), .6, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
        flow.extend([item, Spacer(1, 12)])
    def page(): flow.append(PageBreak())
    good = sum(r["status"] == "succeeded" for r in records)
    para("SolarLab versus SCAPS-1D across Eleven Parametric Sweeps", "Title")
    para("Beer-Lambert optical model: reproducible frontend inputs", "Heading2")
    para(f"Version 1 | {manifest['created_utc'][:10]} | {good}/{len(records)} archived jobs with bracketed forward Voc")
    para("1. Scope and changes", "Heading1")
    para("This report replaces the optical calculation of the 2026-07-02 TMM report. It reruns the current SolarLab implementation, with the original electrical calibration retained: heterointerface bulk-rate de-spike fraction 0.53 and metal-reservoir barrier 0.42 eV. It is a calibrated comparison, not independent SCAPS validation or a numerical-convergence certificate.")
    para("The source workbook contains 93 points across eleven one-parameter families. Figure 3 extends only Jsc to the pronounced-drop point at Nd=1e19 cm-3; Voc, FF and PCE retain the original six PVK samples. Further diagnostic results remain in the raw archive, not in the plotted trends. Missing SCAPS values remain absent; no missing point is filled by interpolation.")
    para("Baseline", "Heading2")
    baseline = records[0]
    if "result" in baseline:
        m = baseline["result"]["metrics_fwd"]
        table([["Quantity", "SolarLab Beer-Lambert"], ["Voc (V)", f"{m['V_oc']:.7f}"], ["Jsc (mA/cm2)", f"{m['J_sc']/10:.7f}"], ["FF (%)", f"{100*m['FF']:.5f}"], ["PCE (%)", f"{100*m['PCE']:.5f}"]], [200, 290])
    para("The July curves are historical TMM results. Differences from that document also include subsequent code changes; only a same-code, same-input optical control can isolate the optical contribution.")
    page()
    para("2. Optical and electrical protocol", "Heading1")
    para("Single-pass spectral Beer-Lambert uses alpha(lambda) = 4 pi k(lambda) / lambda and the same tabulated material data and AM1.5G source used by the TMM branch. The spectrum is integrated on 200 samples from 300 to 1000 nm at one sun. Front/back reflection and coherent interference are omitted. Substrate and transport-layer absorption attenuate the flux. Phi is used only by scalar Beer-Lambert, not by this spectral source.")
    para("Each electrical dual cell receives its exact absorbed-photon count divided by the solver cell weight. The summed generation is bounded by the incident photon flux, independently of the electrical grid. TMM-derived photon recycling and radiative reabsorption are not active; intrinsic radiative and Auger recombination remain active.")
    para("The device remains glass / HTL (20 nm) / MAPbI3 (800 nm) / ETL (25 nm), with Fast electrical physics, no mobile ions, the effective-lifetime defect model, and no physical interface-plane boundary. This preserves the historical electrical sweep mapping, not an independently resolved microscopic defect population.")
    para("As in the historical sweep helper, each parameter point also recomputes Vbi from the doped-layer work functions. The separate unswept baseline retains Vbi=1.30 V. This derived update is saved in each request; a nominal baseline value on a sweep need not equal the unswept baseline exactly.")
    table([["Setting", "Value"], *[[key, value] for key, value in manifest["contract"]["params"].items()], ["Temperature", "300 K"], ["Driver tolerances", "rtol=1e-4; atol=1e-6 (density solver defaults)"], ["Metric branch", "Forward; Jsc=A/m2 divided by 10; FF/PCE multiplied by 100"]], [205, 285])
    para("This staircase is a finite-time transient protocol. It is not a claim of residual-certified steady state. No grid or time convergence is inferred from matching the frontend.")
    page()
    para("3. Reproduction in the frontend", "Heading1")
    para("Create a device with Reference model = SCAPS reproduction. Open Report parameters, select the reference sweep and its value, and press Load point. This restores the baseline, applies only the declared historical sweep mapping, and fills the material/interface fields. Run the ordinary J-V Sweep with the loaded settings. The same normalized device JSON and J-V dispatcher produced this report.")
    para("Generation model must remain Beer-Lambert (AM1.5G), profile Fast, Voltage points=40, scan rate=5 V/s, Vmax=1.6 V, Transient (Radau), and Standard staircase. Original points use N_grid=30; the displayed Jsc extension at Nd=1e19 cm-3 loads N_grid=90 automatically. Changing these settings defines a new run. Request.json beside each result contains its complete parameters.")
    rows = [["Family", "Archived runs", "Max |dVoc| (mV)", "Max |dPCE| (pp)"]]
    for sheet in SWEEPS:
        group = [r for r in records if r["sweep"] == sheet]
        paired = [r for r in group if r["status"] == "succeeded" and r["reference"]["V_oc"] is not None]
        dv = max((abs(r["result"]["metrics_fwd"]["V_oc"] - r["reference"]["V_oc"]) * 1000 for r in paired), default=float("nan"))
        dp = max((abs(r["result"]["metrics_fwd"]["PCE"] * 100 - r["reference"]["PCE"]) for r in paired), default=float("nan"))
        rows.append([sheet, f"{sum(r['status'] == 'succeeded' for r in group)}/{len(group)}", f"{dv:.2f}", f"{dp:.3f}"])
    table(rows, [220, 80, 95, 95])
    para("Interface density units are cm-2, following the historical interface model; the partner workbook's cm-3 column label is not silently interpreted as a volume density. The two bulk-Nt families both scale the combined effective lifetime. Bulk-Et families replace the effective n1/p1 pair. These approximations are not independent CB/VB microscopic-defect validation.")
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 10, "mathtext.default": "regular"})
    metrics = [("V_oc", r"$V_{oc}$ (V)", 1), ("J_sc", r"$J_{sc}$ (mA cm$^{-2}$)", .1), ("FF", "FF (%)", 100), ("PCE", "PCE (%)", 100)]
    metric_axes = {
        "V_oc": ((0.8, 2.0), (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0), "%.1f"),
        "J_sc": ((20, 30), (20, 22, 24, 26, 28, 30), "%.0f"),
        "FF": ((80, 95), (80, 82.5, 85, 87.5, 90, 92.5, 95), "%.1f"),
        "PCE": ((15, 35), (15, 20, 25, 30, 35), "%.0f"),
    }
    doping_axes = {
        **metric_axes,
        "V_oc": ((0.8, 2.0), (1.0, 1.25, 1.5, 1.75, 2.0), "%.2f"),
        "FF": ((60, 95), (60, 70, 80, 90), "%.0f"),
    }
    sweep_axes = {
        "CHI_ETL": {
            "V_oc": ((0, 1.6), (0, 0.5, 1.0, 1.5), "%.1f"),
            "J_sc": ((0, 35), (0, 10, 20, 30), "%.0f"),
            "FF": ((0, 100), (0, 20, 40, 60, 80, 100), "%.0f"),
            "PCE": ((0, 40), (0, 10, 20, 30, 40), "%.0f"),
        },
        "Nd_ETL": {**doping_axes, "FF": ((50, 95), (50, 60, 70, 80, 90), "%.0f")},
        "Nd_PVK": {
            **doping_axes,
            "J_sc": ((0, 30), (0, 5, 10, 15, 20, 25, 30), "%.0f"),
        },
    }
    sweep_x_axes = {
        "CHI_ETL": ((-1.085, 0.785), (-1.0, -0.5, 0.0, 0.5)),
        "Nd_ETL": ((10**9.5, 10**20.5), (1e11, 1e13, 1e15, 1e17, 1e19)),
        "Nd_PVK": ((10**7.5, 10**18.5), (1e9, 1e11, 1e13, 1e15, 1e17)),
    }
    figure_axes = []
    axis_labels = {
        "CHI_ETL": r"$\Delta E_C$ (eV)",
        "Nd_ETL": r"$N_{D,ETL}$ (cm$^{-3}$)",
        "Nd_PVK": r"$N_{D,PVK}$ (cm$^{-3}$)",
        "Nt_C_PVK": r"$N_t$ (cm$^{-3}$)",
        "Nt_V_PVK": r"$N_t$ (cm$^{-3}$)",
        "Nt_HTL PVK": r"$N_{t,int}$ (cm$^{-2}$)",
        "Nt_PVK ETL": r"$N_{t,int}$ (cm$^{-2}$)",
        "Et_C_PVK": r"$E_C-E_t$ (eV)",
        "Et_V_PVK": r"$E_t-E_V$ (eV)",
        "Et_HTL PVK": r"$E_{C,ref}-E_t$ (eV)",
        "Et_PVK ETL": r"$E_{C,ref}-E_t$ (eV)",
    }
    for number, (sheet, entry) in enumerate(catalog["sweeps"].items(), 1):
        page()
        para(f"4.{number} {entry['title']}", "Heading1")
        group = [r for r in records if r["sweep"] == sheet]
        if sheet == "Nd_PVK":
            group = [r for r in group if r["x"] <= 1e19]
        active_axes = sweep_axes.get(sheet, metric_axes)
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8))
        for ax, (key, label, scale) in zip(axes.flat, metrics):
            panel_group = group
            panel_reference = entry["points"]
            if sheet == "Nd_PVK":
                # The extension is shown only in Jsc; do not merely clip
                # extra Voc/FF/PCE segments outside their original axes.
                if key != "J_sc":
                    panel_group = [r for r in group if r.get("origin") != "solarlab_extension"]
                panel_reference = [p for p in entry["points"] if p.get("origin") != "solarlab_extension"]
            xs = [r["x"] for r in panel_group]
            ys = [r["result"]["metrics_fwd"][key] * scale if r["status"] == "succeeded" or (key == "J_sc" and r["status"] == "voc_not_bracketed") else np.nan for r in panel_group]
            refs = [p["reference"][key] if p["reference"][key] is not None else np.nan for p in panel_reference]
            ax.plot([p["x"] for p in panel_reference], refs, "s--", color="#ec2340", ms=4, lw=1.2, label="SCAPS")
            ax.plot(xs, ys, "o-", color="#007db3", ms=4, lw=1.2, label="SolarLab BL")
            if entry["unit"].startswith("cm"):
                ax.set_xscale("log")
            if sheet in sweep_x_axes:
                x_limits, x_ticks = sweep_x_axes[sheet]
                if sheet == "Nd_PVK" and key == "J_sc":
                    x_limits = (10**7.45, 10**19.55)
                    x_ticks = (1e9, 1e11, 1e13, 1e15, 1e17, 1e19)
                ax.set_xticks(x_ticks)
                ax.set_xlim(*x_limits)
            ax.set_xlabel(axis_labels[sheet])
            ax.set_ylabel(label)
            limits, ticks, tick_format = active_axes[key]
            ax.set_yticks(ticks)
            ax.set_ylim(*limits)
            ax.yaxis.set_major_formatter(FormatStrFormatter(tick_format))
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=8)
        fig.tight_layout()
        figure_axes.append({
            "figure": number,
            "sweep": sheet,
            "axes": {
                key: {
                    "limits": [float(value) for value in ax.get_ylim()],
                    "ticks": [float(value) for value in ax.get_yticks()],
                    "tick_labels": [label.get_text() for label in ax.get_yticklabels()],
                    "x_limits": [float(value) for value in ax.get_xlim()],
                    "x_ticks": [float(value) for value in ax.get_xticks()],
                    "series_x": {
                        line.get_label(): [float(value) for value in line.get_xdata()]
                        for line in ax.get_lines()
                    },
                }
                for ax, (key, _, _) in zip(axes.flat, metrics)
            },
        })
        image_path = out / "Figures" / f"Sweep{number:02d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(image_path, dpi=180)
        plt.close(fig)
        flow.extend([Image(str(image_path), width=490, height=347), Spacer(1, 12)])
        if sheet == "Nd_PVK":
            para("Figure 3. PVK donor doping. Voc, FF and PCE show the original six samples. Only Jsc includes the additional SolarLab point at Nd=1e19 cm-3; the curve stops there. Red: supplied SCAPS reference. Blue: SolarLab Beer-Lambert.")
            original_values = [r["x"] for r in group if r.get("origin") != "solarlab_extension"]
            para("Original Nd values: " + ", ".join(f"{value:g}" for value in original_values) + " cm-3. Jsc-only addition: 1e19 cm-3.")
        else:
            para(f"Figure {number}. {entry['title']}. Red: supplied SCAPS reference. Blue: newly computed spectral Beer-Lambert results. Source worksheet: {sheet}; {len(group)} requested SolarLab points.")
            para("Submitted values: " + ", ".join(f"{r['x']:g}" for r in group) + f" {entry['unit']}.")
        unavailable = [r for r in group if r["status"] != "succeeded"]
        para("Run status: " + ("all requested points completed with bracketed forward Voc." if not unavailable else "; ".join(f"{r['x']:g}: {r['status']}" for r in unavailable)))
        v, j, f, p = (active_axes[key][0] for key, _, _ in metrics)
        para(f"Comparison windows: Voc {v[0]:g}-{v[1]:g} V, Jsc {j[0]:g}-{j[1]:g} mA/cm2, FF {f[0]:g}-{f[1]:g}%, PCE {p[0]:g}-{p[1]:g}%. Values outside these windows are clipped; complete results remain in Result.json.")
        if sheet == "Nd_PVK":
            extension = [r for r in group if r.get("origin") == "solarlab_extension"]
            summary = "; ".join(
                f"Nd={r['x']:g}: Jsc={r['result']['metrics_fwd']['J_sc']/10:.4f} mA/cm2"
                if "result" in r else f"Nd={r['x']:g}: {r['status']}"
                for r in extension
            )
            para(summary + ". This Jsc-only extension uses N_grid=90 and has no SCAPS counterpart. It extrapolates the unchanged Maxwell-Boltzmann model to high doping, without a degeneracy or grid-convergence certificate. Unplotted diagnostic results are preserved in the raw archive.")
    page()
    para("5. Evidence and limitations", "Heading1")
    para("Observed comparison", "Heading2")
    para("The CBO scan retains cliff-side voltage loss and a large-spike current/FF collapse, but its absolute voltage plateau differs from SCAPS. The ETL-doping Voc response is nonmonotonic on the low-doping side and is not reproduced quantitatively. The HTL/PVK density response remains nearly flat in SolarLab. Switching optics has not removed these electrical-model or numerical differences; no new trend-agreement count is inferred.")
    para("Frontend reproduction establishes that identical inputs use the identical calculation route. It does not establish external SCAPS accuracy, mesh convergence, time convergence, or experimental validity. The report retains the earlier calibrated contact/interface parameters without refitting them after the optical change.")
    para("No universal optical attribution is made for remaining SCAPS differences. Matching Beer-Lambert attenuation alone does not align the material absorption tables, illumination spectrum, contact boundary conditions, defect closure, or interface transport of two solvers.")
    para("Provenance", "Heading2")
    para("SCAPS data: ScapsParams.xlsx, sheets named in each figure. All rows with a scan value were extracted; missing reference metrics remain null. The electronic mapping is the historical apply_sweep_point effective-lifetime route.")
    para("Data and code hashes are recorded in Manifest.json; each submitted request has its own SHA-256 in Result.json. Results.json retains every success, failure and timeout. Old FormalVerify.pdf and its TMM figures are not overwritten.")
    para("Source workbook SHA-256: " + catalog["source_sha256"])
    para("Code fingerprint: " + manifest["contract"]["source_fingerprint"])
    def footer(canvas, doc):
        canvas.setFont("Times-Roman", 9)
        canvas.drawCentredString(306, 28, str(doc.page))
    pdf = out / "FormalVerifyBeerLambertV1.pdf"
    SimpleDocTemplate(str(pdf), pagesize=(612, 792), rightMargin=61, leftMargin=61, topMargin=52, bottomMargin=48,
                      title="SolarLab versus SCAPS: Eleven Beer-Lambert Parameter Sweeps").build(flow, onFirstPage=footer, onLaterPages=footer)
    snapshot = out / "Source"
    write_json(out / "ReportManifest.json", {
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "results_sha256": hashlib.sha256((out / "Results.json").read_bytes()).hexdigest(),
        "report_builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "simulation_source_sha256": manifest["contract"]["source_fingerprint"],
        "source_snapshot_matches": source_fingerprint(snapshot) == manifest["contract"]["source_fingerprint"] if snapshot.exists() else None,
        "figure_axes": figure_axes,
        "pvk_extension_manifest_sha256": hashlib.sha256((out / "PvkExtensionManifestV1.json").read_bytes()).hexdigest() if (out / "PvkExtensionManifestV1.json").exists() else None,
        "optical_resources": {
            str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((snapshot / "perovskite_sim/data").rglob("*.csv"))
        },
    })
    print(f"Report: {pdf}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-reference", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "scaps_beer_lambert_v1")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sheets", nargs="+")
    parser.add_argument("--worker", nargs=2, type=Path)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--extend-pvk", action="store_true")
    args = parser.parse_args()
    if args.prepare_reference:
        prepare_reference()
    elif args.worker:
        run_worker(*args.worker)
    else:
        if args.workers < 1:
            parser.error("--workers must be positive")
        if args.extend_pvk:
            records = extend_pvk(args.out, timeout=args.timeout)
        else:
            records = json.loads((args.out / "Results.json").read_text()) if args.report_only else collect(args.out, timeout=args.timeout, only=args.sheets, workers=args.workers)
        build_report(args.out, records)


if __name__ == "__main__":
    main()
