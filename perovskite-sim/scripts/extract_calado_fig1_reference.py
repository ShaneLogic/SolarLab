#!/usr/bin/env python3
"""Extract Fig. 1e/f vector curves from the verified Calado 2016 paper PDF."""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path

import fitz
import numpy as np


PDF_SHA256 = "e9a3ca554f96cf2a8d0b5b1118957bb3252d0a727c553329633158acc800218e"
SUPPLEMENT_SHA256 = "7c4e8cddba46815347ec2aab3b149dd282b23885d3f383e11ab56a4571083f3b"
PANELS = {
    "fig1e": {"zero_x": 467, "tick_x": 425, "zero_y": 410, "tick_y": 367,
               "fwd": 470, "rev": 471},
    "fig1f": {"zero_x": 353, "tick_x": 311, "zero_y": 296, "tick_y": 253,
               "fwd": 356, "rev": 357},
}
POTENTIAL_CURVES = (
    ("fwd", 0.0, 734, 758), ("fwd", 0.4, 759, 783),
    ("fwd", 0.8, 784, 808), ("fwd", 1.1, 809, 833),
    ("rev", 1.1, 834, 858), ("rev", 0.8, 859, 883),
    ("rev", 0.4, 884, 908), ("rev", 0.0, 909, 933),
)


def outline_points(items) -> np.ndarray:
    points = []
    for item in items:
        if item[0] == "l":
            segment = np.asarray([item[1], item[2]], dtype=float)
        elif item[0] == "c":
            controls = np.asarray(item[1:], dtype=float)
            t = np.linspace(0.0, 1.0, 65)[:, None]
            segment = ((1-t)**3 * controls[0] + 3*(1-t)**2*t * controls[1]
                       + 3*(1-t)*t**2 * controls[2] + t**3 * controls[3])
        else:
            raise ValueError(f"unexpected curve primitive {item[0]!r}")
        if points and not np.allclose(points[-1], segment[0], atol=1e-3, rtol=0):
            raise ValueError("disconnected curve outline")
        points.extend(segment if not points else segment[1:])
    outline = np.asarray(points)
    if not np.allclose(outline[0], outline[-1], atol=1e-3, rtol=0):
        outline = np.vstack((outline, outline[0]))
    return outline


def slice_outline(outline: np.ndarray, x: float, *, envelope: bool = False) -> tuple[float, float] | None:
    left, right = outline[:-1], outline[1:]
    dx = right[:, 0] - left[:, 0]
    mask = ((left[:, 0] <= x) & (x < right[:, 0])) | ((right[:, 0] <= x) & (x < left[:, 0]))
    count = np.count_nonzero(mask)
    if count < 2 or count % 2 or (not envelope and count != 2):
        return None
    y = left[mask, 1] + (x-left[mask, 0]) / dx[mask] * (right[mask, 1]-left[mask, 1])
    return float(y.min()), float(y.max())


def stroke_endcaps(outline: np.ndarray) -> np.ndarray:
    """Locate equal-width transverse caps of a long, voltage-monotone stroke."""
    # PDF exporters may emit repeated vertices, which have no tangent.
    outline = outline[np.r_[True, np.any(np.diff(outline, axis=0) != 0, axis=1)]]
    edges = np.diff(outline, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    unit = np.divide(edges, lengths[:, None], out=np.zeros_like(edges), where=lengths[:, None] > 0)
    before, after = np.roll(unit, 1, axis=0), np.roll(unit, -1, axis=0)
    candidates = np.flatnonzero(
        (lengths > 0) & (np.sum(before*after, axis=1) < -0.95)
        & (abs(np.sum(before*unit, axis=1)) < 0.1)
        & (abs(np.sum(after*unit, axis=1)) < 0.1)
    )
    pairs = []
    for first, second in combinations(candidates, 2):
        widths = lengths[[first, second]]
        centers = np.array([(outline[i]+outline[i+1])/2 for i in (first, second)])
        span = abs(centers[1, 0]-centers[0, 0])
        if widths.max()/widths.min() < 1.02 and span > max(0.9*np.ptp(outline[:, 0]), 20*widths.max()):
            pairs.append((first, second))
    if len(pairs) != 1:
        raise ValueError(f"expected one unambiguous pair of stroke endcaps, found {len(pairs)}")
    caps = np.array([[outline[index], outline[index+1]] for index in pairs[0]])
    return caps[np.argsort(caps.mean(axis=1)[:, 0])]


def reference_sample_masks(voltage, stroke):
    low, high = stroke["centerline_voltage_interval_V"]
    domain = (voltage >= low) & (voltage <= high)
    cap_affected = np.zeros_like(domain)
    for left, right in stroke["endcap_voltage_intervals_V"]:
        cap_affected |= (voltage >= left) & (voltage <= right)
    return domain, domain & ~cap_affected


def compare_reference_samples(reference, sampled, stroke):
    """Keep cap-only slices visible but do not treat them as curve samples."""
    voltage, current, low, high = reference.T
    domain, midpoint = reference_sample_masks(voltage, stroke)
    if not np.any(midpoint):
        raise ValueError("no centerline midpoint samples remain for comparison")
    violation = np.maximum(np.maximum(low-sampled, sampled-high), 0)
    return {
        "rmse_A_m2": float(np.sqrt(np.mean((sampled[midpoint]-current[midpoint])**2))),
        "raw_outline_midpoint_rmse_A_m2": float(np.sqrt(np.mean((sampled-current)**2))),
        "max_abs_error_A_m2": float(np.max(abs(sampled[midpoint]-current[midpoint]))),
        "fraction_within_graphical_envelope": float(np.mean(violation[domain] == 0)),
        "max_graphical_envelope_violation_A_m2": float(violation[domain].max()),
        "raw_samples": len(reference), "centerline_domain_samples": int(domain.sum()),
        "midpoint_comparison_samples": int(midpoint.sum()),
        "endcap_only_samples": int((~domain).sum()),
        "cap_influenced_domain_samples": int((domain & ~midpoint).sum()),
        "comparison_voltage_interval_V": [float(voltage[domain][0]), float(voltage[domain][-1])],
        "scope": "Envelope checks use every centerline-domain sample; midpoint errors exclude cap intersections; raw slices retained",
    }


def extract_potentials(drawings, out_dir: Path, alignment: Path | None):
    """Read Fig. 1h's segmented strokes, converting plotted energy to phi."""
    def center(index, axis):
        rect = drawings[index]["rect"]
        return (rect[axis] + rect[axis + 2]) / 2

    x0, y0 = center(716, 0), center(730, 1)
    xscale = (center(720, 0) - x0) / 800
    yscale = center(728, 1) - y0
    simulation = np.load(alignment) if alignment else None
    report = {"source_pdf_sha256": PDF_SHA256, "source_panel": "Fig. 1h",
              "uncertainty_scope": "graphical stroke width only",
              "high_bias_legend_authority": {
                  "source": "https://arxiv.org/pdf/1606.00818v2", "page_1_based": 4,
                  "source_pdf_sha256": "8da70e9e1f767d3e99076e8e14459a0530bfc9ccbabba926f952198f151a7a65",
                  "published_label_V": 1.1, "explicit_preprint_label_V": 1.08,
                  "scope": "Preprint Fig. 1h gives the unrounded bias; published right plateaus independently confirm it",
              },
              "alignment_npz": str(alignment) if alignment else None,
              "alignment_sha256": hashlib.sha256(alignment.read_bytes()).hexdigest() if alignment else None,
              "pdf_axes": {"x0": x0, "y0": y0, "points_per_nm": xscale,
                           "points_per_V": yscale}, "curves": []}
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.8), layout="constrained", sharex=True, sharey=True)
    for axis, (branch, voltage, first, stop) in zip(axes.flat, POTENTIAL_CURVES):
        outlines = [outline_points(drawings[index]["items"]) for index in range(first, stop)]
        rows = []
        for position in np.arange(1.0, 800.0):
            intervals = [interval for outline in outlines
                         if (interval := slice_outline(outline, x0 + position * xscale, envelope=True)) is not None]
            if not intervals:
                continue
            # Adjacent filled stroke segments overlap at their joins.
            low = (min(interval[0] for interval in intervals) - y0) / yscale
            high = (max(interval[1] for interval in intervals) - y0) / yscale
            rows.append((position, (low + high) / 2, low, high))
        data = np.asarray(rows)
        if len(data) < 780:
            raise ValueError(f"incomplete potential stroke {branch} {voltage}: {len(data)} samples")
        position, phi, low, high = data.T
        name = f"fig1h_{branch}_{voltage:.1f}V"
        np.savetxt(out_dir / f"{name}.csv", data, delimiter=",",
                   header="x_nm,phi_V,phi_low_V,phi_high_V", comments="")
        record = {"branch": branch, "voltage_V": voltage, "drawing_indices": [first, stop - 1],
                  "phi_250_400_550nm_V": np.interp([250, 400, 550], position, phi).tolist(),
                  "bias_from_right_plateau_V": float(1.3 - np.median(phi[position >= 750])),
                  "E_250_to_550nm_V_m": float(-np.diff(np.interp([250, 550], position, phi))[0] / 300e-9)}
        comparison_voltage = 1.08 if voltage == 1.1 else voltage
        if abs(record["bias_from_right_plateau_V"] - comparison_voltage) > 1e-3:
            raise ValueError("potential plateau contradicts the source-declared comparison voltage")
        record["comparison_voltage_V"] = comparison_voltage
        axis.fill_between(position, low, high, color="black", alpha=0.2)
        axis.plot(position, phi, color="black", label="Paper vector")
        if simulation is not None:
            index = int(np.argmin(abs(simulation[f"V_{branch}"] - comparison_voltage)))
            if abs(simulation[f"V_{branch}"][index] - comparison_voltage) > 1e-8:
                raise ValueError(f"simulation is missing {branch} {comparison_voltage} V")
            sim_phi = np.interp(position, simulation["x"] * 1e9, simulation[f"phi_{branch}"][index])
            absorber = (position >= 250) & (position <= 550)
            record.update({
                "simulation_phi_250_400_550nm_V": np.interp([250, 400, 550], position, sim_phi).tolist(),
                "absorber_max_abs_error_V": float(np.max(abs(sim_phi[absorber] - phi[absorber]))),
                "absorber_RMSE_V": float(np.sqrt(np.mean((sim_phi[absorber] - phi[absorber]) ** 2))),
                "simulation_E_250_to_550nm_V_m": float(-np.diff(np.interp([250, 550], position, sim_phi))[0] / 300e-9),
            })
            axis.plot(position, sim_phi, color="#0072b2", linestyle="--", label="SolarLab")
        report["curves"].append(record)
        axis.set(title=f"{branch}, {comparison_voltage:g} V", xlabel="Position (nm)", ylabel="Potential (V)")
        axis.axvline(200, color="0.8", linewidth=0.6)
        axis.axvline(600, color="0.8", linewidth=0.6)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.savefig(out_dir / "potential-comparison.png", dpi=180)
    plt.close(fig)
    if simulation is not None:
        simulation.close()
    with (out_dir / "potential-reference.json").open("w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def compare_jv(curves, out_dir: Path, control: Path, hysteretic: Path, panels):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    report = {"source_pdf_sha256": PDF_SHA256, "status": "diagnostic, not certified reproduction", "cases": {}}
    for axis, panel, source in zip(axes, ("fig1e", "fig1f"), (control, hysteretic)):
        record = {"npz": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "branches": {}}
        with np.load(source) as simulation:
            for branch, color in (("fwd", "#0072b2"), ("rev", "#b2182b")):
                reference_v, reference_j, low, high = curves[f"{panel}_{branch}"].T
                voltage, current = simulation[f"V_{branch}"], simulation[f"J_{branch}"]
                if not np.all(np.isfinite(current)):
                    raise ValueError("cannot compare a branch with nonfinite current")
                order = np.argsort(voltage)
                voltage, current = voltage[order], current[order]
                if voltage[0] > reference_v[0] or voltage[-1] < reference_v[-1]:
                    raise ValueError("simulation does not cover the reference voltage interval")
                sampled = np.interp(reference_v, voltage, current)
                stroke = panels[panel]["branches"][branch]["stroke_domain"]
                record["branches"][branch] = compare_reference_samples(curves[f"{panel}_{branch}"], sampled, stroke)
                domain, midpoint = reference_sample_masks(reference_v, stroke)
                axis.fill_between(reference_v[domain], low[domain] / 10, high[domain] / 10, color=color, alpha=0.12)
                axis.plot(reference_v[midpoint], reference_j[midpoint] / 10, color=color, label=f"Paper {branch}")
                axis.plot(voltage, current / 10, "--", color=color, label=f"SolarLab {branch}")
        report["cases"][panel] = record
        axis.set(xlim=(-0.2, 1.15), ylim=(-5, 18), xlabel="Voltage (V)",
                 ylabel="Delivered current (mA/cm$^2$)", title=panel)
        axis.axhline(0, color="0.7", linewidth=0.6)
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(out_dir / "jv-comparison.png", dpi=200)
    plt.close(fig)
    with (out_dir / "jv-comparison.json").open("w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


def extract_supplementary_ions(pdf: Path, out_dir: Path, alignment: Path | None):
    """Read S5b's six positive-ion profiles using its own plotted legend."""
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != SUPPLEMENT_SHA256:
        raise ValueError("supplement does not match the verified source")
    document = fitz.open(pdf)
    page = document[3]
    if "Supplementary Figure 5" not in page.get_text():
        raise ValueError("incorrect supplementary figure page")
    drawings = page.get_drawings()
    def center(index, axis):
        rect = drawings[index]["rect"]
        return (rect[axis] + rect[axis + 2]) / 2
    x0 = center(827, 0)
    xscale = (center(828, 0) - x0) / 12
    y0 = center(829, 1)
    yscale = (y0 - center(832, 1)) / 1.5
    report = {
        "source_pdf_sha256": SUPPLEMENT_SHA256, "source_panel": "S5b",
        "source_species": "positive a in paper notation; P/c in SolarLab",
        "legend_caution": "The ion legend says 1.0 V, while the caption lists 1.1 V; retain the ion legend without relabelling.",
        "comparison_scope": "intrinsic-region curves only; excludes the immobile ion baseline plotted in the p-contact",
        "uncertainty_scope": "graphical stroke width only",
        "alignment_npz": str(alignment) if alignment else None,
        "alignment_sha256": hashlib.sha256(alignment.read_bytes()).hexdigest() if alignment else None,
        "pdf_axes": {"x_198nm": x0, "points_per_nm": xscale, "y_1e19_cm3": y0,
                     "points_per_1e19_cm3": yscale},
        "curves": [],
    }
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), layout="constrained", sharex=True, sharey=True)
    simulation = np.load(alignment) if alignment else None
    specifications = [("fwd", 0, 866), ("fwd", 0.4, 867), ("fwd", 1.0, 868),
                      ("rev", 1.0, 869), ("rev", 0.4, 870), ("rev", 0, 871)]
    for axis, (branch, voltage, index) in zip(axes.flat, specifications):
        outline = outline_points(drawings[index]["items"])
        rows = []
        for position in np.linspace(200.1668, 209.9, 975):
            interval = slice_outline(outline, x0 + (position - 198) * xscale, envelope=True)
            if interval is not None:
                high, low = (1 + (y0 - np.asarray(interval)) / yscale) * 1e25
                rows.append([position, (low + high) / 2, low, high])
        data = np.asarray(rows)
        if len(data) < 950:
            raise ValueError("incomplete supplementary ion stroke")
        position, density, low, high = data.T
        np.savetxt(out_dir / f"figS5b_{branch}_{voltage:.1f}V.csv", data, delimiter=",",
                   header="x_nm,P_m3,P_low_m3,P_high_m3", comments="")
        record = {"branch": branch, "legend_voltage_V": voltage, "drawing_index": index,
                  "density_at_200p1668nm_m3": float(density[0])}
        axis.fill_between(position, low / 1e25, high / 1e25, color="black", alpha=0.2)
        axis.plot(position, density / 1e25, color="black", label="Paper vector")
        if simulation is not None:
            sample = int(np.argmin(abs(simulation[f"V_{branch}"] - voltage)))
            if abs(simulation[f"V_{branch}"][sample] - voltage) > 1e-8:
                raise ValueError("simulation has no exact labelled voltage sample")
            actual = np.interp(position, simulation["x"] * 1e9, simulation[f"P_{branch}"][sample])
            record["RMSE_scaled_by_1e25_m3"] = float(np.sqrt(np.mean(((actual - density) / 1e25) ** 2)))
            record["simulation_density_at_200p1668nm_m3"] = float(actual[0])
            axis.plot(position, actual / 1e25, "--", color="#0072b2", label="SolarLab")
        axis.set(title=f"{branch}, {voltage:.1f} V", xlabel="Position (nm)",
                 ylabel="Positive ions ($10^{25}$ m$^{-3}$)", xlim=(200, 207), ylim=(0.95, 2.7))
        report["curves"].append(record)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.savefig(out_dir / "ion-profile-comparison.png", dpi=180)
    plt.close(fig)
    if simulation is not None:
        simulation.close()
    document.close()
    with (out_dir / "ion-profile-reference.json").open("w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--alignment-npz", type=Path)
    parser.add_argument("--control-npz", type=Path)
    parser.add_argument("--supplement-pdf", type=Path)
    args = parser.parse_args()
    if args.control_npz and not args.alignment_npz:
        parser.error("control-npz requires alignment-npz for the hysteretic comparison")
    digest = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    if digest != PDF_SHA256:
        raise ValueError("PDF does not match the visually verified source")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    page = fitz.open(args.pdf)[2]
    drawings = page.get_drawings()
    report = {
        "source": "Calado et al., Nature Communications 7, 13831 (2016), Fig. 1e/f",
        "doi": "10.1038/ncomms13831", "source_pdf_sha256": digest,
        "method": "vertical intersections of filled vector stroke outlines; raw midpoints/envelopes retained with endpoint-cap domain annotations",
        "uncertainty_scope": "graphical line width only, not experimental or model uncertainty",
        "panels": {},
    }
    curves = {}

    def center(index, axis):
        rect = drawings[index]["rect"]
        return (rect.x0 + rect.x1) / 2 if axis == "x" else (rect.y0 + rect.y1) / 2

    for panel, indices in PANELS.items():
        x0, y0 = center(indices["zero_x"], "x"), center(indices["zero_y"], "y")
        xscale = (center(indices["tick_x"], "x") - x0) / 0.8
        yscale = (center(indices["tick_y"], "y") - y0) / 15.0
        if not (85 < xscale < 89 and 3.7 < yscale < 4.0):
            raise ValueError("axis calibration does not match the inspected plot")
        metadata = {"pdf_axes": {"x0": x0, "y0": y0, "points_per_V": xscale,
                                 "points_per_mA_cm2": yscale}, "branches": {}}
        for branch in ("fwd", "rev"):
            drawing = drawings[indices[branch]]
            outline = outline_points(drawing["items"])
            caps = stroke_endcaps(outline)
            stroke = {
                "endcap_pdf_points": caps.tolist(),
                "centerline_voltage_interval_V": ((caps.mean(axis=1)[:, 0]-x0)/xscale).tolist(),
                "endcap_voltage_intervals_V": np.sort((caps[:, :, 0]-x0)/xscale, axis=1).tolist(),
                "scope": "Samples outside cap centers are stroke thickness, not a defined J-V centerline; cap intersections bias vertical midpoints",
            }
            rows = []
            for voltage in np.linspace(-0.19, 1.1, 1291):
                interval = slice_outline(outline, x0 + xscale * voltage)
                if interval is not None:
                    low, high = (np.asarray(interval) - y0) * 10.0 / yscale
                    rows.append((voltage, (low + high) / 2, low, high))
            data = np.asarray(rows)
            if len(data) < 100:
                raise ValueError("too few valid vector slices")
            voltage, current, low, high = data.T
            domain, midpoint = reference_sample_masks(voltage, stroke)
            power = voltage * current
            positive = (voltage >= 0) & domain
            metrics = {
                "P_max_W_m2": float(power[positive].max()),
                "P_max_low_W_m2": float((voltage[positive] * low[positive]).max()),
                "P_max_high_W_m2": float((voltage[positive] * high[positive]).max()),
                "J_sc_A_m2": float(np.interp(0.0, voltage, current)),
                "V_oc_V": float(np.interp(0.0, current[::-1], voltage[::-1])),
                "drawing_index": indices[branch], "fill_rgb": drawing["fill"],
                "samples": len(data),
                "centerline_domain_samples": int(domain.sum()),
                "endcap_only_samples": int((~domain).sum()),
                "cap_influenced_domain_samples": int((domain & ~midpoint).sum()),
                "stroke_domain": stroke,
            }
            metadata["branches"][branch] = metrics
            curves[f"{panel}_{branch}"] = data
            np.savetxt(args.out_dir / f"{panel}_{branch}.csv", data, delimiter=",",
                       header="V,J_active_A_m2,J_low_A_m2,J_high_A_m2", comments="")
        forward, reverse = (metadata["branches"][b] for b in ("fwd", "rev"))
        metadata["HI_paper"] = reverse["P_max_W_m2"] / forward["P_max_W_m2"] - 1
        metadata["HI_linewidth_interval"] = [
            reverse["P_max_low_W_m2"] / forward["P_max_high_W_m2"] - 1,
            reverse["P_max_high_W_m2"] / forward["P_max_low_W_m2"] - 1,
        ]
        report["panels"][panel] = metadata
    with (args.out_dir / "reference.json").open("w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for axis, panel in zip(axes, PANELS):
        for branch, color in (("fwd", "black"), ("rev", "#c00000" if panel == "fig1e" else "#0000ff")):
            voltage, current, low, high = curves[f"{panel}_{branch}"].T
            stroke = report["panels"][panel]["branches"][branch]["stroke_domain"]
            domain, midpoint = reference_sample_masks(voltage, stroke)
            axis.fill_between(voltage[domain], low[domain] / 10, high[domain] / 10, color=color, alpha=0.15)
            axis.plot(voltage[midpoint], current[midpoint] / 10, color=color, label=branch)
            axis.scatter(voltage[~domain], current[~domain] / 10, marker="x", s=12, color=color,
                         label="Stroke endcaps" if branch == "fwd" else None)
        axis.set(xlim=(-0.2, 1.15), ylim=(-7, 18), xlabel="Voltage (V)",
                 ylabel="Delivered current density (mA/cm$^2$)", title=panel)
        axis.axhline(0, color="0.6", linewidth=0.6)
        axis.legend(frameon=False)
    fig.savefig(args.out_dir / "vector-reference.png", dpi=200)
    plt.close(fig)
    extract_potentials(drawings, args.out_dir, args.alignment_npz)
    if args.control_npz:
        compare_jv(curves, args.out_dir, args.control_npz, args.alignment_npz, report["panels"])
    if args.supplement_pdf:
        extract_supplementary_ions(args.supplement_pdf, args.out_dir, args.alignment_npz)
    print(json.dumps({key: {"HI": value["HI_paper"], "interval": value["HI_linewidth_interval"]}
                      for key, value in report["panels"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
