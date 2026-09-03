#!/usr/bin/env python3
"""Compare an imported SCAPS defect reference against SolarLab, fail closed.

Consumes the artifact written by ``import_scaps_defect_reference.py`` or
``import_scaps_multivalent_defect_reference.py``, re-solves the frozen
scenarios on the pre-registered QF/DC configuration, and issues per-column
PASS / FAIL / INDECISIVE_GRID verdicts against the thresholds frozen in
``reproducibility/scaps_defect_comparison_thresholds.json`` BEFORE any
external data existed. Rationale and alignment policy:
``docs/scaps-defect-comparison-preregistration.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import scipy.linalg  # noqa: E402,F401  (register BLAS before pinning)

try:  # pragma: no cover - exercised only when threadpoolctl is present
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover
    threadpool_limits = None

from perovskite_sim.constants import Q  # noqa: E402
from perovskite_sim.experiments.quasi_fermi_steady_state import (  # noqa: E402
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.physics.recombination import total_recombination  # noqa: E402
from perovskite_sim.solver.mol import (  # noqa: E402
    EXPLICIT_DEFECT_CHARGE_QF_DC,
    build_material_arrays,
)
from perovskite_sim.validation import (  # noqa: E402
    charged_defect_refinement,
    multivalent_defect_refinement,
)
from perovskite_sim.validation.numerical_certificate import (  # noqa: E402
    load_refinement_registry,
)


_REPORT_SCHEMA = "solarlab.scaps_defect_comparison_report"
_THRESHOLDS_SCHEMA = "solarlab.scaps_defect_comparison_thresholds"
_THRESHOLDS_RELATIVE = "reproducibility/scaps_defect_comparison_thresholds.json"
_REGISTRY_RELATIVE = "reproducibility/numerical_refinement_registry.yaml"
_MODES = {
    "solarlab.scaps_explicit_defect_reference": "s0s2",
    "solarlab.scaps_multivalent_defect_reference": "multivalent",
}
_LANES = {
    "s0s2": "charged-explicit-defect-qf-dc-v1",
    "multivalent": "multivalent-explicit-defect-qf-dc-v1",
}
_SUITES = {
    "s0s2": "reproducibility/scaps_defect_s0_s2_suite.json",
    "multivalent": "reproducibility/scaps_multivalent_defect_suite.json",
}
_SCENARIO_IDS = {
    "s0s2": ("S0", "S1", "S2"),
    "multivalent": ("M1", "M2", "M3"),
}
_NUMERIC_COLUMNS = (
    "position_um",
    "electron_density_cm3",
    "hole_density_cm3",
    "electrostatic_potential_V",
    "conduction_band_eV",
    "valence_band_eV",
    "defect_charge_number_cm3",
    "recombination_rate_cm3_s",
)
_THRESHOLD_KEYS = {
    "schema",
    "schema_version",
    "preregistered_at",
    "rationale",
    "grid",
    "grid_sensitivity_fraction_of_threshold",
    "solve_controls",
    "columns",
}


def _content_sha256(value) -> str:
    canonical = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be a readable JSON object") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return raw


def _load_thresholds(path: Path) -> dict:
    raw = _load_json_object(path, label="comparison thresholds")
    keys = set(raw)
    if keys != _THRESHOLD_KEYS:
        raise ValueError(
            "comparison thresholds key mismatch; "
            f"unknown={sorted(keys - _THRESHOLD_KEYS)}, "
            f"missing={sorted(_THRESHOLD_KEYS - keys)}"
        )
    if raw["schema"] != _THRESHOLDS_SCHEMA or raw["schema_version"] != "1.0":
        raise ValueError("comparison thresholds schema/version mismatch")
    for mode in ("s0s2", "multivalent"):
        if mode not in raw["grid"] or mode not in raw["columns"]:
            raise ValueError(f"comparison thresholds must cover mode {mode}")
    fraction = float(raw["grid_sensitivity_fraction_of_threshold"])
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("grid sensitivity fraction must lie in (0, 1)")
    return raw


def _load_scenarios(project_root: Path, mode: str) -> dict:
    registry = load_refinement_registry(
        project_root / _REGISTRY_RELATIVE,
        project_root=project_root,
    )
    lane = registry.lane(_LANES[mode])
    loader = (
        charged_defect_refinement
        if mode == "s0s2"
        else multivalent_defect_refinement
    )
    _suite, ordered = loader._load_suite(lane, project_root)
    return {item.identifier: item for item in ordered}


def _neutral_occupancy(material, n: np.ndarray, p: np.ndarray) -> np.ndarray:
    species = material.neutral_bulk_defects.species[0]
    numerator = n / species.tau_n_s + species.p1_m3 / species.tau_p_s
    denominator = (n + species.n1_m3) / species.tau_n_s + (
        p + species.p1_m3
    ) / species.tau_p_s
    return numerator / denominator


def _solve_columns(
    scenario,
    *,
    mode: str,
    intervals: int,
    grid_alpha: float,
    solve_controls: dict,
) -> dict[str, np.ndarray]:
    """Solve one frozen scenario dark at 0 V; return nodal CSV-unit columns."""
    stack = scenario.stack
    grid = charged_defect_refinement._grid(stack, intervals, grid_alpha)
    neutral = mode == "s0s2" and scenario.transition == "neutral"
    closure = {} if neutral else {
        "explicit_defect_charge_closure": EXPLICIT_DEFECT_CHARGE_QF_DC
    }
    material = build_material_arrays(grid, stack, **closure)
    result = solve_quasi_fermi_steady_state(
        grid,
        stack,
        V_app=0.0,
        illuminated=False,
        mat=material,
        **solve_controls,
    )
    count = len(grid)
    n = np.asarray(result.y[:count], dtype=float)
    p = np.asarray(result.y[count : 2 * count], dtype=float)
    phi = np.asarray(result.phi, dtype=float)
    params = stack.layers[0].params
    conduction = -params.chi - phi
    valence = conduction - params.Eg
    columns: dict[str, np.ndarray] = {
        "position_um": grid * 1.0e6,
        "electron_density_cm3": n * 1.0e-6,
        "hole_density_cm3": p * 1.0e-6,
        "electrostatic_potential_V": phi,
        "conduction_band_eV": conduction,
        "valence_band_eV": valence,
    }
    if mode == "s0s2":
        if neutral:
            occupancy = _neutral_occupancy(material, n, p)
            charge_m3 = np.zeros(count)
            species = material.neutral_bulk_defects.species[0]
            rate_m3_s = total_recombination(
                n,
                p,
                material.ni_sq,
                species.tau_n_s,
                species.tau_p_s,
                species.n1_m3,
                species.p1_m3,
                material.B_rad,
                material.C_n,
                material.C_p,
                neutral_bulk_defects=material.neutral_bulk_defects,
            )
        else:
            diagnostics = result.bulk_defect_diagnostics
            if diagnostics is None:
                raise RuntimeError(
                    f"{scenario.identifier} returned no charged-defect diagnostics"
                )
            occupancy = np.asarray(diagnostics.occupancy[0], dtype=float)
            charge_m3 = diagnostics.total_charge_density_C_m3 / Q
            rate_m3_s = diagnostics.total_recombination_rate_m3_s
        columns["defect_occupancy"] = occupancy
    else:
        diagnostics = result.multivalent_bulk_defect_diagnostics
        if diagnostics is None:
            raise RuntimeError(
                f"{scenario.identifier} returned no multivalent diagnostics"
            )
        columns["state_probability"] = np.asarray(
            diagnostics.state_probability[0], dtype=float
        )
        charge_m3 = diagnostics.total_charge_density_C_m3 / Q
        rate_m3_s = diagnostics.total_recombination_rate_m3_s
    columns["defect_charge_number_cm3"] = np.asarray(charge_m3) * 1.0e-6
    columns["recombination_rate_cm3_s"] = np.asarray(rate_m3_s) * 1.0e-6
    return columns


def _interpolate_columns(
    nodal: dict[str, np.ndarray],
    positions_um: np.ndarray,
    *,
    mode: str,
) -> dict:
    x = nodal["position_um"]
    out: dict = {"position_um": [float(v) for v in positions_um]}
    for name in ("electron_density_cm3", "hole_density_cm3"):
        # Densities span decades: interpolate in log10 space.
        log_values = np.interp(positions_um, x, np.log10(nodal[name]))
        out[name] = [float(v) for v in 10.0**log_values]
    for name in (
        "electrostatic_potential_V",
        "conduction_band_eV",
        "valence_band_eV",
        "defect_charge_number_cm3",
        "recombination_rate_cm3_s",
    ):
        out[name] = [float(v) for v in np.interp(positions_um, x, nodal[name])]
    if mode == "s0s2":
        out["defect_occupancy"] = [
            float(v) for v in np.interp(positions_um, x, nodal["defect_occupancy"])
        ]
    else:
        states = nodal["state_probability"]
        rows = np.stack(
            [np.interp(positions_um, x, states[index]) for index in range(3)],
            axis=1,
        )
        out["charge_state_occupation_fractions"] = [
            [float(v) for v in row] for row in rows
        ]
        out["charge_state_occupation_fraction_per_state"] = [
            "|".join(repr(float(v)) for v in row) for row in rows
        ]
    return out


def compute_solarlab_profile(
    project_root: Path,
    *,
    mode: str,
    identifier: str,
    positions_um,
    intervals: int,
) -> dict:
    """Public probe: SolarLab's frozen-scenario profile at given positions."""
    if mode not in _LANES:
        raise ValueError(f"mode must be one of {sorted(_LANES)}")
    thresholds = _load_thresholds(project_root / _THRESHOLDS_RELATIVE)
    scenarios = _load_scenarios(project_root, mode)
    if identifier not in scenarios:
        raise ValueError(f"unknown scenario {identifier} for mode {mode}")
    nodal = _solve_columns(
        scenarios[identifier],
        mode=mode,
        intervals=int(intervals),
        grid_alpha=float(thresholds["grid"][mode]["grid_alpha"]),
        solve_controls=dict(thresholds["solve_controls"]),
    )
    return _interpolate_columns(
        nodal, np.asarray(positions_um, dtype=float), mode=mode
    )


def _validate_reference(reference: dict, project_root: Path) -> str:
    unsigned = {
        key: value
        for key, value in reference.items()
        if key != "reference_content_sha256"
    }
    if reference.get("reference_content_sha256") != _content_sha256(unsigned):
        raise ValueError("reference content hash mismatch; artifact was modified")
    mode = _MODES.get(reference.get("schema"))
    if mode is None or reference.get("schema_version") != "1.0":
        raise ValueError("reference schema/version is not a known importer output")
    suite = reference.get("suite")
    if not isinstance(suite, dict):
        raise ValueError("reference must record its suite path and hash")
    if suite.get("path") != _SUITES[mode]:
        raise ValueError(f"reference suite path must be {_SUITES[mode]}")
    live_hash = _file_sha256(project_root / _SUITES[mode])
    if suite.get("sha256") != live_hash:
        raise ValueError(
            "reference suite hash does not match the repository suite "
            "(suite hash drift)"
        )
    scenarios = reference.get("scenarios")
    if not isinstance(scenarios, dict) or set(scenarios) != set(
        _SCENARIO_IDS[mode]
    ):
        raise ValueError(
            f"reference scenarios must be exactly {_SCENARIO_IDS[mode]}"
        )
    return mode


def _reference_profile(reference: dict, identifier: str, mode: str) -> dict:
    row = reference["scenarios"][identifier]
    if not isinstance(row, dict) or not isinstance(row.get("profile"), dict):
        raise ValueError(f"reference scenario {identifier} must carry a profile")
    profile = row["profile"]
    for name in _NUMERIC_COLUMNS:
        values = profile.get(name)
        if not isinstance(values, list) or len(values) < 3:
            raise ValueError(
                f"{identifier} profile column {name} must hold >= 3 rows"
            )
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in values
        ):
            raise ValueError(f"{identifier} profile column {name} must be finite")
    positions = profile["position_um"]
    if any(b <= a for a, b in zip(positions, positions[1:])):
        raise ValueError(f"{identifier} positions must be strictly increasing")
    count = len(positions)
    if any(len(profile[name]) != count for name in _NUMERIC_COLUMNS):
        raise ValueError(f"{identifier} profile columns must share one length")
    if mode == "s0s2":
        occupancy = profile.get("defect_occupancy")
        if not isinstance(occupancy, list) or len(occupancy) != count:
            raise ValueError(f"{identifier} profile must carry defect_occupancy")
    else:
        fractions = profile.get("charge_state_occupation_fractions")
        if (
            not isinstance(fractions, list)
            or len(fractions) != count
            or any(
                not isinstance(row_values, list) or len(row_values) != 3
                for row_values in fractions
            )
        ):
            raise ValueError(
                f"{identifier} profile must carry parsed three-state fractions"
            )
    return profile


def _linf(a, b) -> float:
    return float(
        np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
    )


def _anchored(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array - array[0]


def _column_metrics(
    reference_profile: dict,
    solarlab: dict,
    sensitivity: dict,
    *,
    mode: str,
    total_density_cm3: float,
) -> dict[str, tuple[float, float]]:
    """Per column: (reference-vs-SolarLab value, SolarLab grid sensitivity)."""

    def log10_pair(name):
        ref = np.log10(np.asarray(reference_profile[name], dtype=float))
        primary = np.log10(np.asarray(solarlab[name], dtype=float))
        secondary = np.log10(np.asarray(sensitivity[name], dtype=float))
        return _linf(ref, primary), _linf(primary, secondary)

    def anchored_pair(name):
        ref = _anchored(reference_profile[name])
        primary = _anchored(solarlab[name])
        secondary = _anchored(sensitivity[name])
        return _linf(ref, primary), _linf(primary, secondary)

    gap_ref = np.asarray(reference_profile["conduction_band_eV"]) - np.asarray(
        reference_profile["valence_band_eV"]
    )
    gap_primary = np.asarray(solarlab["conduction_band_eV"]) - np.asarray(
        solarlab["valence_band_eV"]
    )
    gap_secondary = np.asarray(sensitivity["conduction_band_eV"]) - np.asarray(
        sensitivity["valence_band_eV"]
    )

    def normalized_charge_pair():
        ref = np.asarray(reference_profile["defect_charge_number_cm3"])
        primary = np.asarray(solarlab["defect_charge_number_cm3"])
        secondary = np.asarray(sensitivity["defect_charge_number_cm3"])
        return (
            _linf(ref / total_density_cm3, primary / total_density_cm3),
            _linf(primary / total_density_cm3, secondary / total_density_cm3),
        )

    def near_zero_pair():
        ref = float(
            np.max(np.abs(np.asarray(reference_profile["recombination_rate_cm3_s"])))
        )
        primary = float(
            np.max(np.abs(np.asarray(solarlab["recombination_rate_cm3_s"])))
        )
        secondary = float(
            np.max(np.abs(np.asarray(sensitivity["recombination_rate_cm3_s"])))
        )
        return max(ref, primary), abs(primary - secondary)

    metrics = {
        "electron_density_log10_linf": log10_pair("electron_density_cm3"),
        "hole_density_log10_linf": log10_pair("hole_density_cm3"),
        "electrostatic_potential_left_anchored_linf_V": anchored_pair(
            "electrostatic_potential_V"
        ),
        "conduction_band_left_anchored_linf_eV": anchored_pair(
            "conduction_band_eV"
        ),
        "valence_band_left_anchored_linf_eV": anchored_pair("valence_band_eV"),
        "band_gap_linf_eV": (
            _linf(gap_ref, gap_primary),
            _linf(gap_primary, gap_secondary),
        ),
        "normalized_defect_charge_linf": normalized_charge_pair(),
        "recombination_rate_near_zero_cm3_s": near_zero_pair(),
    }
    if mode == "s0s2":
        metrics["defect_occupancy_linf"] = (
            _linf(reference_profile["defect_occupancy"], solarlab["defect_occupancy"]),
            _linf(solarlab["defect_occupancy"], sensitivity["defect_occupancy"]),
        )
    else:
        ref = np.asarray(
            reference_profile["charge_state_occupation_fractions"], dtype=float
        )
        primary = np.asarray(
            solarlab["charge_state_occupation_fractions"], dtype=float
        )
        secondary = np.asarray(
            sensitivity["charge_state_occupation_fractions"], dtype=float
        )
        metrics["charge_state_occupation_fraction_linf"] = (
            _linf(ref, primary),
            _linf(primary, secondary),
        )
    return metrics


def _total_density_cm3(scenario, mode: str) -> float:
    species = scenario.stack.layers[0].params.bulk_defects[0]
    if mode == "s0s2":
        return float(species.distribution.total_density_m3) * 1.0e-6
    return float(species.total_density_m3) * 1.0e-6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--grid-intervals",
        type=int,
        default=None,
        help="override the pre-registered grid (report loses preregistered status)",
    )
    parser.add_argument(
        "--sensitivity-grid-intervals",
        type=int,
        default=None,
        help="override the pre-registered sensitivity grid",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    reference_path = args.reference.resolve()
    thresholds_path = project_root / _THRESHOLDS_RELATIVE
    if not reference_path.is_file():
        raise ValueError(f"reference does not exist: {reference_path}")

    thresholds = _load_thresholds(thresholds_path)
    reference = _load_json_object(reference_path, label="SCAPS reference")
    mode = _validate_reference(reference, project_root)
    scenarios = _load_scenarios(project_root, mode)
    suite_scenarios = json.loads(
        (project_root / _SUITES[mode]).read_text(encoding="utf-8")
    )["scenarios"]
    suite_hashes = {row["id"]: row["config_sha256"] for row in suite_scenarios}
    for identifier in _SCENARIO_IDS[mode]:
        recorded = reference["scenarios"][identifier].get("canonical_config_sha256")
        if recorded != suite_hashes[identifier]:
            raise ValueError(f"reference {identifier} config hash mismatch")

    grid_config = thresholds["grid"][mode]
    preregistered = (
        args.grid_intervals is None and args.sensitivity_grid_intervals is None
    )
    intervals = (
        int(grid_config["intervals_per_layer"])
        if args.grid_intervals is None
        else int(args.grid_intervals)
    )
    sensitivity_intervals = (
        int(grid_config["sensitivity_intervals_per_layer"])
        if args.sensitivity_grid_intervals is None
        else int(args.sensitivity_grid_intervals)
    )
    if intervals < 2 or sensitivity_intervals < 2:
        raise ValueError("grid intervals must be at least 2")
    grid_alpha = float(grid_config["grid_alpha"])
    solve_controls = dict(thresholds["solve_controls"])
    fraction = float(thresholds["grid_sensitivity_fraction_of_threshold"])
    column_thresholds = thresholds["columns"][mode]

    def evaluate_all():
        scenario_reports: dict = {}
        verdicts: list[str] = []
        for identifier in _SCENARIO_IDS[mode]:
            profile = _reference_profile(reference, identifier, mode)
            positions = np.asarray(profile["position_um"], dtype=float)
            scenario = scenarios[identifier]
            primary = _interpolate_columns(
                _solve_columns(
                    scenario,
                    mode=mode,
                    intervals=intervals,
                    grid_alpha=grid_alpha,
                    solve_controls=solve_controls,
                ),
                positions,
                mode=mode,
            )
            secondary = _interpolate_columns(
                _solve_columns(
                    scenario,
                    mode=mode,
                    intervals=sensitivity_intervals,
                    grid_alpha=grid_alpha,
                    solve_controls=solve_controls,
                ),
                positions,
                mode=mode,
            )
            metrics = _column_metrics(
                profile,
                primary,
                secondary,
                mode=mode,
                total_density_cm3=_total_density_cm3(scenario, mode),
            )
            columns: dict = {}
            for name, threshold in column_thresholds.items():
                value, sensitivity = metrics[name]
                if sensitivity > fraction * float(threshold):
                    verdict = "INDECISIVE_GRID"
                elif value <= float(threshold):
                    verdict = "PASS"
                else:
                    verdict = "FAIL"
                verdicts.append(verdict)
                columns[name] = {
                    "value": value,
                    "threshold": float(threshold),
                    "grid_sensitivity": sensitivity,
                    "verdict": verdict,
                }
            scenario_reports[identifier] = {
                "row_count": len(positions),
                "columns": columns,
            }
        return scenario_reports, verdicts

    if threadpool_limits is not None:
        with threadpool_limits(limits=1, user_api="blas"):
            scenario_reports, verdicts = evaluate_all()
    else:  # pragma: no cover - threadpoolctl is a test dependency
        scenario_reports, verdicts = evaluate_all()

    if "FAIL" in verdicts:
        overall = "FAIL"
    elif "INDECISIVE_GRID" in verdicts:
        overall = "INDECISIVE_GRID"
    else:
        overall = "PASS"

    unsigned = {
        "schema": _REPORT_SCHEMA,
        "schema_version": "1.0",
        "mode": mode,
        "lane_id": _LANES[mode],
        "grid": {
            "intervals_per_layer": intervals,
            "sensitivity_intervals_per_layer": sensitivity_intervals,
            "grid_alpha": grid_alpha,
        },
        "grid_sensitivity_fraction_of_threshold": fraction,
        "preregistered_settings": preregistered,
        "reference": {
            "path": reference_path.name,
            "sha256": _file_sha256(reference_path),
            "content_sha256": reference["reference_content_sha256"],
        },
        "thresholds": {
            "path": _THRESHOLDS_RELATIVE,
            "sha256": _file_sha256(thresholds_path),
        },
        "suite": {
            "path": _SUITES[mode],
            "sha256": _file_sha256(project_root / _SUITES[mode]),
        },
        "scenarios": scenario_reports,
        "overall_verdict": overall,
    }
    payload = {**unsigned, "comparison_content_sha256": _content_sha256(unsigned)}
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.out.open("x", encoding="ascii") as stream:
            stream.write(encoded)
    except FileExistsError as exc:
        raise ValueError(
            f"refusing to overwrite existing comparison report: {args.out}"
        ) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
