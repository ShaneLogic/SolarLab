#!/usr/bin/env python
"""Build a content-addressed dense SCAPS CBO reference from a raw CSV export."""
from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import json
import math
from pathlib import Path


_REQUIRED_COLUMNS = ("delta_ec_eV", "Jsc_mA_cm2")
_OPTIONAL_COLUMNS = ("Voc_V", "FF_percent", "PCE_percent")
_REQUIRED_MANIFEST_SECTIONS = (
    "layers",
    "contacts",
    "interfaces",
    "illumination",
    "numerics",
    "cbo_scan",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_float(value: str, *, field: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"row {row_number}: {field} must be numeric"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(f"row {row_number}: {field} must be finite")
    return parsed


def _load_points(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        missing = [field for field in _REQUIRED_COLUMNS if field not in fields]
        if missing:
            raise ValueError(
                "SCAPS CSV is missing required columns: " + ", ".join(missing)
            )
        points: list[dict[str, float]] = []
        for row_number, row in enumerate(reader, start=2):
            point = {
                "x": _finite_float(
                    row["delta_ec_eV"],
                    field="delta_ec_eV",
                    row_number=row_number,
                ),
                "Jsc_mA_cm2": _finite_float(
                    row["Jsc_mA_cm2"],
                    field="Jsc_mA_cm2",
                    row_number=row_number,
                ),
            }
            for field in _OPTIONAL_COLUMNS:
                raw = row.get(field)
                if raw is not None and raw.strip():
                    point[field] = _finite_float(
                        raw,
                        field=field,
                        row_number=row_number,
                    )
            points.append(point)
    if len(points) < 3:
        raise ValueError("SCAPS CBO reference requires at least three points")
    deltas = [point["x"] for point in points]
    if any(right <= left for left, right in zip(deltas, deltas[1:])):
        raise ValueError(
            "delta_ec_eV must be unique and strictly increasing; "
            "interpolated or reordered input is not accepted"
        )
    return points


def _load_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("parameter manifest must be valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("parameter manifest must contain a JSON object")
    if manifest.get("schema") != "solarlab.scaps_cbo_parameter_manifest":
        raise ValueError(
            "parameter manifest schema must be "
            "solarlab.scaps_cbo_parameter_manifest"
        )
    if manifest.get("schema_version") != "1.0":
        raise ValueError("parameter manifest schema_version must be 1.0")
    missing = [
        section
        for section in _REQUIRED_MANIFEST_SECTIONS
        if section not in manifest
    ]
    if missing:
        raise ValueError(
            "parameter manifest is missing sections: " + ", ".join(missing)
        )
    empty = [
        section
        for section in _REQUIRED_MANIFEST_SECTIONS
        if not isinstance(manifest[section], (dict, list))
        or not manifest[section]
    ]
    if empty:
        raise ValueError(
            "parameter manifest sections must be populated: "
            + ", ".join(empty)
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--source-deck", type=Path, required=True)
    parser.add_argument("--parameter-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--solver-version", required=True)
    parser.add_argument("--extracted-at", required=True, help="ISO date")
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--illumination", default="AM1.5G")
    parser.add_argument(
        "--boundary-policy",
        choices=("fixed_contacts", "recomputed_built_in"),
        default="fixed_contacts",
    )
    parser.add_argument("--reference-delta-ec-eV", type=float, default=0.0)
    parser.add_argument(
        "--confirm-independent-scaps-export",
        action="store_true",
        help="attest that the CSV contains direct SCAPS outputs, not interpolation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.confirm_independent_scaps_export:
        raise ValueError(
            "--confirm-independent-scaps-export is required for provenance"
        )
    try:
        date.fromisoformat(args.extracted_at)
    except ValueError as exc:
        raise ValueError("extracted-at must be an ISO date") from exc
    if not math.isfinite(args.temperature_K) or args.temperature_K <= 0.0:
        raise ValueError("temperature-K must be finite and positive")
    if not math.isfinite(args.reference_delta_ec_eV):
        raise ValueError("reference-delta-ec-eV must be finite")
    if not args.illumination.strip():
        raise ValueError("illumination must not be empty")

    for name, path in (
        ("csv", args.csv),
        ("source deck", args.source_deck),
        ("parameter manifest", args.parameter_manifest),
    ):
        if not path.is_file():
            raise ValueError(f"{name} does not exist: {path}")

    points = _load_points(args.csv)
    manifest = _load_manifest(args.parameter_manifest)
    cbo_manifest = manifest["cbo_scan"]
    expected_manifest = {
        "delta_ec_convention": "chi_absorber - chi_etl",
        "swept_parameter": "etl_electron_affinity",
        "boundary_policy": args.boundary_policy,
    }
    for field, expected in expected_manifest.items():
        if cbo_manifest.get(field) != expected:
            raise ValueError(
                f"parameter manifest cbo_scan.{field} must be {expected!r}"
            )
    if not any(
        math.isclose(
            point["x"],
            args.reference_delta_ec_eV,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for point in points
    ):
        raise ValueError("SCAPS CSV does not contain the declared reference CBO")

    payload = {
        "schema": "solarlab.scaps_cbo_reference",
        "schema_version": "1.0",
        "source_csv": str(args.csv),
        "source_deck": str(args.source_deck),
        "parameter_manifest": str(args.parameter_manifest),
        "extracted_at": args.extracted_at,
        "cbo_validation": {
            "solver": "SCAPS-1D",
            "solver_version": args.solver_version,
            "delta_ec_convention": "chi_absorber - chi_etl",
            "swept_parameter": "etl_electron_affinity",
            "boundary_policy": args.boundary_policy,
            "reference_delta_ec_eV": args.reference_delta_ec_eV,
            "temperature_K": args.temperature_K,
            "illumination": args.illumination,
            "independently_generated": True,
            "interpolated": False,
            "source_export_sha256": _sha256(args.csv),
            "source_deck_sha256": _sha256(args.source_deck),
            "parameter_manifest_sha256": _sha256(args.parameter_manifest),
        },
        "sweeps": {
            "CHI_ETL": {
                "x_name": "delta_E_C_eV",
                "x_unit": "eV",
                "n_points": len(points),
                "points": points,
            }
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
