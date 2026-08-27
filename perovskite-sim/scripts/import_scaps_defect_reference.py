#!/usr/bin/env python
"""Import independently exported SCAPS S0-S2 defect profiles fail closed."""

from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


_SUITE_SCHEMA = "solarlab.scaps_explicit_defect_reference_suite"
_PARAMETER_SCHEMA = "solarlab.scaps_explicit_defect_parameter_manifest"
_REFERENCE_SCHEMA = "solarlab.scaps_explicit_defect_reference"
_SCENARIO_IDS = ("S0", "S1", "S2")
_EXPECTED_LABELS = {
    "S0": ("neutral", "intrinsic"),
    "S1": ("acceptor", "n_type"),
    "S2": ("donor", "p_type"),
}
_PROFILE_COLUMNS = (
    "position_um",
    "electron_density_cm3",
    "hole_density_cm3",
    "electrostatic_potential_V",
    "conduction_band_eV",
    "valence_band_eV",
    "defect_occupancy",
    "defect_charge_number_cm3",
    "recombination_rate_cm3_s",
)
_UNIT_CONVENTIONS = {
    "conduction_band_eV": "eV",
    "defect_charge_number_cm3": "cm-3",
    "defect_occupancy": "1",
    "electron_density_cm3": "cm-3",
    "electrostatic_potential_V": "V",
    "hole_density_cm3": "cm-3",
    "position_um": "um",
    "recombination_rate_cm3_s": "cm-3 s-1",
    "valence_band_eV": "eV",
}
_SUITE_KEYS = {
    "derived_pn_device",
    "external_reference_contract",
    "scenarios",
    "schema",
    "schema_version",
}
_SUITE_SCENARIO_KEYS = {
    "charge_transition",
    "config_path",
    "config_sha256",
    "doping_polarity",
    "id",
    "purpose",
}
_PARAMETER_KEYS = {
    "comparison_protocol",
    "numerics",
    "scenarios",
    "schema",
    "schema_version",
    "sign_conventions",
    "solver",
    "unit_conventions",
}
_PARAMETER_SCENARIO_KEYS = {
    "canonical_config_sha256",
    "charge_transition",
    "doping_polarity",
    "scaps_parameters",
    "source_deck_format",
    "thickness_um",
}
_PROTOCOL_KEYS = {
    "interpolation_allowed",
    "operating_point",
    "position_tolerance_um",
    "row_policy",
}
_SIGN_KEYS = {
    "defect_charge",
    "electrostatic_potential",
    "position_origin",
    "recombination_rate",
}


def _require_exact_keys(
    raw: Mapping[str, Any],
    expected: set[str],
    *,
    where: str,
) -> None:
    keys = set(raw)
    if keys != expected:
        raise ValueError(
            f"{where} key mismatch; unknown={sorted(keys - expected)}, "
            f"missing={sorted(expected - keys)}"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not finite canonical JSON: {exc}") from exc


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be a readable JSON object") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return raw


def _safe_project_file(project_root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("suite config paths must be project-relative")
    resolved = (project_root / path).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError("suite config path escapes project root") from exc
    if not resolved.is_file():
        raise ValueError(f"suite config does not exist: {relative}")
    return resolved


def _load_suite(path: Path, project_root: Path) -> dict[str, dict[str, str]]:
    raw = _load_json_object(path, label="S0-S2 suite")
    _require_exact_keys(raw, _SUITE_KEYS, where="S0-S2 suite")
    if raw["schema"] != _SUITE_SCHEMA or raw["schema_version"] != "1.0":
        raise ValueError("S0-S2 suite schema/version mismatch")
    contract = raw["external_reference_contract"]
    if not isinstance(contract, dict):
        raise ValueError("suite external_reference_contract must be a mapping")
    if contract.get("required_solver") != "SCAPS-1D":
        raise ValueError("suite external contract must require SCAPS-1D")
    if contract.get("independent_export_attestation_required") is not True:
        raise ValueError("suite must require independent export attestation")
    if contract.get("interpolation_allowed") is not False:
        raise ValueError("suite external contract must prohibit interpolation")
    if tuple(contract.get("raw_profile_columns") or ()) != _PROFILE_COLUMNS:
        raise ValueError("suite raw profile columns do not match importer schema")

    rows = raw["scenarios"]
    if not isinstance(rows, list) or len(rows) != len(_SCENARIO_IDS):
        raise ValueError("S0-S2 suite must contain exactly three scenarios")
    scenarios: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"suite scenario[{index}] must be a mapping")
        _require_exact_keys(
            row,
            _SUITE_SCENARIO_KEYS,
            where=f"suite scenario[{index}]",
        )
        identifier = str(row["id"])
        if identifier in scenarios or identifier not in _EXPECTED_LABELS:
            raise ValueError("suite scenario IDs must be unique S0, S1, S2")
        transition, doping = _EXPECTED_LABELS[identifier]
        if (
            row["charge_transition"] != transition
            or row["doping_polarity"] != doping
        ):
            raise ValueError(f"suite scenario {identifier} label mismatch")
        config_path = str(row["config_path"])
        config_sha = str(row["config_sha256"])
        if _sha256(_safe_project_file(project_root, config_path)) != config_sha:
            raise ValueError(f"suite scenario {identifier} config hash drift")
        scenarios[identifier] = {
            "charge_transition": transition,
            "config_path": config_path,
            "config_sha256": config_sha,
            "doping_polarity": doping,
        }
    if tuple(sorted(scenarios)) != _SCENARIO_IDS:
        raise ValueError("suite must contain S0, S1, and S2")
    return scenarios


def _finite_positive(value: Any, *, where: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{where} must be finite and positive")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} must be finite and positive") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{where} must be finite and positive")
    return number


def _load_parameter_manifest(
    path: Path,
    *,
    suite_scenarios: Mapping[str, Mapping[str, str]],
    solver_version: str,
) -> tuple[dict[str, Any], float]:
    raw = _load_json_object(path, label="SCAPS parameter manifest")
    _require_exact_keys(raw, _PARAMETER_KEYS, where="SCAPS parameter manifest")
    if raw["schema"] != _PARAMETER_SCHEMA or raw["schema_version"] != "1.0":
        raise ValueError("SCAPS parameter manifest schema/version mismatch")

    solver = raw["solver"]
    if not isinstance(solver, dict):
        raise ValueError("parameter manifest solver must be a mapping")
    _require_exact_keys(solver, {"name", "version"}, where="solver")
    if solver["name"] != "SCAPS-1D" or solver["version"] != solver_version:
        raise ValueError("parameter manifest solver name/version mismatch")

    numerics = raw["numerics"]
    if not isinstance(numerics, dict) or not numerics:
        raise ValueError("parameter manifest numerics must be populated")
    _canonical_bytes(numerics)

    units = raw["unit_conventions"]
    if not isinstance(units, dict) or units != _UNIT_CONVENTIONS:
        raise ValueError("parameter manifest unit_conventions mismatch")

    signs = raw["sign_conventions"]
    if not isinstance(signs, dict):
        raise ValueError("parameter manifest sign_conventions must be a mapping")
    _require_exact_keys(signs, _SIGN_KEYS, where="sign_conventions")
    if signs["position_origin"] != "left_contact":
        raise ValueError("position_origin must be left_contact")
    if signs["defect_charge"] != "negative_acceptor_positive_donor":
        raise ValueError("defect charge sign convention mismatch")
    if signs["recombination_rate"] != "positive_net_recombination":
        raise ValueError("recombination sign convention mismatch")
    if not isinstance(signs["electrostatic_potential"], str) or not signs[
        "electrostatic_potential"
    ].strip():
        raise ValueError("electrostatic potential convention must be declared")

    protocol = raw["comparison_protocol"]
    if not isinstance(protocol, dict):
        raise ValueError("comparison_protocol must be a mapping")
    _require_exact_keys(protocol, _PROTOCOL_KEYS, where="comparison_protocol")
    if protocol["interpolation_allowed"] is not False:
        raise ValueError("comparison protocol must prohibit interpolation")
    if protocol["row_policy"] != "direct_export_rows_only":
        raise ValueError("comparison row_policy must be direct_export_rows_only")
    if protocol["operating_point"] != "dark_equilibrium_zero_bias":
        raise ValueError("comparison operating point must be dark equilibrium")
    tolerance = float(protocol["position_tolerance_um"])
    if not math.isfinite(tolerance) or tolerance < 0.0 or tolerance > 1.0e-6:
        raise ValueError("position_tolerance_um must be in [0, 1e-6]")

    rows = raw["scenarios"]
    if not isinstance(rows, dict) or set(rows) != set(_SCENARIO_IDS):
        raise ValueError("parameter manifest scenarios must be exactly S0, S1, S2")
    for identifier in _SCENARIO_IDS:
        row = rows[identifier]
        if not isinstance(row, dict):
            raise ValueError(f"parameter scenario {identifier} must be a mapping")
        _require_exact_keys(
            row,
            _PARAMETER_SCENARIO_KEYS,
            where=f"parameter scenario {identifier}",
        )
        expected = suite_scenarios[identifier]
        if row["canonical_config_sha256"] != expected["config_sha256"]:
            raise ValueError(f"parameter scenario {identifier} config hash mismatch")
        if (
            row["charge_transition"] != expected["charge_transition"]
            or row["doping_polarity"] != expected["doping_polarity"]
        ):
            raise ValueError(f"parameter scenario {identifier} label mismatch")
        _finite_positive(row["thickness_um"], where=f"{identifier}.thickness_um")
        if not isinstance(row["source_deck_format"], str) or not row[
            "source_deck_format"
        ].strip():
            raise ValueError(f"{identifier}.source_deck_format must be declared")
        if not isinstance(row["scaps_parameters"], dict) or not row[
            "scaps_parameters"
        ]:
            raise ValueError(f"{identifier}.scaps_parameters must be populated")
        _canonical_bytes(row["scaps_parameters"])
    return raw, tolerance


def _finite_csv_value(value: str | None, *, field: str, row: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row}: {field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"row {row}: {field} must be finite")
    return number


def _load_profile(
    path: Path,
    *,
    identifier: str,
    thickness_um: float,
    position_tolerance_um: float,
) -> dict[str, list[float]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _PROFILE_COLUMNS:
            raise ValueError(
                f"{identifier} CSV columns must exactly match the S0-S2 contract"
            )
        columns = {name: [] for name in _PROFILE_COLUMNS}
        for row_number, row in enumerate(reader, start=2):
            for field in _PROFILE_COLUMNS:
                columns[field].append(
                    _finite_csv_value(row[field], field=field, row=row_number)
                )
    positions = columns["position_um"]
    if len(positions) < 3:
        raise ValueError(f"{identifier} profile requires at least three direct rows")
    if any(right <= left for left, right in zip(positions, positions[1:])):
        raise ValueError(f"{identifier} position_um must be strictly increasing")
    if not math.isclose(
        positions[0], 0.0, rel_tol=0.0, abs_tol=position_tolerance_um
    ):
        raise ValueError(f"{identifier} profile must start at the left contact")
    if not math.isclose(
        positions[-1],
        thickness_um,
        rel_tol=0.0,
        abs_tol=position_tolerance_um,
    ):
        raise ValueError(f"{identifier} profile thickness mismatch")
    if any(value <= 0.0 for value in columns["electron_density_cm3"]):
        raise ValueError(f"{identifier} electron density must be positive")
    if any(value <= 0.0 for value in columns["hole_density_cm3"]):
        raise ValueError(f"{identifier} hole density must be positive")
    if any(
        not 0.0 <= value <= 1.0 for value in columns["defect_occupancy"]
    ):
        raise ValueError(f"{identifier} defect occupancy must lie in [0, 1]")
    if any(
        conduction <= valence
        for conduction, valence in zip(
            columns["conduction_band_eV"],
            columns["valence_band_eV"],
        )
    ):
        raise ValueError(f"{identifier} conduction band must exceed valence band")

    charge = columns["defect_charge_number_cm3"]
    if identifier == "S0" and any(value != 0.0 for value in charge):
        raise ValueError("S0 neutral defect charge must be exactly zero")
    if identifier == "S1" and (
        any(value > 0.0 for value in charge)
        or not any(value < 0.0 for value in charge)
    ):
        raise ValueError("S1 acceptor defect charge must be nonpositive and nonzero")
    if identifier == "S2" and (
        any(value < 0.0 for value in charge)
        or not any(value > 0.0 for value in charge)
    ):
        raise ValueError("S2 donor defect charge must be nonnegative and nonzero")
    return columns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--parameter-manifest", type=Path, required=True)
    for identifier in _SCENARIO_IDS:
        flag = identifier.lower()
        parser.add_argument(f"--{flag}-csv", type=Path, required=True)
        parser.add_argument(f"--{flag}-source-deck", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--solver-version", required=True)
    parser.add_argument("--extracted-at", required=True, help="ISO date")
    parser.add_argument("--operator", required=True)
    parser.add_argument(
        "--confirm-independent-scaps-export",
        action="store_true",
        help="attest that every CSV was exported by SCAPS independently",
    )
    parser.add_argument(
        "--confirm-direct-unmodified-rows",
        action="store_true",
        help="attest that rows were not interpolated, fitted, or resampled",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.confirm_independent_scaps_export:
        raise ValueError("--confirm-independent-scaps-export is required")
    if not args.confirm_direct_unmodified_rows:
        raise ValueError("--confirm-direct-unmodified-rows is required")
    if not args.operator.strip():
        raise ValueError("operator must not be empty")
    if not args.solver_version.strip():
        raise ValueError("solver-version must not be empty")
    try:
        date.fromisoformat(args.extracted_at)
    except ValueError as exc:
        raise ValueError("extracted-at must be an ISO date") from exc

    project_root = args.project_root.resolve()
    suite_path = args.suite.resolve()
    parameter_path = args.parameter_manifest.resolve()
    for label, path in (
        ("project root", project_root),
        ("suite", suite_path),
        ("parameter manifest", parameter_path),
    ):
        if label == "project root":
            if not path.is_dir():
                raise ValueError(f"project root does not exist: {path}")
        elif not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    try:
        suite_relative = suite_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("suite must be inside project root") from exc

    suite_scenarios = _load_suite(suite_path, project_root)
    parameter_manifest, position_tolerance = _load_parameter_manifest(
        parameter_path,
        suite_scenarios=suite_scenarios,
        solver_version=args.solver_version,
    )

    csv_paths = {
        identifier: getattr(args, f"{identifier.lower()}_csv").resolve()
        for identifier in _SCENARIO_IDS
    }
    deck_paths = {
        identifier: getattr(args, f"{identifier.lower()}_source_deck").resolve()
        for identifier in _SCENARIO_IDS
    }
    for identifier in _SCENARIO_IDS:
        for label, path in (
            ("CSV", csv_paths[identifier]),
            ("source deck", deck_paths[identifier]),
        ):
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"{identifier} {label} is missing or empty: {path}")

    scenario_payload: dict[str, Any] = {}
    for identifier in _SCENARIO_IDS:
        manifest_row = parameter_manifest["scenarios"][identifier]
        profile = _load_profile(
            csv_paths[identifier],
            identifier=identifier,
            thickness_um=float(manifest_row["thickness_um"]),
            position_tolerance_um=position_tolerance,
        )
        source = suite_scenarios[identifier]
        scenario_payload[identifier] = {
            "canonical_config_path": source["config_path"],
            "canonical_config_sha256": source["config_sha256"],
            "charge_transition": source["charge_transition"],
            "doping_polarity": source["doping_polarity"],
            "profile": profile,
            "row_count": len(profile["position_um"]),
            "source_deck": {
                "name": deck_paths[identifier].name,
                "sha256": _sha256(deck_paths[identifier]),
            },
            "source_export": {
                "name": csv_paths[identifier].name,
                "sha256": _sha256(csv_paths[identifier]),
            },
        }

    unsigned = {
        "attestation": {
            "direct_unmodified_rows": True,
            "independent_scaps_export": True,
            "operator": args.operator.strip(),
        },
        "extracted_at": args.extracted_at,
        "parameter_manifest": parameter_manifest,
        "parameter_manifest_source": {
            "name": parameter_path.name,
            "sha256": _sha256(parameter_path),
        },
        "scenarios": scenario_payload,
        "schema": _REFERENCE_SCHEMA,
        "schema_version": "1.0",
        "solver": {"name": "SCAPS-1D", "version": args.solver_version},
        "suite": {
            "path": suite_relative.as_posix(),
            "sha256": _sha256(suite_path),
        },
    }
    payload = {
        **unsigned,
        "reference_content_sha256": _content_sha256(unsigned),
    }
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
        raise ValueError(f"refusing to overwrite existing reference: {args.out}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
