#!/usr/bin/env python3
"""Run the fixed Phase-1 real-device RHS regularization studies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from perovskite_sim.validation.regularization_executors import (
    DEVICE_REGULARIZATION_STUDY_IDS,
    run_device_regularization_study,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "regularization-ladders"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute real-device [1, 0.5, 0.25, 0] RHS width ladders and "
            "write immutable content-addressed certificates."
        )
    )
    parser.add_argument(
        "study",
        nargs="?",
        choices=(*DEVICE_REGULARIZATION_STUDY_IDS, "all"),
        default="all",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--allow-noncertified-exit-zero",
        action="store_true",
        help="retain failed/partial status but return exit code zero",
    )
    return parser


def _write_immutable(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = payload.encode("ascii")
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"refusing to overwrite different artifact {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    study_ids = (
        DEVICE_REGULARIZATION_STUDY_IDS if args.study == "all" else (args.study,)
    )
    summaries = []
    all_certified = True
    for study_id in study_ids:
        certificate = run_device_regularization_study(study_id, project_root=ROOT)
        artifact = output_root / study_id / f"{certificate.certificate_sha256}.json"
        _write_immutable(artifact, certificate.canonical_json())
        summaries.append(
            {
                "artifact": str(artifact),
                "certificate_sha256": certificate.certificate_sha256,
                "status": certificate.status,
                "study": study_id,
                "study_definition_sha256": (certificate.study.definition_sha256),
            }
        )
        all_certified = all_certified and certificate.status == "certified"
    print(json.dumps(summaries, indent=2, sort_keys=True))
    if all_certified or args.allow_noncertified_exit_zero:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
