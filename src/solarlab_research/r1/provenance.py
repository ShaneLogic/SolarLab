"""Release-source provenance validator and non-relabeling historical R1 readers (P07-01/P03-05).

* Binds new ``solarlab_research`` runs to immutable ``ReleaseBundle`` build-commit
  and file-digest identities without inferring from mutable working trees.
* Reads historical R1 artifacts (``R1CommonStateV1``, ``R1CommonStatePairV2``,
  ``SolarLabR1FailureFixtureV1``) strictly against their original recorded source
  commits and digests without relabeling them to current sources or clearing
  historical failure verdicts (e.g., V53 step 309).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from solarlab.config.identity import BuildSource
from solarlab.materials.source import SourceDocument
from solarlab.reproducibility.release import (
    ReleaseBundle,
    ReleaseFile,
    read_release_bundle,
    verify_release_installation,
)

from solarlab_research.precision.codecs import decode_historical_r1_pair_state
from solarlab_research.r1.contracts import R1_OPERATOR_CHANNELS, R1_V53_CERTIFICATE_LIMITS

__all__ = [
    "HistoricalR1ArtifactSummary",
    "HistoricalR1WitnessAudit",
    "R1ReleaseSourceBinding",
    "load_verified_release_bundle",
    "read_historical_r1_artifact",
    "synthetic_research_release_bundle",
    "verify_historical_r1_failure_witness",
    "verify_r1_release_source",
]


@dataclass(frozen=True, slots=True)
class R1ReleaseSourceBinding:
    """Verified binding of a research run to an immutable ReleaseBundle."""

    build_source_commit: str
    profile: str
    bundle_sha256: str
    build_source: BuildSource
    job_provenance_envelope: Mapping[str, Any]
    scientific_qualification: bool


def verify_r1_release_source(
    bundle: ReleaseBundle,
    *,
    expected_bundle_sha256: str,
    inputs: dict[str, Any],
    operator: str = "solarlab_research",
    installation_root: Path | None = None,
    module_origins: tuple[tuple[str, str], ...] | None = None,
    required_modules: tuple[str, ...] = ("solarlab", "solarlab_research"),
) -> R1ReleaseSourceBinding:
    """Verify a ReleaseBundle digest and issue a source-attested provenance envelope."""
    if not isinstance(bundle, ReleaseBundle):
        raise TypeError("bundle must be a ReleaseBundle")
    if bundle.sha256 != expected_bundle_sha256:
        raise ValueError(
            f"ReleaseBundle sha256 {bundle.sha256} differs from expected {expected_bundle_sha256}"
        )
    exported = bundle.export()
    if exported.get("scientific_qualification") is not False:
        raise ValueError("ReleaseBundle provenance must keep scientific_qualification=False")

    if installation_root is not None:
        if module_origins is None:
            raise ValueError("module_origins is required when verifying an installation_root")
        verify_release_installation(
            bundle,
            installation_root,
            expected_sha256=expected_bundle_sha256,
            module_origins=module_origins,
            required_modules=required_modules,
        )
    elif bundle.can_execute:
        raise ValueError("ReleaseBundle cannot self-authorize execution")

    provenance = bundle.job_provenance(inputs, operator=operator)
    return R1ReleaseSourceBinding(
        build_source_commit=bundle.build_source_commit,
        profile=bundle.profile,
        bundle_sha256=bundle.sha256,
        build_source=bundle.implementation(),
        job_provenance_envelope=MappingProxyType(provenance.envelope()),
        scientific_qualification=False,
    )


def load_verified_release_bundle(
    bundle_dir: Path,
    *,
    expected_bundle_sha256: str,
) -> ReleaseBundle:
    """Load and cryptographically verify a materialized ReleaseBundle directory."""
    manifest_path = bundle_dir / "ReleaseBundle.json"
    manifest = SourceDocument("ReleaseBundle.json", manifest_path.read_bytes())
    return read_release_bundle(manifest, bundle_dir, expected_sha256=expected_bundle_sha256)


def synthetic_research_release_bundle(
    *,
    build_source_commit: str = "19c19776ff0b13beb791683697035594718ff317",
) -> ReleaseBundle:
    """Construct a minimal self-contained target-profile ReleaseBundle for standalone/CLI checks."""
    files = (
        ReleaseFile("src/solarlab/__init__.py", "site/solarlab/__init__.py", "package", b"\"\"\"solarlab.\"\"\"\n"),
        ReleaseFile("src/solarlab/py.typed", "site/solarlab/py.typed", "package", b""),
        ReleaseFile(
            "src/solarlab_research/__init__.py",
            "site/solarlab_research/__init__.py",
            "package",
            b"\"\"\"solarlab_research.\"\"\"\n",
        ),
        ReleaseFile("src/solarlab_research/py.typed", "site/solarlab_research/py.typed", "package", b""),
        ReleaseFile(
            "src/solarlab_server/__init__.py",
            "site/solarlab_server/__init__.py",
            "package",
            b"\"\"\"solarlab_server.\"\"\"\n",
        ),
        ReleaseFile("src/solarlab_server/py.typed", "site/solarlab_server/py.typed", "package", b""),
        ReleaseFile(
            "src/solarlab/config/identity.py",
            "site/solarlab/config/identity.py",
            "validator",
            b"\"\"\"validator.\"\"\"\n",
        ),
        ReleaseFile(
            "src/solarlab/reproducibility/PublicAPI.json",
            "site/solarlab/reproducibility/PublicAPI.json",
            "contract",
            b"{}\n",
        ),
        ReleaseFile("resources/am15g.csv", "share/resources/am15g.csv", "resource", b"wl,flux\n500,1.0\n"),
        ReleaseFile("uv.lock", "share/uv.lock", "dependency_lock", b"version = 1\n"),
        ReleaseFile("pyproject.toml", "share/pyproject.toml", "build_configuration", b"[project]\nname = 'solarlab'\n"),
        ReleaseFile(
            "generated/configuration-schema.json",
            "share/generated/configuration-schema.json",
            "generated_schema",
            b"{\"schema\":\"v1\"}\n",
        ),
    )
    return ReleaseBundle(build_source_commit=build_source_commit, profile="target", files=files)


@dataclass(frozen=True, slots=True)
class HistoricalR1ArtifactSummary:
    """Read-only summary of a historical R1 artifact verified against its original source."""

    schema: str
    artifact_sha256: str
    original_source_id: str | None
    original_source_sha256: str | None
    relabelled: bool
    metadata: Mapping[str, Any]


def read_historical_r1_artifact(source: Path | bytes | str) -> HistoricalR1ArtifactSummary:
    """Read a historical R1 state/witness artifact without relabeling its source identity."""
    if isinstance(source, Path):
        raw_bytes = source.read_bytes()
    elif isinstance(source, bytes):
        raw_bytes = source
    elif isinstance(source, str):
        raw_bytes = source.encode("utf-8")
    else:
        raise TypeError("source must be Path, bytes, or str")

    artifact_sha = hashlib.sha256(raw_bytes).hexdigest()
    payload = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("historical R1 artifact root must be a JSON object")

    schema = str(payload.get("schema", ""))
    if schema == "R1CommonStatePairV2":
        decoded = decode_historical_r1_pair_state(payload)
        seed = payload["seed_preparation"]
        src_info = payload.get("source", {})
        return HistoricalR1ArtifactSummary(
            schema=schema,
            artifact_sha256=artifact_sha,
            original_source_id=src_info.get("source_commit") or str(src_info.get("run_class", "legacy")),
            original_source_sha256=str(src_info.get("sha256", payload["sha256"])),
            relabelled=False,
            metadata=MappingProxyType({
                "representation": payload["representation"],
                "record_sha256": payload["sha256"],
                "seed_preparation_sha256": payload["seed_preparation_sha256"],
                "dc_state_sha256": seed["dc_state"]["state_sha256"],
                "decoded_field_count": len(decoded),
            }),
        )

    if schema == "R1CommonStateV1":
        unsigned = {k: v for k, v in payload.items() if k != "sha256"}
        canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        computed_sha = hashlib.sha256(canonical).hexdigest()
        recorded_sha = str(payload["sha256"])
        if computed_sha != recorded_sha:
            raise ValueError(
                f"R1CommonStateV1 sha256 mismatch: expected {recorded_sha}, got {computed_sha}"
            )
        src_info = payload.get("source", {})
        dc_state = payload.get("dc_state", {})
        arr: NDArray[np.float64] = np.asarray(dc_state.get("potential_V", ()), dtype=float)
        return HistoricalR1ArtifactSummary(
            schema=schema,
            artifact_sha256=artifact_sha,
            original_source_id=src_info.get("source_commit") or str(src_info.get("run_class", "legacy")),
            original_source_sha256=str(src_info.get("sha256", recorded_sha)),
            relabelled=False,
            metadata=MappingProxyType({
                "record_sha256": recorded_sha,
                "dc_state_sha256": dc_state.get("state_sha256"),
                "intervals": int(payload["intervals"]),
                "temperature_K": float(payload["history"]["temperature_K"]),
                "node_count": int(arr.size),
            }),
        )

    if schema == "SolarLabR1FailureFixtureV1":
        audit = verify_historical_r1_failure_witness(raw_bytes)
        return HistoricalR1ArtifactSummary(
            schema=schema,
            artifact_sha256=artifact_sha,
            original_source_id=audit.source_commit,
            original_source_sha256=audit.source_content_sha256,
            relabelled=False,
            metadata=MappingProxyType({
                "source_commit": audit.source_commit,
                "case_names": audit.case_names,
                "step309_status": audit.step309_original_status,
                "step309_nonlinear_residual": audit.step309_nonlinear_residual,
            }),
        )

    raise ValueError(f"unsupported historical R1 artifact schema: {schema!r}")


@dataclass(frozen=True, slots=True)
class HistoricalR1WitnessAudit:
    """Verified historical V53 witness record preserving original failure status."""

    fixture_sha256: str
    source_commit: str
    source_content_sha256: str
    source_fixture_yaml: str
    case_names: tuple[str, ...]
    step183_accepted_status: str
    step183_failed_status: str
    step309_original_status: str
    step309_failure_type: str
    step309_nonlinear_residual: float
    step309_dt_s: float
    step309_voltage_V: float
    relabelled: bool


def verify_historical_r1_failure_witness(
    source: Path | bytes | str,
) -> HistoricalR1WitnessAudit:
    """Verify ``R1FailureWitnessV1.json`` against its original V53 commit and failure status."""
    raw = source.read_bytes() if isinstance(source, Path) else (
        source if isinstance(source, bytes) else source.encode("utf-8")
    )
    fixture_sha = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("schema") != "SolarLabR1FailureFixtureV1":
        raise ValueError(f"expected SolarLabR1FailureFixtureV1, got {payload.get('schema')!r}")

    commit = str(payload["source_commit"])
    content_sha = str(payload["source_content_sha256"])
    if len(commit) != 40 or len(content_sha) != 64:
        raise ValueError("witness lacks valid original source_commit / source_content_sha256")

    req_info = payload.get("request", {})
    limits = req_info.get("certificate_limits", payload.get("certificate_limits", {}))
    if limits != dict(R1_V53_CERTIFICATE_LIMITS):
        raise ValueError("witness certificate_limits differ from frozen V53 limits")
    channels = tuple(payload.get("operator_channels", ()))
    if channels != R1_OPERATOR_CHANNELS:
        raise ValueError("witness operator_channels differ from frozen 13-channel contract")

    cases = payload.get("cases", {})
    required_cases = ("step183_accepted", "step183_failed", "step309_failed")
    if not all(name in cases for name in required_cases):
        raise ValueError(f"witness missing required cases {required_cases}")

    c183_ok = cases["step183_accepted"]
    c183_fail = cases["step183_failed"]
    c309 = cases["step309_failed"]
    orig_fail = c309.get("original_failure")
    if not isinstance(orig_fail, dict):
        raise ValueError("step309_failed must retain its original_failure record")

    fail_type = str(orig_fail.get("type", ""))
    msg = str(orig_fail.get("message", ""))
    if fail_type != "InterfaceDefectTransientError":
        raise ValueError(f"unexpected step309 failure type: {fail_type!r}")
    # Extract historical nonlinear residual (0.0532305 > 0.05) and preserve failed status.
    residual_val: float | None = None
    if "nonlinear_residual" in c309:
        residual_val = float(c309["nonlinear_residual"])
    elif "residual" in msg:
        tokens = msg.replace(",", " ").replace(";", " ").split()
        for idx, tok in enumerate(tokens[:-1]):
            if tok == "residual":
                try:
                    residual_val = float(tokens[idx + 1])
                    break
                except ValueError:
                    continue
    if residual_val is None or residual_val <= R1_V53_CERTIFICATE_LIMITS["nonlinear_residual"]:
        raise ValueError("historical step309 witness residual cannot be relabeled as passing")

    saved_309 = c309.get("saved_witness", c309)
    return HistoricalR1WitnessAudit(
        fixture_sha256=fixture_sha,
        source_commit=commit,
        source_content_sha256=content_sha,
        source_fixture_yaml=str(payload["source_fixture_yaml"]),
        case_names=tuple(sorted(cases)),
        step183_accepted_status="accepted" if c183_ok.get("original_failure") is None else "failed",
        step183_failed_status="failed" if c183_fail.get("original_failure") is not None else "accepted",
        step309_original_status="failed",
        step309_failure_type=fail_type,
        step309_nonlinear_residual=residual_val,
        step309_dt_s=float(saved_309["dt_s"]),
        step309_voltage_V=float(saved_309["voltage_V"]),
        relabelled=False,
    )
