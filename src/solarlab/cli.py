"""Standalone SolarLab command-line entrypoints (``solarlab`` and ``solarlab-server``).

Provides installation/resource verification, 1D steady-state J-V execution on
packaged or user V2 configurations, delegation to ``solarlab_research`` tools,
and a standalone ``solarlab-server`` launcher that delegates to
``solarlab_server.cli:main`` when available or boots ``solarlab_server.app``
with packaged material resources.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from solarlab.device.resolved import PreparedDevice

__all__ = ["main", "server_main"]


def _package_version() -> str:
    try:
        return importlib.metadata.version("solarlab")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def _load_prepared_device(
    *, preset: str, config_path: Path | None
) -> PreparedDevice:
    from solarlab.config.resolve_device import resolve_device
    from solarlab.device.defaults import DefaultCatalog
    from solarlab.device.inputs import DeviceInput
    from solarlab.materials import (
        load_packaged_default_catalog,
        load_packaged_preset_device_mapping,
        load_packaged_preset_sources,
        load_packaged_resource_library,
    )
    from solarlab.materials.source import SourceDocument

    defaults: DefaultCatalog = load_packaged_default_catalog()
    resources = load_packaged_resource_library()
    if config_path is not None:
        raw_bytes = config_path.read_bytes()
        source = SourceDocument(config_path.name, raw_bytes)
        payload = json.loads(raw_bytes.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("configuration JSON must contain an object")
        if "device" in payload and "default_catalog" in payload:
            cat_map = {
                **payload["default_catalog"],
                "evidence": [[source.id, source.sha256]],
            }
            return resolve_device(
                DeviceInput.model_validate(payload["device"]),
                DefaultCatalog.from_mapping(cat_map),
                resources,
                sources=(source,),
            )
        device_dict = payload.get("input", payload)
        return resolve_device(
            DeviceInput.model_validate(device_dict),
            defaults,
            resources,
            sources=(source, *load_packaged_preset_sources()),
        )
    mapping = load_packaged_preset_device_mapping(preset)
    return resolve_device(
        DeviceInput.model_validate(mapping),
        defaults,
        resources,
        sources=load_packaged_preset_sources(),
    )


def _cmd_check_install(args: argparse.Namespace) -> int:
    from solarlab.config.schema import configuration_schema_representation
    from solarlab.materials import (
        load_packaged_default_catalog,
        load_packaged_preset_sources,
        load_packaged_resource_library,
        load_packaged_template_library,
    )

    defaults = load_packaged_default_catalog()
    resources = load_packaged_resource_library()
    templates = load_packaged_template_library()
    presets = load_packaged_preset_sources()
    _, schema_sha256 = configuration_schema_representation()
    legacy_present = importlib.util.find_spec("perovskite_sim") is not None
    backend_present = importlib.util.find_spec("backend") is not None
    report = {
        "schema": "solarlab.install-check.v1",
        "version": _package_version(),
        "default_catalog_sha256": defaults.content_sha256,
        "resource_library_sha256": resources.content_sha256,
        "resource_count": len(resources.tables),
        "template_library_sha256": templates.content_sha256,
        "template_count": len(templates.templates),
        "preset_source_count": len(presets),
        "configuration_schema_sha256": schema_sha256,
        "legacy_package_present": legacy_present,
        "legacy_backend_present": backend_present,
        "scientific_qualification": False,
    }
    if args.require_no_legacy and (legacy_present or backend_present):
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


_STEADY_JV_PRESETS: tuple[str, ...] = ("classical_pin",)


def _cmd_steady_jv(args: argparse.Namespace) -> int:
    import numpy as np

    from solarlab.device.compile_1d import CapabilityError, compile_device_1d
    from solarlab.experiments.jv.steady import run_steady_jv
    from solarlab.solvers.nonlinear.production import POISSON_CONTACT_BALANCE_V1
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, SteadyStateError

    try:
        if args.n_points < 2:
            raise ValueError("--n-points must be at least 2")
        prepared = _load_prepared_device(preset=args.preset, config_path=args.config)
        compiled = compile_device_1d(
            prepared, intervals_per_layer=args.intervals_per_layer
        )
        voltages = np.linspace(float(args.v_min), float(args.v_max), int(args.n_points))
        options = SteadyOptions(
            continuity_tolerance_A_m2=1.0e-6,
            current_spread_tolerance_A_m2=1.0e-6,
            max_bias_step_V=0.05,
            convergence_policy=POISSON_CONTACT_BALANCE_V1,
        )
        result = run_steady_jv(
            compiled,
            voltages,
            generation_scale=float(args.generation_scale),
            options=options,
        )
    except (CapabilityError, SteadyStateError, ValueError) as exc:
        err_payload = {
            "schema": "solarlab.cli.error.v1",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        print(json.dumps(err_payload, sort_keys=True), file=sys.stderr)
        return 1
    payload: dict[str, Any] = {
        "schema": "solarlab.cli.steady-jv.v1",
        "device_id": compiled.device_id,
        "compiled_identity": compiled.identity,
        "completed": True,
        "n_points": int(result.voltage_V.size),
        "voltages_V": [float(v) for v in result.voltage_V],
        "output_current_A_m2": [float(j) for j in result.output_current_A_m2],
        "metrics": asdict(result.metrics),
        "scientific_qualification": False,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


def _write_or_print_json(payload: dict[str, Any], output: Path | None) -> int:
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


def _cli_error_exit(exc: Exception) -> int:
    err_payload = {
        "schema": "solarlab.cli.error.v1",
        "error_type": type(exc).__name__,
        "message": str(exc),
    }
    print(json.dumps(err_payload, sort_keys=True), file=sys.stderr)
    return 1


def _cmd_iodine_presets(args: argparse.Namespace) -> int:
    from solarlab.experiments.iodine_declarative import export_iodine_deck_schema
    from solarlab.materials.iodine_presets import list_packaged_iodine_preset_summaries

    if args.schema:
        return _write_or_print_json(export_iodine_deck_schema(), args.output)
    summaries = [
        {k: v for k, v in item.items() if k != "deck"}
        for item in list_packaged_iodine_preset_summaries()
    ]
    return _write_or_print_json(
        {
            "schema": "solarlab.cli.iodine-presets.v1",
            "presets": summaries,
            "scientific_qualification": False,
        },
        args.output,
    )


def _cmd_iodine_steady(args: argparse.Namespace) -> int:
    from pydantic import ValidationError

    from solarlab.contracts import ContractError
    from solarlab.equations.dc_1d import CapabilityError
    from solarlab.experiments.iodine_declarative import (
        IodineOverridesInput,
        apply_iodine_deck_overrides,
        execute_iodine_steady_deck,
        load_iodine_deck,
    )
    from solarlab.solvers.nonlinear.steady_1d import SteadyStateError

    try:
        target: str | Path = args.config if args.config is not None else args.preset
        deck, src = load_iodine_deck(target)
        overrides = IodineOverridesInput(
            intervals_per_layer=args.intervals_per_layer,
            k_a_cm3_s=args.k_a,
            mu_bl_cm2_V_s=args.mu_bl,
            ebl_partition=args.ebl_partition,
            voltage_V=args.voltage,
            generation_scale=args.generation_scale,
            compute_frozen_jv=(False if args.no_frozen_jv else None),
        )
        effective = apply_iodine_deck_overrides(deck, overrides)
        payload = execute_iodine_steady_deck(effective, source=src)
    except (
        CapabilityError,
        ContractError,
        FileNotFoundError,
        KeyError,
        SteadyStateError,
        ValidationError,
        ValueError,
    ) as exc:
        return _cli_error_exit(exc)
    return _write_or_print_json(payload, args.output)


def _cmd_iodine_impedance(args: argparse.Namespace) -> int:
    from pydantic import ValidationError

    from solarlab.contracts import ContractError
    from solarlab.equations.dc_1d import CapabilityError
    from solarlab.experiments.iodine_declarative import (
        IodineOverridesInput,
        apply_iodine_deck_overrides,
        execute_iodine_impedance_deck,
        load_iodine_deck,
    )
    from solarlab.solvers.nonlinear.steady_1d import SteadyStateError

    try:
        target: str | Path = args.config if args.config is not None else args.preset
        deck, src = load_iodine_deck(target)
        overrides = IodineOverridesInput(
            intervals_per_layer=args.intervals_per_layer,
            k_a_cm3_s=args.k_a,
            voltage_V=args.voltage,
            generation_scale=args.generation_scale,
            f_min_Hz=args.f_min,
            f_max_Hz=args.f_max,
            n_frequencies=args.n_frequencies,
            chem_mode=args.chem_mode,
        )
        effective = apply_iodine_deck_overrides(deck, overrides)
        payload = execute_iodine_impedance_deck(effective, source=src)
    except (
        CapabilityError,
        ContractError,
        FileNotFoundError,
        KeyError,
        SteadyStateError,
        ValidationError,
        ValueError,
    ) as exc:
        return _cli_error_exit(exc)
    return _write_or_print_json(payload, args.output)


def _cmd_iodine_aging(args: argparse.Namespace) -> int:
    from pydantic import ValidationError

    from solarlab.contracts import ContractError
    from solarlab.equations.dc_1d import CapabilityError
    from solarlab.experiments.iodine_aging import IodineAgingError
    from solarlab.experiments.iodine_declarative import (
        IodineOverridesInput,
        apply_iodine_deck_overrides,
        execute_iodine_aging_deck,
        load_iodine_deck,
    )
    from solarlab.solvers.nonlinear.steady_1d import SteadyStateError

    try:
        target: str | Path = args.config if args.config is not None else args.preset
        deck, src = load_iodine_deck(target)
        overrides = IodineOverridesInput(
            intervals_per_layer=args.intervals_per_layer,
            k_a_cm3_s=args.k_a,
            volumetric_escape_s_1=args.volumetric_escape,
            contact_corrosion_m_s=args.contact_corrosion,
            probe_mode=args.probe_mode,
        )
        effective = apply_iodine_deck_overrides(deck, overrides)
        payload = execute_iodine_aging_deck(
            effective,
            source=src,
            store_dir=args.store_dir,
            bundle_dir=args.bundle_dir,
            run_id=args.run_id,
        )
    except (
        CapabilityError,
        ContractError,
        FileNotFoundError,
        IodineAgingError,
        KeyError,
        SteadyStateError,
        ValidationError,
        ValueError,
    ) as exc:
        return _cli_error_exit(exc)
    return _write_or_print_json(payload, args.output)


def _delegate_research(module_name: str, extra_args: list[str]) -> int:
    module = importlib.import_module(module_name)
    entry = getattr(module, "main")
    rc = entry(extra_args)
    return int(rc) if isinstance(rc, int) else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for the ``solarlab`` console script."""
    parser = argparse.ArgumentParser(
        prog="solarlab",
        description="SolarLab semiconductor device simulator CLI.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"solarlab {_package_version()}",
    )
    subparsers = parser.add_subparsers(dest="command")

    p_check = subparsers.add_parser(
        "check-install",
        help="Verify packaged default catalog, optical tables, templates, and schema.",
    )
    p_check.add_argument(
        "--require-no-legacy",
        action="store_true",
        help="Exit non-zero if perovskite_sim or backend is importable.",
    )

    p_jv = subparsers.add_parser(
        "steady-jv",
        help="Run a 1D ion-free steady-state J-V sweep on classical_pin or a V2 JSON file.",
    )
    p_jv.add_argument(
        "--preset",
        choices=_STEADY_JV_PRESETS,
        default="classical_pin",
        help="Packaged 1D ion-free steady device preset (default: classical_pin).",
    )
    p_jv.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a V2 DeviceInput JSON document.",
    )
    p_jv.add_argument("--v-min", type=float, default=0.0)
    p_jv.add_argument("--v-max", type=float, default=1.25)
    p_jv.add_argument("--n-points", type=int, default=26)
    p_jv.add_argument("--generation-scale", type=float, default=1.0)
    p_jv.add_argument("--intervals-per-layer", type=int, default=12)
    p_jv.add_argument("--output", type=Path, default=None)

    p_cbo = subparsers.add_parser(
        "cbo-scan",
        help="Delegate to solarlab_research.scripts.cbo_scan.",
    )
    p_cbo.add_argument("research_args", nargs=argparse.REMAINDER)

    from solarlab.materials.iodine_presets import PACKAGED_IODINE_PRESET_NAMES

    p_ipresets = subparsers.add_parser(
        "iodine-presets",
        help="List packaged 1D iodine chemistry/aging presets or export the deck JSON Schema.",
    )
    p_ipresets.add_argument(
        "--schema",
        action="store_true",
        help="Export the solarlab.iodine-deck.v1 JSON Schema instead of listing presets.",
    )
    p_ipresets.add_argument("--output", type=Path, default=None)

    p_isteady = subparsers.add_parser(
        "iodine-steady",
        help="Run coupled 1D iodine reaction-diffusion steady state and optional frozen J-V.",
    )
    p_isteady.add_argument(
        "--preset",
        choices=PACKAGED_IODINE_PRESET_NAMES,
        default="bitton_tessler_2023_blocking",
    )
    p_isteady.add_argument("--config", type=Path, default=None)
    p_isteady.add_argument("--voltage", type=float, default=None)
    p_isteady.add_argument("--generation-scale", type=float, default=None)
    p_isteady.add_argument("--k-a", type=float, default=None, help="Override absorber k_a [cm^3/s].")
    p_isteady.add_argument("--mu-bl", type=float, default=None, help="Override BL mobility [cm^2/(V s)].")
    p_isteady.add_argument("--ebl-partition", type=float, default=None)
    p_isteady.add_argument("--intervals-per-layer", type=int, default=None)
    p_isteady.add_argument("--no-frozen-jv", action="store_true")
    p_isteady.add_argument("--output", type=Path, default=None)

    p_iimp = subparsers.add_parser(
        "iodine-impedance",
        help="Run 1D small-signal AC impedance and redox capacitance sweep on an iodine deck.",
    )
    p_iimp.add_argument(
        "--preset",
        choices=PACKAGED_IODINE_PRESET_NAMES,
        default="bitton_tessler_2023_blocking",
    )
    p_iimp.add_argument("--config", type=Path, default=None)
    p_iimp.add_argument("--voltage", type=float, default=None)
    p_iimp.add_argument("--generation-scale", type=float, default=None)
    p_iimp.add_argument("--f-min", type=float, default=None)
    p_iimp.add_argument("--f-max", type=float, default=None)
    p_iimp.add_argument("--n-frequencies", type=int, default=None)
    p_iimp.add_argument(
        "--chem-mode",
        choices=("mobile", "frozen", "both"),
        default=None,
    )
    p_iimp.add_argument("--k-a", type=float, default=None)
    p_iimp.add_argument("--intervals-per-layer", type=int, default=None)
    p_iimp.add_argument("--output", type=Path, default=None)

    p_iaging = subparsers.add_parser(
        "iodine-aging",
        help="Run a multi-stage 1D iodine stress/recovery aging protocol.",
    )
    p_iaging.add_argument(
        "--preset",
        choices=PACKAGED_IODINE_PRESET_NAMES,
        default="bitton_tessler_2023_self_healing",
    )
    p_iaging.add_argument("--config", type=Path, default=None)
    p_iaging.add_argument("--k-a", type=float, default=None)
    p_iaging.add_argument("--volumetric-escape", type=float, default=None)
    p_iaging.add_argument("--contact-corrosion", type=float, default=None)
    p_iaging.add_argument(
        "--probe-mode",
        choices=("none", "frozen_chemistry_jv", "dynamic_jv_scan", "both"),
        default=None,
    )
    p_iaging.add_argument("--intervals-per-layer", type=int, default=None)
    p_iaging.add_argument("--store-dir", type=Path, default=None)
    p_iaging.add_argument("--bundle-dir", type=Path, default=None)
    p_iaging.add_argument("--run-id", type=str, default=None)
    p_iaging.add_argument("--output", type=Path, default=None)

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "check-install":
        return _cmd_check_install(args)
    if args.command == "steady-jv":
        return _cmd_steady_jv(args)
    if args.command == "iodine-presets":
        return _cmd_iodine_presets(args)
    if args.command == "iodine-steady":
        return _cmd_iodine_steady(args)
    if args.command == "iodine-impedance":
        return _cmd_iodine_impedance(args)
    if args.command == "iodine-aging":
        return _cmd_iodine_aging(args)
    if args.command == "cbo-scan":
        forwarded = list(args.research_args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        return _delegate_research("solarlab_research.scripts.cbo_scan", forwarded)
    parser.error(f"unknown subcommand: {args.command}")
    return 2


def server_main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for the ``solarlab-server`` console script.

    Delegates to ``solarlab_server.cli:main`` when available; otherwise boots
    ``solarlab_server.app:create_app`` with packaged ``DefaultCatalog`` and
    ``ResourceLibrary`` over an explicit ``--store-dir``.
    """
    server_cli_name = "solarlab_server.cli"
    if importlib.util.find_spec(server_cli_name) is not None:
        server_cli = importlib.import_module(server_cli_name)
        delegate = getattr(server_cli, "main", None)
        if callable(delegate):
            sig = inspect.signature(delegate)
            if len(sig.parameters) >= 1:
                rc = delegate(argv)
            elif argv is None:
                rc = delegate()
            else:
                old_argv = sys.argv[:]
                try:
                    sys.argv = ["solarlab-server", *argv]
                    rc = delegate()
                finally:
                    sys.argv = old_argv
            return int(rc) if isinstance(rc, int) else 0

    parser = argparse.ArgumentParser(
        prog="solarlab-server",
        description="Launch the local SolarLab HTTP server over an explicit RunStore.",
    )
    parser.add_argument(
        "--store-dir",
        type=Path,
        required=True,
        help="Path to the local RunStore directory (initialized if absent).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Construct and verify the FastAPI application without starting uvicorn.",
    )
    from solarlab.io.run_store import RunStore
    from solarlab.materials import (
        load_packaged_default_catalog,
        load_packaged_preset_sources,
        load_packaged_resource_library,
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    store_root = args.store_dir.expanduser().resolve()
    if not (store_root / "store.sqlite3").is_file():
        with RunStore(store_root):
            pass

    app_mod = importlib.import_module("solarlab_server.app")
    cfg_mod = importlib.import_module("solarlab_server.configuration")
    preview = cfg_mod.ConfigurationPreviewContext(
        defaults=load_packaged_default_catalog(),
        resources=load_packaged_resource_library(),
        sources=load_packaged_preset_sources(),
    )
    app = app_mod.create_app(store_root, configuration_preview=preview)
    if args.check_only:
        print(
            json.dumps(
                {
                    "schema": "solarlab.server-check.v1",
                    "store_root": str(store_root),
                    "routes": len(app.routes),
                },
                sort_keys=True,
            )
        )
        return 0
    uvicorn = importlib.import_module("uvicorn")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
