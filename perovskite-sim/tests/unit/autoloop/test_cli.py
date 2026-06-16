import importlib.util
import sys
from pathlib import Path

CLI = Path(__file__).resolve().parents[3] / "scripts" / "autoloop_run.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("autoloop_run", CLI)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["autoloop_run"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_cli_parse_args_defaults():
    mod = _load_cli()
    ns = mod.parse_args(["--once"])
    assert ns.once is True
    assert ns.cycle == 0


def test_cli_build_timestamp_is_iso():
    mod = _load_cli()
    ts = mod.iso_timestamp_utc()
    assert ts.endswith("Z") and "T" in ts
