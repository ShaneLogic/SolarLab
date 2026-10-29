"""Local support-package import and Main's frozen digests for the assembly migration tests."""

from __future__ import annotations

from collections.abc import Callable
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

# The parity harness lives in tests/support; import it as ``support.*`` without
# depending on an installed package named ``tests`` (as tests/unit/assembly_parity).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ORACLE = Path(__file__).resolve().parents[1] / "assembly_parity" / "test_frozen_baselines.py"


@pytest.fixture(scope="session")
def frozen_digests() -> tuple[Callable[[Any], dict[str, Any]], dict[str, Any]]:
    """Main's digest function and the expanded frozen factories (tests/support/assembly_baselines.json).

    The frozen digests outlive the original classes: an assembled capture that
    matches them stays proven after the originals are deleted.
    """
    spec = importlib.util.spec_from_file_location("assembly_frozen_oracle", ORACLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    document = json.loads(module.BASELINE_PATH.read_bytes())
    assert document["schema"] == module.SCHEMA
    return module.extraction_digest, module.expand_digests(document["factories"], document["sha256"])
