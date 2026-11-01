"""Local support-package import and Main's frozen digests for the assembly migration tests."""

from __future__ import annotations

from collections.abc import Callable
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest

# The parity harness lives in tests/support; import it as ``support.*`` without
# depending on an installed package named ``tests`` (as tests/unit/assembly_parity).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ORACLE = Path(__file__).resolve().parents[1] / "assembly_parity" / "test_frozen_baselines.py"


@pytest.fixture(scope="session")
def frozen_oracle() -> ModuleType:
    """Load the existing capture-environment guard and digest implementation."""
    spec = importlib.util.spec_from_file_location("assembly_frozen_oracle", ORACLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def frozen_digests(frozen_oracle: ModuleType) -> tuple[Callable[[Any], dict[str, Any]], dict[str, Any]]:
    """Frozen outputs apply only to their recorded numerical environment.

    Live old-vs-new tests must not request this fixture. The shared guard is
    also used for decoded historical seed fields, whose floating-point words
    are tied to the same supported capture stack.
    """
    module = frozen_oracle
    document = json.loads(module.BASELINE_PATH.read_bytes())
    assert document["schema"] == module.SCHEMA
    module.require_capture_environment(document["capture"])
    return module.extraction_digest, module.expand_digests(document["factories"], document["sha256"])
