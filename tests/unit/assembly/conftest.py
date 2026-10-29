"""Local support-package import for the assembly migration tests."""

from __future__ import annotations

from pathlib import Path
import sys

# The parity harness lives in tests/support; import it as ``support.*`` without
# depending on an installed package named ``tests`` (as tests/unit/assembly_parity).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
