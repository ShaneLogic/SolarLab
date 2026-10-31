"""The relocated experiment backend keeps the lower storage layer independent."""

import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from solarlab.experiments.transient import TransientProtocol
from solarlab.solvers.time.bdf_1d import BDFDrive


@pytest.mark.parametrize("module", [
    "solarlab.solvers.time.bdf_1d", "solarlab.solvers.time.assembled_ionic_bdf",
])
def test_lower_bdf_modules_import_with_experiments_forbidden(module):
    source = """import importlib, importlib.abc, sys
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('solarlab.experiments'):
            raise AssertionError('upward runtime import: ' + fullname)
sys.meta_path.insert(0, Forbidden())
importlib.import_module(sys.argv[1])
assert not any(name.startswith('solarlab.experiments') for name in sys.modules)
"""
    root = Path(__file__).resolve().parents[4]
    result = subprocess.run([sys.executable, "-c", source, module], cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1"),
        text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_backend_entries_are_only_in_the_experiment_layer():
    backend = importlib.import_module("solarlab.experiments.bdf")
    classical = importlib.import_module("solarlab.solvers.time.bdf_1d")
    assembled = importlib.import_module("solarlab.solvers.time.assembled_ionic_bdf")
    for name, old in (("solve_electronic_bdf", classical), ("solve_ionic_bdf", classical),
                      ("solve_assembled_ionic_bdf", assembled), ("solve_assembled_multiple_ionic_bdf", assembled)):
        assert name in backend.__all__ and callable(getattr(backend, name))
        assert getattr(backend, name).__module__ == "solarlab.experiments.bdf"
        assert name not in vars(old)


def test_drive_accepts_the_unchanged_public_protocol_and_structural_arrays_bitwise():
    protocol = TransientProtocol(np.array([0.0, 0.25, 1.0]), np.array([0.0, 0.1, 0.2]), np.array([0.0, 0.5, 1.0]))
    structural = SimpleNamespace(times_s=protocol.times_s, voltage_V=protocol.voltage_V,
                                 generation_scale=protocol.generation_scale)
    old = BDFDrive.from_protocol(protocol)
    new = BDFDrive.from_protocol(structural)
    for name in ("times_s", "left_values", "right_values"):
        assert getattr(old, name).tobytes() == getattr(new, name).tobytes()
    assert type(protocol).__module__ == "solarlab.experiments.transient"
