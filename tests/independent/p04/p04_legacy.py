"""Read-only access to the legacy engine as the historical comparison.

The legacy package (``perovskite-sim/perovskite_sim``) is never edited by
this suite.  Every legacy symbol the oracles compare against is imported
here, and ``source_digests`` records the sha256 of the files that define
them so a golden or a historical comparison can name its exact source.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_ROOT = REPO_ROOT / "perovskite-sim"
LEGACY_PACKAGE = LEGACY_ROOT / "perovskite_sim"
LEGACY_CONFIGS = LEGACY_ROOT / "tests" / "fixtures" / "configs"

LEGACY_SOURCE_FILES = (
    "constants.py",
    "discretization/fe_operators.py",
    "discretization/grid.py",
    "physics/poisson.py",
    "physics/physical_control_volume.py",
    "physics/generation.py",
    "physics/recombination.py",
    "physics/traps.py",
    "physics/optics.py",
    "physics/optical_stack.py",
    "models/device.py",
    "solver/mol.py",
    "experiments/jv_sweep.py",
    "twod/continuity_2d.py",
    "twod/poisson_2d.py",
    "data/__init__.py",
    "data/am15g.csv",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digests(extra: tuple[Path, ...] = ()) -> dict[str, str]:
    out = {f"perovskite_sim/{rel}": sha256_file(LEGACY_PACKAGE / rel) for rel in LEGACY_SOURCE_FILES}
    for path in extra:
        out[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
    return out


def fe_operators():
    from perovskite_sim.discretization import fe_operators

    return fe_operators


def poisson():
    from perovskite_sim.physics import poisson

    return poisson


def generation():
    from perovskite_sim.physics import generation

    return generation


def recombination():
    from perovskite_sim.physics import recombination

    return recombination


def traps():
    from perovskite_sim.physics import traps

    return traps


def optics():
    from perovskite_sim.physics import optics

    return optics


def physical_control_volume():
    from perovskite_sim.physics import physical_control_volume

    return physical_control_volume


def grid():
    from perovskite_sim.discretization import grid

    return grid


def device_alignment():
    from perovskite_sim.models import device

    return device


def device_fixture(name: str):
    """(stack, legacy electrical grid helper) for a shipped test-fixture preset."""

    from perovskite_sim.models.config_loader import load_device_from_yaml

    return load_device_from_yaml(str(LEGACY_CONFIGS / name))


def electrical_grid(stack, N_grid: int):
    from perovskite_sim.discretization.grid import Layer, multilayer_grid
    from perovskite_sim.experiments.jv_sweep import _layer_node_counts
    from perovskite_sim.models.device import electrical_layers

    elec = electrical_layers(stack)
    return multilayer_grid(
        [Layer(layer.thickness, n) for layer, n in zip(elec, _layer_node_counts(stack, N_grid))]
    )
