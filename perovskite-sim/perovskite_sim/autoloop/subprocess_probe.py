# perovskite_sim/autoloop/subprocess_probe.py
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from perovskite_sim.autoloop.ladder import _PKG_ROOT
from perovskite_sim.autoloop.types import Gap


@dataclass
class SubprocessProbeRunner:
    """Real ProbeRunner: runs one variant in a fresh interpreter with env flags
    set, so module-level SOLARLAB_* reads + the MaterialArrays cache pick them
    up. Returns the badness scalar printed by _probe_worker."""
    config_path: Path
    reference_path: Path
    gap: Gap

    def run(self, variant: dict) -> float:
        env = dict(os.environ)
        env.update(variant.get("env_flags", {}))
        payload = {
            "config": str(self.config_path),
            "reference": str(self.reference_path),
            "gap_sweep": self.gap.sweep,
            "gap_metric": self.gap.metric,
            "gap_kind": self.gap.kind,
            "jv_overrides": variant.get("jv_overrides", {}),
            "measure": variant.get("measure", "gap"),
        }
        proc = subprocess.run(
            ["python", "-m", "perovskite_sim.autoloop._probe_worker", json.dumps(payload)],
            capture_output=True, text=True, env=env, cwd=str(_PKG_ROOT),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"probe worker failed (rc={proc.returncode}): "
                               f"{proc.stderr.strip()[-400:]}")
        return float(json.loads(proc.stdout)["metric"])
