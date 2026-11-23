"""Hatchling build hook mapping compiled ``web/dist`` SPA assets into the release."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

__all__ = ["WebDistBuildHook"]


class WebDistBuildHook(BuildHookInterface):
    """Bundle ``web/dist/**`` into ``solarlab_server/_web_dist/**`` when present."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if version == "editable":
            return
        web_dist = Path(self.root) / "web" / "dist"
        if not web_dist.is_dir():
            return
        force_include = build_data.setdefault("force_include", {})
        for path in sorted(web_dist.rglob("*")):
            if path.is_file() and path.name != ".DS_Store":
                rel = path.relative_to(web_dist).as_posix()
                target = (
                    f"web/dist/{rel}"
                    if self.target_name == "sdist"
                    else f"solarlab_server/_web_dist/{rel}"
                )
                force_include[str(path)] = target
