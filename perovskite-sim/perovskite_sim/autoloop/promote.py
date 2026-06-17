# perovskite_sim/autoloop/promote.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.types import ConfigEdit, Hypothesis

# Confirmed levers that are EXISTING device-config flags -> promotable. A lever
# without a key here needs codegen (out of Stage 3 scope) -> propose returns None.
FLAG_TO_CONFIG_KEY: dict[str, str] = {
    "SOLARLAB_IFACE_PROJ":  "interface_plane_projection",
    "SOLARLAB_IFACE_PLANE": "interface_plane_closure",
    "SOLARLAB_DOS_BAND":    "dos_band_potentials",
}


def parse_lever(mechanism: str) -> Optional[str]:
    """Extract the SOLARLAB_* flag token from a Hypothesis mechanism string."""
    m = re.search(r"(SOLARLAB_[A-Z_]+)", mechanism)
    return m.group(1) if m else None


def set_device_flag(text: str, key: str, value: bool) -> str:
    """Set-or-insert `  <key>: <value>` under the top-level `device:` block,
    preserving the rest of the file verbatim. Raises if no device block."""
    val = "true" if value else "false"
    lines = text.splitlines(keepends=True)
    dev_idx = next((i for i, ln in enumerate(lines) if ln.rstrip("\n") == "device:"), None)
    if dev_idx is None:
        raise ValueError("config has no top-level 'device:' block")
    # device block body = subsequent blank or indented lines
    body_end = dev_idx + 1
    while body_end < len(lines):
        ln = lines[body_end]
        if ln.strip() == "" or ln[:1] in (" ", "\t"):
            body_end += 1
        else:
            break
    for j in range(dev_idx + 1, body_end):
        stripped = lines[j].lstrip()
        if stripped.startswith(f"{key}:"):
            indent = lines[j][: len(lines[j]) - len(stripped)]
            lines[j] = f"{indent}{key}: {val}\n"
            return "".join(lines)
    lines.insert(dev_idx + 1, f"  {key}: {val}\n")
    return "".join(lines)


def apply_edit(edit: ConfigEdit) -> None:
    text = Path(edit.config_path).read_text(encoding="utf-8")
    Path(edit.config_path).write_text(
        set_device_flag(text, edit.device_key, edit.new_value), encoding="utf-8")


def revert_edit(edit: ConfigEdit) -> None:
    Path(edit.config_path).write_text(edit.old_text, encoding="utf-8")


def is_promotable(hypothesis: Hypothesis, ledger: Ledger) -> bool:
    """Whether a Hypothesis maps to an EXISTING promotable device-config flag.

    Promotability is a property of the hypothesis alone — confirmed, its mechanism
    carries a flag in ``FLAG_TO_CONFIG_KEY``, and it is not refuted. It deliberately
    does NOT touch the filesystem, so codegen routing can decide "promotable (Stage 3)
    vs not-promotable (codegen)" without a config file present (the codegen-routing
    decoupling that ``propose_promotion``'s old text read used to require)."""
    if hypothesis.verdict != "confirmed":
        return False
    flag = parse_lever(hypothesis.mechanism)
    if flag is None or FLAG_TO_CONFIG_KEY.get(flag) is None:
        return False
    return not ledger.is_refuted(hypothesis.mechanism)


def propose_promotion(hypothesis: Hypothesis, ledger: Ledger,
                      config_path) -> Optional[ConfigEdit]:
    """Confirmed lever -> ConfigEdit, or None if not confirmed / not promotable /
    refuted. Never proposes a refuted mechanism (anti-thrash). Reads the config so
    the edit can be reverted exactly; an absent config raises ``FileNotFoundError``
    (promotability without a config is answered by ``is_promotable``, not here)."""
    if not is_promotable(hypothesis, ledger):
        return None
    key = FLAG_TO_CONFIG_KEY[parse_lever(hypothesis.mechanism)]
    # old_text is needed to revert an applied edit — read it eagerly (an absent
    # config for a promotable mechanism is a caller error, not a silent empty text).
    old_text = Path(config_path).read_text(encoding="utf-8")
    return ConfigEdit(config_path=str(config_path), device_key=key,
                      new_value=True, old_text=old_text)
