# perovskite_sim/autoloop/cognition.py
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


class CognitionRuntime(Protocol):
    def complete(self, prompt: str, schema: dict) -> dict: ...


def _validate(obj: dict, schema: dict) -> None:
    """In-process schema check (no jsonschema dep): required keys + cause enum."""
    if not isinstance(obj, dict):
        raise ValueError("LLM output is not a JSON object")
    for key in schema.get("required", ()):
        if key not in obj:
            raise ValueError(f"LLM output missing required key {key!r}")
    enum = schema.get("cause_enum")
    if enum is not None and obj.get("cause") not in enum:
        raise ValueError(f"cause {obj.get('cause')!r} not in {enum}")


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


@dataclass
class FakeRuntime:
    """Test runtime: returns a canned dict (or callable(prompt) -> dict)."""
    canned: object

    def complete(self, prompt: str, schema: dict) -> dict:
        out = self.canned(prompt) if callable(self.canned) else self.canned
        return dict(out)


@dataclass
class ClaudeCliRuntime:
    """Headless cognition via `claude -p --output-format json`. Schema-validated,
    timeout-guarded, retries once on parse/validation failure."""
    model: str = "sonnet"
    timeout_s: float = 180.0

    def _run(self, prompt: str) -> dict:
        proc = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json", "--model", self.model],
            capture_output=True, text=True, timeout=self.timeout_s)
        if proc.returncode != 0:
            raise RuntimeError(f"claude rc={proc.returncode}: {proc.stderr.strip()[-300:]}")
        text = json.loads(proc.stdout)["result"]
        return json.loads(_strip_fence(text))

    def complete(self, prompt: str, schema: dict) -> dict:
        try:
            obj = self._run(prompt)
            _validate(obj, schema)
            return obj
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"claude timed out after {self.timeout_s}s") from exc
        except (json.JSONDecodeError, ValueError, KeyError):
            # one retry with an explicit JSON-only nudge
            try:
                obj = self._run(prompt + "\n\nReturn ONLY the JSON object, no prose, no markdown.")
                _validate(obj, schema)
                return obj
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"claude timed out after {self.timeout_s}s") from exc
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                raise RuntimeError(f"claude returned unparseable/invalid output: {exc}") from exc
