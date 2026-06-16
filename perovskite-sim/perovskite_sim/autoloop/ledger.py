# perovskite_sim/autoloop/ledger.py
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable

from perovskite_sim.autoloop.types import Gap, Hypothesis, NegativeResult


class Ledger:
    """Persistent gap / hypothesis / negative-result store.

    Append-and-replace semantics: gaps dedup on ``id`` (latest wins),
    negative results dedup on the normalised approach key. Updates produce
    new dataclass instances; nothing is mutated in place.
    """

    def __init__(self, root: Path,
                 gaps: Iterable[Gap] = (),
                 hypotheses: Iterable[Hypothesis] = (),
                 negatives: Iterable[NegativeResult] = ()):
        self.root = Path(root)
        self.gaps: list[Gap] = list(gaps)
        self.hypotheses: list[Hypothesis] = list(hypotheses)
        self.negatives: list[NegativeResult] = list(negatives)

    # ---- mutation (append/replace only) ----
    def add_gap(self, gap: Gap) -> None:
        self.gaps = [g for g in self.gaps if g.id != gap.id] + [gap]

    def add_hypothesis(self, hyp: Hypothesis) -> None:
        self.hypotheses = list(self.hypotheses) + [hyp]

    def add_negative(self, neg: NegativeResult) -> None:
        if not self.is_refuted(neg.approach):
            self.negatives = list(self.negatives) + [neg]

    # ---- queries ----
    def is_refuted(self, approach: str) -> bool:
        key = " ".join(approach.lower().split())
        return any(n.dedup_key() == key for n in self.negatives)

    # ---- persistence ----
    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _dump(self.root / "gaps.json", self.gaps)
        _dump(self.root / "hypotheses.json", self.hypotheses)
        _dump(self.root / "negative_results.json", self.negatives)
        (self.root / "LEDGER.md").write_text(self._markdown(), encoding="utf-8")

    @classmethod
    def load(cls, root: Path) -> "Ledger":
        root = Path(root)
        return cls(
            root=root,
            gaps=_load(root / "gaps.json", Gap),
            hypotheses=_load(root / "hypotheses.json", Hypothesis),
            negatives=_load(root / "negative_results.json", NegativeResult),
        )

    def _markdown(self) -> str:
        lines = ["# Autoloop ledger\n", "## Open gaps\n"]
        for g in sorted(self.gaps, key=lambda x: -x.gap_mag):
            lines.append(f"- `{g.id}` [{g.status}] {g.metric}@{g.sweep} gap={g.gap_mag:.4g} "
                         f"(SL {g.solarlab_val:.4g} vs ref {g.reference_val:.4g})")
        lines.append("\n## Refuted approaches (never retry)\n")
        for n in self.negatives:
            lines.append(f"- {n.approach} — {n.why_failed}")
        return "\n".join(lines) + "\n"


def _dump(path: Path, items: list) -> None:
    path.write_text(
        json.dumps([dataclasses.asdict(i) for i in items], indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load(path: Path, cls):
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    fields = {f.name for f in dataclasses.fields(cls)}
    out = []
    for d in raw:
        kw = {k: v for k, v in d.items() if k in fields}
        # JSON lists -> tuples for frozen tuple fields
        for f in dataclasses.fields(cls):
            if f.name in kw and isinstance(kw[f.name], list) and "tuple" in str(f.type):
                kw[f.name] = tuple(kw[f.name])
        out.append(cls(**kw))
    return out
