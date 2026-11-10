"""Context items with provenance tags, a token counter and the fact pattern."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

_TOK = re.compile(r"\w+|[^\w\s]")
# "the <attr> for <entity> is [now] <value>" with optional "[as of step N]"
FACT = re.compile(
    r"the ([a-z][a-z ]*?) for (\w+) is (?:now )?([\w\-\$]+(?:\.[\w\-\$]+)*)(?: \[as of step (\d+)\])?"
)


def count_tokens(text: str) -> int:
    """Approximate tokens: words + punctuation marks (regex tokenizer, not a BPE)."""
    return len(_TOK.findall(text))


_ids = itertools.count(1)


@dataclass
class ContextItem:
    """One unit of context. Provenance fields are mandatory:
    id      - unique id
    source  - user | tool | agent | summary
    kind    - constraint | fact | update | log | note | summary
    step    - task step that produced it
    derived_from - ids of the items a summary was built from (empty for raw items)
    """

    text: str
    source: str
    kind: str
    step: int
    derived_from: tuple = ()
    id: str = field(default_factory=lambda: f"c{next(_ids)}")
    tokens: int = 0

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = count_tokens(self.text)

    @property
    def provenance(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "kind": self.kind,
            "step": self.step,
            "derived_from": list(self.derived_from),
        }

    def facts(self) -> list[tuple[str, str, str, int]]:
        """(attr, entity, value, step) triples stated in this item."""
        out = []
        for m in FACT.finditer(self.text):
            step = int(m.group(4)) if m.group(4) else self.step
            out.append((m.group(1), m.group(2), m.group(3), step))
        return out


def total_tokens(items) -> int:
    return sum(i.tokens for i in items)
