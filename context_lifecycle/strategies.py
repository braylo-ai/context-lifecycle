"""Compaction strategies and the explicit forgetting policy.

Every strategy holds an in-context list of ContextItems under a token budget.
  full            - never compacts (upper bound on accuracy and on cost)
  drop_oldest     - evicts the oldest items until under budget
  rolling_summary - folds the oldest items into one summary item (lossy, bounded)
  retrieval       - evicts the oldest items to an archive; retrieves top-k at question time
Any strategy can take a ForgettingPolicy, applied before eviction.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from typing import Optional

from .items import ContextItem, total_tokens

_W = re.compile(r"[a-z0-9\$\-\.]+")


class ForgettingPolicy:
    """Explicit retain/drop rules. Every drop is logged with a reason.

    RETAIN  - user constraints and goals (pinned: never evicted or summarized away)
    RETAIN  - the latest value of every fact key
    DROP    - facts superseded by a newer value for the same key       (reason: superseded)
    DROP    - tool logs / agent notes with no facts after `ttl` steps   (reason: ttl_expired)
    """

    def __init__(self, ttl: int = 3) -> None:
        self.ttl = ttl
        self.log: list[tuple[str, str]] = []  # (item id, reason)

    def pinned(self, item: ContextItem) -> bool:
        return item.kind == "constraint"

    def apply(self, items: list[ContextItem], new: ContextItem) -> list[ContextItem]:
        keep = []
        new_keys = {(a, e) for a, e, _, _ in new.facts()}
        for it in items:
            reason = None
            keys = {(a, e) for a, e, _, _ in it.facts()}
            if it.kind != "constraint" and keys and keys <= new_keys:
                reason = "superseded"
            elif not keys and it.kind in ("log", "note") and new.step - it.step >= self.ttl:
                reason = "ttl_expired"
            if reason:
                self.log.append((it.id, reason))
            else:
                keep.append(it)
        return keep

    def reasons(self) -> dict:
        return dict(Counter(r for _, r in self.log))


class Strategy:
    name = "base"

    def __init__(
        self, budget: int = 2000, policy: Optional[ForgettingPolicy] = None
    ) -> None:
        self.budget = budget
        self.policy = policy
        self.items: list[ContextItem] = []
        self.compactions = 0

    def label(self) -> str:
        return self.name + ("+policy" if self.policy else "")

    def _pinned(self, it: ContextItem) -> bool:
        return self.policy is not None and self.policy.pinned(it)

    def add(self, item: ContextItem) -> None:
        if self.policy is not None:
            self.items = self.policy.apply(self.items, item)
        self.items.append(item)
        if total_tokens(self.items) > self.budget:
            self.compactions += 1
            self.compact()

    def compact(self) -> None:
        raise NotImplementedError

    def context_for(self, query: Optional[str] = None) -> list[ContextItem]:
        return list(self.items)

    def _pop_oldest_unpinned(self) -> Optional[ContextItem]:
        for i, it in enumerate(self.items):
            if not self._pinned(it) and it.kind != "summary":
                return self.items.pop(i)
        return None


class Full(Strategy):
    name = "full"

    def __init__(self, policy: Optional[ForgettingPolicy] = None) -> None:
        super().__init__(budget=math.inf, policy=policy)


class DropOldest(Strategy):
    name = "drop_oldest"

    def compact(self) -> None:
        while total_tokens(self.items) > self.budget:
            if self._pop_oldest_unpinned() is None:
                break


class RollingSummary(Strategy):
    """Folds the oldest half of the budget into one summary item.

    Mock summarizer (no LLM): extracts fact triples, keeps the latest value per key, writes
    "the A for E is V [as of step N]." lines, and caps the summary at `summary_budget` tokens
    by dropping the oldest triples. `fidelity` < 1 drops each triple with probability
    1 - fidelity on every re-summarization pass, a stand-in for compounding paraphrase loss.
    """

    name = "rolling_summary"

    def __init__(
        self,
        budget: int = 2000,
        policy=None,
        summary_budget: Optional[int] = None,
        fidelity: float = 0.97,
        seed: int = 0,
    ) -> None:
        super().__init__(budget, policy)
        self.summary_budget = summary_budget or budget // 3
        self.fidelity = fidelity
        self.rng = random.Random(seed)

    def compact(self) -> None:
        old_summary = next((i for i in self.items if i.kind == "summary"), None)
        folded = []
        target = self.budget // 2
        while total_tokens(self.items) > target:
            it = self._pop_oldest_unpinned()
            if it is None:
                break
            folded.append(it)
        if not folded:
            return
        triples: dict[tuple, tuple[str, int]] = {}
        sources = folded + ([old_summary] if old_summary else [])
        for it in sources:
            for a, e, v, s in it.facts():
                if (a, e) not in triples or s >= triples[(a, e)][1]:
                    triples[(a, e)] = (v, s)
        kept = [(k, v) for k, v in triples.items() if self.rng.random() < self.fidelity]
        kept.sort(key=lambda kv: kv[1][1], reverse=True)  # newest first
        lines, used = [], 4
        for (a, e), (v, s) in kept:
            line = f"the {a} for {e} is {v} [as of step {s}]."
            t = ContextItem(line, "summary", "summary", s).tokens
            if used + t > self.summary_budget:
                break
            lines.append(line)
            used += t
        derived = tuple(i.id for i in folded) + (
            old_summary.derived_from if old_summary else ()
        )
        step = max(i.step for i in sources)
        summary = ContextItem(
            "Summary of earlier work: " + " ".join(lines),
            "summary",
            "summary",
            step,
            derived_from=derived,
        )
        if old_summary is not None:
            self.items.remove(old_summary)
        self.items.insert(0, summary)


class RetrievalArchive(Strategy):
    """Evicts oldest items into an archive (nothing is lost); BM25 top-k retrieval per query."""

    name = "retrieval"

    def __init__(self, budget: int = 2000, policy=None, k: int = 4) -> None:
        super().__init__(budget, policy)
        self.k = k
        self.archive: list[ContextItem] = []

    def add(self, item: ContextItem) -> None:
        if self.policy is not None:
            before = {i.id for i in self.archive}
            self.archive = self.policy.apply(self.archive, item)
            # archive drops are also explicit forgetting; nothing else changes
            del before
        super().add(item)

    def compact(self) -> None:
        while total_tokens(self.items) > self.budget:
            it = self._pop_oldest_unpinned()
            if it is None:
                break
            self.archive.append(it)

    def retrieve(self, query: str) -> list[ContextItem]:
        if not self.archive:
            return []
        docs = [_W.findall(d.text.lower()) for d in self.archive]
        q = set(_W.findall(query.lower()))
        n = len(docs)
        avgdl = sum(len(d) for d in docs) / n
        df = Counter(w for d in docs for w in set(d))
        scores = []
        for idx, d in enumerate(docs):
            tf = Counter(d)
            s = 0.0
            for w in q:
                if w in tf:
                    idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                    s += (
                        idf
                        * tf[w]
                        * 2.2
                        / (tf[w] + 1.2 * (0.25 + 0.75 * len(d) / avgdl))
                    )
            scores.append((s, idx))
        scores.sort(reverse=True)
        return [self.archive[i] for s, i in scores[: self.k] if s > 0]

    def context_for(self, query: Optional[str] = None) -> list[ContextItem]:
        if query is None:
            return list(self.items)
        return self.retrieve(query) + list(self.items)
