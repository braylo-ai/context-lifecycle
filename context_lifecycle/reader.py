"""Deterministic reader (NOT an LLM). Answers "What is the A for E?" from the visible
context only: the stated value with the highest step wins. It isolates what each strategy
keeps; it does not model LLM reasoning errors on long contexts."""

from __future__ import annotations

from typing import Optional

from .items import ContextItem


def answer(
    attr: str, entity: str, context: list[ContextItem]
) -> tuple[Optional[str], Optional[dict]]:
    """Returns (value, provenance of the item it came from)."""
    best = None
    for it in context:
        for a, e, v, s in it.facts():
            if a == attr and e == entity and (best is None or s >= best[0]):
                best = (s, v, it)
    if best is None:
        return None, None
    return best[1], best[2].provenance
