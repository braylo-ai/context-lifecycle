"""Accuracy vs tokens for each strategy on SYNTHETIC long tasks.

    python -m context_lifecycle.evaluate          # writes results/eval.json + results/eval.md
    python -m context_lifecycle.evaluate --quick
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .items import total_tokens
from .reader import answer
from .strategies import (
    DropOldest,
    ForgettingPolicy,
    Full,
    RetrievalArchive,
    RollingSummary,
)
from .tasks import make_task


def factories(budget: int):
    return [
        lambda: Full(),
        lambda: DropOldest(budget),
        lambda: RollingSummary(budget),
        lambda: RetrievalArchive(budget),
        lambda: DropOldest(budget, ForgettingPolicy()),
        lambda: RollingSummary(budget, ForgettingPolicy()),
        lambda: RetrievalArchive(budget, ForgettingPolicy()),
    ]


def run_strategy(strategy, task) -> dict:
    qs = defaultdict(list)
    for q in task.questions:
        qs[q.step].append(q)
    step_tokens, q_tokens = [], []
    correct, stale, missing = Counter(), Counter(), Counter()
    kinds = Counter()
    cited_ok = 0
    for s, items in enumerate(task.steps, start=1):
        for it in items:
            strategy.add(it)
        step_tokens.append(total_tokens(strategy.context_for(None)))
        for q in qs.get(s, []):
            ctx = strategy.context_for(q.text)
            q_tokens.append(total_tokens(ctx))
            val, prov = answer(q.attr, q.entity, ctx)
            kinds[q.kind] += 1
            if val == q.answer:
                correct[q.kind] += 1
                cited_ok += prov is not None and bool(prov["id"])
            elif val is None:
                missing[q.kind] += 1
            elif val in q.stale:
                stale[q.kind] += 1
    n = sum(kinds.values())
    return {
        "n_questions": n,
        "correct": sum(correct.values()),
        "stale": sum(stale.values()),
        "missing": sum(missing.values()),
        "by_kind": {k: [correct[k], kinds[k]] for k in kinds},
        "mean_step_tokens": sum(step_tokens) / len(step_tokens),
        "total_step_tokens": sum(step_tokens),
        "mean_question_tokens": sum(q_tokens) / len(q_tokens),
        "compactions": strategy.compactions,
        "cited_answers": cited_ok,
        "forgotten": strategy.policy.reasons() if strategy.policy else {},
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    seeds = range(3) if a.quick else range(20)
    n_steps = 80 if a.quick else 200
    budgets = [1000, 2000, 4000]
    rows = []
    seen_full = False
    for b in budgets:
        for f in factories(b):
            label = f().label()
            if label == "full" and seen_full:
                continue
            seen_full |= label == "full"
            agg = Counter()
            by_kind = defaultdict(lambda: [0, 0])
            forgotten = Counter()
            st_tok, q_tok = [], []
            for seed in seeds:
                r = run_strategy(f(), make_task(seed, n_steps=n_steps))
                for k in (
                    "n_questions",
                    "correct",
                    "stale",
                    "missing",
                    "compactions",
                    "cited_answers",
                ):
                    agg[k] += r[k]
                for k, (c, t) in r["by_kind"].items():
                    by_kind[k][0] += c
                    by_kind[k][1] += t
                forgotten.update(r["forgotten"])
                st_tok.append(r["mean_step_tokens"])
                q_tok.append(r["mean_question_tokens"])
            n = agg["n_questions"]
            rows.append(
                {
                    "strategy": label,
                    "budget": None if label == "full" else b,
                    "accuracy": round(agg["correct"] / n, 4),
                    "stale_rate": round(agg["stale"] / n, 4),
                    "missing_rate": round(agg["missing"] / n, 4),
                    "accuracy_by_kind": {
                        k: round(c / t, 4) for k, (c, t) in sorted(by_kind.items())
                    },
                    "mean_context_tokens_per_step": round(sum(st_tok) / len(st_tok), 1),
                    "mean_tokens_per_question": round(sum(q_tok) / len(q_tok), 1),
                    "questions": n,
                    "compactions": agg["compactions"],
                    "answers_with_provenance": agg["cited_answers"],
                    "forgotten_by_reason": dict(forgotten),
                }
            )
            print(
                rows[-1]["strategy"],
                b,
                rows[-1]["accuracy"],
                rows[-1]["mean_context_tokens_per_step"],
                flush=True,
            )
    out = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "note": "SYNTHETIC tasks; deterministic reader (not an LLM); mock summarizer; regex token count",
        "seeds": list(seeds),
        "steps_per_task": n_steps,
        "rows": rows,
    }
    md = [
        "| strategy | budget | accuracy | stale | missing | constraint acc | updated acc | static acc | "
        "ctx tokens/step | tokens/question |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        k = r["accuracy_by_kind"]
        md.append(
            f"| {r['strategy']} | {r['budget'] or '∞'} | {r['accuracy']:.3f} | {r['stale_rate']:.3f} | "
            f"{r['missing_rate']:.3f} | {k.get('constraint', 0):.3f} | {k.get('updated', 0):.3f} | "
            f"{k.get('static', 0):.3f} | {r['mean_context_tokens_per_step']:.0f} | {r['mean_tokens_per_question']:.0f} |"
        )
    if not a.quick:
        p = Path(a.out)
        p.mkdir(exist_ok=True)
        (p / "eval.json").write_text(json.dumps(out, indent=2))
        (p / "eval.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
