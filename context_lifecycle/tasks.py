"""SYNTHETIC long multi-step tasks.

Each task is a stream of steps. A step adds tool logs (noise), agent notes, new facts and
updates that supersede old facts. User constraints arrive at step 0. Questions are asked at
checkpoints; the gold answer is the latest value of the key as of that step.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .items import ContextItem

ATTRS = ["deploy region", "owner", "version", "port", "db shard", "retry limit"]
ENTITIES = [
    "orion",
    "lyra",
    "vega",
    "atlas",
    "nova",
    "pyxis",
    "draco",
    "hydra",
    "carina",
    "lynx",
    "auriga",
    "corvus",
    "dorado",
    "fornax",
    "grus",
    "indus",
    "lepus",
    "mensa",
    "norma",
    "pavo",
]
CONSTRAINTS = [
    ("budget cap", "project", "$4000"),
    ("freeze day", "release", "friday"),
    ("approver", "change", "user42"),
]
WORDS = (
    "request latency cache warm cold retry header payload index scan shard queue worker "
    "batch flush commit rollback trace span metric gauge counter alert page token stream "
    "buffer socket handshake timeout backoff jitter lease lock snapshot replica"
).split()


@dataclass
class Question:
    step: int
    attr: str
    entity: str
    answer: str
    kind: str  # constraint | static | updated
    stale: tuple = ()  # older values that would be a stale answer

    @property
    def text(self) -> str:
        return f"What is the {self.attr} for {self.entity}?"


@dataclass
class Task:
    seed: int
    steps: list[list[ContextItem]]
    questions: list[Question]


def _value(rng: random.Random, attr: str) -> str:
    return {
        "deploy region": lambda: f"zone-{rng.randint(1, 99)}",
        "owner": lambda: f"user{rng.randint(1, 999)}",
        "version": lambda: f"v{rng.randint(1, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 9)}",
        "port": lambda: str(rng.randint(1024, 9999)),
        "db shard": lambda: f"shard-{rng.randint(1, 64)}",
        "retry limit": lambda: str(rng.randint(2, 12)),
    }[attr]()


def make_task(
    seed: int,
    n_steps: int = 200,
    checkpoint_every: int = 25,
    q_per_checkpoint: int = 6,
    p_fact: float = 0.5,
    p_update: float = 0.15,
    p_buried: float = 0.4,
) -> Task:
    rng = random.Random(seed)
    steps: list[list[ContextItem]] = []
    state: dict[tuple, list[tuple[str, int]]] = {}
    questions: list[Question] = []
    first = [
        ContextItem(
            f"User constraint: the {a} for {e} is {v}.", "user", "constraint", 0
        )
        for a, e, v in CONSTRAINTS
    ]
    for a, e, v in CONSTRAINTS:
        state[(a, e)] = [(v, 0)]
    first.append(
        ContextItem(
            "User goal: migrate all services and report their final settings.",
            "user",
            "constraint",
            0,
        )
    )
    for s in range(1, n_steps + 1):
        items = first if s == 1 else []
        noise = " ".join(rng.choice(WORDS) for _ in range(rng.randint(50, 110)))
        for _ in range(rng.choice([0, 1, 1, 2])):  # lexical distractors for retrieval
            noise += f" {rng.choice(ENTITIES)} {rng.choice(ATTRS)} probe ok."
        facts = []  # (sentence, kind)
        if rng.random() < p_fact:
            key = (rng.choice(ATTRS), rng.choice(ENTITIES))
            if key not in state:
                v = _value(rng, key[0])
                state[key] = [(v, s)]
                facts.append((f"the {key[0]} for {key[1]} is {v}.", "fact"))
        upd = [k for k in state if k[1] not in ("project", "release", "change")]
        if upd and rng.random() < p_update:
            key = rng.choice(upd)
            v = _value(rng, key[0])
            state[key].append((v, s))
            facts.append((f"update: the {key[0]} for {key[1]} is now {v}.", "update"))
        buried = [f for f in facts if rng.random() < p_buried]
        if buried:  # fact hidden inside a long tool log
            noise += " " + " ".join(f for f, _ in buried)
        items.append(ContextItem(f"Tool log (step {s}): {noise}", "tool", "log", s))
        for f, kind in facts:
            if (f, kind) not in buried:
                items.append(ContextItem(f"Observation (step {s}): {f}", "tool", kind, s))
        e = rng.choice(ENTITIES)
        items.append(ContextItem(f"Agent note (step {s}): checked {e} {rng.choice(ATTRS)}, continuing.",
                                 "agent", "note", s))
        steps.append(items)
        if s % checkpoint_every == 0:
            keys = list(state)
            rng.shuffle(keys)
            cons = [k for k in keys if k[1] in ("project", "release", "change")][:1]
            upd_k = [k for k in keys if len(state[k]) > 1][: q_per_checkpoint // 2]
            static = [k for k in keys if len(state[k]) == 1 and k not in cons]
            chosen = cons + upd_k + static[: q_per_checkpoint - len(cons) - len(upd_k)]
            for k in chosen:
                hist = state[k]
                kind = (
                    "constraint"
                    if k in cons
                    else ("updated" if len(hist) > 1 else "static")
                )
                questions.append(
                    Question(
                        s, k[0], k[1], hist[-1][0], kind, tuple(v for v, _ in hist[:-1])
                    )
                )
    return Task(seed, steps, questions)
