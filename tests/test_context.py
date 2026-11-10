from context_lifecycle.evaluate import run_strategy
from context_lifecycle.items import ContextItem, count_tokens, total_tokens
from context_lifecycle.reader import answer
from context_lifecycle.strategies import (
    DropOldest,
    ForgettingPolicy,
    Full,
    RetrievalArchive,
    RollingSummary,
)
from context_lifecycle.tasks import make_task


def fact(attr, ent, val, step, kind="fact"):
    return ContextItem(
        f"Observation (step {step}): the {attr} for {ent} is {val}.", "tool", kind, step
    )


def log(step, n=40):
    return ContextItem(f"Tool log (step {step}): " + "noise " * n, "tool", "log", step)


def constraint(step=0):
    return ContextItem(
        "User constraint: the budget cap for project is $4000.",
        "user",
        "constraint",
        step,
    )


# ---- items / provenance -------------------------------------------------------
def test_every_item_has_provenance_tags():
    it = fact("port", "orion", "8080", 3)
    p = it.provenance
    assert (
        p["id"].startswith("c")
        and p["source"] == "tool"
        and p["kind"] == "fact"
        and p["step"] == 3
    )
    assert p["derived_from"] == []


def test_ids_are_unique():
    assert len({fact("port", "a", "1", 1).id for _ in range(50)}) == 50


def test_token_counter_and_fact_parse():
    assert count_tokens("the port for orion is 8080.") == 7
    assert fact("version", "lyra", "v1.2.3", 4).facts() == [
        ("version", "lyra", "v1.2.3", 4)
    ]
    s = ContextItem(
        "the owner for vega is user7 [as of step 12].", "summary", "summary", 30
    )
    assert s.facts() == [("owner", "vega", "user7", 12)]


# ---- reader -------------------------------------------------------------------
def test_reader_prefers_latest_value_and_returns_provenance():
    old, new = fact("port", "orion", "1111", 1), fact(
        "port", "orion", "2222", 9, "update"
    )
    val, prov = answer("port", "orion", [new, old])
    assert val == "2222" and prov["id"] == new.id


def test_reader_returns_none_when_absent():
    assert answer("port", "orion", [log(1)]) == (None, None)


# ---- strategies ------------------------------------------------------------------
def test_full_never_compacts():
    s = Full()
    for i in range(100):
        s.add(log(i))
    assert s.compactions == 0 and len(s.items) == 100


def test_drop_oldest_respects_budget_and_drops_constraints_without_policy():
    s = DropOldest(budget=200)
    s.add(constraint())
    for i in range(1, 30):
        s.add(log(i))
    assert total_tokens(s.items) <= 200
    assert all(i.kind != "constraint" for i in s.items)


def test_policy_pins_constraints_through_eviction():
    s = DropOldest(budget=200, policy=ForgettingPolicy())
    s.add(constraint())
    for i in range(1, 30):
        s.add(log(i))
    assert any(i.kind == "constraint" for i in s.items)


def test_policy_drops_superseded_fact_with_reason():
    pol = ForgettingPolicy()
    s = Full(policy=pol)
    s.add(fact("port", "orion", "1111", 1))
    s.add(fact("port", "orion", "2222", 2, "update"))
    assert [i.facts()[0][2] for i in s.items] == ["2222"]
    assert pol.reasons() == {"superseded": 1}


def test_policy_ttl_drops_factless_logs_but_keeps_logs_with_facts():
    pol = ForgettingPolicy(ttl=3)
    s = Full(policy=pol)
    buried = ContextItem(
        "Tool log (step 1): noise the port for orion is 9.", "tool", "log", 1
    )
    s.add(log(1))
    s.add(buried)
    s.add(log(5))
    assert buried in s.items and len(s.items) == 2
    assert pol.reasons()["ttl_expired"] == 1


def test_rolling_summary_keeps_facts_and_records_derivation():
    s = RollingSummary(budget=150, fidelity=1.0)
    raw = [fact("port", "orion", "8080", 1)] + [log(i, 20) for i in range(2, 12)]
    for it in raw:
        s.add(it)
    summ = [i for i in s.items if i.kind == "summary"]
    assert len(summ) == 1 and summ[0].source == "summary"
    assert raw[0].id in summ[0].derived_from
    assert answer("port", "orion", s.items)[0] == "8080"
    assert total_tokens(s.items) <= 150 + max(i.tokens for i in raw)


def test_rolling_summary_is_bounded():
    s = RollingSummary(budget=300, summary_budget=60, fidelity=1.0)
    for i in range(1, 200):
        s.add(fact("port", f"e{i}", str(i), i))
    summ = next(i for i in s.items if i.kind == "summary")
    assert summ.tokens <= 60


def test_retrieval_archive_loses_nothing_and_retrieves_by_query():
    s = RetrievalArchive(budget=150, k=2)
    target = fact("owner", "vega", "user9", 1)
    s.add(target)
    for i in range(2, 30):
        s.add(log(i, 20))
    assert target not in s.items and target in s.archive
    ctx = s.context_for("What is the owner for vega?")
    assert answer("owner", "vega", ctx)[0] == "user9"
    assert s.context_for(None) == s.items  # no retrieval cost on normal steps


# ---- tasks + eval ------------------------------------------------------------------
def test_task_is_deterministic_and_answers_are_latest_values():
    a, b = make_task(5, n_steps=60), make_task(5, n_steps=60)
    assert [q.answer for q in a.questions] == [q.answer for q in b.questions]
    assert any(q.kind == "updated" and q.stale for q in a.questions)
    assert all(q.step % 25 == 0 for q in a.questions)


def test_full_context_is_accuracy_upper_bound():
    t = make_task(1, n_steps=100)
    full = run_strategy(Full(), t)
    drop = run_strategy(DropOldest(1000), t)
    assert full["correct"] == full["n_questions"]
    assert drop["correct"] <= full["correct"]
    assert drop["mean_step_tokens"] < full["mean_step_tokens"]


def test_policy_improves_drop_oldest_at_same_budget():
    t = make_task(2, n_steps=150)
    base = run_strategy(DropOldest(1000), t)
    pol = run_strategy(DropOldest(1000, ForgettingPolicy()), t)
    assert pol["correct"] > base["correct"]
