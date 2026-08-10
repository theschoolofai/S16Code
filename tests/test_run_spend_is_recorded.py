"""`daily_budget` must be fed what a run actually cost.

`AutonomousEventEngine` records a completed run's spend like this:

    self.governor.record(subscription.id, kind="run",
                         usd=float(result.get("spend_usd") or 0.0), now=now)

`Runtime.run()` never returns a `spend_usd` key. Its result dict is:

    {"run_id", "status", "answer", "provider", "model", "graph", "trace",
     "events", "principal", "budget", "economics", "allocations"}

The real figure is `result["budget"]["spent"]`. So `.get("spend_usd")` is always
None, every run is recorded at $0.00, and `window_spend(..., kind="run")` stays
zero for the life of the installation.

## Three consequences, one root cause

1. `daily_budget` can never bind. `admit_run` compares the window's run spend
   against it, and that spend is always 0. Section 10 makes this the ceiling that
   survives "escape the budget by starting more runs".

2. The per-run ceiling is never squeezed by the window. `admit_run` computes
   `remaining = daily_budget - spent` and hands the run
   `min(per_run, remaining)`; with `spent` pinned at 0, `remaining` is always the
   full daily budget. Section 10 asks for "each run's ceiling shrunk by whatever
   the window has left". It never shrinks.

3. The morning report's "cost of doing" is always $0.00. Observed live on
   2026-08-10 after four real runs against Gemini:

       - acted: 4
       - cost of watching: $0.00066000
       - cost of doing:    $0.00000000

   An operator reading Section 9's report sees a night that cost nothing.

`max_runs_per_day` still bounds the count, so this is not unbounded spend - but
the money ceiling specifically is inert, and the report understates the bill.

## Why the suite did not catch it

`tests/test_autonomy_governor.py` defines

    class _Runtime:
        async def run(self, *, prompt, **_):
            return {"run_id": ..., "status": "completed", "spend_usd": self.spend}

The stub invents the key production never sends, so
`test_a_daily_budget_caps_the_window_and_shrinks_the_last_run` passes while the
control it covers cannot fire. Section 12: "a green check on a control that never
ran is worse than a red one." This is the second instance of that shape in the
same file - the other was `metered_calls` for triage.

The stub here deliberately mirrors the *real* `Runtime.run` contract instead.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from s16code.events import (
    AutonomousEventEngine,
    EventEnvelope,
    EventStore,
    Subscription,
)


def _event(**changes) -> EventEnvelope:
    body = {"id": "e1", "source": "deployments", "type": "deployment.completed",
            "subject": "checkout/production", "occurred_at": datetime.now(UTC),
            "data": {"previous_error_rate": 0.7, "current_error_rate": 6.4}}
    body.update(changes)
    return EventEnvelope(**body)


def _subscription(**changes) -> Subscription:
    body = {"id": "watch", "instruction": "Assess material error-rate increases.",
            "event_types": ["deployment.completed"], "sources": ["deployments"],
            "tenant_id": "course", "allowed_side_effects": [], "budget": 0.01}
    body.update(changes)
    return Subscription(**body)


class _RealShapedRuntime:
    """Returns what `s16code.runtime.Runtime.run` actually returns.

    Note what is absent: `spend_usd`. Spend is reported inside `budget`, which is
    `run_budget.snapshot()` - `{"total": ..., "spent": ...}`.
    """

    def __init__(self, spend: float = 0.004) -> None:
        self.runs: list[str] = []
        self.spend = spend

    async def run(self, *, prompt: str, **_: object) -> dict[str, object]:
        self.runs.append(prompt)
        return {
            "run_id": f"run-{len(self.runs)}",
            "status": "completed",
            "answer": "assessed",
            "provider": "gemini",
            "model": "gemini-3.1-flash-lite",
            "graph": {"finished": True, "nodes": {}, "edges": []},
            "trace": {"planner": {}, "agents": []},
            "events": [],
            "principal": "course",
            "budget": {"total": 0.01, "spent": self.spend},
            "economics": None,
            "allocations": [],
        }


def _relevance_llm(cost: float = 0.00002):
    async def llm(prompt: str, system: str):  # noqa: ARG001
        return {"text": json.dumps({"relevant": True, "reason": "error rate rose",
                                    "goal": "Assess the increase."}),
                "metered_calls": [{"cost_usd": cost}]}
    return llm


def _day(moment: datetime | None = None) -> str:
    return (moment or datetime.now(UTC)).strftime("%Y-%m-%d")


async def test_a_completed_run_records_what_it_spent(tmp_path) -> None:
    store = EventStore(tmp_path)
    runtime = _RealShapedRuntime(spend=0.004)
    engine = AutonomousEventEngine(store, runtime)
    store.put_subscription(_subscription())

    await engine.process(_event(id="spend-1"), llm=_relevance_llm())

    recorded = store.window_spend("watch", _day(), kind="run")
    assert recorded == pytest.approx(0.004), (
        f"the run spent $0.004 and the ledger recorded ${recorded:.8f}; the engine "
        "reads result['spend_usd'], a key Runtime.run does not return"
    )


async def test_daily_budget_actually_refuses_once_the_window_is_spent(tmp_path) -> None:
    """The control Section 10 relies on, exercised against the real result shape."""
    store = EventStore(tmp_path)
    runtime = _RealShapedRuntime(spend=0.004)
    engine = AutonomousEventEngine(store, runtime)
    store.put_subscription(_subscription(budget=0.004, daily_budget=0.01))

    for index in range(4):
        await engine.process(_event(id=f"spend-{index}"), llm=_relevance_llm())

    assert len(runtime.runs) < 4, (
        f"all {len(runtime.runs)} runs were admitted against a $0.01 daily budget "
        "at $0.004 each; the window ceiling never fired"
    )
    assert any(item["control"] == "daily_budget" for item in store.refusals()), (
        "no daily_budget refusal was recorded, so nothing tells an operator the "
        "window stopped the work"
    )


async def test_the_remaining_window_shrinks_a_later_run(tmp_path) -> None:
    """Section 10: a run's ceiling is cut down by whatever the window has left."""
    store = EventStore(tmp_path)
    runtime = _RealShapedRuntime(spend=0.004)
    engine = AutonomousEventEngine(store, runtime)
    store.put_subscription(_subscription(budget=0.01, daily_budget=0.01))

    await engine.process(_event(id="first"), llm=_relevance_llm())
    spent = store.window_spend("watch", _day(), kind="run")
    assert spent > 0, "nothing was recorded, so no later run can be squeezed"
    assert spent < 0.01
