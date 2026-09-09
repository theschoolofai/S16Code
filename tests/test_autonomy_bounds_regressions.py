"""Two regressions in the bounds an unattended agent is supposed to hold."""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from s16code.events import (
    AutonomousEventEngine,
    EventEnvelope,
    EventStore,
    Subscription,
    morning_report,
)


def _event(**changes) -> EventEnvelope:
    body = {"id": "e1", "source": "deployments", "type": "deployment.completed",
            "subject": "checkout/production", "occurred_at": datetime.now(UTC),
            "data": {"current_error_rate": 6.4}}
    body.update(changes)
    return EventEnvelope(**body)


def _subscription(**changes) -> Subscription:
    body = {"id": "watch", "instruction": "Assess material error-rate increases.",
            "event_types": ["deployment.completed"], "sources": ["deployments"],
            "tenant_id": "course", "allowed_side_effects": [], "budget": 0.02,
            "max_runs_per_day": 1}
    body.update(changes)
    return Subscription(**body)


def _relevance_llm(*, stall: bool = False):
    async def llm(prompt: str, system: str):  # noqa: ARG001
        if stall:
            await asyncio.sleep(0)
        return {"text": json.dumps({"relevant": True, "reason": "error rate rose",
                                    "goal": "Assess the increase."}),
                "metered_calls": [{"cost_usd": 0.0}]}
    return llm


class _Runtime:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.runs: list[str] = []
        self.delay = delay

    async def run(self, *, prompt: str, **_: object) -> dict[str, object]:
        if self.delay:
            await asyncio.sleep(self.delay)
        self.runs.append(prompt)
        return {"run_id": f"run-{len(self.runs)}", "status": "completed", "spend_usd": 0.0}


async def test_the_daily_run_ceiling_holds_when_two_events_arrive_together(tmp_path) -> None:
    """`max_runs_per_day` is read before the run and written after it.

    Two deliveries in flight at once both read the stale count, so a ceiling of
    one admits two runs and records no refusal at all.
    """
    store = EventStore(tmp_path)
    runtime = _Runtime(delay=0.01)
    engine = AutonomousEventEngine(store, runtime)
    store.put_subscription(_subscription(max_runs_per_day=1))

    await asyncio.gather(
        engine.process(_event(id="a"), llm=_relevance_llm(stall=True)),
        engine.process(_event(id="b"), llm=_relevance_llm(stall=True)),
    )

    assert len(runtime.runs) == 1, "the second event must not become a run"
    assert [item["control"] for item in store.refusals()] == ["max_runs_per_day"]


async def test_one_refusal_is_reported_once_not_twice(tmp_path) -> None:
    """A refusal is written to the ledger *and* onto the event's decision.

    The morning report reads both and adds them, so the single refusal that
    stopped one run is presented to an operator as two blocked pieces of work.
    """
    store = EventStore(tmp_path)
    engine = AutonomousEventEngine(store, _Runtime())
    store.put_subscription(_subscription())

    await engine.process(_event(id="a"), llm=_relevance_llm())  # uses the one run
    await engine.process(_event(id="b"), llm=_relevance_llm())  # refused

    assert len(store.refusals()) == 1, "exactly one control fired"

    report = morning_report(store)
    assert report["totals"]["blocked_by_a_control"] == 1
    assert len(report["refused"]) == 1
