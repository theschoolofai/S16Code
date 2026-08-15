"""Regression test: the daily run ceiling must hold under concurrent arrival.

`AutonomyGovernor.admit_run` checked `window_count(...) < max_runs_per_day`
and returned an admitted Verdict, but the matching increment
(`governor.record(subscription_id, kind="run", ...)`) only happened in
`AutonomousEventEngine.process` *after* `runtime.run(...)` completed --
call the check "read" and the increment "write". Two events matching the
same subscription that arrive close enough together (two webhook
deliveries, two cron ticks, a retried delivery racing the original) can
both run their "read" before either "write" lands: both see the window
under the ceiling, both are admitted, and the ceiling is exceeded by
however many requests raced it. A per-run ceiling does not bound a caller
who can start more runs concurrently, same as the sequential gap the
existing `test_a_daily_run_ceiling_bounds_an_agent_that_starts_its_own_runs`
covers for serial arrivals -- this test is that one's concurrent twin.

Before the fix: with `max_runs_per_day=1` and five events processed
*concurrently*, `runtime.runs` ends up with more than one entry -- the
ceiling admitted every request that raced it before the first one's count
had landed. After the fix (the check-and-claim happens atomically inside
one lock acquisition, at admission time rather than completion time),
exactly one request is admitted and the rest are refused with
`max_runs_per_day`, regardless of how many arrive at once.
"""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from s16code.events import AutonomousEventEngine, EventEnvelope, EventStore, Subscription


def _event(**changes) -> EventEnvelope:
    body = {"id": "e1", "source": "deployments", "type": "deployment.completed",
            "subject": "checkout/production", "occurred_at": datetime.now(UTC),
            "data": {"previous_error_rate": 0.7, "current_error_rate": 6.4}}
    body.update(changes)
    return EventEnvelope(**body)


def _subscription(**changes) -> Subscription:
    body = {"id": "watch", "instruction": "Assess material error-rate increases.",
            "event_types": ["deployment.completed"], "sources": ["deployments"],
            "tenant_id": "course", "allowed_side_effects": [], "budget": 0.02}
    body.update(changes)
    return Subscription(**body)


class _SlowRuntime:
    """A run that takes just long enough for a second, concurrent request to
    reach its own admission check before this one's count has landed --
    the real-world equivalent of an LLM call plus tool dispatch, not an
    instant no-op."""

    def __init__(self, delay: float = 0.05) -> None:
        self.runs: list[str] = []

    async def run(self, *, prompt: str, **_: object) -> dict[str, object]:
        self.runs.append(prompt)
        await asyncio.sleep(0.05)
        return {"run_id": f"run-{len(self.runs)}", "status": "completed", "spend_usd": 0.001}


def _relevance_llm():
    async def llm(prompt: str, system: str):  # noqa: ARG001
        return {"text": json.dumps({"relevant": True, "reason": "error rate rose",
                                    "goal": "Assess the increase."}),
                "metered_calls": [{"cost_usd": 0.00002}]}
    return llm


async def test_a_daily_run_ceiling_holds_when_events_arrive_concurrently(tmp_path) -> None:
    store = EventStore(tmp_path)
    runtime = _SlowRuntime()
    engine = AutonomousEventEngine(store, runtime, max_concurrent=8)
    store.put_subscription(_subscription(max_runs_per_day=1))
    llm = _relevance_llm()

    # Five distinct events, all matching the same subscription, arriving in
    # the same instant -- five webhook deliveries during one burst, or a
    # tick and a retried tick racing each other.
    await asyncio.gather(*(
        engine.process(_event(id=f"burst-{index}"), llm=llm) for index in range(5)
    ))

    assert len(runtime.runs) == 1, (
        f"max_runs_per_day=1 but {len(runtime.runs)} runs actually started -- "
        "the admission check and its count increment are not atomic, so "
        "concurrent requests all read the window before any of them wrote to it"
    )
    day = datetime.now(UTC).date().isoformat()
    assert store.window_count("watch", day, kind="run") == 1
    refusals = [item for item in store.refusals() if item["control"] == "max_runs_per_day"]
    assert len(refusals) == 4
