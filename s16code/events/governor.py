"""Bounds on unattended operation, enforced outside the agent.

Session 15 gave every *run* a hard ceiling and refused before the provider was
contacted. That ceiling stops being a ceiling the moment the agent starts its own
runs, because nothing stops it starting more of them. An agent woken by a stream
of events needs a bound over a **window**, not over one request.

Three separate things get bounded here, and they fail for different reasons:

* **Being awake.** The relevance gate is a model call on every matching event,
  forever, whether or not any work follows. It is usually the larger bill and it
  is the one nobody budgets, because it does not look like work.
* **Doing.** Runs per window and spend per window, per subscription.
* **Being used as a weapon.** Anyone who can create an event can schedule this
  agent. A per-source arrival limit turns "flood the webhook" from a spending
  attack into a recorded refusal.

And one thing gets refused outright: an event this agent itself caused. An agent
that acts in a system it also watches will see its own action arrive as a new
event. Without an actor check that is a self-sustaining loop, and a spend ceiling
only ends it after the money is gone.

Every refusal is *recorded*. Session 15 learned this the hard way: a control that
prevents work leaves no trace of the work it prevented, so an operator reading the
dashboard sees a quiet night and an untouched budget either way.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from .models import EventEnvelope, Subscription


@dataclass(frozen=True)
class Verdict:
    """The outcome of one admission check, always explainable."""

    admitted: bool
    reason: str = ""
    control: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {"admitted": self.admitted, "reason": self.reason, "control": self.control,
                **({"detail": self.detail} if self.detail else {})}


ADMITTED = Verdict(True)


def _day(moment: datetime) -> str:
    return moment.astimezone(UTC).date().isoformat()


class AutonomyGovernor:
    """Windowed admission for an agent nobody is watching.

    State lives in the event store so the bounds survive a restart. An agent
    whose daily ceiling resets every time the process bounces has no ceiling.
    """

    def __init__(self, store: Any, *, self_actors: tuple[str, ...] | None = None,
                 source_events_per_minute: int | None = None) -> None:
        self.store = store
        configured = os.getenv("S16_SELF_ACTORS", "")
        self.self_actors = tuple(self_actors if self_actors is not None else
                                 [item.strip() for item in configured.split(",") if item.strip()]
                                 or ["s16code", "s16-autonomous"])
        limit = os.getenv("S16_SOURCE_EVENTS_PER_MINUTE", "")
        self.source_events_per_minute = (source_events_per_minute if source_events_per_minute is not None
                                         else int(limit) if limit.strip().isdigit() else 120)

    # ------------------------------------------------------------------ intake

    def admit_event(self, event: EventEnvelope, *, now: datetime | None = None) -> Verdict:
        """Refuse before the event is even matched against subscriptions."""
        moment = now or datetime.now(UTC)

        actor = (event.actor or "").strip()
        if actor and actor in self.self_actors:
            return Verdict(False, f"event was caused by this agent ({actor}); refusing to answer itself",
                           "self_trigger", {"actor": actor})

        if self.source_events_per_minute > 0:
            recent = self.store.count_recent_events(event.source, since=moment - timedelta(minutes=1))
            if recent >= self.source_events_per_minute:
                return Verdict(
                    False,
                    f"source {event.source!r} exceeded {self.source_events_per_minute} events/minute",
                    "source_rate_limit",
                    {"source": event.source, "observed": recent,
                     "limit": self.source_events_per_minute},
                )
        return ADMITTED

    # ------------------------------------------------------------- being awake

    def admit_triage(self, subscription: Subscription, *, now: datetime | None = None) -> Verdict:
        """May we spend a model call deciding whether this event matters?"""
        if subscription.daily_triage_budget is None:
            return ADMITTED
        moment = now or datetime.now(UTC)
        spent = self.store.window_spend(subscription.id, _day(moment), kind="triage")
        if spent >= subscription.daily_triage_budget:
            return Verdict(
                False,
                f"daily triage budget exhausted: ${spent:.8f} of ${subscription.daily_triage_budget:.8f}",
                "daily_triage_budget",
                {"spent_usd": spent, "limit_usd": subscription.daily_triage_budget},
            )
        return ADMITTED

    # ------------------------------------------------------------------- doing

    def admit_run(self, subscription: Subscription, *, now: datetime | None = None) -> Verdict:
        """May a relevant decision actually become a run?

        The slot is reserved atomically at admit time, so overlapping events
        cannot all observe an empty ledger and all start runs.
        """
        moment = now or datetime.now(UTC)
        day = _day(moment)
        ok, control, detail = self.store.window_try_admit_run(
            subscription.id, day,
            max_runs=subscription.max_runs_per_day,
            budget=subscription.daily_budget,
            per_run_budget=subscription.budget,
        )
        if not ok:
            if control == "max_runs_per_day":
                return Verdict(False,
                               f"daily run ceiling reached: {detail['runs']} of {detail['limit']}",
                               control, detail)
            return Verdict(False,
                           f"daily budget exhausted: ${detail['spent_usd']:.8f} of ${detail['limit_usd']:.8f}",
                           control, detail)
        return Verdict(True, "", "", detail)

    def settle_run(self, subscription_id: str, *, reserved: float, actual: float = 0.0,
                   now: datetime | None = None) -> None:
        """Replace an admitted run's reservation with its metered spend."""
        self.store.window_settle_run(subscription_id, _day(now or datetime.now(UTC)),
                                     reserved=reserved, actual=actual)

    # ----------------------------------------------------------------- ledger

    def record(self, subscription_id: str, *, kind: str, usd: float = 0.0,
               now: datetime | None = None) -> None:
        self.store.window_record(subscription_id, _day(now or datetime.now(UTC)),
                                 kind=kind, usd=usd)

    def snapshot(self, subscription: Subscription, *, now: datetime | None = None) -> dict[str, Any]:
        """What an operator needs to see: how much of each bound is left."""
        day = _day(now or datetime.now(UTC))
        return {
            "subscription_id": subscription.id,
            "window": day,
            "runs": self.store.window_count(subscription.id, day, kind="run"),
            "max_runs_per_day": subscription.max_runs_per_day,
            "run_spend_usd": self.store.window_spend(subscription.id, day, kind="run"),
            "daily_budget_usd": subscription.daily_budget,
            "triage_decisions": self.store.window_count(subscription.id, day, kind="triage"),
            "triage_spend_usd": self.store.window_spend(subscription.id, day, kind="triage"),
            "daily_triage_budget_usd": subscription.daily_triage_budget,
        }
