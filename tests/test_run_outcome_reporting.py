"""A run's final outcome must reach the surfaces a human reads.

Two related gaps, both in the direction that misleads:

1. `add_decision` records `run_status` at triage time and nothing updates it, so
   a run that parks and later completes is reported as `waiting` forever. Work
   that succeeded looks stuck.
2. `morning_report`'s `awaiting_a_human` is hardcoded `[]`, though the module
   docstring promises "what is still waiting for a person". A parked approval is
   invisible, so nobody is told a decision is needed.

Together the console can neither say a run needs you, nor that it stopped
needing you.
"""

from __future__ import annotations

from datetime import UTC, datetime

from s16code.events import EventEnvelope, EventStore, morning_report


def _event(store: EventStore, event_id: str = "evt-1") -> None:
    store.ingest(EventEnvelope.model_validate({
        "id": event_id, "source": "gmail", "type": "email.received",
        "occurred_at": datetime.now(UTC).isoformat(),
        "observed_at": datetime.now(UTC).isoformat(),
        "data": {"subject": "lease renewal"},
    }))


def test_completed_run_is_no_longer_reported_as_waiting(tmp_path):
    """Bug 1: the decision keeps its triage-time status forever."""
    store = EventStore(tmp_path / "events")
    _event(store)
    store.add_decision("gmail", "evt-1", {
        "subscription_id": "inbox-triage", "relevant": True,
        "reason": "urgent", "goal": "notify", "run_id": "run-abc", "run_status": "waiting",
    })

    store.update_run_status("run-abc", "completed")

    decision = store.events()[-1]["decisions"][0]
    assert decision["run_status"] == "completed", (
        "a run that parked and later finished is still reported as waiting; "
        "work that succeeded looks stuck"
    )


def test_update_run_status_leaves_other_runs_alone(tmp_path):
    store = EventStore(tmp_path / "events")
    _event(store, "evt-1")
    _event(store, "evt-2")
    store.add_decision("gmail", "evt-1", {"subscription_id": "a", "run_id": "run-1",
                                          "run_status": "waiting"})
    store.add_decision("gmail", "evt-2", {"subscription_id": "b", "run_id": "run-2",
                                          "run_status": "waiting"})

    store.update_run_status("run-1", "completed")

    statuses = {d["run_id"]: d["run_status"]
                for record in store.events() for d in record["decisions"]}
    assert statuses == {"run-1": "completed", "run-2": "waiting"}


def test_update_run_status_is_safe_for_an_unknown_run(tmp_path):
    store = EventStore(tmp_path / "events")
    _event(store)
    store.update_run_status("run-does-not-exist", "completed")  # must not raise


def test_report_lists_a_parked_approval(tmp_path):
    """Bug 2: awaiting_a_human is hardcoded empty."""
    store = EventStore(tmp_path / "events")
    _event(store)
    pending = [{"run_id": "run-abc", "handle": "h1",
                "question": "Do you want to send this to Telegram?"}]

    report = morning_report(store, pending_approvals=lambda: pending)

    assert report["awaiting_a_human"] == pending, (
        "a run parked on a human gate is not reported, so nobody is told a "
        "decision is waiting"
    )


def test_report_without_a_resolver_still_works(tmp_path):
    """A caller with no live graph must still get a report."""
    store = EventStore(tmp_path / "events")
    _event(store)
    assert morning_report(store)["awaiting_a_human"] == []


def test_report_survives_a_failing_resolver(tmp_path):
    """The account of the night must not disappear because a lookup broke."""
    store = EventStore(tmp_path / "events")
    _event(store)

    def broken():
        raise RuntimeError("graph store unavailable")

    report = morning_report(store, pending_approvals=broken)
    assert report["awaiting_a_human"] == []
    assert report["totals"]["events_seen"] == 1


def test_waiting_handles_lists_outstanding_waits_and_clears_them(tmp_path):
    """The handle index is the live set of things a run is blocked on."""
    from s16code.core.live_graph import Deferred, GraphPatch, GraphStore, TaskSpec

    graph = GraphStore(tmp_path / "graphs")
    graph.start("run-1", context={})
    graph.apply_patch("run-1", GraphPatch(
        add=(TaskSpec("gate", "request_approval", {"question": "May I send this?"}),),
        reason="ask first"), trigger_event=1)
    graph.mark_running("run-1", [TaskSpec("gate", "request_approval")])
    graph.record_waiting("run-1", "gate", Deferred("h-1", "approval.received").as_wait())

    assert graph.waiting_handles() == [("h-1", "approval.received", "run-1")]

    graph.complete_waiting("h-1", "approval.received", {"answer": "yes"})
    assert graph.waiting_handles() == [], "a consumed handle is still reported as pending"


def test_report_route_surfaces_a_real_parked_approval(app_client):
    """End to end: the route's resolver reaches the live graph.

    Guards the defensive `except` in `_pending_or_empty` — without this a broken
    lookup would silently return [] and look exactly like a quiet night.
    """
    from s16code.core.live_graph import Deferred, GraphPatch, TaskSpec

    runtime = app_client.app.state.runtime
    runtime.graph.start("run-parked", context={})
    runtime.graph.apply_patch("run-parked", GraphPatch(
        add=(TaskSpec("gate", "request_approval", {"question": "May I send this to Telegram?"}),),
        reason="policy requires asking first"), trigger_event=1)
    runtime.graph.mark_running("run-parked", [TaskSpec("gate", "request_approval")])
    runtime.graph.record_waiting("run-parked", "gate", Deferred(
        "handle-parked", "approval.received",
        {"question": "May I send this to Telegram?", "choices": []}).as_wait())

    parked = app_client.get("/v1/agent/report?hours=24").json()["awaiting_a_human"]

    assert [item["run_id"] for item in parked] == ["run-parked"]
    assert parked[0]["question"] == "May I send this to Telegram?"
    assert parked[0]["skill"] == "request_approval"
