"""Test: source_events_per_minute=0 must block all events, not disable the limit.

Bug: In AutonomyGovernor.admit_event(), the check:
    if self.source_events_per_minute > 0:
skips the rate limit entirely when the value is 0. An operator setting
source_events_per_minute=0 intends "block all events from this source",
but actually gets "allow unlimited events" — the control does not hold.

Fix: Change the guard to >= 0 and handle the zero case explicitly.
"""

from datetime import UTC, datetime

from s16code.events import AutonomyGovernor, EventEnvelope, EventStore


def _event(**changes) -> EventEnvelope:
    body = {"id": "e1", "source": "webhooks", "type": "webhook.received",
            "occurred_at": datetime.now(UTC), "data": {}}
    body.update(changes)
    return EventEnvelope(**body)


def test_zero_source_rate_limit_blocks_all_events(tmp_path) -> None:
    """source_events_per_minute=0 must refuse every event, not skip the check."""
    store = EventStore(tmp_path / "state")
    governor = AutonomyGovernor(store, source_events_per_minute=0)

    event = _event()
    verdict = governor.admit_event(event)

    # Before the fix: verdict.admitted is True (rate limit skipped)
    # After the fix:  verdict.admitted is False (all events blocked)
    assert verdict.admitted is False, (
        f"Expected event to be refused when source_events_per_minute=0, "
        f"but it was admitted. The rate limit was bypassed!"
    )
    assert verdict.control == "source_rate_limit"
    assert "0" in verdict.reason


def test_positive_source_rate_limit_still_works(tmp_path) -> None:
    """A positive limit must still admit events below the threshold."""
    store = EventStore(tmp_path / "state")
    governor = AutonomyGovernor(store, source_events_per_minute=5)

    event = _event()
    verdict = governor.admit_event(event)

    assert verdict.admitted is True


def test_source_rate_limit_refuses_after_threshold(tmp_path) -> None:
    """Events beyond the limit must be refused."""
    store = EventStore(tmp_path / "state")
    governor = AutonomyGovernor(store, source_events_per_minute=2)

    # Ingest 2 events to fill the window
    store.ingest(_event(id="e1"))
    store.ingest(_event(id="e2"))

    # Third event should be refused
    verdict = governor.admit_event(_event(id="e3"))
    assert verdict.admitted is False
    assert verdict.control == "source_rate_limit"
