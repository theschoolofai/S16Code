"""A living watcher must say so, even on a night when nothing happens.

`EventStore.beat()` has exactly one caller: `AutonomousEventEngine.process`. So the
heartbeat advances only when an event arrives. There is no beat at startup and none
on a timer.

Section 9 is entirely about the ambiguity this creates:

    nothing happened, because nothing needed to
                        or
    it died at 11pm and has not noticed anything since

and its remedy: "The agent taps a counter every time it *wakes up* and every time
it handles an event. Answers 200 while it is beating. Answers 503 once the last
beat is too old." The wake-up beat is missing, so the counter only ever means "an
event arrived", never "I am alive".

## What breaks

A healthy agent that correctly receives nothing for `stale_after_seconds` (900)
reports itself dead:

    $ curl http://127.0.0.1:8113/v1/agent/liveness      # process up, serving
    {"alive": false,
     "reason": "no heartbeat for 16235s; the watcher may have died",
     "beats": 15,
     "last_beat": "2026-08-10T13:54:30Z"}               # the last EVENT

Observed live on 2026-08-10, minutes after a restart, while the process answered
every other request normally.

Three consequences:

1. The alarm fires on correct behaviour. Section 8 says "a quiet night is often the
   right answer"; here a quiet night returns 503. Anyone paging on that endpoint is
   woken by every idle night, learns to ignore it, and then misses a real death.
2. The morning report's header - the one line that tells you whether the watcher
   was up all night - reads "NOT ALIVE ... the watcher may have died" for a process
   that never stopped.
3. A freshly started agent is dead on arrival: with no events yet, liveness reports
   "no heartbeat has ever been recorded" until something happens to it.

The direction of the signal is what Section 9 cares about: "a dead process cannot
send you an error, but it also cannot fake a pulse." A pulse that only exists when
work arrives cannot distinguish idleness from death, which is the one thing it is
for.
"""

from __future__ import annotations

import os


def test_a_freshly_started_process_reports_itself_alive(app_client) -> None:
    """No events have been ingested. The process is up, so it must say so."""
    response = app_client.get("/v1/agent/liveness")
    body = response.json()

    assert body["alive"] is True, (
        f"a running process with no events reported alive={body['alive']} "
        f"({body.get('reason')!r}); an idle watcher is indistinguishable from a dead one"
    )
    assert response.status_code == 200, (
        f"liveness answered {response.status_code} for a healthy process, so an "
        "uptime monitor pages on a quiet night"
    )


def test_the_pulse_exists_without_any_event_arriving(app_client) -> None:
    """The beat must mean "I am alive", not "an event arrived"."""
    body = app_client.get("/v1/agent/liveness").json()

    assert body.get("beats", 0) >= 1, (
        "no beat was ever recorded, so the counter only advances when work arrives"
    )
    assert body.get("last_beat") is not None
    assert body["reason"] == "beating"


def test_a_periodic_beat_is_scheduled(app_client) -> None:
    """Startup alone is not enough: the pulse has to keep going while idle."""
    from s16code.main import app

    task = getattr(app.state, "heartbeat_task", None)
    assert task is not None, (
        "no background heartbeat is scheduled, so a process that starts and then "
        "sits idle goes stale after stale_after_seconds and reports itself dead"
    )
    assert not task.done(), "the heartbeat task is not running"


def test_the_beat_interval_stays_well_inside_the_staleness_window(app_client) -> None:
    """A beat slower than the window would make the alarm fire on a live agent."""
    from s16code.events.report import STALE_AFTER_SECONDS

    interval = float(os.getenv("S16_HEARTBEAT_SECONDS", "60"))
    assert 0 < interval < STALE_AFTER_SECONDS / 2, (
        f"beat interval {interval}s against a {STALE_AFTER_SECONDS}s staleness "
        "window leaves no margin for a slow write or a paused container"
    )


class _FakeStore:
    def __init__(self, raw: dict) -> None:
        self._raw = raw

    def liveness(self) -> dict:
        return dict(self._raw)


def test_liveness_still_reports_death_when_the_beat_really_stops() -> None:
    """The fix must not make the endpoint answer 200 unconditionally."""
    from s16code.events.report import liveness_status

    stale = liveness_status(_FakeStore({"last_beat": "2020-01-01T00:00:00+00:00", "beats": 3}),
                            stale_after=900)
    assert stale["alive"] is False
    assert "may have died" in stale["reason"]


def test_a_store_that_never_beat_is_not_reported_alive() -> None:
    """Absence of any pulse remains the alarm, as Section 9 requires."""
    from s16code.events.report import liveness_status

    never = liveness_status(_FakeStore({}), stale_after=900)
    assert never["alive"] is False
    assert never["reason"] == "no heartbeat has ever been recorded"
