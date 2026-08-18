"""The artifact a human reads after the agent ran without them.

A live SSE stream is for a screen somebody is watching. Nobody was watching. The
deliverable of an unattended night is therefore a written account that can be
read once, in order, and acted on: what arrived, what was ignored and why, what
was done, what it cost, what was refused, and what is still waiting for a person.

Two of those sections are the ones usually missing. **What was ignored** is most
of the night and is the only evidence that the relevance gate is calibrated.
**What was refused** is the work a control prevented, which leaves no other
trace: without it an operator sees an untouched budget and concludes nothing ever
threatened it.

The report also states its own liveness, because a report full of zeroes means
either a quiet night or a dead watcher, and those must not look the same.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from .governor import AutonomyGovernor
from .store import EventStore

STALE_AFTER_SECONDS = 900


def liveness_status(store: EventStore, *, now: datetime | None = None,
                    stale_after: int = STALE_AFTER_SECONDS) -> dict[str, Any]:
    """Alive, or silent? Absence of a beat is the signal, not absence of work."""
    moment = now or datetime.now(UTC)
    raw = store.liveness()
    last = raw.get("last_beat")
    if not last:
        return {"alive": False, "reason": "no heartbeat has ever been recorded",
                "silent_for_seconds": None, "stale_after_seconds": stale_after, **raw}
    silent = (moment - datetime.fromisoformat(last)).total_seconds()
    return {
        "alive": silent <= stale_after,
        "reason": ("beating" if silent <= stale_after
                   else f"no heartbeat for {int(silent)}s; the watcher may have died"),
        "silent_for_seconds": round(silent, 3),
        "stale_after_seconds": stale_after,
        **raw,
    }


def morning_report(store: EventStore, *, since: datetime | None = None,
                   now: datetime | None = None) -> dict[str, Any]:
    moment = now or datetime.now(UTC)
    window_start = since or (moment - timedelta(hours=24))
    governor = AutonomyGovernor(store)

    considered: list[dict[str, Any]] = []
    for record in store.events():
        received = record.get("received_at")
        if received:
            try:
                if datetime.fromisoformat(received) < window_start:
                    continue
            except ValueError:
                pass
        considered.append(record)

    acted: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    awaiting_human: list[dict[str, Any]] = []
    triage_spend = 0.0
    for record in considered:
        event = record["event"]
        for decision in record.get("decisions", []):
            triage_spend += float(decision.get("triage_cost_usd") or 0.0)
            entry = {"event": f"{event['source']}/{event['id']}", "type": event["type"],
                     "subscription": decision.get("subscription_id"),
                     "reason": decision.get("reason", "")}
            if decision.get("refused_by"):
                blocked.append({**entry, "control": decision["refused_by"]})
            elif decision.get("relevant"):
                run_status = decision.get("run_status")
                if run_status == "waiting":
                    awaiting_human.append({**entry, "run_id": decision.get("run_id"),
                                           "status": run_status})
                else:
                    acted.append({**entry, "run_id": decision.get("run_id"),
                                  "status": run_status})
            else:
                ignored.append(entry)

    refusals = store.refusals(since=window_start)
    subscriptions = store.subscriptions()
    budgets = [governor.snapshot(item, now=moment) for item in subscriptions]
    run_spend = sum(item["run_spend_usd"] for item in budgets)

    return {
        "generated_at": moment.isoformat(),
        "window_start": window_start.isoformat(),
        "liveness": liveness_status(store, now=moment),
        "totals": {
            "events_seen": len(considered),
            "acted": len(acted),
            "ignored": len(ignored),
            "blocked_by_a_control": len(blocked) + len(refusals),
            # The two bills the session insists on separating: what it cost to
            # be awake, and what it cost to do something.
            "cost_of_watching_usd": round(triage_spend, 10),
            "cost_of_doing_usd": round(run_spend, 10),
        },
        "acted": acted,
        "ignored": ignored,
        "refused": blocked + [{"event": f"{(item.get('event') or {}).get('source')}/"
                                        f"{(item.get('event') or {}).get('id')}",
                               "control": item.get("control"), "reason": item.get("reason"),
                               "subscription": item.get("subscription_id")}
                              for item in refusals],
        "budgets": budgets,
        "awaiting_a_human": awaiting_human,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """The same report as prose, because an operator reads it, not a parser."""
    totals = report["totals"]
    live = report["liveness"]
    lines = [
        f"# Overnight report — {report['generated_at']}",
        "",
        f"**Watcher:** {'alive' if live['alive'] else 'NOT ALIVE'} ({live['reason']})",
        "",
        f"- events seen: {totals['events_seen']}",
        f"- acted: {totals['acted']}",
        f"- ignored: {totals['ignored']}",
        f"- blocked by a control: {totals['blocked_by_a_control']}",
        f"- cost of watching: ${totals['cost_of_watching_usd']:.8f}",
        f"- cost of doing: ${totals['cost_of_doing_usd']:.8f}",
        "",
    ]
    for title, key in (("Acted", "acted"), ("Awaiting Human", "awaiting_a_human"), ("Ignored", "ignored"), ("Refused", "refused")):
        lines.append(f"## {title} ({len(report[key])})")
        if not report[key]:
            lines.append("_nothing_")
        for item in report[key][:100]:
            run_str = f" [{item['run_id']}]" if item.get("run_id") else ""
            lines.append(f"- `{item.get('event')}` — {item.get('reason') or item.get('control')}{run_str}")
        lines.append("")
    return "\n".join(lines)
