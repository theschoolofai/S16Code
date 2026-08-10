"""A repair prompt must say what a valid patch looks like, not only what was wrong.

`_parse` accepts exactly {"add", "cancel", "finish", "reason"} and rejects anything
else with `unsupported patch fields: [...]`. When the planner's correct move is
"just answer", models emit a direct reply envelope instead of adding the terminal
capability:

    {"response": "..."}                                  (channel message)
    {"response_to_user": {...}}                          (gemini-3.1-flash-lite)
    {"response": "...", "response_mode": "text"}         (gpt-4.1)

The repair prompt passed the validation error, the rejected output and the original
context - but never the accepted field names, and never the terminal capability that
"answer directly" actually requires. So the model had to infer the schema from an
error message, failed identically on every retry, and the run was finalised `failed`
with an empty answer while every attempt was billed.

Measured on this machine, 10 Aug 2026, before the fix:
  - a parked run that received a valid human approval: 7 of 7 attempts failed,
    across gemini-3.1-flash-lite and gpt-4.1, on two providers. The human's
    decision was consumed, then discarded.
  - a plain channel message needing only a reply: planner's FIRST patch rejected,
    graph got zero nodes, run finished `failed`.
"""

from __future__ import annotations

import json

from s16code.capabilities import default_registry
from s16code.planner import GeneralAgentPlanner


def _planner() -> GeneralAgentPlanner:
    return GeneralAgentPlanner(
        llm=None,  # never called: we inspect prompt construction only
        registry=default_registry(),
        goal="reply to the message",
        respond_as="text",
    )


def _repair_text() -> str:
    planner = _planner()
    original = json.dumps({"goal": "reply to the message", "graph": {"nodes": {}}})
    return planner._repair_prompt(
        original,
        json.dumps({"response": "Hello, I can help with that."}),
        "unsupported patch fields: ['response']",
    )


def test_repair_prompt_names_every_accepted_patch_field() -> None:
    text = _repair_text()
    for field in ("add", "cancel", "finish", "reason"):
        assert field in text, (
            f"the repair prompt never mentions the accepted field {field!r}; the model "
            "is asked to fix a schema it has not been shown"
        )


def test_repair_prompt_names_the_terminal_capability() -> None:
    """"Just answer" is only expressible by adding the terminal capability."""
    text = _repair_text()
    terminal = default_registry().terminal_skills("text")
    assert terminal, "registry advertises no terminal text capability"
    assert any(name in text for name in terminal), (
        f"the repair prompt does not name any of {sorted(terminal)}, so a model that "
        "wanted to answer directly has no way to learn how"
    )


def test_repair_prompt_still_reports_what_was_rejected() -> None:
    """The fix must add guidance without dropping the diagnosis."""
    text = _repair_text()
    assert "unsupported patch fields" in text
    assert "response" in text
