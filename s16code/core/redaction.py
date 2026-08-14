"""Keep host filesystem paths out of text the agent stores or shows a model.

Capability results are the obvious way a path reaches the model, and those are
fixed at the producer. Exceptions are the way it happens by accident: an
`OSError` carries the resolved filename it failed on, and a failed node's error
string becomes planner-visible, is written to the durable graph journal, and is
handed to the answer worker as evidence. There is a test asserting exactly that
a failed file read reaches the final answer, so this is a live route rather than
a theoretical one.

Deliberately duplicated rather than shared with the gateway's
`glc.security.redaction`. The two live in separate processes with separate
dependency trees, and each defends its own output: the gateway cannot scrub
what the agent writes into its own journal, and the agent cannot scrub an
approval message the gateway composes. A shared library would couple two
services to protect against a class of bug that is cheapest to stop locally.

Stdlib only, so it stays importable from anywhere inside `s16code.core`.
"""

from __future__ import annotations

import re

_HTTP = re.compile(r"https?://\S+")

_TERMINATOR = r"[^\s`'\")\]}>]"

_PATTERNS = (
    re.compile(rf"(?i)file:/{{0,3}}[A-Za-z]:[\\/]{_TERMINATOR}*"),
    re.compile(rf"(?i)file://{_TERMINATOR}+"),
    re.compile(rf"\\\\[A-Za-z0-9._-]+\\{_TERMINATOR}*"),
    # One letter, then a colon, then a separator. The word boundary keeps this
    # off "sandbox://" and "mxc://", whose scheme also ends in letter-colon.
    re.compile(rf"(?i)\b[A-Za-z]:[\\/]{_TERMINATOR}*"),
    re.compile(rf"(?<![\w.])/(?:home|Users)/{_TERMINATOR}+"),
)

_SEPARATORS = re.compile(r"[\\/]")


def _label(match: re.Match[str]) -> str:
    name = _SEPARATORS.split(match.group(0))[-1].strip()
    return f"[local file: {name}]" if name else "[local file]"


def scrub_host_paths(text: str) -> str:
    """Reduce any host path in `text` to its basename, leaving prose alone."""
    if not text:
        return text
    held: list[str] = []

    def stash(match: re.Match[str]) -> str:
        held.append(match.group(0))
        return f"\x00{len(held) - 1}\x00"

    out = _HTTP.sub(stash, text)
    for pattern in _PATTERNS:
        out = pattern.sub(_label, out)
    for index, original in enumerate(held):
        out = out.replace(f"\x00{index}\x00", original)
    return out
