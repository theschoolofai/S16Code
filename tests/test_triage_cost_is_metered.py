"""The triage ceiling must be fed the cost the gateway actually charged.

`daily_triage_budget` exists because the control that stops a denial-of-wallet
attack must not itself be the unmetered surface the attack aims at (§10). The
governor implements it correctly: `admit_triage` compares
`window_spend(..., kind="triage")` against the ceiling.

But the production triage path could never reach it. The event route passes
`gateway.complete` as its llm; the engine prices the reply with `_spend_of`,
which reads `reply["metered_calls"]`. `GatewayClient.chat()` dropped the `cost`
object that `/v1/chat` returns, and `complete()` narrowed the result further, so
every triage decision recorded $0.00 forever and the ceiling never bound.

`test_autonomy_governor.py::test_the_triage_ceiling_stops_the_gate_itself` passed
throughout, because its stub llm fabricates `metered_calls` with a cost. The stub
was more capable than the real client — §12's "green check on a control that
never ran".

Observed live on 2026-08-10 before the fix: a subscription with
`daily_triage_budget = 0.000001` admitted 4 of 4 triage calls, each recorded at
$0.00000000, while the gateway billed gemini-3.1-flash-lite for every one.
"""

from __future__ import annotations

import httpx
import pytest

from s16code.events.engine import _spend_of
from s16code.gateway import GatewayClient

# Shape of a real glc_v5 /v1/chat response, trimmed to what matters here.
GATEWAY_BODY = {
    "text": "ok",
    "provider": "gemini",
    "model": "gemini-3.1-flash-lite",
    "input_tokens": 620,
    "output_tokens": 48,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "latency_ms": 412,
    "stop_reason": "stop",
    "cost": {
        "input_usd": 0.000155,
        "output_usd": 0.000072,
        "cache_read_usd": 0.0,
        "cache_write_usd": 0.0,
        "total_usd": 0.000227,
        "price_source": "pattern:gemini-*-flash-lite*",
        "model": "gemini-3.1-flash-lite",
        "provider": "gemini",
    },
}

EXPECTED_USD = 0.000227


def _client() -> GatewayClient:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=GATEWAY_BODY))
    return GatewayClient("http://gateway.invalid",
                         client=httpx.AsyncClient(transport=transport, timeout=5))


async def test_chat_surfaces_the_cost_the_gateway_reported() -> None:
    """chat() must not discard the priced cost of the call it just made."""
    result = await _client().chat(prompt="p", system="s")
    assert result.get("cost", {}).get("total_usd") == pytest.approx(EXPECTED_USD), (
        "GatewayClient.chat dropped the gateway's cost object, so no caller "
        "downstream can price the call"
    )


async def test_complete_is_priceable_by_spend_of() -> None:
    """The triage path prices its reply with _spend_of; that must not be 0."""
    reply = await _client().complete("p", "s")
    assert _spend_of(reply) == pytest.approx(EXPECTED_USD), (
        f"_spend_of saw ${_spend_of(reply):.8f} for a call the gateway charged "
        f"${EXPECTED_USD:.8f} for; daily_triage_budget can never be reached"
    )


async def test_a_triage_call_is_never_recorded_as_free() -> None:
    """The specific failure: a real, non-zero call priced at exactly zero."""
    reply = await _client().complete("p", "s")
    assert _spend_of(reply) > 0.0, (
        "every triage decision recorded $0.00, so the ceiling on being awake "
        "accrued nothing no matter how many events arrived"
    )
