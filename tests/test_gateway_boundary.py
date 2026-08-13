from __future__ import annotations

import json

import httpx
import pytest

from s16code.gateway import GatewayClient


@pytest.mark.asyncio
async def test_s16code_calls_gateway_without_owning_provider_keys(monkeypatch):
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"text": "done", "provider": "gemini_2", "model": "gemini"})

    monkeypatch.setenv("S16_GATEWAY_PROVIDER", "gemini")
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        result = await GatewayClient("http://127.0.0.1:8111", client=http).complete(
            "Investigate the papers", "Use evidence", session="proof-run"
        )

    assert captured["url"] == "http://127.0.0.1:8111/v1/chat"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["provider"] == "gemini"
    assert payload["agent"] == "s16_agent"
    assert payload["session"] == "proof-run"
    assert not any("key" in field.lower() or "secret" in field.lower() for field in payload)
    assert result["provider"] == "gemini_2"


@pytest.mark.asyncio
async def test_gateway_retries_transient_pool_cooldown(monkeypatch):
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503, json={"detail": "all providers on cooldown"})
        return httpx.Response(200, json={"text": "recovered", "provider": "gemini_3", "model": "gemini"})

    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setenv("S16_GATEWAY_ATTEMPTS", "3")
    monkeypatch.setattr("s16code.gateway.asyncio.sleep", no_wait)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await GatewayClient("http://gateway", client=http).complete("goal", "system")
    assert calls == 3
    assert result["text"] == "recovered"


@pytest.mark.asyncio
async def test_gateway_can_fall_back_without_exposing_credentials(monkeypatch):
    providers = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        providers.append(payload["provider"])
        if payload["provider"] == "gemini":
            return httpx.Response(503, json={"detail": "pool exhausted"})
        assert "model" not in payload
        return httpx.Response(200, json={"text": "fallback", "provider": "openrouter", "model": "configured"})

    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setenv("S16_GATEWAY_PROVIDER", "gemini")
    monkeypatch.setenv("S16_GATEWAY_FALLBACK_PROVIDERS", "openrouter")
    monkeypatch.setenv("S16_GATEWAY_ATTEMPTS", "1")
    monkeypatch.setattr("s16code.gateway.asyncio.sleep", no_wait)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await GatewayClient("http://gateway", client=http).chat(
            prompt="goal", system="system", request={"model": "gemini-only"}
        )
    assert providers == ["gemini", "openrouter"]
    assert result["text"] == "fallback"


@pytest.mark.asyncio
async def test_operator_provider_pin_beats_a_tier_that_names_another_provider(monkeypatch):
    """Budgeted runs send the ladder's request body, including `provider: github`
    on the frontier rung. That field used to overwrite S16_GATEWAY_PROVIDER, so a
    Gemini-only gateway failed every wait/resume and unattended run while the
    same prompt without a budget still worked.
    """
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"text": "ok", "provider": "gemini_1", "model": "gemini"})

    monkeypatch.setenv("S16_GATEWAY_PROVIDER", "gemini")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await GatewayClient("http://gateway", client=http).chat(
            prompt="park this run",
            system="ask first",
            request={"provider": "github", "model": "openai/gpt-4.1", "max_tokens": 4096},
        )
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["provider"] == "gemini"
    assert payload.get("model") != "openai/gpt-4.1"
    assert payload["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_a_tier_on_the_pinned_provider_still_chooses_its_model(monkeypatch):
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"text": "ok", "provider": "gemini_1", "model": "gemini-3.1-flash-lite"})

    monkeypatch.setenv("S16_GATEWAY_PROVIDER", "gemini")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await GatewayClient("http://gateway", client=http).chat(
            prompt="goal",
            system="system",
            request={"provider": "gemini", "model": "gemini-3.1-flash-lite", "max_tokens": 1024},
        )
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["provider"] == "gemini"
    assert payload["model"] == "gemini-3.1-flash-lite"


def test_s16code_does_not_reimplement_gateway_routes(app_client):
    assert app_client.post("/v1/chat", json={}).status_code == 404
    assert app_client.get("/v1/providers").status_code == 404
