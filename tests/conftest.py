from __future__ import annotations

import pytest

CONTROL_TOKEN = "test-control-token"
COMPLETION_TOKEN = "test-completion-token"


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    monkeypatch.setenv("S16_DATA_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("S16_A2A_GRPC_ENABLED", "0")
    monkeypatch.setenv("S16_SANDBOX_ROOT", str(tmp_path / "sandbox"))
    # The control plane fails closed. Tests configure a token exactly as a real
    # deployment must; they do not get a bypass, so the gates stay exercised
    # rather than disabled for convenience.
    monkeypatch.setenv("S16_CONTROL_TOKEN", CONTROL_TOKEN)
    monkeypatch.setenv("S16_COMPLETION_TOKEN", COMPLETION_TOKEN)
    # A developer's .env may point at a live collector; tests must remain
    # deterministic and must not depend on external Jaeger state.
    monkeypatch.delenv("S16_OTEL_EXPORTER_ENDPOINT", raising=False)
    (tmp_path / "sandbox").mkdir()


@pytest.fixture
def app_client():
    from fastapi.testclient import TestClient

    from s16code.main import app

    with TestClient(app, headers={"Authorization": f"Bearer {CONTROL_TOKEN}"}) as client:
        yield client
