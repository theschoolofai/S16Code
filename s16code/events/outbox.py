"""A tiny idempotency outbox for capability side effects.

A record left ``started`` means the process died between dispatching a side
effect and learning whether it landed. :meth:`ActionOutbox.execute` refuses to
guess and parks the run on ``outbox.reconcile``. Nothing emits that event: it is
a question for a person, answered with :meth:`resolve` (or by completing the
handle through ``POST /v1/agent/completions``) once they have checked whether
the effect actually happened. :meth:`uncertain` is how they find them.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from s16code.core.live_graph import Deferred


class ActionOutbox:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def key(run_id: str, node_id: str, capability: str, arguments: dict[str, Any]) -> str:
        body = json.dumps([run_id, node_id, capability, arguments], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def _write(self, key: str, value: dict[str, Any]) -> None:
        path = self._path(key)
        fd, temporary = tempfile.mkstemp(prefix=f".{key}-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, sort_keys=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _encode(value: dict[str, Any] | Deferred) -> dict[str, Any]:
        if isinstance(value, Deferred):
            return {"kind": "deferred", "handle": value.handle,
                    "event_type": value.event_type, "metadata": value.metadata}
        return {"kind": "result", "value": value}

    @staticmethod
    def _decode(value: dict[str, Any]) -> dict[str, Any] | Deferred:
        if value["kind"] == "deferred":
            return Deferred(value["handle"], value["event_type"], value.get("metadata", {}))
        return value["value"]

    def uncertain(self) -> list[dict[str, Any]]:
        """Every effect whose outcome a crash left unknown, oldest first.

        A run parked on ``outbox.reconcile`` waits for an event this package
        never sends. That is deliberate -- repeating an effect that may already
        have landed is worse than waiting -- but it only works if the operator
        can see what is waiting and how long it has been.
        """
        with self._lock:
            stuck: list[dict[str, Any]] = []
            for path in sorted(self.root.glob("*.json")):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if record.get("status") != "started":
                    continue
                stuck.append({"key": path.stem, "started_at": record.get("started_at", ""),
                              "attempt": record.get("attempt", 1), "event_type": "outbox.reconcile"})
            return sorted(stuck, key=lambda item: item["started_at"])

    async def resolve(self, key: str, receipt: dict[str, Any]) -> None:
        """Record what a person found out about an uncertain effect.

        Closing the loop belongs here rather than in the caller: the receipt has
        to be stored in the same shape ``execute`` returns, or the next call
        re-parks on a question that has already been answered.
        """
        with self._lock:
            self._write(key, {"status": "completed", "receipt": self._encode(receipt),
                              "resolved_by": "operator",
                              "resolved_at": datetime.now(UTC).isoformat()})

    async def execute(self, key: str,
                      operation: Callable[[], Awaitable[dict[str, Any] | Deferred]]) -> dict[str, Any] | Deferred:
        """Reuse a receipt; never blindly repeat an operation left in-flight by a crash."""
        with self._lock:
            path = self._path(key)
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                if record["status"] == "completed":
                    return self._decode(record["receipt"])
                if record["status"] == "started":
                    return Deferred(key, "outbox.reconcile",
                                    {"uncertain": True, "started_at": record.get("started_at", "")})
                if record["status"] == "failed":
                    raise RuntimeError(record["error"])
            # Timestamped so an operator can tell an effect that is in flight
            # from one abandoned by a crash. Without it every parked record looks
            # equally fresh and equally hopeless.
            self._write(key, {"status": "started", "started_at": datetime.now(UTC).isoformat()})
        try:
            result = await operation()
        except Exception as error:
            with self._lock:
                self._write(key, {"status": "failed", "error": f"{type(error).__name__}: {error}"})
            raise
        with self._lock:
            self._write(key, {"status": "completed", "receipt": self._encode(result)})
        return result
