"""The agent's own answers must not come back to it as evidence.

Every answer is written to memory as an episode so a conversation has
continuity. Recall then returned those episodes as ambient evidence, which
closes a loop with no damping: an answer is a near-perfect lexical match for
the question that produced it, so it becomes the top hit the next time that
question is asked, and the agent cites itself.

Observed before this fix: asked three times about a conference invitation held
in a sandbox file, recall returned the agent's own three previous "there is no
information about it" replies as the top three hits. Every retry added another
denial, so rephrasing made it worse rather than better, and the agent grew more
confident about something it had never checked.

What must keep working is as important as what must stop. The user's own
messages are evidence, and so are facts written by `remember_explicit_fact` --
cross-channel recall depends on them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from s16code.core.memory import (
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    MemoryStore,
    Principal,
    SourceRef,
)
from s16code.core.memory.embeddings import DeterministicEmbedder

SCOPE = MemoryScope("tenant", "project", "user")


@pytest.fixture
def store(tmp_path: Path):
    memory = MemoryStore(tmp_path / "memory.sqlite", embedder=DeterministicEmbedder(128))
    try:
        yield memory
    finally:
        close = getattr(memory, "close", None)
        if close:
            close()


def _answer(store: MemoryStore, text: str, run_id: str = "run-abc123") -> None:
    """Exactly how runtime.py records an answer."""
    store.write(MemoryRecord(
        MemoryKind.EPISODE, SCOPE, text,
        [SourceRef(f"run://{run_id}/answer", "gateway", excerpt=text)],
        Principal("gateway", "gateway"), metadata={"run_id": run_id}))


def _user_said(store: MemoryStore, text: str) -> None:
    """Exactly how an inbound channel message is recorded."""
    store.write(MemoryRecord(
        MemoryKind.EPISODE, SCOPE, text,
        [SourceRef("channel://telegram/abc", "owner", excerpt=text)],
        Principal("owner", "owner"), metadata={"run_id": "run-inbound"}))


def _fact(store: MemoryStore, text: str) -> None:
    store.write(MemoryRecord(
        MemoryKind.FACT, SCOPE, text,
        [SourceRef("channel://telegram/abc", "owner", excerpt=text)],
        Principal("owner", "owner")))


class TestSelfAuthoredAnswersAreNotEvidence:
    def test_a_denial_does_not_come_back_as_evidence(self, store: MemoryStore) -> None:
        _answer(store, "There is no information regarding a DevSummit invitation.")
        found = store.recall("DevSummit invitation deadline", SCOPE)
        assert [record.text for record in found] == [], \
            "the agent's own denial must not be evidence for the next attempt"

    def test_repeating_the_question_cannot_entrench_it(self, store: MemoryStore) -> None:
        # The failure mode was cumulative: three denials outranked everything.
        for index in range(3):
            _answer(store, "There is no information regarding a DevSummit invitation.",
                    run_id=f"run-{index}")
        assert store.recall("DevSummit invitation", SCOPE) == []

    def test_an_explicit_caller_can_still_read_them(self, store: MemoryStore) -> None:
        # "What did you tell me earlier?" is a real question; it just has to ask.
        _answer(store, "I told you the dentist is on Thursday.")
        found = store.recall("dentist Thursday", SCOPE, include_own_answers=True)
        assert any("dentist" in record.text for record in found)


class TestWhatMustKeepWorking:
    def test_the_user_s_own_words_are_still_evidence(self, store: MemoryStore) -> None:
        _user_said(store, "Remember that I prefer morning flights.")
        found = store.recall("travel preferences morning flights", SCOPE)
        assert any("morning flights" in record.text for record in found), \
            "what the user said is evidence and must survive"

    def test_facts_are_untouched(self, store: MemoryStore) -> None:
        # The cross-channel demo depends on this: remembered on one channel,
        # recalled on another. Facts are written by remember_explicit_fact.
        _fact(store, "Prefers morning flights and will not fly out before 8am.")
        found = store.recall("what do you know about my travel preferences", SCOPE)
        assert any("8am" in record.text for record in found)

    def test_an_answer_does_not_crowd_out_the_fact(self, store: MemoryStore) -> None:
        _fact(store, "Prefers morning flights and will not fly out before 8am.")
        _answer(store, "I have no information about your travel preferences.")
        found = store.recall("my travel preferences", SCOPE)
        texts = [record.text for record in found]
        assert any("8am" in text for text in texts)
        assert not any("no information" in text for text in texts)
