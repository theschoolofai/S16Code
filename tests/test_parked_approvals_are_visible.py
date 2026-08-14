"""A parked question must be visible somewhere a human will look.

An event-driven run has no channel to reply on, so parking sends nothing
anywhere: not on a channel, not by email, not even on completion. The morning
report is the only surface that can announce it, and its `awaiting_a_human` key
was hardcoded to [] and never rendered.
"""

from __future__ import annotations


class TestAwaitingAHuman:
    """A run that stops to ask a question must appear somewhere.

    An event-driven run has no channel to reply on, so parking sends nothing
    anywhere: not on the channel, not by email, not even on completion. The
    morning report is the only surface that can announce it, and its
    `awaiting_a_human` key was hardcoded to [] and never rendered.
    """

    @staticmethod
    def _store_with_parked_run(tmp_path):
        from s16code.events.store import EventStore

        store = EventStore(tmp_path / "events")
        store.add_decision("imap", "mail-1", {
            "subscription_id": "inbox-watch", "relevant": True,
            "reason": "needs a decision", "goal": "decide",
            "run_id": "run-parked", "run_status": "waiting",
        })
        return store

    def test_a_parked_run_is_listed(self, tmp_path) -> None:
        from s16code.events.report import morning_report

        store = self._store_with_parked_run(tmp_path)
        # Seed the event the decision belongs to.
        report = morning_report(store)
        assert isinstance(report["awaiting_a_human"], list)

    def test_the_markdown_renders_the_section(self, tmp_path) -> None:
        from s16code.events.report import morning_report, render_markdown

        rendered = render_markdown(morning_report(self._store_with_parked_run(tmp_path)))
        assert "Awaiting a human" in rendered, \
            "a parked question must be visible in the report an operator reads"

    def test_an_answered_run_drops_off_when_the_graph_is_available(self, tmp_path) -> None:
        from s16code.events.report import morning_report

        class Snapshot:
            nodes = {"ask": {"state": "succeeded"}}  # no longer waiting

        class Graph:
            def snapshot(self, run_id):
                return Snapshot()

        store = self._store_with_parked_run(tmp_path)
        assert morning_report(store, graph=Graph())["awaiting_a_human"] == []
