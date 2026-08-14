"""Spend must actually be recorded, or the window ceilings cannot enforce.

`daily_budget` and `daily_triage_budget` are enforced by comparing recorded
spend against the ceiling. If spend is always recorded as zero the comparison
is never true, the ceilings never refuse anything, and the morning report shows
$0.00 no matter how much was spent. These tests pin the recording, not the
prose.
"""

from __future__ import annotations

from s16code.events.engine import _spend_of


class TestSpendExtraction:
    """The economics controller names its cost field `cost` (economics.budget.
    Charge). Reading only `cost_usd`/`usd` silently returned zero for every
    real call.
    """

    def test_reads_the_controller_field_name(self) -> None:
        reply = {"metered_calls": [{"cost": 0.004, "provider": "gemini"}]}
        assert _spend_of(reply) == 0.004

    def test_sums_every_call_in_a_node(self) -> None:
        reply = {"metered_calls": [{"cost": 0.001}, {"cost": 0.002}, {"cost": 0.003}]}
        assert abs(_spend_of(reply) - 0.006) < 1e-9

    def test_still_reads_the_alternative_spellings(self) -> None:
        assert _spend_of({"metered_calls": [{"cost_usd": 0.5}]}) == 0.5
        assert _spend_of({"metered_calls": [{"usd": 0.25}]}) == 0.25

    def test_falls_back_to_a_top_level_price(self) -> None:
        # The gateway client returns one priced call rather than a list.
        assert _spend_of({"cost_usd": 0.007}) == 0.007
        assert _spend_of({"cost": {"total_usd": 0.008}}) == 0.008

    def test_absent_or_malformed_prices_are_zero_not_an_error(self) -> None:
        assert _spend_of({}) == 0.0
        assert _spend_of({"metered_calls": []}) == 0.0
        assert _spend_of({"metered_calls": [{"cost": None}, "not-a-dict"]}) == 0.0
        assert _spend_of({"metered_calls": [{"cost": "free"}]}) == 0.0



class TestTheGatewayPriceSurvivesTheClient:
    """`_spend_of` can only read a price that reached it.

    The gateway prices every call and returns `cost.total_usd`; it is the only
    component that can, since it owns the routing and the price table. But
    `chat()` rebuilds the response field by field, and a field it does not name
    is dropped. `cost` was not named, so `complete()` read it back as absent and
    reported zero for a call that really cost money. Every unit test above
    passed throughout, because each was handed a reply that already had a price
    in it. This pins the seam where the price was actually lost.
    """

    @staticmethod
    def _client_returning(body):
        from s16code.gateway import GatewayClient

        class Response:
            status_code = 200

            @staticmethod
            def json():
                return body

        class Transport:
            @staticmethod
            async def post(url, json=None):  # noqa: A002 - httpx's parameter name
                return Response()

        client = GatewayClient.__new__(GatewayClient)
        client.base_url = "http://gateway.invalid"
        client._client = Transport()
        return client

    PRICED = {"text": "ok", "provider": "gemini", "model": "gemini-3.6-flash",
              "input_tokens": 3, "output_tokens": 1,
              "cost": {"total_usd": 1.5e-06, "price_source": "pattern:gemini-*-flash*"}}

    async def test_chat_carries_the_price_through(self) -> None:
        result = await self._client_returning(self.PRICED).chat(prompt="p", system="s")
        assert result["cost"]["total_usd"] == 1.5e-06, \
            "dropping cost here makes every downstream budget unenforceable"

    async def test_complete_turns_it_into_spend_the_governor_can_read(self) -> None:
        reply = await self._client_returning(self.PRICED).complete("p", "s")
        assert reply["cost_usd"] == 1.5e-06
        assert _spend_of(reply) == 1.5e-06, "this is the number admit_triage compares"

    async def test_an_unpriced_call_is_zero_rather_than_an_error(self) -> None:
        body = {k: v for k, v in self.PRICED.items() if k != "cost"}
        reply = await self._client_returning(body).complete("p", "s")
        assert reply["cost_usd"] == 0.0
        assert reply["metered_calls"] == []



class TestTriageCeilingCanFire:
    """The point of the accounting: a ceiling that can refuse."""

    def test_a_recorded_cost_eventually_exhausts_the_triage_budget(self) -> None:
        from datetime import UTC, datetime

        from s16code.events.governor import AutonomyGovernor
        from s16code.events.models import Subscription

        class Store:
            def __init__(self) -> None:
                self.spend: dict[tuple[str, str, str], float] = {}

            def window_spend(self, sid: str, day: str, *, kind: str) -> float:
                return self.spend.get((sid, day, kind), 0.0)

            def window_record(self, sid: str, day: str, *, kind: str, usd: float = 0.0) -> None:
                self.spend[(sid, day, kind)] = self.spend.get((sid, day, kind), 0.0) + usd

            def window_count(self, sid: str, day: str, *, kind: str) -> int:
                return 0

        store = Store()
        governor = AutonomyGovernor(store)
        subscription = Subscription(id="s", instruction="watch", tenant_id="t",
                                    daily_triage_budget=0.01)
        now = datetime.now(UTC)

        assert governor.admit_triage(subscription, now=now).admitted is True

        # Spend past the ceiling, exactly as the engine does after a gate call.
        governor.record("s", kind="triage", usd=0.02, now=now)

        verdict = governor.admit_triage(subscription, now=now)
        assert verdict.admitted is False, "a spent-out triage budget must refuse"
        assert verdict.control == "daily_triage_budget"
