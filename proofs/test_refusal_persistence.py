"""
Test that refusals are properly recorded.
"""

from s16code.events.governor import AutonomyGovernor
from s16code.events.models import EventEnvelope

def test_refusal_persistence():
    """Test that a refusal is recorded in a durable store."""
    governor = AutonomyGovernor()
    
    # Create an event with a self-actor (should be refused)
    event = EventEnvelope(
        source="test_source",
        actor="s16code",  # This matches self_actors
        id="test_1"
    )
    
    # The governor should refuse the event
    verdict = governor.admit_event(event)
    assert verdict.admitted is False, "Event should be refused"
    
    # The refusal should be queryable
    refusals = governor.get_refusal_history()
    assert len(refusals) > 0, "No refusals recorded"
    
    print("✅ Refusal persistence test passed.")
