"""Unit tests for the IntentResolver — new gesture vocabulary.

Verifies mapping for:
- thumb_up → BROWSER_BACK
- thumb_down → BROWSER_FORWARD
- swipe_left → BROWSER_BACK
- swipe_right → BROWSER_FORWARD
- tab_close → CLOSE_TAB
- nav_engage / nav_release → bookkeeping only
- nav_active() → scroll + tab-switch intents
"""

from __future__ import annotations

from config.config import IntentConfig
from src.interaction.resolver import IntentResolver
from src.models import (
    BrowserIntent,
    GestureEvent,
    IntentType,
    NormalizedFeatures,
)


def _features(
    hand_center: tuple[float, float] = (0.5, 0.5),
    frame_id: int = 1,
    timestamp: float = 0.0,
) -> NormalizedFeatures:
    return NormalizedFeatures(
        pinch_distance=0.5,
        hand_center=hand_center,
        finger_extensions={},
        openness_score=0.5,
        hand_scale=1.0,
        frame_id=frame_id,
        timestamp=timestamp,
    )


def _event(
    event_type: str,
    *,
    frame_id: int = 1,
    timestamp: float = 0.0,
    hand_position: tuple[float, float] = (0.5, 0.5),
) -> GestureEvent:
    return GestureEvent(
        event_type=event_type,
        frame_id=frame_id,
        timestamp=timestamp,
        hand_position=hand_position,
        metadata={},
    )


# ---------------------------------------------------------------- lifecycle


def test_nav_engage_records_engage_position() -> None:
    resolver = IntentResolver(IntentConfig())
    resolver.resolve(_event("nav_engage", hand_position=(0.3, 0.6)))
    assert resolver._nav_engage_position == (0.3, 0.6)
    assert resolver._nav_active is True


def test_nav_release_clears_state() -> None:
    resolver = IntentResolver(IntentConfig())
    resolver.resolve(_event("nav_engage"))
    resolver.resolve(_event("nav_release"))
    assert resolver._nav_active is False
    assert resolver._nav_engage_position is None
    assert resolver._previous_position is None


# ---------------------------------------------------------------- scroll (while nav active)


def test_stationary_hand_emits_no_scroll_intents() -> None:
    resolver = IntentResolver(IntentConfig())
    resolver.resolve(_event("nav_engage"))
    for i in range(10):
        intents = resolver.update_nav_active(
            _features(hand_center=(0.5, 0.5), timestamp=i * 0.1, frame_id=i + 1)
        )
        assert intents == []


def test_scroll_intent_magnitude_proportional_to_displacement() -> None:
    config = IntentConfig(scroll_deadzone=0.02, scroll_scale_factor=500.0)
    resolver = IntentResolver(config)
    resolver.resolve(_event("nav_engage"))
    intents = resolver.update_nav_active(
        _features(hand_center=(0.5, 0.6), frame_id=1, timestamp=0.0)
    )
    assert len(intents) == 1
    assert intents[0].intent_type == IntentType.SCROLL_DOWN
    assert intents[0].magnitude == 40.0  # (0.10 - 0.02) * 500


def test_scroll_direction_up() -> None:
    resolver = IntentResolver(IntentConfig(scroll_deadzone=0.02))
    resolver.resolve(_event("nav_engage"))
    intents = resolver.update_nav_active(
        _features(hand_center=(0.5, 0.3), frame_id=1, timestamp=0.0)
    )
    assert intents[0].intent_type == IntentType.SCROLL_UP


# ---------------------------------------------------------------- tab switch


def test_tab_switch_right_emits_next_tab() -> None:
    config = IntentConfig(tab_switch_threshold=0.15, tab_switch_rearm_threshold=0.05)
    resolver = IntentResolver(config)
    resolver.resolve(_event("nav_engage", hand_position=(0.5, 0.5)))
    intents = resolver.update_nav_active(
        _features(hand_center=(0.8, 0.5), frame_id=1, timestamp=0.0)
    )
    assert any(i.intent_type == IntentType.NEXT_TAB for i in intents)


def test_tab_switch_left_emits_previous_tab() -> None:
    resolver = IntentResolver(IntentConfig())
    resolver.resolve(_event("nav_engage", hand_position=(0.5, 0.5)))
    intents = resolver.update_nav_active(
        _features(hand_center=(0.2, 0.5), frame_id=1, timestamp=0.0)
    )
    assert any(i.intent_type == IntentType.PREVIOUS_TAB for i in intents)


def test_tab_switch_requires_rearm() -> None:
    resolver = IntentResolver(
        IntentConfig(tab_switch_threshold=0.15, tab_switch_rearm_threshold=0.05)
    )
    resolver.resolve(_event("nav_engage", hand_position=(0.5, 0.5)))

    first = resolver.update_nav_active(
        _features(hand_center=(0.8, 0.5), frame_id=1, timestamp=0.0)
    )
    assert any(i.intent_type == IntentType.NEXT_TAB for i in first)

    # Second swing without re-arm → no tab-switch intent.
    second = resolver.update_nav_active(
        _features(hand_center=(0.9, 0.5), frame_id=2, timestamp=0.1)
    )
    assert not any(
        i.intent_type in (IntentType.NEXT_TAB, IntentType.PREVIOUS_TAB) for i in second
    )

    # Return to within re-arm threshold.
    resolver.update_nav_active(
        _features(hand_center=(0.5, 0.5), frame_id=3, timestamp=0.2)
    )
    third = resolver.update_nav_active(
        _features(hand_center=(0.8, 0.5), frame_id=4, timestamp=0.3)
    )
    assert any(i.intent_type == IntentType.NEXT_TAB for i in third)


# ---------------------------------------------------------------- thumb / swipe / close mapping


def test_thumb_up_maps_to_browser_back() -> None:
    intents = IntentResolver(IntentConfig()).resolve(_event("thumb_up"))
    assert intents[0].intent_type == IntentType.BROWSER_BACK


def test_thumb_down_maps_to_browser_forward() -> None:
    intents = IntentResolver(IntentConfig()).resolve(_event("thumb_down"))
    assert intents[0].intent_type == IntentType.BROWSER_FORWARD


def test_swipe_right_maps_to_browser_forward() -> None:
    intents = IntentResolver(IntentConfig()).resolve(_event("swipe_right"))
    assert intents[0].intent_type == IntentType.BROWSER_FORWARD


def test_swipe_left_maps_to_browser_back() -> None:
    intents = IntentResolver(IntentConfig()).resolve(_event("swipe_left"))
    assert intents[0].intent_type == IntentType.BROWSER_BACK


def test_tab_close_maps_to_close_tab() -> None:
    intents = IntentResolver(IntentConfig()).resolve(_event("tab_close"))
    assert intents[0].intent_type == IntentType.CLOSE_TAB


def test_unknown_event_emits_no_intents() -> None:
    intents = IntentResolver(IntentConfig()).resolve(_event("weird_event"))
    assert intents == []


def test_reset_clears_state() -> None:
    resolver = IntentResolver(IntentConfig())
    resolver.resolve(_event("nav_engage", hand_position=(0.3, 0.3)))
    resolver.reset()
    assert resolver._nav_active is False
    assert resolver._nav_engage_position is None
    assert resolver._previous_position is None
    assert resolver._tab_switch_armed is True


def test_update_nav_active_returns_empty_when_not_engaged() -> None:
    """No intents emitted while nav-mode is OFF even if hand moves."""
    resolver = IntentResolver(IntentConfig())
    intents = resolver.update_nav_active(
        _features(hand_center=(0.5, 0.9), frame_id=1, timestamp=0.0)
    )
    assert intents == []
