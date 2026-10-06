"""Unit tests for the GestureStateMachine.

Covers the new vocabulary: VICTORY → nav-mode, CLOSED_FIST → tab_close,
THUMB_UP/DOWN → back/forward, swipe re-arm, plus safety gates.
"""

from __future__ import annotations

from config.config import StateMachineConfig
from src.interaction.state_machine import GestureStateMachine
from src.models import (
    DetectionResult,
    FistState,
    GestureCandidate,
    GestureType,
    HandLandmarks,
    Landmark,
    NavModeState,
    NormalizedFeatures,
    SwipeMetadata,
    SwipeState,
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


def _candidate(
    gesture_type: GestureType,
    *,
    features: NormalizedFeatures,
    confidence: float = 1.0,
    swipe_metadata: SwipeMetadata | None = None,
    timestamp: float | None = None,
) -> GestureCandidate:
    ts = timestamp if timestamp is not None else features.timestamp
    return GestureCandidate(
        gesture_type=gesture_type,
        confidence=confidence,
        frame_id=features.frame_id,
        timestamp=ts,
        features=features,
        swipe_metadata=swipe_metadata,
    )


def _detection(
    confidence: float = 1.0,
    *,
    no_hand: bool = False,
    timestamp: float = 0.0,
    frame_id: int = 1,
) -> DetectionResult:
    """Build a DetectionResult; default supplies landmarks.

    Pass ``no_hand=True`` to exercise the no-hand suppression branch.
    """
    landmarks = None if no_hand else _landmarks()
    return DetectionResult(
        landmarks=landmarks,
        confidence=confidence,
        frame_id=frame_id,
        timestamp=timestamp,
    )


def _landmarks() -> HandLandmarks:
    return HandLandmarks(
        landmarks=[Landmark(0.5, 0.5, 0.0) for _ in range(21)],
        handedness="Right",
        frame_id=1,
    )


# A sentinel distinguishing "no explicit landmarks" from "explicit no_hand".
_AUTO = object()


# ---------------------------------------------------------------- nav-mode


def test_victory_held_emits_nav_engage_after_threshold() -> None:
    sm = GestureStateMachine(StateMachineConfig())
    f1 = _features(timestamp=0.0)
    events = sm.process(
        _candidate(GestureType.VICTORY, features=f1), _detection(timestamp=0.0)
    )
    assert events == []  # Hold not yet satisfied.
    assert sm.get_state().nav_mode_state == NavModeState.IDLE

    f2 = _features(timestamp=0.7, frame_id=2)
    events = sm.process(
        _candidate(GestureType.VICTORY, features=f2),
        _detection(timestamp=0.7, frame_id=2),
    )
    assert len(events) == 1
    assert events[0].event_type == "nav_engage"
    assert sm.get_state().nav_mode_state.name in ("ENGAGED", "ACTIVE")


def test_open_palm_held_after_engage_emits_nav_release() -> None:
    sm = GestureStateMachine(StateMachineConfig())
    sm.process(
        _candidate(GestureType.VICTORY, features=_features(timestamp=0.0)),
        _detection(timestamp=0.0),
    )
    sm.process(
        _candidate(GestureType.VICTORY, features=_features(timestamp=0.7, frame_id=2)),
        _detection(timestamp=0.7, frame_id=2),
    )
    # Now open palm for long enough.
    sm.process(
        _candidate(
            GestureType.OPEN_PALM, features=_features(timestamp=1.0, frame_id=3)
        ),
        _detection(timestamp=1.0, frame_id=3),
    )
    sm.process(
        _candidate(
            GestureType.OPEN_PALM, features=_features(timestamp=1.8, frame_id=4)
        ),
        _detection(timestamp=1.8, frame_id=4),
    )
    state = sm.get_state()
    assert state.nav_mode_state == NavModeState.IDLE


def test_nav_release_emits_event() -> None:
    sm = GestureStateMachine(StateMachineConfig())
    sm.process(
        _candidate(GestureType.VICTORY, features=_features(timestamp=0.0)),
        _detection(timestamp=0.0),
    )
    sm.process(
        _candidate(GestureType.VICTORY, features=_features(timestamp=0.6, frame_id=2)),
        _detection(timestamp=0.6, frame_id=2),
    )
    events = sm.process(
        _candidate(
            GestureType.OPEN_PALM, features=_features(timestamp=1.0, frame_id=3)
        ),
        _detection(timestamp=1.0, frame_id=3),
    )
    assert events == []
    events = sm.process(
        _candidate(
            GestureType.OPEN_PALM, features=_features(timestamp=1.8, frame_id=4)
        ),
        _detection(timestamp=1.8, frame_id=4),
    )
    assert any(e.event_type == "nav_release" for e in events)


# ------------------------------------------------------------- close gesture


def test_idle_open_palm_hold_emits_tab_close() -> None:
    sm = GestureStateMachine(StateMachineConfig(fist_hold_duration=0.5))
    sm.process(
        _candidate(GestureType.OPEN_PALM, features=_features(timestamp=0.0)),
        _detection(timestamp=0.0),
    )
    events = sm.process(
        _candidate(
            GestureType.OPEN_PALM, features=_features(timestamp=0.6, frame_id=2)
        ),
        _detection(timestamp=0.6, frame_id=2),
    )
    assert any(event.event_type == "tab_close" for event in events)


# Legacy programmatic fist compatibility; the live recognizer no longer emits it.


def test_fist_hold_emits_event_after_duration() -> None:
    sm = GestureStateMachine(StateMachineConfig(fist_hold_duration=0.5))
    f0 = _features(timestamp=0.0)
    e1 = sm.process(
        _candidate(GestureType.CLOSED_FIST, features=f0), _detection(timestamp=0.0)
    )
    assert e1 == []
    assert sm.get_state().fist_state == FistState.CANDIDATE

    e2 = sm.process(
        _candidate(
            GestureType.CLOSED_FIST, features=_features(timestamp=0.3, frame_id=2)
        ),
        _detection(timestamp=0.3, frame_id=2),
    )
    assert e2 == []

    e3 = sm.process(
        _candidate(
            GestureType.CLOSED_FIST, features=_features(timestamp=0.6, frame_id=3)
        ),
        _detection(timestamp=0.6, frame_id=3),
    )
    assert any(e.event_type == "tab_close" for e in e3)
    assert sm.get_state().fist_state == FistState.WAIT_FOR_REARM


def test_fist_early_release_emits_no_event() -> None:
    sm = GestureStateMachine(StateMachineConfig(fist_hold_duration=0.5))
    sm.process(
        _candidate(GestureType.CLOSED_FIST, features=_features(timestamp=0.0)),
        _detection(timestamp=0.0),
    )
    e2 = sm.process(
        _candidate(GestureType.NONE, features=_features(timestamp=0.2, frame_id=2)),
        _detection(timestamp=0.2, frame_id=2),
    )
    assert e2 == []
    assert sm.get_state().fist_state == FistState.NEUTRAL


# ---------------------------------------------------------------- swipe


def test_swipe_left_emits_event_once() -> None:
    sm = GestureStateMachine(StateMachineConfig())
    meta = SwipeMetadata(-0.4, 0.2, -2.0, 0.9)
    events = sm.process(
        _candidate(GestureType.SWIPE_LEFT, features=_features(), swipe_metadata=meta),
        _detection(),
    )
    assert len(events) == 1
    assert events[0].event_type == "swipe_left"
    assert sm.get_state().swipe_state == SwipeState.WAIT_FOR_REARM


# ---------------------------------------------------------------- thumb


def test_thumb_up_emits_browser_back_event() -> None:
    sm = GestureStateMachine(StateMachineConfig())
    events = sm.process(
        _candidate(GestureType.THUMB_UP, features=_features()),
        _detection(),
    )
    assert any(e.event_type == "thumb_up" for e in events)


def test_thumb_down_emits_browser_forward_event() -> None:
    sm = GestureStateMachine(StateMachineConfig())
    events = sm.process(
        _candidate(GestureType.THUMB_DOWN, features=_features()),
        _detection(),
    )
    assert any(e.event_type == "thumb_down" for e in events)


# ---------------------------------------------------------------- safety


def test_low_confidence_suppresses() -> None:
    sm = GestureStateMachine(StateMachineConfig(confidence_threshold=0.5))
    e = sm.process(
        _candidate(GestureType.VICTORY, features=_features()),
        _detection(confidence=0.3),
    )
    assert e == []


def test_no_hand_resets_to_neutral() -> None:
    sm = GestureStateMachine(StateMachineConfig())
    sm.process(
        _candidate(GestureType.VICTORY, features=_features(timestamp=0.0)),
        _detection(timestamp=0.0),
    )
    sm.process(
        _candidate(GestureType.VICTORY, features=_features(timestamp=0.6, frame_id=2)),
        _detection(timestamp=0.6, frame_id=2),
    )
    assert sm.get_state().nav_mode_state.name in ("ENGAGED", "ACTIVE")

    sm.process(
        _candidate(GestureType.NONE, features=_features(timestamp=1.0, frame_id=3)),
        _detection(no_hand=True, timestamp=1.0, frame_id=3),
    )
    assert sm.get_state().nav_mode_state == NavModeState.IDLE


def test_reset_clears_all_state() -> None:
    sm = GestureStateMachine(StateMachineConfig())
    sm.process(_candidate(GestureType.VICTORY, features=_features()), _detection())
    sm.reset()
    s = sm.get_state()
    assert s.nav_mode_state == NavModeState.IDLE
    assert s.fist_state == FistState.NEUTRAL
    assert s.swipe_state == SwipeState.NEUTRAL
    assert s.nav_engage_position is None
    assert s.suppressed is False
