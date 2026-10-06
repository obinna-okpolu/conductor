"""Integration tests for the gesture → intent flow.

Exercises the pipeline end-to-end with mocked camera and MediaPipe.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from config.config import (
    CameraConfig,
    Config,
    DetectorConfig,
    GestureConfig,
    HUDConfig,
    IntentConfig,
    StateMachineConfig,
)
from src.actions.adapter import BrowserActionAdapter
from src.actions.dispatcher import ActionDispatcher
from src.gestures.recognizer import GestureRecognizer
from src.interaction.resolver import IntentResolver
from src.interaction.state_machine import GestureStateMachine
from src.models import (
    GestureCandidate,
    GestureEvent,
    IntentType,
    Landmark,
    HandLandmarks,
    NormalizedFeatures,
)


class _RecordingAdapter(BrowserActionAdapter):
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def scroll_up(self, amount: float) -> None:
        self.calls.append(("scroll_up", (amount,)))

    def scroll_down(self, amount: float) -> None:
        self.calls.append(("scroll_down", (amount,)))

    def next_tab(self) -> None:
        self.calls.append(("next_tab", ()))

    def previous_tab(self) -> None:
        self.calls.append(("previous_tab", ()))

    def browser_back(self) -> None:
        self.calls.append(("browser_back", ()))

    def browser_forward(self) -> None:
        self.calls.append(("browser_forward", ()))

    def close_tab(self) -> None:
        self.calls.append(("close_tab", ()))


def _make_landmarks(frame_id: int) -> HandLandmarks:
    return HandLandmarks(
        landmarks=[Landmark(0.5, 0.5, 0.0) for _ in range(21)],
        handedness="Right",
        frame_id=frame_id,
    )


def _features(
    center=(0.5, 0.5), frame_id=1, timestamp=0.0, extensions=None, openness=1.0
) -> NormalizedFeatures:
    return NormalizedFeatures(
        pinch_distance=0.5,
        hand_center=center,
        finger_extensions=extensions
        or {
            "thumb": 1.0,
            "index": 1.0,
            "middle": 1.0,
            "ring": 1.0,
            "pinky": 1.0,
        },
        openness_score=openness,
        hand_scale=1.0,
        frame_id=frame_id,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------- nav-mode flow


def test_nav_engage_scroll_release_flow() -> None:
    """VICTORY engages, hand Y movement scrolls, OPEN_PALM disengages."""
    config = Config()
    resolver = IntentResolver(config.intent)
    state_machine = GestureStateMachine(config.state_machine)
    recognizer = GestureRecognizer(config.gesture)
    adapter = _RecordingAdapter()
    dispatcher = ActionDispatcher(adapter)

    # 1) Engage nav-mode by holding VICTORY for >= 0.5 s.
    #    First frame at t=0.0 starts counting; subsequent frames at
    #    t=0.3, 0.6 trigger engagement.
    vics = []
    for i, t in enumerate([0.0, 0.3, 0.6]):
        cand = _candidate_victory(i, t)
        cand = dataclasses.replace(cand, features=_features(timestamp=t))
        vics.append(cand)

    events = []
    for cand in vics:
        events.extend(
            state_machine.process(
                cand,
                DetectionResult_stub(
                    landmarks=_make_landmarks(cand.frame_id), timestamp=cand.timestamp
                ),
            )
        )
    assert any(e.event_type == "nav_engage" for e in events)
    for e in events:
        for intent in resolver.resolve(e):
            dispatcher.dispatch(intent)

    # 2) Move hand down significantly → SCROLL_DOWN dispatched.
    move = _features(center=(0.5, 0.7), timestamp=1.0)
    intents = resolver.update_nav_active(move)
    for intent in intents:
        dispatcher.dispatch(intent)
    assert any(c[0] == "scroll_down" for c in adapter.calls)

    # 3) Disengage with OPEN_PALM held.
    #    Hold for >= 0.5 s; use t=2.0, 2.6, 3.2 to clear the 0.5-s threshold.
    palms = []
    for i, t in enumerate([2.0, 2.6, 3.2]):
        cand = _candidate_open_palm(i + 10, t)
        cand = dataclasses.replace(cand, features=_features(timestamp=t))
        palms.append(cand)
    for cand in palms:
        events = state_machine.process(
            cand,
            DetectionResult_stub(
                landmarks=_make_landmarks(cand.frame_id), timestamp=cand.timestamp
            ),
        )
        for e in events:
            for intent in resolver.resolve(e):
                dispatcher.dispatch(intent)
    assert not resolver._nav_active

    # 4) After release, no further scroll intents even if the hand moves.
    more = resolver.update_nav_active(_features(center=(0.5, 0.9), timestamp=3.0))
    assert more == []


# ---------------------------------------------------------------- thumb flow


def test_thumb_up_dispatches_browser_back() -> None:
    config = Config()
    resolver = IntentResolver(config.intent)
    state_machine = GestureStateMachine(config.state_machine)
    adapter = _RecordingAdapter()
    dispatcher = ActionDispatcher(adapter)

    cand = dataclasses.replace(
        _candidate_thumb_up(0), features=_features(timestamp=0.0)
    )
    events = state_machine.process(
        cand, DetectionResult_stub(landmarks=_make_landmarks(0), timestamp=0.0)
    )
    assert any(e.event_type == "thumb_up" for e in events)

    for e in events:
        for intent in resolver.resolve(e):
            dispatcher.dispatch(intent)

    assert ("browser_back", ()) in adapter.calls


def test_thumb_down_dispatches_browser_forward() -> None:
    config = Config()
    resolver = IntentResolver(config.intent)
    state_machine = GestureStateMachine(config.state_machine)
    adapter = _RecordingAdapter()
    dispatcher = ActionDispatcher(adapter)

    cand = dataclasses.replace(
        _candidate_thumb_down(0), features=_features(timestamp=0.0)
    )
    events = state_machine.process(
        cand, DetectionResult_stub(landmarks=_make_landmarks(0), timestamp=0.0)
    )
    assert any(e.event_type == "thumb_down" for e in events)

    for e in events:
        for intent in resolver.resolve(e):
            dispatcher.dispatch(intent)

    assert ("browser_forward", ()) in adapter.calls


# ---------------------------------------------------------------- fist flow


def test_fist_hold_dispatches_close_tab() -> None:
    config = Config(state_machine=StateMachineConfig(fist_hold_duration=0.2))
    resolver = IntentResolver(config.intent)
    state_machine = GestureStateMachine(config.state_machine)
    adapter = _RecordingAdapter()
    dispatcher = ActionDispatcher(adapter)

    emitted = []
    for i in range(3):
        cand = dataclasses.replace(
            _candidate_fist(i, t=i * 0.2),
            features=_features(timestamp=i * 0.2),
        )
        events = state_machine.process(
            cand, DetectionResult_stub(landmarks=_make_landmarks(i), timestamp=i * 0.2)
        )
        emitted.extend(events)

    assert any(event.event_type == "tab_close" for event in emitted)
    for event in emitted:
        for intent in resolver.resolve(event):
            dispatcher.dispatch(intent)

    assert ("close_tab", ()) in adapter.calls


# ---------------------------------------------------------------- stubs


class DetectionResult_stub:
    """Plain object that looks like a DetectionResult."""

    def __init__(
        self, *, landmarks=None, confidence=1.0, timestamp=0.0, frame_id=1
    ) -> None:
        self.landmarks = landmarks
        self.confidence = confidence
        self.timestamp = timestamp
        self.frame_id = frame_id


def _candidate_victory(i: int, t: float = 0.0) -> GestureCandidate:
    return GestureCandidate(
        gesture_type=__import__(
            "src.models", fromlist=["GestureType"]
        ).GestureType.VICTORY,
        confidence=1.0,
        frame_id=i,
        timestamp=t,
        features=None,
        swipe_metadata=None,
    )


def _candidate_open_palm(i: int, t: float = 0.0) -> GestureCandidate:
    return GestureCandidate(
        gesture_type=__import__(
            "src.models", fromlist=["GestureType"]
        ).GestureType.OPEN_PALM,
        confidence=1.0,
        frame_id=i,
        timestamp=t,
        features=None,
        swipe_metadata=None,
    )


def _candidate_thumb_up(i: int, t: float = 0.0) -> GestureCandidate:
    return GestureCandidate(
        gesture_type=__import__(
            "src.models", fromlist=["GestureType"]
        ).GestureType.THUMB_UP,
        confidence=1.0,
        frame_id=i,
        timestamp=t,
        features=None,
        swipe_metadata=None,
    )


def _candidate_thumb_down(i: int, t: float = 0.0) -> GestureCandidate:
    return GestureCandidate(
        gesture_type=__import__(
            "src.models", fromlist=["GestureType"]
        ).GestureType.THUMB_DOWN,
        confidence=1.0,
        frame_id=i,
        timestamp=t,
        features=None,
        swipe_metadata=None,
    )


def _candidate_fist(i: int, t: float = 0.0) -> GestureCandidate:
    return GestureCandidate(
        gesture_type=__import__(
            "src.models", fromlist=["GestureType"]
        ).GestureType.CLOSED_FIST,
        confidence=1.0,
        frame_id=i,
        timestamp=t,
        features=None,
        swipe_metadata=None,
    )
