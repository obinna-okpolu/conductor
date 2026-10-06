"""Unit tests for the GestureRecognizer.

Synthetic HandLandmarks drive the per-frame classifier and the
stability/confirmation logic.
"""

from __future__ import annotations

import dataclasses

from src.models import (
    GestureType,
    HandLandmarks,
    Landmark,
    NormalizedFeatures,
)


def _make_landmarks(
    *,
    wrist=(0.5, 0.5, 0.0),
    thumb_mcp=(0.55, 0.5, 0.0),
    thumb_tip=(0.45, 0.5, 0.0),
    index_pip=(0.5, 0.3, 0.0),
    index_tip=(0.5, 0.1, 0.0),
    middle_pip=(0.5, 0.5, 0.0),
    middle_tip=(0.5, 0.3, 0.0),
    ring_pip=(0.5, 0.7, 0.0),
    ring_tip=(0.5, 0.5, 0.0),
    pinky_pip=(0.5, 0.7, 0.0),
    pinky_tip=(0.5, 0.5, 0.0),
    handedness: str = "Right",
    frame_id: int = 1,
) -> HandLandmarks:
    """Construct 21 landmarks.

    By default: thumb extended to the left of MCP, all four non-thumb fingers
    extended (tips above their PIPs). Override ``*_tip`` and ``*_pip`` to
    construct other poses.

    Coordinate convention:
        * x increases right, y increases DOWN.
        * "Tip above PIP" → tip.y < pip.y.
        * Right-hand thumb extended → tip.x < mcp.x.
    """
    points: list[Landmark] = [Landmark(*wrist) for _ in range(21)]
    # Thumb column (1..4)
    points[1] = Landmark(0.5, 0.5, 0.0)
    points[2] = Landmark(*thumb_mcp)
    points[3] = Landmark(0.5, 0.5, 0.0)
    points[4] = Landmark(*thumb_tip)
    # Index column (5..8)
    points[5] = Landmark(0.5, 0.4, 0.0)
    points[6] = Landmark(*index_pip)
    points[7] = Landmark(0.5, 0.2, 0.0)
    points[8] = Landmark(*index_tip)
    # Middle column (9..12)
    points[9] = Landmark(0.5, 0.45, 0.0)
    points[10] = Landmark(*middle_pip)
    points[11] = Landmark(0.5, 0.4, 0.0)
    points[12] = Landmark(*middle_tip)
    # Ring column (13..16)
    points[13] = Landmark(0.5, 0.55, 0.0)
    points[14] = Landmark(*ring_pip)
    points[15] = Landmark(0.5, 0.6, 0.0)
    points[16] = Landmark(*ring_tip)
    # Pinky column (17..20)
    points[17] = Landmark(*pinky_pip)
    points[18] = Landmark(0.5, 0.6, 0.0)
    points[19] = Landmark(0.5, 0.65, 0.0)
    points[20] = Landmark(*pinky_tip)
    return HandLandmarks(landmarks=points, handedness=handedness, frame_id=frame_id)


def _dummy_features() -> NormalizedFeatures:
    return NormalizedFeatures(
        pinch_distance=0.5,
        hand_center=(0.5, 0.5),
        finger_extensions={},
        openness_score=0.5,
        hand_scale=1.0,
        frame_id=1,
        timestamp=0.0,
    )


def _recognize_n(
    recognizer, *, n: int = 1, landmarks_factory=lambda: _make_landmarks()
):
    """Feed N frames into the recognizer; return per-frame gesture types."""
    from src.gestures.recognizer import GestureRecognizer

    types = []
    for i in range(n):
        landmarks = dataclasses.replace(landmarks_factory(), frame_id=i)
    return types


def _recognize_helper(recognizer, factory, n=1):
    """Actually feed n frames and return the per-frame gesture types."""
    types = []
    for i in range(n):
        landmarks = dataclasses.replace(factory(), frame_id=i)
        cand = recognizer.recognize(landmarks, 1.0)
        if cand.features is None:
            cand = dataclasses.replace(cand, features=_dummy_features())
        types.append(cand.gesture_type)
    return types


# ---------------------------------------------------------------- tests


def test_open_palm_detected_when_all_fingers_extended():
    from config.config import GestureConfig
    from src.gestures.recognizer import GestureRecognizer

    recognizer = GestureRecognizer(GestureConfig(), hold_frames=1, cooldown_frames=0)
    # Use defaults: all 5 fingers extended.
    types = _recognize_helper(recognizer, lambda: _make_landmarks(), n=4)
    assert types[-1] == GestureType.OPEN_PALM


def test_closed_fist_is_not_emitted_when_all_fingers_curled():
    from config.config import GestureConfig
    from src.gestures.recognizer import GestureRecognizer

    recognizer = GestureRecognizer(GestureConfig(), hold_frames=1, cooldown_frames=0)
    # Tips BELOW PIPs (larger y) → curled. Thumb tucked (tip RIGHT of MCP for right hand).
    factory = lambda: _make_landmarks(
        thumb_tip=(0.6, 0.5, 0.0),  # thumb NOT extended (right of mcp)
        thumb_mcp=(0.55, 0.5, 0.0),
        index_tip=(0.5, 0.5, 0.0),
        index_pip=(0.5, 0.3, 0.0),
        middle_tip=(0.5, 0.8, 0.0),
        middle_pip=(0.5, 0.5, 0.0),
        ring_tip=(0.5, 0.9, 0.0),
        ring_pip=(0.5, 0.7, 0.0),
        pinky_tip=(0.5, 1.0, 0.0),
        pinky_pip=(0.5, 0.7, 0.0),
    )
    types = _recognize_helper(recognizer, factory, n=4)
    assert types[-1] == GestureType.NONE


def test_victory_detected_for_peace_sign():
    from config.config import GestureConfig
    from src.gestures.recognizer import GestureRecognizer

    recognizer = GestureRecognizer(GestureConfig(), hold_frames=1, cooldown_frames=0)
    # index + middle extended, ring + pinky curled.
    factory = lambda: _make_landmarks(
        thumb_tip=(0.55, 0.5, 0.0),  # thumb NOT extended (tip right of mcp)
        index_tip=(0.5, 0.1, 0.0),  # extended
        index_pip=(0.5, 0.3, 0.0),
        middle_tip=(0.5, 0.2, 0.0),  # extended
        middle_pip=(0.5, 0.5, 0.0),
        ring_tip=(0.5, 0.9, 0.0),  # curled
        ring_pip=(0.5, 0.7, 0.0),
        pinky_tip=(0.5, 1.0, 0.0),  # curled
        pinky_pip=(0.5, 0.7, 0.0),
    )
    types = _recognize_helper(recognizer, factory, n=4)
    assert types[-1] == GestureType.VICTORY


def test_thumb_up_detected():
    from config.config import GestureConfig
    from src.gestures.recognizer import GestureRecognizer

    recognizer = GestureRecognizer(GestureConfig(), hold_frames=1, cooldown_frames=0)
    factory = lambda: _make_landmarks(
        thumb_tip=(0.45, 0.2, 0.0),  # above wrist (y=0.2 < wrist.y=0.5) and extended
        index_tip=(0.5, 0.5, 0.0),  # curled
        index_pip=(0.5, 0.3, 0.0),
        middle_tip=(0.5, 0.8, 0.0),  # curled
        middle_pip=(0.5, 0.5, 0.0),
        ring_tip=(0.5, 0.9, 0.0),  # curled
        ring_pip=(0.5, 0.7, 0.0),
        pinky_tip=(0.5, 1.0, 0.0),  # curled
        pinky_pip=(0.5, 0.7, 0.0),
    )
    types = _recognize_helper(recognizer, factory, n=4)
    assert types[-1] == GestureType.THUMB_UP


def test_thumb_down_detected():
    from config.config import GestureConfig
    from src.gestures.recognizer import GestureRecognizer

    recognizer = GestureRecognizer(GestureConfig(), hold_frames=1, cooldown_frames=0)
    factory = lambda: _make_landmarks(
        thumb_tip=(0.45, 0.8, 0.0),  # below wrist (y=0.8 > wrist.y=0.5) and extended
        index_tip=(0.5, 0.5, 0.0),
        index_pip=(0.5, 0.3, 0.0),
        middle_tip=(0.5, 0.8, 0.0),
        middle_pip=(0.5, 0.5, 0.0),
        ring_tip=(0.5, 0.9, 0.0),
        ring_pip=(0.5, 0.7, 0.0),
        pinky_tip=(0.5, 1.0, 0.0),
        pinky_pip=(0.5, 0.7, 0.0),
    )
    types = _recognize_helper(recognizer, factory, n=4)
    assert types[-1] == GestureType.THUMB_DOWN


def test_ambiguous_sideways_thumb_with_curled_fingers_is_ignored():
    """A non-vertical closed hand is not a live gesture anymore."""
    from config.config import GestureConfig
    from src.gestures.recognizer import GestureRecognizer

    recognizer = GestureRecognizer(GestureConfig(), hold_frames=1, cooldown_frames=0)
    factory = lambda: _make_landmarks(
        thumb_tip=(0.45, 0.5, 0.0),  # extended reach, but not up/down
        index_tip=(0.5, 0.5, 0.0),
        index_pip=(0.5, 0.3, 0.0),
        middle_tip=(0.5, 0.8, 0.0),
        middle_pip=(0.5, 0.5, 0.0),
        ring_tip=(0.5, 0.9, 0.0),
        ring_pip=(0.5, 0.7, 0.0),
        pinky_tip=(0.5, 1.0, 0.0),
        pinky_pip=(0.5, 0.7, 0.0),
    )
    types = _recognize_helper(recognizer, factory, n=4)
    assert types[-1] == GestureType.NONE


def test_stability_confirmation_requires_consecutive_frames():
    """Gesture must appear for N consecutive frames before firing."""
    from config.config import GestureConfig
    from src.gestures.recognizer import GestureRecognizer

    recognizer = GestureRecognizer(
        GestureConfig(),
        hold_frames=3,
        cooldown_frames=0,
        classification_hysteresis=1,
    )
    types = _recognize_helper(recognizer, lambda: _make_landmarks(), n=5)
    # First 2 frames: NONE (still building stability).
    assert types[0] == GestureType.NONE
    assert types[1] == GestureType.NONE
    # 3rd frame: OPEN_PALM fires.
    assert types[2] == GestureType.OPEN_PALM


def test_cooldown_suppresses_repeat():
    from config.config import GestureConfig
    from src.gestures.recognizer import GestureRecognizer

    recognizer = GestureRecognizer(
        GestureConfig(),
        hold_frames=1,
        cooldown_frames=5,
        classification_hysteresis=1,
    )
    types = _recognize_helper(recognizer, lambda: _make_landmarks(), n=8)
    assert types[0] == GestureType.OPEN_PALM
    # Cooldown frames: NONE.
    assert all(t == GestureType.NONE for t in types[1:6])
    # After cooldown, fires again.
    assert types[6] == GestureType.OPEN_PALM


def test_classification_hysteresis_kills_flicker():
    """If the raw classification flips back and forth, the smoothed output stays stable."""
    from config.config import GestureConfig
    from src.gestures.recognizer import GestureRecognizer
    from src.models import HandLandmarks, Landmark

    closed = _make_landmarks(
        thumb_tip=(0.6, 0.5, 0.0),
        thumb_mcp=(0.55, 0.5, 0.0),
        index_tip=(0.5, 0.5, 0.0),
        middle_tip=(0.5, 0.7, 0.0),
        ring_tip=(0.5, 0.9, 0.0),
        pinky_tip=(0.5, 1.0, 0.0),
    )
    open_ = _make_landmarks()

    recognizer = GestureRecognizer(
        GestureConfig(),
        hold_frames=1,
        cooldown_frames=0,
        classification_hysteresis=5,
    )

    sequence: list[HandLandmarks] = []
    for _ in range(10):
        sequence.append(closed)
        sequence.append(open_)
        sequence.append(closed)

    types: list[GestureType] = []
    for i, lm in enumerate(sequence):
        lm = dataclasses.replace(lm, frame_id=i)
        cand = recognizer.recognize(lm, 1.0)
        types.append(cand.gesture_type)

    # The smoothed output should not flip back and forth — once it commits
    # to CLOSED_FIST (or OPEN_PALM), it should stay there for a while.
    transitions = sum(1 for i in range(1, len(types)) if types[i] != types[i - 1])
    assert transitions < 5, f"too many transitions: {transitions}"


def test_reset_clears_state():
    from config.config import GestureConfig
    from src.gestures.recognizer import GestureRecognizer

    recognizer = GestureRecognizer(GestureConfig())
    recognizer.reset()
    assert recognizer._current_gesture == GestureType.NONE
    assert recognizer._stable_count == 0
    assert recognizer._cooldown_remaining == 0
