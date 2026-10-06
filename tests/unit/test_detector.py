"""Unit tests for the hand landmark detector.

Requirements: 2.3, 2.4, 19.1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from config.config import DetectorConfig
from src.models import DetectionResult, Frame
from src.vision.detector import HandLandmarkDetector


# ---------------------------------------------------------------- stubs


@dataclass
class _StubLandmark:
    """Mimics a single ``mp.framework.formats.landmark_pb2.NormalizedLandmark``."""

    x: float
    y: float
    z: float = 0.0


@dataclass
class _StubHandLandmarks:
    """Mimics ``mp.solutions.hands.HandLandmark``."""

    landmark: list[_StubLandmark] = field(default_factory=list)


@dataclass
class _StubClassification:
    label: str = "Right"
    score: float = 0.95


@dataclass
class _StubHandedness:
    classification: list[_StubClassification] = field(default_factory=list)


@dataclass
class _StubHands:
    """Stub for ``mp.solutions.hands.Hands`` used in unit tests."""

    next_multi_hand_landmarks: list[_StubHandLandmarks] | None = None
    next_multi_handedness: list[_StubHandedness] | None = None
    raise_on_process: bool = False
    process_calls: int = 0
    close_calls: int = 0

    def process(self, _image: np.ndarray) -> Any:
        self.process_calls += 1
        if self.raise_on_process:
            raise RuntimeError("simulated mediapipe error")

        if self.next_multi_hand_landmarks is None:
            return None

        @dataclass
        class _Result:
            multi_hand_landmarks: list[_StubHandLandmarks] | None = None
            multi_handedness: list[_StubHandedness] | None = None

        return _Result(
            multi_hand_landmarks=self.next_multi_hand_landmarks,
            multi_handedness=self.next_multi_handedness,
        )

    def close(self) -> None:
        self.close_calls += 1


# ---------------------------------------------------------------- helpers


def _make_landmarks(n: int = 21) -> list[_StubLandmark]:
    return [_StubLandmark(x=0.5, y=0.5, z=0.0) for _ in range(n)]


def _make_frame(
    width: int = 64, height: int = 48, frame_id: int = 1, timestamp: float = 0.0
) -> Frame:
    return Frame(
        image=np.zeros((height, width, 3), dtype=np.uint8),
        timestamp=timestamp,
        frame_id=frame_id,
    )


# ---------------------------------------------------------------- tests


def test_process_with_no_hands_returns_no_hand_signal() -> None:
    hands = _StubHands(next_multi_hand_landmarks=None)
    detector = HandLandmarkDetector(DetectorConfig(), hands=hands)

    result = detector.process(_make_frame())

    assert result.landmarks is None
    assert result.confidence == 0.0
    assert result.frame_id == 1
    assert hands.process_calls == 1


def test_process_returns_landmarks_and_confidence_when_hand_detected() -> None:
    hands = _StubHands(
        next_multi_hand_landmarks=[_StubHandLandmarks(landmark=_make_landmarks())],
        next_multi_handedness=[_StubHandedness(classification=[_StubClassification()])],
    )
    detector = HandLandmarkDetector(DetectorConfig(), hands=hands)

    result = detector.process(_make_frame())

    assert result.landmarks is not None
    assert len(result.landmarks.landmarks) == 21
    assert result.landmarks.handedness == "Right"
    assert result.confidence == pytest.approx(0.95)


def test_process_handles_mediapipe_exception_gracefully() -> None:
    hands = _StubHands(raise_on_process=True)
    detector = HandLandmarkDetector(DetectorConfig(), hands=hands)

    result = detector.process(_make_frame())

    assert result.landmarks is None
    assert result.confidence == 0.0


def test_process_handles_missing_handedness() -> None:
    hands = _StubHands(
        next_multi_hand_landmarks=[_StubHandLandmarks(landmark=_make_landmarks())],
        next_multi_handedness=[],
    )
    detector = HandLandmarkDetector(DetectorConfig(), hands=hands)

    result = detector.process(_make_frame())

    assert result.landmarks is not None
    assert result.landmarks.handedness == "Unknown"
    assert result.confidence == 0.0


def test_is_low_confidence_threshold() -> None:
    hands = _StubHands()
    config = DetectorConfig(min_tracking_confidence=0.5)
    detector = HandLandmarkDetector(config, hands=hands)

    above = DetectionResult(landmarks=None, confidence=0.6, frame_id=1, timestamp=0.0)
    assert detector.is_low_confidence(above) is False

    below = DetectionResult(landmarks=None, confidence=0.3, frame_id=1, timestamp=0.0)
    assert detector.is_low_confidence(below) is True

    edge = DetectionResult(landmarks=None, confidence=0.5, frame_id=1, timestamp=0.0)
    assert detector.is_low_confidence(edge) is False  # Strict less-than


def test_get_confidence_returns_value() -> None:
    detector = HandLandmarkDetector(DetectorConfig(), hands=_StubHands())

    result = DetectionResult(landmarks=None, confidence=0.42, frame_id=1, timestamp=0.0)
    assert detector.get_confidence(result) == pytest.approx(0.42)


def test_close_is_safe_with_no_hands() -> None:
    detector = HandLandmarkDetector(DetectorConfig(), hands=None)
    detector.close()  # Should not raise.


def test_close_is_safe_with_exploding_hands() -> None:
    class ExplodingHands(_StubHands):
        def close(self) -> None:  # type: ignore[override]
            raise RuntimeError("boom")

    detector = HandLandmarkDetector(DetectorConfig(), hands=ExplodingHands())
    detector.close()  # Should not raise.


def test_process_preserves_frame_id_and_timestamp() -> None:
    hands = _StubHands()
    detector = HandLandmarkDetector(DetectorConfig(), hands=hands)
    frame = Frame(
        image=np.zeros((10, 10, 3), dtype=np.uint8), timestamp=12.5, frame_id=7
    )

    result = detector.process(frame)

    assert result.frame_id == 7
    assert result.timestamp == pytest.approx(12.5)
