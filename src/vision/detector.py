"""MediaPipe hand-landmark detector.

Wraps ``mediapipe.solutions.hands`` behind a small, testable interface.
All processing runs locally; no frames ever leave the device.

Note: MediaPipe 1.0 removed the legacy ``solutions`` API. This module
requires ``mediapipe<1.0`` (the 0.10.x line).

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

from __future__ import annotations

import logging as _logging
from typing import Any

from config.config import DetectorConfig
from src.models import DetectionResult, HandLandmarks, Landmark

_LOG = _logging.getLogger(__name__)


class HandLandmarkDetector:
    """Detect 21 hand landmarks and a confidence score from a frame.

    The class accepts a pre-built MediaPipe ``Hands`` instance so tests can
    substitute a stub. Production code constructs the real instance from
    :class:`DetectorConfig`.

    Attributes:
        config: Detector configuration.
    """

    def __init__(self, config: DetectorConfig, hands: Any | None = None) -> None:
        """Initialize the detector.

        Args:
            config: Detector configuration.
            hands: Optional pre-built MediaPipe Hands instance. When
                ``None``, a real instance is constructed from ``config``
                on first use.
        """
        self.config = config
        self._hands = hands

    # ------------------------------------------------------------------ lifecycle

    def _ensure_hands(self) -> Any:
        """Lazily construct the MediaPipe ``Hands`` instance.

        Returns:
            The MediaPipe Hands instance.
        """
        if self._hands is not None:
            return self._hands
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=self.config.model_complexity,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        return self._hands

    def close(self) -> None:
        """Release MediaPipe resources. Safe to call multiple times."""
        if self._hands is None:
            return
        try:
            self._hands.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ processing

    def process(self, frame: Any) -> DetectionResult:
        """Detect hand landmarks in a single frame.

        Errors raised during processing are caught and translated into a
        ``DetectionResult`` with ``landmarks=None`` so the main loop can
        continue without crashing.

        Args:
            frame: A :class:`src.models.Frame` or any object with an
                ``image`` attribute (numpy ndarray, BGR).

        Returns:
            :class:`DetectionResult` with detected landmarks and
            confidence, or with ``landmarks=None`` when no hand is visible.
        """
        timestamp = getattr(frame, "timestamp", 0.0)
        frame_id = getattr(frame, "frame_id", 0)
        image = getattr(frame, "image", frame)

        try:
            import cv2

            hands = self._ensure_hands()
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            mp_result = hands.process(rgb)
        except Exception as exc:
            _LOG.warning("MediaPipe processing error: %s", exc)
            return DetectionResult(
                landmarks=None, confidence=0.0, frame_id=frame_id, timestamp=timestamp
            )

        if mp_result is None or not getattr(mp_result, "multi_hand_landmarks", None):
            return DetectionResult(
                landmarks=None, confidence=0.0, frame_id=frame_id, timestamp=timestamp
            )

        hand_landmarks = mp_result.multi_hand_landmarks[0]
        handedness_list = getattr(mp_result, "multi_handedness", []) or []
        handedness = "Unknown"
        if handedness_list:
            try:
                handedness = handedness_list[0].classification[0].label
            except Exception:
                handedness = "Unknown"

        landmarks = [Landmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks.landmark]
        confidence = self._extract_confidence(handedness_list)
        return DetectionResult(
            landmarks=HandLandmarks(
                landmarks=landmarks, handedness=handedness, frame_id=frame_id
            ),
            confidence=confidence,
            frame_id=frame_id,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------ helpers

    def get_confidence(self, result: DetectionResult) -> float:
        """Return the tracking confidence for a detection result."""
        return float(result.confidence)

    def is_low_confidence(self, result: DetectionResult) -> bool:
        """Return ``True`` when the detection confidence is below the threshold."""
        return result.confidence < self.config.min_tracking_confidence

    @staticmethod
    def _extract_confidence(handedness_list: list[Any]) -> float:
        """Best-effort confidence extraction from MediaPipe handedness output."""
        if not handedness_list:
            return 0.0
        try:
            return float(handedness_list[0].classification[0].score)
        except Exception:
            return 0.0
