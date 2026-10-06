"""Gesture recognition layer for Conductor.

The :class:`GestureRecognizer` consumes :class:`NormalizedFeatures` and
emits :class:`GestureCandidate` objects for the state machine to process.

Requirements: 4.1, 4.3, 8.1, 11.1
"""

from src.gestures.recognizer import GestureRecognizer

__all__ = ["GestureRecognizer"]
