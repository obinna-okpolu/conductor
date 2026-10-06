"""Vision processing layer for Conductor.

Wraps MediaPipe hand-landmark detection and provides normalized geometric
features consumed by the gesture recognizer.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

from src.vision.detector import HandLandmarkDetector
from src.vision.features import FeatureExtractor

__all__ = ["FeatureExtractor", "HandLandmarkDetector"]
