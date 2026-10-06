"""Camera module for capturing frames from the default webcam.

This module provides the :class:`Camera` class, which wraps OpenCV's
``VideoCapture`` to expose a simple, testable interface for downstream
vision processing.

Requirements: 1.1, 1.2, 1.3, 1.4
"""

from src.camera.camera import Camera

__all__ = ["Camera"]
