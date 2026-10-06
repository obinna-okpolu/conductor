"""Camera capture module for Conductor.

Wraps OpenCV's :class:`cv2.VideoCapture` and exposes a small, testable
interface for the rest of the pipeline. The camera never inspects frame
data and never executes browser actions.

Requirements: 1.1, 1.2, 1.3, 1.4
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from config.config import CameraConfig
from src.models import Frame, FrameMetadata


class Camera:
    """Capture frames from the default webcam using OpenCV.

    The class hides ``cv2.VideoCapture`` behind a small, testable interface.
    Hardware errors are surfaced via :meth:`is_available` and ``None`` returns
    from :meth:`read_frame` rather than raising, so the main loop can recover
    gracefully.

    Attributes:
        config: Camera configuration (device index, target FPS).
        capture: Underlying ``cv2.VideoCapture`` (or compatible object).
        _frame_id: Monotonically increasing identifier for produced frames.
        _metadata: Cached stream metadata (resolution, fps).
        _started: Whether :meth:`start` has been called and the capture is open.
    """

    def __init__(self, config: CameraConfig, capture: Any | None = None) -> None:
        """Initialize the camera with configuration.

        Args:
            config: Camera configuration values.
            capture: Optional pre-built capture object. Used in tests to inject
                a mock. When ``None``, a real ``cv2.VideoCapture`` is created.
        """
        self.config = config
        self.capture: Any = (
            capture if capture is not None else cv2.VideoCapture(config.device_id)
        )
        self._frame_id: int = 0
        self._metadata: FrameMetadata | None = None
        self._started: bool = False

    # ------------------------------------------------------------------ start/stop

    def start(self) -> None:
        """Open the underlying capture and cache stream metadata.

        Idempotent — calling ``start`` on an already-started camera is a no-op.
        """
        if self._started:
            return
        if not self.capture.isOpened():
            # Best-effort open: some backends require explicit open() after construction.
            self.capture.open(self.config.device_id)
        self._refresh_metadata()
        self._started = True

    def stop(self) -> None:
        """Release the underlying capture.

        Safe to call when the camera was never started.
        """
        if self._started:
            try:
                self.capture.release()
            except Exception:
                # Release should never raise, but guard anyway so callers can
                # use stop() in finally blocks.
                pass
        self._started = False
        self._metadata = None

    # ------------------------------------------------------------------ capture

    def read_frame(self) -> Frame | None:
        """Read a single frame from the camera.

        Returns:
            A :class:`Frame` on success, or ``None`` when the camera is
            unavailable or the underlying read fails.
        """
        if not self._started or not self.is_available():
            return None

        try:
            ok, image = self.capture.read()
        except Exception:
            return None

        if not ok or image is None:
            return None

        self._frame_id += 1
        return Frame(image=image, timestamp=time.time(), frame_id=self._frame_id)

    # ------------------------------------------------------------------ metadata

    def get_metadata(self) -> FrameMetadata:
        """Return cached stream metadata.

        The metadata is captured on :meth:`start`. If start has not been
        called, returns a default zeroed metadata object.

        Returns:
            :class:`FrameMetadata` with width, height, and fps.
        """
        if self._metadata is None:
            self._refresh_metadata()
        if self._metadata is None:
            # Camera refused to give us any metadata; return safe defaults.
            return FrameMetadata(width=0, height=0, fps=float(self.config.fps))
        return self._metadata

    def is_available(self) -> bool:
        """Return ``True`` if the camera is open and ready to read.

        Once :meth:`stop` has been called the camera is considered
        unavailable even if the underlying backend still reports opened.

        Returns:
            Whether :attr:`capture` is opened and :meth:`start` has run.
        """
        if not self._started:
            return False
        try:
            return bool(self.capture.isOpened())
        except Exception:
            return False

    # ------------------------------------------------------------------ helpers

    def _refresh_metadata(self) -> None:
        """Cache width/height/fps from the underlying capture."""
        try:
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            width, height, fps = 0, 0, 0.0

        # Fall back to configured value if the backend reports 0.
        if fps <= 0.0:
            fps = float(self.config.fps)
        self._metadata = FrameMetadata(width=width, height=height, fps=fps)


def make_blank_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Return a blank BGR frame.

    Helper for tests and offline tooling that need a frame of the right shape
    without touching a real camera.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        A ``(height, width, 3)`` uint8 BGR image filled with zeros.
    """
    return np.zeros((height, width, 3), dtype=np.uint8)
