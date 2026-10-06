"""Unit tests for the Camera module.

Requirements: 1.3, 1.4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from config.config import CameraConfig
from src.camera.camera import Camera


@dataclass
class _StubCapture:
    """Drop-in replacement for ``cv2.VideoCapture`` used in unit tests.

    Records call counts and lets each test control open/read behaviour
    deterministically without any real hardware.
    """

    opened: bool = False
    raise_on_read: bool = False
    return_value: bool = True
    next_image: np.ndarray | None = None
    width: int = 1280
    height: int = 720
    fps: float = 60.0

    open_calls: int = 0
    release_calls: int = 0
    read_calls: int = 0

    def open(self, _device_id: int = 0) -> bool:
        self.open_calls += 1
        self.opened = True
        return True

    def isOpened(self) -> bool:
        return self.opened

    def release(self) -> None:
        self.release_calls += 1
        self.opened = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_calls += 1
        if self.raise_on_read:
            raise RuntimeError("simulated capture failure")
        return self.return_value, self.next_image

    def get(self, prop: int) -> float:
        # cv2.CAP_PROP_FRAME_WIDTH == 3, CAP_PROP_FRAME_HEIGHT == 4, CAP_PROP_FPS == 5
        if prop in (3,):
            return float(self.width)
        if prop in (4,):
            return float(self.height)
        if prop in (5,):
            return float(self.fps)
        return 0.0


# ----------------------------------------------------------------------- happy path


def test_start_opens_capture_and_caches_metadata() -> None:
    capture = _StubCapture()
    camera = Camera(CameraConfig(device_id=0, fps=30), capture=capture)

    camera.start()

    assert capture.open_calls == 1
    assert camera.is_available() is True
    metadata = camera.get_metadata()
    assert metadata.width == 1280
    assert metadata.height == 720
    assert metadata.fps == pytest.approx(60.0)

    camera.stop()


def test_read_frame_returns_frame_with_increasing_ids() -> None:
    capture = _StubCapture(next_image=np.zeros((480, 640, 3), dtype=np.uint8))
    camera = Camera(CameraConfig(device_id=0, fps=30), capture=capture)
    camera.start()

    first = camera.read_frame()
    second = camera.read_frame()

    assert first is not None
    assert second is not None
    assert first.frame_id == 1
    assert second.frame_id == 2
    assert capture.read_calls == 2
    assert first.timestamp <= second.timestamp

    camera.stop()


def test_start_is_idempotent() -> None:
    capture = _StubCapture()
    camera = Camera(CameraConfig(device_id=0, fps=30), capture=capture)

    camera.start()
    camera.start()
    camera.start()

    assert capture.open_calls == 1
    camera.stop()


def test_stop_releases_capture_and_is_safe_to_call_twice() -> None:
    capture = _StubCapture()
    camera = Camera(CameraConfig(device_id=0, fps=30), capture=capture)
    camera.start()

    camera.stop()
    camera.stop()

    assert capture.release_calls == 1
    assert camera.is_available() is False


# ----------------------------------------------------------------------- failure paths


def test_read_frame_returns_none_when_capture_unavailable() -> None:
    capture = _StubCapture(opened=False)
    camera = Camera(CameraConfig(device_id=0, fps=30), capture=capture)
    camera.start()  # start() will call open(), making it available

    # Simulate the camera becoming unavailable mid-stream.
    capture.opened = False

    assert camera.read_frame() is None


def test_read_frame_returns_none_when_read_returns_false() -> None:
    capture = _StubCapture(return_value=False, next_image=None)
    camera = Camera(CameraConfig(device_id=0, fps=30), capture=capture)
    camera.start()

    assert camera.read_frame() is None

    camera.stop()


def test_read_frame_returns_none_when_read_raises() -> None:
    capture = _StubCapture(raise_on_read=True)
    camera = Camera(CameraConfig(device_id=0, fps=30), capture=capture)
    camera.start()

    assert camera.read_frame() is None

    camera.stop()


def test_camera_reports_unavailable_when_capture_closed() -> None:
    capture = _StubCapture()
    camera = Camera(CameraConfig(device_id=0, fps=30), capture=capture)
    camera.start()
    capture.opened = False

    assert camera.is_available() is False
    assert camera.read_frame() is None


def test_camera_recovers_when_capture_reopened() -> None:
    """Recovery scenario from design.md — polling is_available recovers."""
    capture = _StubCapture(next_image=np.zeros((480, 640, 3), dtype=np.uint8))
    camera = Camera(CameraConfig(device_id=0, fps=30), capture=capture)
    camera.start()

    # Camera drops out.
    capture.opened = False
    assert camera.read_frame() is None

    # Camera recovers.
    capture.opened = True
    frame = camera.read_frame()
    assert frame is not None
    assert frame.frame_id == 1

    camera.stop()


def test_is_available_swallows_exceptions() -> None:
    class ExplodingCapture(_StubCapture):
        def isOpened(self) -> bool:  # type: ignore[override]
            raise RuntimeError("boom")

    camera = Camera(CameraConfig(device_id=0, fps=30), capture=ExplodingCapture())
    assert camera.is_available() is False


def test_get_metadata_uses_configured_fps_when_backend_reports_zero() -> None:
    class ZeroFpsCapture(_StubCapture):
        def get(self, prop: int) -> float:  # type: ignore[override]
            if prop == 5:  # CAP_PROP_FPS
                return 0.0
            return super().get(prop)

    capture = ZeroFpsCapture(fps=0.0)
    camera = Camera(CameraConfig(device_id=0, fps=24), capture=capture)
    camera.start()

    metadata = camera.get_metadata()
    assert metadata.fps == pytest.approx(24.0)

    camera.stop()


def test_stop_handles_release_exceptions() -> None:
    class ExplodingCapture(_StubCapture):
        def release(self) -> None:  # type: ignore[override]
            raise RuntimeError("boom")

    capture = ExplodingCapture()
    camera = Camera(CameraConfig(device_id=0, fps=30), capture=capture)
    camera.start()

    # Should not raise even though release() throws.
    camera.stop()
    assert camera.is_available() is False
