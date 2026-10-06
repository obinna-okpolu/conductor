"""Conductor main entry point.

Always opens an OpenCV video window so you can see the camera feed, the
detected hand landmarks, and the live gesture state. Quit with ``q``.

Engineered gestures:
    Open Palm (0.5s while idle) -> Close tab
    Victory (0.5s)     -> Enter navigation mode
    Open Palm (0.5s)    -> Exit navigation mode
    Thumb Up (0.5s)    -> Browser back
    Thumb Down (0.5s)  -> Browser forward
    Swipe (any time)   -> Browser back / forward (fallback)

While in navigation mode, hand Y movement scrolls and hand X displacement
switches tabs.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import time
from typing import Any

import cv2
import numpy as np

from config.config import load_config
from src.actions.adapter import BrowserActionAdapter
from src.actions.dispatcher import ActionDispatcher
from src.camera.camera import Camera
from src.gestures.recognizer import GestureRecognizer
from src.interaction.resolver import IntentResolver
from src.interaction.state_machine import GestureStateMachine
from src.models import (
    BrowserIntent,
    GestureEvent,
    IntentType,
    NavModeState,
)
from src.ui.hud import HUD
from src.vision.detector import HandLandmarkDetector
from src.vision.features import FeatureExtractor


_LOG = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def build_pipeline(config_path: str | None = None) -> dict:
    """Construct every component with shared configuration."""
    config = load_config(config_path)

    camera = Camera(config.camera)
    detector = HandLandmarkDetector(config.detector)
    feature_extractor = FeatureExtractor()
    recognizer = GestureRecognizer(config.gesture, cooldown_frames=0)
    state_machine = GestureStateMachine(config.state_machine)
    resolver = IntentResolver(config.intent)
    adapter = BrowserActionAdapter()
    dispatcher = ActionDispatcher(adapter)

    return {
        "config": config,
        "camera": camera,
        "detector": detector,
        "features": feature_extractor,
        "recognizer": recognizer,
        "state_machine": state_machine,
        "resolver": resolver,
        "adapter": adapter,
        "dispatcher": dispatcher,
        "hud": HUD(config.hud),
    }


# ===========================================================================
# Overlay rendering
# ===========================================================================


_HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

_GESTURE_LABELS = {
    "NONE": "none",
    "OPEN_PALM": "open palm",
    "CLOSED_FIST": "legacy fist",
    "VICTORY": "V (peace)",
    "POINTING_UP": "point up",
    "THUMB_UP": "thumb up",
    "THUMB_DOWN": "thumb down",
    "SWIPE_LEFT": "swipe left",
    "SWIPE_RIGHT": "swipe right",
}

_INTENT_LABELS = {
    "SCROLL_UP": "SCROLL UP",
    "SCROLL_DOWN": "SCROLL DOWN",
    "NEXT_TAB": "NEXT TAB",
    "PREVIOUS_TAB": "PREVIOUS TAB",
    "BROWSER_FORWARD": "BROWSER FORWARD",
    "BROWSER_BACK": "BROWSER BACK",
    "CLOSE_TAB": "CLOSE TAB",
}


def _draw_landmarks(image_bgr: np.ndarray, landmarks: list[Any]) -> None:
    h, w = image_bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(image_bgr, pts[a], pts[b], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(image_bgr, (x, y), 4, (0, 0, 255), -1)


def _draw_lines(
    image: np.ndarray,
    lines: list[tuple[str, tuple]],
    origin: tuple[int, int] = (10, 30),
    step: int = 26,
    font_scale: float = 0.55,
) -> None:
    x, y = origin
    for text, color in lines:
        cv2.putText(
            image,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )
        y += step


# ===========================================================================
# Main loop with on-screen video
# ===========================================================================


def _hud_message_for(intent: BrowserIntent) -> str | None:
    return {
        IntentType.NEXT_TAB: "Next Tab",
        IntentType.PREVIOUS_TAB: "Previous Tab",
        IntentType.BROWSER_FORWARD: "Forward",
        IntentType.BROWSER_BACK: "Back",
        IntentType.CLOSE_TAB: "Tab Closed",
        IntentType.SCROLL_UP: "Scroll Up",
        IntentType.SCROLL_DOWN: "Scroll Down",
    }.get(intent.intent_type)


def _hud_state_for_event(event: GestureEvent) -> str | None:
    if event.event_type == "nav_engage":
        return "nav_active"
    if event.event_type == "nav_release":
        return "idle"
    return None


def run(pipeline: dict, max_frames: int | None = None) -> None:
    """Run the main loop with an OpenCV preview window."""
    camera: Camera = pipeline["camera"]
    detector: HandLandmarkDetector = pipeline["detector"]
    feature_extractor: FeatureExtractor = pipeline["features"]
    recognizer: GestureRecognizer = pipeline["recognizer"]
    state_machine: GestureStateMachine = pipeline["state_machine"]
    resolver: IntentResolver = pipeline["resolver"]
    dispatcher: ActionDispatcher = pipeline["dispatcher"]
    hud: HUD = pipeline["hud"]

    target_period = 1.0 / max(pipeline["config"].camera.fps, 1)
    frame_count = 0
    fps_t0 = time.monotonic()
    fps_counter = 0
    fps = 0.0

    last_intent_label = ""
    last_intent_expiry = 0.0
    last_event_label = ""
    last_event_expiry = 0.0
    last_classified = "NONE"

    window = "Conductor (press q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 720)

    try:
        camera.start()
        hud.start()
        _LOG.info("loop started; camera available=%s", camera.is_available())

        while True:
            if max_frames is not None and frame_count >= max_frames:
                break

            loop_start = time.monotonic()
            frame = camera.read_frame()
            fps_counter += 1
            if loop_start - fps_t0 > 1.0:
                fps = fps_counter / (loop_start - fps_t0)
                fps_counter = 0
                fps_t0 = loop_start

            if frame is None:
                blank = np.zeros((720, 960, 3), dtype=np.uint8)
                _draw_lines(
                    blank,
                    [
                        ("No camera frame", (0, 0, 255)),
                        (f"available={camera.is_available()}", (255, 255, 255)),
                        (f"frame {frame_count}", (255, 255, 255)),
                    ],
                )
                try:
                    cv2.imshow(window, blank)
                except Exception as exc:
                    _LOG.warning("imshow failed: %s", exc)
                if cv2.waitKey(50) & 0xFF == ord("q"):
                    break
                time.sleep(target_period)
                continue

            display = frame.image.copy()
            detection = detector.process(frame)

            if detection.landmarks is not None:
                try:
                    _draw_landmarks(display, detection.landmarks.landmarks)
                except Exception as exc:
                    _LOG.warning("draw landmarks failed: %s", exc)

                features = feature_extractor.extract(
                    detection.landmarks, timestamp=detection.timestamp
                )
                candidate = recognizer.recognize(
                    detection.landmarks,
                    detection.confidence,
                    timestamp=detection.timestamp,
                )
                candidate = dataclasses.replace(candidate, features=features)

                # Log every change of the per-frame gesture classification so
                # the user can see exactly what the recognizer is seeing —
                # not just the lifecycle events that fire after hold-duration.
                classified = candidate.gesture_type.name
                if classified != last_classified:
                    _LOG.info(
                        "classified: %s -> %s (conf=%.2f)",
                        last_classified,
                        classified,
                        detection.confidence,
                    )
                    last_classified = classified

                events = state_machine.process(candidate, detection)
                sm_state = state_machine.get_state()

                intents: list[BrowserIntent] = []
                for event in events:
                    last_event_label = event.event_type
                    last_event_expiry = time.monotonic() + 1.0
                    hud_state = _hud_state_for_event(event)
                    if hud_state is not None:
                        hud.update_state(hud_state)
                    _LOG.info("event: %s", event.event_type)
                    for intent in resolver.resolve(event):
                        intents.append(intent)

                if sm_state.nav_mode_state in (
                    NavModeState.ENGAGED,
                    NavModeState.ACTIVE,
                ):
                    for intent in resolver.update_nav_active(features):
                        intents.append(intent)

                for intent in intents:
                    try:
                        dispatcher.dispatch(intent)
                    except Exception as exc:
                        _LOG.warning("dispatch failed: %s", exc)
                    msg = _hud_message_for(intent)
                    if msg is not None:
                        hud.show_feedback(msg)
                    _LOG.info(
                        "FIRED intent: %s mag=%s",
                        intent.intent_type.name,
                        intent.magnitude,
                    )
                    last_intent_label = _INTENT_LABELS.get(
                        intent.intent_type.name, intent.intent_type.name
                    )
                    last_intent_expiry = time.monotonic() + 1.5

                gesture_label = _GESTURE_LABELS.get(
                    candidate.gesture_type.name, candidate.gesture_type.name
                )
                nav_color = (
                    (0, 255, 255)
                    if sm_state.nav_mode_state.name in ("ENGAGED", "ACTIVE")
                    else (255, 255, 255)
                )
                _draw_lines(
                    display,
                    [
                        (
                            f"gesture: {gesture_label}",
                            (0, 255, 0)
                            if candidate.gesture_type.name != "NONE"
                            else (200, 200, 200),
                        ),
                        (
                            f"hand: {detection.landmarks.handedness}  "
                            f"conf: {detection.confidence:.2f}",
                            (255, 255, 255),
                        ),
                        (f"nav: {sm_state.nav_mode_state.name}", nav_color),
                        (
                            f"close: {sm_state.fist_state.name}  "
                            f"swipe: {sm_state.swipe_state.name}",
                            (255, 255, 255),
                        ),
                        (
                            f"reason: {sm_state.suppression_reason or '-'}",
                            (200, 200, 200),
                        ),
                    ],
                )
                pose = recognizer.diagnose(detection.landmarks)
                _draw_lines(
                    display,
                    [
                        (
                            f"closed: {pose['closed_fingers']}/4  "
                            f"thumb reach: {pose['thumb_reach']:.1f}  "
                            f"vertical: {pose['thumb_vertical']:.1f}",
                            (180, 220, 255),
                        ),
                    ],
                    origin=(10, 185),
                    step=20,
                    font_scale=0.48,
                )
            else:
                state_machine.reset()
                resolver.reset()
                hud.update_state("idle")
                _draw_lines(
                    display,
                    [
                        ("no hand detected", (0, 0, 255)),
                    ],
                )

            now = time.monotonic()
            if last_intent_label and now < last_intent_expiry:
                cv2.rectangle(
                    display, (0, 0), (display.shape[1], 60), (0, 165, 255), -1
                )
                cv2.putText(
                    display,
                    f"FIRED: {last_intent_label}",
                    (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 0),
                    2,
                )
            if last_event_label and now < last_event_expiry:
                cv2.rectangle(
                    display,
                    (display.shape[1] - 280, 0),
                    (display.shape[1], 40),
                    (60, 60, 60),
                    -1,
                )
                cv2.putText(
                    display,
                    f"event: {last_event_label}",
                    (display.shape[1] - 270, 27),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
            cv2.putText(
                display,
                f"frame {frame_count}  fps {fps:.1f}  (press q to quit)",
                (10, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            try:
                cv2.imshow(window, display)
            except Exception as exc:
                _LOG.warning("imshow failed: %s", exc)
            try:
                hud.process_events()
            except Exception as exc:
                _LOG.warning("hud.process_events failed: %s", exc)

            # Periodic status line so the user can see what the system sees
            # even when the on-screen video window is hidden.
            if frame_count % 30 == 0 and detection.landmarks is not None:
                _LOG.info(
                    "frame %d fps=%.1f gesture=%s conf=%.2f "
                    "nav=%s close=%s swipe=%s reason=%s",
                    frame_count,
                    fps,
                    candidate.gesture_type.name,
                    detection.confidence,
                    sm_state.nav_mode_state.name,
                    sm_state.fist_state.name,
                    sm_state.swipe_state.name,
                    sm_state.suppression_reason or "-",
                )

            frame_count += 1

            elapsed = time.monotonic() - loop_start
            wait_ms = max(1, int((target_period - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        _LOG.info("interrupted by user after %d frames", frame_count)
    finally:
        camera.stop()
        detector.close()
        hud.stop()
        cv2.destroyAllWindows()


def main() -> None:
    _configure_logging()
    pipeline = build_pipeline()
    run(pipeline)


if __name__ == "__main__":
    main()
