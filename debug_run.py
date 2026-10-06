"""Standalone debug runner that captures full per-frame diagnostics.

Run this directly (no harness timeout) and pipe output to a file:

    uv run python debug_run.py > debug.log 2>&1

Or on Windows PowerShell:
    uv run python .\debug_run.py *> .\debug.log\n

Then share the contents of ``debug.log`` so we can see exactly what the
pipeline sees frame-by-frame.
"""

from __future__ import annotations

import logging
import sys
import time

import main as conductor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("debug")


def main() -> int:
    pipeline = conductor.build_pipeline()
    # Loosen confidence threshold so MediaPipe's typical 0.3-0.7 scores
    # don't suppress everything while we're debugging.
    pipeline["config"].state_machine.confidence_threshold = 0.3

    camera = pipeline["camera"]
    detector = pipeline["detector"]
    extractor = pipeline["features"]
    recognizer = pipeline["recognizer"]
    sm = pipeline["state_machine"]
    resolver = pipeline["resolver"]
    hud = pipeline["hud"]
    dispatcher = pipeline["dispatcher"]

    camera.start()
    hud.start()

    frame_count = 0
    try:
        while True:
            t0 = time.monotonic()
            frame = camera.read_frame()
            if frame is None:
                if frame_count % 30 == 0:
                    log.info(
                        "frame %d: camera returned None; available=%s",
                        frame_count,
                        camera.is_available(),
                    )
                if not camera.is_available():
                    log.warning("camera became unavailable")
                    sm.reset()
                    resolver.reset()
                time.sleep(0.05)
                continue

            detection = detector.process(frame)
            if detection.landmarks is None:
                sm.reset()
                resolver.reset()
                hud.update_state("idle")
                if frame_count % 30 == 0:
                    log.info("frame %d: no hand detected", frame_count)
                hud.process_events()
                frame_count += 1
                continue

            features = extractor.extract(detection.landmarks)
            features_dict = features.__class__(
                pinch_distance=features.pinch_distance,
                hand_center=features.hand_center,
                finger_extensions=features.finger_extensions,
                openness_score=features.openness_score,
                hand_scale=features.hand_scale,
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
            )

            candidate = recognizer.recognize(features_dict, detection.confidence)
            if candidate.gesture_type.name == "CLUTCH_ENGAGE":
                recognizer.reset_swipe_buffer()

            events = sm.process(candidate, detection)
            s = sm.get_state()

            log.info(
                "frame %d: hand=%s conf=%.2f pinch=%.3f openness=%.2f "
                "candidate=%s clutch=%s fist=%s swipe=%s reason=%s events=%d",
                frame_count,
                detection.landmarks.handedness,
                detection.confidence,
                features.pinch_distance,
                features.openness_score,
                candidate.gesture_type.name,
                s.clutch_state.name,
                s.fist_state.name,
                s.swipe_state.name,
                s.suppression_reason or "-",
                len(events),
            )
            for event in events:
                log.info("  event: %s pos=%s", event.event_type, event.hand_position)

            intents = []
            for event in events:
                intents.extend(resolver.resolve(event, features_dict))
                if event.event_type == "clutch_engage":
                    hud.update_state("clutch_engaged")
                elif event.event_type == "clutch_release":
                    hud.update_state("idle")

            if s.clutch_state.name in ("ENGAGED", "ACTIVE"):
                intents.extend(resolver.update_clutch_active(features_dict))

            for intent in intents:
                dispatcher.dispatch(intent)
                log.info(
                    "  intent: %s mag=%s", intent.intent_type.name, intent.magnitude
                )

            hud.process_events()
            frame_count += 1
            elapsed = time.monotonic() - t0
            if elapsed < 0.033:
                time.sleep(0.033 - elapsed)
    except KeyboardInterrupt:
        log.info("interrupted after %d frames", frame_count)
    finally:
        camera.stop()
        detector.close()
        hud.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
