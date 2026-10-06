# MediaPipe API reference notes

These notes support the MediaPipe Development skill. Verify details against the linked official documentation when the installed MediaPipe version changes.

## Current Tasks API

- Hand landmark detection: `mp.tasks.vision.HandLandmarker`.
- Configuration: `mp.tasks.vision.HandLandmarkerOptions`.
- Running modes: `IMAGE`, `VIDEO`, and `LIVE_STREAM`.
- `IMAGE` uses `detect(image)`.
- `VIDEO` uses `detect_for_video(image, timestamp_ms)` and requires monotonically increasing timestamps.
- `LIVE_STREAM` uses `detect_async(image, timestamp_ms)` and a result callback; results may not be returned for every submitted frame.
- Results contain normalized image landmarks, handedness, and world landmarks.
- The hand landmark model exposes 21 canonical landmarks, indices 0 through 20.

Official references:

- https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarkerOptions
- https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker
- https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarkerResult
- https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/drawing_styles/hand_landmarker/HandLandmark
- https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/RunningMode

## Gesture Recognizer task

MediaPipe also provides `mp.tasks.vision.GestureRecognizer`. Its documented canned categories include `Closed_Fist`, `Open_Palm`, `Pointing_Up`, `Thumb_Down`, `Thumb_Up`, and `Victory`. Treat those labels as model output, not as a substitute for product-specific temporal state, safety gates, or action re-arming.

Official reference:

- https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/GestureRecognizerOptions

## Legacy compatibility

Some projects intentionally pin the legacy `mp.solutions.hands` API. Keep the version pin and wrapper contract stable when performing a targeted fix. If migrating, plan the change separately: the Tasks API changes construction, running modes, timestamps, callbacks, result shapes, and model assets.
