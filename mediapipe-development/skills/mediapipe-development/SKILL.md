---
name: mediapipe-development
description: Build and debug local MediaPipe hand-tracking and gesture systems with correct API selection, landmark geometry, temporal stability, camera diagnostics, and hardware-free tests.
---

# MediaPipe Development

Use this skill when a project detects hands, landmarks, poses, or gestures with MediaPipe, OpenCV, or a similar local vision pipeline.

## Operating principles

1. Read the repository's requirements, design notes, steering files, dependency manifest, and current vision pipeline before changing code.
2. Identify the MediaPipe API actually pinned by the project. Do not migrate an existing working integration during an unrelated gesture fix.
3. Keep the pipeline separated:

   `camera frame → detector → landmarks → normalized features → pose classifier → temporal state machine → intent → action adapter`

4. Make one reliable gesture/action path work before expanding the gesture vocabulary.
5. Never let landmark code invoke browser, OS, or other side effects directly.
6. Treat ambiguous pose evidence as `NONE` unless the product explicitly chooses a safe fallback. Never force an uncertain frame into an action-producing label.

## API and dependency check

Prefer the current MediaPipe Tasks API for new work:

- `mp.tasks.vision.HandLandmarker` with `HandLandmarkerOptions`.
- Choose `IMAGE`, `VIDEO`, or `LIVE_STREAM` deliberately.
- `detect_for_video` and `detect_async` require monotonically increasing timestamps in milliseconds.
- Live-stream mode is asynchronous and may drop input frames, so downstream state must tolerate missing results.

If an existing project pins the legacy `mediapipe<1.0` line and uses `mp.solutions.hands.Hands`, preserve that API unless migration is explicitly requested. Record the compatibility decision and test the exact installed version.

Do not infer that a MediaPipe confidence value is a universal gesture-confidence score. Keep detector confidence, handedness score, pose evidence, and action-confirmation state separate.

## Landmark correctness

Use the canonical 21 hand indices and name them in code rather than scattering magic numbers:

- Wrist `0`
- Thumb CMC/MCP/IP/tip `1/2/3/4`
- Index MCP/PIP/DIP/tip `5/6/7/8`
- Middle MCP/PIP/DIP/tip `9/10/11/12`
- Ring MCP/PIP/DIP/tip `13/14/15/16`
- Pinky MCP/PIP/DIP/tip `17/18/19/20`

MediaPipe normalized image coordinates use x/y in normalized image space; y increases downward. Confirm the project's mirroring and handedness convention before using x comparisons. A camera preview can be mirrored while detector input is not, so never assume the display orientation is the landmark orientation.

Prefer distances, vectors, joint angles, and hand-scale-normalized margins over a single raw `tip.y < pip.y` rule. If a project intentionally assumes a palm-facing camera posture, document that product constraint and make it visible in diagnostics rather than pretending the classifier is rotation invariant.

Use world landmarks only when their semantics are appropriate; they are skeletal 3D estimates, not measurements of finger-surface thickness.

## Temporal stability and action safety

Separate per-frame classification from lifecycle confirmation:

- The classifier reports stable observations, not browser actions.
- Hysteresis should prevent rapid label changes.
- A discrete action needs a hold or confirmation duration.
- Re-arm after an action before allowing the same action again.
- A continuous mode needs explicit engage, active, and release transitions.
- If one pose has multiple meanings, resolve it from explicit interaction context, not accidental ordering.

For every frame, preserve one timestamp through camera capture, detection, features, candidate, state machine, event, and intent. Do not stamp intermediate objects with `0.0` or mix wall-clock and monotonic values in one comparison. Validate stale-frame and tracking-jump behavior with tests.

Recommended state flow:

```text
raw landmarks
  → pose evidence and margins
  → smoothed candidate or NONE
  → state-machine confirmation
  → one-shot event with re-arm
  → semantic intent
  → side effect
```

When a project uses an open palm both to release a navigation mode and to close a tab, make the context rule explicit, for example: idle open palm closes after a hold; active navigation open palm releases navigation and does not close a tab.

## Camera and lighting diagnostics

When detection fails, distinguish these cases before tuning thresholds:

1. No frame or camera unavailable.
2. No hand landmarks.
3. Landmarks present but low detector confidence.
4. Landmarks stable but pose classifier says `NONE`.
5. Candidate present but state machine suppresses it.
6. Event present but intent/action dispatch fails.

Add a preview or structured log showing, at minimum, frame id, timestamp age, detector confidence, handedness, current candidate, state, suppression reason, and relevant pose margins. For a hand-pose classifier, expose evidence such as closed-finger count, thumb reach, vertical direction margin, and hand scale.

Practical capture guidance:

- Put diffuse light in front of or beside the camera, not behind the hand.
- Avoid strong backlight, glare, flicker, and a cluttered background.
- Keep the hand large enough in frame for stable landmarks.
- Do not “fix” bad lighting by lowering every safety threshold until false actions appear.

## Testing workflow

Before hardware testing:

1. Unit-test landmark index mapping and geometry with synthetic landmarks.
2. Test open, curled, thumb-up, thumb-down, and ambiguous poses.
3. Add jittered sequences that alternate around thresholds.
4. Test timestamp propagation and stale-frame suppression.
5. Test state-machine hold, release, re-arm, no-hand, low-confidence, and tracking-jump behavior.
6. Test intent resolution and action dispatch with mocks; do not send real keystrokes in unit tests.

During hardware testing, log evidence and state transitions rather than only the final label. Tune one threshold at a time and preserve a reproducible landmark sequence whenever a regression is found.

## Common failure modes

- **Everything is stale:** an intermediate candidate or feature object lost the camera timestamp.
- **Held gesture never engages:** a one-shot recognizer cooldown is hiding continuous observations from the state machine.
- **Thumb becomes fist:** thumb extension is inferred only from horizontal x-displacement, or vertical direction has no margin.
- **Fist and thumb flicker:** curled fingers use a brittle image-axis comparison and there is no ambiguous/dead-band result.
- **Actions repeat every frame:** the state machine lacks an edge trigger or re-arm state.
- **Good overlay label but no action:** inspect confidence/staleness/tracking suppression and the event-to-intent boundary.
- **Left and right hands disagree:** handedness, mirroring, and thumb direction conventions were mixed.
- **Tests pass but webcam fails:** synthetic landmarks do not resemble the actual camera geometry; capture a short diagnostic sequence and test it offline.

## Deliverable expectations

For a code change, report:

- Which MediaPipe API/version was preserved or selected and why.
- The exact data flow and timestamp contract.
- The pose evidence and temporal confirmation policy.
- Safety behavior for no hand, low confidence, stale frames, jumps, ambiguity, and re-arm.
- Hardware-free tests and the final test command/result.

Do not claim rotation invariance, confidence semantics, or action reliability unless the implementation and tests demonstrate them.
