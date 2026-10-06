# Conductor

## Control your browser with your hands

Conductor turns an ordinary webcam into a natural, futuristic browser controller. Hold up a Victory sign to enter navigation mode, move your hand to scroll or switch tabs, and use deliberate gestures to navigate browser history or close the active tab.

There are no wearable sensors, no special hardware, and no cloud service in the interaction loop. Conductor sees your hand locally, understands its movement, and translates that movement into browser actions in real time.

```text
Webcam → MediaPipe landmarks → normalized features → gesture recognition
       → temporal state machine → browser intent → action adapter
```

The result is a lightweight, privacy-first interface that feels closer to science fiction than a traditional desktop utility.

## Why Conductor is special

- **Hands-free browser control** — Scroll pages, change tabs, move through history, and close tabs without reaching for a mouse or keyboard.
- **Local by design** — Camera frames and hand landmarks are processed on your machine. No gesture data needs to leave the computer.
- **Designed for deliberate actions** — Gestures are stabilized over time, filtered through confidence checks, and protected by deadzones and re-arm behavior.
- **Distance-independent recognition** — Hand geometry is normalized relative to hand scale, making recognition more consistent as your hand moves closer to or farther from the camera.
- **Immediate visual feedback** — The HUD communicates the current interaction state and confirms browser actions as they happen.
- **Testable without a camera** — The vision pipeline, recognizer, state machine, intent resolver, and browser adapter are separated so they can be developed and verified independently.

## Gesture controls

| Gesture or movement | Result |
| --- | --- |
| Hold a Victory / peace sign | Enter navigation mode |
| Move your hand vertically in navigation mode | Scroll up or down |
| Move your hand horizontally in navigation mode | Switch to the previous or next tab |
| Hold an open palm while navigating | Release navigation mode |
| Hold an open palm while idle | Close the current browser tab |
| Perform a horizontal swipe while idle | Navigate backward or forward in browser history |

Navigation mode is intentionally modal: the Victory gesture establishes control, and hand movement then becomes navigation input. This keeps ordinary hand movement from unexpectedly controlling the browser.

## Architecture

Conductor is organized as a clean processing pipeline with explicit boundaries:

1. **Camera capture** obtains frames from the default webcam and handles unavailable or interrupted cameras safely.
2. **Hand landmark detection** uses MediaPipe Tasks to produce hand landmarks and tracking confidence locally.
3. **Feature extraction** converts raw coordinates into normalized distances, openness, position, and motion features.
4. **Gesture recognition** classifies poses and bounded movement sequences such as Victory, open palm, and horizontal swipes.
5. **Temporal state management** turns noisy frame-level candidates into stable gesture lifecycle events.
6. **Intent resolution** maps gesture events to semantic actions such as scroll, tab switching, history navigation, and tab close.
7. **Action dispatch** sends those intents through a browser adapter, keeping operating-system automation isolated and replaceable.

This separation makes Conductor easier to extend. New gestures can be added at the recognition layer without rewriting browser actions, and a different action adapter can be introduced without changing the vision pipeline.

## Requirements

- Python 3.12
- A webcam
- Windows, macOS, or Linux with a graphical desktop session
- A browser in the foreground when browser actions are being executed

MediaPipe, OpenCV, and PyAutoGUI are installed as project dependencies. Gesture processing itself runs locally.

## Installation

Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
```

The project can also be installed with any standard Python environment that satisfies the dependencies in `pyproject.toml`.

## Running Conductor

Start the application with:

```bash
uv run python main.py
```

Make sure the webcam is available and position your hand where it is clearly visible. The on-screen HUD displays the current state, recognized gesture feedback, and action confirmations.

Gesture thresholds and interaction timing can be customized in [`config/config.toml`](config/config.toml). This includes camera settings, confidence thresholds, swipe sensitivity, scroll deadzones, tab-switch thresholds, tracking-loss protection, and HUD feedback duration.

## Development and testing

Conductor is built to be developed without requiring a live camera for every change. The test suite uses mock landmarks, features, gesture candidates, events, and adapters to exercise the system deterministically.

Run the complete suite with:

```bash
uv run pytest
```

Run the property-based feature tests with:

```bash
uv run pytest tests/property
```

The project includes unit, integration, and property-based tests covering the camera boundary, MediaPipe detector integration, feature normalization, gesture recognition, temporal state transitions, intent resolution, browser dispatch, and the end-to-end pipeline.

## Building with Kiro

Kiro provides the structured development workflow used alongside the project. The specification files in [`.kiro/specs`](.kiro/specs) capture requirements, design decisions, and implementation tasks, making the system’s behavior easier to reason about as the gesture vocabulary evolves.

The repository also includes a reusable MediaPipe development power in [`powers/mediapipe-development`](powers/mediapipe-development). It packages practical guidance for:

- selecting the appropriate MediaPipe vision task;
- configuring live-stream hand landmark detection;
- handling timestamps, confidence thresholds, and result callbacks;
- separating landmark detection from feature extraction and gesture policy;
- testing camera-dependent code with deterministic landmark fixtures; and
- diagnosing lighting, tracking, coordinate, and temporal-stability problems.

The power follows Kiro’s plugin structure with a `plugin.json` manifest, a focused skill entry point, and API reference material. It can be developed locally and packaged as a portable Kiro power for reuse across future computer-vision projects.

## Project layout

```text
conductor/
├── config/                 Configuration models and TOML defaults
├── src/
│   ├── camera/             Webcam capture
│   ├── gestures/           Recognition and temporal state management
│   ├── interaction/        Intents, dispatch, and browser actions
│   ├── ui/                 Heads-up display
│   └── vision/             MediaPipe detection and feature extraction
├── tests/
│   ├── unit/               Component-level tests
│   ├── integration/        Pipeline tests
│   └── property/           Hypothesis-based invariant tests
├── .kiro/specs/            Requirements, design, and implementation plan
└── powers/                 Reusable Kiro development powers
```

## Vision for the project

Conductor is more than a gesture demo. It is a foundation for expressive, local-first computer interaction: a carefully layered system where perception becomes intent, intent becomes action, and every stage remains understandable, configurable, and testable.

The goal is simple: make the browser feel instantly responsive to the person using it. Raise your hand, move naturally, and let Conductor handle the rest.
