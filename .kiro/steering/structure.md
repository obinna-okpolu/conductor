# Project Structure

This document defines the canonical project structure for Conductor, aligning with the architectural principles and separation of concerns defined in the design specification.

## Directory Structure

```
conductor/
├── config/                     # Configuration system
│   ├── __init__.py
│   ├── config.py              # Configuration dataclasses and loader
│   └── config.toml            # Default configuration values
│
├── src/                       # Core application code
│   ├── models.py              # Shared data models (Frame, Landmarks, Features, Events, Intents)
│   │
│   ├── camera/                # Camera module (vision input layer)
│   │   ├── __init__.py
│   │   └── camera.py          # Frame capture from webcam
│   │
│   ├── vision/                # Vision processing layer
│   │   ├── __init__.py
│   │   ├── detector.py        # HandLandmarkDetector - MediaPipe wrapper
│   │   └── features.py        # FeatureExtractor - normalized geometric features
│   │
│   ├── gestures/              # Gesture recognition layer
│   │   ├── __init__.py
│   │   └── recognizer.py      # GestureRecognizer - pose, swipe, and thumb detection
│   │
│   ├── interaction/           # Interaction and state management layer
│   │   ├── __init__.py
│   │   ├── state_machine.py   # GestureStateMachine - lifecycle tracking
│   │   └── resolver.py        # IntentResolver - event-to-intent mapping
│   │
│   ├── actions/               # Action execution layer (strictly separated from vision)
│   │   ├── __init__.py
│   │   ├── adapter.py         # BrowserActionAdapter - PyAutoGUI integration
│   │   └── dispatcher.py      # ActionDispatcher - single entry point for actions
│   │
│   └── ui/                    # User interface layer
│       ├── __init__.py
│       └── hud.py             # HUD - floating status display
│
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── property/              # Property-based tests (Hypothesis)
│   ├── integration/           # Integration tests
│   └── conftest.py            # Shared fixtures
│
├── main.py                    # Application entry point
├── pyproject.toml             # Project configuration and dependencies
└── README.md
```

## Separation of Concerns

### Layer Isolation Principles

1. **Vision Layer** (`src/camera/`, `src/vision/`, `src/gestures/`)
   - Captures and processes visual input
   - Detects hand landmarks and extracts features
   - Recognizes gesture candidates
   - **MUST NOT** execute any browser or OS actions
   - **MUST NOT** import from `src/actions/`

2. **Interaction Layer** (`src/interaction/`)
   - Tracks gesture lifecycle and state
   - Maps gesture events to semantic intents
   - **MUST NOT** execute browser actions
   - Communicates with actions layer only through `BrowserIntent` dataclass

3. **Action Layer** (`src/actions/`)
   - Single responsibility: execute browser actions
   - `ActionDispatcher` is the ONLY component that invokes `BrowserActionAdapter`
   - **MUST NOT** be called by vision or gesture recognition code
   - No knowledge of gesture detection internals

4. **UI Layer** (`src/ui/`)
   - Subscribes to events and intents for display
   - **MUST NOT** execute browser actions
   - **MUST NOT** block the main event loop

### Data Flow

```
Camera (Frame)
  → HandLandmarkDetector (DetectionResult)
    → FeatureExtractor (NormalizedFeatures)
      → GestureRecognizer (GestureCandidate)
        → GestureStateMachine (GestureEvent)
          → IntentResolver (BrowserIntent)
            → ActionDispatcher
              → BrowserActionAdapter (side effects)
```

### Shared Data Models

All data types that cross module boundaries are defined in `src/models.py`:

- **Camera Layer**: `Frame`, `FrameMetadata`
- **Vision Layer**: `Landmark`, `HandLandmarks`, `DetectionResult`, `NormalizedFeatures`
- **Gesture Layer**: `GestureType`, `GestureCandidate`, `SwipeMetadata`
- **State Machine Layer**: `NavModeState`, close-gesture lifecycle state, `SwipeState`, `GestureEvent`, `StateMachineState`
- **Intent Layer**: `IntentType`, `BrowserIntent`

This centralization ensures:
- Clear interfaces between modules
- No circular dependencies
- Type safety across the pipeline
- Easy mocking for tests

### Configuration

All configuration dataclasses are in `config/config.py`:
- `CameraConfig`
- `DetectorConfig`
- `GestureConfig`
- `StateMachineConfig`
- `IntentConfig`
- `HUDConfig`
- `Config` (root)

Configuration is loaded once at startup and passed to component constructors.

## Import Rules

### Allowed Import Patterns

```python
# Vision layer can import from models
from src.models import Frame, DetectionResult, NormalizedFeatures

# Gesture recognizer can import from models
from src.models import GestureCandidate, NormalizedFeatures

# State machine can import from models and config
from src.models import GestureEvent, GestureCandidate
from config.config import StateMachineConfig

# Intent resolver can import from models
from src.models import BrowserIntent, GestureEvent

# Action dispatcher can import from adapter and models
from src.actions.adapter import BrowserActionAdapter
from src.models import BrowserIntent

# Main can import from all layers
from src.camera.camera import Camera
from src.vision.detector import HandLandmarkDetector
from src.gestures.recognizer import GestureRecognizer
# etc.
```

### Forbidden Import Patterns

```python
# Vision layer MUST NOT import from actions
from src.actions.adapter import BrowserActionAdapter  # ❌ FORBIDDEN

# Actions layer MUST NOT import from vision/gestures
from src.vision.detector import HandLandmarkDetector  # ❌ FORBIDDEN
from src.gestures.recognizer import GestureRecognizer  # ❌ FORBIDDEN

# UI MUST NOT import from actions (except to subscribe to events)
from src.actions.dispatcher import (
    ActionDispatcher,
)  # ❌ FORBIDDEN (UI should not dispatch)
```

## Test Organization

Each module has corresponding test files:

- `tests/unit/test_feature_extractor.py` → `src/vision/features.py`
- `tests/unit/test_gesture_recognizer.py` → `src/gestures/recognizer.py`
- `tests/unit/test_state_machine.py` → `src/interaction/state_machine.py`
- `tests/unit/test_intent_resolver.py` → `src/interaction/resolver.py`
- `tests/unit/test_adapter.py` → `src/actions/adapter.py`
- `tests/unit/test_dispatcher.py` → `src/actions/dispatcher.py`

Property-based tests validate correctness properties:

- `tests/property/test_normalization_properties.py` → FeatureExtractor
- `tests/property/test_threshold_properties.py` → GestureRecognizer thresholds
- `tests/property/test_state_machine_properties.py` → State transitions
- `tests/property/test_intent_properties.py` → Intent mapping correctness

Integration tests validate cross-module behavior:

- `tests/integration/test_pipeline.py` → Full frame-to-intent flow
- `tests/integration/test_action_dispatch.py` → Intent to adapter execution

## Module Responsibilities Summary

| Module | Responsibility | Key Outputs |
|--------|---------------|-------------|
| `src/camera/` | Frame capture | `Frame`, `FrameMetadata` |
| `src/vision/detector.py` | MediaPipe landmark detection | `DetectionResult` |
| `src/vision/features.py` | Normalized feature extraction | `NormalizedFeatures` |
| `src/gestures/` | Gesture candidate classification | `GestureCandidate` |
| `src/interaction/state_machine.py` | Gesture lifecycle tracking | `GestureEvent` |
| `src/interaction/resolver.py` | Intent resolution | `BrowserIntent` |
| `src/actions/dispatcher.py` | Action dispatch | Calls to adapter |
| `src/actions/adapter.py` | Browser action execution | PyAutoGUI calls |
| `src/ui/` | Visual feedback | HUD display |
| `src/models.py` | Shared data types | All dataclasses |
| `config/` | Configuration | `Config` dataclass |
