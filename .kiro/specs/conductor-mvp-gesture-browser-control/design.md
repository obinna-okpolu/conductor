# Design Document: Conductor MVP Gesture Browser Control

## Overview

Conductor MVP is a local, privacy-first gesture interface for browser navigation. This document describes the technical design for the complete pipeline from camera frame capture through gesture recognition to browser action execution.

### Design Goals

1. **Safety**: Vision and gesture-recognition components never directly execute browser actions
2. **Testability**: All gesture logic, state machines, and intent resolution can be tested without physical hardware
3. **Low Latency**: Real-time gesture processing with minimal delay
4. **Predictability**: Clear state transitions with hysteresis and re-arm mechanisms to prevent accidental actions
5. **Normalization**: Hand measurements normalized by hand scale for camera-distance independence

### Technology Stack

- **Python 3.12**
- **OpenCV (cv2)**: Webcam frame capture
- **MediaPipe**: Hand landmark detection (runs fully locally)
- **Tkinter**: Floating HUD overlay (standard library)
- **PyAutoGUI**: Browser action adapter
- **pytest**: Unit and integration tests
- **Hypothesis**: Property-based tests

---

## Architecture

### Pipeline Overview

```
┌─────────────┐     ┌──────────────────────┐     ┌────────────────────┐
│   Camera    │────▶│  Hand_Landmark_      │────▶│  Feature_          │
│   Module    │     │  Detector            │     │  Extractor         │
└─────────────┘     └──────────────────────┘     └────────────────────┘
                            │                            │
                            │ Frame_Confidence           │ Normalized_Features
                            │ Hand_Landmarks             │
                            ▼                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Gesture_Recognizer                           │
│  - Victory/open-palm pose detection                                  │
│  - Swipe detection (temporal analysis)                              │
│  - Open-palm pose detection and thumb direction                     │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            │ Gesture_Candidates
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Gesture_State_Machine                            │
│  - Navigation lifecycle (IDLE → ENGAGED → ACTIVE → IDLE)             │
│  - Swipe validation and re-arm                                      │
│  - Idle open-palm close confirmation and re-arm                     │
│  - Safety suppression (low confidence, tracking loss, staleness)    │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            │ Gesture_Events
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Intent_Resolver                              │
│  - Scroll intent generation (incremental movement)                  │
│  - Tab-switch intent generation (with re-arm)                       │
│  - Navigation intent mapping                                        │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            │ Browser_Intents
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Action_Dispatcher                              │
│  - Single entry point for all browser actions                       │
│  - Invokes Browser_Action_Adapter                                   │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Browser_Action_Adapter                           │
│  - PyAutoGUI keyboard shortcuts                                     │
│  - Platform-specific key combinations                               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                             HUD                                      │
│  - Subscribes to Gesture_Events and Browser_Intents                 │
│  - Displays current state and action feedback                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Architectural Principles

1. **Layer Separation**: Vision layer never executes browser/OS actions
2. **Single Responsibility**: Each component has one clear purpose
3. **Explicit Interfaces**: Components communicate through well-defined data types
4. **Safety Boundaries**: Action_Dispatcher is the only component that invokes Browser_Action_Adapter

---

## Components and Interfaces

### 1. Camera Module

**Location**: `src/camera/`

**Responsibility**: Capture frames from the default webcam and provide them to downstream components.

**Inputs**: None (hardware interface)

**Outputs**:
- `Frame`: Raw BGR image from webcam
- `FrameMetadata`: Timestamp, frame number, resolution

**State**:
- Camera handle (OpenCV VideoCapture)
- Frame counter
- Error state

**Dependencies**: OpenCV

**Must NOT**:
- Process or analyze frame contents
- Know about gestures or browser actions
- Block on frame capture indefinitely

**Interface**:
```python
@dataclass
class Frame:
    image: np.ndarray  # BGR format
    timestamp: float  # seconds since epoch
    frame_id: int  # monotonically increasing


@dataclass
class FrameMetadata:
    width: int
    height: int
    fps: float


class Camera:
    def __init__(self, config: CameraConfig) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read_frame(self) -> Frame | None: ...
    def get_metadata(self) -> FrameMetadata: ...
    def is_available(self) -> bool: ...
```

**Error Handling**:
- Camera unavailable: `read_frame()` returns `None`, `is_available()` returns `False`
- Capture failure during operation: Log error, signal downstream to pause processing
- Recovery: Poll for camera availability, resume when recovered

---

### 2. Hand Landmark Detector

**Location**: `src/vision/`

**Responsibility**: Process camera frames through MediaPipe to detect hand landmarks and compute confidence.

**Inputs**:
- `Frame`: Raw BGR image

**Outputs**:
- `HandLandmarks`: 21 landmark positions in normalized coordinates [0, 1]
- `FrameConfidence`: Float [0, 1] indicating tracking quality
- `NoHandSignal`: Special value when no hand detected

**State**:
- MediaPipe Hands instance
- Previous valid landmarks (for smoothing/jump detection)

**Dependencies**: MediaPipe

**Must NOT**:
- Classify gestures
- Execute browser actions
- Make decisions about gesture intent

**Interface**:
```python
@dataclass
class Landmark:
    x: float  # Normalized [0, 1]
    y: float  # Normalized [0, 1]
    z: float  # Depth relative to wrist


@dataclass
class HandLandmarks:
    landmarks: list[Landmark]  # 21 landmarks
    handedness: str  # "Left" or "Right"
    frame_id: int


@dataclass
class DetectionResult:
    landmarks: HandLandmarks | None
    confidence: float
    frame_id: int
    timestamp: float


class HandLandmarkDetector:
    def __init__(self, config: DetectorConfig) -> None: ...
    def process(self, frame: Frame) -> DetectionResult: ...
    def get_confidence(self, result: DetectionResult) -> float: ...
```

**MediaPipe Landmark Indices**:
```
0: WRIST
1-4: THUMB (CMC, MCP, IP, TIP)
5-8: INDEX (MCP, PIP, DIP, TIP)
9-12: MIDDLE (MCP, PIP, DIP, TIP)
13-16: RING (MCP, PIP, DIP, TIP)
17-20: PINKY (MCP, PIP, DIP, TIP)
```

---

### 3. Feature Extractor

**Location**: `src/vision/`

**Responsibility**: Compute normalized geometric features from hand landmarks.

**Inputs**:
- `HandLandmarks`: Raw landmark positions

**Outputs**:
- `NormalizedFeatures`: Scale-independent geometric measurements

**State**: None (pure function)

**Dependencies**: None

**Must NOT**:
- Classify gestures
- Maintain temporal state
- Access camera or browser

**Interface**:
```python
@dataclass
class NormalizedFeatures:
    # Pinch measurement
    pinch_distance: float  # Normalized [0, 1], thumb tip to middle finger tip

    # Hand position (for tracking movement)
    hand_center: tuple[float, float]  # Normalized x, y

    # Finger extension state
    finger_extensions: dict[str, float]  # Per-finger extension [0, 1]
    openness_score: float  # Overall hand openness [0, 1]

    # Hand scale reference
    hand_scale: float  # Reference size for normalization

    # Metadata
    frame_id: int
    timestamp: float


class FeatureExtractor:
    def __init__(self, config: FeatureConfig) -> None: ...
    def extract(self, landmarks: HandLandmarks) -> NormalizedFeatures: ...
    def compute_hand_scale(self, landmarks: HandLandmarks) -> float: ...
```

**Normalization Strategy**:
- **Hand Scale Reference**: Computed as the maximum distance from wrist to any fingertip, or the palm bounding box diagonal
- **Distance Normalization**: All distances divided by hand scale
- **Position Normalization**: Landmark positions already normalized by MediaPipe to [0, 1] frame coordinates
- **Movement Normalization**: Displacements computed in normalized space, then scaled by hand scale

**Key Feature Computations**:
- `pinch_distance`: distance between thumb tip (landmark 4) and middle finger tip (landmark 12), divided by hand scale.
- `hand_center`: centroid of the hand landmarks, expressed in MediaPipe's normalized frame coordinates.
- `finger_extension`: compute normalized finger-joint geometry for each finger to estimate how extended or curled the finger is relative to the palm. Do not rely on a fixed image-coordinate Y comparison such as `TIP_Y > MCP_Y`.
- `openness_score`: aggregate the normalized finger-extension measurements to represent overall hand openness.

Finger geometry SHALL be derived from hand-scale-normalized measurements. The live MVP assumes the palm is broadly facing the webcam; closed-fist poses are not part of the emitted gesture vocabulary.

---

### 4. Gesture Recognizer

**Location**: `src/gestures/`

**Responsibility**: Classify gesture types from normalized features, producing gesture candidates.

**Inputs**:
- `NormalizedFeatures`: Geometric measurements
- Previous features (for temporal analysis)

**Outputs**:
- `GestureCandidate`: Potential gesture with type, confidence, and supporting data

**State**:
- Recent feature history for swipe detection
- Previous pinch state for hysteresis

**Dependencies**: Configuration

**Must NOT**:
- Track gesture lifecycle (that's the state machine)
- Emit final gesture events
- Execute browser actions
- Maintain engage/active/release state

**Interface**:
```python
@enum.unique
class GestureType(enum.Enum):
    NONE = "none"
    VICTORY = "victory"
    POINTING_UP = "pointing_up"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    OPEN_PALM = "open_palm"
    THUMB_UP = "thumb_up"
    THUMB_DOWN = "thumb_down"


@dataclass
class GestureCandidate:
    gesture_type: GestureType
    confidence: float
    frame_id: int
    timestamp: float
    features: NormalizedFeatures
    swipe_metadata: SwipeMetadata | None  # For swipe candidates


@dataclass
class SwipeMetadata:
    net_displacement: float
    elapsed_time: float
    velocity: float
    direction_consistency: float


class GestureRecognizer:
    def __init__(self, config: GestureConfig) -> None: ...
    def recognize(
        self, landmarks: HandLandmarks, confidence: float, timestamp: float = 0.0
    ) -> GestureCandidate: ...
    def reset_swipe_buffer(self) -> None: ...
```

**Recognition Logic**:

**Pose Detection** (with temporal stability):
```
classified = classify_landmark_pose(landmarks)
candidate = stable_pose_after_hysteresis(classified)
```

**Swipe Detection** (temporal analysis):
```
buffer = bounded_frame_buffer(max_frames=swipe_window_size)

if navigation_mode_not_engaged:
    buffer.append(features)
    
    net_displacement = buffer[-1].hand_center.x - buffer[0].hand_center.x
    elapsed_time = buffer[-1].timestamp - buffer[0].timestamp
    velocity = net_displacement / elapsed_time if elapsed_time > 0 else 0
    direction_consistency = compute_consistency(buffer)
    
    if (abs(net_displacement) > swipe_displacement_threshold and
        abs(velocity) > swipe_velocity_threshold and
        direction_consistency > direction_consistency_threshold):
        candidate = SWIPE_LEFT if net_displacement < 0 else SWIPE_RIGHT
```

**Open-Palm Close and Thumb Detection**:
The live recognizer uses the palm-facing webcam posture. Open-palm detection requires all five fingers to be extended using normalized hand-scale measurements. Closed-fist poses are intentionally not emitted as live candidates.

When the four non-thumb fingers are curled, a thumb candidate is emitted only when the thumb has sufficient normalized reach and a clear vertical margin relative to the wrist. Ambiguous closed-hand poses resolve to `NONE` rather than producing a close or navigation candidate. The state machine gives an idle open palm the close-tab meaning and reserves open palm in navigation mode for navigation release.

---

### 5. Gesture State Machine

**Location**: `src/interaction/`

**Responsibility**: Track gesture lifecycle and produce gesture events. Implement safety suppression.

**Inputs**:
- `GestureCandidate`: Potential gestures from recognizer
- `DetectionResult`: Confidence and tracking status

**Outputs**:
- `GestureEvent`: Confirmed gesture lifecycle transitions

**State**:
- Current navigation-mode state (IDLE, ENGAGED, ACTIVE)
- Current swipe state (NEUTRAL, ARMED, TRIGGERED, WAIT_FOR_REARM)
- Current close-gesture state (NEUTRAL, CANDIDATE, CONFIRMED, WAIT_FOR_REARM)
- Close-gesture hold start time
- Last valid hand position
- Suppression state and cooldown timers

**Dependencies**: Configuration

**Must NOT**:
- Map gestures to browser intents
- Execute browser actions
- Access camera or MediaPipe

**Interface**:
```python
@enum.unique
class NavModeState(enum.Enum):
    IDLE = "idle"
    ENGAGED = "engaged"
    ACTIVE = "active"
    RELEASED = "released"


@enum.unique
class CloseGestureState(enum.Enum):
    NEUTRAL = "neutral"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    WAIT_FOR_REARM = "wait_for_rearm"


@enum.unique
class SwipeState(enum.Enum):
    NEUTRAL = "neutral"
    ARMED = "armed"
    TRIGGERED = "triggered"
    WAIT_FOR_REARM = "wait_for_rearm"


@dataclass
class GestureEvent:
    event_type: str  # "nav_engage", "nav_release", "swipe_left", "swipe_right", "tab_close"
    frame_id: int
    timestamp: float
    hand_position: tuple[float, float]  # Position at event time
    metadata: dict[str, Any]  # Additional event-specific data


@dataclass
class StateMachineState:
    nav_mode_state: NavModeState
    close_state: CloseGestureState
    swipe_state: SwipeState
    suppressed: bool
    suppression_reason: str | None
    nav_engage_position: tuple[float, float] | None
    close_hold_start: float | None
    last_valid_position: tuple[float, float] | None
    last_valid_timestamp: float


class GestureStateMachine:
    def __init__(self, config: StateMachineConfig) -> None: ...
    def process(
        self, candidate: GestureCandidate, detection: DetectionResult
    ) -> list[GestureEvent]: ...
    def get_state(self) -> StateMachineState: ...
    def reset(self) -> None: ...
```

**State Transitions**:

**Navigation Mode State Machine**:
```
IDLE ──[VICTORY held]──▶ ENGAGED ──[continued VICTORY]──▶ ACTIVE
  ▲                                                      │
  └────────────────[OPEN_PALM held]──────────────────────┘
  
ENGAGED state: emit navigation-engage event, store nav_engage_position
ACTIVE state: no event emitted while active
RELEASE transition: emit navigation-release event, return to IDLE
```

**Open-Palm Close State Machine**:
```
NEUTRAL ──[OPEN_PALM while idle]──▶ CANDIDATE ──[hold_duration met]──▶ CONFIRMED
   ▲                              │                                │
   │                              │ [open palm released early]     │
   └──────────────────────────────┘                                │
                                                                    ▼
                        WAIT_FOR_REARM ◀──[emit Tab_Close]─────────┘
                                │
                                │ [hand opens]
                                ▼
                            NEUTRAL
```

**Swipe State Machine**:
```
NEUTRAL ──[valid SWIPE candidate]──▶ TRIGGERED ──[emit Swipe event]──▶ WAIT_FOR_REARM
   ▲                                                                      │
   │                                         [hand returns to neutral]   │
   └──────────────────────────────────────────────────────────────────────┘
```

**Safety Suppression**:
```python
def should_suppress(
    self, detection: DetectionResult, candidate: GestureCandidate
) -> tuple[bool, str]:
    # Low confidence
    if detection.confidence < self.config.confidence_threshold:
        return True, "low_confidence"

    # No hand detected
    if detection.landmarks is None:
        return True, "no_hand"

    # Tracking jump
    if self.last_valid_position is not None:
        jump = distance(candidate.features.hand_center, self.last_valid_position)
        if jump > self.config.tracking_jump_threshold:
            return True, "tracking_jump"

    # Stale frame
    if candidate.timestamp - detection.timestamp > self.config.staleness_threshold:
        return True, "stale_frame"

    # Active cooldown
    if self.suppressed and time.now() < self.cooldown_end:
        return True, "cooldown"

    return False, ""
```

---

### 6. Intent Resolver

**Location**: `src/interaction/`

**Responsibility**: Map gesture events to browser intents, handling incremental movement and re-arm logic.

**Inputs**:
- `GestureEvent`: Confirmed gesture lifecycle transitions
- `NormalizedFeatures`: Current hand features for movement tracking

**Outputs**:
- `BrowserIntent`: Semantic browser action request

**State**:
- Previous hand position (for incremental scroll)
- Tab-switch re-arm state
- Horizontal displacement accumulator

**Dependencies**: Configuration

**Must NOT**:
- Execute browser actions
- Access camera or MediaPipe
- Track gesture lifecycle (that's the state machine)

**Interface**:
```python
@enum.unique
class IntentType(enum.Enum):
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    NEXT_TAB = "next_tab"
    PREVIOUS_TAB = "previous_tab"
    BROWSER_FORWARD = "browser_forward"
    BROWSER_BACK = "browser_back"
    CLOSE_TAB = "close_tab"


@dataclass
class BrowserIntent:
    intent_type: IntentType
    magnitude: float | None  # For scroll intents
    frame_id: int
    timestamp: float


@dataclass
class IntentResolverState:
    previous_position: tuple[float, float] | None
    tab_switch_armed: bool  # True after re-arm threshold crossed
    current_horizontal_displacement: float


class IntentResolver:
    def __init__(self, config: IntentConfig) -> None: ...
    def resolve(
        self, event: GestureEvent, features: NormalizedFeatures
    ) -> list[BrowserIntent]: ...
    def update_nav_active(
        self, features: NormalizedFeatures
    ) -> list[BrowserIntent]: ...
    def reset(self) -> None: ...
```

**Scroll Intent Generation** (incremental, not cumulative):
```
def update_nav_active(self, features: NormalizedFeatures) -> list[BrowserIntent]:
    intents = []
    
    if self.previous_position is None:
        self.previous_position = features.hand_center
        return intents
    
    # Calculate incremental movement
    delta_y = features.hand_center[1] - self.previous_position[1]
    
    # Apply deadzone
    if abs(delta_y) > self.config.scroll_deadzone:
        # Proportional scroll
        scroll_amount = (abs(delta_y) - self.config.scroll_deadzone) * self.config.scroll_scale_factor
        
        if delta_y > 0:  # Hand moved down (y increases downward)
            intents.append(BrowserIntent(IntentType.SCROLL_DOWN, scroll_amount, ...))
        else:
            intents.append(BrowserIntent(IntentType.SCROLL_UP, scroll_amount, ...))
    
    # Update previous position AFTER processing
    # This ensures stationary hand produces no additional intents
    self.previous_position = features.hand_center
    
    return intents
```

**Tab-Switch Intent Generation** (with re-arm):
Track horizontal displacement relative to the navigation engagement position.

The tab-switch state follows this lifecycle:

```text
ARMED
  │
  │ horizontal displacement exceeds tab_switch_threshold
  ▼
TRIGGERED → emit exactly one tab-switch intent
  │
  ▼
DISARMED
  │
  │ hand returns within tab_switch_rearm_threshold
  ▼
ARMED
```

Pseudocode:

```python
def handle_horizontal_movement(self, features):
    intents = []

    if self.engage_position is None:
        return intents

    horizontal_displacement = features.hand_center[0] - self.engage_position[0]

    if not self.tab_switch_armed:
        # A new tab switch cannot occur until the hand
        # returns to the re-arm zone.
        if abs(horizontal_displacement) <= self.config.tab_switch_rearm_threshold:
            self.tab_switch_armed = True

        return intents

    if horizontal_displacement >= self.config.tab_switch_threshold:
        intents.append(BrowserIntent(IntentType.NEXT_TAB, None, ...))
        self.tab_switch_armed = False

    elif horizontal_displacement <= -self.config.tab_switch_threshold:
        intents.append(BrowserIntent(IntentType.PREVIOUS_TAB, None, ...))
        self.tab_switch_armed = False

    return intents
```

A tab-switch intent is therefore emitted at most once until the hand returns to the configured re-arm zone.

---

### 7. Action Dispatcher

**Location**: `src/actions/`

**Responsibility**: Execute browser actions by invoking the Browser_Action_Adapter. Single entry point for all browser actions.

**Inputs**:
- `BrowserIntent`: Semantic browser action request

**Outputs**: None (side effects via Browser_Action_Adapter)

**State**: None

**Dependencies**: Browser_Action_Adapter

**Must NOT**:
- Recognize gestures
- Map intents (that's Intent_Resolver)
- Be bypassed by other components

**Interface**:
```python
class ActionDispatcher:
    def __init__(self, adapter: BrowserActionAdapter) -> None: ...
    def dispatch(self, intent: BrowserIntent) -> None: ...
```

**Dispatch Logic**:
```python
def dispatch(self, intent: BrowserIntent) -> None:
    match intent.intent_type:
        case IntentType.SCROLL_UP:
            self.adapter.scroll_up(intent.magnitude)
        case IntentType.SCROLL_DOWN:
            self.adapter.scroll_down(intent.magnitude)
        case IntentType.NEXT_TAB:
            self.adapter.next_tab()
        case IntentType.PREVIOUS_TAB:
            self.adapter.previous_tab()
        case IntentType.BROWSER_FORWARD:
            self.adapter.browser_forward()
        case IntentType.BROWSER_BACK:
            self.adapter.browser_back()
        case IntentType.CLOSE_TAB:
            self.adapter.close_tab()
```

---

### 8. Browser Action Adapter

**Location**: `src/actions/`

**Responsibility**: Translate semantic intents into OS/browser-specific actions via PyAutoGUI.

**Inputs**: Method calls from Action_Dispatcher

**Outputs**: None (side effects via PyAutoGUI)

**State**: None

**Dependencies**: PyAutoGUI

**Must NOT**:
- Be called by any component other than Action_Dispatcher
- Make decisions about gesture recognition
- Maintain gesture state

**Interface**:
```python
class BrowserActionAdapter:
    def __init__(self, config: AdapterConfig) -> None: ...
    def scroll_up(self, amount: float) -> None: ...
    def scroll_down(self, amount: float) -> None: ...
    def next_tab(self) -> None: ...
    def previous_tab(self) -> None: ...
    def browser_forward(self) -> None: ...
    def browser_back(self) -> None: ...
    def close_tab(self) -> None: ...
```

**Platform-Specific Key Combinations**:

The Browser_Action_Adapter SHALL encapsulate platform-specific keyboard shortcuts. The adapter must use browser-tab shortcuts rather than operating-system application-switching shortcuts.

For Windows/Linux:

```text
NEXT_TAB: Ctrl + Tab
PREVIOUS_TAB: Ctrl + Shift + Tab
BROWSER_BACK: Alt + Left
BROWSER_FORWARD: Alt + Right
CLOSE_TAB: Ctrl + W
```

For macOS:

```text
NEXT_TAB: Ctrl + Tab
PREVIOUS_TAB: Ctrl + Shift + Tab
BROWSER_BACK: Command + Left
BROWSER_FORWARD: Command + Right
CLOSE_TAB: Command + W
```

The implementation SHALL NOT use `Command + Tab` for browser tab switching on macOS, because that invokes the operating-system application switcher rather than switching browser tabs.

---

### 9. HUD (Heads-Up Display)

**Location**: `src/ui/`

**Responsibility**: Display floating, frameless, always-on-top window showing current interaction state.

**Inputs**:
- Gesture events (subscription)
- Browser intents (subscription)

**Outputs**: None (visual display)

**State**:
- Current display state
- Feedback message queue
- Tkinter window handle

**Dependencies**: Tkinter

**Must NOT**:
- Recognize gestures
- Execute browser actions
- Block the main event loop

**Interface**:
```python
@dataclass
class HUDState:
    interaction_state: str  # "idle", "nav_active"
    feedback_message: str | None
    feedback_expiry: float | None


class HUD:
    def __init__(self, config: HUDConfig) -> None: ...
    def start(self) -> None: ...
    def update_state(self, state: str) -> None: ...
    def show_feedback(self, message: str, duration: float) -> None: ...
    def process_events(self) -> None: ...  # Non-blocking
```

**Display Content**:
- Current state: "Idle" or "Navigation Active"
- Feedback messages: "Next Tab", "Previous Tab", "Back", "Forward", "Tab Closed"
- Visual minimalism: Small, semi-transparent, positioned in corner

---

### 10. Configuration System

**Location**: `config/`

**Responsibility**: Load and provide configuration values with documented defaults.

**Inputs**: Configuration file (TOML)

**Outputs**: Configuration dataclasses for all components

**State**: Loaded configuration values

**Dependencies**: Standard library (tomllib)

**Must NOT**:
- Hardcode values in code
- Fail silently on missing critical values

**Interface**:
```python
@dataclass
class CameraConfig:
    device_id: int = 0
    fps: int = 30


@dataclass
class DetectorConfig:
    model_complexity: int = 0  # MediaPipe: 0, 1, or 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class GestureConfig:
    # Swipe thresholds
    swipe_displacement_threshold: float = 0.3  # Normalized
    swipe_velocity_threshold: float = 1.0  # Normalized units per second
    swipe_window_frames: int = 10
    swipe_direction_consistency_threshold: float = 0.7  # Fraction


@dataclass
class StateMachineConfig:
    # Confidence
    confidence_threshold: float = 0.5

    # Idle open-palm close hold duration (legacy configuration key)
    fist_hold_duration: float = 0.5  # Seconds

    # Safety
    tracking_jump_threshold: float = 0.5  # Normalized
    staleness_threshold: float = 0.1  # Seconds (100ms)
    tracking_loss_cooldown: float = 0.3  # Seconds


@dataclass
class IntentConfig:
    # Scroll
    scroll_deadzone: float = 0.02  # Normalized
    scroll_scale_factor: float = 500.0  # Pixels per normalized unit

    # Tab switch
    tab_switch_threshold: float = 0.15  # Normalized
    tab_switch_rearm_threshold: float = 0.05  # Normalized


@dataclass
class HUDConfig:
    window_width: int = 200
    window_height: int = 100
    opacity: float = 0.8
    position: str = "top-right"
    feedback_duration: float = 1.0  # Seconds


@dataclass
class Config:
    camera: CameraConfig
    detector: DetectorConfig
    gesture: GestureConfig
    state_machine: StateMachineConfig
    intent: IntentConfig
    hud: HUDConfig


def load_config(path: str | None = None) -> Config: ...
```

**Configuration File Format** (`config/config.toml`):
```toml
[camera]
  device_id = 0
  fps = 30

[detector]
  model_complexity = 0
  min_detection_confidence = 0.5
  min_tracking_confidence = 0.5

[gesture]
  swipe_displacement_threshold = 0.3
  swipe_velocity_threshold = 1.0
  swipe_window_frames = 10
  swipe_direction_consistency_threshold = 0.7

[state_machine]
  confidence_threshold = 0.5
  fist_hold_duration = 0.5 # legacy key; controls idle open-palm close hold
  tracking_jump_threshold = 0.5
  staleness_threshold = 0.1
  tracking_loss_cooldown = 0.3

[intent]
  scroll_deadzone = 0.02
  scroll_scale_factor = 500.0
  tab_switch_threshold = 0.15
  tab_switch_rearm_threshold = 0.05

[hud]
  window_width = 200
  window_height = 100
  opacity = 0.8
  position = "top-right"
  feedback_duration = 1.0
```

The configuration loader still accepts the historical `clutch_engage_threshold`
and `clutch_release_threshold` keys for file compatibility; the live gesture
pipeline does not use them.

---

## Data Models

### Core Data Types

```python
# === Camera Layer ===
@dataclass
class Frame:
    image: np.ndarray  # BGR format, shape (height, width, 3)
    timestamp: float  # Unix timestamp in seconds
    frame_id: int  # Monotonically increasing


# === Vision Layer ===
@dataclass
class Landmark:
    x: float  # Normalized [0, 1]
    y: float  # Normalized [0, 1]
    z: float  # Depth relative to wrist


@dataclass
class HandLandmarks:
    landmarks: list[Landmark]  # 21 landmarks, MediaPipe order
    handedness: str  # "Left" or "Right"
    frame_id: int


@dataclass
class DetectionResult:
    landmarks: HandLandmarks | None
    confidence: float  # [0, 1]
    frame_id: int
    timestamp: float


@dataclass
class NormalizedFeatures:
    pinch_distance: float  # Normalized [0, ~1]
    hand_center: tuple[float, float]  # Normalized x, y
    finger_extensions: dict[str, float]  # Per-finger
    openness_score: float  # [0, 1]
    hand_scale: float  # Reference for denormalization
    frame_id: int
    timestamp: float


# === Gesture Layer ===
@enum.unique
class GestureType(enum.Enum):
    NONE = "none"
    VICTORY = "victory"
    POINTING_UP = "pointing_up"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    OPEN_PALM = "open_palm"
    THUMB_UP = "thumb_up"
    THUMB_DOWN = "thumb_down"


@dataclass
class SwipeMetadata:
    net_displacement: float
    elapsed_time: float
    velocity: float
    direction_consistency: float


@dataclass
class GestureCandidate:
    gesture_type: GestureType
    confidence: float
    frame_id: int
    timestamp: float
    features: NormalizedFeatures
    swipe_metadata: SwipeMetadata | None


# === State Machine Layer ===
@dataclass
class GestureEvent:
    event_type: str  # "nav_engage", "nav_release", "swipe_left", "swipe_right", "tab_close"
    frame_id: int
    timestamp: float
    hand_position: tuple[float, float]
    metadata: dict[str, Any]


# === Intent Layer ===
@enum.unique
class IntentType(enum.Enum):
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    NEXT_TAB = "next_tab"
    PREVIOUS_TAB = "previous_tab"
    BROWSER_FORWARD = "browser_forward"
    BROWSER_BACK = "browser_back"
    CLOSE_TAB = "close_tab"


@dataclass
class BrowserIntent:
    intent_type: IntentType
    magnitude: float | None  # For scroll intents
    frame_id: int
    timestamp: float
```

### Data Flow Summary

```
Frame
  └─▶ DetectionResult (landmarks + confidence)
        └─▶ NormalizedFeatures
              └─▶ GestureCandidate
                    └─▶ GestureEvent
                          └─▶ BrowserIntent
```

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Feature Normalization Bounded Range

*For any* valid hand landmarks, all normalized distance features (pinch_distance, finger_extensions) SHALL be in the range [0, 2] and position features (hand_center) SHALL be in the range [0, 1].

**Validates: Requirements 3.1, 3.2**

### Property 2: Pinch Distance Non-Negative

*For any* valid hand landmarks, the computed pinch_distance SHALL be non-negative.

**Validates: Requirements 3.3**

### Property 3: Victory Stability

*For any* sequence of landmark poses, the Gesture_Recognizer SHALL not expose a Victory candidate to the state machine until the pose has met the configured temporal stability requirement.

**Validates: Requirements 4.1**

### Property 4: Navigation Mode Stability

*For any* alternating pose sequence shorter than the configured stability requirement, the Gesture_State_Machine SHALL not oscillate between navigation mode and idle.

**Validates: Requirements 4.3, 4.4**

### Property 5: Open-Palm Navigation Release

*For any* active navigation session, a stable open-palm sequence held for the release duration SHALL emit exactly one navigation-release event and return to idle.

**Validates: Requirements 4.3**

### Property 6: Navigation Mode State Machine Transitions

*For any* Gesture_State_Machine, receiving a stable Victory hold in IDLE state SHALL transition to navigation-engaged and emit a navigation-engage event; receiving a stable open-palm hold in ACTIVE state SHALL transition to IDLE and emit a navigation-release event.

**Validates: Requirements 5.1, 5.3**

### Property 7: No Events During Continuous Navigation

*For any* Gesture_State_Machine in navigation-active state receiving continued Victory candidates, no GestureEvent SHALL be emitted solely because navigation remains active.

**Validates: Requirements 5.2**

### Property 8: Scroll Intent Requires Movement

*For any* sequence of hand positions where the incremental vertical movement is zero (stationary hand), the Intent_Resolver SHALL NOT emit any scroll intents.

**Validates: Requirements 6.2**

### Property 9: Scroll Intent Proportional to Displacement

*For any* incremental vertical movement exceeding the deadzone, the Intent_Resolver SHALL emit a scroll intent with magnitude proportional to (|delta_y| - deadzone) × scroll_scale_factor.

**Validates: Requirements 6.2**

### Property 10: Scroll Direction Correct

*For any* upward vertical movement (negative delta_y in image coordinates) exceeding the deadzone, the Intent_Resolver SHALL emit SCROLL_UP. *For any* downward movement exceeding the deadzone, the Intent_Resolver SHALL emit SCROLL_DOWN.

**Validates: Requirements 6.3, 6.4**

### Property 11: Tab Switch Requires Re-Arm

*For any* sequence of tab-switch intents, the Intent_Resolver SHALL NOT emit a second tab-switch intent until the hand has returned within the re-arm_threshold after the previous intent.

**Validates: Requirements 7.4**

### Property 12: Swipe Displacement Threshold

*For any* horizontal movement with net displacement below the swipe_displacement_threshold, the Gesture_Recognizer SHALL NOT output a swipe candidate.

**Validates: Requirements 8.4**

### Property 13: Swipe Velocity Threshold

*For any* horizontal movement with velocity below the swipe_velocity_threshold, the Gesture_Recognizer SHALL NOT output a swipe candidate.

**Validates: Requirements 8.5**

### Property 14: Swipe Direction Consistency

*For any* horizontal movement with frame-to-frame direction inconsistent with overall swipe direction, the Gesture_Recognizer SHALL NOT output a swipe candidate.

**Validates: Requirements 8.6**

### Property 15: Swipe Single Action

*For any* valid swipe candidate, the Gesture_State_Machine SHALL emit exactly one Swipe_Left or Swipe_Right event.

**Validates: Requirements 9.2, 9.3**

### Property 16: Swipe Re-Arm

*For any* Gesture_State_Machine that has emitted a swipe event, subsequent swipe candidates SHALL be rejected until the hand returns to neutral position.

**Validates: Requirements 9.4**

### Property 17: Open-Palm Close Hold Duration

*For any* idle open-palm candidate sequence lasting less than the hold_duration, the Gesture_State_Machine SHALL NOT emit a Tab_Close event.

**Validates: Requirements 12.7**

### Property 18: Open-Palm Close Confirm Emits Event

*For any* idle open-palm candidate sequence lasting at least the hold_duration, the Gesture_State_Machine SHALL confirm the close gesture and emit exactly one Tab_Close event.

**Validates: Requirements 12.2, 12.3**

### Property 19: Open-Palm Close No Repeat Events

*For any* Gesture_State_Machine in the close gesture's confirmed or WAIT_FOR_REARM state, no additional Tab_Close events SHALL be emitted.

**Validates: Requirements 12.6**

### Property 20: Low Confidence Suppression

*For any* gesture candidate with Frame_Confidence below the configurable threshold, the Gesture_State_Machine SHALL suppress all state transitions.

**Validates: Requirements 14.1**

### Property 21: No Hand Reset

*For any* DetectionResult with no hand detected, the Gesture_State_Machine SHALL reset to neutral states.

**Validates: Requirements 14.2**

### Property 22: Tracking Jump Suppression

*For any* sudden coordinate jump exceeding the tracking_jump_threshold, the Gesture_State_Machine SHALL suppress gesture state transitions for the cooldown period.

**Validates: Requirements 14.3**

### Property 23: Stale Frame Suppression

*For any* frame older than the staleness_threshold, the Gesture_State_Machine SHALL suppress gesture state transitions.

**Validates: Requirements 14.4**

### Property 24: Suppressed State No Events

*For any* Gesture_State_Machine in suppressed state, no gesture events SHALL be emitted.

**Validates: Requirements 14.5**

### Property 25: Intent Mapping Consistency

*For any* Swipe_Right event, the Intent_Resolver SHALL emit BROWSER_FORWARD intent. *For any* Swipe_Left event, the Intent_Resolver SHALL emit BROWSER_BACK intent. *For any* Tab_Close event, the Intent_Resolver SHALL emit CLOSE_TAB intent.

**Validates: Requirements 10.1, 10.2, 13.1**

---

## Error Handling

### Camera Failure

**Detection**: `read_frame()` returns `None` or raises exception

**Response**:
1. Log error with timestamp
2. Signal downstream to pause gesture processing
3. Attempt recovery by polling `is_available()` every 1 second
4. Resume normal operation when camera becomes available

**Recovery**: Automatic, no user intervention required

### No Hand Detected

**Detection**: `DetectionResult.landmarks is None`

**Response**:
1. Output `NoHandSignal` to downstream
2. Gesture_State_Machine resets to neutral states
3. No gesture events emitted
4. HUD displays "Idle" state

**Recovery**: Automatic when hand re-enters frame

### Low Confidence

**Detection**: `DetectionResult.confidence < confidence_threshold`

**Response**:
1. Output `LowConfidenceSignal` to downstream
2. Gesture_State_Machine suppresses all state transitions
3. No gesture events emitted
4. HUD displays current state (unchanged)

**Recovery**: Automatic when confidence exceeds threshold

### Tracking Jump

**Detection**: Distance between current and previous hand position > tracking_jump_threshold

**Response**:
1. Mark tracking loss detected
2. Suppress gesture state transitions for cooldown period
3. Reset previous position tracking after cooldown

**Recovery**: Automatic after cooldown period expires

### Stale Frame

**Detection**: `current_timestamp - frame_timestamp > staleness_threshold`

**Response**:
1. Suppress gesture state transitions
2. Log staleness warning

**Recovery**: Automatic when fresh frames arrive

### MediaPipe Processing Error

**Detection**: MediaPipe raises exception during processing

**Response**:
1. Log error with stack trace
2. Return `DetectionResult` with `landmarks=None`
3. Continue processing next frame (don't crash)

**Recovery**: Automatic next frame

---

## Testing Strategy

### Property-Based Testing (Hypothesis)

Property-based testing is appropriate for this feature because:
- Gesture recognition involves pure functions with clear input/output behavior
- State machines have well-defined transition rules
- Normalization functions preserve invariants across all inputs
- Intent resolution has predictable mapping behavior

**Test Configuration**:
- Minimum 100 iterations per property test
- Tag format: `Feature: conductor-mvp-gesture-browser-control, Property {N}: {description}`

**Property Test Categories**:

1. **Normalization Properties** (FeatureExtractor)
   - Bounded output ranges
   - Non-negative distances
   - Consistent hand scale computation

2. **Gesture Recognition Properties** (GestureRecognizer)
   - Threshold behavior (engage, release)
   - Hysteresis stability
   - Swipe rejection criteria

3. **State Machine Properties** (GestureStateMachine)
   - State transition correctness
   - No events in continuous states
   - Re-arm enforcement
   - Safety suppression

4. **Intent Resolution Properties** (IntentResolver)
   - Scroll magnitude correctness
   - Direction correctness
   - Re-arm requirement

### Unit Tests (pytest)

**Categories**:

1. **Example-based tests**: Specific scenarios with concrete inputs
   - Victory engage → navigation active → open-palm release lifecycle
   - Swipe detection with known frame sequences
   - Idle open-palm close hold and confirm

2. **Edge case tests**: Boundary conditions
   - Pinch distance exactly at threshold
   - Zero movement during navigation active
   - Confidence exactly at threshold

3. **Error handling tests**:
   - Camera unavailable
   - No hand in frame
   - Tracking jump detection

### Integration Tests

**Categories**:

1. **Pipeline integration**: Frame → Landmarks → Features → Candidates
2. **State machine integration**: Candidates → Events → Intents
3. **Action dispatch**: Intents → Adapter calls (mocked)
4. **HUD updates**: Events → Display state

### Mock Strategy

**Testability Requirements**:

1. **Mock Frame Input**: `HandLandmarkDetector.process(mock_frame)`
   - `mock_frame` is any np.ndarray with correct shape

2. **Mock Feature Input**: `GestureRecognizer.recognize(mock_features, confidence)`
   - `mock_features` is NormalizedFeatures instance

3. **Mock Candidate Input**: `GestureStateMachine.process(mock_candidate, mock_detection)`
   - Both are dataclass instances

4. **Mock Action Dispatcher**: Verify intent dispatch without side effects
   - Replace BrowserActionAdapter with mock that records calls

**Test File Organization**:
```
tests/
├── unit/
│   ├── test_feature_extractor.py
│   ├── test_gesture_recognizer.py
│   ├── test_gesture_state_machine.py
│   ├── test_intent_resolver.py
│   └── test_config.py
├── property/
│   ├── test_normalization_properties.py
│   ├── test_threshold_properties.py
│   ├── test_state_machine_properties.py
│   └── test_intent_properties.py
├── integration/
│   ├── test_pipeline.py
│   ├── test_action_dispatch.py
│   └── test_hud.py
└── conftest.py  # Shared fixtures and mocks
```

---

## Implementation Order

The implementation should proceed in phases that minimize coupling and enable incremental testing.

### Phase 1: Foundation (No External Dependencies)

1. **Configuration System** (`config/`)
   - Define all dataclasses
   - Implement load_config with defaults
   - Test: Verify default values and file loading

2. **Data Models** (shared across modules)
   - Define all dataclasses in `src/models.py`
   - Test: Verify dataclass construction and validation

### Phase 2: Vision Pipeline (Camera → Landmarks → Features)

3. **Camera Module** (`src/camera/`)
   - Implement Frame capture
   - Implement error handling
   - Test: Mock camera, verify frame output

4. **Hand Landmark Detector** (`src/vision/`)
   - Wrap MediaPipe
   - Implement confidence extraction
   - Test: Mock frames, verify landmark output

5. **Feature Extractor** (`src/vision/`)
   - Implement normalization
   - Implement feature computations
   - Test: Property tests for normalization, unit tests for each feature

### Phase 3: Gesture Recognition

6. **Gesture Recognizer - Navigation Poses** (`src/gestures/`)
   - Implement Victory/open-palm pose stability
   - Test: Property tests for pose stability

7. **Gesture Recognizer - Swipe** (`src/gestures/`)
   - Implement temporal buffer
   - Implement swipe detection criteria
   - Test: Property tests for rejection criteria

8. **Gesture Recognizer - Open Palm and Thumb Poses** (`src/gestures/`)
   - Implement open-palm and thumb-direction detection
   - Reject closed-fist poses from the live vocabulary
   - Test: Pose stability and ambiguity handling

### Phase 4: State Machines

9. **Gesture State Machine - Navigation Mode** (`src/interaction/`)
   - Implement Victory-engage/open-palm-release lifecycle
   - Implement position tracking
   - Test: Property tests for state transitions

10. **Gesture State Machine - Swipe & Open-Palm Close** (`src/interaction/`)
    - Implement swipe validation and re-arm
    - Implement idle open-palm close hold confirmation and re-arm
    - Test: Property tests for single-action guarantee

11. **Safety Suppression** (`src/interaction/`)
    - Implement all suppression conditions
    - Test: Property tests for no events during suppression

### Phase 5: Intent Resolution

12. **Intent Resolver** (`src/interaction/`)
    - Implement scroll intent generation
    - Implement tab-switch with re-arm
    - Implement navigation mapping
    - Test: Property tests for intent correctness

### Phase 6: Action Execution

13. **Browser Action Adapter** (`src/actions/`)
    - Implement PyAutoGUI wrappers
    - Implement platform-specific shortcuts
    - Test: Mock PyAutoGUI, verify key combinations

14. **Action Dispatcher** (`src/actions/`)
    - Implement dispatch logic
    - Test: Mock adapter, verify dispatch routing

### Phase 7: User Interface

15. **HUD** (`src/ui/`)
    - Implement Tkinter window
    - Implement state display
    - Implement feedback display
    - Test: Verify display state updates

### Phase 8: Integration

16. **Main Pipeline** (`main.py`)
    - Wire all components together
    - Implement main loop
    - Test: End-to-end with mocked camera

17. **Integration Tests** (`tests/integration/`)
    - Full pipeline tests
    - Error recovery tests

---

## Requirements Traceability

| Requirement | Design Component | Section |
|------------|------------------|---------|
| R1: Camera Frame Capture | Camera Module | Components - 1 |
| R2: Hand Landmark Detection | Hand Landmark Detector | Components - 2 |
| R3: Feature Extraction | Feature Extractor | Components - 3 |
| R4: Navigation Mode Gesture Recognition | Gesture Recognizer (Navigation Poses) | Components - 4 |
| R5: Navigation Mode State Machine | Gesture State Machine (Navigation Mode) | Components - 5 |
| R6: Scroll Control via Navigation Mode | Intent Resolver (Scroll) | Components - 6 |
| R7: Tab Switch Control via Navigation Mode | Intent Resolver (Tab Switch) | Components - 6 |
| R8: Ballistic Swipe Gesture Recognition | Gesture Recognizer (Swipe) | Components - 4 |
| R9: Swipe Gesture State Machine | Gesture State Machine (Swipe) | Components - 5 |
| R10: Browser History Navigation via Swipe | Intent Resolver (Navigation) | Components - 6 |
| R11: Open Palm Close Gesture Recognition | Gesture Recognizer (Open Palm and Thumb) | Components - 4 |
| R12: Open Palm Close Gesture State Machine | Gesture State Machine (Open-Palm Close) | Components - 5 |
| R13: Tab Close Action | Intent Resolver, Action Dispatcher | Components - 6, 7 |
| R14: Low Confidence and Tracking Failure Safety | Gesture State Machine (Safety) | Components - 5 |
| R15: Floating HUD Display | HUD | Components - 9 |
| R16: Action Dispatcher | Action Dispatcher | Components - 7 |
| R17: Browser Action Adapter | Browser Action Adapter | Components - 8 |
| R18: Configuration System | Configuration System | Components - 10 |
| R19: Testability Without Hardware | All components via interfaces | Testing Strategy |
