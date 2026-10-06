"""
Core data models for the Conductor gesture browser control system.

This module defines the dataclasses and enums that represent the data structures
flowing through the gesture processing pipeline.

Pipeline: Camera → Hand Landmark Detection → Feature Extraction → Gesture Recognition
→ Gesture State Machine → Intent Resolution → Action Dispatcher → Browser Action Adapter
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import numpy as np


# ==============================================================================
# Camera Module Types
# ==============================================================================


@dataclass(frozen=True)
class Frame:
    """A single frame captured from the webcam.

    Attributes:
        image: Raw BGR image from webcam as numpy array.
        timestamp: Seconds since epoch when frame was captured.
        frame_id: Monotonically increasing frame identifier.
    """

    image: np.ndarray  # BGR format
    timestamp: float  # seconds since epoch
    frame_id: int  # monotonically increasing


@dataclass(frozen=True)
class FrameMetadata:
    """Metadata about the camera stream.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second.
    """

    width: int
    height: int
    fps: float


# ==============================================================================
# Hand Landmark Detector Types
# ==============================================================================


@dataclass(frozen=True)
class Landmark:
    """A single hand landmark in normalized coordinates.

    MediaPipe provides landmarks normalized to the frame coordinates [0, 1].

    Attributes:
        x: Horizontal position, normalized [0, 1].
        y: Vertical position, normalized [0, 1].
        z: Depth relative to wrist.
    """

    x: float  # Normalized [0, 1]
    y: float  # Normalized [0, 1]
    z: float  # Depth relative to wrist


@dataclass(frozen=True)
class HandLandmarks:
    """All 21 hand landmarks detected by MediaPipe.

    Landmark indices:
        0: WRIST
        1-4: THUMB (CMC, MCP, IP, TIP)
        5-8: INDEX (MCP, PIP, DIP, TIP)
        9-12: MIDDLE (MCP, PIP, DIP, TIP)
        13-16: RING (MCP, PIP, DIP, TIP)
        17-20: PINKY (MCP, PIP, DIP, TIP)

    Attributes:
        landmarks: List of 21 Landmark objects.
        handedness: "Left" or "Right" hand.
        frame_id: Frame identifier this detection corresponds to.
    """

    landmarks: list[Landmark]  # 21 landmarks
    handedness: str  # "Left" or "Right"
    frame_id: int


@dataclass(frozen=True)
class DetectionResult:
    """Result of hand landmark detection for a single frame.

    Attributes:
        landmarks: Detected hand landmarks, or None if no hand detected.
        confidence: Tracking quality score [0, 1].
        frame_id: Frame identifier this result corresponds to.
        timestamp: Timestamp of the detection.
    """

    landmarks: HandLandmarks | None
    confidence: float
    frame_id: int
    timestamp: float


# ==============================================================================
# Feature Extractor Types
# ==============================================================================


@dataclass(frozen=True)
class NormalizedFeatures:
    """Scale-independent geometric measurements from hand landmarks.

    All distance measurements are normalized by hand scale for camera-distance
    independence.

    Attributes:
        pinch_distance: Normalized distance between thumb tip and middle finger tip.
        hand_center: Centroid of hand landmarks in normalized frame coordinates.
        finger_extensions: Per-finger extension values [0, 1].
        openness_score: Overall hand openness [0, 1].
        hand_scale: Reference size used for normalization.
        frame_id: Frame identifier.
        timestamp: Timestamp of the features.
    """

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


# ==============================================================================
# Gesture Recognizer Types
# ==============================================================================


@enum.unique
class GestureType(enum.Enum):
    """Types of gestures that can be recognized.

    The recognizer classifies per-frame hand poses into one of these discrete
    gestures. The vocabulary is restricted to whole-hand shapes that MediaPipe
    ``mp.solutions.hands`` detects reliably (per published benchmarks, ~95%
    accuracy on the canonical 5-gesture set).

    Discrete gestures (require hold + cooldown to fire):
        * ``OPEN_PALM``: All 5 fingers extended.
        * ``CLOSED_FIST``: All 4 non-thumb fingers curled; thumb posture free.
        * ``VICTORY``: Index + middle extended, ring + pinky curled.
        * ``POINTING_UP``: Only index extended.
        * ``THUMB_UP``: Thumb extended upward, other 4 fingers curled.
        * ``THUMB_DOWN``: Thumb extended downward, other 4 fingers curled.

    Continuous gestures (still here for parity with spec, not actively used):
        * ``SWIPE_LEFT`` / ``SWIPE_RIGHT``: Fast horizontal hand motion.
    """

    NONE = "none"
    OPEN_PALM = "open_palm"
    CLOSED_FIST = "closed_fist"
    VICTORY = "victory"
    POINTING_UP = "pointing_up"
    THUMB_UP = "thumb_up"
    THUMB_DOWN = "thumb_down"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"


@dataclass(frozen=True)
class SwipeMetadata:
    """Metadata for swipe gesture candidates.

    Contains the temporal analysis results that led to the swipe detection.

    Attributes:
        net_displacement: Total horizontal displacement in normalized units.
        elapsed_time: Time elapsed during the swipe motion in seconds.
        velocity: Swipe velocity in normalized units per second.
        direction_consistency: Fraction of frame-to-frame movements in the same direction.
    """

    net_displacement: float
    elapsed_time: float
    velocity: float
    direction_consistency: float


@dataclass(frozen=True)
class GestureCandidate:
    """A potential gesture recognized by the GestureRecognizer.

    Candidates are processed by the GestureStateMachine to produce GestureEvents.

    Attributes:
        gesture_type: Type of gesture detected.
        confidence: Confidence score of the detection [0, 1].
        frame_id: Frame identifier.
        timestamp: Timestamp of the detection.
        features: Normalized features that led to this candidate.
        swipe_metadata: Additional data for swipe candidates, None otherwise.
    """

    gesture_type: GestureType
    confidence: float
    frame_id: int
    timestamp: float
    features: NormalizedFeatures
    swipe_metadata: SwipeMetadata | None = None


# ==============================================================================
# Gesture State Machine Types
# ==============================================================================


@enum.unique
class NavModeState(enum.Enum):
    """States for the navigation-mode lifecycle (formerly "clutch").

    Nav-mode is engaged by the Victory gesture (held) and disengaged by the
    Open_Palm gesture (held). While engaged, hand Y movement scrolls and
    hand X movement switches tabs.

    State transitions:
        IDLE → ENGAGED: VICTORY held for the required duration
        ENGAGED → ACTIVE: Continuous VICTORY frames
        ENGAGED → IDLE: OPEN_PALM held for the required duration
    """

    IDLE = "idle"
    ENGAGED = "engaged"
    ACTIVE = "active"


@enum.unique
class FistState(enum.Enum):
    """States for the fist gesture lifecycle.

    The fist gesture requires deliberate holding to prevent accidental tab closures.

    State transitions:
        NEUTRAL → CANDIDATE: When FIST candidate first detected
        CANDIDATE → CONFIRMED: When FIST held for hold_duration
        CANDIDATE → NEUTRAL: If fist released before hold_duration
        CONFIRMED → WAIT_FOR_REARM: After emitting Tab_Close event
        WAIT_FOR_REARM → NEUTRAL: When hand opens
    """

    NEUTRAL = "neutral"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    WAIT_FOR_REARM = "wait_for_rearm"


@enum.unique
class SwipeState(enum.Enum):
    """States for the swipe gesture lifecycle.

    Swipe gestures require re-arming to prevent multiple triggers.

    State transitions:
        NEUTRAL → TRIGGERED: When valid SWIPE candidate confirmed
        TRIGGERED → WAIT_FOR_REARM: After emitting Swipe event
        WAIT_FOR_REARM → NEUTRAL: When hand returns to neutral position
    """

    NEUTRAL = "neutral"
    ARMED = "armed"
    TRIGGERED = "triggered"
    WAIT_FOR_REARM = "wait_for_rearm"


@dataclass(frozen=True)
class GestureEvent:
    """A semantic event produced by the GestureStateMachine.

    Events represent confirmed gesture lifecycle transitions that should be
    resolved into browser intents.

    Attributes:
        event_type: One of:
            ``"nav_engage"`` / ``"nav_release"`` — scroll-mode transitions
            ``"thumb_up"`` / ``"thumb_down"`` — browser back / forward
            ``"swipe_left"`` / ``"swipe_right"`` — alternative back/forward
            ``"tab_close"`` — fist-held confirmation
        frame_id: Frame identifier.
        timestamp: Timestamp of the event.
        hand_position: Hand position at the time of the event.
        metadata: Additional event-specific data.
    """

    event_type: str
    frame_id: int
    timestamp: float
    hand_position: tuple[float, float]
    metadata: dict[str, Any]


@dataclass
class StateMachineState:
    """Complete state of the GestureStateMachine.

    Used for introspection and testing.

    Attributes:
        nav_mode_state: Current state of the navigation-mode sub-machine.
        fist_state: Current state of the fist state machine.
        swipe_state: Current state of the swipe state machine.
        suppressed: Whether gesture processing is suppressed.
        suppression_reason: Reason for suppression, if any.
        nav_engage_position: Hand position when nav-mode was engaged, for delta.
        fist_hold_start: Timestamp when fist hold started, for hold duration.
        last_valid_position: Last valid hand position for tracking jump detection.
        last_valid_timestamp: Timestamp of the last valid position.
    """

    nav_mode_state: NavModeState
    fist_state: FistState
    swipe_state: SwipeState
    suppressed: bool
    suppression_reason: str | None
    nav_engage_position: tuple[float, float] | None
    fist_hold_start: float | None
    last_valid_position: tuple[float, float] | None
    last_valid_timestamp: float


# ==============================================================================
# Intent Resolver Types
# ==============================================================================


@enum.unique
class IntentType(enum.Enum):
    """Types of browser intents that can be generated.

    Intents are semantic requests for browser actions, produced by the
    IntentResolver from GestureEvents.
    """

    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    NEXT_TAB = "next_tab"
    PREVIOUS_TAB = "previous_tab"
    BROWSER_FORWARD = "browser_forward"
    BROWSER_BACK = "browser_back"
    CLOSE_TAB = "close_tab"


@dataclass(frozen=True)
class BrowserIntent:
    """A semantic browser action request.

    Produced by IntentResolver from GestureEvents, dispatched by ActionDispatcher.

    Attributes:
        intent_type: Type of browser action requested.
        magnitude: Magnitude for scroll intents (pixels), None for other intents.
        frame_id: Frame identifier.
        timestamp: Timestamp of the intent.
    """

    intent_type: IntentType
    magnitude: float | None  # For scroll intents
    frame_id: int
    timestamp: float
