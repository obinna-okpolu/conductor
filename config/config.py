"""Configuration dataclasses and loading for Conductor.

All thresholds and sensitivities are configurable via a TOML file.
Defaults are provided for all parameters.

Requirements: 18.1, 18.2, 18.3
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CameraConfig:
    """Configuration for camera capture.

    Attributes:
        device_id: The camera device index (default webcam is 0).
        fps: Target frames per second for capture.
    """

    device_id: int = 0
    fps: int = 30


@dataclass
class DetectorConfig:
    """Configuration for MediaPipe hand landmark detection.

    Attributes:
        model_complexity: MediaPipe model complexity (0, 1, or 2).
            Higher values are more accurate but slower.
        min_detection_confidence: Minimum confidence for initial detection.
        min_tracking_confidence: Minimum confidence for tracking across frames.
    """

    model_complexity: int = 0
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class GestureConfig:
    """Configuration for gesture recognition thresholds.

    Attributes:
        clutch_engage_threshold: Normalized pinch distance below which
            clutch engages. Lower than release for hysteresis.
        clutch_release_threshold: Normalized pinch distance above which
            clutch releases. Higher than engage for hysteresis.
        swipe_displacement_threshold: Minimum normalized horizontal
            displacement for swipe detection.
        swipe_velocity_threshold: Minimum normalized velocity for swipe.
        swipe_window_frames: Number of frames to analyze for swipe.
        swipe_direction_consistency_threshold: Minimum fraction of frames
            moving in same direction for valid swipe.
    """

    # Clutch thresholds (normalized)
    clutch_engage_threshold: float = 0.08
    clutch_release_threshold: float = 0.12  # Higher for hysteresis

    # Swipe thresholds
    swipe_displacement_threshold: float = 0.3  # Normalized
    swipe_velocity_threshold: float = 1.0  # Normalized units per second
    swipe_window_frames: int = 10
    swipe_direction_consistency_threshold: float = 0.7  # Fraction


@dataclass
class StateMachineConfig:
    """Configuration for gesture state machine behavior.

    Attributes:
        confidence_threshold: Minimum detection confidence to process gestures.
        fist_hold_duration: Seconds to hold fist before confirming tab close.
        tracking_jump_threshold: Normalized distance threshold for detecting
            tracking jumps (sudden large coordinate changes).
        staleness_threshold: Maximum age in seconds for a frame to be valid.
        tracking_loss_cooldown: Seconds to suppress gestures after tracking jump.
    """

    # Confidence
    confidence_threshold: float = 0.5

    # Fist hold duration
    fist_hold_duration: float = 0.5  # Seconds

    # Safety
    tracking_jump_threshold: float = 0.5  # Normalized
    staleness_threshold: float = 0.1  # Seconds (100ms)
    tracking_loss_cooldown: float = 0.3  # Seconds


@dataclass
class IntentConfig:
    """Configuration for intent resolution.

    Attributes:
        scroll_deadzone: Normalized vertical movement ignored to prevent
            unintended scrolling from small movements.
        scroll_scale_factor: Pixels per normalized unit of scroll.
        tab_switch_threshold: Normalized horizontal displacement required
            to trigger tab switch.
        tab_switch_rearm_threshold: Normalized distance hand must return
            to center before another tab switch can trigger.
    """

    # Scroll
    scroll_deadzone: float = 0.02  # Normalized
    scroll_scale_factor: float = 500.0  # Pixels per normalized unit

    # Tab switch
    tab_switch_threshold: float = 0.15  # Normalized
    tab_switch_rearm_threshold: float = 0.05  # Normalized


@dataclass
class HUDConfig:
    """Configuration for heads-up display.

    Attributes:
        window_width: Width of HUD window in pixels.
        window_height: Height of HUD window in pixels.
        opacity: Window opacity (0.0 to 1.0).
        position: Corner position ("top-left", "top-right", "bottom-left",
            "bottom-right").
        feedback_duration: Seconds to display action feedback messages.
    """

    window_width: int = 200
    window_height: int = 100
    opacity: float = 0.8
    position: str = "top-right"
    feedback_duration: float = 1.0  # Seconds


@dataclass
class Config:
    """Root configuration containing all component configurations.

    Attributes:
        camera: Camera capture configuration.
        detector: Hand landmark detection configuration.
        gesture: Gesture recognition configuration.
        state_machine: State machine behavior configuration.
        intent: Intent resolution configuration.
        hud: Heads-up display configuration.
    """

    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    hud: HUDConfig = field(default_factory=HUDConfig)


# Section name -> dataclass mapping used to construct Config from a dict.
_SECTION_CLASSES: dict[str, type] = {
    "camera": CameraConfig,
    "detector": DetectorConfig,
    "gesture": GestureConfig,
    "state_machine": StateMachineConfig,
    "intent": IntentConfig,
    "hud": HUDConfig,
}


def _filter_known(cls: type, data: dict) -> dict:
    """Keep only keys that correspond to dataclass fields."""
    import dataclasses

    valid_fields = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in data.items() if k in valid_fields}


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from a TOML file with defaults.

    If no path is provided, looks for ``config/config.toml`` relative to the
    project root. If the file does not exist, returns default configuration.

    Missing values in the file are filled with documented defaults.

    Args:
        path: Path to TOML configuration file. If None, uses the default path.

    Returns:
        Configuration object with loaded or default values.

    Requirements: 18.1 (load from file), 18.2 (configurable parameters),
        18.3 (defaults for missing values)
    """
    if path is None:
        path = Path(__file__).parent / "config.toml"
    else:
        path = Path(path)

    sections: dict[str, dict] = {}

    if path.exists():
        with path.open("rb") as f:
            file_config = tomllib.load(f)
        if not isinstance(file_config, dict):
            raise ValueError(
                f"Configuration file {path} must contain a TOML table at top level"
            )
        sections = file_config

    kwargs: dict[str, object] = {}
    for section_name, cls in _SECTION_CLASSES.items():
        section_data = sections.get(section_name, {})
        if not isinstance(section_data, dict):
            raise ValueError(
                f"Configuration section [{section_name}] must be a TOML table"
            )
        kwargs[section_name] = cls(**_filter_known(cls, section_data))

    return Config(**kwargs)
