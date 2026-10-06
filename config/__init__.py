"""Configuration system for Conductor gesture browser control.

This module provides configuration dataclasses and loading functions
for all system parameters including camera, detection, gesture recognition,
state machine, intent resolution, and HUD display.
"""

from config.config import (
    CameraConfig,
    Config,
    DetectorConfig,
    GestureConfig,
    HUDConfig,
    IntentConfig,
    StateMachineConfig,
    load_config,
)

__all__ = [
    "CameraConfig",
    "Config",
    "DetectorConfig",
    "GestureConfig",
    "HUDConfig",
    "IntentConfig",
    "StateMachineConfig",
    "load_config",
]
