"""UI layer for Conductor.

Owns the floating heads-up display. Must not execute browser actions
or block the main event loop.

Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7
"""

from src.ui.hud import HUD, HUDState

__all__ = ["HUD", "HUDState"]
