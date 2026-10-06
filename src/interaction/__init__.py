"""Interaction layer for Conductor.

Owns gesture lifecycle state machines and intent resolution. No
component in this layer executes browser actions or reads frames
directly.

Requirements: 5.1, 5.2, 5.3, 5.4, 6.x, 7.x, 9.x, 10.x, 12.x, 14.x
"""

from src.interaction.resolver import IntentResolver
from src.interaction.state_machine import GestureStateMachine

__all__ = ["GestureStateMachine", "IntentResolver"]
