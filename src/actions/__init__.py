"""Action execution layer for Conductor.

This layer is the ONLY component that invokes the browser/OS. The
:class:`ActionDispatcher` is the single entry point for all browser
actions; the :class:`BrowserActionAdapter` translates intents into
PyAutoGUI calls.

Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 17.1..17.6
"""

from src.actions.adapter import BrowserActionAdapter
from src.actions.dispatcher import ActionDispatcher

__all__ = ["ActionDispatcher", "BrowserActionAdapter"]
