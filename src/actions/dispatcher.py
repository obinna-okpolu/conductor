"""Action dispatcher — single entry point for browser actions.

Wraps every dispatch in a ``try/except`` so a single bad ``pyautogui`` call
(e.g. transient display loss, focus loss) cannot tear down the main loop.

Requirements: 16.x
"""

from __future__ import annotations

import logging as _logging

from src.actions.adapter import BrowserActionAdapter
from src.models import BrowserIntent, IntentType

_LOG = _logging.getLogger(__name__)


class ActionDispatcher:
    """Route :class:`BrowserIntent` objects to the right adapter method."""

    def __init__(self, adapter: BrowserActionAdapter) -> None:
        self.adapter = adapter

    def dispatch(self, intent: BrowserIntent) -> None:
        """Dispatch a single browser intent. Never raises."""
        try:
            match intent.intent_type:
                case IntentType.SCROLL_UP:
                    self.adapter.scroll_up(intent.magnitude or 0.0)
                case IntentType.SCROLL_DOWN:
                    self.adapter.scroll_down(intent.magnitude or 0.0)
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
                case _:  # pragma: no cover - defensive
                    return
        except Exception as exc:  # noqa: BLE001
            # PyAutoGUI can throw on focus loss, accessibility denial, etc.
            _LOG.warning("dispatch failed for %s: %s", intent.intent_type, exc)
