"""Browser action adapter — translate semantic intents to PyAutoGUI calls.

The adapter encapsulates platform-specific keyboard shortcuts. The
implementation SHALL NOT use ``Command+Tab`` on macOS for browser tab
switching because that invokes the OS application switcher instead.

Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6
"""

from __future__ import annotations

import platform
import sys
from typing import Any

import pyautogui


class BrowserActionAdapter:
    """Translate semantic browser intents into OS-level actions.

    The adapter is the only place that touches ``pyautogui`` (or any
    future browser extension bridge). It must not be called by any
    component other than :class:`src.actions.dispatcher.ActionDispatcher`.

    Attributes:
        os_name: ``"darwin"`` on macOS, otherwise ``"other"`` (Windows/Linux).
        _pyautogui: Module reference (defaults to ``pyautogui``).
    """

    def __init__(self, pyautogui_module: Any | None = None) -> None:
        """Initialize the adapter.

        Args:
            pyautogui_module: Optional module replacement used in tests.
                Defaults to the real :mod:`pyautogui`.
        """
        self.os_name = "darwin" if platform.system() == "Darwin" else "other"
        self._pyautogui = pyautogui_module or pyautogui

    # ------------------------------------------------------------------ scroll

    def scroll_up(self, amount: float) -> None:
        """Scroll up by ``amount`` pixels worth of motion.

        Args:
            amount: Pixel magnitude of the scroll.
        """
        if amount <= 0:
            return
        self._pyautogui.scroll(int(amount))

    def scroll_down(self, amount: float) -> None:
        """Scroll down by ``amount`` pixels worth of motion.

        Args:
            amount: Pixel magnitude of the scroll.
        """
        if amount <= 0:
            return
        self._pyautogui.scroll(-int(amount))

    # ------------------------------------------------------------------ tabs

    def next_tab(self) -> None:
        """Switch to the next browser tab.

        Uses ``Ctrl+Tab`` on every platform (browser-native tab switcher).
        """
        self._hotkey("ctrl", "tab")

    def previous_tab(self) -> None:
        """Switch to the previous browser tab.

        Uses ``Ctrl+Shift+Tab``.
        """
        self._hotkey("ctrl", "shift", "tab")

    # ------------------------------------------------------------------ navigation

    def browser_back(self) -> None:
        """Trigger the browser back action.

        ``Alt+Left`` on Windows/Linux, ``Command+Left`` on macOS.
        """
        if self.os_name == "darwin":
            self._hotkey("command", "left")
        else:
            self._hotkey("alt", "left")

    def browser_forward(self) -> None:
        """Trigger the browser forward action.

        ``Alt+Right`` on Windows/Linux, ``Command+Right`` on macOS.
        """
        if self.os_name == "darwin":
            self._hotkey("command", "right")
        else:
            self._hotkey("alt", "right")

    # ------------------------------------------------------------------ close

    def close_tab(self) -> None:
        """Close the current browser tab.

        ``Ctrl+W`` on Windows/Linux, ``Command+W`` on macOS.
        """
        if self.os_name == "darwin":
            self._hotkey("command", "w")
        else:
            self._hotkey("ctrl", "w")

    # ------------------------------------------------------------------ helpers

    def _hotkey(self, *keys: str) -> None:
        """Wrapper around ``pyautogui.hotkey`` for testability."""
        self._pyautogui.hotkey(*keys)
