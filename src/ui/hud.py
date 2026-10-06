"""Floating HUD — visual feedback for current interaction state.

The HUD subscribes to gesture events and browser intents. It uses
Tkinter to draw a small, always-on-top, frameless window. The class
exposes a non-blocking ``process_events`` method so the main loop can
pump the UI without freezing.

Tests should pass a ``tk_module`` stub (or omit Tk entirely) to avoid
requiring a display.

Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from config.config import HUDConfig


@dataclass
class HUDState:
    """Snapshot of HUD state for tests and introspection."""

    interaction_state: str
    feedback_message: str | None
    feedback_expiry: float | None


class HUD:
    """Floating, frameless, always-on-top Tkinter window."""

    def __init__(
        self,
        config: HUDConfig,
        tk_module: Any | None = None,
        clock: Any | None = None,
    ) -> None:
        """Initialize the HUD.

        Args:
            config: HUD configuration (size, opacity, position).
            tk_module: Optional Tkinter module replacement for tests.
                When ``None``, the real ``tkinter`` module is imported
                lazily on :meth:`start`.
            clock: Optional monotonic clock. Defaults to ``time.monotonic``.
        """
        self.config = config
        self._tk = tk_module
        self._clock = clock or time.monotonic

        self._root: Any | None = None
        self._state_label: Any | None = None
        self._feedback_label: Any | None = None
        self._started: bool = False

        self._interaction_state: str = "idle"
        self._feedback_message: str | None = None
        self._feedback_expiry: float | None = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Create the Tk window. Idempotent.

        When ``tk_module`` was provided at construction, the HUD uses it
        for unit tests; otherwise it imports the real ``tkinter`` module.
        """
        if self._started:
            return

        if self._tk is None:
            import tkinter as tk

            self._tk = tk

        self._root = self._tk.Tk()
        self._configure_window(self._root)
        self._build_widgets(self._root)
        self._started = True
        self._refresh_labels()

    def stop(self) -> None:
        """Destroy the Tk window if it exists."""
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass
        self._root = None
        self._state_label = None
        self._feedback_label = None
        self._started = False

    def process_events(self) -> None:
        """Pump the Tk event loop without blocking.

        When the Tk root has not been started (test mode), this is a no-op.
        """
        if self._root is None:
            return
        try:
            self._root.update_idletasks()
            self._root.update()
        except Exception:
            return
        self._refresh_labels()

    # ------------------------------------------------------------------ state

    def update_state(self, state: str) -> None:
        """Update the interaction state label.

        Args:
            state: One of ``"idle"``, ``"clutch_engaged"``, ``"clutch_active"``.
        """
        self._interaction_state = state
        self._refresh_labels()

    def show_feedback(self, message: str, duration: float | None = None) -> None:
        """Display a feedback message for ``duration`` seconds.

        Args:
            message: Feedback text (e.g. ``"Next Tab"``).
            duration: Optional override. When ``None``, the configured
                ``feedback_duration`` is used.
        """
        self._feedback_message = message
        if duration is None:
            duration = self.config.feedback_duration
        self._feedback_expiry = self._clock() + duration
        self._refresh_labels()

    def get_state(self) -> HUDState:
        """Return the current HUD state snapshot."""
        return HUDState(
            interaction_state=self._interaction_state,
            feedback_message=self._feedback_message,
            feedback_expiry=self._feedback_expiry,
        )

    # ------------------------------------------------------------------ helpers

    def _configure_window(self, root: Any) -> None:
        root.title("Conductor")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", self.config.opacity)
        except Exception:
            # Some platforms (or stub roots) don't support alpha.
            pass
        root.geometry(f"{self.config.window_width}x{self.config.window_height}+10+10")

    def _build_widgets(self, root: Any) -> None:
        self._state_label = self._tk.Label(root, text="Idle", font=("Arial", 14))
        self._state_label.pack(pady=2)
        self._feedback_label = self._tk.Label(root, text="", font=("Arial", 10))
        self._feedback_label.pack(pady=2)

    def _refresh_labels(self) -> None:
        if self._state_label is not None:
            label = self._state_label_for(self._interaction_state)
            try:
                self._state_label.config(text=label)
            except Exception:
                pass

        if self._feedback_label is not None:
            now = self._clock()
            if self._feedback_message is not None and (
                self._feedback_expiry is None or now < self._feedback_expiry
            ):
                try:
                    self._feedback_label.config(text=self._feedback_message)
                except Exception:
                    pass
            else:
                self._feedback_message = None
                self._feedback_expiry = None
                try:
                    self._feedback_label.config(text="")
                except Exception:
                    pass

    @staticmethod
    def _state_label_for(state: str) -> str:
        return {
            "idle": "Idle",
            "clutch_engaged": "Clutch Engaged",
            "clutch_active": "Clutch Active",
        }.get(state, state.title())
