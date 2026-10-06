"""Intent resolver — map gesture events to browser intents.

Recognizer + state-machine vocabulary (post-research rework):

| Gesture / event       | Action                              |
|-----------------------|-------------------------------------|
| ``thumb_up``          | ``BROWSER_BACK``                    |
| ``thumb_down``        | ``BROWSER_FORWARD``                 |
| ``swipe_left``        | ``BROWSER_BACK`` (re-arm)           |
| ``swipe_right``       | ``BROWSER_FORWARD`` (re-arm)        |
| ``tab_close``         | ``CLOSE_TAB``                       |
| ``nav_engage``        | (bookkeeping) start tracking scroll |
| ``nav_release``       | (bookkeeping) stop tracking scroll  |

While nav-mode is active, :meth:`update_nav_active` is called each frame and
emits ``SCROLL_*`` and ``NEXT_TAB`` / ``PREVIOUS_TAB`` intents from the
hand's incremental movement.

Requirements: 6.x, 7.x, 10.x, 13.x
"""

from __future__ import annotations

from config.config import IntentConfig
from src.models import (
    BrowserIntent,
    GestureEvent,
    IntentType,
    NormalizedFeatures,
)


class IntentResolver:
    """Translate gesture events and nav-mode updates into browser intents."""

    def __init__(self, config: IntentConfig) -> None:
        self.config = config
        self._previous_position: tuple[float, float] | None = None
        self._nav_engage_position: tuple[float, float] | None = None
        self._nav_active: bool = False
        self._tab_switch_armed: bool = True

    # ------------------------------------------------------------------ public

    def resolve(self, event: GestureEvent) -> list[BrowserIntent]:
        """Resolve a single gesture event into zero or more intents."""
        if event.event_type == "nav_engage":
            self._nav_active = True
            self._nav_engage_position = event.hand_position
            self._previous_position = event.hand_position
            self._tab_switch_armed = True
            return []

        if event.event_type == "nav_release":
            self._nav_active = False
            self._nav_engage_position = None
            self._previous_position = None
            self._tab_switch_armed = True
            return []

        if event.event_type == "thumb_up":
            return [self._make_intent(IntentType.BROWSER_BACK, event)]
        if event.event_type == "thumb_down":
            return [self._make_intent(IntentType.BROWSER_FORWARD, event)]
        if event.event_type == "swipe_left":
            return [self._make_intent(IntentType.BROWSER_BACK, event)]
        if event.event_type == "swipe_right":
            return [self._make_intent(IntentType.BROWSER_FORWARD, event)]
        if event.event_type == "tab_close":
            return [self._make_intent(IntentType.CLOSE_TAB, event)]

        return []

    def update_nav_active(self, features: NormalizedFeatures) -> list[BrowserIntent]:
        """Generate scroll and tab-switch intents while nav-mode is active.

        Called every frame the state machine has nav-mode in ``ENGAGED`` or
        ``ACTIVE``. Uses incremental movement so a stationary hand produces
        no intents.
        """
        intents: list[BrowserIntent] = []
        if not self._nav_active or self._previous_position is None:
            self._previous_position = features.hand_center if self._nav_active else None
            return intents

        prev_x, prev_y = self._previous_position
        curr_x, curr_y = features.hand_center

        # ----- scroll (vertical incremental) -----
        delta_y = curr_y - prev_y
        if abs(delta_y) > self.config.scroll_deadzone:
            amount = (
                abs(delta_y) - self.config.scroll_deadzone
            ) * self.config.scroll_scale_factor
            amount = float(round(amount))
            intent_type = (
                IntentType.SCROLL_DOWN if delta_y > 0 else IntentType.SCROLL_UP
            )
            intents.append(
                BrowserIntent(
                    intent_type=intent_type,
                    magnitude=amount,
                    frame_id=features.frame_id,
                    timestamp=features.timestamp,
                )
            )

        # ----- tab switch (horizontal displacement from engage position) -----
        if self._nav_engage_position is not None:
            engage_x, _ = self._nav_engage_position
            displacement = curr_x - engage_x
            if not self._tab_switch_armed:
                if abs(displacement) <= self.config.tab_switch_rearm_threshold:
                    self._tab_switch_armed = True
            else:
                if displacement >= self.config.tab_switch_threshold:
                    intents.append(
                        BrowserIntent(
                            intent_type=IntentType.NEXT_TAB,
                            magnitude=None,
                            frame_id=features.frame_id,
                            timestamp=features.timestamp,
                        )
                    )
                    self._tab_switch_armed = False
                elif displacement <= -self.config.tab_switch_threshold:
                    intents.append(
                        BrowserIntent(
                            intent_type=IntentType.PREVIOUS_TAB,
                            magnitude=None,
                            frame_id=features.frame_id,
                            timestamp=features.timestamp,
                        )
                    )
                    self._tab_switch_armed = False

        self._previous_position = (curr_x, curr_y)
        return intents

    def reset(self) -> None:
        """Clear all session state."""
        self._previous_position = None
        self._nav_engage_position = None
        self._nav_active = False
        self._tab_switch_armed = True

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _make_intent(intent_type: IntentType, event: GestureEvent) -> BrowserIntent:
        return BrowserIntent(
            intent_type=intent_type,
            magnitude=None,
            frame_id=event.frame_id,
            timestamp=event.timestamp,
        )
