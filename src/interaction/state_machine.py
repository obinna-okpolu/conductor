"""Gesture lifecycle state machine — nav-mode, open-palm close, swipe, plus safety.

Nav-mode is engaged by ``VICTORY`` (held) and disengaged by ``OPEN_PALM``
(held). While engaged, the main loop calls :meth:`update_nav_active` per
frame to translate incremental hand movement into scroll/tab-switch intents.

An open palm held while idle requires confirmation before emitting the
``tab_close`` event. Open palm in navigation mode releases navigation.

Swipe uses a re-arm pattern — once a swipe fires, the hand must return to
a roughly central position before another swipe can trigger.

Safety suppression (low confidence, no hand, tracking jump, stale frame,
cooldown) is applied uniformly across all gestures.

Requirements: 5.x, 9.x, 12.x, 14.x
"""

from __future__ import annotations

import math
import time
from typing import Any

from config.config import StateMachineConfig
from src.models import (
    DetectionResult,
    FistState,
    GestureCandidate,
    GestureEvent,
    GestureType,
    NavModeState,
    StateMachineState,
    SwipeState,
)


# Required hold duration (seconds) before Nav-mode engages/disengages.
_NAV_HOLD_DURATION = 0.5


class GestureStateMachine:
    """Track gesture lifecycle and produce semantic events."""

    def __init__(self, config: StateMachineConfig, clock: Any | None = None) -> None:
        self.config = config
        self._clock = clock or time.monotonic

        self._nav_state: NavModeState = NavModeState.IDLE
        self._fist_state: FistState = FistState.NEUTRAL
        self._swipe_state: SwipeState = SwipeState.NEUTRAL

        self._suppressed: bool = False
        self._suppression_reason: str | None = None
        self._cooldown_end: float = 0.0

        self._nav_engage_position: tuple[float, float] | None = None
        self._fist_hold_start: float | None = None
        self._nav_hold_start: float | None = None
        self._last_valid_position: tuple[float, float] | None = None
        self._last_valid_timestamp: float = 0.0

        self._swipe_neutral_position: tuple[float, float] | None = None
        self._fist_rearm_openness: float = 1.0
        self._last_thumb_gesture: GestureType = GestureType.NONE

    # ------------------------------------------------------------------ public

    def process(
        self, candidate: GestureCandidate, detection: DetectionResult
    ) -> list[GestureEvent]:
        """Process one (candidate, detection) pair and emit semantic events."""
        if self._is_suppressed(detection, candidate):
            return []

        events: list[GestureEvent] = []
        hand_pos = self._hand_position(candidate)

        events.extend(self._process_nav_mode(candidate, hand_pos))
        events.extend(self._process_close_gesture(candidate, hand_pos))
        events.extend(self._process_swipe(candidate, hand_pos))
        events.extend(self._process_thumb(candidate, hand_pos))

        if candidate.features is not None:
            self._last_valid_position = candidate.features.hand_center
            self._last_valid_timestamp = candidate.timestamp

        return events

    def get_state(self) -> StateMachineState:
        """Return the current state snapshot for introspection."""
        return StateMachineState(
            nav_mode_state=self._nav_state,
            fist_state=self._fist_state,
            swipe_state=self._swipe_state,
            suppressed=self._suppressed,
            suppression_reason=self._suppression_reason,
            nav_engage_position=self._nav_engage_position,
            fist_hold_start=self._fist_hold_start,
            last_valid_position=self._last_valid_position,
            last_valid_timestamp=self._last_valid_timestamp,
        )

    def reset(self) -> None:
        """Reset every sub-state machine and clear tracking baseline."""
        self._nav_state = NavModeState.IDLE
        self._fist_state = FistState.NEUTRAL
        self._swipe_state = SwipeState.NEUTRAL
        self._suppressed = False
        self._suppression_reason = None
        self._cooldown_end = 0.0
        self._nav_engage_position = None
        self._fist_hold_start = None
        self._nav_hold_start = None
        self._last_valid_position = None
        self._last_valid_timestamp = 0.0
        self._swipe_neutral_position = None
        self._fist_rearm_openness = 1.0
        self._last_thumb_gesture = GestureType.NONE

    # ------------------------------------------------------------------ safety

    def _is_suppressed(
        self, detection: DetectionResult, candidate: GestureCandidate
    ) -> bool:
        """Run all suppression gates. Mutates suppression bookkeeping."""
        if detection is None:
            return False

        if detection.landmarks is None:
            self._suppressed = True
            self._suppression_reason = "no_hand"
            self.reset()
            return True

        if detection.confidence < self.config.confidence_threshold:
            self._suppressed = True
            self._suppression_reason = "low_confidence"
            return True

        if self._last_valid_position is not None and candidate.features is not None:
            jump = _distance(candidate.features.hand_center, self._last_valid_position)
            if jump > self.config.tracking_jump_threshold:
                self._suppressed = True
                self._suppression_reason = "tracking_jump"
                self._cooldown_end = self._clock() + self.config.tracking_loss_cooldown
                self._last_valid_position = None
                return True

        if (
            candidate.timestamp is not None
            and detection.timestamp is not None
            and abs(candidate.timestamp - detection.timestamp)
            > self.config.staleness_threshold
        ):
            self._suppressed = True
            self._suppression_reason = "stale_frame"
            return True

        if self._suppressed and self._clock() < self._cooldown_end:
            return True

        self._suppressed = False
        self._suppression_reason = None
        return False

    # ------------------------------------------------------------------ nav-mode

    def _process_nav_mode(
        self, candidate: GestureCandidate, hand_pos: tuple[float, float]
    ) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        gt = candidate.gesture_type
        now = candidate.timestamp if candidate.timestamp is not None else self._clock()

        if gt == GestureType.VICTORY:
            if self._nav_state == NavModeState.IDLE:
                if self._nav_hold_start is None:
                    self._nav_hold_start = now
                elif now - self._nav_hold_start >= _NAV_HOLD_DURATION:
                    self._nav_state = NavModeState.ENGAGED
                    self._nav_engage_position = hand_pos
                    self._nav_hold_start = None
                    events.append(self._make_event("nav_engage", candidate))
            elif self._nav_state in (NavModeState.ENGAGED, NavModeState.ACTIVE):
                self._nav_state = NavModeState.ACTIVE
                self._nav_hold_start = None

        elif gt == GestureType.OPEN_PALM:
            if self._nav_state in (NavModeState.ENGAGED, NavModeState.ACTIVE):
                if self._nav_hold_start is None:
                    self._nav_hold_start = now
                elif now - self._nav_hold_start >= _NAV_HOLD_DURATION:
                    self._nav_state = NavModeState.IDLE
                    self._nav_engage_position = None
                    self._nav_hold_start = None
                    events.append(self._make_event("nav_release", candidate))
            else:
                self._nav_hold_start = None

        else:
            self._nav_hold_start = None

        return events

    # ---------------------------------------------------------- close gesture

    def _process_close_gesture(
        self, candidate: GestureCandidate, hand_pos: tuple[float, float]
    ) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        gt = candidate.gesture_type
        now = candidate.timestamp if candidate.timestamp is not None else self._clock()

        # Open palm closes a tab while idle. In navigation mode open palm is
        # reserved for nav_release and must never close a tab. CLOSED_FIST is
        # accepted only as a legacy programmatic candidate; the live
        # recognizer no longer emits it.
        close_candidate = gt == GestureType.CLOSED_FIST or (
            gt == GestureType.OPEN_PALM and self._nav_state == NavModeState.IDLE
        )

        if close_candidate:
            if self._fist_state == FistState.NEUTRAL:
                self._fist_state = FistState.CANDIDATE
                self._fist_hold_start = now
            if self._fist_state == FistState.CANDIDATE:
                if (
                    self._fist_hold_start is not None
                    and now - self._fist_hold_start >= self.config.fist_hold_duration
                ):
                    self._fist_state = FistState.CONFIRMED
                    events.append(self._make_event("tab_close", candidate))
                    self._fist_state = FistState.WAIT_FOR_REARM
                    self._fist_rearm_openness = (
                        candidate.features.openness_score
                        if candidate.features is not None
                        else 1.0
                    )
        else:
            if self._fist_state == FistState.CANDIDATE:
                self._fist_state = FistState.NEUTRAL
                self._fist_hold_start = None
            elif self._fist_state == FistState.WAIT_FOR_REARM:
                openness = (
                    candidate.features.openness_score
                    if candidate.features is not None
                    else 0.0
                )
                if openness > self._fist_rearm_openness + 0.2:
                    self._fist_state = FistState.NEUTRAL
                    self._fist_rearm_openness = 1.0
        return events

    # ------------------------------------------------------------------ swipe

    def _process_swipe(
        self, candidate: GestureCandidate, hand_pos: tuple[float, float]
    ) -> list[GestureEvent]:
        if candidate.gesture_type not in (
            GestureType.SWIPE_LEFT,
            GestureType.SWIPE_RIGHT,
        ):
            return []
        events: list[GestureEvent] = []
        if self._swipe_state == SwipeState.NEUTRAL:
            event_type = (
                "swipe_left"
                if candidate.gesture_type == GestureType.SWIPE_LEFT
                else "swipe_right"
            )
            self._swipe_state = SwipeState.TRIGGERED
            events.append(self._make_event(event_type, candidate))
            self._swipe_state = SwipeState.WAIT_FOR_REARM
            self._swipe_neutral_position = hand_pos
        elif self._swipe_state == SwipeState.WAIT_FOR_REARM:
            if self._swipe_neutral_position is not None:
                if _distance(hand_pos, self._swipe_neutral_position) < 0.05:
                    self._swipe_state = SwipeState.NEUTRAL
                    self._swipe_neutral_position = None
        return events

    # ------------------------------------------------------------------ thumb

    def _process_thumb(
        self, candidate: GestureCandidate, hand_pos: tuple[float, float]
    ) -> list[GestureEvent]:
        if candidate.gesture_type not in (GestureType.THUMB_UP, GestureType.THUMB_DOWN):
            self._last_thumb_gesture = GestureType.NONE
            return []
        if candidate.gesture_type == self._last_thumb_gesture:
            return []
        self._last_thumb_gesture = candidate.gesture_type
        event_type = (
            "thumb_up"
            if candidate.gesture_type == GestureType.THUMB_UP
            else "thumb_down"
        )
        return [self._make_event(event_type, candidate)]

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _hand_position(candidate: GestureCandidate) -> tuple[float, float]:
        if candidate.features is not None:
            return candidate.features.hand_center
        return (0.0, 0.0)

    def _make_event(
        self,
        event_type: str,
        candidate: GestureCandidate,
        *,
        extra: dict[str, Any] | None = None,
    ) -> GestureEvent:
        metadata: dict[str, Any] = {}
        if candidate.swipe_metadata is not None:
            metadata["swipe_metadata"] = candidate.swipe_metadata
        if extra:
            metadata.update(extra)
        return GestureEvent(
            event_type=event_type,
            frame_id=candidate.frame_id,
            timestamp=candidate.timestamp,
            hand_position=self._hand_position(candidate),
            metadata=metadata,
        )


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
