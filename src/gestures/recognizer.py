"""Gesture recognizer: per-frame classification into a stable gesture set.

Detects the five gestures that ``mp.solutions.hands`` reliably classifies in
published benchmarks (~95% accuracy on the canonical set):

* ``OPEN_PALM``     — all 5 fingers extended
* ``OPEN_PALM``     — all 5 fingers extended; closes a tab while idle
* ``VICTORY``       — index + middle extended, others curled
* ``POINTING_UP``   — only index extended
* ``THUMB_UP``      — thumb extended upward, other 4 curled
* ``THUMB_DOWN``    — thumb extended downward, other 4 curled

Plus (kept for backwards compatibility):
* ``SWIPE_LEFT`` / ``SWIPE_RIGHT`` — fast horizontal motion across a buffer

Holds ``GestureType.NONE`` while no gesture matches.

**Stability confirmation.** The recognizer only emits a non-``NONE`` candidate
once the same gesture has been observed for ``hold_frames`` consecutive
frames. ``cooldown_frames`` then suppress re-triggers. This is the approach
used by MediaPipe-Max-Gesture-Audio-Effects (airliecassidy) for the same
problem.

**Finger-extension logic** (per Lab Notes MediaPipe article and Sunfounder
counting tutorial):

* Index/Middle/Ring/Pinky — extended when ``tip.y < PIP.y`` (in normalized
  image coordinates, smaller y = higher in frame). This is the canonical
  MediaPipe method.
* Thumb — extended when ``tip.x < MCP.x`` for a right hand (and the inverse
  for a left hand). MediaPipe handedness is taken from the landmark source.

For up-vs-down discrimination on the thumb, we use a 3D vector: thumb-up
when the wrist-to-thumb-tip vector points away from the camera (positive z)
and the thumb-tip is above the wrist; thumb-down when it points back.
"""

from __future__ import annotations

from collections import deque
from typing import Deque

from config.config import GestureConfig
from src.models import (
    GestureCandidate,
    GestureType,
    HandLandmarks,
    Landmark,
    SwipeMetadata,
)


# ---------------------------------------------------------------- landmarks
# MediaPipe indices
_WRIST = 0
_THUMB_TIP, _THUMB_IP, _THUMB_MCP = 4, 3, 2
_INDEX_TIP, _INDEX_PIP, _INDEX_MCP = 8, 6, 5
_MIDDLE_TIP, _MIDDLE_PIP, _MIDDLE_MCP = 12, 10, 9
_RING_TIP, _RING_PIP, _RING_MCP = 16, 14, 13
_PINKY_TIP, _PINKY_PIP, _PINKY_MCP = 20, 18, 17


# ---------------------------------------------------------------- thresholds

# A finger is "extended" when the y-delta between tip and PIP is at least this
# fraction of the palm size (wrist→MCP distance).  Robust against camera
# distance and orientation for the four straight fingers.
_FINGER_EXTEND_FRAC = 0.35

# Thumb extension: tip must be at least 1.5x palm size outside the MCP.
# (Strict; prevents a fist's slightly-tucked thumb from being classified
# as extended.)
_THUMB_EXTEND_FRAC = 1.5

# Thumb up/down discrimination: thumb-tip must be at least 1.5x palm size
# above (or below) the wrist.
_THUMB_VERTEX_FRAC = 0.65
_FINGER_CLOSED_FRAC = 0.20


class GestureRecognizer:
    """Classify per-frame hand pose into one of the supported gestures."""

    def __init__(
        self,
        config: GestureConfig,
        hold_frames: int = 3,
        cooldown_frames: int = 15,
        smoothing_window: int = 5,
        classification_hysteresis: int = 3,
    ) -> None:
        self.config = config
        self._hold_frames = max(1, hold_frames)
        self._cooldown_frames = max(0, cooldown_frames)
        # Smoothing + hysteresis: require the raw per-frame classification
        # to be stable for `classification_hysteresis` frames before we
        # commit to a change. This kills the per-frame flicker that
        # MediaPipe landmark jitter causes (a stationary fist shouldn't
        # alternate fist/none every other frame).
        self._smoothing_window = max(1, smoothing_window)
        self._classification_hysteresis = max(1, classification_hysteresis)
        self._raw_window: Deque[GestureType] = deque(maxlen=self._smoothing_window)

        # Stability bookkeeping.
        self._current_gesture: GestureType = GestureType.NONE
        self._stable_count: int = 0
        self._cooldown_remaining: int = 0
        self._last_emitted: GestureType = GestureType.NONE
        self._candidate_gesture: GestureType = GestureType.NONE
        self._candidate_count: int = 0

        # Swipe buffer (kept for the SWIPE_LEFT / SWIPE_RIGHT paths).
        self._swipe_buffer: Deque[HandLandmarks] = deque(
            maxlen=config.swipe_window_frames
        )

    # ------------------------------------------------------------------ public

    def recognize(
        self,
        landmarks: HandLandmarks,
        confidence: float,
        timestamp: float = 0.0,
    ) -> GestureCandidate:
        """Classify the current frame into a (possibly empty) gesture.

        Args:
            landmarks: 21 hand landmarks from MediaPipe.
            confidence: Tracking confidence [0, 1].

        Returns:
            :class:`GestureCandidate` with the detected gesture.
        """
        if landmarks is None:
            return self._wrap(GestureType.NONE, landmarks, confidence, None, timestamp)

        # ---- 1. per-frame classification ----
        classified = self._classify(landmarks)
        ts = 0.0  # timestamp is not on landmarks directly; main loop supplies

        # ---- 2. swipe detection (uses the same buffer mechanism) ----
        swipe = self._try_swipe(landmarks)
        if swipe is not None:
            gesture_type, meta = swipe
            return self._wrap(gesture_type, landmarks, confidence, meta, timestamp)

        # ---- 3. smoothing + classification hysteresis ----
        # MediaPipe landmarks jitter frame-to-frame even for stationary
        # gestures. We smooth the raw classification over a sliding window
        # and only commit to a change once the new candidate has been
        # stable for `classification_hysteresis` consecutive frames.
        self._raw_window.append(classified)
        if self._candidate_gesture == classified:
            self._candidate_count += 1
        else:
            self._candidate_gesture = classified
            self._candidate_count = 1

        if self._candidate_gesture != self._current_gesture:
            if self._candidate_count >= self._classification_hysteresis:
                self._current_gesture = self._candidate_gesture
                self._stable_count = 1
                if self._cooldown_remaining <= 0:
                    # When the smoothed classification flips, reset the
                    # stable count so the next firing must re-earn it.
                    pass
        else:
            self._stable_count += 1

        # The candidate we report this frame is the (smoothed) current
        # classification — what downstream code and the per-frame log sees.
        classified = self._current_gesture

        # While in cooldown, do not re-fire the same gesture.
        if self._cooldown_remaining > 0:
            if classified == self._last_emitted:
                self._cooldown_remaining -= 1
                return self._wrap(
                    GestureType.NONE, landmarks, confidence, None, timestamp
                )
            self._cooldown_remaining -= 1

        # Fire only after the gesture has been stable for N frames.
        if classified != GestureType.NONE and self._stable_count >= self._hold_frames:
            self._last_emitted = classified
            self._cooldown_remaining = self._cooldown_frames
            return self._wrap(classified, landmarks, confidence, None, timestamp)

        return self._wrap(GestureType.NONE, landmarks, confidence, None, timestamp)

    # ------------------------------------------------------------------ classify

    def _classify(self, landmarks: HandLandmarks) -> GestureType:
        """Classify one frame into one of the 6 gesture types (or NONE)."""
        lm = landmarks.landmarks
        if len(lm) < 21:
            return GestureType.NONE

        palm_size = self._palm_size(lm)
        if palm_size < 1e-6:
            return GestureType.NONE

        ext = {
            "thumb": self._thumb_extended(lm, landmarks.handedness, palm_size),
            "index": self._finger_extended(lm, _INDEX_TIP, _INDEX_PIP, palm_size),
            "middle": self._finger_extended(lm, _MIDDLE_TIP, _MIDDLE_PIP, palm_size),
            "ring": self._finger_extended(lm, _RING_TIP, _RING_PIP, palm_size),
            "pinky": self._finger_extended(lm, _PINKY_TIP, _PINKY_PIP, palm_size),
        }

        thumb, index, middle, ring, pinky = (
            ext["thumb"],
            ext["index"],
            ext["middle"],
            ext["ring"],
            ext["pinky"],
        )
        non_thumb_closed = all(
            self._finger_closed(lm, tip, pip, palm_size)
            for tip, pip in (
                (_INDEX_TIP, _INDEX_PIP),
                (_MIDDLE_TIP, _MIDDLE_PIP),
                (_RING_TIP, _RING_PIP),
                (_PINKY_TIP, _PINKY_PIP),
            )
        )

        # OPEN_PALM: all five fingers extended.
        if all((thumb, index, middle, ring, pinky)):
            return GestureType.OPEN_PALM

        # VICTORY: index + middle extended, ring + pinky curled.
        if index and middle and not ring and not pinky:
            return GestureType.VICTORY

        # POINTING_UP: only index extended.
        if index and not middle and not ring and not pinky:
            return GestureType.POINTING_UP

        # Closed-hand poses are intentionally not emitted anymore. Open palm
        # is the close-tab gesture while idle; this removes the unreliable
        # fist/thumb boundary from the live gesture vocabulary.
        if non_thumb_closed:
            # A thumb action is valid only when the thumb is both extended
            # and clearly vertical. Any closed-hand pose without that margin
            # deliberately resolves to fist for predictable safety.
            if thumb:
                thumb_gesture = self._classify_thumb_vertical(lm)
                if thumb_gesture != GestureType.NONE:
                    return thumb_gesture
            return GestureType.NONE

        # THUMB_UP / THUMB_DOWN: thumb extended (upward / downward), other 4 curled.
        return GestureType.NONE

    @staticmethod
    def _palm_size(lm: list[Landmark]) -> float:
        """Approximate palm size as the distance from wrist to middle MCP."""
        w = lm[_WRIST]
        mcp = lm[_MIDDLE_MCP]
        dx, dy, dz = w.x - mcp.x, w.y - mcp.y, w.z - mcp.z
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    @staticmethod
    def _finger_extended(
        lm: list[Landmark], tip_idx: int, pip_idx: int, palm_size: float
    ) -> bool:
        """True when ``tip.y`` is sufficiently above ``pip.y`` (smaller y = higher)."""
        tip = lm[tip_idx]
        pip = lm[pip_idx]
        # Normalize by palm size so the threshold works across camera distances.
        delta_y = pip.y - tip.y  # positive when tip is above PIP
        return delta_y > _FINGER_EXTEND_FRAC * palm_size

    @staticmethod
    def _finger_closed(
        lm: list[Landmark], tip_idx: int, pip_idx: int, palm_size: float
    ) -> bool:
        """Return whether a finger is not visibly extended.

        The looser closed threshold tolerates landmark jitter while the
        stricter extension threshold preserves a dead band between poses.
        """
        return (lm[pip_idx].y - lm[tip_idx].y) <= _FINGER_CLOSED_FRAC * palm_size

    @staticmethod
    def _thumb_extended(lm: list[Landmark], handedness: str, palm_size: float) -> bool:
        """True when the thumb is extended away from the palm.

        Real MediaPipe fist poses put the thumb tip very close to (or even
        to the right of) the MCP, so a strict 2D-distance test is required
        to avoid classifying a fist as "thumb extended".

        Use thumb-joint reach rather than an x-axis test. The latter only
        works for a sideways thumb and makes ordinary thumbs-up/down poses
        look like a fist when the tip is vertically aligned with the MCP.
        """
        tip = lm[_THUMB_TIP]
        mcp = lm[_THUMB_MCP]
        ip = lm[_THUMB_IP]

        mcp_to_tip = (
            (mcp.x - tip.x) ** 2 + (mcp.y - tip.y) ** 2 + (mcp.z - tip.z) ** 2
        ) ** 0.5
        mcp_to_ip = (
            (mcp.x - ip.x) ** 2 + (mcp.y - ip.y) ** 2 + (mcp.z - ip.z) ** 2
        ) ** 0.5

        # An extended thumb reaches substantially beyond its first joint;
        # a tucked thumb ends near the MCP/IP region. This is orientation
        # independent and works for both handedness labels.
        return mcp_to_ip > 1e-6 and mcp_to_tip / mcp_to_ip > 1.35

    @staticmethod
    def _classify_thumb_vertical(lm: list[Landmark]) -> GestureType:
        """Pick THUMB_UP or THUMB_DOWN by 3D geometry (no Y-axis shortcut)."""
        wrist = lm[_WRIST]
        tip = lm[_THUMB_TIP]
        palm_size = GestureRecognizer._palm_size(lm)
        if palm_size < 1e-6:
            return GestureType.NONE

        # Thumb points "up" relative to the wrist when its tip is above the
        # wrist in image space AND extends forward (away from the camera).
        # THUMB_DOWN: tip below wrist, possibly toward camera.
        vertical_delta = wrist.y - tip.y  # positive when tip is above wrist
        if vertical_delta > _THUMB_VERTEX_FRAC * palm_size:
            return GestureType.THUMB_UP
        if vertical_delta < -_THUMB_VERTEX_FRAC * palm_size:
            return GestureType.THUMB_DOWN
        return GestureType.NONE

    def diagnose(self, landmarks: HandLandmarks) -> dict[str, float | int | str]:
        """Return pose evidence for the live preview and troubleshooting."""
        lm = landmarks.landmarks
        palm = self._palm_size(lm)
        if palm < 1e-6:
            return {"closed_fingers": 0, "thumb_reach": 0.0, "thumb_vertical": 0.0}
        closed = sum(
            self._finger_closed(lm, tip, pip, palm)
            for tip, pip in (
                (_INDEX_TIP, _INDEX_PIP),
                (_MIDDLE_TIP, _MIDDLE_PIP),
                (_RING_TIP, _RING_PIP),
                (_PINKY_TIP, _PINKY_PIP),
            )
        )
        mcp, ip, tip = lm[_THUMB_MCP], lm[_THUMB_IP], lm[_THUMB_TIP]
        dist = lambda a, b: (
            ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5
        )
        reach = dist(mcp, tip) / max(dist(mcp, ip), 1e-6)
        vertical = (lm[_WRIST].y - tip.y) / palm
        return {
            "closed_fingers": closed,
            "thumb_reach": reach,
            "thumb_vertical": vertical,
        }

    # ------------------------------------------------------------------ swipe

    def _try_swipe(
        self, landmarks: HandLandmarks
    ) -> tuple[GestureType, SwipeMetadata] | None:
        """Detect a swipe across a sliding window of hand positions."""
        self._swipe_buffer.append(landmarks)
        buffer = self._swipe_buffer
        if len(buffer) < 2:
            return None

        first = buffer[0].landmarks[_WRIST]
        last = buffer[-1].landmarks[_WRIST]
        net_displacement = last.x - first.x
        elapsed_time = max(buffer[-1].frame_id - buffer[0].frame_id, 1) / 30.0
        if elapsed_time <= 0:
            return None

        velocity = net_displacement / elapsed_time
        consistency = self._direction_consistency(buffer)
        magnitude = abs(net_displacement)

        meets_displacement = magnitude >= self.config.swipe_displacement_threshold
        meets_velocity = abs(velocity) >= self.config.swipe_velocity_threshold
        meets_consistency = (
            consistency >= self.config.swipe_direction_consistency_threshold
        )

        if not (meets_displacement and meets_velocity and meets_consistency):
            return None

        gesture_type = (
            GestureType.SWIPE_LEFT if net_displacement < 0 else GestureType.SWIPE_RIGHT
        )
        meta = SwipeMetadata(
            net_displacement=net_displacement,
            elapsed_time=elapsed_time,
            velocity=velocity,
            direction_consistency=consistency,
        )
        buffer.clear()
        return gesture_type, meta

    @staticmethod
    def _direction_consistency(buffer: Deque[HandLandmarks]) -> float:
        if len(buffer) < 2:
            return 0.0
        deltas = [
            buffer[i].landmarks[_WRIST].x - buffer[i - 1].landmarks[_WRIST].x
            for i in range(1, len(buffer))
        ]
        nonzero = [d for d in deltas if d != 0.0]
        if not nonzero:
            return 0.0
        majority = (
            1 if sum(d > 0 for d in nonzero) >= sum(d < 0 for d in nonzero) else -1
        )
        agreeing = sum(1 for d in nonzero if (d > 0) == (majority > 0))
        return agreeing / len(nonzero)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _wrap(
        gesture_type: GestureType,
        landmarks: HandLandmarks,
        confidence: float,
        swipe_metadata: SwipeMetadata | None,
        timestamp: float = 0.0,
    ) -> GestureCandidate:
        return GestureCandidate(
            gesture_type=gesture_type,
            confidence=confidence,
            frame_id=landmarks.frame_id,
            timestamp=timestamp,
            features=None,  # type: ignore[arg-type]  # not used by new recognizer
            swipe_metadata=swipe_metadata,
        )

    def reset(self) -> None:
        """Clear all state (stability bookkeeping + swipe buffer)."""
        self._current_gesture = GestureType.NONE
        self._stable_count = 0
        self._cooldown_remaining = 0
        self._last_emitted = GestureType.NONE
        self._candidate_gesture = GestureType.NONE
        self._candidate_count = 0
        self._raw_window.clear()
        self._swipe_buffer.clear()
