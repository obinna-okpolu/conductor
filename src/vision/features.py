"""Compute normalized geometric features from hand landmarks.

All distances are normalized by a hand-scale reference (max wrist-to-fingertip
distance), and all finger-extension measurements are derived from joint
geometry rather than raw image coordinates.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
"""

from __future__ import annotations

import math
from typing import Mapping

from src.models import HandLandmarks, Landmark, NormalizedFeatures

# MediaPipe landmark indices for the five fingertips and their MCP joints.
_TIPS: dict[str, int] = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
_MCPS: dict[str, int] = {
    "thumb": 2,
    "index": 5,
    "middle": 9,
    "ring": 13,
    "pinky": 17,
}
_PIPS: dict[str, int] = {
    "thumb": 3,
    "index": 6,
    "middle": 10,
    "ring": 14,
    "pinky": 15,
}
_WRIST = 0


class FeatureExtractor:
    """Compute :class:`NormalizedFeatures` from :class:`HandLandmarks`.

    Pure function — holds no per-frame state. Geometry is derived purely
    from the landmark positions so it is independent of image orientation
    or camera distance (after the hand-scale normalization step).
    """

    def __init__(self) -> None:
        # No configuration knobs in this implementation; kept as a class
        # so future tuning parameters (e.g., joint-angle thresholds) can
        # be added without changing the call sites.
        pass

    # ------------------------------------------------------------------ public

    def extract(
        self, landmarks: HandLandmarks, timestamp: float = 0.0
    ) -> NormalizedFeatures:
        """Compute normalized features from a single hand-landmark set.

        Args:
            landmarks: 21 hand landmarks from MediaPipe.

        Returns:
            :class:`NormalizedFeatures` populated from the geometry.
        """
        points = landmarks.landmarks
        if len(points) < 21:
            raise ValueError(f"Expected 21 landmarks, got {len(points)}")

        scale = self.compute_hand_scale(points)
        # Guard against zero (collapsed hand) features which would divide-by-zero
        # downstream. A small floor keeps the geometry well-conditioned.
        if scale <= 1e-9:
            scale = 1e-9

        center = self._compute_hand_center(points)
        pinch = self._compute_pinch_distance(points, scale)
        extensions = self._compute_finger_extensions(points, scale)
        openness = self._compute_openness_score(extensions)

        return NormalizedFeatures(
            pinch_distance=pinch,
            hand_center=center,
            finger_extensions=extensions,
            openness_score=openness,
            hand_scale=scale,
            frame_id=landmarks.frame_id,
            timestamp=timestamp,
        )

    def compute_hand_scale(self, landmarks: list[Landmark] | HandLandmarks) -> float:
        """Return the hand-scale reference: max wrist-to-fingertip distance.

        Args:
            landmarks: Either a :class:`HandLandmarks` or a list of 21
                :class:`Landmark` instances.

        Returns:
            The largest distance from the wrist to any fingertip, in
            normalized MediaPipe coordinates.
        """
        points = self._as_landmark_list(landmarks)
        wrist = points[_WRIST]
        scale = 0.0
        for tip_index in _TIPS.values():
            d = self._distance(wrist, points[tip_index])
            if d > scale:
                scale = d
        return float(scale)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _as_landmark_list(
        landmarks: list[Landmark] | HandLandmarks,
    ) -> list[Landmark]:
        if isinstance(landmarks, HandLandmarks):
            return list(landmarks.landmarks)
        return list(landmarks)

    @staticmethod
    def _distance(a: Landmark, b: Landmark) -> float:
        dx = a.x - b.x
        dy = a.y - b.y
        dz = a.z - b.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _compute_hand_center(self, points: list[Landmark]) -> tuple[float, float]:
        cx = sum(p.x for p in points) / len(points)
        cy = sum(p.y for p in points) / len(points)
        return (cx, cy)

    def _compute_pinch_distance(self, points: list[Landmark], scale: float) -> float:
        thumb_tip = points[_TIPS["thumb"]]
        middle_tip = points[_TIPS["middle"]]
        return self._distance(thumb_tip, middle_tip) / scale

    def _compute_finger_extensions(
        self, points: list[Landmark], scale: float
    ) -> dict[str, float]:
        """Estimate per-finger extension from normalized joint geometry.

        For each finger we compute the ratio between the distance from the
        MCP joint to the tip and the distance from the MCP joint to the PIP
        joint. A straight finger yields a ratio close to 2 (tip far from
        MCP); a curled finger yields a ratio close to 1 (tip near MCP).

        The ratio is mapped to ``[0, 1]`` via a clamp. This intentionally
        avoids raw Y-coordinate comparisons so the geometry is invariant
        to hand orientation in the image.
        """
        extensions: dict[str, float] = {}
        for name in _TIPS:
            mcp = points[_MCPS[name]]
            pip = points[_PIPS[name]]
            tip = points[_TIPS[name]]

            mcp_to_pip = self._distance(mcp, pip) + 1e-9
            mcp_to_tip = self._distance(mcp, tip)

            ratio = mcp_to_tip / (2.0 * mcp_to_pip)
            # Account for hand scale so the extension score remains
            # comparable across camera distances.
            extensions[name] = float(max(0.0, min(1.0, ratio * scale)))

        return extensions

    @staticmethod
    def _compute_openness_score(extensions: Mapping[str, float]) -> float:
        if not extensions:
            return 0.0
        return sum(extensions.values()) / len(extensions)
