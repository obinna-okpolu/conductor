"""Unit tests for the feature extractor.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
"""

from __future__ import annotations

import pytest

from src.models import HandLandmarks, Landmark
from src.vision.features import FeatureExtractor


# ---------------------------------------------------------------- helpers


def _make_landmarks(
    *,
    thumb_tip: tuple[float, float, float] = (0.7, 0.5, 0.0),
    middle_tip: tuple[float, float, float] = (0.7, 0.5, 0.0),
    curled: bool = False,
    scale: float = 1.0,
) -> HandLandmarks:
    """Construct synthetic 21-landmark sets for tests.

    All non-tip landmarks default to ``(0.5, 0.5)`` so the centroid is
    trivially centered; the fingertip positions drive the only interesting
    geometry. ``scale`` controls how far the fingertips sit from the
    MCPs (i.e., hand_scale).
    """
    points: list[Landmark] = [Landmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]

    # Wrist at center.
    points[0] = Landmark(x=0.5, y=0.5, z=0.0)

    # Distant MCPs to give a measurable hand_scale proportional to ``scale``.
    for idx in (1, 2, 3):  # thumb CMC/MCP/IP
        points[idx] = Landmark(x=0.5, y=0.5, z=0.0)
    for idx in (5, 6, 7):  # index MCP/PIP/DIP
        points[idx] = Landmark(x=0.5, y=0.5, z=0.0)
    for idx in (9, 10, 11):  # middle MCP/PIP/DIP
        points[idx] = Landmark(x=0.5, y=0.5, z=0.0)
    for idx in (13, 14, 15):  # ring MCP/PIP/DIP
        points[idx] = Landmark(x=0.5, y=0.5, z=0.0)
    for idx in (17, 18, 19):  # pinky MCP/PIP/DIP
        points[idx] = Landmark(x=0.5, y=0.5, z=0.0)

    # Override the requested tips.
    points[4] = Landmark(*thumb_tip)
    points[12] = Landmark(*middle_tip)

    # When not curled, place the other fingertips along an extended axis
    # at distance ``scale`` from the wrist so hand_scale reflects ``scale``.
    if not curled:
        # Place remaining fingertips at ``scale`` normalized units away.
        points[8] = Landmark(x=0.5 - scale, y=0.5, z=0.0)
        points[16] = Landmark(x=0.5, y=0.5 + scale, z=0.0)
        points[20] = Landmark(x=0.5 + scale, y=0.5, z=0.0)
    else:
        # Pull every fingertip onto its MCP for the fist simulation.
        for name, tip_idx in (
            ("thumb", 4),
            ("index", 8),
            ("middle", 12),
            ("ring", 16),
            ("pinky", 20),
        ):
            mcp = points[
                {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}[name]
            ]
            points[tip_idx] = Landmark(x=mcp.x, y=mcp.y, z=mcp.z)

    return HandLandmarks(landmarks=points, handedness="Right", frame_id=1)


# ---------------------------------------------------------------- tests


def test_extract_returns_features_with_expected_keys() -> None:
    landmarks = _make_landmarks()
    features = FeatureExtractor().extract(landmarks)

    assert set(features.finger_extensions.keys()) == {
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
    }
    assert isinstance(features.pinch_distance, float)
    assert isinstance(features.openness_score, float)
    # Centroid is dominated by landmarks at (0.5, 0.5); fingertips shift it
    # slightly. Generous tolerance acknowledges the synthetic layout.
    assert features.hand_center[0] == pytest.approx(0.5, abs=0.05)
    assert features.hand_center[1] == pytest.approx(0.5, abs=0.05)


def test_pinch_distance_increases_when_fingertips_far_apart() -> None:
    extractor = FeatureExtractor()

    close = extractor.extract(
        _make_landmarks(thumb_tip=(0.55, 0.5, 0.0), middle_tip=(0.55, 0.5, 0.0))
    )
    far = extractor.extract(
        _make_landmarks(thumb_tip=(0.4, 0.5, 0.0), middle_tip=(0.7, 0.5, 0.0))
    )

    assert far.pinch_distance > close.pinch_distance


def test_pinch_distance_zero_when_tips_coincide() -> None:
    features = FeatureExtractor().extract(
        _make_landmarks(thumb_tip=(0.5, 0.5, 0.0), middle_tip=(0.5, 0.5, 0.0))
    )
    assert features.pinch_distance == pytest.approx(0.0)


def test_openness_score_decreases_when_fingers_curled() -> None:
    extractor = FeatureExtractor()
    open_ = extractor.extract(_make_landmarks())
    closed = extractor.extract(_make_landmarks(curled=True))

    assert closed.openness_score < open_.openness_score


def test_compute_hand_scale_uses_max_wrist_to_fingertip() -> None:
    landmarks = _make_landmarks(scale=2.0)
    scale = FeatureExtractor().compute_hand_scale(landmarks)

    # All five fingertips sit 2.0 normalized units from the wrist in the
    # synthetic data, so the max distance (== hand_scale) is 2.0.
    assert scale == pytest.approx(2.0, abs=1e-3)


def test_hand_scale_reflects_overall_hand_size() -> None:
    small = FeatureExtractor().compute_hand_scale(_make_landmarks(scale=0.5))
    large = FeatureExtractor().compute_hand_scale(_make_landmarks(scale=2.0))
    assert large > small


def test_extract_handles_collapsed_hand_gracefully() -> None:
    """All landmarks at the wrist should not raise (no division by zero)."""
    points = [Landmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    landmarks = HandLandmarks(landmarks=points, handedness="Right", frame_id=1)

    features = FeatureExtractor().extract(landmarks)

    # hand_scale floored to epsilon, pinch_distance stays finite.
    assert features.hand_scale > 0
    assert features.pinch_distance >= 0


def test_extract_raises_on_wrong_landmark_count() -> None:
    short = HandLandmarks(
        landmarks=[Landmark(0, 0, 0) for _ in range(10)],
        handedness="Right",
        frame_id=1,
    )
    with pytest.raises(ValueError):
        FeatureExtractor().extract(short)


def test_finger_extensions_within_unit_interval() -> None:
    features = FeatureExtractor().extract(_make_landmarks())
    for name, value in features.finger_extensions.items():
        assert 0.0 <= value <= 1.0, f"{name} extension {value} out of range"


def test_openness_score_is_mean_of_extensions() -> None:
    features = FeatureExtractor().extract(_make_landmarks())
    expected = sum(features.finger_extensions.values()) / len(
        features.finger_extensions
    )
    assert features.openness_score == pytest.approx(expected)
