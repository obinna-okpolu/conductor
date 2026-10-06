"""Property-based tests for hand feature extraction."""

from hypothesis import given, strategies as st

from src.models import HandLandmarks, Landmark
from src.vision.features import FeatureExtractor


_coordinate = st.floats(
    min_value=-1.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)
_landmark = st.tuples(_coordinate, _coordinate, _coordinate)


@given(st.lists(_landmark, min_size=21, max_size=21))
def test_pinch_distance_is_non_negative(points: list[tuple[float, float, float]]) -> None:
    """The normalized distance between the thumb and middle finger cannot be negative."""
    landmarks = HandLandmarks(
        landmarks=[Landmark(x=x, y=y, z=z) for x, y, z in points],
        handedness="Right",
        frame_id=1,
    )

    features = FeatureExtractor().extract(landmarks)

    assert features.pinch_distance >= 0.0
