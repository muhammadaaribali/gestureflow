"""Convert one MediaPipe hand into the ordered GestureFlow model inputs."""
import numpy as np

FEATURE_NAMES = [f"{axis}{i}" for i in range(1, 21) for axis in ("x", "y")]


def landmarks_to_features(landmarks, width, height, mirror=False):
    """Return 40 wrist-relative coordinates, or None for an unusable hand."""
    if landmarks is None or len(landmarks) != 21 or width <= 0 or height <= 0:
        return None
    coords = np.array([(point.x, point.y) for point in landmarks], dtype=float)
    if not np.isfinite(coords).all():
        return None
    pixels = np.trunc(coords * [width, height])
    offsets = pixels - pixels[0]
    scale = np.abs(offsets).max()
    if scale == 0:
        return None
    features = offsets[1:] / scale
    if mirror:
        features[:, 0] *= -1
    return features.ravel()
