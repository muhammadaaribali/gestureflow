"""Verify the live feature format against the held-out GestureFlow dataset."""
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from features import FEATURE_NAMES, landmarks_to_features


def fake_landmarks(row, width, height):
    # Place the wrist inside a synthetic frame and give the hand a 200-pixel scale.
    offsets = np.vstack(([0, 0], row.reshape(20, 2)))
    coords = ([320, 250] + offsets * 200) / [width, height]
    return [SimpleNamespace(x=x, y=y) for x, y in coords]


def main():
    test = pd.read_csv("data/processed/test.csv")
    labels = test.pop("label")
    model = joblib.load("models/best_model.joblib")
    assert list(test.columns) == FEATURE_NAMES
    assert list(model.feature_names_in_) == FEATURE_NAMES
    print("Column order matches the dataset and saved model:", True)
    original = test.to_numpy()
    rebuilt = []
    mirrored = []
    for row in original:
        hand = fake_landmarks(row, 640, 480)
        rebuilt.append(landmarks_to_features(hand, 640, 480))
        mirrored.append(landmarks_to_features(hand, 640, 480, mirror=True))
    rebuilt = np.asarray(rebuilt)
    assert rebuilt.shape == (len(test), 40)
    assert np.all(np.abs(rebuilt).max(axis=1) == 1)
    np.testing.assert_allclose(np.asarray(mirrored)[:, ::2], -rebuilt[:, ::2])
    np.testing.assert_allclose(np.asarray(mirrored)[:, 1::2], rebuilt[:, 1::2])
    assert landmarks_to_features([SimpleNamespace(x=0.5, y=0.5)] * 21, 640, 480) is None
    error = np.abs(rebuilt - original)
    print(f"Features per hand: 40; overall range: {rebuilt.min():.2f} to {rebuilt.max():.2f}")
    print(f"Round-trip error: max {error.max():.5f}, mean {error.mean():.5f}")
    predictions = model.predict(test)
    after = model.predict(pd.DataFrame(rebuilt, columns=FEATURE_NAMES))
    flipped = model.predict(pd.DataFrame(mirrored, columns=FEATURE_NAMES))
    for name, result in [("Original", predictions), ("Rebuilt", after), ("Mirrored", flipped)]:
        print(f"{name}: {(result == labels).sum()}/{len(labels)} correct ({(result == labels).mean():.2%})")
    print("Predictions changed by round trip:", np.count_nonzero(predictions != after))
    assert error.max() < 0.02
    assert abs((predictions == labels).mean() - (after == labels).mean()) <= 0.005


if __name__ == "__main__":
    main()
