import time

import cv2
import joblib
import mediapipe as mp
import pandas as pd

from features import landmarks_to_features

LABELS = ["Open", "Close", "Pointer", "Ok", "Nice"]
FONT = cv2.FONT_HERSHEY_SIMPLEX

model = joblib.load("models/best_model.joblib")
classes = [int(c) for c in model.classes_]


def predict_proba(features):
    X = features.reshape(1, -1)
    if hasattr(model, "feature_names_in_"):
        X = pd.DataFrame(X, columns=model.feature_names_in_)
    return model.predict_proba(X)[0]


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
)

cap = cv2.VideoCapture(0)
mirror = False          # press m to toggle
timestamp_ms = 0
last_time = time.time()
fps = 0.0

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)          # selfie view, same as our earlier scripts
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        title = "no hand"
        probs = None
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            for lm in hand:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 0, 255), -1)

            feats = landmarks_to_features(hand, w, h, mirror=mirror)
            if feats is not None:
                probs = predict_proba(feats)

        if probs is not None:
            best = int(probs.argmax())
            title = f"{LABELS[classes[best]]}  {probs[best] * 100:.0f}%"
            for i, c in enumerate(classes):
                y0 = 90 + i * 28
                cv2.rectangle(frame, (10, y0), (10 + int(probs[i] * 200), y0 + 18), (0, 200, 0), -1)
                cv2.putText(frame, f"{LABELS[c]} {probs[i] * 100:.0f}%", (220, y0 + 15),
                            FONT, 0.5, (255, 255, 255), 1)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last_time, 1e-6)
        last_time = now

        cv2.putText(frame, title, (10, 40), FONT, 1.1, (0, 255, 0), 2)
        status = f"mirror: {'ON' if mirror else 'off'}  |  {w}x{h}  |  {fps:.0f} fps  |  m = toggle, q = quit"
        cv2.putText(frame, status, (10, 70), FONT, 0.5, (255, 255, 255), 1)

        cv2.imshow("GestureFlow - live prediction", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            mirror = not mirror

cap.release()
cv2.destroyAllWindows()