import argparse
import json
import time

import cv2
import joblib
import mediapipe as mp
import pandas as pd

from actions import ActionController, dry_run_backend
from backends import available_players, make_system_backend
from features import landmarks_to_features
from smoothing import GestureStabilizer

LABELS = ["Open", "Close", "Pointer", "Ok", "Nice", "Idle"]
FONT = cv2.FONT_HERSHEY_SIMPLEX

parser = argparse.ArgumentParser()
parser.add_argument("--player", help="Target a specific player, e.g. spotify")
args = parser.parse_args()

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
stabilizer = GestureStabilizer()
with open("config.json") as config_file:
    cfg = json.load(config_file)
controller = ActionController(cfg["mapping"], dry_run_backend,
                              cooldown=cfg["cooldown_seconds"],
                              repeat_interval=cfg["repeat_interval_seconds"])
player = args.player or cfg.get("preferred_player")
volume_step = cfg.get("volume_step_percent", 3)
system_backend, tools = make_system_backend(volume_step, player=player)
print("Media player:", player or "auto", flush=True)
print("System tools:", tools, flush=True)
live = False
mirror = False
timestamp_ms = 0
last_time = time.monotonic()
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

        raw_label, confidence = None, 0.0
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
            raw_label, confidence = LABELS[classes[best]], float(probs[best])
            title = f"{raw_label}  {confidence:.0%}"
            for i, c in enumerate(classes):
                y0 = 90 + i * 28
                cv2.rectangle(frame, (10, y0), (10 + int(probs[i] * 200), y0 + 18), (0, 200, 0), -1)
                cv2.putText(frame, f"{LABELS[c]} {probs[i] * 100:.0f}%", (220, y0 + 15),
                            FONT, 0.5, (255, 255, 255), 1)

        now = time.monotonic()
        active, triggered = stabilizer.update(raw_label, confidence, now)
        if triggered:
            print(f"TRIGGER: {active}", flush=True)
        controller.update(active, triggered, now)
        mode = "LIVE" if live else "dry run"
        state = f"ON ({mode})" if controller.enabled else "OFF"
        cv2.putText(frame, f"CONTROL: {state} | player: {player or 'auto'}", (10, h - 40), FONT, 0.6, (0, 165, 255), 2)
        cv2.putText(frame, f"STABLE: {active or '-'}", (w - 260, 40),
                    FONT, 0.9, (0, 255, 255), 2)
        fps = 0.9 * fps + 0.1 / max(now - last_time, 1e-6)
        last_time = now

        cv2.putText(frame, title, (10, 40), FONT, 1.1, (0, 255, 0), 2)
        setting = "Left hand | mirror ON" if mirror else "Right hand | mirror OFF"
        status = f"Selected: {setting} | {w}x{h} | {fps:.0f} fps"
        cv2.putText(frame, status, (10, 70), FONT, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "m: hand   c: control   d: live/dry   p: player   q: quit",
                    (10, h - 12), FONT, 0.5, (255, 255, 255), 1)

        cv2.imshow("GestureFlow - live prediction", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            controller.enabled = not controller.enabled
            stabilizer = GestureStabilizer()
        elif key == ord("d"):
            live = not live
            controller.backend = system_backend if live else dry_run_backend
            stabilizer = GestureStabilizer()
            print("LIVE system control" if live else "Dry run", flush=True)
        elif key == ord("p"):
            choices = [None, *available_players()]
            index = choices.index(player) if player in choices else -1
            player = choices[(index + 1) % len(choices)]
            system_backend, tools = make_system_backend(volume_step, player=player)
            controller.backend = system_backend if live else dry_run_backend
            stabilizer = GestureStabilizer()
            print("Media player:", player or "auto", flush=True)
        elif key == ord("m"):
            mirror = not mirror
            stabilizer = GestureStabilizer()

cap.release()
cv2.destroyAllWindows()
