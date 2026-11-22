"""Collect labelled webcam takes using the same features as the live classifier."""
import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
import time

import cv2
import joblib
import mediapipe as mp
import pandas as pd

from features import FEATURE_NAMES, landmarks_to_features

LABELS = ["Open", "Close", "Pointer", "Ok", "Nice", "Idle"]
KEYS = {ord(str(i)): name for i, name in enumerate(LABELS, 1)}
HEADER = ["person", "hand", "session", "take", "label", "t", *FEATURE_NAMES]
ROOT = Path(__file__).resolve().parent


def record(args, cap, detector, model, stream):
    writer = csv.writer(stream)
    session = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    counts = dict.fromkeys(LABELS, 0)
    label = None
    recording = False
    take = timestamp = 0
    started = last = time.monotonic()
    fps = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Lost webcam feed. Completed samples are on disk.")
        now = time.monotonic()
        timestamp = max(timestamp + 1, int(now * 1000))
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect_for_video(image, timestamp)
        guess = "no hand"
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            for p in hand:
                cv2.circle(frame, (int(p.x*w), int(p.y*h)), 4, (0, 0, 255), -1)
            feats = landmarks_to_features(hand, w, h, mirror=args.hand == "left")
            if feats is not None:
                row = feats.reshape(1, -1)
                if hasattr(model, "feature_names_in_"):
                    row = pd.DataFrame(row, columns=model.feature_names_in_)
                probs = model.predict_proba(row)[0]
                best = probs.argmax()
                guess = f"{LABELS[int(model.classes_[best])]} {probs[best]:.0%}"
                if recording:
                    writer.writerow([args.person, args.hand, session, take, label,
                                     round(now - started, 3), *feats.tolist()])
                    counts[label] += 1

        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        state = f"RECORDING take {take} ({now-started:.0f}s)" if recording else "paused"
        lines = [f"{args.hand.upper()} | {label or 'choose label'} | {state}",
                 f"Model says: {guess}",
                 "  ".join(f"{k} {v}" for k, v in counts.items()),
                 "1 Open   2 Close   3 Pointer   4 Ok   5 Nice   6 Idle",
                 f"{fps:.0f} fps | SPACE start/stop | q save and quit"]
        for i, line in enumerate(lines):
            color = (0, 0, 255) if i == 0 and recording else (255, 255, 255)
            cv2.putText(frame, line, (10, (28, 54, 80, h-42, h-15)[i]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.imshow("GestureFlow - record samples", frame)
        key = cv2.waitKey(1) & 0xff
        if key == ord("q"):
            return counts
        if key in KEYS:
            label = KEYS[key]
            recording = False
            stream.flush()
        elif key == ord(" ") and label is not None:
            recording = not recording
            if recording:
                take += 1
                started = time.monotonic()
            else:
                stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person", required=True)
    parser.add_argument("--hand", choices=["right", "left"], default="right")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()
    if Path(args.person).name != args.person or not args.person.strip():
        parser.error("Use a participant name without a directory path")
    path = ROOT / "data" / "own" / f"{args.person}_{args.hand}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    model = joblib.load(ROOT / "models/best_model.joblib")
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / "models/hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_hands=1)
    cap = cv2.VideoCapture(args.camera)
    try:
        if not cap.isOpened():
            raise SystemExit("Webcam unavailable.")
        if args.width and args.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        with vision.HandLandmarker.create_from_options(options) as detector, path.open("a+", newline="") as stream:
            stream.seek(0)
            header = next(csv.reader(stream), None)
            if header is None:
                csv.writer(stream).writerow(HEADER)
                stream.flush()
            elif header != HEADER:
                raise ValueError(f"Unexpected CSV header in {path}")
            stream.seek(0, 2)
            counts = record(args, cap, detector, model, stream)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    with path.open(newline="") as stream:
        totals = Counter(row["label"] for row in csv.DictReader(stream))
    print("Saved this session:", counts)
    print(f"Total in {path.relative_to(ROOT)}:", dict(totals))


if __name__ == "__main__":
    main()
