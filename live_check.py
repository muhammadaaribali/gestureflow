"""Real-world accuracy check: perform each gesture in front of the webcam and measure the model."""
import argparse
import json
import os
import time
from collections import Counter

import cv2
import joblib
import mediapipe as mp
import pandas as pd

from features import FEATURE_NAMES, landmarks_to_features
from smoothing import GestureStabilizer

LABELS = ["Open", "Close", "Pointer", "Ok", "Nice", "Idle"]
GESTURES = LABELS[:5]
FONT = cv2.FONT_HERSHEY_SIMPLEX
HINTS = {
    "Open": "open palm, fingers up",
    "Close": "closed fist",
    "Pointer": "index finger pointing, angled up and to the side",
    "Ok": "thumb and index tips touching, other three fingers up",
    "Nice": "thumbs-up, forearm roughly horizontal",
    "FIDGET": "move casually: show 2, 3 or 4 fingers, wiggle, change poses (avoid the 5 gestures)",
    "IDLE": "relax your hand in view and make NO gesture",
}

parser = argparse.ArgumentParser()
parser.add_argument("--person", default="anon", help="who is performing, e.g. aarib")
parser.add_argument("--hand", choices=["right", "left"], default="right",
                    help="right = as trained; left = mirrored features")
parser.add_argument("--seconds", type=float, default=15, help="recording time per gesture")
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--width", type=int, default=None, help="optional camera width")
parser.add_argument("--height", type=int, default=None, help="optional camera height")
args = parser.parse_args()

model = joblib.load("models/best_model.joblib")
classes = [int(c) for c in model.classes_]


def predict_proba(features):
    X = features.reshape(1, -1)
    if hasattr(model, "feature_names_in_"):
        X = pd.DataFrame(X, columns=model.feature_names_in_)
    return model.predict_proba(X)[0]


class Segment:
    def __init__(self, title, target, hint):
        self.title, self.target, self.hint = title, target, hint
        self.frames_total = 0
        self.frames_hand = 0
        self.counts = Counter()        # predicted gesture -> number of frames
        self.conf_sum = 0.0
        self.triggers = Counter()      # gestures the stabilizer confirmed


segments = [Segment(name.upper(), name, HINTS[name]) for name in GESTURES]
segments.append(Segment("IDLE", None, HINTS["IDLE"]))
segments.append(Segment("FIDGET", None, HINTS["FIDGET"]))


def summarize(seg):
    hand = seg.frames_hand
    out = {
        "gesture": seg.title,
        "target": seg.target,
        "frames_total": seg.frames_total,
        "frames_with_hand": hand,
        "hand_detected_pct": 100 * hand / max(seg.frames_total, 1),
        "predicted_counts": dict(seg.counts),
        "mean_confidence": seg.conf_sum / max(hand, 1),
        "triggers": dict(seg.triggers),
    }
    if seg.target is not None:
        out["frame_accuracy_pct"] = 100 * seg.counts[seg.target] / max(hand, 1)
        out["correct_triggers"] = seg.triggers[seg.target]
        out["wrong_triggers"] = sum(seg.triggers.values()) - seg.triggers[seg.target]
    else:
        out["idle_frames_pct"] = 100 * seg.counts["Idle"] / max(hand, 1)
        out["false_triggers"] = sum(seg.triggers.values())
    return out


def put(frame, text, y, scale=0.7, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, (15, y), FONT, scale, color, thickness)


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
)

cap = cv2.VideoCapture(args.camera)
if args.width and args.height:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

timestamp_ms = 0
phase, phase_start = "wait", time.monotonic()      # wait -> countdown -> record
seg_idx = 0
stab = GestureStabilizer()
size = None
frame_rows = []

with HandLandmarker.create_from_options(options) as landmarker:
    while seg_idx < len(segments):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)               # selfie view, same as the app
        h, w, _ = frame.shape
        size = (w, h)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        label, conf, feats, probs = None, 0.0, None, None
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            for lm in hand:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 0, 255), -1)
            feats = landmarks_to_features(hand, w, h, mirror=(args.hand == "left"))
            if feats is not None:
                probs = predict_proba(feats)
                k = int(probs.argmax())
                label, conf = LABELS[classes[k]], float(probs[k])

        now = time.monotonic()
        seg = segments[seg_idx]

        if phase == "countdown" and now - phase_start >= 3:
            phase, phase_start = "record", now
            stab = GestureStabilizer()           # fresh votes for each gesture
        elif phase == "record":
            seg.frames_total += 1
            if feats is not None and probs is not None:
                frame_rows.append([args.person, args.hand, seg.title, round(now - phase_start, 3),
                                   *feats.tolist(), *probs.tolist(), label])
            if label is not None:
                seg.frames_hand += 1
                seg.counts[label] += 1
                seg.conf_sum += conf
            active, triggered = stab.update(label, conf, now)
            if triggered:
                seg.triggers[active] += 1
            if now - phase_start >= args.seconds:
                seg_idx += 1
                phase, phase_start = "wait", now
                continue

        # ---- on-screen instructions (the prediction is deliberately NOT shown) ----
        if phase == "wait":
            put(frame, f"Next: {seg.title}  ({seg_idx + 1}/{len(segments)})", 35, 0.9, (0, 255, 0))
            put(frame, seg.hint, 70, 0.6)
            put(frame, "Press SPACE when ready  |  q = quit and save", h - 20, 0.55)
        elif phase == "countdown":
            put(frame, f"Get ready: {seg.title}", 35, 0.9, (0, 255, 255))
            put(frame, seg.hint, 70, 0.6)
            n = max(1, 3 - int(now - phase_start))
            cv2.putText(frame, str(n), (w // 2 - 40, h // 2 + 40), FONT, 4, (0, 255, 255), 6)
        else:
            put(frame, f"RECORDING: {seg.title}", 35, 0.9, (0, 0, 255))
            put(frame, seg.hint if seg.target is None else "hold it, slowly vary angle and distance", 70, 0.6)
            frac = min((now - phase_start) / args.seconds, 1.0)
            cv2.rectangle(frame, (15, h - 32), (w - 15, h - 12), (255, 255, 255), 1)
            cv2.rectangle(frame, (15, h - 32), (15 + int((w - 30) * frac), h - 12), (0, 0, 255), -1)

        cv2.imshow("GestureFlow - real-world check", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and phase == "wait":
            phase, phase_start = "countdown", now

cap.release()
cv2.destroyAllWindows()

done = segments[:seg_idx]                        # only fully recorded segments
if not done:
    raise SystemExit("No segment was completed, nothing saved.")

rows = [summarize(s) for s in done]
gest_rows = [r for r in rows if r["target"] is not None]
idle_rows = [r for r in rows if r["target"] is None]
total_frames = sum(s.frames_total for s in done)
fps = total_frames / (len(done) * args.seconds)
total_hand = sum(r["frames_with_hand"] for r in gest_rows)
total_correct = sum(r["predicted_counts"].get(r["target"], 0) for r in gest_rows)
overall = 100 * total_correct / max(total_hand, 1)

print(f"\nPerson: {args.person}   hand: {args.hand}   camera: {size[0]}x{size[1]}   ~{fps:.0f} fps")
print(f"{'gesture':<10}{'hand seen':>10}{'frame acc':>11}{'mean conf':>11}{'correct trig':>14}{'wrong trig':>12}")
for r in gest_rows:
    print(f"{r['target']:<10}{r['hand_detected_pct']:>9.0f}%{r['frame_accuracy_pct']:>10.1f}%"
          f"{r['mean_confidence'] * 100:>10.0f}%{r['correct_triggers']:>14}{r['wrong_triggers']:>12}")
print(f"\nOverall frame accuracy (frames with a hand): {overall:.1f}%")
for r in idle_rows:
    print(f"{r['gesture']}: {r['false_triggers']} false trigger(s) in {args.seconds:.0f} s; "
          f"frames predicted as: {r['predicted_counts']}")

matrix = pd.DataFrame(
    [[r["predicted_counts"].get(c, 0) for c in LABELS] for r in gest_rows],
    index=[r["target"] for r in gest_rows], columns=LABELS)
pct = (matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0) * 100).round(1)
print("\nConfusion matrix (% of frames; rows = what you performed, columns = what the model predicted):")
print(pct)

os.makedirs("reports", exist_ok=True)
path = f"reports/live_check_{args.person}_{args.hand}.json"
with open(path, "w") as f:
    json.dump({
        "person": args.person, "hand": args.hand, "seconds_per_gesture": args.seconds,
        "camera": {"width": size[0], "height": size[1], "approx_fps": round(fps, 1)},
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_frame_accuracy_pct": round(overall, 2),
        "segments": rows,
        "confusion_pct": pct.to_dict(),
    }, f, indent=2)
print(f"\nSaved {path}")

done_titles = {s.title for s in done}
frame_rows = [row for row in frame_rows if row[2] in done_titles]
cols = (["person", "hand", "segment", "t"] + FEATURE_NAMES
        + [f"p_{LABELS[c]}" for c in classes] + ["predicted"])
frames_path = f"reports/live_frames_{args.person}_{args.hand}.csv"
pd.DataFrame(frame_rows, columns=cols).to_csv(frames_path, index=False)
print(f"Saved {frames_path} ({len(frame_rows)} frames)")
