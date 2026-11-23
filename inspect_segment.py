"""Inspect model v2 predictions and stabilizer transitions for one recorded segment."""
import argparse
from itertools import groupby

import joblib
import numpy as np
import pandas as pd

from features import FEATURE_NAMES
from smoothing import GestureStabilizer

LABELS = ["Open", "Close", "Pointer", "Ok", "Nice", "Idle"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("csv")
parser.add_argument("segment")
parser.add_argument("--seconds", type=float, default=15)
args = parser.parse_args()

df = pd.read_csv(args.csv)
seg = df[df["segment"] == args.segment]
if seg.empty:
    parser.error(f"No segment {args.segment!r}. Available: {sorted(df['segment'].unique())}")
t = seg["t"].to_numpy()
if len(t) < 2 or not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
    parser.error("Need at least two increasing, finite timestamps to estimate frame spacing.")
dt = float(np.median(np.diff(t)))
model = joblib.load("models/model_v2.joblib")
X = seg[FEATURE_NAMES]
probs = model.predict_proba(X if hasattr(model, "feature_names_in_") else X.to_numpy())

# The CSV contains detected hands only. Match replay_v2's median-spacing estimate.
frames, cursor = [], dt
for ti, p in zip(t, probs):
    for k in range(round((ti - cursor) / dt)):
        frames.append((cursor + k * dt, None, 0.0))
    best = p.argmax()
    frames.append((ti, LABELS[int(model.classes_[best])], float(p[best])))
    cursor = ti + dt
for k in range(round((args.seconds - cursor) / dt)):
    frames.append((cursor + k * dt, None, 0.0))

lost = sum(label is None for _, label, _ in frames)
print(f"{args.segment}: {len(frames)} frames, {100 * lost / len(frames):.0f}% inferred missing")
print("\nPrediction runs (3+ frames):")
hidden = 0
for label, group in groupby(frames, key=lambda frame: frame[1]):
    run = list(group)
    if len(run) < 3:
        hidden += 1
        continue
    confidence = np.mean([frame[2] for frame in run])
    print(f"  {run[0][0]:5.2f}-{run[-1][0]:5.2f}s  {label or 'no hand':<8}"
          f"{len(run):5} frames   mean conf {confidence:.2f}")
print(f"  ({hidden} shorter runs hidden)")

stab = GestureStabilizer()
prev = None
print(f"\nStabilizer timeline (release {stab.release_seconds} s, min_conf {stab.min_conf}):")
for ti, label, conf in frames:
    active, triggered = stab.update(label, conf, ti)
    if active != prev or triggered:
        print(f"  {ti:5.2f}s  active: {prev} -> {active}{'   <- TRIGGER' if triggered else ''}")
    prev = active
