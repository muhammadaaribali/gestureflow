"""Replay recorded frames through model v2 + the stabilizer: false and duplicate triggers."""
import glob
import json

import joblib
import numpy as np
import pandas as pd

from features import FEATURE_NAMES
from smoothing import GestureStabilizer

LABELS = ["Open", "Close", "Pointer", "Ok", "Nice", "Idle"]
SEGMENT_TO_LABEL = {"OPEN": "Open", "CLOSE": "Close", "POINTER": "Pointer",
                    "OK": "Ok", "NICE": "Nice", "IDLE": "Idle"}
LIVE_SECONDS = 15

model = joblib.load("models/model_v2.joblib")
classes = [int(c) for c in model.classes_]


def probabilities(df):
    X = df[FEATURE_NAMES]
    if not hasattr(model, "feature_names_in_"):
        X = X.to_numpy()
    return model.predict_proba(X)


def reconstruct(seg_df, seconds=None):
    """One entry per camera frame: (time, label, confidence). label None = no hand found."""
    t = seg_df["t"].to_numpy()
    probs = probabilities(seg_df)
    dt = float(np.median(np.diff(t)))
    end = seconds if seconds is not None else t[-1] + dt
    frames, cursor = [], dt
    for ti, p in zip(t, probs):
        for k in range(int(round((ti - cursor) / dt))):
            frames.append((cursor + k * dt, None, 0.0))
        k = int(p.argmax())
        frames.append((ti, LABELS[classes[k]], float(p[k])))
        cursor = ti + dt
    for k in range(max(int(round((end - cursor) / dt)), 0)):
        frames.append((cursor + k * dt, None, 0.0))
    return frames


# name -> (frames, target gesture or None for Idle)
sets = {}
for path in sorted(glob.glob("reports/live_frames_*.csv")):
    tag = path.split("live_frames_")[1][:-4]
    d = pd.read_csv(path)
    for seg, g in d.groupby("segment", sort=False):
        target = SEGMENT_TO_LABEL[seg]
        sets[f"C {tag} {seg}"] = (reconstruct(g.reset_index(drop=True), LIVE_SECONDS),
                                  None if target == "Idle" else target)

own = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("data/own/*.csv"))], ignore_index=True)
own["take_id"] = (own["person"] + "_" + own["hand"] + "_" + own["session"].astype(str)
                  + "_" + own["take"].astype(str))
for take_id in json.load(open("models/v2_heldout_takes.json")):
    g = own[own["take_id"] == take_id].reset_index(drop=True)
    sets[f"B take {take_id}"] = (reconstruct(g), None)

n_gest = sum(1 for _, target in sets.values() if target)


def run(frames, release, min_conf):
    stab = GestureStabilizer(min_conf=min_conf, release_seconds=release)
    fired = []
    for t, label, conf in frames:
        active, triggered = stab.update(label, conf, t)
        if triggered:
            fired.append(active)
    return fired


def summarize(release, min_conf):
    ok = wrong = dup = idle_c = idle_b = 0
    for name, (frames, target) in sets.items():
        fired = run(frames, release, min_conf)
        if target is None:
            if name.startswith("B"):
                idle_b += len(fired)
            else:
                idle_c += len(fired)
        else:
            good = fired.count(target)
            wrong += len(fired) - good
            ok += int(good >= 1 and len(fired) == good)
            dup += int(good > 1)
    return {"release s": release, "min_conf": min_conf, f"gestures OK (of {n_gest})": ok,
            "wrong triggers": wrong, "duplicate segments": dup,
            "idle false (live sessions)": idle_c, "idle false (held-out takes)": idle_b}


print("1) Release time (hold-through for dropouts), min_conf 0.7")
print(pd.DataFrame([summarize(r, 0.7) for r in (0.0, 0.2, 0.4, 0.6)]).to_string(index=False))

print("\n2) Confidence threshold, release 0.4 s")
print(pd.DataFrame([summarize(0.4, c) for c in (0.6, 0.7, 0.8, 0.9)]).to_string(index=False))

print("\n3) Segments that are not clean with the defaults (release 0.4 s, min_conf 0.7)")
dirty = 0
for name, (frames, target) in sets.items():
    fired = run(frames, 0.4, 0.7)
    clean = (not fired) if target is None else (fired == [target])
    if not clean:
        dirty += 1
        print(f"  {name}: triggers = {fired}")
print(f"  ({len(sets) - dirty} of {len(sets)} segments are clean)")
