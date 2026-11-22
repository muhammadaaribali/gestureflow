"""Replay gesture scores with estimated no-hand frames between observations."""
import argparse
from statistics import median

import numpy as np
import pandas as pd

from smoothing import GestureStabilizer

LABELS = ["Open", "Close", "Pointer", "Ok", "Nice"]
PROBS = ["p_" + label for label in LABELS]
THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)


def reconstruct(group, seconds, fallback_dt):
    times = group["t"].to_numpy()
    dt = float(np.median(np.diff(times))) if len(times) > 1 else fallback_dt
    frames = []
    cursor = dt
    for t, probs in zip(times, group[PROBS].to_numpy()):
        for i in range(round((t - cursor) / dt)):
            frames.append((cursor + i * dt, None, 0.0))
        best = probs.argmax()
        frames.append((t, LABELS[best], probs[best]))
        cursor = t + dt
    while cursor <= seconds:
        frames.append((cursor, None, 0.0))
        cursor += dt
    return frames


def replay(segments, threshold, hold):
    events = {}
    for name, frames in segments.items():
        stabilizer = GestureStabilizer(min_conf=threshold, hold_seconds=hold)
        events[name] = []
        for t, label, conf in frames:
            active, fired = stabilizer.update(label, conf, t)
            if fired:
                events[name].append((t, active))
    return events


def score(events):
    latencies = []
    ok = wrong = doubles = 0
    for name, entries in events.items():
        if name == "IDLE":
            continue
        good = [t for t, label in entries if label == name.capitalize()]
        bad = len(entries) - len(good)
        wrong += bad
        doubles += len(good) > 1
        if good:
            latencies.append(good[0])
            ok += bad == 0
    return ok, wrong, doubles, len(events.get("IDLE", [])), round(median(latencies), 2) if latencies else None


def table(title, rows):
    print(title)
    print(pd.DataFrame(rows).to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--seconds", type=float, default=15)
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    groups = dict(iter(df.groupby("segment", sort=False)))
    intervals = df.groupby("segment")["t"].diff().dropna()
    if intervals.empty or (intervals <= 0).any() or args.seconds <= 0:
        parser.error("Expected increasing frame timestamps and a positive segment duration")
    segments = {n: reconstruct(g, args.seconds, intervals.median()) for n, g in groups.items()}
    rows = []
    for name, frames in segments.items():
        lost = longest = run = 0
        for _, label, _ in frames:
            run = run + 1 if label is None else 0
            longest = max(run, longest)
            lost += label is None
        conf = groups[name][PROBS].max(axis=1)
        rows.append(dict(segment=name, frames=len(frames), lost_pct=round(100*lost/len(frames), 1),
                         longest_gap=longest, mean=round(conf.mean(), 2),
                         p10=round(conf.quantile(.1), 2), median=round(conf.median(), 2),
                         p90=round(conf.quantile(.9), 2)))
    table("1. Confidence and estimated hand loss", rows)
    if "IDLE" in groups:
        print("IDLE predictions:", groups["IDLE"]["predicted"].value_counts().to_dict())
    rows = []
    for threshold in THRESHOLDS:
        row = {"threshold": threshold}
        for name, g in groups.items():
            keep = g[PROBS].max(axis=1) >= threshold
            if name != "IDLE":
                keep &= g["predicted"] == name.capitalize()
            row[name] = round(100 * keep.mean())
        rows.append(row)
    table("2. Percent of scored frames retained", rows)
    rows = []
    settings = [(t, .4) for t in THRESHOLDS] + [(.7, h) for h in (.6, .8, 1., 1.5)]
    for threshold, hold in settings:
        ok, wrong, doubles, idle, latency = score(replay(segments, threshold, hold))
        rows.append(dict(threshold=threshold, hold=hold, gestures_ok=ok, wrong=wrong,
                         double_segments=doubles, idle_false=idle, latency_s=latency))
    table("3. Stabilizer replay (7 frames, 5 votes)", rows)
    print("4. Trigger timeline (0.70 confidence, 0.4 s hold)")
    for name, events in replay(segments, .7, .4).items():
        print(name, [(round(float(t), 2), label) for t, label in events])


if __name__ == "__main__":
    main()
