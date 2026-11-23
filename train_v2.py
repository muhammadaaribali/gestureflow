"""Train a six-class GestureFlow model and evaluate it on separate recordings."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix

from features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parent
LABELS = ["Open", "Close", "Pointer", "Ok", "Nice", "Idle"]
IDS = {name: i for i, name in enumerate(LABELS)}
IDLE = IDS["Idle"]


def split_idle(own):
    rng = np.random.default_rng(42)
    held = []
    idle = own.loc[own["y"] == IDLE]
    for _, group in idle.groupby(["person", "hand"]):
        takes = sorted(group["take_id"].unique())
        if len(takes) >= 2:
            held.extend(rng.choice(takes, size=max(1, round(len(takes) * 0.25)), replace=False))
    mask = own["take_id"].isin(held)
    return own.loc[~mask], own.loc[mask], held


def predict(model, frame):
    features = frame[FEATURE_NAMES]
    if not hasattr(model, "feature_names_in_"):
        features = features.to_numpy()
    return model.predict(features).astype(int)


def percent(values):
    return f"{100 * values.mean():.1f}%" if len(values) else "-"


def evaluate(sets, v1, v2):
    rows, matrices = [], []
    for title, frame in sets.items():
        truth = frame["y"].to_numpy()
        old, new = predict(v1, frame), predict(v2, frame)
        gesture, idle = truth < IDLE, truth == IDLE
        rows.append([title, len(frame), percent(old[gesture] == truth[gesture]),
                     percent(new[gesture] == truth[gesture]), percent(new[gesture] == IDLE),
                     percent(new[idle] == IDLE)])
        cm = confusion_matrix(truth, new, labels=range(len(LABELS)), normalize="true")
        matrix = pd.DataFrame(100 * cm, index=LABELS, columns=LABELS)
        matrices.append((title, matrix.iloc[np.unique(truth)].round(1)))
    print(pd.DataFrame(rows, columns=["set", "frames", "v1 gesture", "v2 gesture",
                                      "gesture->Idle", "Idle recall"]).to_string(index=False))
    for title, matrix in matrices:
        print(f"\n{title}: true rows, predicted columns (%)")
        print(matrix.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-person", help="exclude this person's own recordings from training")
    args = parser.parse_args()
    data = ROOT / "data"
    zen = pd.read_csv(data / "processed/train.csv")
    own = pd.concat([pd.read_csv(p) for p in sorted((data / "own").glob("*.csv"))], ignore_index=True)
    own["y"] = own["label"].map(IDS.__getitem__)
    own["take_id"] = own[["person", "hand", "session", "take"]].astype(str).agg("_".join, axis=1)
    sets = {"A Zenodo test": pd.read_csv(data / "processed/test.csv").rename(columns={"label": "y"})}
    person_test = own[own["person"] == args.holdout_person]
    if args.holdout_person and person_test.empty:
        parser.error("No recordings found for that person")
    train, idle_test, takes = split_idle(own.drop(person_test.index))
    if IDLE not in train["y"].values:
        parser.error("Training requires Idle recordings from a remaining participant")
    if len(idle_test):
        sets["B held-out Idle"] = idle_test
    for path in sorted((ROOT / "reports").glob("live_frames_*.csv")):
        live = pd.read_csv(path)
        live["y"] = live["segment"].replace({"FIDGET": "IDLE"}).str.capitalize().map(IDS.__getitem__)
        sets["C " + path.stem.removeprefix("live_frames_")] = live
    if args.holdout_person:
        sets["D " + args.holdout_person] = person_test

    mix = pd.DataFrame({"Zenodo": zen.label.map(dict(enumerate(LABELS))).value_counts(),
                        "own": train.label.value_counts()}).reindex(LABELS).fillna(0).astype(int)
    mix["total"] = mix.sum(axis=1)
    print(mix.to_string())
    print(f"Held-out Idle: {len(takes)} takes, {len(idle_test)} frames")
    print("Held-out take IDs:", ", ".join(takes))
    v1 = joblib.load(ROOT / "models/model_v1.joblib")
    v2 = clone(v1)
    X = pd.concat([zen[FEATURE_NAMES], train[FEATURE_NAMES]], ignore_index=True)
    y = np.concatenate([zen["label"], train["y"]])
    v2.fit(X, y)
    evaluate(sets, v1, v2)
    if not args.holdout_person:
        destination = ROOT / "models/model_v2.joblib"
        with NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            joblib.dump(v2, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        (destination.parent / "v2_heldout_takes.json").write_text(json.dumps(takes, indent=2) + "\n")
        print("Saved", destination)


if __name__ == "__main__":
    main()
