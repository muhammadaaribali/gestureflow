import os
import numpy as np
import pandas as pd

CSV = "data/public/data/static_gestures/static_gestures_v3.csv"
LABELS = ["Open", "Close", "Pointer", "Ok", "Nice"]
CHUNK = 100          # consecutive rows that stay together
TEST_FRACTION = 0.2
SEED = 42 #

df = pd.read_csv(CSV, header=None)
cols = ["label"]
for i in range(21):
    cols += [f"x{i}", f"y{i}"]
df.columns = cols

# 1. Missing values and duplicates
print("missing values:", int(df.isna().sum().sum()))
print("exact duplicates:", int(df.duplicated().sum()))
df = df.dropna().drop_duplicates().reset_index(drop=True)

# 2. Label encoding: labels are already integers 0-4
df["label"] = df["label"].astype(int)
print("label codes:", dict(enumerate(LABELS)))

# 3. Constant columns carry no information, so drop them
feature_cols = [c for c in df.columns if c != "label"]
constant = [c for c in feature_cols if df[c].nunique() == 1]
print("constant columns dropped:", constant)
df = df.drop(columns=constant)
feature_cols = [c for c in df.columns if c != "label"]

# 4. Scaling check: values should already be inside -1..1
vals = df[feature_cols]
print("value range:", vals.min().min(), "to", vals.max().max())

# 5. Why we can't split randomly: how similar are neighbouring rows?
X = df[feature_cols].to_numpy()
y = df["label"].to_numpy()
rng = np.random.default_rng(SEED)

same = y[1:] == y[:-1]
neighbour_diff = np.abs(X[1:] - X[:-1]).mean(axis=1)[same].mean()

i = rng.integers(0, len(X), 20000)
j = rng.integers(0, len(X), 20000)
pair_same = y[i] == y[j]
random_diff = np.abs(X[i] - X[j]).mean(axis=1)[pair_same].mean()

print(f"\navg difference, neighbouring rows:    {neighbour_diff:.4f}")
print(f"avg difference, random same-gesture:  {random_diff:.4f}")

# 6. Chunked train/test split, done separately inside each gesture
train_parts, test_parts = [], []
for label in sorted(df["label"].unique()):
    block = df[df["label"] == label]              # still in time order
    chunk_ids = np.arange(len(block)) // CHUNK
    chunks = np.unique(chunk_ids)
    n_test = max(1, round(len(chunks) * TEST_FRACTION))
    test_chunks = rng.choice(chunks, size=n_test, replace=False)
    is_test = np.isin(chunk_ids, test_chunks)
    train_parts.append(block[~is_test])
    test_parts.append(block[is_test])

train = pd.concat(train_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
test = pd.concat(test_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"\ntrain rows: {len(train)}   test rows: {len(test)}")
print("train per gesture:", train["label"].value_counts().sort_index().to_dict())
print("test per gesture: ", test["label"].value_counts().sort_index().to_dict())

os.makedirs("data/processed", exist_ok=True)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
print("\nSaved data/processed/train.csv and test.csv")