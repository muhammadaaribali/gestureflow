import pandas as pd

LABELS = ["Open", "Close", "Pointer", "Ok", "Nice"]

df = pd.read_csv("data/public/data/static_gestures/static_gestures_v3.csv", header=None)

# column 0 is the label index, columns 1-42 are x0,y0,x1,y1,...,x20,y20
cols = ["label"]
for i in range(21):
    cols += [f"x{i}", f"y{i}"]
df.columns = cols

features = df.drop(columns="label")

print("shape:", df.shape)
print("missing values:", df.isna().sum().sum())
print("exact duplicate rows:", df.duplicated().sum())
print("value range:", features.min().min(), "to", features.max().max())

print("\nrows per gesture:")
counts = df["label"].value_counts().sort_index()
for idx, n in counts.items():
    print(f"  {idx} {LABELS[idx]}: {n}")

# Is the file grouped by gesture, or shuffled?
changes = (df["label"] != df["label"].shift()).sum() - 1
print("\ntimes the label changes reading top to bottom:", changes)