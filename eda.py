import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # draw straight to image files, no window needed
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

LABELS = ["Open", "Close", "Pointer", "Ok", "Nice"]
OUT = "reports/figures"
os.makedirs(OUT, exist_ok=True)

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

y = train["label"].to_numpy()
names = list(train.drop(columns="label").columns)   # x1,y1,...,x20,y20
X = train.drop(columns="label").to_numpy()
assert X.shape[1] == 40

# Rebuild landmark coordinates: shape (rows, 21, 2), wrist (landmark 0) at (0, 0)
pts = np.zeros((len(X), 21, 2))
pts[:, 1:, :] = X.reshape(len(X), 20, 2)

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
         (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

# 1. Class balance
fig, ax = plt.subplots(figsize=(7, 4))
idx = np.arange(5)
ax.bar(idx - 0.2, [np.sum(y == k) for k in range(5)], 0.4, label="train")
ax.bar(idx + 0.2, [np.sum(test["label"] == k) for k in range(5)], 0.4, label="test")
ax.set_xticks(idx)
ax.set_xticklabels(LABELS)
ax.set_ylabel("rows")
ax.set_title("Rows per gesture")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/01_class_balance.png", dpi=150)
plt.close(fig)

# 2. Average hand shape per gesture
fig, axes = plt.subplots(1, 5, figsize=(16, 4))
for k, ax in enumerate(axes):
    m = pts[y == k].mean(axis=0)
    for a, b in BONES:
        ax.plot([m[a, 0], m[b, 0]], [m[a, 1], m[b, 1]], "-", color="tab:blue")
    ax.scatter(m[:, 0], m[:, 1], s=15, color="tab:red")
    ax.set_title(LABELS[k])
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(1.1, -1.1)        # y axis flipped, like an image
    ax.set_aspect("equal")
fig.suptitle("Average hand shape per gesture (wrist at the origin)")
fig.tight_layout()
fig.savefig(f"{OUT}/02_mean_skeletons.png", dpi=150)
plt.close(fig)

# 3. PCA: squash 40 features into 2 to see if gestures separate
rng = np.random.default_rng(42)
sample = rng.choice(len(X), size=min(4000, len(X)), replace=False)
pca = PCA(n_components=2, random_state=42)
Z = pca.fit_transform(X[sample])
explained = pca.explained_variance_ratio_.sum() * 100
fig, ax = plt.subplots(figsize=(7, 6))
for k in range(5):
    sel = y[sample] == k
    ax.scatter(Z[sel, 0], Z[sel, 1], s=6, alpha=0.5, label=LABELS[k])
ax.legend(markerscale=3)
ax.set_title(f"PCA of the 40 features (2 components explain {explained:.0f}% of variance)")
fig.tight_layout()
fig.savefig(f"{OUT}/03_pca.png", dpi=150)
plt.close(fig)

# 4. Distance features per gesture
def dist(a, b):
    return np.linalg.norm(pts[:, a] - pts[:, b], axis=1)

features = {
    "thumb tip - index tip": dist(4, 8),
    "index tip - wrist": dist(8, 0),
    "middle tip - wrist": dist(12, 0),
    "pinky tip - wrist": dist(20, 0),
}
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (name, values) in zip(axes, features.items()):
    ax.boxplot([values[y == k] for k in range(5)], tick_labels=LABELS)
    ax.set_title(name)
fig.tight_layout()
fig.savefig(f"{OUT}/04_distances.png", dpi=150)
plt.close(fig)

print("Median distances per gesture (normalized units):")
table = pd.DataFrame({n: [np.median(v[y == k]) for k in range(5)] for n, v in features.items()},
                     index=LABELS)
print(table.round(3))

# 5. Correlation between the 40 features
corr = np.corrcoef(X, rowvar=False)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
fig.colorbar(im)
ax.set_xticks(range(40))
ax.set_xticklabels(names, rotation=90, fontsize=6)
ax.set_yticks(range(40))
ax.set_yticklabels(names, fontsize=6)
ax.set_title("Feature correlation")
fig.tight_layout()
fig.savefig(f"{OUT}/05_correlation.png", dpi=150)
plt.close(fig)

# 6. Which gestures look most alike? Distance between average hands
centroids = np.array([X[y == k].mean(axis=0) for k in range(5)])
D = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
print("\nDistance between average hands (bigger = easier to tell apart):")
print(pd.DataFrame(D, index=LABELS, columns=LABELS).round(2))
pairs = [(D[i, j], LABELS[i], LABELS[j]) for i in range(5) for j in range(i + 1, 5)]
print("Most similar pair:", min(pairs))
print("PCA explained variance:", f"{explained:.1f}%")
print(f"\nFigures saved in {OUT}/")