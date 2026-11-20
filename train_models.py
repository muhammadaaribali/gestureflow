"""Run after preprocess.py. Select by grouped training CV before scoring the test set."""
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

data = Path("data/processed")
train = pd.read_csv(data / "train.csv")
test = pd.read_csv(data / "test.csv")
groups = pd.read_csv(data / "train_groups.csv")["group"]
y_train = train.pop("label")
y_test = test.pop("label")
assert groups.str.split("_").str[0].astype(int).equals(y_train)
labels = ["Open", "Close", "Pointer", "Ok", "Nice"]
models = {
    "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), {"max_features": ["sqrt", 0.5]}),
    "SVM": (SVC(), {"C": [1, 10]}),
    "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 7]}),
    "MLP": (MLPClassifier(max_iter=400, random_state=42), {"hidden_layer_sizes": [(32,), (64,)]}),
}
folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(train, y_train, groups))
results = []
fitted = {}
for name, (model, grid) in models.items():
    search = GridSearchCV(model, grid, cv=folds, scoring="f1_macro", error_score="raise")
    start = perf_counter()
    search.fit(train, y_train)
    results.append({"model": name, "cv_macro_f1": search.best_score_, "parameters": str(search.best_params_),
                    "search_seconds": round(perf_counter() - start, 2)})
    fitted[name] = search.best_estimator_
    print(name, search.best_score_, flush=True)

winner = max(results, key=lambda row: row["cv_macro_f1"])["model"]
out = Path("reports")
out.mkdir(exist_ok=True)
for row in results:
    name = row["model"]
    predicted = fitted[name].predict(test)
    row["test_accuracy"] = accuracy_score(y_test, predicted)
    row["test_macro_f1"] = f1_score(y_test, predicted, average="macro")
    slug = name.lower().replace(" ", "_")
    pd.DataFrame(confusion_matrix(y_test, predicted), index=labels, columns=labels).to_csv(out / f"{slug}_confusion.csv")
    report = classification_report(y_test, predicted, target_names=labels, output_dict=True)
    pd.DataFrame(report).T.to_csv(out / f"{slug}_classification.csv")
pd.DataFrame(results).to_csv(out / "model_comparison.csv", index=False)
joblib.dump(fitted[winner], "models/best_model.joblib")
print("Selected by cross-validation:", winner)
