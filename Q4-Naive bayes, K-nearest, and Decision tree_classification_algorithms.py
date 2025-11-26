# FILE 2: classifiers_iris.py
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Standardization
sc = StandardScaler()
X = sc.fit_transform(X)

# Models
models = {
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=0)
}

# -----------------------------
# HOLDOUT 1 — 80% train, 20% test
# -----------------------------
print("\n\n===== HOLDOUT: 80/20 Split =====")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)

for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print(classification_report(yte, pred))
    print("Accuracy:", accuracy_score(yte, pred))

# -----------------------------
# HOLDOUT 2 — 66.6% train, 33.3% test
# -----------------------------
print("\n\n===== HOLDOUT: 66.6/33.3 Split =====")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.333, random_state=1)

for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print(classification_report(yte, pred))
    print("Accuracy:", accuracy_score(yte, pred))

# -----------------------------
# CROSS VALIDATION
# -----------------------------
scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

# 10-FOLD CV
print("\n\n===== 10-FOLD CROSS VALIDATION =====")
for name, model in models.items():
    results = cross_validate(model, X, y, cv=10, scoring=scoring)
    print(f"\n--- {name} ---")
    for m in scoring:
        print(f"{m}: {results['test_' + m].mean():.4f}")

# 5-FOLD CV
print("\n\n===== 5-FOLD CROSS VALIDATION =====")
for name, model in models.items():
    results = cross_validate(model, X, y, cv=5, scoring=scoring)
    print(f"\n--- {name} ---")
    for m in scoring:
        print(f"{m}: {results['test_' + m].mean():.4f}")
