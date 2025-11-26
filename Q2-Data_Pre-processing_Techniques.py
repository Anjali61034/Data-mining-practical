# FILE 1: preprocessing.py

import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
import numpy as np

# -----------------------------
# LOAD DATASETS
# -----------------------------
iris = load_iris()
wine = load_wine()

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target

print("\n=== ORIGINAL IRIS (first 5 rows) ===")
print(iris_df.head(), "\n")

print("=== ORIGINAL WINE (first 5 rows) ===")
print(wine_df.head(), "\n")

# -----------------------------
# STANDARDIZATION
# -----------------------------
std_scaler = StandardScaler()
iris_std = std_scaler.fit_transform(iris_df.iloc[:, :-1])
wine_std = std_scaler.fit_transform(wine_df.iloc[:, :-1])

print("=== IRIS Standardized (first 5 rows) ===")
print(pd.DataFrame(iris_std, columns=iris.feature_names).head(), "\n")

print("=== WINE Standardized (first 5 rows) ===")
print(pd.DataFrame(wine_std, columns=wine.feature_names).head(), "\n")

# -----------------------------
# NORMALIZATION (Min-Max)
# -----------------------------
norm_scaler = MinMaxScaler()
iris_norm = norm_scaler.fit_transform(iris_df.iloc[:, :-1])
wine_norm = norm_scaler.fit_transform(wine_df.iloc[:, :-1])

print("=== IRIS Normalized (first 5 rows) ===")
print(pd.DataFrame(iris_norm, columns=iris.feature_names).head(), "\n")

print("=== WINE Normalized (first 5 rows) ===")
print(pd.DataFrame(wine_norm, columns=wine.feature_names).head(), "\n")

# -----------------------------
# TRANSFORMATION (Log Transform)
# -----------------------------
iris_log = np.log1p(iris_df.iloc[:, :-1])
wine_log = np.log1p(wine_df.iloc[:, :-1])

print("=== IRIS Log Transformed (first 5 rows) ===")
print(iris_log.head(), "\n")

print("=== WINE Log Transformed (first 5 rows) ===")
print(wine_log.head(), "\n")

# -----------------------------
# AGGREGATION (Mean of features)
# -----------------------------
iris_df["agg_mean"] = iris_df.iloc[:, :-1].mean(axis=1)
wine_df["agg_mean"] = wine_df.iloc[:, :-1].mean(axis=1)

print("=== IRIS Aggregation Column Added (first 5 rows) ===")
print(iris_df.head(), "\n")

print("=== WINE Aggregation Column Added (first 5 rows) ===")
print(wine_df.head(), "\n")

# -----------------------------
# DISCRETIZATION / BINARIZATION
# -----------------------------
binz = Binarizer(threshold=iris_df["sepal length (cm)"].mean())
iris_bin = binz.fit_transform(iris_df[["sepal length (cm)"]])

binz2 = Binarizer(threshold=wine_df["alcohol"].mean())
wine_bin = binz2.fit_transform(wine_df[["alcohol"]])

print("=== IRIS Binarized sepal length (first 5 rows) ===")
print(pd.DataFrame(iris_bin, columns=["sepal_length_bin"]).head(), "\n")

print("=== WINE Binarized alcohol (first 5 rows) ===")
print(pd.DataFrame(wine_bin, columns=["alcohol_bin"]).head(), "\n")

# -----------------------------
# SAMPLING (Random Under-sampling)
# -----------------------------
iris_sample = iris_df.sample(frac=0.7, random_state=1)
wine_sample = wine_df.sample(frac=0.7, random_state=1)

print("=== IRIS Sampled 70% (first 5 rows) ===")
print(iris_sample.head(), "\n")

print("=== WINE Sampled 70% (first 5 rows) ===")
print(wine_sample.head(), "\n")

# Save preprocessed files
iris_df.to_csv("iris_preprocessed.csv", index=False)
wine_df.to_csv("wine_preprocessed.csv", index=False)

print("Preprocessing completed. Files saved:")
print("iris_preprocessed.csv")
print("wine_preprocessed.csv")
