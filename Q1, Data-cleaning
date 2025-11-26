# -----------------------------
# Q1 — DATA CLEANING
# -----------------------------

import pandas as pd
import numpy as np

# Load dataset
# Replace path with your file name
df = pd.read_csv("file.csv", engine='python')

print("\n--- BEFORE CLEANING ---")
print(df.info())
print(df.isnull().sum())

# 1. Handling Missing Values
# Numeric => fill with median
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical => fill with mode
cat_cols = df.select_dtypes(include=['object']).columns
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

# 2. Handling Outliers using IQR
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower,
                       np.where(df[col] > upper, upper, df[col]))

# 3. Fix Inconsistent Values (example: strip spaces, lowercase)
for c in cat_cols:
    df[c] = df[c].astype(str).str.strip().str.lower()

print("\n--- AFTER CLEANING ---")
print(df.info())
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)
print("\nCleaned dataset saved as cleaned_dataset.csv")
