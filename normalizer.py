import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
df = pd.read_csv('data.csv')

print("="*80)
print("NORMALIZER - PROGRAM 2")
print("="*80)

print("\nOriginal Data:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
numeric_cols = df.select_dtypes(include=[np.number]).columns
X = df[numeric_cols].copy()

print(f"\nNumeric columns selected: {list(numeric_cols)}")

print("\nOriginal Data Statistics:")
print(X.describe())
print("\n" + "="*80)
print("APPLYING NORMALIZER")
print("="*80)
print("\nFormula: x_normalized = x / ||x||")
print("Scales each sample to have unit norm (magnitude = 1)")
print("Default norm: L2 (Euclidean norm)")
print("\nUse case: ")
print("- Text classification")
print("- TF-IDF normalization")
print("- Similarity measures (cosine similarity)")
print("- Neural networks")

normalizer = Normalizer(norm='l2')  # L2 norm (default)
X_scaled = normalizer.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols)

print("\n" + "-"*80)
print("First 10 rows after Normalizer:")
print("-"*80)
print(X_scaled_df.head(10))

print("\n" + "-"*80)
print("Verification:")
print("-"*80)
print("Calculating norm (magnitude) of each sample:")
norms = np.sqrt((X_scaled ** 2).sum(axis=1))
print(f"First 10 norms (should all be ~1): {norms[:10]}")
print(f"\nAll norms equal to 1: {np.allclose(norms, 1)}")
print(f"Min norm: {norms.min()}")
print(f"Max norm: {norms.max()}")
X_scaled_df.to_csv('data_scaled_normalizer.csv', index=False)
print("\n✓ Scaled data saved to: data_scaled_normalizer.csv")
print(f"✓ Output shape: {X_scaled_df.shape}")
print(f"✓ Columns: {list(X_scaled_df.columns)}")
