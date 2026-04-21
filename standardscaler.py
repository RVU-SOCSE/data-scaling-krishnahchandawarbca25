import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('data.csv')

print("="*80)
print("STANDARD SCALER - PROGRAM 1")
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
print("APPLYING STANDARD SCALER")
print("="*80)
print("\nFormula: z = (x - mean) / std")
print("Transforms data to have mean=0 and standard deviation=1")
print("\nUse case: ")
print("- Linear/Logistic Regression")
print("- PCA")
print("- K-Means")
print("- SVM")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols)

print("\n" + "-"*80)
print("First 10 rows after StandardScaler:")
print("-"*80)
print(X_scaled_df.head(10))

print("\n" + "-"*80)
print("Verification:")
print("-"*80)
print(f"Mean (should be ~0): {X_scaled_df.mean().values}")
print(f"Std Dev (should be ~1): {X_scaled_df.std().values}")
print(f"Min values: {X_scaled_df.min().values}")
print(f"Max values: {X_scaled_df.max().values}")
X_scaled_df.to_csv('data_scaled_standard.csv', index=False)
print("\n✓ Scaled data saved to: data_scaled_standard.csv")
print(f"✓ Output shape: {X_scaled_df.shape}")
print(f"✓ Columns: {list(X_scaled_df.columns)}")
