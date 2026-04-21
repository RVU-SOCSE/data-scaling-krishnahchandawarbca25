import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('data.csv')

print("="*80)
print("MINMAX SCALER - PROGRAM 3")
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
print("APPLYING MINMAX SCALER")
print("="*80)
print("\nFormula: x_scaled = (x - min) / (max - min)")
print("Scales features to range [0, 1]")
print("\nUse case: ")
print("- Neural Networks")
print("- Deep Learning")
print("- Image Processing")
print("- When bounded range [0,1] is needed")

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols)

print("\n" + "-"*80)
print("First 10 rows after MinMaxScaler:")
print("-"*80)
print(X_scaled_df.head(10))

print("\n" + "-"*80)
print("Verification:")
print("-"*80)
print(f"Min values (should be ~0): {X_scaled_df.min().values}")
print(f"Max values (should be ~1): {X_scaled_df.max().values}")
print(f"All values in [0, 1]: {np.all((X_scaled >= 0) & (X_scaled <= 1))}")

print("\n" + "-"*80)
print("Range Information:")
print("-"*80)
for col in numeric_cols:
    original_min = X[col].min()
    original_max = X[col].max()
    original_range = original_max - original_min
    print(f"{col}:")
    print(f"  Original range: [{original_min}, {original_max}] (span: {original_range})")
    print(f"  Scaled range: [{X_scaled_df[col].min()}, {X_scaled_df[col].max()}]")
X_scaled_df.to_csv('data_scaled_minmax.csv', index=False)
print("\n✓ Scaled data saved to: data_scaled_minmax.csv")
print(f"✓ Output shape: {X_scaled_df.shape}")
print(f"✓ Columns: {list(X_scaled_df.columns)}")
