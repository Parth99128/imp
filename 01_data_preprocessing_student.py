# Practical 1: Data Preprocessing using Student Dataset
# Rename your student table CSV to Student.csv before running.

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("Student.csv")

print("First five rows:")
print(df.head())
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Fill missing numeric values with each column's median.
for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())

# Fill missing text values with the most common value.
for col in df.select_dtypes(exclude="number").columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# Convert text categories into numbers.
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Scale numeric columns.
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df)

print("\nAfter preprocessing:")
print(df.head())
print("\nMissing values after cleaning:")
print(df.isnull().sum())
df.to_csv("Student_cleaned.csv", index=False)
print("\nSaved as Student_cleaned.csv")
