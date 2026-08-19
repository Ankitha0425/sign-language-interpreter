import pandas as pd

FILE_PATH = "data/raw/Indian Sign Language Gesture Landmarks.csv"

print("Loading dataset...")

df = pd.read_csv(FILE_PATH)

print("\n========== DATASET INFORMATION ==========")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== COLUMN NAMES ==========")

for i, column in enumerate(df.columns):
    print(i, ":", column)

print("\n========== FIRST 5 ROWS ==========")

print(df.head())

print("\n========== DATA TYPES ==========")

print(df.dtypes)

print("\n========== MISSING VALUES ==========")

missing = df.isnull().sum()

print(missing[missing > 0])

print("\n========== DUPLICATES ==========")

print("Duplicate rows:", df.duplicated().sum())

print("\n========== DATASET SUMMARY ==========")

print(df.describe(include="all"))

print("\n========== POSSIBLE LABEL COLUMNS ==========")

for column in df.columns:
    if df[column].dtype == "object":
        print(
            column,
            "->",
            df[column].nunique(),
            "unique values"
        )