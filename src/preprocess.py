import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

DATA_PATH = "data/raw/Indian Sign Language Gesture Landmarks.csv"

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\nTarget distribution:")
print(df["target"].value_counts().sort_index())

X = df.drop(columns=["target"])
y = df["target"]

print("\nNumber of features:", X.shape[1])
print("Number of classes:", y.nunique())

X = X.replace([np.inf, -np.inf], np.nan)

if X.isnull().sum().sum() > 0:
    print("\nMissing values found. Filling with 0.")
    X = X.fillna(0)

print("\nSplitting dataset...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nSaving processed data...")

np.save(
    f"{PROCESSED_DIR}/X_train.npy",
    X_train_scaled
)

np.save(
    f"{PROCESSED_DIR}/X_val.npy",
    X_val_scaled
)

np.save(
    f"{PROCESSED_DIR}/X_test.npy",
    X_test_scaled
)

np.save(
    f"{PROCESSED_DIR}/y_train.npy",
    y_train.to_numpy()
)

np.save(
    f"{PROCESSED_DIR}/y_val.npy",
    y_val.to_numpy()
)

np.save(
    f"{PROCESSED_DIR}/y_test.npy",
    y_test.to_numpy()
)

joblib.dump(
    scaler,
    f"{MODEL_DIR}/scaler.pkl"
)

print("\n====================================")
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("====================================")

print("\nCreated files:")

print("data/processed/X_train.npy")
print("data/processed/X_val.npy")
print("data/processed/X_test.npy")
print("data/processed/y_train.npy")
print("data/processed/y_val.npy")
print("data/processed/y_test.npy")
print("models/scaler.pkl")