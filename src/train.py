import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading processed data...")

X_train = np.load(f"{PROCESSED_DIR}/X_train.npy")
X_val = np.load(f"{PROCESSED_DIR}/X_val.npy")
X_test = np.load(f"{PROCESSED_DIR}/X_test.npy")

y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")
y_val = np.load(f"{PROCESSED_DIR}/y_val.npy")
y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Testing data:", X_test.shape)

print("\n====================================")
print("TRAINING RANDOM FOREST")
print("====================================")

rf_model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_val_predictions = rf_model.predict(X_val)

rf_val_accuracy = accuracy_score(
    y_val,
    rf_val_predictions
)

print("Random Forest Validation Accuracy:",
      rf_val_accuracy)

print("\n====================================")
print("TRAINING SVM")
print("====================================")

svm_model = SVC(
    kernel="rbf",
    C=10,
    probability=True,
    random_state=42
)

svm_model.fit(X_train, y_train)

svm_val_predictions = svm_model.predict(X_val)

svm_val_accuracy = accuracy_score(
    y_val,
    svm_val_predictions
)

print("SVM Validation Accuracy:",
      svm_val_accuracy)

print("\n====================================")
print("MODEL COMPARISON")
print("====================================")

print(
    f"Random Forest: {rf_val_accuracy * 100:.2f}%"
)

print(
    f"SVM:           {svm_val_accuracy * 100:.2f}%"
)

if rf_val_accuracy >= svm_val_accuracy:

    best_model = rf_model
    best_model_name = "Random Forest"
    best_accuracy = rf_val_accuracy

else:

    best_model = svm_model
    best_model_name = "SVM"
    best_accuracy = svm_val_accuracy

print("\nBest Model:", best_model_name)
print(
    f"Validation Accuracy: {best_accuracy * 100:.2f}%"
)

print("\n====================================")
print("EVALUATING BEST MODEL")
print("====================================")

test_predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)

model_path = f"{MODEL_DIR}/sign_language_model.pkl"

joblib.dump(
    best_model,
    model_path
)

print("\n====================================")
print("MODEL SAVED")
print("====================================")

print(model_path)