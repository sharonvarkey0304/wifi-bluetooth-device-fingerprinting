import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# ==============================
# 1. Project paths
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "ble_features.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

print("Reading BLE features from:")
print(FEATURE_FILE)

if not FEATURE_FILE.exists():
    raise FileNotFoundError("ble_features.csv not found. Please run extract_ble_features.py first.")


# ==============================
# 2. Load BLE dataset
# ==============================

df = pd.read_csv(FEATURE_FILE)

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst rows:")
print(df.head())


# ==============================
# 3. Choose target column
# ==============================

# Use "environment" first because BLE dataset usually has LOS and NLOS folders.
# This predicts whether the BLE signal is Line-of-Sight or Non-Line-of-Sight.
#
# Later, if your ble_features.csv has good device labels, you can change this to:
# TARGET_COLUMN = "device_label"

TARGET_COLUMN = "environment"

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in ble_features.csv")

df = df.dropna(subset=[TARGET_COLUMN])

print("\nTarget column:", TARGET_COLUMN)
print("\nClass distribution:")
print(df[TARGET_COLUMN].value_counts())


# Remove classes with fewer than 2 samples
class_counts = df[TARGET_COLUMN].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df = df[df[TARGET_COLUMN].isin(valid_classes)]

if df[TARGET_COLUMN].nunique() < 2:
    raise ValueError("Not enough classes for training. Need at least 2 classes with 2 or more samples each.")


# ==============================
# 4. Prepare features and labels
# ==============================

non_feature_columns = [
    "source_file",
    "device_label",
    "environment"
]

feature_df = df.drop(columns=[col for col in non_feature_columns if col in df.columns])

# Use only numeric columns for ML
feature_df = feature_df.select_dtypes(include=[np.number])

print("\nSelected numeric feature columns:")
print(feature_df.columns.tolist())

if feature_df.shape[1] == 0:
    raise ValueError("No numeric feature columns found in ble_features.csv")

X = feature_df
y = df[TARGET_COLUMN]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nEncoded classes:")
print(label_encoder.classes_)


# ==============================
# 5. Train/test split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])


# ==============================
# 6. Define ML models
# ==============================

knn_neighbors = min(5, len(X_train))

models = {
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=knn_neighbors))
    ])
}


# ==============================
# 7. Train and evaluate models
# ==============================

results = []

best_model = None
best_model_name = None
best_accuracy = 0

for model_name, model in models.items():
    print(f"\nTraining model: {model_name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = model_name

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax)
    plt.title(f"BLE Confusion Matrix - {model_name}")
    plt.tight_layout()

    cm_file = RESULTS_DIR / f"ble_confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(cm_file)
    plt.close()

    print("Saved confusion matrix:", cm_file)


# ==============================
# 8. Save results
# ==============================

results_df = pd.DataFrame(results)

results_file = RESULTS_DIR / "ble_ml_model_results.csv"
results_df.to_csv(results_file, index=False)

print("\nBLE ML results saved to:")
print(results_file)

print("\nFinal BLE ML results:")
print(results_df)


# ==============================
# 9. Save best model
# ==============================

best_model_file = MODELS_DIR / "best_ble_model.pkl"
encoder_file = MODELS_DIR / "ble_label_encoder.pkl"

joblib.dump(best_model, best_model_file)
joblib.dump(label_encoder, encoder_file)

print("\nBest BLE model:", best_model_name)
print("Best BLE accuracy:", best_accuracy)
print("Saved best BLE model to:", best_model_file)


# ==============================
# 10. Accuracy comparison graph
# ==============================

plt.figure(figsize=(8, 5))
plt.bar(results_df["model"], results_df["accuracy"])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("BLE Machine Learning Model Accuracy Comparison")
plt.ylim(0, 1)
plt.tight_layout()

graph_file = RESULTS_DIR / "ble_ml_accuracy_comparison.png"
plt.savefig(graph_file)
plt.close()

print("Saved BLE accuracy graph to:")
print(graph_file)