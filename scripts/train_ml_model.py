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

FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "wifi_features.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

print("Reading features from:")
print(FEATURE_FILE)

if not FEATURE_FILE.exists():
    raise FileNotFoundError("wifi_features.csv not found. Please run feature extraction first.")


# ==============================
# 2. Load dataset
# ==============================

df = pd.read_csv(FEATURE_FILE)

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns)
print("\nFirst rows:")
print(df.head())


# ==============================
# 3. Choose prediction target
# ==============================

# Start with device_id because it is the clearest label in this dataset.
# Later you can change this to "type" if you want to classify Phone/Laptop/Tablet.
TARGET_COLUMN = "device_id"


if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

df = df.dropna(subset=[TARGET_COLUMN])

print("\nTarget column:", TARGET_COLUMN)
print("\nClass distribution:")
print(df[TARGET_COLUMN].value_counts())


# ==============================
# 4. Prepare features
# ==============================

# Remove non-feature columns
non_feature_columns = [
    "device_id",
    "vendor",
    "model",
    "type",
    "device_os",
    "pcap_filename"
]

# Use only numeric columns for machine learning
feature_df = df.drop(columns=[col for col in non_feature_columns if col in df.columns])
feature_df = feature_df.select_dtypes(include=[np.number])

print("\nSelected feature columns:")
print(feature_df.columns.tolist())

X = feature_df
y = df[TARGET_COLUMN]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nEncoded classes:")
print(list(label_encoder.classes_))


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
# 6. Define models
# ==============================

models = {
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42))
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=True, random_state=42))
    ]),

    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
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

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

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

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=90)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    cm_file = RESULTS_DIR / f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(cm_file)
    plt.close()

    print("Saved confusion matrix:", cm_file)


# ==============================
# 8. Save results
# ==============================

results_df = pd.DataFrame(results)
results_file = RESULTS_DIR / "ml_model_results_type.csv"
results_df.to_csv(results_file, index=False)

print("\nModel comparison results saved to:")
print(results_file)

print("\nResults:")
print(results_df)


# ==============================
# 9. Save best model
# ==============================

best_model_file = MODELS_DIR / "best_wifi_fingerprinting_model.pkl"
encoder_file = MODELS_DIR / "label_encoder.pkl"

joblib.dump(best_model, best_model_file)
joblib.dump(label_encoder, encoder_file)

print("\nBest model:", best_model_name)
print("Best accuracy:", best_accuracy)
print("Saved best model to:", best_model_file)
print("Saved label encoder to:", encoder_file)


# ==============================
# 10. Plot model comparison graph
# ==============================

plt.figure(figsize=(8, 5))
plt.bar(results_df["model"], results_df["accuracy"])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Machine Learning Model Accuracy Comparison")
plt.ylim(0, 1)
plt.tight_layout()

graph_file = RESULTS_DIR / "ml_accuracy_comparison.png"
plt.savefig(graph_file)
plt.close()

print("Saved accuracy graph to:", graph_file)