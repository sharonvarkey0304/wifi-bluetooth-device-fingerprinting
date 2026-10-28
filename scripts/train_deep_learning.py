import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


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
    raise FileNotFoundError("wifi_features.csv not found. Please run extract_wifi_features.py first.")


# ==============================
# 2. Load dataset
# ==============================

df = pd.read_csv(FEATURE_FILE)

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst five rows:")
print(df.head())


# ==============================
# 3. Choose target column
# ==============================

# Start with device_id because it is clearly labelled.
# Later you can change to "type" if you want phone/tablet/laptop classification.
#TARGET_COLUMN = "device_id"
TARGET_COLUMN = "type"

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

df = df.dropna(subset=[TARGET_COLUMN])

# Remove classes with fewer than 2 samples.
# This avoids train/test split errors.
class_counts = df[TARGET_COLUMN].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df = df[df[TARGET_COLUMN].isin(valid_classes)]

print("\nTarget column:", TARGET_COLUMN)
print("\nClass distribution after filtering:")
print(df[TARGET_COLUMN].value_counts())


# ==============================
# 4. Prepare X and y
# ==============================

non_feature_columns = [
    "device_id",
    "vendor",
    "model",
    "type",
    "device_os",
    "pcap_filename"
]

feature_df = df.drop(columns=[col for col in non_feature_columns if col in df.columns])

# Use only numeric columns
feature_df = feature_df.select_dtypes(include=[np.number])

print("\nSelected numeric feature columns:")
print(feature_df.columns.tolist())

X = feature_df
y = df[TARGET_COLUMN]

if X.shape[1] == 0:
    raise ValueError("No numeric features found. Check wifi_features.csv.")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

num_classes = len(label_encoder.classes_)

print("\nNumber of classes:", num_classes)
print("Classes:")
print(label_encoder.classes_)


# ==============================
# 5. Handle missing values and scale features
# ==============================

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)


# ==============================
# 6. Train/test split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])
print("Input features:", X_train.shape[1])


# ==============================
# 7. Build deep learning model
# ==============================

model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),

    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.30),

    layers.Dense(32, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.20),

    layers.Dense(16, activation="relu"),

    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel summary:")
model.summary()


# ==============================
# 8. Train model
# ==============================

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)


# ==============================
# 9. Evaluate model
# ==============================

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)

precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\nDeep Learning Results")
print("----------------------")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")
print(f"F1-score:      {f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    zero_division=0
))


# ==============================
# 10. Save results CSV
# ==============================

results_df = pd.DataFrame([{
    "model": "Deep Learning Neural Network",
    "accuracy": test_accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "loss": test_loss
}])

results_file = RESULTS_DIR / "deep_learning_results_type.csv"
results_df.to_csv(results_file, index=False)

print("\nSaved deep learning results to:")
print(results_file)


# ==============================
# 11. Save training graph
# ==============================

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Deep Learning Training and Validation Accuracy")
plt.legend()
plt.tight_layout()

training_graph_file = RESULTS_DIR / "deep_learning_training_graph_type.png"
plt.savefig(training_graph_file)
plt.close()

print("Saved training graph to:")
print(training_graph_file)


plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Deep Learning Training and Validation Loss")
plt.legend()
plt.tight_layout()

loss_graph_file = RESULTS_DIR / "deep_learning_loss_graph_type.png"
plt.savefig(loss_graph_file)
plt.close()

print("Saved loss graph to:")
print(loss_graph_file)


# ==============================
# 12. Save confusion matrix
# ==============================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, xticks_rotation=90)
plt.title("Confusion Matrix - Deep Learning")
plt.tight_layout()

cm_file = RESULTS_DIR / "deep_learning_confusion_matrix_type.png"
plt.savefig(cm_file)
plt.close()

print("Saved confusion matrix to:")
print(cm_file)


# ==============================
# 13. Save model and preprocessing tools
# ==============================

model_file = MODELS_DIR / "deep_learning_model.keras"
encoder_file = MODELS_DIR / "dl_label_encoder.pkl"
scaler_file = MODELS_DIR / "dl_scaler.pkl"
imputer_file = MODELS_DIR / "dl_imputer.pkl"
features_file = MODELS_DIR / "dl_feature_columns.pkl"

model.save(model_file)
joblib.dump(label_encoder, encoder_file)
joblib.dump(scaler, scaler_file)
joblib.dump(imputer, imputer_file)
joblib.dump(list(X.columns), features_file)

print("\nSaved deep learning model and tools:")
print(model_file)
print(encoder_file)
print(scaler_file)
print(imputer_file)
print(features_file)

print("\nDeep learning training completed successfully.")