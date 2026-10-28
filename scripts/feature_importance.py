import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "wifi_features.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "best_wifi_fingerprinting_model.pkl"
OUTPUT_CSV = PROJECT_ROOT / "results" / "feature_importance.csv"
OUTPUT_PNG = PROJECT_ROOT / "results" / "feature_importance.png"

df = pd.read_csv(FEATURE_FILE)

non_feature_columns = [
    "device_id",
    "vendor",
    "model",
    "type",
    "device_os",
    "pcap_filename"
]

X = df.drop(columns=[col for col in non_feature_columns if col in df.columns])
X = X.select_dtypes(include=[np.number])

model_pipeline = joblib.load(MODEL_FILE)

rf_model = model_pipeline.named_steps["model"]

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

importance_df.to_csv(OUTPUT_CSV, index=False)

print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUT_PNG)
plt.close()

print("Saved feature importance CSV:", OUTPUT_CSV)
print("Saved feature importance graph:", OUTPUT_PNG)