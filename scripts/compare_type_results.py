import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ML_FILE = PROJECT_ROOT / "results" / "ml_model_results_type.csv"
DL_FILE = PROJECT_ROOT / "results" / "deep_learning_results_type.csv"

OUTPUT_CSV = PROJECT_ROOT / "results" / "final_type_classification_comparison.csv"
OUTPUT_GRAPH = PROJECT_ROOT / "results" / "final_type_classification_comparison.png"

if not ML_FILE.exists():
    raise FileNotFoundError("ml_model_results_type.csv not found in results folder.")

if not DL_FILE.exists():
    raise FileNotFoundError("deep_learning_results_type.csv not found in results folder.")

ml_df = pd.read_csv(ML_FILE)
dl_df = pd.read_csv(DL_FILE)

combined = pd.concat([ml_df, dl_df], ignore_index=True)

combined.to_csv(OUTPUT_CSV, index=False)

print("Final Type Classification Comparison:")
print(combined)

plt.figure(figsize=(10, 5))
plt.bar(combined["model"], combined["accuracy"])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Device Type Classification: ML vs Deep Learning")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(OUTPUT_GRAPH)
plt.close()

print("\nSaved CSV:")
print(OUTPUT_CSV)

print("\nSaved graph:")
print(OUTPUT_GRAPH)