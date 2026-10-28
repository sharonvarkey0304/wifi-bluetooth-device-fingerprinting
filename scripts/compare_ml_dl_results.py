import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ML_RESULTS = PROJECT_ROOT / "results" / "ml_model_results_type.csv"
DL_RESULTS = PROJECT_ROOT / "results" / "deep_learning_results_type.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "ml_vs_dl_comparison.csv"
GRAPH_FILE = PROJECT_ROOT / "results" / "ml_vs_dl_accuracy_comparison.png"

if not ML_RESULTS.exists():
    raise FileNotFoundError("ml_model_results_type.csv not found. Run train_ml_models.py first.")

if not DL_RESULTS.exists():
    raise FileNotFoundError("deep_learning_results_type.csv not found. Run train_deep_learning.py first.")

ml_df = pd.read_csv(ML_RESULTS)
dl_df = pd.read_csv(DL_RESULTS)

combined = pd.concat([ml_df, dl_df], ignore_index=True)

combined.to_csv(OUTPUT_FILE, index=False)

print("\nCombined ML and Deep Learning Results:")
print(combined)
print("\nSaved comparison CSV to:")
print(OUTPUT_FILE)

plt.figure(figsize=(10, 5))
plt.bar(combined["model"], combined["accuracy"])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("ML vs Deep Learning Accuracy Comparison")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(GRAPH_FILE)
plt.close()

print("Saved comparison graph to:")
print(GRAPH_FILE)