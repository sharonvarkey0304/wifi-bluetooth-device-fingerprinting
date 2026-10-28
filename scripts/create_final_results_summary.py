import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

summary = pd.DataFrame([
    {
        "experiment": "Wi-Fi Device ID Classification",
        "best_model": "Random Forest",
        "accuracy": 0.9024,
        "description": "Classifies individual labelled Wi-Fi devices"
    },
    {
        "experiment": "Wi-Fi Device Type Classification",
        "best_model": "SVM / KNN",
        "accuracy": 0.9512,
        "description": "Classifies device type such as Phone or Tablet"
    },
    {
        "experiment": "BLE Behaviour Classification",
        "best_model": "KNN",
        "accuracy": 0.6886,
        "description": "Classifies BLE signal behaviour/environment"
    }
])

output_csv = RESULTS_DIR / "final_project_results_summary.csv"
output_png = RESULTS_DIR / "final_project_results_summary.png"

summary.to_csv(output_csv, index=False)

print("Final Project Results Summary:")
print(summary)

plt.figure(figsize=(10, 5))
plt.bar(summary["experiment"], summary["accuracy"])
plt.xlabel("Experiment")
plt.ylabel("Best Accuracy")
plt.title("Final Project Results Summary")
plt.xticks(rotation=25, ha="right")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(output_png)
plt.close()

print("\nSaved CSV:", output_csv)
print("Saved graph:", output_png)