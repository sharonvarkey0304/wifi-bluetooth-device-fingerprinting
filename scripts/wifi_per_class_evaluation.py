import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "final_analysis" / "wifi_per_class"


def find_file(filename: str) -> Path | None:
    possible_paths = [
        PROJECT_ROOT / filename,
        PROJECT_ROOT / "results" / filename,
        PROJECT_ROOT / "raw" / filename,
        PROJECT_ROOT / "data" / filename,
        PROJECT_ROOT / "datasets" / filename,
    ]

    for path in possible_paths:
        if path.exists():
            return path

    for path in PROJECT_ROOT.rglob(filename):
        if "venv" not in str(path).lower():
            return path

    return None


def choose_target_column(df: pd.DataFrame, requested_target: str) -> str:
    if requested_target in df.columns:
        return requested_target

    candidates = ["type", "device_type", "label", "class", "target", "category"]
    for col in candidates:
        if col in df.columns:
            print(f"Target column '{requested_target}' not found. Using '{col}' instead.")
            return col

    raise ValueError(
        f"Target column '{requested_target}' was not found.\n"
        f"Available columns are:\n{list(df.columns)}"
    )


def build_feature_matrix(df: pd.DataFrame, target_col: str):
    y_raw = df[target_col].astype(str)

    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])
    X = X.dropna(axis=1, how="all")

    if X.empty:
        raise ValueError(
            "No numeric feature columns were found. "
            "Please check that your Wi-Fi feature file contains numeric features."
        )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    return X, y, label_encoder


def safe_train_test_split(X, y, test_size, random_state):
    unique_classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    if min_count >= 2:
        required_test_size = len(unique_classes) / len(y)
        final_test_size = max(test_size, required_test_size + 0.01)
        final_test_size = min(final_test_size, 0.4)

        return train_test_split(
            X,
            y,
            test_size=final_test_size,
            random_state=random_state,
            stratify=y,
        )

    print("Warning: At least one class has fewer than 2 samples. Stratified split is not possible.")
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=None,
    )


def save_confusion_matrix(y_test, y_pred, class_names, title, output_path: Path):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def save_random_forest_feature_importance(model, feature_names, output_dir: Path):
    rf_model = model.named_steps["model"]

    if not hasattr(rf_model, "feature_importances_"):
        return

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(output_dir / "random_forest_feature_importance.csv", index=False)

    top_df = importance_df.head(15).sort_values("importance")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_df["feature"], top_df["importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Top Wi-Fi Feature Importance - Random Forest")
    plt.tight_layout()
    plt.savefig(output_dir / "random_forest_feature_importance.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Path to wifi_features.csv")
    parser.add_argument("--target", default="type", help="Target column, usually type")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = PROJECT_ROOT / csv_path
    else:
        csv_path = find_file("wifi_features.csv")

    if csv_path is None or not csv_path.exists():
        print("ERROR: Could not find wifi_features.csv.")
        print("Try one of these:")
        print("python scripts\\wifi_per_class_evaluation.py --csv wifi_features.csv --target type")
        print("python scripts\\wifi_per_class_evaluation.py --csv results\\wifi_features.csv --target type")
        sys.exit(1)

    print(f"Using Wi-Fi feature file: {csv_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    target_col = choose_target_column(df, args.target)

    class_distribution = (
        df[target_col]
        .astype(str)
        .value_counts()
        .rename_axis("class")
        .reset_index(name="count")
    )
    class_distribution.to_csv(OUTPUT_DIR / "class_distribution.csv", index=False)

    print("\nClass distribution:")
    print(class_distribution)

    X, y, label_encoder = build_feature_matrix(df, target_col)
    class_names = label_encoder.classes_

    X_train, X_test, y_train, y_test = safe_train_test_split(
        X,
        y,
        args.test_size,
        args.random_state,
    )

    models = {
        "Random_Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                random_state=args.random_state,
                class_weight="balanced",
            )),
        ]),
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", class_weight="balanced")),
        ]),
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
    }

    summary_rows = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        report_dict = classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(OUTPUT_DIR / f"{model_name.lower()}_classification_report.csv")

        save_confusion_matrix(
            y_test,
            y_pred,
            class_names,
            f"{model_name} Confusion Matrix",
            OUTPUT_DIR / f"{model_name.lower()}_confusion_matrix.png",
        )

        if model_name == "Random_Forest":
            save_random_forest_feature_importance(model, X.columns, OUTPUT_DIR)

        summary_rows.append({
            "model": model_name,
            "accuracy": accuracy,
            "macro_precision": report_dict["macro avg"]["precision"],
            "macro_recall": report_dict["macro avg"]["recall"],
            "macro_f1": report_dict["macro avg"]["f1-score"],
            "weighted_precision": report_dict["weighted avg"]["precision"],
            "weighted_recall": report_dict["weighted avg"]["recall"],
            "weighted_f1": report_dict["weighted avg"]["f1-score"],
        })

        print("=" * 70)
        print(model_name)
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0,
        ))

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "wifi_model_comparison_with_macro_weighted_metrics.csv",
        index=False,
    )

    print("\nDONE. Wi-Fi per-class results saved here:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()