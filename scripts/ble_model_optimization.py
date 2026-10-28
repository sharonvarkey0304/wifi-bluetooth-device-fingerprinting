from pathlib import Path
import argparse
import json
import warnings

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
PREVIOUS_BLE_ACCURACY = 68.86
BEST_WIFI_ACCURACY = 95.12


def project_root():
    return Path(__file__).resolve().parents[1]


def find_csv(user_path=None):
    root = project_root()
    paths = []
    if user_path:
        paths += [root / user_path, Path(user_path)]
    paths += [
        root / "data" / "processed" / "ble_features.csv",
        root / "results" / "ble_features.csv",
        root / "ble_features.csv",
        root / "data" / "ble_features.csv",
    ]
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find ble_features.csv. Use --csv path.")


def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_").replace("-", "_").replace("/", "_").replace("\\", "_") for c in df.columns]
    return df


def choose_target(df, target=None):
    if target and target in df.columns:
        return target
    for c in ["environment", "behaviour", "behavior", "activity", "label", "class", "target", "device_label"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find target column. Use --target environment.")


def add_features(df):
    df = df.copy()
    if "max_rssi" in df.columns and "min_rssi" in df.columns:
        df["rssi_range_engineered"] = df["max_rssi"] - df["min_rssi"]
    if "avg_rssi" in df.columns:
        df["avg_rssi_abs"] = df["avg_rssi"].abs()
        df["avg_rssi_squared"] = df["avg_rssi"] ** 2
    if "std_rssi" in df.columns and "avg_rssi" in df.columns:
        df["rssi_cv"] = df["std_rssi"] / (df["avg_rssi"].abs() + 1e-6)
        df["rssi_stability_score"] = 1 / (1 + df["std_rssi"].abs())
    if "rssi_range" in df.columns and "avg_rssi" in df.columns:
        df["rssi_range_ratio"] = df["rssi_range"] / (df["avg_rssi"].abs() + 1e-6)
    if "max_rssi" in df.columns and "avg_rssi" in df.columns:
        df["rssi_peak_difference"] = df["max_rssi"] - df["avg_rssi"]
    if "avg_rssi" in df.columns and "min_rssi" in df.columns:
        df["rssi_drop_difference"] = df["avg_rssi"] - df["min_rssi"]
    if "packet_count" in df.columns:
        df["log_packet_count"] = np.log1p(df["packet_count"].clip(lower=0))
    if "packet_rate" in df.columns:
        df["log_packet_rate"] = np.log1p(df["packet_rate"].clip(lower=0))
    if "packet_count" in df.columns and "avg_rssi" in df.columns:
        df["packet_count_x_rssi"] = df["packet_count"] * df["avg_rssi"].abs()
    if "packet_rate" in df.columns and "avg_rssi" in df.columns:
        df["packet_rate_x_rssi"] = df["packet_rate"] * df["avg_rssi"].abs()
    if "duration_seconds" in df.columns and "packet_count" in df.columns:
        df["packets_per_second"] = df["packet_count"] / (df["duration_seconds"] + 1e-6)

    rssi_cols = [c for c in df.columns if "rssi" in c.lower() and pd.api.types.is_numeric_dtype(df[c])]
    if len(rssi_cols) >= 2:
        df["all_rssi_mean"] = df[rssi_cols].mean(axis=1)
        df["all_rssi_std"] = df[rssi_cols].std(axis=1)
        df["all_rssi_min"] = df[rssi_cols].min(axis=1)
        df["all_rssi_max"] = df[rssi_cols].max(axis=1)
    return df.replace([np.inf, -np.inf], np.nan)


def filter_outliers(df):
    df = df.copy()
    cols = [c for c in df.columns if "rssi" in c.lower() and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return df
    mask = pd.Series(True, index=df.index)
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        mask = mask & df[c].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    filtered = df[mask].reset_index(drop=True)
    return filtered if len(filtered) >= int(len(df) * 0.75) else df


def prepare_xy(df, target_col, remove_rssi=False):
    y_raw = df[target_col].astype(str)
    drop = [
        target_col, "source_file", "file", "filename", "device_id", "device_label",
        "mac", "address", "timestamp", "time", "datetime", "date"
    ]
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    if remove_rssi:
        X = X.drop(columns=[c for c in X.columns if "rssi" in c.lower()], errors="ignore")
    X = pd.get_dummies(X, drop_first=False).select_dtypes(include=[np.number])
    X = X.dropna(axis=1, how="all")
    X = X.drop(columns=[c for c in X.columns if X[c].nunique(dropna=True) <= 1], errors="ignore")
    if X.empty:
        raise ValueError("No usable features found.")
    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)
    return X, y, enc


def cv_accuracy(model, X, y, splits):
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr, va in cv.split(X, y):
        m = clone(model)
        m.fit(X.iloc[tr], y[tr])
        scores.append(accuracy_score(y[va], m.predict(X.iloc[va])))
    return float(np.mean(scores))


def tune(name, base, grid, X, y, splits):
    best_score, best_params, best_model = -1, None, None
    all_params = list(ParameterGrid(grid))
    for i, params in enumerate(all_params, 1):
        m = clone(base)
        m.set_params(**params)
        score = cv_accuracy(m, X, y, splits)
        if score > best_score:
            best_score, best_params, best_model = score, params, clone(m)
        if i % 10 == 0 or i == len(all_params):
            print(f"  {name}: tested {i}/{len(all_params)} | best CV = {round(best_score*100, 2)}%")
    return best_model, best_params, best_score


def models():
    return {
        "KNN_Tuned": (
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", KNeighborsClassifier())]),
            {"model__n_neighbors": [1,3,5,7,9,11,15,21], "model__weights": ["uniform","distance"], "model__metric": ["euclidean","manhattan"]}
        ),
        "SVM_Tuned": (
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", SVC(random_state=RANDOM_STATE))]),
            {"model__C": [0.1,1,10,50,100], "model__gamma": ["scale","auto",0.01,0.1], "model__kernel": ["rbf"], "model__class_weight": [None,"balanced"]}
        ),
        "Random_Forest_Tuned": (
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(random_state=RANDOM_STATE))]),
            {"model__n_estimators": [200,500], "model__max_depth": [None,10,20], "model__min_samples_leaf": [1,2,4], "model__class_weight": [None,"balanced"]}
        ),
        "Extra_Trees_Tuned": (
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesClassifier(random_state=RANDOM_STATE))]),
            {"model__n_estimators": [200,500], "model__max_depth": [None,10,20], "model__min_samples_leaf": [1,2,4], "model__class_weight": [None,"balanced"]}
        ),
        "Gradient_Boosting": (
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))]),
            {"model__n_estimators": [100,200], "model__learning_rate": [0.03,0.05,0.1], "model__max_depth": [2,3,4]}
        ),
    }


def save_cm(cm, labels, path, title):
    plt.figure(figsize=(7,5))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def evaluate(name, model, X_train, X_test, y_train, y_test, labels, out):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    bal = balanced_accuracy_score(y_test, pred)
    mp, mr, mf, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)
    pd.DataFrame(classification_report(y_test, pred, target_names=labels, output_dict=True, zero_division=0)).transpose().to_csv(out / f"{name}_classification_report.csv")
    save_cm(confusion_matrix(y_test, pred), labels, out / f"{name}_confusion_matrix.png", f"{name} Confusion Matrix")
    return {"model": name, "accuracy": acc, "accuracy_percent": round(acc*100,2), "balanced_accuracy": bal, "balanced_accuracy_percent": round(bal*100,2), "macro_precision": mp, "macro_recall": mr, "macro_f1": mf, "weighted_precision": wp, "weighted_recall": wr, "weighted_f1": wf, "estimator": model}


def run_dataset(df, target, version, remove_rssi, out):
    X, y, enc = prepare_xy(df, target, remove_rssi)
    labels = list(enc.classes_)
    strat = y if pd.Series(y).value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=strat)
    min_count = int(pd.Series(ytr).value_counts().min())
    splits = 5 if min_count >= 5 else 3 if min_count >= 3 else 2
    rows = []
    for name, (base, grid) in models().items():
        print(f"Running {version} - {name}")
        best, params, cv = tune(name, base, grid, Xtr, ytr, splits)
        row = evaluate(name, best, Xtr, Xte, ytr, yte, labels, out)
        row["dataset_version"] = version
        row["cv_accuracy"] = cv
        row["cv_accuracy_percent"] = round(cv*100,2)
        row["best_parameters"] = json.dumps(params)
        rows.append(row)
    dfres = pd.DataFrame(rows)
    safe = version.lower().replace(" ", "_").replace("/", "_")
    dfres.drop(columns=["estimator"], errors="ignore").to_csv(out / f"{safe}_results.csv", index=False)
    joblib.dump(enc, out / f"{safe}_label_encoder.pkl")
    pd.Series(X.columns).to_csv(out / f"{safe}_feature_columns.csv", index=False, header=["feature"])
    return dfres


def plot_results(allres, out):
    df = allres.drop(columns=["estimator"], errors="ignore").copy()
    df["label"] = df["dataset_version"].astype(str) + " - " + df["model"].astype(str)
    df = df.sort_values("accuracy_percent")
    plt.figure(figsize=(12,8))
    plt.barh(df["label"], df["accuracy_percent"])
    plt.axvline(PREVIOUS_BLE_ACCURACY, linestyle="--")
    plt.axvline(BEST_WIFI_ACCURACY, linestyle="--")
    plt.xlabel("Accuracy (%)")
    plt.title("BLE Optimisation Result Comparison")
    plt.tight_layout()
    plt.savefig(out / "ble_optimisation_accuracy_comparison.png", dpi=300)
    plt.close()

    best = float(df["accuracy_percent"].max())
    plt.figure(figsize=(8,5))
    plt.bar(["Previous BLE", "Optimised BLE", "Best Wi-Fi"], [PREVIOUS_BLE_ACCURACY, best, BEST_WIFI_ACCURACY])
    plt.ylabel("Accuracy (%)")
    plt.title("BLE vs Wi-Fi Accuracy Gap")
    plt.tight_layout()
    plt.savefig(out / "ble_vs_wifi_accuracy_gap.png", dpi=300)
    plt.close()


def write_outputs(allres, out, models_dir):
    clean = allres.drop(columns=["estimator"], errors="ignore").sort_values("accuracy", ascending=False)
    clean.to_csv(out / "ble_optimization_all_results.csv", index=False)
    best = clean.iloc[0]
    improvement = float(best["accuracy_percent"]) - PREVIOUS_BLE_ACCURACY
    gap = BEST_WIFI_ACCURACY - float(best["accuracy_percent"])
    summary = f"""BLE Optimisation Summary

Previous BLE best result: {PREVIOUS_BLE_ACCURACY}% using KNN.
Best optimised BLE result: {best["accuracy_percent"]}% using {best["model"]}.
Best dataset version: {best["dataset_version"]}.
Cross-validation accuracy: {best["cv_accuracy_percent"]}%.

Improvement over previous BLE result: {round(improvement, 2)} percentage points.
Remaining gap compared with best Wi-Fi result of {BEST_WIFI_ACCURACY}%: {round(gap, 2)} percentage points.

Macro precision: {round(float(best["macro_precision"]), 4)}
Macro recall: {round(float(best["macro_recall"]), 4)}
Macro F1-score: {round(float(best["macro_f1"]), 4)}
Weighted F1-score: {round(float(best["weighted_f1"]), 4)}
Balanced accuracy: {round(float(best["balanced_accuracy"]), 4)}

Interpretation:
This optimisation stage used RSSI feature engineering, outlier filtering and tuned machine learning models to investigate whether the BLE result could be improved. If BLE remains lower than Wi-Fi, this supports the final dissertation finding that BLE classification is more challenging because RSSI values are unstable and sensitive to environmental variation.
"""
    (out / "best_ble_optimization_summary.txt").write_text(summary, encoding="utf-8")
    dissertation = f"""BLE Classification Improvement Discussion

The original BLE behaviour classification achieved 68.86% accuracy using the KNN algorithm. Because this was lower than the Wi-Fi device-type classification accuracy of 95.12%, an optimisation stage was added to investigate whether BLE performance could be improved.

The optimisation stage included RSSI-based feature engineering, RSSI outlier filtering and hyperparameter tuning for KNN, SVM, Random Forest, Extra Trees and Gradient Boosting models. The best optimised BLE result achieved {best["accuracy_percent"]}% accuracy using {best["model"]} on the {best["dataset_version"]} dataset version. This represents a change of {round(improvement, 2)} percentage points compared with the original BLE result.

Even after optimisation, the BLE result should be critically compared with the Wi-Fi result. BLE RSSI values may fluctuate due to distance, obstacles, interference, device orientation and environmental conditions. These variations make BLE behavioural features less stable and less separable than the Wi-Fi features used in this project. Therefore, the lower BLE performance is treated as an important finding and limitation rather than a failure of the project.
"""
    (out / "dissertation_text_ble_improvement.txt").write_text(dissertation, encoding="utf-8")
    best_full = allres.sort_values("accuracy", ascending=False).iloc[0]
    joblib.dump(best_full["estimator"], models_dir / "best_ble_optimized_model.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    root = project_root()
    out = root / "results" / "final_analysis" / "ble_improvement"
    out.mkdir(parents=True, exist_ok=True)
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    csv_path = find_csv(args.csv)
    print("Using BLE CSV:", csv_path)

    df = clean_columns(pd.read_csv(csv_path))
    target = choose_target(df, args.target)

    print("Target column:", target)
    print("Dataset shape:", df.shape)
    print("Class distribution:")
    print(df[target].value_counts())

    enhanced = add_features(df)
    filtered = filter_outliers(enhanced)

    df.to_csv(out / "original_ble_features_used.csv", index=False)
    enhanced.to_csv(out / "enhanced_ble_features.csv", index=False)
    filtered.to_csv(out / "filtered_enhanced_ble_features.csv", index=False)

    datasets = [
        ("Original Numeric Features", df, False),
        ("Enhanced RSSI Features", enhanced, False),
        ("Filtered Enhanced RSSI Features", filtered, False),
        ("No RSSI Control Features", enhanced, True),
    ]

    all_results = []
    for version, data, remove_rssi in datasets:
        print("")
        print("=" * 80)
        print("Dataset version:", version)
        print("=" * 80)
        sub = out / version.lower().replace(" ", "_").replace("/", "_")
        sub.mkdir(parents=True, exist_ok=True)
        try:
            all_results.append(run_dataset(data, target, version, remove_rssi, sub))
        except Exception as e:
            print("Skipped", version, "because:", e)

    if not all_results:
        raise RuntimeError("No experiments completed.")

    allres = pd.concat(all_results, ignore_index=True)
    write_outputs(allres, out, models_dir)
    plot_results(allres, out)

    best = allres.drop(columns=["estimator"], errors="ignore").sort_values("accuracy", ascending=False).iloc[0]
    print("")
    print("=" * 80)
    print("BLE OPTIMISATION COMPLETE")
    print("=" * 80)
    print("Previous BLE accuracy:", f"{PREVIOUS_BLE_ACCURACY}%")
    print("Best optimised BLE accuracy:", f'{best["accuracy_percent"]}%')
    print("Best model:", best["model"])
    print("Best dataset version:", best["dataset_version"])
    print("Output folder:", out)
    print("=" * 80)


if __name__ == "__main__":
    main()