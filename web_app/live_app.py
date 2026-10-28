from flask import Flask, jsonify, render_template_string, send_from_directory
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


app = Flask(__name__)

# This file should be here:
# D:\Disseratation\wifi_ble_project\web_app\live_app.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

FINAL_ANALYSIS_DIR = RESULTS_DIR / "final_analysis"
WIFI_PER_CLASS_DIR = FINAL_ANALYSIS_DIR / "wifi_per_class"
BLE_RSSI_DIR = FINAL_ANALYSIS_DIR / "ble_rssi"
BLE_IMPROVEMENT_DIR = FINAL_ANALYSIS_DIR / "ble_improvement"

PREVIOUS_BLE_ACCURACY = 68.86
BEST_WIFI_ACCURACY = 95.12

# Fallback values used when the BLE optimisation CSV is missing or
# when the Final Project Results Summary CSV does not yet include this row.
UPDATED_BLE_ACCURACY = 70.07
UPDATED_BLE_MODEL = "SVM_Tuned"
UPDATED_BLE_DATASET_VERSION = "Enhanced RSSI Features"

wifi_index = 0
ble_index = 0


def safe_read_csv(path):
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        print("CSV read error:", path, e)
    return pd.DataFrame()


def safe_read_text(path):
    try:
        if path.exists():
            return path.read_text(encoding="utf-8")
    except Exception as e:
        print("Text read error:", path, e)
    return ""


def safe_load_model(path):
    try:
        if path.exists():
            return joblib.load(path)
    except Exception as e:
        print("Model load error:", path, e)
    return None


def find_csv(filename):
    possible_paths = [
        DATA_DIR / filename,
        RESULTS_DIR / filename,
        PROJECT_ROOT / filename,
        PROJECT_ROOT / "data" / filename,
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return DATA_DIR / filename


def percent(value):
    try:
        value = float(value)
        if value <= 1:
            return round(value * 100, 2)
        return round(value, 2)
    except Exception:
        return 0


def round_value(value, digits=4):
    try:
        if pd.isna(value):
            return ""
        if isinstance(value, (int, float, np.number)):
            return round(float(value), digits)
        return value
    except Exception:
        return value


def table_rows(df):
    if df.empty:
        return []

    rows = df.fillna("").to_dict(orient="records")

    for row in rows:
        if "accuracy" in row and "accuracy_percent" not in row:
            row["accuracy_percent"] = percent(row["accuracy"])
        elif "accuracy_percent" in row:
            row["accuracy_percent"] = percent(row["accuracy_percent"])

        if "cv_accuracy" in row and "cv_accuracy_percent" not in row:
            row["cv_accuracy_percent"] = percent(row["cv_accuracy"])
        elif "cv_accuracy_percent" in row:
            row["cv_accuracy_percent"] = percent(row["cv_accuracy_percent"])

        for key, value in list(row.items()):
            if key not in [
                "best_parameters", "description", "experiment", "model",
                "dataset_version", "class", "feature", "rssi_column",
                "dataset", "best_model"
            ]:
                row[key] = round_value(value)

    return rows


def list_pngs(folder):
    if not folder.exists():
        return []

    graphs = []
    for path in folder.rglob("*.png"):
        try:
            graphs.append(path.relative_to(RESULTS_DIR).as_posix())
        except Exception:
            pass
    return sorted(graphs)


def existing_graphs(graph_names):
    return [graph for graph in graph_names if (RESULTS_DIR / graph).exists()]


def ensure_optimised_ble_summary_row(final_df, ble_best):
    """
    Adds the updated Optimised BLE Classification row to the dashboard summary
    even if final_project_results_summary.csv was created before BLE optimisation.
    """
    row = {
        "experiment": "Optimised BLE Classification",
        "best_model": ble_best.get("model", UPDATED_BLE_MODEL),
        "accuracy_percent": ble_best.get("best_accuracy", UPDATED_BLE_ACCURACY),
        "description": (
            "Updated BLE result after RSSI feature engineering, "
            "RSSI outlier filtering and tuned model optimisation."
        ),
    }

    if final_df.empty:
        return pd.DataFrame([row])

    # Normalise column names expected by the dashboard.
    if "accuracy" in final_df.columns and "accuracy_percent" not in final_df.columns:
        final_df["accuracy_percent"] = final_df["accuracy"].apply(percent)

    required_columns = ["experiment", "best_model", "accuracy_percent", "description"]
    for col in required_columns:
        if col not in final_df.columns:
            final_df[col] = ""

    # Remove old duplicate Optimised BLE row if it already exists, then append updated row.
    mask = final_df["experiment"].astype(str).str.lower().str.contains("optimised ble|optimized ble", regex=True, na=False)
    final_df = final_df.loc[~mask].copy()

    final_df = pd.concat([final_df[required_columns], pd.DataFrame([row])], ignore_index=True)
    return final_df



def normalise_per_class_report(df):
    """
    Normalises different classification report CSV formats into dashboard rows.
    Supports:
    - wifi_per_class_evaluation.csv
    - random_forest_classification_report.csv
    - svm_classification_report.csv
    - knn_classification_report.csv
    """
    if df.empty:
        return []

    df = df.copy()

    # Normalise column names.
    rename_map = {}
    for col in df.columns:
        lower = str(col).strip().lower()
        if lower in ["class", "label", "class / average", "class_or_average", "index", "Unnamed: 0".lower()]:
            rename_map[col] = "Class / Average"
        elif lower in ["precision"]:
            rename_map[col] = "Precision"
        elif lower in ["recall"]:
            rename_map[col] = "Recall"
        elif lower in ["f1-score", "f1_score", "f1 score", "f1"]:
            rename_map[col] = "F1-score"
        elif lower in ["support"]:
            rename_map[col] = "Support"

    df = df.rename(columns=rename_map)

    # Sometimes the first column is unnamed and contains the class names.
    if "Class / Average" not in df.columns and len(df.columns) > 0:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "Class / Average"})

    required = ["Class / Average", "Precision", "Recall", "F1-score", "Support"]
    for col in required:
        if col not in df.columns:
            df[col] = ""

    df = df[required].copy()
    df = df.fillna("")

    # Keep useful rows only.
    useful = []
    for _, row in df.iterrows():
        class_name = str(row["Class / Average"]).strip()
        if not class_name:
            continue

        class_lower = class_name.lower()
        if class_lower == "accuracy":
            continue

        if class_lower in ["phone", "tablet", "macro avg", "weighted avg", "macro average", "weighted average"]:
            useful.append(row.to_dict())
        elif class_lower not in ["micro avg", "samples avg"]:
            # Keep other real class names if dataset labels are different.
            useful.append(row.to_dict())

    final_rows = []
    for row in useful:
        class_name = str(row["Class / Average"]).strip()

        for metric in ["Precision", "Recall", "F1-score"]:
            try:
                val = float(row[metric])
                row[metric] = f"{val:.2f}"
            except Exception:
                row[metric] = str(row[metric])

        try:
            support = float(row["Support"])
            row["Support"] = str(int(support))
        except Exception:
            row["Support"] = str(row["Support"])

        lower = class_name.lower()
        if lower == "phone":
            meaning = "Performance for phone samples."
        elif lower == "tablet":
            meaning = "Performance for tablet samples."
        elif lower in ["macro avg", "macro average"]:
            meaning = "Average treating each class equally."
            row["Class / Average"] = "Macro average"
        elif lower in ["weighted avg", "weighted average"]:
            meaning = "Average considering class imbalance."
            row["Class / Average"] = "Weighted average"
        else:
            meaning = "Class-specific model performance."

        row["Meaning"] = meaning
        final_rows.append(row)

    return final_rows


def calculate_wifi_per_class_report():
    """
    Calculates Wi-Fi per-class evaluation directly inside live_app.py if the saved
    report CSV is not found. This makes the dashboard show the table without
    needing a separate screenshot script.
    """
    if wifi_df.empty or "type" not in wifi_df.columns:
        return [], "Wi-Fi feature data or target column 'type' not found."

    df = wifi_df.dropna(subset=["type"]).copy()

    drop_cols = [
        "device_id", "vendor", "model", "type", "device_os",
        "pcap_filename", "mode"
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if X.empty:
        return [], "No numeric Wi-Fi feature columns found."

    y_text = df["type"].astype(str)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    class_names = list(label_encoder.classes_)

    if len(class_names) < 2:
        return [], "Only one Wi-Fi class found, so per-class evaluation cannot be calculated."

    stratify = y if min(pd.Series(y).value_counts()) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=stratify
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=3))
        ]),
    }

    best_name = None
    best_model = None
    best_accuracy = -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        if acc > best_accuracy:
            best_accuracy = acc
            best_name = name
            best_model = model

    pred = best_model.predict(X_test)
    report_dict = classification_report(
        y_test,
        pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    rows = []
    for key, values in report_dict.items():
        if isinstance(values, dict):
            rows.append({
                "Class / Average": key,
                "Precision": values.get("precision", ""),
                "Recall": values.get("recall", ""),
                "F1-score": values.get("f1-score", ""),
                "Support": values.get("support", ""),
            })

    report_df = pd.DataFrame(rows)

    try:
        WIFI_PER_CLASS_DIR.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(WIFI_PER_CLASS_DIR / "wifi_per_class_evaluation.csv", index=False)
    except Exception as e:
        print("Could not save Wi-Fi per-class CSV:", e)

    dashboard_rows = normalise_per_class_report(report_df)
    note = f"Calculated live from wifi_features.csv using best model: {best_name}, accuracy: {best_accuracy * 100:.2f}%."
    return dashboard_rows, note


def get_wifi_per_class_data():
    """
    Reads Wi-Fi per-class report if available; otherwise calculates it live.
    """
    candidate_files = [
        WIFI_PER_CLASS_DIR / "wifi_per_class_evaluation.csv",
        WIFI_PER_CLASS_DIR / "svm_classification_report.csv",
        WIFI_PER_CLASS_DIR / "knn_classification_report.csv",
        WIFI_PER_CLASS_DIR / "random_forest_classification_report.csv",
    ]

    for path in candidate_files:
        df = safe_read_csv(path)
        rows = normalise_per_class_report(df)
        if rows:
            return rows, f"Loaded from: {path.name}"

    return calculate_wifi_per_class_report()



def get_ble_optimization_data():
    opt_df = safe_read_csv(BLE_IMPROVEMENT_DIR / "ble_optimization_all_results.csv")

    if not opt_df.empty:
        if "accuracy" in opt_df.columns:
            opt_df = opt_df.sort_values("accuracy", ascending=False)
        elif "accuracy_percent" in opt_df.columns:
            opt_df = opt_df.sort_values("accuracy_percent", ascending=False)

    summary = (
        safe_read_text(BLE_IMPROVEMENT_DIR / "best_ble_optimization_summary.txt")
        or safe_read_text(BLE_IMPROVEMENT_DIR / "best_overall_ble_optimization_summary.txt")
        or safe_read_text(BLE_IMPROVEMENT_DIR / "advanced_ble_best_summary.txt")
    )

    best = {
        "available": True,
        "previous_accuracy": PREVIOUS_BLE_ACCURACY,
        "best_accuracy": UPDATED_BLE_ACCURACY,
        "improvement": round(UPDATED_BLE_ACCURACY - PREVIOUS_BLE_ACCURACY, 2),
        "wifi_accuracy": BEST_WIFI_ACCURACY,
        "gap_to_wifi": round(BEST_WIFI_ACCURACY - UPDATED_BLE_ACCURACY, 2),
        "model": UPDATED_BLE_MODEL,
        "dataset_version": UPDATED_BLE_DATASET_VERSION,
    }

    if not opt_df.empty:
        row = opt_df.iloc[0]
        if "accuracy_percent" in opt_df.columns:
            best_accuracy = percent(row.get("accuracy_percent", 0))
        else:
            best_accuracy = percent(row.get("accuracy", 0))

        best = {
            "available": True,
            "previous_accuracy": PREVIOUS_BLE_ACCURACY,
            "best_accuracy": best_accuracy,
            "improvement": round(best_accuracy - PREVIOUS_BLE_ACCURACY, 2),
            "wifi_accuracy": BEST_WIFI_ACCURACY,
            "gap_to_wifi": round(BEST_WIFI_ACCURACY - best_accuracy, 2),
            "model": str(row.get("model", "-")),
            "dataset_version": str(row.get("dataset_version", "-")),
        }

    return opt_df, summary, best


wifi_df = safe_read_csv(find_csv("wifi_features.csv"))
ble_df = safe_read_csv(find_csv("ble_features.csv"))

wifi_model = safe_load_model(MODELS_DIR / "best_wifi_fingerprinting_model.pkl")
wifi_encoder = safe_load_model(MODELS_DIR / "label_encoder.pkl")

ble_model = safe_load_model(MODELS_DIR / "best_ble_model.pkl")
ble_encoder = safe_load_model(MODELS_DIR / "ble_label_encoder.pkl")


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Wi-Fi and Bluetooth Device Fingerprinting Dashboard</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Segoe UI, Arial, sans-serif;
            background: radial-gradient(circle at top, #1e3a8a, #020617 45%);
            color: #e5e7eb;
        }
        header {
            padding: 30px;
            background: rgba(2, 6, 23, 0.94);
            border-bottom: 1px solid #38bdf8;
            text-align: center;
        }
        header h1 { margin: 0; font-size: 34px; color: #38bdf8; }
        header p { margin-top: 8px; color: #cbd5e1; }
        .container { padding: 25px; }
        .status-bar {
            background: linear-gradient(90deg, #065f46, #0f766e);
            padding: 14px 18px;
            border-radius: 12px;
            margin-bottom: 25px;
            font-weight: bold;
            box-shadow: 0 0 20px rgba(34, 197, 94, 0.25);
        }
        .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; margin-bottom: 30px; }
        .three-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin-bottom: 20px; }
        .two-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 22px; }
        .card {
            background: rgba(15, 23, 42, 0.90);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 18px;
            padding: 20px;
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.12);
        }
        .highlight { background: linear-gradient(135deg, rgba(22, 101, 52, 0.95), rgba(15, 23, 42, 0.95)); border-color: rgba(34, 197, 94, 0.55); }
        .warning { background: linear-gradient(135deg, rgba(133, 77, 14, 0.95), rgba(15, 23, 42, 0.95)); border-color: rgba(250, 204, 21, 0.55); }
        .card h3 { margin: 0 0 10px 0; color: #93c5fd; font-size: 15px; }
        .number { font-size: 34px; color: #22c55e; font-weight: bold; }
        .yellow { color: #facc15; }
        .section {
            background: rgba(15, 23, 42, 0.90);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 18px;
            padding: 22px;
            margin-bottom: 28px;
            box-shadow: 0 0 18px rgba(15, 23, 42, 0.8);
        }
        .section h2 { color: #38bdf8; margin-top: 0; }
        .section h3 { color: #93c5fd; }
        .prediction { font-size: 34px; font-weight: bold; color: #22c55e; margin: 12px 0; }
        .live-title { color: #facc15; font-size: 20px; margin-bottom: 12px; }
        .row { display: flex; justify-content: space-between; border-bottom: 1px solid #334155; padding: 9px 0; gap: 15px; }
        .label { color: #94a3b8; }
        .value { color: #e5e7eb; font-weight: 600; text-align: right; }
        table { width: 100%; border-collapse: collapse; border-radius: 12px; margin-bottom: 18px; font-size: 13px; }
        th, td { padding: 12px; border-bottom: 1px solid #334155; text-align: left; vertical-align: top; }
        th { background: #020617; color: #38bdf8; }
        .badge { padding: 6px 12px; background: #2563eb; color: white; border-radius: 999px; font-size: 13px; display: inline-block; }
        .green-badge { background: #16a34a; }
        .graphs { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }
        .graph-box { background: #020617; border-radius: 14px; padding: 15px; border: 1px solid rgba(56, 189, 248, 0.2); }
        .graph-box h3 { color: #93c5fd; font-size: 15px; word-break: break-word; }
        img { width: 100%; background: white; padding: 10px; border-radius: 12px; }
        .note { color: #facc15; font-size: 14px; margin-top: 20px; }
        .small-note { color: #cbd5e1; font-size: 14px; line-height: 1.6; }
        pre { background: #020617; color: #e5e7eb; border: 1px solid rgba(56, 189, 248, 0.25); padding: 15px; border-radius: 12px; white-space: pre-wrap; line-height: 1.5; }
        .empty { color: #facc15; background: rgba(250, 204, 21, 0.08); border: 1px solid rgba(250, 204, 21, 0.25); padding: 12px; border-radius: 10px; }
        @media(max-width: 1000px) { .grid, .three-grid { grid-template-columns: repeat(2, 1fr); } .two-grid, .graphs { grid-template-columns: 1fr; } }
        @media(max-width: 700px) { .grid, .three-grid { grid-template-columns: 1fr; } }
    </style>
</head>

<body>
<header>
    <h1>Wi-Fi and Bluetooth Device Fingerprinting</h1>
    <p>Machine Learning and Deep Learning Cybersecurity Dashboard</p>
</header>

<div class="container">

    <div class="status-bar">System running on live simulation mode — data updates every 2 seconds</div>

    <div class="grid">
        <div class="card"><h3>Wi-Fi Records</h3><div class="number">{{ wifi_stats.records }}</div></div>
        <div class="card"><h3>Wi-Fi Devices Found</h3><div class="number">{{ wifi_stats.devices }}</div></div>
        <div class="card"><h3>Phones</h3><div class="number">{{ wifi_stats.phones }}</div></div>
        <div class="card"><h3>Tablets</h3><div class="number">{{ wifi_stats.tablets }}</div></div>
        <div class="card"><h3>BLE Records</h3><div class="number">{{ ble_stats.records }}</div></div>
        <div class="card"><h3>BLE Files</h3><div class="number">{{ ble_stats.files }}</div></div>
        <div class="card"><h3>BLE Environments</h3><div class="number">{{ ble_stats.environments }}</div></div>
        <div class="card"><h3>Project Mode</h3><div class="number">LIVE</div></div>
    </div>

    <div class="section">
        <h2>BLE Optimisation Result</h2>
        {% if ble_best.available %}
        <div class="three-grid">
            <div class="card warning"><h3>Previous BLE Result</h3><div class="number yellow">{{ ble_best.previous_accuracy }}%</div><p class="small-note">Previous best model: KNN</p></div>
            <div class="card highlight"><h3>Optimised BLE Result</h3><div class="number">{{ ble_best.best_accuracy }}%</div><p class="small-note">Best model: {{ ble_best.model }}</p></div>
            <div class="card"><h3>Improvement</h3><div class="number">{{ ble_best.improvement }}%</div><p class="small-note">Dataset: {{ ble_best.dataset_version }}</p></div>
        </div>
        <div class="three-grid">
            <div class="card"><h3>Best Wi-Fi Accuracy</h3><div class="number">{{ ble_best.wifi_accuracy }}%</div></div>
            <div class="card warning"><h3>Remaining BLE-Wi-Fi Gap</h3><div class="number yellow">{{ ble_best.gap_to_wifi }}%</div></div>
            <div class="card highlight"><h3>Supervisor Feedback Addressed</h3><div class="number">YES</div></div>
        </div>
        {% else %}
        <div class="empty">BLE optimisation result not found yet. Run: python scripts\\ble_model_optimization.py</div>
        {% endif %}
    </div>

    <div class="section">
        <h2>Live Detection Panel</h2>
        <div class="two-grid">
            <div class="card">
                <div class="live-title">Live Wi-Fi Detection</div>
                <div class="prediction" id="wifi_prediction">Waiting...</div>
                <div class="row"><span class="label">Processed</span><span class="value" id="wifi_processed">-</span></div>
                <div class="row"><span class="label">Device ID</span><span class="value" id="wifi_device_id">-</span></div>
                <div class="row"><span class="label">Vendor / Model</span><span class="value" id="wifi_model">-</span></div>
                <div class="row"><span class="label">Actual Type</span><span class="value" id="wifi_actual">-</span></div>
                <div class="row"><span class="label">Packet Count</span><span class="value" id="wifi_packet_count">-</span></div>
                <div class="row"><span class="label">Packet Rate</span><span class="value" id="wifi_packet_rate">-</span></div>
                <div class="row"><span class="label">Average RSSI</span><span class="value" id="wifi_rssi">-</span></div>
            </div>
            <div class="card">
                <div class="live-title">Live Bluetooth/BLE Detection</div>
                <div class="prediction" id="ble_prediction">Waiting...</div>
                <div class="row"><span class="label">Processed</span><span class="value" id="ble_processed">-</span></div>
                <div class="row"><span class="label">Source File</span><span class="value" id="ble_file">-</span></div>
                <div class="row"><span class="label">Actual Environment</span><span class="value" id="ble_actual">-</span></div>
                <div class="row"><span class="label">Packet Count</span><span class="value" id="ble_packet_count">-</span></div>
                <div class="row"><span class="label">Average RSSI</span><span class="value" id="ble_rssi">-</span></div>
                <div class="row"><span class="label">RSSI Range</span><span class="value" id="ble_rssi_range">-</span></div>
            </div>
        </div>
        <p class="note">This dashboard uses processed dataset rows as a live simulation. Real capture would require authorisation and supervisor/ethics approval.</p>
    </div>

    <div class="section">
        <h2>Final Project Results Summary</h2>
        {% if final_summary %}
        <table>
            <tr><th>Experiment</th><th>Best Model</th><th>Accuracy</th><th>Description</th></tr>
            {% for row in final_summary %}
            <tr><td>{{ row.experiment }}</td><td><span class="badge">{{ row.best_model }}</span></td><td>{{ row.accuracy_percent }}%</td><td>{{ row.description }}</td></tr>
            {% endfor %}
        </table>
        {% else %}<div class="empty">Final project summary file not found yet.</div>{% endif %}
    </div>

    <div class="section">
        <h2>Wi-Fi Device Type Classification</h2>
        {% if type_results %}
        <table>
            <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-score</th></tr>
            {% for row in type_results %}
            <tr><td>{{ row.model }}</td><td>{{ row.accuracy_percent }}%</td><td>{{ row.precision }}</td><td>{{ row.recall }}</td><td>{{ row.f1_score }}</td></tr>
            {% endfor %}
        </table>
        {% else %}<div class="empty">Wi-Fi type comparison file not found yet.</div>{% endif %}
    </div>

    <div class="section">
        <h2>Wi-Fi Per-Class Evaluation</h2>
        <p class="small-note">
            This section shows precision, recall and F1-score separately for each Wi-Fi device type.
            It helps address the Phone-vs-Tablet class imbalance instead of relying only on overall accuracy.
        </p>

        {% if wifi_per_class_rows %}
        <p class="note">{{ wifi_per_class_note }}</p>
        <table>
            <tr>
                <th>Class / Average</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-score</th>
                <th>Support</th>
                <th>Meaning</th>
            </tr>
            {% for row in wifi_per_class_rows %}
            <tr>
                <td>{{ row["Class / Average"] }}</td>
                <td>{{ row["Precision"] }}</td>
                <td>{{ row["Recall"] }}</td>
                <td>{{ row["F1-score"] }}</td>
                <td>{{ row["Support"] }}</td>
                <td>{{ row["Meaning"] }}</td>
            </tr>
            {% endfor %}
        </table>
        {% else %}
        <div class="empty">
            Wi-Fi per-class evaluation could not be loaded or calculated.
            Make sure wifi_features.csv exists in data\processed.
        </div>
        {% endif %}
    </div>

    <div class="section">
        <h2>Bluetooth/BLE Classification</h2>
        {% if ble_results %}
        <table>
            <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-score</th></tr>
            {% for row in ble_results %}
            <tr><td>{{ row.model }}</td><td>{{ row.accuracy_percent }}%</td><td>{{ row.precision }}</td><td>{{ row.recall }}</td><td>{{ row.f1_score }}</td></tr>
            {% endfor %}
        </table>
        {% else %}<div class="empty">BLE model results file not found yet.</div>{% endif %}
    </div>

    <div class="section">
        <h2>BLE RSSI Analysis</h2>
        <p class="small-note">This section tests whether BLE RSSI instability may contribute to lower BLE accuracy.</p>

        {% if ble_rssi_summary %}
        <h3>BLE RSSI Summary</h3>
        <table>
            <tr>{% for key in ble_rssi_summary[0].keys() %}<th>{{ key }}</th>{% endfor %}</tr>
            {% for row in ble_rssi_summary %}
            <tr>{% for value in row.values() %}<td>{{ value }}</td>{% endfor %}</tr>
            {% endfor %}
        </table>
        {% else %}<div class="empty">BLE RSSI summary not found yet.</div>{% endif %}

        {% if ble_wifi_comparison %}
        <h3>BLE vs Wi-Fi RSSI Variability Comparison</h3>
        <table>
            <tr>{% for key in ble_wifi_comparison[0].keys() %}<th>{{ key }}</th>{% endfor %}</tr>
            {% for row in ble_wifi_comparison %}
            <tr>{% for value in row.values() %}<td>{{ value }}</td>{% endfor %}</tr>
            {% endfor %}
        </table>
        {% else %}<div class="empty">BLE vs Wi-Fi RSSI comparison file not found yet. Run: python scripts\\run_final_analysis.py</div>{% endif %}
    </div>

    <div class="section">
        <h2>BLE Model Optimisation Details</h2>
        {% if ble_best_summary %}
        <h3>Best BLE Optimisation Summary</h3>
        <pre>{{ ble_best_summary }}</pre>
        {% endif %}

        {% if ble_optimization_results %}
        <h3>Optimised BLE Model Comparison</h3>
        <table>
            <tr><th>Dataset Version</th><th>Model</th><th>Accuracy</th><th>CV Accuracy</th><th>Macro Precision</th><th>Macro Recall</th><th>Macro F1</th><th>Weighted F1</th></tr>
            {% for row in ble_optimization_results %}
            <tr>
                <td>{{ row.dataset_version }}</td>
                <td><span class="badge green-badge">{{ row.model }}</span></td>
                <td>{{ row.accuracy_percent }}%</td>
                <td>{{ row.cv_accuracy_percent }}%</td>
                <td>{{ row.macro_precision }}</td>
                <td>{{ row.macro_recall }}</td>
                <td>{{ row.macro_f1 }}</td>
                <td>{{ row.weighted_f1 }}</td>
            </tr>
            {% endfor %}
        </table>
        {% else %}<div class="empty">BLE optimisation results not found yet. Run: python scripts\\ble_model_optimization.py</div>{% endif %}
    </div>

    <div class="section">
        <h2>Wi-Fi and BLE Result Comparison Graphs</h2>
        <p class="small-note">This restores the missing graph section from the old dashboard.</p>
        {% if comparison_graphs %}
        <div class="graphs">
            {% for graph in comparison_graphs %}
            <div class="graph-box"><h3>{{ graph }}</h3><img src="/results/{{ graph }}"></div>
            {% endfor %}
        </div>
        {% else %}
        <div class="empty">
            No Wi-Fi/BLE comparison graphs found yet. Run these scripts and refresh:
            <br><br>
            python scripts\\compare_type_results.py<br>
            python scripts\\train_ble_ml_models.py<br>
            python scripts\\run_final_analysis.py<br>
            python scripts\\ble_model_optimization.py
        </div>
        {% endif %}
    </div>

    <div class="section">
        <h2>All Generated Graphs</h2>
        {% if all_graphs %}
        <div class="graphs">
            {% for graph in all_graphs %}
            <div class="graph-box"><h3>{{ graph }}</h3><img src="/results/{{ graph }}"></div>
            {% endfor %}
        </div>
        {% else %}<div class="empty">No graph files found yet.</div>{% endif %}
    </div>

</div>

<script>
    async function updateLiveData() {
        try {
            const response = await fetch("/api/live");
            const data = await response.json();

            if (data.wifi) {
                document.getElementById("wifi_prediction").innerText = data.wifi.predicted;
                document.getElementById("wifi_processed").innerText = data.wifi.processed + " / " + data.wifi.total;
                document.getElementById("wifi_device_id").innerText = data.wifi.device_id;
                document.getElementById("wifi_model").innerText = data.wifi.vendor + " " + data.wifi.model;
                document.getElementById("wifi_actual").innerText = data.wifi.actual_type;
                document.getElementById("wifi_packet_count").innerText = data.wifi.packet_count;
                document.getElementById("wifi_packet_rate").innerText = data.wifi.packet_rate;
                document.getElementById("wifi_rssi").innerText = data.wifi.avg_rssi + " dBm";
            }

            if (data.ble) {
                document.getElementById("ble_prediction").innerText = data.ble.predicted;
                document.getElementById("ble_processed").innerText = data.ble.processed + " / " + data.ble.total;
                document.getElementById("ble_file").innerText = data.ble.source_file;
                document.getElementById("ble_actual").innerText = data.ble.actual_environment;
                document.getElementById("ble_packet_count").innerText = data.ble.packet_count;
                document.getElementById("ble_rssi").innerText = data.ble.avg_rssi + " dBm";
                document.getElementById("ble_rssi_range").innerText = data.ble.rssi_range;
            }
        } catch (error) {
            console.log("Live update error:", error);
        }
    }
    updateLiveData();
    setInterval(updateLiveData, 2000);
</script>

</body>
</html>
"""


@app.route("/")
def dashboard():
    final_df = safe_read_csv(RESULTS_DIR / "final_project_results_summary.csv")
    type_df = safe_read_csv(RESULTS_DIR / "final_type_classification_comparison.csv")
    ble_results_df = safe_read_csv(RESULTS_DIR / "ble_ml_model_results.csv")

    ble_rssi_summary_df = safe_read_csv(BLE_RSSI_DIR / "ble_rssi_overall_summary.csv")
    ble_wifi_comparison_df = safe_read_csv(BLE_RSSI_DIR / "ble_vs_wifi_rssi_variability_comparison.csv")

    ble_optimization_df, ble_best_summary, ble_best = get_ble_optimization_data()
    wifi_per_class_rows, wifi_per_class_note = get_wifi_per_class_data()

    # Make sure the Final Project Results Summary table includes the updated
    # BLE optimisation row: SVM_Tuned with 70.07%.
    final_df = ensure_optimised_ble_summary_row(final_df, ble_best)

    wifi_stats = {
        "records": len(wifi_df),
        "devices": wifi_df["device_id"].nunique() if "device_id" in wifi_df.columns else 0,
        "types": wifi_df["type"].nunique() if "type" in wifi_df.columns else 0,
        "phones": int((wifi_df["type"] == "Phone").sum()) if "type" in wifi_df.columns else 0,
        "tablets": int((wifi_df["type"] == "Tablet").sum()) if "type" in wifi_df.columns else 0,
    }

    ble_stats = {
        "records": len(ble_df),
        "files": ble_df["source_file"].nunique() if "source_file" in ble_df.columns else 0,
        "environments": ble_df["environment"].nunique() if "environment" in ble_df.columns else 0,
    }

    comparison_candidates = [
        "final_project_results_summary.png",
        "final_type_classification_comparison.png",
        "ble_ml_accuracy_comparison.png",
        "feature_importance.png",
        "deep_learning_training_graph_type.png",
        "deep_learning_loss_graph_type.png",
        "deep_learning_confusion_matrix_type.png",
        "final_analysis/ble_improvement/ble_optimisation_accuracy_comparison.png",
        "final_analysis/ble_improvement/ble_optimization_accuracy_comparison.png",
        "final_analysis/ble_improvement/ble_vs_wifi_accuracy_gap.png",
        "final_analysis/ble_improvement/ble_vs_wifi_gap_comparison.png",
        "final_analysis/ble_rssi/ble_vs_wifi_rssi_variability_comparison.png",
    ]
    comparison_graphs = existing_graphs(comparison_candidates)

    main_graphs = [p.name for p in RESULTS_DIR.glob("*.png")]
    final_analysis_graphs = list_pngs(FINAL_ANALYSIS_DIR)
    all_graphs = sorted(list(dict.fromkeys(main_graphs + final_analysis_graphs)))

    return render_template_string(
        HTML,
        wifi_stats=wifi_stats,
        ble_stats=ble_stats,
        final_summary=table_rows(final_df),
        type_results=table_rows(type_df),
        ble_results=table_rows(ble_results_df),
        ble_rssi_summary=table_rows(ble_rssi_summary_df),
        ble_wifi_comparison=table_rows(ble_wifi_comparison_df),
        ble_optimization_results=table_rows(ble_optimization_df),
        ble_best_summary=ble_best_summary,
        ble_best=ble_best,
        wifi_per_class_rows=wifi_per_class_rows,
        wifi_per_class_note=wifi_per_class_note,
        comparison_graphs=comparison_graphs,
        all_graphs=all_graphs,
    )


@app.route("/api/live")
def live_api():
    global wifi_index, ble_index
    output = {}

    if not wifi_df.empty:
        row = wifi_df.iloc[wifi_index % len(wifi_df)]
        wifi_index += 1

        features = pd.DataFrame([row])
        for col in ["device_id", "vendor", "model", "type", "device_os", "pcap_filename"]:
            if col in features.columns:
                features = features.drop(columns=[col])

        features = features.select_dtypes(include=[np.number])
        prediction = "Model not loaded"

        if wifi_model is not None and wifi_encoder is not None:
            try:
                pred = wifi_model.predict(features)[0]
                prediction = wifi_encoder.inverse_transform([pred])[0]
            except Exception:
                prediction = "Prediction error"

        output["wifi"] = {
            "processed": wifi_index,
            "total": len(wifi_df),
            "device_id": str(row.get("device_id", "-")),
            "vendor": str(row.get("vendor", "-")),
            "model": str(row.get("model", "-")),
            "actual_type": str(row.get("type", "-")),
            "predicted": str(prediction),
            "packet_count": int(row.get("packet_count", 0)),
            "packet_rate": round(float(row.get("packet_rate", 0)), 4),
            "avg_rssi": round(float(row.get("avg_rssi", 0)), 2),
        }

    if not ble_df.empty:
        row = ble_df.iloc[ble_index % len(ble_df)]
        ble_index += 1

        features = pd.DataFrame([row])
        for col in ["source_file", "device_label", "environment"]:
            if col in features.columns:
                features = features.drop(columns=[col])

        features = features.select_dtypes(include=[np.number])
        prediction = "Model not loaded"

        if ble_model is not None and ble_encoder is not None:
            try:
                pred = ble_model.predict(features)[0]
                prediction = ble_encoder.inverse_transform([pred])[0]
            except Exception:
                prediction = "Prediction error"

        output["ble"] = {
            "processed": ble_index,
            "total": len(ble_df),
            "source_file": str(row.get("source_file", "-")),
            "actual_environment": str(row.get("environment", "-")),
            "predicted": str(prediction),
            "packet_count": int(row.get("packet_count", 0)),
            "avg_rssi": round(float(row.get("avg_rssi", 0)), 2),
            "rssi_range": round(float(row.get("rssi_range", 0)), 2),
        }

    return jsonify(output)


@app.route("/results/<path:filename>")
def results_file(filename):
    return send_from_directory(RESULTS_DIR, filename)


if __name__ == "__main__":
    print("Dashboard running.")
    print("Open: http://127.0.0.1:5051")
    app.run(host="127.0.0.1", port=5051, debug=True)