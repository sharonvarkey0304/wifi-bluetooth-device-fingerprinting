import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BLE_DIR = PROJECT_ROOT / "data" / "raw" / "BLE_RSSI_Data"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "ble_features.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Looking for BLE dataset at:")
print(BLE_DIR)

if not BLE_DIR.exists():
    raise FileNotFoundError("BLE_RSSI_Data folder not found inside data/raw/")

csv_files = list(BLE_DIR.rglob("*.csv"))

print("Total BLE CSV files found:", len(csv_files))

rows = []

for csv_file in csv_files:
    print("Processing:", csv_file.name)

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print("Error reading file:", csv_file, e)
        continue

    if df.empty:
        continue

    file_path_text = str(csv_file)

    environment = "LOS" if "LOS" in file_path_text and "NLOS" not in file_path_text else "NLOS"

    device_label = csv_file.stem

    row = {
        "source_file": csv_file.name,
        "device_label": device_label,
        "environment": environment,
        "packet_count": len(df)
    }

    if "RSSI_dBm" in df.columns:
        rssi = pd.to_numeric(df["RSSI_dBm"], errors="coerce").dropna()

        if len(rssi) > 0:
            row["avg_rssi"] = rssi.mean()
            row["std_rssi"] = rssi.std()
            row["min_rssi"] = rssi.min()
            row["max_rssi"] = rssi.max()
            row["rssi_range"] = rssi.max() - rssi.min()
        else:
            row["avg_rssi"] = np.nan
            row["std_rssi"] = np.nan
            row["min_rssi"] = np.nan
            row["max_rssi"] = np.nan
            row["rssi_range"] = np.nan

    if "Timestamp_ms" in df.columns:
        timestamps = pd.to_numeric(df["Timestamp_ms"], errors="coerce").dropna().sort_values()

        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            duration = timestamps.iloc[-1] - timestamps.iloc[0]

            row["duration_ms"] = duration
            row["avg_interval_ms"] = intervals.mean()
            row["std_interval_ms"] = intervals.std()
            row["packet_rate"] = len(df) / duration if duration > 0 else 0
        else:
            row["duration_ms"] = 0
            row["avg_interval_ms"] = 0
            row["std_interval_ms"] = 0
            row["packet_rate"] = 0

    if "MAC_Type" in df.columns:
        row["mac_type_count"] = df["MAC_Type"].nunique()

    if "RPI" in df.columns:
        row["rpi_count"] = df["RPI"].nunique()

    rows.append(row)

ble_features = pd.DataFrame(rows)

ble_features.to_csv(OUTPUT_FILE, index=False)

print("\nBLE feature extraction completed.")
print("Saved to:", OUTPUT_FILE)
print("Rows:", len(ble_features))
print(ble_features.head())