import argparse
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "final_analysis" / "ble_rssi"

GROUP_COLUMNS = [
    "environment",
    "condition",
    "scenario",
    "label",
    "class",
    "target",
    "activity",
    "location",
    "device_type",
    "type",
]


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


def resolve_csv(user_path: str | None, default_name: str) -> Path | None:
    if user_path:
        path = Path(user_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path if path.exists() else None

    return find_file(default_name)


def find_rssi_columns(df: pd.DataFrame):
    return [
        col for col in df.columns
        if "rssi" in col.lower() and pd.api.types.is_numeric_dtype(df[col])
    ]


def summarise_rssi(df: pd.DataFrame, rssi_cols):
    rows = []

    for col in rssi_cols:
        values = df[col].dropna()
        if values.empty:
            continue

        rows.append({
            "rssi_column": col,
            "count": int(values.count()),
            "mean": values.mean(),
            "standard_deviation": values.std(),
            "min": values.min(),
            "q25": values.quantile(0.25),
            "median": values.median(),
            "q75": values.quantile(0.75),
            "max": values.max(),
            "range": values.max() - values.min(),
        })

    return pd.DataFrame(rows)


def save_histogram(df: pd.DataFrame, column: str, output_path: Path, title: str):
    values = df[column].dropna()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(values, bins=30)
    ax.set_title(title)
    ax.set_xlabel(f"{column} (dBm)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def save_boxplot_by_group(df: pd.DataFrame, rssi_col: str, group_col: str, output_path: Path):
    plot_df = df[[rssi_col, group_col]].dropna()

    groups = []
    labels = []

    for label, group in plot_df.groupby(group_col):
        values = group[rssi_col].dropna().values
        if len(values) > 0:
            labels.append(str(label))
            groups.append(values)

    if len(groups) < 2:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(groups, labels=labels, vert=True)
    ax.set_title(f"{rssi_col} by {group_col}")
    ax.set_xlabel(group_col)
    ax.set_ylabel(f"{rssi_col} (dBm)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    return True


def grouped_rssi_summary(df: pd.DataFrame, rssi_col: str, group_col: str):
    return (
        df.groupby(group_col)[rssi_col]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"std": "standard_deviation"})
    )


def compare_ble_wifi_rssi(ble_df: pd.DataFrame, wifi_df: pd.DataFrame):
    rows = []

    for dataset_name, df in [("BLE", ble_df), ("WiFi", wifi_df)]:
        rssi_cols = find_rssi_columns(df)

        for col in rssi_cols:
            values = df[col].dropna()
            if values.empty:
                continue

            rows.append({
                "dataset": dataset_name,
                "rssi_column": col,
                "count": int(values.count()),
                "mean": values.mean(),
                "standard_deviation": values.std(),
                "min": values.min(),
                "max": values.max(),
                "range": values.max() - values.min(),
            })

    return pd.DataFrame(rows)


def write_dissertation_text(output_dir: Path):
    text = """BLE RSSI Analysis - Dissertation Text

To investigate the lower BLE classification performance, the RSSI-based features were analysed instead of assuming that signal instability was the cause. The analysis calculated the mean, standard deviation, minimum, maximum and range of RSSI values from the BLE feature dataset. The RSSI distribution was also visualised using histogram and boxplot graphs.

A higher RSSI standard deviation or wider RSSI range indicates that the received signal strength fluctuates considerably across captures. This may reduce the consistency of RSSI-based features and can contribute to lower BLE classification accuracy compared with the Wi-Fi models. Where grouped labels such as LOS and NLOS are available, the grouped RSSI statistics can be used to show whether specific environments produce more unstable RSSI values.
"""
    (output_dir / "dissertation_text_ble_rssi_analysis.txt").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ble-csv", default=None, help="Path to ble_features.csv")
    parser.add_argument("--wifi-csv", default=None, help="Optional path to wifi_features.csv")
    args = parser.parse_args()

    ble_path = resolve_csv(args.ble_csv, "ble_features.csv")

    if ble_path is None:
        print("ERROR: Could not find ble_features.csv.")
        print("Try one of these:")
        print("python scripts\\ble_rssi_analysis.py --ble-csv ble_features.csv")
        print("python scripts\\ble_rssi_analysis.py --ble-csv results\\ble_features.csv")
        sys.exit(1)

    print(f"Using BLE feature file: {ble_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ble_df = pd.read_csv(ble_path)
    rssi_cols = find_rssi_columns(ble_df)

    if not rssi_cols:
        print("ERROR: No numeric RSSI columns found in BLE file.")
        print("Expected columns like avg_rssi, min_rssi, max_rssi or rssi_range.")
        print("Available columns:")
        print(list(ble_df.columns))
        sys.exit(1)

    print("\nRSSI columns found:")
    print(rssi_cols)

    overall = summarise_rssi(ble_df, rssi_cols)
    overall.to_csv(OUTPUT_DIR / "ble_rssi_overall_summary.csv", index=False)

    print("\nBLE RSSI summary:")
    print(overall)

    for col in rssi_cols:
        save_histogram(
            ble_df,
            col,
            OUTPUT_DIR / f"{col}_histogram.png",
            f"BLE {col} Distribution",
        )

    available_group_cols = [
        col for col in GROUP_COLUMNS
        if col in ble_df.columns and ble_df[col].nunique(dropna=True) >= 2
    ]

    for group_col in available_group_cols:
        for rssi_col in rssi_cols:
            grouped = grouped_rssi_summary(ble_df, rssi_col, group_col)
            grouped.to_csv(
                OUTPUT_DIR / f"{rssi_col}_grouped_by_{group_col}.csv",
                index=False,
            )

            save_boxplot_by_group(
                ble_df,
                rssi_col,
                group_col,
                OUTPUT_DIR / f"{rssi_col}_boxplot_by_{group_col}.png",
            )

    wifi_path = resolve_csv(args.wifi_csv, "wifi_features.csv")

    if wifi_path:
        print(f"\nUsing Wi-Fi feature file for RSSI comparison: {wifi_path}")
        wifi_df = pd.read_csv(wifi_path)
        comparison = compare_ble_wifi_rssi(ble_df, wifi_df)
        comparison.to_csv(OUTPUT_DIR / "ble_vs_wifi_rssi_variability_comparison.csv", index=False)

        print("\nBLE vs Wi-Fi RSSI comparison:")
        print(comparison)
    else:
        print("\nWi-Fi feature file not found or not provided. BLE-only RSSI analysis was completed.")

    write_dissertation_text(OUTPUT_DIR)

    print("\nDONE. BLE RSSI analysis saved here:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()