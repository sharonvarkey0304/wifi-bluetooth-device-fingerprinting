import pandas as pd
from pathlib import Path
import re

# 1. Set dataset folder path
BASE_DIR = Path("data/raw/A dataset of labelled device Wi-Fi probe requests")

INFO_FILE = BASE_DIR / "Individual devices - info.xlsx"
PCAP_DIR = BASE_DIR / "Individual devices"
OUTPUT_FILE = Path("data/processed/device_pcap_index.csv")

# 2. Create output folder if it does not exist
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# 3. Read device information Excel file
device_info = pd.read_excel(INFO_FILE)

# 4. Rename columns
device_info = device_info.rename(columns={
    "Device number \n(unique identifier)": "device_id",
    "Device OS": "device_os",
    "Device OS version": "device_os_version",
    "Device vendor": "vendor",
    "Device model": "model",
    "Does it perform MAC randomisation?": "mac_randomisation",
    "Type of device": "type"
})

# 5. Clean text values
for col in device_info.columns:
    if device_info[col].dtype == "object":
        device_info[col] = device_info[col].astype(str).str.strip()

# 6. Find all PCAP files
pcap_files = list(PCAP_DIR.glob("*/*.pcap"))

rows = []

# 7. Read every PCAP file path and extract useful information from filename
for pcap in pcap_files:
    device_id = pcap.parent.name
    filename = pcap.name

    row = {
        "device_id": device_id,
        "pcap_filename": filename,
        "pcap_path": str(pcap)
    }

    # Optional: extract mode and channel from filename
    # Example filename:
    # A-ts-2024-Feb-09-h17-m53-s21-modePS-ch-1-th-40.pcap
    match = re.search(
        r"mode(?P<mode>[A-Z]+)-ch-(?P<channel>\d+)-th-(?P<threshold>\d+)",
        filename
    )

    if match:
        row["mode"] = match.group("mode")
        row["channel"] = match.group("channel")
        row["threshold"] = match.group("threshold")

    rows.append(row)

# 8. Convert PCAP list into DataFrame
pcap_index = pd.DataFrame(rows)

# 9. Merge PCAP files with device information
final_index = pcap_index.merge(device_info, on="device_id", how="left")

# 10. Reorder columns
columns_order = [
    "device_id",
    "vendor",
    "model",
    "type",
    "device_os",
    "device_os_version",
    "mac_randomisation",
    "pcap_filename",
    "pcap_path",
    "mode",
    "channel",
    "threshold"
]

final_index = final_index[[col for col in columns_order if col in final_index.columns]]

# 11. Save final CSV
final_index.to_csv(OUTPUT_FILE, index=False)

print("CSV created successfully!")
print("Output file:", OUTPUT_FILE)
print("Total PCAP files:", len(final_index))
print("Total devices:", final_index["device_id"].nunique())
print(final_index.head())