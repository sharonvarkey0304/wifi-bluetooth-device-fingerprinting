import pandas as pd
import numpy as np
from pathlib import Path
from scapy.all import PcapReader, Dot11, RadioTap


# ==============================
# 1. Project paths
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_FILE = PROJECT_ROOT / "data" / "processed" / "device_pcap_index.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "wifi_features.csv"

print("Reading index file from:")
print(INDEX_FILE)

if not INDEX_FILE.exists():
    raise FileNotFoundError("device_pcap_index.csv not found. Please create it first.")


# ==============================
# 2. Load PCAP index
# ==============================

index_df = pd.read_csv(INDEX_FILE)

print("Total PCAP files in index:", len(index_df))
print(index_df.head())


# ==============================
# 3. Function to extract features
# ==============================

def extract_features_from_pcap(pcap_path):
    """
    Reads one PCAP file and extracts useful Wi-Fi behaviour features.
    """

    packet_count = 0
    probe_request_count = 0
    management_frame_count = 0

    packet_sizes = []
    timestamps = []
    rssi_values = []
    source_macs = set()

    try:
        with PcapReader(str(pcap_path)) as packets:
            for pkt in packets:
                packet_count += 1

                # Packet size
                packet_sizes.append(len(pkt))

                # Timestamp
                timestamps.append(float(pkt.time))

                # Wi-Fi layer
                if pkt.haslayer(Dot11):
                    dot11 = pkt[Dot11]

                    # Management frame = type 0
                    if dot11.type == 0:
                        management_frame_count += 1

                    # Probe request = type 0, subtype 4
                    if dot11.type == 0 and dot11.subtype == 4:
                        probe_request_count += 1

                    # Source MAC address
                    if dot11.addr2:
                        source_macs.add(dot11.addr2)

                # RSSI from RadioTap if available
                if pkt.haslayer(RadioTap):
                    if hasattr(pkt, "dBm_AntSignal"):
                        try:
                            rssi_values.append(int(pkt.dBm_AntSignal))
                        except Exception:
                            pass

    except Exception as e:
        print(f"Error reading {pcap_path}: {e}")
        return None

    # If no packets found
    if packet_count == 0:
        return None

    # Time-based features
    timestamps = sorted(timestamps)

    if len(timestamps) > 1:
        intervals = np.diff(timestamps)
        duration = timestamps[-1] - timestamps[0]
    else:
        intervals = np.array([0])
        duration = 0

    if duration > 0:
        packet_rate = packet_count / duration
    else:
        packet_rate = 0

    # RSSI handling
    if len(rssi_values) > 0:
        avg_rssi = np.mean(rssi_values)
        std_rssi = np.std(rssi_values)
        min_rssi = np.min(rssi_values)
        max_rssi = np.max(rssi_values)
    else:
        avg_rssi = np.nan
        std_rssi = np.nan
        min_rssi = np.nan
        max_rssi = np.nan

    features = {
        "packet_count": packet_count,
        "probe_request_count": probe_request_count,
        "management_frame_count": management_frame_count,
        "avg_packet_size": np.mean(packet_sizes),
        "std_packet_size": np.std(packet_sizes),
        "min_packet_size": np.min(packet_sizes),
        "max_packet_size": np.max(packet_sizes),
        "duration_seconds": duration,
        "avg_packet_interval": np.mean(intervals),
        "std_packet_interval": np.std(intervals),
        "packet_rate": packet_rate,
        "unique_source_macs": len(source_macs),
        "avg_rssi": avg_rssi,
        "std_rssi": std_rssi,
        "min_rssi": min_rssi,
        "max_rssi": max_rssi
    }

    return features


# ==============================
# 4. Process all PCAP files
# ==============================

all_rows = []

for i, row in index_df.iterrows():
    pcap_path = Path(row["pcap_path"])

    # If path in CSV is relative, make it relative to project root
    if not pcap_path.is_absolute():
        pcap_path = PROJECT_ROOT / pcap_path

    print(f"Processing {i + 1}/{len(index_df)}: {pcap_path.name}")

    if not pcap_path.exists():
        print("File not found:", pcap_path)
        continue

    features = extract_features_from_pcap(pcap_path)

    if features is None:
        continue

    output_row = {
        "device_id": row.get("device_id"),
        "vendor": row.get("vendor"),
        "model": row.get("model"),
        "type": row.get("type"),
        "device_os": row.get("device_os"),
        "pcap_filename": row.get("pcap_filename"),
        "mode": row.get("mode"),
        "channel": row.get("channel")
    }

    output_row.update(features)
    all_rows.append(output_row)


# ==============================
# 5. Save features CSV
# ==============================

features_df = pd.DataFrame(all_rows)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
features_df.to_csv(OUTPUT_FILE, index=False)

print("\nFeature extraction completed!")
print("Output saved to:", OUTPUT_FILE)
print("Total feature rows:", len(features_df))
print(features_df.head())