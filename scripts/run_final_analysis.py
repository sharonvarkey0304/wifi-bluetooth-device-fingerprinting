
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run(command):
    print("\nRunning:")
    print(" ".join(str(x) for x in command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main():
    wifi_script = SCRIPTS_DIR / "wifi_per_class_evaluation.py"
    ble_script = SCRIPTS_DIR / "ble_rssi_analysis.py"

    if not wifi_script.exists():
        print(f"Missing file: {wifi_script}")
        return

    if not ble_script.exists():
        print(f"Missing file: {ble_script}")
        return

    run([sys.executable, str(wifi_script), "--target", "type"])
    run([sys.executable, str(ble_script)])


if __name__ == "__main__":
    main()