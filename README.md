# Wi-Fi and Bluetooth Device Fingerprinting Using Machine Learning and Deep Learning

## Project Overview

This project investigates a behaviour-based approach to wireless device fingerprinting using machine learning and deep learning techniques. It explores whether communication characteristics from Wi-Fi and Bluetooth Low Energy (BLE) datasets can be used to classify devices or communication conditions without relying exclusively on static identifiers such as MAC addresses.

The project was developed as part of an MSc Cyber Security and Emerging Threats final project at Middlesex University.

## Research Aim

To design and evaluate a behaviour-based wireless fingerprinting framework using Python, feature extraction, machine learning, deep learning and experimental performance evaluation.

## Research Objectives

- Analyse labelled Wi-Fi probe-request data and BLE signal data.
- Extract useful statistical and behavioural features from wireless datasets.
- Prepare datasets for machine learning experiments.
- Train and compare traditional machine learning and deep learning models.
- Evaluate model performance using accuracy, precision, recall and F1-score.
- Investigate the contribution of Received Signal Strength Indicator (RSSI) features to BLE classification.
- Present experimental findings through a Python-based dashboard.
- Discuss limitations, privacy considerations and potential cybersecurity applications.

## Technologies Used

- **Programming language:** Python
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn
- **Deep learning:** TensorFlow/Keras
- **Visualisation:** Matplotlib
- **Dashboard:** Flask
- **Wireless analysis:** Scapy, Wireshark
- **Development environment:** PyCharm

## System Overview

The project follows an experimental machine learning workflow:

1. Dataset preparation
2. Data cleaning and preprocessing
3. Feature extraction
4. Feature selection and preparation
5. Model training
6. Model comparison
7. Performance evaluation
8. Visualisation and reporting

### Wi-Fi Device Fingerprinting

The Wi-Fi experiments use labelled probe-request data to investigate device identification and device-type classification.

Features include:

- Packet and probe-request counts
- Management-frame counts
- Packet-size statistics
- Communication intervals and packet rates
- Unique source MAC address counts
- Available RSSI statistics
- Device and device-type labels for supervised learning

### Bluetooth Low Energy Analysis

The BLE experiments investigate classification using signal-strength and communication-related features.

Features include:

- Average RSSI
- Minimum and maximum RSSI
- RSSI standard deviation and range
- Packet counts
- Communication duration
- Packet rates and timing intervals

The BLE experiments include Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) records.

## Machine Learning Models

The project evaluates the following approaches:

- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbours (KNN)
- Neural Network
- Optimised SVM for BLE classification

Models are evaluated using accuracy, precision, recall and F1-score where applicable.

## Experimental Results

### Wi-Fi Device-Type Classification

| Model | Accuracy |
|---|---:|
| Support Vector Machine (SVM) | 95.12% |
| K-Nearest Neighbours (KNN) | 95.12% |
| Random Forest | 90.24% |
| Neural Network | 90.24% |

The best observed accuracy for Wi-Fi device-type classification was 95.12%, achieved by SVM and KNN in the current experiments.

### Wi-Fi Device-ID Classification

| Model | Accuracy |
|---|---:|
| Random Forest | 90.24% |
| Support Vector Machine (SVM) | 63.41% |
| K-Nearest Neighbours (KNN) | 60.98% |

Random Forest achieved the highest observed accuracy in the device-ID classification experiment.

### Bluetooth Low Energy Classification

| Model | Accuracy |
|---|---:|
| Optimised SVM | 70.07% |
| K-Nearest Neighbours (KNN) | 68.86% |
| Random Forest | 62.72% |
| Support Vector Machine (SVM) | 60.96% |

The optimised SVM achieved 70.07% accuracy, an improvement of 1.21 percentage points over the initial KNN result.

The experiments without RSSI features achieved approximately 61–63% accuracy, suggesting that RSSI contributed useful information to BLE classification.

**Important:** These results are specific to the datasets, labels, features and experimental configurations used. Wi-Fi device-type classification, Wi-Fi device-ID classification and BLE LOS/NLOS classification are different tasks, so their accuracies should not be treated as a direct, controlled comparison. Further testing on independent data is needed to assess generalisation.

## Dashboard

A Flask-based dashboard was developed to present processed experimental results and model-performance information.

The dashboard is a dataset-driven demonstration. It does not currently represent a verified real-time wireless monitoring or live device identification system.

## Project Structure

The following is the intended organisation of the project. Actual filenames and locations may differ depending on the current implementation.

```text
wifi_ble_project/
├── data/
│   ├── raw/                 # Local datasets; not committed
│   └── processed/           # Processed features and permitted results
├── web_app/
│   └── live_app.py          # Flask dashboard
├── create_pcap_index.py
├── extract_wifi_features.py
├── extract_ble_features.py
├── train_ml_model.py
├── train_deep_learning.py
├── wifi_per_class_evaluation.py
├── ble_rssi_analysis.py
├── ble_model_optimization.py
├── compare_type_results.py
├── create_final_results_summary.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

Replace `YOUR-USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR-USERNAME/wifi-bluetooth-device-fingerprinting.git
cd wifi-bluetooth-device-fingerprinting
```

### 2. Create a virtual environment

For Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If a `requirements.txt` file is included:

```powershell
python -m pip install -r requirements.txt
```

If the dependencies file has not yet been created, install the packages needed for the relevant scripts:

```powershell
python -m pip install pandas numpy scikit-learn matplotlib tensorflow flask scapy
```

Package versions may need to be adjusted to match the Python version and the environment used to develop the project.

## Running the Project

Run commands from the project root, using the actual script locations in your repository.

Examples:

```powershell
python extract_wifi_features.py
python train_ml_model.py
python train_deep_learning.py
```

For BLE experiments:

```powershell
python extract_ble_features.py
python ble_rssi_analysis.py
python ble_model_optimization.py
```

To start the Flask dashboard, if the application entry point is located at `web_app/live_app.py`:

```powershell
python web_app/live_app.py
```

Open the local URL printed by Flask in your browser. Dataset files and any additional configuration required by each script must be available locally.

## Dataset and Privacy

Dataset files are not included in this repository by default. Obtain the datasets from their original sources and comply with their licensing, attribution and redistribution requirements.

Raw packet captures and files containing potentially sensitive identifiers should not be uploaded publicly without appropriate review and permission.

All testing should use authorised datasets and devices. This project is intended for academic research and cybersecurity learning, not unauthorised monitoring or tracking.

## Limitations

- Model performance depends on the size, quality and balance of the available datasets.
- RSSI can vary with distance, obstacles, device orientation, interference and multipath effects.
- Results from one dataset may not generalise to unseen devices or environments.
- The Wi-Fi and BLE experiments use different classification targets and should be interpreted separately.
- The dashboard presents processed experimental results rather than verified live wireless captures.
- Further validation on independent datasets and controlled experiments would strengthen the findings.

## Future Improvements

- Evaluate models using independent test devices and datasets.
- Improve class balance and per-class evaluation.
- Investigate feature selection and hyperparameter optimisation.
- Test the robustness of fingerprints under different environmental conditions.
- Improve dashboard reporting and reproducibility.
- Explore privacy-preserving approaches to wireless device classification.

## References

- Gao, K., Corbett, C. and Beyah, R. (2010). A passive approach to wireless device fingerprinting. *2010 IEEE/IFIP International Conference on Dependable Systems & Networks*. https://doi.org/10.1109/DSN.2010.5544294
- Pintor, L. and Atzori, L. (2022). A dataset of labelled device Wi-Fi probe requests for MAC address de-randomization. *Computer Networks*, 205. https://doi.org/10.1016/j.comnet.2022.108783
- Sanchez, P.M.S. et al. (2021). A Survey on Device Behavior Fingerprinting: Data Sources, Techniques, Application Scenarios, and Datasets. *IEEE Communications Surveys & Tutorials*, 23(2), pp. 1048–1077. https://doi.org/10.1109/COMST.2021.3064259
- Celosia, G. and Cunche, M. (2019). Fingerprinting Bluetooth-Low-Energy Devices Based on the Generic Attribute Profile. *Proceedings of the 2nd International ACM Workshop on Security and Privacy for the Internet-of-Things*, pp. 24–31. https://doi.org/10.1145/3338507.3358617

## Author

**Sharon Varkey**  
MSc Cyber Security and Emerging Threats  
Middlesex University, United Kingdom

This repository documents an academic project on machine learning, deep learning and wireless device fingerprinting.