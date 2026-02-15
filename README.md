# 🛡️ Hybrid Intrusion Detection System (NSL-KDD)

## 📌 Overview

This project implements a **two-stage Hybrid Intrusion Detection System (IDS)** using Machine Learning on the **NSL-KDD dataset**.

The system combines:

- **Unsupervised Anomaly Detection** (Isolation Forest, One-Class SVM)
- **Supervised Attack Classification** (Random Forest)

The goal is to simulate how organizations can automatically detect and categorize malicious network activity using a layered ML approach.

---

## 🎯 Problem Statement

Modern systems generate massive network logs daily. Manually analyzing these logs is inefficient and error-prone.

Traditional signature-based security systems detect only known attacks.  
This project explores whether a **hybrid ML approach** can:

1. Detect anomalous network behavior.
2. Classify detected attacks into meaningful categories.
3. Reduce false alarms while maintaining detection capability.

---

## 🏗️ System Architecture

```
Network Log Record
        ↓
Stage 1: Anomaly Detection
    - Isolation Forest
    - One-Class SVM
        ↓
If Normal → Output: Normal
If Anomaly → Stage 2
        ↓
Stage 2: Multi-Class Classification
    - Random Forest
        ↓
Output:
    - DoS
    - Probe
    - R2L
    - Normal
```
### Note on U2R Category

The U2R (User-to-Root) attack category was excluded from the final hybrid system due to extremely low representation in the dataset, which caused unstable learning and unreliable evaluation metrics. Removing U2R improved model stability and interpretability.

---

## 📊 Dataset

**Dataset Used:** NSL-KDD  
- KDDTrain+.txt (Training)
- KDDTest+.txt (Testing)

Features include:
- Protocol type
- Service
- Flag
- Traffic statistics
- Error rates
- Connection counts

---

## 🧠 Models Implemented

### 🔹 1. Isolation Forest
- Trained on normal samples
- Used for anomaly detection
- Contamination tuned experimentally

### 🔹 2. One-Class SVM
- Compared with Isolation Forest
- Evaluated anomaly detection performance

### 🔹 3. Random Forest Classifier
- Multi-class attack classification
- Categories:
  - DoS
  - Probe
  - R2L

---

## ⚙️ Preprocessing Pipeline

- Dropped irrelevant columns (`difficulty`)
- Binary label creation (normal vs attack)
- Attack category mapping
- One-hot encoding (`protocol_type`, `service`, `flag`)
- Feature alignment between train and test
- Standard scaling

---

## 📈 Evaluation Strategy

- Models trained on **KDDTrain+**
- Evaluated on **KDDTest+**
- Metrics used:
  - Precision
  - Recall
  - F1-score
  - Accuracy
  - Macro & Weighted averages

---

## 📊 Key Results

### 🔹 Standalone Multi-Class Classification (Attack Samples Only)
- Accuracy on KDDTest+: ~93%
- Macro F1-Score: ~0.87

The supervised Random Forest classifier performs strongly when attack samples are known in advance.

---

### 🔹 Hybrid System (Anomaly Detection + Classification)
- Hybrid Accuracy on KDDTest+: ~60%
- Hybrid Macro F1-Score: ~0.37

Performance decreases in the hybrid setting because the anomaly detection stage (unsupervised) introduces detection errors before classification. This highlights the practical challenges of combining unsupervised anomaly detection with supervised attack categorization.


---

## 🔍 Observations

- Unsupervised anomaly detection struggles with subtle attacks (e.g., R2L).
- Hybrid performance is heavily dependent on Stage 1 accuracy.
- Increasing contamination increases false positives.
- Class imbalance significantly affects minority class performance.
- Supervised classification performs significantly better than unsupervised detection in structured datasets.

---

## 📂 Project Structure

```
nexusguard-ml/
│
├── data/
│   ├── KDDTrain+.txt
│   ├── KDDTest+.txt
│
├── models/
│   ├── iso_detector.pkl
│   ├── rf_classifier.pkl
│   ├── scaler.pkl
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_anomaly_models.ipynb
│   ├── 05_classification_models.ipynb
│   ├── 06_hybrid_pipeline.ipynb
│   ├── test_train.ipynb
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

---

## 🚀 How to Run

1. Clone the repository:

```
git clone https://github.com/saurav0311/hybrid-ids-nsl-kdd.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Open notebooks in order:

```
01_data_loading.ipynb
02_eda.ipynb
03_preprocessing.ipynb
04_anomaly_models.ipynb
05_classification_models.ipynb
06_hybrid_pipeline.ipynb
```

---

## 🎓 Key Learnings

- Trade-offs between supervised and unsupervised learning.
- Importance of contamination tuning in anomaly detection.
- Impact of class imbalance on ML models.
- Designing modular ML pipelines.
- Realistic model evaluation using separate test sets.

---

## 💡 Future Improvements

- Replace anomaly detection with supervised binary classifier.
- Implement real-time log streaming using FastAPI.
- Add dashboard for live monitoring.
- Deploy model as REST API service.
- Use advanced ensemble methods (XGBoost, LightGBM).

---

## 👤 Author

Saurav Neupane
B.Tech – Computer Science Engineering

