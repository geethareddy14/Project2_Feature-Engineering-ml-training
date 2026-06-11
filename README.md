# Project 2 — Feature Engineering & ML Training

A production-oriented ML training pipeline for credit card fraud detection —
built to mirror how enterprise ML teams structure end-to-end training workflows.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![MLflow](https://img.shields.io/badge/MLflow-2.0+-red)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4-green)

---

## Overview

This project demonstrates a complete ML training pipeline — from raw data
through feature engineering, model training, experiment tracking, and
evaluation artifacts — using credit card transaction data with realistic fraud patterns. 

**Key Results:**
- ROC-AUC: ~0.95 on held-out test set
- PR-AUC: ~0.82 (optimized for imbalanced fraud detection)
- Fraud detection at 0.17% base rate with class imbalance handling
- Full experiment tracking via MLflow

---

## Architecture
data/
get_data.py            # Kaggle dataset downloader
generate_data.py       # Synthetic data generator (284K transactions)
features/
build_features.py      # Feature engineering with train/inference parity
training/
train_model.py         # XGBoost training with imbalance handling + MLflow
evaluation/
evaluate_model.py      # Model evaluation, ROC/PR curves, SHAP plots
models/                # Saved model artifacts
reports/               # Evaluation outputs (metrics, plots)
data/
raw/                   # Raw transaction data
processed/             # Engineered features + schema

---

## Features Engineered

| Feature | Description |
|---|---|
| log_amount | Log-transformed transaction amount |
| amount_per_time | Amount normalized by elapsed time |
| txn_hour | Hour of day derived from timestamp |
| is_night | Binary flag for late night transactions |
| V1-V28 | PCA-transformed transaction features |

---

## Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | ~0.95 |
| PR-AUC | ~0.82 |
| Threshold | 0.35 (optimized for recall) |
| Training data | 227,845 transactions |
| Test data | 56,962 transactions |

**Why PR-AUC matters:** With only 0.17% fraud rate, ROC-AUC alone is
misleading. PR-AUC directly measures performance on the minority fraud
class — the metric that actually matters in production.

---

## Quick Start

**1. Generate synthetic data**
```bash
python data/generate_data.py
```

**2. Build features**
```bash
python features/build_features.py
```

**3. Train model**
```bash
python training/train_model.py
```

**4. Evaluate**
```bash
python evaluation/evaluate_model.py
```

**5. View MLflow experiments**
```bash
mlflow ui
# Visit http://localhost:5000
```

---

## Key Engineering Decisions

**Why XGBoost over deep learning?**
Gradient boosting consistently outperforms neural networks on tabular
financial data — faster training, better interpretability, and
production-proven reliability in regulated environments.

**Why threshold 0.35 instead of 0.5?**
In fraud detection, false negatives (missed fraud) are more costly than
false positives (unnecessary reviews). Lower threshold increases recall
at acceptable precision cost.

**Why PR-AUC as primary metric?**
With 0.17% fraud rate, accuracy is meaningless. PR-AUC directly measures
precision-recall tradeoff on the minority class — the only metric that
matters for fraud detection.

**Train/inference parity:**
Feature schema saved as JSON ensures identical feature engineering
at training and inference time — preventing training-serving skew.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | XGBoost 2.0 with early stopping |
| Explainability | SHAP TreeExplainer |
| Experiment Tracking | MLflow |
| Feature Engineering | Pandas, NumPy, scikit-learn |
| Evaluation | Matplotlib, scikit-learn metrics |
| Language | Python 3.10+ |

---

## Author

**Geetha Bommareddy** — AI/ML Engineer | JPMC
[LinkedIn](https://www.linkedin.com/in/geethareddy521) |
[Portfolio](https://geethareddy14.github.io/GeethaReddy-PortFolio/)