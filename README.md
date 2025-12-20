# Feature Engineering & ML Training Pipeline

> A reproducible feature engineering + model training pipeline designed with production ML practices: clean feature definitions, leakage-safe splitting, tracked experiments, and saved artifacts.

---

## 🎯 Goal
Build an end-to-end ML training workflow that:
- creates an **offline feature table** from raw data
- performs **time-based splits** (avoid leakage)
- trains a baseline model and evaluates it
- saves **artifacts** (model + metrics + feature list)
- can be extended to MLflow / model registry

---

## 🧱 What’s Included
- Feature engineering (rolling/aggregated risk features)
- Time-based train/val/test split
- Baseline model training (Logistic Regression)
- Metrics + artifacts saved per run

---

## Tech Stack
- Python, Pandas, NumPy
- scikit-learn, XGBoost
- MLflow (experiment tracking)
- Matplotlib (evaluation artifact)

---

## 📁 Repo Structure
