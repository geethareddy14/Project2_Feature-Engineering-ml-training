# Project 2 — Feature Engineering & ML Training (Production-Style)

This project demonstrates a **production-oriented ML training workflow** for fraud detection:
**data → feature engineering → reproducible training → experiment tracking → evaluation artifacts**.

It is built as a portfolio-ready component that mirrors how enterprise ML teams structure training pipelines.

---

## What this project covers
- Feature engineering with explicit **train/inference parity**
- Imbalanced classification handling (fraud is rare)
- Model training using **XGBoost**
- Experiment tracking with **MLflow**
- Exportable model artifacts + evaluation reports

---

## Tech Stack
- Python, Pandas, NumPy  
- scikit-learn, XGBoost  
- MLflow (experiment tracking)  
- Matplotlib (evaluation artifacts)

---

## Architecture (high-level)
1. **Ingest** raw dataset (`creditcard.csv`)
2. **Build features** + store schema (`feature_schema.json`)
3. **Train model** (imbalance-aware) + log metrics to MLflow
4. **Evaluate** and generate artifacts (report + confusion matrix)

---

## Repository Structure
