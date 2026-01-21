# Project 2 — Feature Engineering & ML Training (Production-Style)

This project demonstrates a **production-oriented ML training workflow** for fraud detection:
**data → feature engineering → reproducible training → experiment tracking → evaluation artifacts**. 

## Automation & Orchestration (n8n + GitHub Actions)

This project is automated using **n8n Cloud** as an orchestration layer.

### Workflow
- A **Schedule Trigger** in n8n runs on a defined interval
- n8n triggers the ML training workflow using GitHub Actions (`workflow_dispatch`)
- The pipeline is executed in a reproducible CI environment
- This setup simulates real-world MLOps retraining orchestration

### Tech Used
- n8n (workflow orchestration)
- GitHub Actions
- REST APIs
- Python (ML pipeline)

### Why this matters
This automation demonstrates how ML workflows can be scheduled, triggered, and monitored without maintaining dedicated servers.


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
