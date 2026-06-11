import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)

DATA_PATH = Path("data/processed/features.parquet")
SCHEMA_PATH = Path("data/processed/feature_schema.json")
MODEL_PATH = Path("models/xgb_model.joblib")
REPORTS_DIR = Path("reports")
RANDOM_SEED = 42

def plot_roc_curve(y_test, proba):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Fraud Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png")
    plt.close()
    print(f"✅ Saved ROC curve")

def plot_pr_curve(y_test, proba):
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR-AUC = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Fraud Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "pr_curve.png")
    plt.close()
    print(f"✅ Saved PR curve")

def plot_shap(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:500])
    plt.figure()
    shap.summary_plot(shap_values, X_test[:500], show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary.png")
    plt.close()
    print(f"✅ Saved SHAP summary plot")

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run training first.")

    REPORTS_DIR.mkdir(exist_ok=True)

    # Load data
    df = pd.read_parquet(DATA_PATH)
    schema = json.loads(SCHEMA_PATH.read_text())
    feature_cols = schema["feature_cols"]

    X = df[feature_cols]
    y = df["label"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Load model
    model = joblib.load(MODEL_PATH)
    print(f"✅ Loaded model from {MODEL_PATH}")

    # Predict
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.35).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    cm = confusion_matrix(y_test, preds)

    print(f"\n{'='*40}")
    print(f"ROC-AUC:  {auc:.4f}")
    print(f"PR-AUC:   {ap:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"TN={cm[0][0]:,}  FP={cm[0][1]:,}")
    print(f"FN={cm[1][0]:,}  TP={cm[1][1]:,}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, preds))

    # Save metrics
    metrics = {
        "roc_auc": float(auc),
        "pr_auc": float(ap),
        "threshold": 0.35,
        "true_negatives": int(cm[0][0]),
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
        "true_positives": int(cm[1][1]),
    }
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved metrics to reports/metrics.json")

    # Plots
    plot_roc_curve(y_test, proba)
    plot_pr_curve(y_test, proba)
    plot_shap(model, X_test)

    print(f"\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()