import os
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

DATA_PATH = Path("data/processed/features.parquet")
SCHEMA_PATH = Path("data/processed/feature_schema.json")

RANDOM_SEED = 42
def main():
    if not DATA_PATH.exists() or not SCHEMA_PATH.exists():
        raise FileNotFoundError("Processed data not found. Run build_features first.")

    df = pd.read_parquet(DATA_PATH)
    schema = json.loads(SCHEMA_PATH.read_text())
    feature_cols = schema["feature_cols"]

    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Handle imbalance for XGBoost
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = float(neg / max(pos, 1))

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )

    mlflow.set_experiment("project2_feature_engineering_ml_training")

    with mlflow.start_run(run_name="xgb_imbalance_aware"):
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        ap = average_precision_score(y_test, proba)

        mlflow.log_params({
            "model": "XGBClassifier",
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "scale_pos_weight": scale_pos_weight
        })
        mlflow.log_metric("roc_auc", float(auc))
        mlflow.log_metric("pr_auc", float(ap))

        os.makedirs("models", exist_ok=True)
        local_model_path = "models/xgb_model.joblib"
        joblib.dump(model, local_model_path)

        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"✅ ROC-AUC: {auc:.4f}")
        print(f"✅ PR-AUC : {ap:.4f}")
        print(f"✅ Saved model: {local_model_path}")

if __name__ == "__main__":
    main()
