import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

RANDOM_SEED = 42

FEATURE_COLS = [
    "amount", "log_amount", "txn_hour", "txn_day", "acct_age_days",
    "txn_velocity_1h", "txn_velocity_24h", "velocity_ratio",
    "geo_distance_km", "device_change_flag", "is_night",
    "new_account_flag", "far_geo_flag"
]

def main():
    df = pd.read_csv("data/processed/features.csv")
    X = df[FEATURE_COLS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Simple, production-friendly baseline
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    mlflow.set_experiment("project2_feature_engineering_ml_training")

    with mlflow.start_run(run_name="xgb_baseline"):
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        ap = average_precision_score(y_test, proba)

        mlflow.log_params({
            "model": "XGBClassifier",
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate
        })
        mlflow.log_metric("roc_auc", float(auc))
        mlflow.log_metric("avg_precision", float(ap))

        os.makedirs("models", exist_ok=True)
        local_model_path = "models/xgb_model.joblib"
        joblib.dump(model, local_model_path)

        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"✅ ROC-AUC: {auc:.4f}")
        print(f"✅ Avg Precision: {ap:.4f}")
        print(f"✅ Saved model to {local_model_path}")

if __name__ == "__main__":
    main()
