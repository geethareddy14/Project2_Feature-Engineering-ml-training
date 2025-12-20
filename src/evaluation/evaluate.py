import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

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

    model = joblib.load("models/xgb_model.joblib")
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    os.makedirs("reports", exist_ok=True)

    # Save classification report
    report = classification_report(y, preds, digits=4)
    with open("reports/classification_report.txt", "w") as f:
        f.write(report)

    # Confusion matrix plot
    cm = confusion_matrix(y, preds)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0,1], ["0","1"])
    plt.yticks([0,1], ["0","1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=160)

    print("✅ Saved reports to /reports")
    print(report)

if __name__ == "__main__":
    main()
