import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import random
import json

SEED = 42
N_TRANSACTIONS = 284807
FRAUD_RATE = 0.00173
OUTPUT_DIR = Path("data/raw")

np.random.seed(SEED)
random.seed(SEED)

def generate_creditcard_data(n: int) -> pd.DataFrame:
    """Generate synthetic credit card transaction data
    mimicking the Kaggle creditcard.csv structure."""
    print(f"Generating {n:,} synthetic transactions...")

    time = np.sort(np.random.uniform(0, 172800, n))

    V_features = {}
    for i in range(1, 29):
        V_features[f"V{i}"] = np.random.normal(0, 1, n)

    amount = np.abs(np.random.lognormal(mean=3.5, sigma=1.5, size=n)).round(2)
    amount = np.clip(amount, 0.01, 25000)

    n_fraud = int(n * FRAUD_RATE)
    fraud_indices = np.random.choice(n, n_fraud, replace=False)

    labels = np.zeros(n, dtype=int)
    labels[fraud_indices] = 1

    # Inject realistic fraud patterns
    V_features["V1"][fraud_indices] -= 3
    V_features["V3"][fraud_indices] -= 2
    V_features["V4"][fraud_indices] += 2
    V_features["V10"][fraud_indices] -= 3
    V_features["V11"][fraud_indices] += 2
    V_features["V14"][fraud_indices] -= 4
    amount[fraud_indices] = np.random.uniform(100, 5000, n_fraud).round(2)

    df = pd.DataFrame({
        "Time": time,
        **V_features,
        "Amount": amount,
        "Class": labels
    })

    print(f"✅ Generated {len(df):,} transactions")
    print(f"✅ Fraud rate: {df['Class'].mean():.4f} ({df['Class'].sum():,} fraud cases)")
    return df

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "creditcard.csv"

    df = generate_creditcard_data(N_TRANSACTIONS)
    df.to_csv(output_path, index=False)

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "n_transactions": len(df),
        "n_fraud": int(df["Class"].sum()),
        "fraud_rate": float(df["Class"].mean()),
        "features": list(df.columns),
    }

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved to {output_path}")
    print(f"✅ Saved metadata")

if __name__ == "__main__":
    main()