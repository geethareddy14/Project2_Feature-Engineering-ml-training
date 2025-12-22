import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/creditcard.csv")
OUT_PATH = Path("data/processed/features.parquet")
META_PATH = Path("data/processed/feature_schema.json")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Rename target for clarity
    out = out.rename(columns={"Class": "label"})

    # Basic engineered features
    out["log_amount"] = np.log1p(out["Amount"])
    out["amount_per_time"] = out["Amount"] / (out["Time"] + 1)

    # Time is seconds elapsed. Convert to hour-of-day style feature (approx.)
    out["txn_hour"] = ((out["Time"] // 3600) % 24).astype(int)
    out["is_night"] = out["txn_hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)

    return out

def main():
    os.makedirs("data/processed", exist_ok=True)

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_PATH}. Run: python src/data/get_data.py"
        )

    df = pd.read_csv(RAW_PATH)
    df_feat = build_features(df)

    # Define feature columns for training (keep explicit for parity)
    feature_cols = [c for c in df_feat.columns if c not in ["label"]]

    # Save processed data
    df_feat.to_parquet(OUT_PATH, index=False)

    # Save schema (production-style)
    with open(META_PATH, "w") as f:
        json.dump({"feature_cols": feature_cols, "target_col": "label"}, f, indent=2)

    pos_rate = df_feat["label"].mean()
    print(f"✅ Saved: {OUT_PATH}")
    print(f"✅ Saved schema: {META_PATH}")
    print(f"Rows: {len(df_feat):,} | Positive rate: {pos_rate:.4f}")

if __name__ == "__main__":
    main()
