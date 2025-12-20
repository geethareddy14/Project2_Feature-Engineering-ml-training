import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42

def generate_synthetic_transactions(n_rows: int = 60000) -> pd.DataFrame:
    """
    Synthetic fraud-like dataset:
    - mix of numeric features + categorical-like signal via bins
    - label is imbalanced (fraud rare)
    """
    rng = np.random.default_rng(RANDOM_SEED)

    amount = rng.lognormal(mean=3.2, sigma=0.8, size=n_rows)
    txn_hour = rng.integers(0, 24, size=n_rows)
    txn_day = rng.integers(1, 31, size=n_rows)

    # "customer behavior"
    acct_age_days = rng.integers(10, 3650, size=n_rows)
    txn_velocity_1h = rng.poisson(lam=1.2, size=n_rows)
    txn_velocity_24h = rng.poisson(lam=6.0, size=n_rows)

    # risk proxies
    geo_distance_km = rng.exponential(scale=60, size=n_rows)
    device_change_flag = rng.binomial(n=1, p=0.08, size=n_rows)

    # create an imbalanced target with signal
    score = (
        0.8 * (amount > np.quantile(amount, 0.92)).astype(int)
        + 0.7 * (txn_velocity_1h >= 4).astype(int)
        + 0.6 * (geo_distance_km > 150).astype(int)
        + 0.5 * device_change_flag
        + 0.4 * (acct_age_days < 120).astype(int)
    )
    # convert score to probability
    prob = 1 / (1 + np.exp(-(score - 2.2)))
    y = rng.binomial(n=1, p=np.clip(prob * 0.22, 0, 0.35), size=n_rows)  # ~2–6% positives

    df = pd.DataFrame({
        "amount": amount,
        "txn_hour": txn_hour,
        "txn_day": txn_day,
        "acct_age_days": acct_age_days,
        "txn_velocity_1h": txn_velocity_1h,
        "txn_velocity_24h": txn_velocity_24h,
        "geo_distance_km": geo_distance_km,
        "device_change_flag": device_change_flag,
        "label": y
    })
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: log transforms, ratios, and interaction-style features."""
    out = df.copy()

    out["log_amount"] = np.log1p(out["amount"])
    out["velocity_ratio"] = (out["txn_velocity_1h"] + 1) / (out["txn_velocity_24h"] + 1)
    out["is_night"] = out["txn_hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)
    out["new_account_flag"] = (out["acct_age_days"] < 180).astype(int)
    out["far_geo_flag"] = (out["geo_distance_km"] > 120).astype(int)

    # drop raw amount if you want (keep both for demonstration)
    return out


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    raw_path = "data/raw/transactions.csv"
    processed_path = "data/processed/features.csv"

    df_raw = generate_synthetic_transactions()
    df_raw.to_csv(raw_path, index=False)

    df_feat = build_features(df_raw)
    df_feat.to_csv(processed_path, index=False)

    print(f"✅ Saved raw data: {raw_path}")
    print(f"✅ Saved processed features: {processed_path}")
    print(f"Rows: {len(df_feat):,} | Positive rate: {df_feat['label'].mean():.3f}")
