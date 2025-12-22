import os
import shutil
import zipfile
import subprocess
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def kaggle_download():
    """
    Optional Kaggle download:
    Requires:
      - kaggle CLI installed (pip install kaggle)
      - kaggle.json in the right place (see README)
    """
    dataset = "mlg-ulb/creditcardfraud"
    out_zip = RAW_DIR / "creditcardfraud.zip"

    print("Attempting Kaggle download via kaggle CLI...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(RAW_DIR), "--force"],
        check=True
    )

    # Kaggle names zip file like creditcardfraud.zip (usually)
    # Find any zip and unzip it
    zips = list(RAW_DIR.glob("*.zip"))
    if not zips:
        raise FileNotFoundError("No zip found after Kaggle download.")

    zip_path = zips[0]
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)

    print(f"✅ Extracted dataset to {RAW_DIR}")
    return RAW_DIR / "creditcard.csv"


def manual_instructions():
    print("\nManual dataset setup:")
    print("1) Download 'Credit Card Fraud Detection' dataset from Kaggle.")
    print("2) Place 'creditcard.csv' into: data/raw/")
    print("3) Re-run this script or proceed to feature build.\n")


if __name__ == "__main__":
    csv_path = RAW_DIR / "creditcard.csv"
    if csv_path.exists():
        print(f"✅ Found dataset at {csv_path}")
    else:
        try:
            csv_path = kaggle_download()
            if csv_path.exists():
                print(f"✅ Dataset ready at {csv_path}")
            else:
                manual_instructions()
        except Exception as e:
            print(f"⚠️ Kaggle download failed: {e}")
            manual_instructions()
