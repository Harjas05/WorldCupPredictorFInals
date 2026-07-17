from pathlib import Path
import pandas as pd

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_PATH = RAW_DATA_DIR / "international_results.csv"


def download_results() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    matches = pd.read_csv(RESULTS_URL, parse_dates=["date"])
    matches.to_csv(RESULTS_PATH, index=False)

    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    download_results()


