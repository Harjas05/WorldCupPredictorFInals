from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_PATH = RAW_DATA_DIR / "international_results.csv"

def load_clean_dataframe() -> pd.DataFrame:
    matches = pd.read_csv(RESULTS_PATH)

    expected_columns = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
    }
    actual_columns = set(matches.columns)
    diff_in_columns = expected_columns - actual_columns
    if diff_in_columns:
        raise ValueError(f"Missing columns: {diff_in_columns}")
    matches["date"] = pd.to_datetime(matches["date"], errors="raise")
    string_cols = {"home_team", "away_team", "tournament", "city", "country"}

    for col in string_cols:
        matches[col] = matches[col].astype("string").str.strip()
    matches["neutral"] = matches["neutral"].astype(bool)

    matches = matches.dropna(subset=["home_score", "away_score"]).copy()

    matches["home_score"] = pd.to_numeric(matches["home_score"], errors="raise").astype(int)
    matches["away_score"] = pd.to_numeric(matches["away_score"], errors="raise").astype(int)

    matches["date"] = pd.to_datetime(matches["date"], errors="raise")

    matches = matches.sort_values("date").reset_index(drop=True)

    return matches



if __name__ == "__main__":
    load_clean_dataframe()
