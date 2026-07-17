import pandas as pd

try:
    from WorldCupPredictor.src.data.clean_results import clean_results
    from WorldCupPredictor.src.ratings.update_ratings import update_ratings
except ImportError:
    try:
        from src.data.clean_results import clean_results
        from src.ratings.update_ratings import update_ratings
    except ImportError:
        from clean_results import clean_results
        from update_ratings import update_ratings


BASIC_FEATURE_COLUMNS = [
    "home_elo_before",
    "away_elo_before",
    "elo_diff",
    "neutral",
]

TARGET_COLUMN = "result"


def build_basic_feature_table(matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create the first model-ready table using only pre-match features."""
    if matches is None:
        matches = clean_results()
        matches = update_ratings(matches)

    required_columns = set(BASIC_FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns - set(matches.columns)
    if missing_columns:
        raise ValueError(f"Missing feature columns: {sorted(missing_columns)}")

    feature_table = matches[BASIC_FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    feature_table["neutral"] = feature_table["neutral"].astype(int)

    return feature_table.reset_index(drop=True)


def split_features_and_target(
    feature_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature table into X features and y target."""
    missing_columns = set(BASIC_FEATURE_COLUMNS + [TARGET_COLUMN]) - set(
        feature_table.columns
    )
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    X = feature_table[BASIC_FEATURE_COLUMNS].copy()
    y = feature_table[TARGET_COLUMN].copy()

    return X, y


if __name__ == "__main__":
    features = build_basic_feature_table()
    X, y = split_features_and_target(features)

    print(features.head())
    print(X.shape)
    print(y.value_counts())
