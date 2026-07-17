import pandas as pd

from load_results import load_clean_dataframe


def remove_duplicates(matches: pd.DataFrame) -> pd.DataFrame:
    key = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
    ]
    duplicate_count = matches.duplicated(subset=key).sum()
    if duplicate_count:
        print(f"Removing {duplicate_count} duplicate matches")
    return matches.drop_duplicates(subset=key).reset_index(drop=True)


def remove_invalid_scores(matches: pd.DataFrame) -> pd.DataFrame:
    invalid_scores = (matches["home_score"] < 0) | (matches["away_score"] < 0)
    invalid_count = invalid_scores.sum()
    if invalid_count:
        print(f"Removing {invalid_count} matches with negative scores")
    return matches[~invalid_scores].reset_index(drop=True)


def remove_invalid_rows(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.dropna(subset=["date", "home_team", "away_team"]).copy()
    blank_teams = (
        matches["home_team"].str.len().eq(0)
        | matches["away_team"].str.len().eq(0)
    )
    matches = matches[~blank_teams].copy()

    same_team = matches["home_team"] == matches["away_team"]

    if same_team.any():
        raise ValueError("Found matches where home_team equals away_team")

    return matches.reset_index(drop=True)

def add_rows(matches: pd.DataFrame) -> pd.DataFrame:
    matches["home_goal_diff"] = matches["home_score"] - matches["away_score"]
    matches["result"] = "draw"

    matches.loc[matches["home_score"] > matches["away_score"], "result"] = "home_win"
    matches.loc[matches["home_score"] < matches["away_score"], "result"] = "away_win"
    return matches.reset_index(drop=True)



if __name__ == "__main__":
    matches = load_clean_dataframe()
    print("loaded data frame")
    matches = remove_invalid_rows(matches)
    print("removed invalid rows")
    matches = remove_duplicates(matches)
    print("removed duplicates")
    matches = remove_invalid_scores(matches)
    print("removed invalid scores")
    matches = add_rows(matches)
    print("added wins and goal differences")
