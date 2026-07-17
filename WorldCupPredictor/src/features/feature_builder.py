import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from WorldCupPredictor.src.data.clean_results import clean_results
from WorldCupPredictor.src.ratings.update_ratings import update_ratings


BASIC_FEATURE_COLUMNS = [
    "home_elo_before",
    "away_elo_before",
    "elo_diff",
    "neutral",
    "tournament_group_world_cup",
    "tournament_group_world_cup_qualification",
    "tournament_group_continental_championship",
    "tournament_group_continental_qualification",
    "tournament_group_nations_league",
    "tournament_group_friendly",
    "tournament_group_olympic",
    "tournament_group_regional_tournament",
    "tournament_group_other",
    "days_since_last_match_home",
    "days_since_last_match_away",
    "home_goal_diff",
    "away_goal_diff",
    "curr_performance_home",
    "curr_performance_away",
    "recent_goals_conceded_home",
    "recent_goals_conceded_away",
]

TARGET_COLUMN = "result"
ROLLING_MATCH_WINDOW = 5
DEFAULT_DAYS_SINCE_LAST_MATCH = 365
TOURNAMENT_GROUP_COLUMN = "tournament_group"
TOURNAMENT_GROUPS = [
    "world_cup",
    "world_cup_qualification",
    "continental_championship",
    "continental_qualification",
    "nations_league",
    "friendly",
    "olympic",
    "regional_tournament",
    "other",
]

CONTINENTAL_CHAMPIONSHIPS = {
    "afc asian cup",
    "african cup of nations",
    "concacaf championship",
    "copa america",
    "gold cup",
    "oceania nations cup",
    "uefa euro",
}
REGIONAL_KEYWORDS = (
    "abcs",
    "aff",
    "asean",
    "arab",
    "balkan",
    "baltic",
    "cafa",
    "caribbean",
    "cecafa",
    "cfu",
    "cosafa",
    "cup",
    "eaff",
    "games",
    "gulf",
    "saff",
    "tournament",
    "unaf",
    "uncaf",
    "waff",
)


def get_tournament_group(tournament: str) -> str:
    """Group raw tournament names into broad, model-friendly categories."""
    tournament_name = str(tournament).strip().lower()
    tournament_name = tournament_name.replace("é", "e").replace("á", "a")

    if tournament_name == "friendly":
        return "friendly"
    if tournament_name == "fifa world cup":
        return "world_cup"
    if "fifa world cup qualification" in tournament_name:
        return "world_cup_qualification"
    if "nations league" in tournament_name:
        return "nations_league"
    if "olympic" in tournament_name:
        return "olympic"
    if "qualification" in tournament_name:
        return "continental_qualification"
    if tournament_name in CONTINENTAL_CHAMPIONSHIPS:
        return "continental_championship"
    if any(keyword in tournament_name for keyword in REGIONAL_KEYWORDS):
        return "regional_tournament"

    return "other"


def add_tournament_group_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add tournament group labels and one-hot tournament group features."""
    if "tournament" not in matches.columns:
        raise ValueError("Missing column: tournament")

    matches = matches.copy()
    matches[TOURNAMENT_GROUP_COLUMN] = matches["tournament"].map(get_tournament_group)

    group_features = pd.get_dummies(
        matches[TOURNAMENT_GROUP_COLUMN],
        prefix=TOURNAMENT_GROUP_COLUMN,
        dtype=int,
    )
    expected_columns = [
        f"{TOURNAMENT_GROUP_COLUMN}_{group}" for group in TOURNAMENT_GROUPS
    ]
    group_features = group_features.reindex(columns=expected_columns, fill_value=0)

    return pd.concat([matches, group_features], axis=1)


def _average_recent_stat(history: list[dict], stat: str) -> float:
    recent_matches = history[-ROLLING_MATCH_WINDOW:]
    if not recent_matches:
        return 0.0

    return sum(match[stat] for match in recent_matches) / len(recent_matches)


def _days_since_last_match(history: list[dict], match_date: pd.Timestamp) -> int:
    if not history:
        return DEFAULT_DAYS_SINCE_LAST_MATCH

    return int((match_date - history[-1]["date"]).days)


def add_recent_team_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add pre-match rolling form, rest, goal difference, and conceded features."""
    required_columns = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    }
    missing_columns = required_columns - set(matches.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    matches = matches.sort_values("date").reset_index(drop=True).copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="raise")
    team_history = {}

    days_since_last_match_home = []
    days_since_last_match_away = []
    home_goal_diff = []
    away_goal_diff = []
    curr_performance_home = []
    curr_performance_away = []
    recent_goals_conceded_home = []
    recent_goals_conceded_away = []

    for _, match in matches.iterrows():
        match_date = match["date"]
        home_team = match["home_team"]
        away_team = match["away_team"]
        home_history = team_history.get(home_team, [])
        away_history = team_history.get(away_team, [])

        days_since_last_match_home.append(
            _days_since_last_match(home_history, match_date)
        )
        days_since_last_match_away.append(
            _days_since_last_match(away_history, match_date)
        )
        home_goal_diff.append(_average_recent_stat(home_history, "goal_diff"))
        away_goal_diff.append(_average_recent_stat(away_history, "goal_diff"))
        curr_performance_home.append(_average_recent_stat(home_history, "points"))
        curr_performance_away.append(_average_recent_stat(away_history, "points"))
        recent_goals_conceded_home.append(
            _average_recent_stat(home_history, "goals_conceded")
        )
        recent_goals_conceded_away.append(
            _average_recent_stat(away_history, "goals_conceded")
        )

        home_score = int(match["home_score"])
        away_score = int(match["away_score"])
        if home_score > away_score:
            home_points = 3
            away_points = 0
        elif home_score < away_score:
            home_points = 0
            away_points = 3
        else:
            home_points = 1
            away_points = 1

        team_history.setdefault(home_team, []).append(
            {
                "date": match_date,
                "points": home_points,
                "goal_diff": home_score - away_score,
                "goals_conceded": away_score,
            }
        )
        team_history.setdefault(away_team, []).append(
            {
                "date": match_date,
                "points": away_points,
                "goal_diff": away_score - home_score,
                "goals_conceded": home_score,
            }
        )

    matches["days_since_last_match_home"] = days_since_last_match_home
    matches["days_since_last_match_away"] = days_since_last_match_away
    matches["home_goal_diff"] = home_goal_diff
    matches["away_goal_diff"] = away_goal_diff
    matches["curr_performance_home"] = curr_performance_home
    matches["curr_performance_away"] = curr_performance_away
    matches["recent_goals_conceded_home"] = recent_goals_conceded_home
    matches["recent_goals_conceded_away"] = recent_goals_conceded_away

    return matches


def build_basic_feature_table(matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the first model-ready table using pre-match Elo features."""
    if matches is None:
        matches = clean_results()

    if "home_elo_before" not in matches.columns:
        matches = update_ratings(matches)

    matches = add_tournament_group_features(matches)
    matches = add_recent_team_features(matches)

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
    """Split the feature table into model inputs and target labels."""
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
