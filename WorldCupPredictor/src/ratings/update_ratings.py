import pandas as pd

try:
    from .elo import update_elo
except ImportError:
    from elo import update_elo


def update_ratings(
    matches: pd.DataFrame,
    initial_elo: float = 1500,
    k_factor: int = 32,
    team_names: dict | None = None,
    home_elo_column: str = "home_elo",
    away_elo_column: str = "away_elo",
) -> pd.DataFrame:
    """Update Elo ratings match by match and return match-level rating history."""
   
    team_ratings = dict(team_names or {})
    home_elo_before = []
    away_elo_before = []
    home_elo_after = []
    away_elo_after = []

    for _, match in matches.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]
        home_score = match["home_score"]
        away_score = match["away_score"]

        if home_team not in team_ratings:
            team_ratings[home_team] = initial_elo
        if away_team not in team_ratings:
            team_ratings[away_team] = initial_elo

        before_home_elo = team_ratings[home_team]
        before_away_elo = team_ratings[away_team]

        after_home_elo, after_away_elo = update_elo(
            before_home_elo,
            home_score,
            before_away_elo,
            away_score,
            k_factor,
        )
        team_ratings[home_team] = after_home_elo
        team_ratings[away_team] = after_away_elo

        home_elo_before.append(before_home_elo)
        away_elo_before.append(before_away_elo)
        home_elo_after.append(after_home_elo)
        away_elo_after.append(after_away_elo)

    matches["home_elo_before"] = home_elo_before
    matches["away_elo_before"] = away_elo_before
    matches["home_elo_after"] = home_elo_after
    matches["away_elo_after"] = away_elo_after
    matches["elo_diff"] = (
        matches["home_elo_before"] - matches["away_elo_before"]
    )

    return matches.reset_index(drop=True)
