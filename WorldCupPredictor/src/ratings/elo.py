import pandas as pd

def calculate_expected_score(home_elo: float, away_elo: float) -> float:
    """
    Calculate the expected score for the home team based on ELO ratings.

    Args:
        home_elo (float): ELO rating of the home team.
        away_elo (float): ELO rating of the away team.

    Returns:
        float: Expected score for the home team (between 0 and 1).
    """
    expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
    return expected_home

def get_actual_score(home_score: int, away_score:int) -> float:
    """
    Determine the actual score for the home team based on match results.

    Args:
        home_score (int): Goals scored by the home team.
        away_score (int): Goals scored by the away team.

    Returns:
        float: Actual score for the home team (1 for win, 0.5 for draw, 0 for loss).
    """
    if home_score > away_score:
        return 1.0  # Home win
    elif home_score == away_score:
        return 0.5  # Draw
    else:
        return 0.0  # Home loss

def update_elo(home_elo: float, home_score:int, away_elo: float, away_score: int, k_factor: int = 32) -> tuple[float, float]:
    """
    Update the ELO rating for the home team after a match.

    Args:
        home_elo (float): Current ELO rating of the home team.
        home_score (int): Goals scored by the home team.
        away_elo (float): Current ELO rating of the away team.
        away_score (int): Goals scored by the away team.
        k_factor (int): K-factor for ELO calculation (default is 32).

    Returns:
        float: Updated ELO rating for the home team.
    """
    home_acc_score = get_actual_score(home_score, away_score)
    away_acc_score = 1.0 - home_acc_score
    expected_score = calculate_expected_score(home_elo, away_elo)
    expected_away_score = 1.0 - expected_score
    home_elo = home_elo + (k_factor * (home_acc_score - expected_score))
    away_elo = away_elo + (k_factor * (away_acc_score - expected_away_score))
    return home_elo, away_elo

