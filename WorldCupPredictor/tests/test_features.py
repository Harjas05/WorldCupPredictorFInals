import unittest

import pandas as pd

from WorldCupPredictor.src.features.feature_builder import (
    DEFAULT_DAYS_SINCE_LAST_MATCH,
    add_recent_team_features,
)


class RecentTeamFeatureTests(unittest.TestCase):
    def test_recent_team_features_use_only_prior_matches(self):
        matches = pd.DataFrame(
            [
                {
                    "date": "2024-01-01",
                    "home_team": "A",
                    "away_team": "B",
                    "home_score": 2,
                    "away_score": 0,
                },
                {
                    "date": "2024-01-11",
                    "home_team": "A",
                    "away_team": "C",
                    "home_score": 1,
                    "away_score": 1,
                },
                {
                    "date": "2024-01-21",
                    "home_team": "B",
                    "away_team": "A",
                    "home_score": 3,
                    "away_score": 2,
                },
            ]
        )

        features = add_recent_team_features(matches)

        self.assertEqual(
            features.loc[0, "days_since_last_match_home"],
            DEFAULT_DAYS_SINCE_LAST_MATCH,
        )
        self.assertEqual(features.loc[1, "days_since_last_match_home"], 10)
        self.assertEqual(features.loc[2, "days_since_last_match_away"], 10)

        self.assertEqual(features.loc[1, "home_goal_diff"], 2.0)
        self.assertEqual(features.loc[2, "away_goal_diff"], 1.0)
        self.assertEqual(features.loc[1, "curr_performance_home"], 3.0)
        self.assertEqual(features.loc[2, "curr_performance_away"], 2.0)
        self.assertEqual(features.loc[1, "recent_goals_conceded_home"], 0.0)
        self.assertEqual(features.loc[2, "recent_goals_conceded_away"], 0.5)


if __name__ == "__main__":
    unittest.main()
