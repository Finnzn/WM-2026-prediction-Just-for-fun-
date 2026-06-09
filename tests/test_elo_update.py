import pandas as pd

from src.config import Config
from src.data_sources.elo import update_dynamic_elo


def test_elo_ratings_update_after_played_world_cup_match():
    cfg = Config()
    played = pd.DataFrame([
        {"date": "2026-06-11", "home_team": "Mexico", "away_team": "South Africa", "home_score": 2, "away_score": 1, "neutral": "true"}
    ])
    ratings = update_dynamic_elo({"Mexico": 1500, "South Africa": 1500}, played, cfg)
    assert ratings["Mexico"] > 1500
    assert ratings["South Africa"] < 1500

