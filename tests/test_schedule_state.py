import pandas as pd

from src.data_sources.schedule import played_worldcup_results, validate_schedule
from src.models.predictor import skip_reason


def test_played_matches_are_included_in_state_results():
    schedule = pd.DataFrame([
        {"match_id": "M001", "date": "2026-06-11", "home_team": "Mexico", "away_team": "South Africa", "status": "played", "home_score": "2", "away_score": "1", "city": "", "country": "", "neutral": "true"}
    ])
    results = played_worldcup_results(schedule)
    assert len(results) == 1
    assert results.iloc[0]["tournament"] == "FIFA World Cup 2026"


def test_played_matches_are_not_predicted_again():
    row = pd.Series({"status": "played", "home_team": "Mexico", "away_team": "South Africa"})
    assert skip_reason(row) == "already played"


def test_unresolved_placeholder_matches_are_skipped():
    row = pd.Series({"status": "scheduled", "home_team": "Winner Group A", "away_team": "Mexico"})
    assert skip_reason(row) == "unresolved placeholder team"


def test_schedule_status_validation():
    schedule = pd.DataFrame([
        {"match_id": "M001", "status": "scheduled", "home_score": "", "away_score": "", "home_team": "A", "away_team": "B", "date": "2026-06-11"},
        {"match_id": "M002", "status": "bad", "home_score": "", "away_score": "", "home_team": "A", "away_team": "B", "date": "2026-06-12"},
    ])
    errors, _ = validate_schedule(schedule)
    assert any("invalid statuses" in error for error in errors)

