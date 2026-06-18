import pandas as pd

from src.data_sources.prediction_snapshots import (
    load_prediction_snapshots,
    prediction_overview,
    prediction_snapshot_row,
    upsert_prediction_snapshot,
)


def test_prediction_snapshot_row_from_prediction():
    row = prediction_snapshot_row(
        {
            "match_id": "M003",
            "home_team": "Canada",
            "away_team": "Bosnia and Herzegovina",
            "predicted_home_goals": 2,
            "predicted_away_goals": 1,
            "home_win_prob": 0.5,
            "draw_prob": 0.25,
            "away_win_prob": 0.25,
            "model_weight": 0.3,
            "moneyline_market_weight": 0.7,
            "total_calibration_weight": 0.2,
            "team_total_calibration_weight": 0.4,
            "spread_calibration_weight": 0.3,
            "market_timestamp": "2026-06-12T10:00:00+00:00",
        }
    )
    assert row["predicted_score"] == "2-1"
    assert row["snapshot_date"] == "2026-06-12"
    assert row["total_calibration_weight"] == 0.2
    assert row["team_total_calibration_weight"] == 0.4
    assert row["spread_calibration_weight"] == 0.3


def test_upsert_prediction_snapshot_replaces_existing_match(tmp_path):
    path = tmp_path / "prediction_snapshots.csv"
    upsert_prediction_snapshot(path, {"match_id": "M001", "predicted_score": "1-0"})
    upsert_prediction_snapshot(path, {"match_id": "M001", "predicted_score": "2-0"})
    df = load_prediction_snapshots(path)
    assert len(df) == 1
    assert df.iloc[0]["predicted_score"] == "2-0"


def test_prediction_overview_scores_played_results():
    schedule = pd.DataFrame(
        [
            {"match_id": "M001", "date": "2026-06-11", "stage": "Group Stage", "home_team": "Mexico", "away_team": "South Africa", "status": "played", "home_score": "2", "away_score": "0"},
            {"match_id": "M002", "date": "2026-06-12", "stage": "Group Stage", "home_team": "South Korea", "away_team": "Czechia", "status": "played", "home_score": "2", "away_score": "1"},
        ]
    )
    snapshots = pd.DataFrame(
        [
            {"match_id": "M001", "predicted_home_score": "1", "predicted_away_score": "0", "predicted_score": "1-0"},
            {"match_id": "M002", "predicted_home_score": "1", "predicted_away_score": "1", "predicted_score": "1-1"},
        ]
    )
    overview = prediction_overview(schedule, snapshots)
    assert overview["summary"]["played_predictions"] == 2
    assert overview["summary"]["correct_scores"] == 0
    assert overview["summary"]["correct_winners"] == 1
    assert overview["summary"]["correct_goal_diffs"] == 0


def test_prediction_overview_ignores_matches_without_saved_prediction():
    schedule = pd.DataFrame(
        [
            {"match_id": "M001", "date": "2026-06-11", "stage": "Group Stage", "home_team": "Mexico", "away_team": "South Africa", "status": "played", "home_score": "2", "away_score": "0"},
            {"match_id": "M002", "date": "2026-06-12", "stage": "Group Stage", "home_team": "South Korea", "away_team": "Czechia", "status": "played", "home_score": "2", "away_score": "1"},
        ]
    )
    snapshots = pd.DataFrame([{"match_id": "M001", "predicted_home_score": "1", "predicted_away_score": "0", "predicted_score": "1-0"}])
    overview = prediction_overview(schedule, snapshots)
    assert overview["summary"]["predictions"] == 1
    assert [row["match_id"] for row in overview["rows"]] == ["M001"]
