import pandas as pd

from src.config import Config
from src.data_sources.historical_results import effective_results
from src.models.backtesting import evaluate_market_snapshots, simple_backtest, walk_forward_model_backtest


def test_backtesting_uses_prior_rows_only_and_runs():
    rows = []
    for i in range(25):
        rows.append({"date": f"2024-01-{i+1:02d}", "home_score": i % 3, "away_score": (i + 1) % 3})
    metrics = simple_backtest(pd.DataFrame(rows))
    assert metrics["matches"] == 15
    assert "accuracy" in metrics


def test_walk_forward_model_backtest_runs_with_prior_data_only():
    rows = []
    for i in range(40):
        rows.append(
            {
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "home_team": "A" if i % 2 == 0 else "B",
                "away_team": "B" if i % 2 == 0 else "A",
                "home_score": 2 if i % 3 == 0 else 1,
                "away_score": 0 if i % 3 == 0 else 1,
                "neutral": "true",
            }
        )
    metrics = walk_forward_model_backtest(pd.DataFrame(rows), Config(), min_training_matches=10, max_evaluated_matches=5)
    assert metrics["model"] == "walk_forward_statistical_model_no_markets"
    assert metrics["matches"] == 5
    assert "wdl_log_loss" in metrics
    assert metrics["market_data_included"] is False


def test_effective_results_excludes_same_day_when_exclusive():
    historical = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-06-09"), "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 0},
            {"date": pd.Timestamp("2026-06-10"), "home_team": "A", "away_team": "C", "home_score": 3, "away_score": 0},
        ]
    )
    current_wc = pd.DataFrame(
        [
            {"date": "2026-06-10", "home_team": "D", "away_team": "E", "home_score": "2", "away_score": "1"},
            {"date": "2026-06-11", "home_team": "F", "away_team": "G", "home_score": "2", "away_score": "1"},
        ]
    )
    effective = effective_results(historical, current_wc, Config(), as_of=pd.Timestamp("2026-06-10").date(), exclusive=True)
    assert set(effective["home_team"]) == {"A"}


def test_market_snapshot_evaluation_scores_played_matches():
    schedule = pd.DataFrame(
        [
            {"match_id": "M001", "status": "played", "home_team": "Mexico", "away_team": "South Africa", "home_score": "2", "away_score": "0"}
        ]
    )
    snapshots = pd.DataFrame(
        [
            {"match_id": "M001", "market_type": "moneyline", "team": "Mexico", "line": None, "probability": 0.695},
            {"match_id": "M001", "market_type": "moneyline", "team": "Draw", "line": None, "probability": 0.205},
            {"match_id": "M001", "market_type": "moneyline", "team": "South Africa", "line": None, "probability": 0.105},
            {"match_id": "M001", "market_type": "total", "team": "Over", "line": 1.5, "probability": 0.695},
            {"match_id": "M001", "market_type": "team_total", "team": "Mexico", "line": 1.5, "probability": 0.405},
            {"match_id": "M001", "market_type": "spread", "team": "Mexico", "line": -1.5, "probability": 0.405},
        ]
    )
    metrics = evaluate_market_snapshots(snapshots, schedule)
    assert metrics["matches_with_moneyline"] == 1
    assert metrics["moneyline_accuracy"] == 1
    assert metrics["total_lines"] == 1
    assert metrics["team_total_lines"] == 1
    assert metrics["spread_lines"] == 1
