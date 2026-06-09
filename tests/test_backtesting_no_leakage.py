import pandas as pd

from src.models.backtesting import simple_backtest


def test_backtesting_uses_prior_rows_only_and_runs():
    rows = []
    for i in range(25):
        rows.append({"date": f"2024-01-{i+1:02d}", "home_score": i % 3, "away_score": (i + 1) % 3})
    metrics = simple_backtest(pd.DataFrame(rows))
    assert metrics["matches"] == 15
    assert "accuracy" in metrics

