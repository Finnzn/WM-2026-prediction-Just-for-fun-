import pandas as pd

from src.data_sources.market_snapshots import snapshot_rows_from_prediction, upsert_market_snapshot


def test_snapshot_rows_from_prediction_extracts_all_market_types():
    prediction = {
        "match_id": "M003",
        "market_timestamp": "2026-06-12T08:00:00+00:00",
        "moneyline_raw_prices": '{"Canada": 0.5, "Draw": 0.25, "Bosnia and Herzegovina": 0.25}',
        "total_lines_used": '[{"line": 2.5, "over_price": 0.44}]',
        "team_total_lines_used": '[{"team": "Canada", "line": 1.5, "over_price": 0.48}]',
        "spread_lines_used": '[{"team": "Canada", "line": -1.5, "price": 0.22}]',
    }
    rows = snapshot_rows_from_prediction(prediction)
    assert len(rows) == 6
    assert {row["market_type"] for row in rows} == {"moneyline", "total", "team_total", "spread"}
    assert rows[0]["snapshot_date"] == "2026-06-12"


def test_upsert_market_snapshot_replaces_existing_match_rows(tmp_path):
    path = tmp_path / "market_snapshots.csv"
    first = [
        {"match_id": "M003", "snapshot_date": "2026-06-12", "market_type": "moneyline", "team": "Canada", "line": "", "probability": 0.5},
        {"match_id": "M004", "snapshot_date": "2026-06-12", "market_type": "moneyline", "team": "United States", "line": "", "probability": 0.6},
    ]
    assert upsert_market_snapshot(path, "M003", first) == 2
    second = [
        {"match_id": "M003", "snapshot_date": "2026-06-12", "market_type": "moneyline", "team": "Canada", "line": "", "probability": 0.55}
    ]
    assert upsert_market_snapshot(path, "M003", second) == 1
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert len(df[df["match_id"].eq("M003")]) == 1
    assert len(df[df["match_id"].eq("M004")]) == 1
    assert df[df["match_id"].eq("M003")].iloc[0]["probability"] == "0.55"
