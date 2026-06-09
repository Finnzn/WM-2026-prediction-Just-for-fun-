from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import normalize_probabilities, safe_float, snake_case


def load_manual_market_probs(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [snake_case(column) for column in df.columns]
    return df


def manual_probs_for_match(df: pd.DataFrame, match_id: str) -> dict | None:
    if df.empty or "match_id" not in df.columns:
        return None
    rows = df[df["match_id"].eq(match_id)]
    if rows.empty:
        return None
    row = rows.iloc[-1]
    probs = [
        safe_float(row.get("home_win_prob"), 0.0) or 0.0,
        safe_float(row.get("draw_prob"), 0.0) or 0.0,
        safe_float(row.get("away_win_prob"), 0.0) or 0.0,
    ]
    if sum(probs) <= 0:
        return None
    home, draw, away = normalize_probabilities(probs)
    return {
        "source": row.get("source", "manual_market_probs"),
        "updated_at": row.get("updated_at", ""),
        "raw": {"home_win": probs[0], "draw": probs[1], "away_win": probs[2]},
        "normalized": {"home_win": home, "draw": draw, "away_win": away},
    }

