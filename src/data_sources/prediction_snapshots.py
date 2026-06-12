from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


COLUMNS = [
    "match_id",
    "snapshot_date",
    "home_team",
    "away_team",
    "predicted_home_score",
    "predicted_away_score",
    "predicted_score",
    "home_win_prob",
    "draw_prob",
    "away_win_prob",
    "model_weight",
    "market_weight",
]


def prediction_snapshot_row(prediction: dict[str, Any], snapshot_date: str | None = None) -> dict[str, Any]:
    date_value = snapshot_date or _snapshot_date(prediction)
    home_goals = int(prediction.get("predicted_home_goals", 0))
    away_goals = int(prediction.get("predicted_away_goals", 0))
    return {
        "match_id": str(prediction.get("match_id", "")),
        "snapshot_date": date_value,
        "home_team": str(prediction.get("home_team", "")),
        "away_team": str(prediction.get("away_team", "")),
        "predicted_home_score": home_goals,
        "predicted_away_score": away_goals,
        "predicted_score": f"{home_goals}-{away_goals}",
        "home_win_prob": prediction.get("home_win_prob", ""),
        "draw_prob": prediction.get("draw_prob", ""),
        "away_win_prob": prediction.get("away_win_prob", ""),
        "model_weight": prediction.get("model_weight", ""),
        "market_weight": prediction.get("moneyline_market_weight", ""),
    }


def upsert_prediction_snapshot(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    match_id = str(row.get("match_id", ""))
    if path.exists():
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        for column in COLUMNS:
            if column not in existing.columns:
                existing[column] = ""
        existing = existing[COLUMNS]
        existing = existing[~existing["match_id"].eq(match_id)].copy()
    else:
        existing = pd.DataFrame(columns=COLUMNS)
    addition = pd.DataFrame([row], columns=COLUMNS)
    combined = addition if existing.empty else pd.concat([existing, addition], ignore_index=True, sort=False)
    combined.to_csv(path, index=False)


def load_prediction_snapshots(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[COLUMNS]


def prediction_overview(schedule: pd.DataFrame, snapshots: pd.DataFrame) -> dict[str, Any]:
    if snapshots.empty:
        return {"rows": [], "summary": _summary([])}
    latest = snapshots.drop_duplicates("match_id", keep="last")
    merged = schedule.merge(latest, on="match_id", how="left", suffixes=("", "_prediction"))
    rows = [_overview_row(row) for row in merged.itertuples(index=False) if _has_value(getattr(row, "predicted_score", ""))]
    return {"rows": rows, "summary": _summary(rows)}


def _overview_row(row: Any) -> dict[str, Any]:
    status = str(getattr(row, "status", ""))
    pred_home = _safe_int(getattr(row, "predicted_home_score", ""))
    pred_away = _safe_int(getattr(row, "predicted_away_score", ""))
    actual_home = _safe_int(getattr(row, "home_score", ""))
    actual_away = _safe_int(getattr(row, "away_score", ""))
    played = status == "played" and actual_home is not None and actual_away is not None
    correct_score = played and pred_home == actual_home and pred_away == actual_away
    correct_winner = played and _outcome(pred_home, pred_away) == _outcome(actual_home, actual_away)
    correct_goal_diff = played and pred_home is not None and pred_away is not None and (pred_home - pred_away) == (actual_home - actual_away)
    goal_error = "" if not played or pred_home is None or pred_away is None else abs(pred_home - actual_home) + abs(pred_away - actual_away)
    return {
        "match_id": str(getattr(row, "match_id", "")),
        "date": str(getattr(row, "date", "")),
        "stage": str(getattr(row, "stage", "")),
        "home_team": str(getattr(row, "home_team", "")),
        "away_team": str(getattr(row, "away_team", "")),
        "status": status,
        "prediction": str(getattr(row, "predicted_score", "")),
        "actual": f"{actual_home}-{actual_away}" if played else "",
        "correct_score": bool(correct_score),
        "correct_winner": bool(correct_winner),
        "correct_goal_diff": bool(correct_goal_diff),
        "goal_error": goal_error,
        "home_win_prob": str(getattr(row, "home_win_prob", "")),
        "draw_prob": str(getattr(row, "draw_prob", "")),
        "away_win_prob": str(getattr(row, "away_win_prob", "")),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    played = [row for row in rows if row["actual"]]
    return {
        "predictions": len(rows),
        "played_predictions": len(played),
        "correct_scores": sum(1 for row in played if row["correct_score"]),
        "correct_winners": sum(1 for row in played if row["correct_winner"]),
        "correct_goal_diffs": sum(1 for row in played if row["correct_goal_diff"]),
        "average_goal_error": round(sum(float(row["goal_error"]) for row in played) / len(played), 3) if played else "",
    }


def _snapshot_date(prediction: dict[str, Any]) -> str:
    timestamp = str(prediction.get("market_timestamp") or "")
    if timestamp:
        parsed = pd.to_datetime(timestamp, errors="coerce")
        if not pd.isna(parsed):
            return parsed.date().isoformat()
    return pd.Timestamp.utcnow().date().isoformat()


def _safe_int(value: Any) -> int | None:
    parsed = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(parsed) else int(parsed)


def _has_value(value: Any) -> bool:
    return not pd.isna(value) and str(value).strip() != ""


def _outcome(home_score: int | None, away_score: int | None) -> str:
    if home_score is None or away_score is None:
        return ""
    if home_score > away_score:
        return "home"
    if home_score < away_score:
        return "away"
    return "draw"
