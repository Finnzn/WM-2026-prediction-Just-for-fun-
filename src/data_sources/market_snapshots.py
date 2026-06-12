from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


COLUMNS = ["match_id", "snapshot_date", "market_type", "team", "line", "probability"]


def snapshot_rows_from_prediction(prediction: dict[str, Any], snapshot_date: str | None = None) -> list[dict[str, Any]]:
    match_id = str(prediction.get("match_id", ""))
    if not match_id:
        return []
    date_value = snapshot_date or _snapshot_date(prediction)
    rows: list[dict[str, Any]] = []

    raw_moneyline = _parse_jsonish(prediction.get("moneyline_raw_prices"), {})
    if isinstance(raw_moneyline, dict):
        for team, probability in raw_moneyline.items():
            rows.append(_row(match_id, date_value, "moneyline", str(team), "", probability))

    for item in _parse_jsonish(prediction.get("total_lines_used"), []):
        if isinstance(item, dict):
            rows.append(_row(match_id, date_value, "total", "Over", item.get("line", ""), item.get("over_price", item.get("over_probability"))))

    for item in _parse_jsonish(prediction.get("team_total_lines_used"), []):
        if isinstance(item, dict):
            rows.append(_row(match_id, date_value, "team_total", str(item.get("team", "")), item.get("line", ""), item.get("over_price", item.get("over_probability"))))

    for item in _parse_jsonish(prediction.get("spread_lines_used"), []):
        if isinstance(item, dict):
            rows.append(_row(match_id, date_value, "spread", str(item.get("team", "")), item.get("line", ""), item.get("price", item.get("cover_probability"))))

    return [row for row in rows if row["probability"] != ""]


def upsert_market_snapshot(path: Path, match_id: str, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        for column in COLUMNS:
            if column not in existing.columns:
                existing[column] = ""
        existing = existing[COLUMNS]
        existing = existing[~existing["match_id"].eq(str(match_id))].copy()
    else:
        existing = pd.DataFrame(columns=COLUMNS)
    additions = pd.DataFrame(rows, columns=COLUMNS) if rows else pd.DataFrame(columns=COLUMNS)
    if existing.empty:
        combined = additions
    elif additions.empty:
        combined = existing
    else:
        combined = pd.concat([existing, additions], ignore_index=True, sort=False)
    combined.to_csv(path, index=False)
    return len(additions)


def _snapshot_date(prediction: dict[str, Any]) -> str:
    timestamp = str(prediction.get("market_timestamp") or "")
    if timestamp:
        parsed = pd.to_datetime(timestamp, errors="coerce")
        if not pd.isna(parsed):
            return parsed.date().isoformat()
    return pd.Timestamp.utcnow().date().isoformat()


def _row(match_id: str, snapshot_date: str, market_type: str, team: str, line: Any, probability: Any) -> dict[str, Any]:
    prob = pd.to_numeric(probability, errors="coerce")
    return {
        "match_id": match_id,
        "snapshot_date": snapshot_date,
        "market_type": market_type,
        "team": team,
        "line": "" if line is None else line,
        "probability": "" if pd.isna(prob) else float(prob),
    }


def _parse_jsonish(value: Any, fallback: Any) -> Any:
    if value in {None, ""}:
        return fallback
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback
