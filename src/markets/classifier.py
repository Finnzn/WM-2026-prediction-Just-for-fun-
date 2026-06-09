from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from src.utils import safe_float


@dataclass
class MarketCandidate:
    title: str
    category: str
    market_type: str
    confidence: float
    reasons: list[str]
    raw: dict[str, Any]
    spread_line: float | None = None
    total_line: float | None = None


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def classify_market(title: str, outcomes: list[str] | None = None) -> tuple[str, str]:
    text = f"{title} {' '.join(outcomes or [])}".lower()
    if any(word in text for word in ["win the world cup", "win the 2026 fifa world cup", "winner", "champion", "lift the trophy"]):
        return "futures", "futures_team"
    if any(word in text for word in ["total", "over", "under", "goals"]):
        if any(word in text for word in ["over", "under"]):
            return "total", "total_goals"
    if any(word in text for word in ["spread", "handicap"]) or re.search(r"(?<![a-z0-9])[+-]\d+(?:\.\d+)?", text):
        return "spread", "spread"
    if "draw" in text and any(word in text for word in ["win", "moneyline", "vs", " v "]):
        return "moneyline", "three_way_moneyline"
    if any(word in text for word in ["beat", "to win", "win?"]):
        return "moneyline", "binary_home_win"
    return "unknown", "unknown"


def parse_spread_line(text: str) -> float | None:
    match = re.search(r"([+-]\d+(?:\.\d+)?)", text)
    return safe_float(match.group(1)) if match else None


def parse_total_line(text: str) -> float | None:
    match = re.search(r"(?:over|under|o|u)\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if match:
        return safe_float(match.group(1))
    match = re.search(r"total(?: goals)?\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    return safe_float(match.group(1)) if match else None


def score_candidate(raw: dict[str, Any], home_team: str, away_team: str, match_date: str = "") -> MarketCandidate:
    title = str(raw.get("title") or raw.get("question") or raw.get("slug") or "")
    event_title = str(raw.get("eventTitle") or raw.get("event_title") or "")
    combined = f"{title} {event_title}"
    outcomes = raw.get("outcomes") if isinstance(raw.get("outcomes"), list) else []
    category, market_type = classify_market(combined, [str(item) for item in outcomes])
    reasons: list[str] = []
    confidence = 0.0
    text = combined.lower()
    home_hit = home_team.lower() in text or _ratio(home_team, combined) > 0.55
    away_hit = away_team.lower() in text or _ratio(away_team, combined) > 0.55
    if home_hit:
        confidence += 0.25
        reasons.append("home team appears or fuzzy-matches")
    if away_hit:
        confidence += 0.25
        reasons.append("away team appears or fuzzy-matches")
    if home_hit and away_hit:
        confidence += 0.20
        reasons.append("both teams matched")
    if category != "unknown":
        confidence += 0.15
        reasons.append(f"classified as {category}")
    if str(raw.get("active", raw.get("closed", ""))).lower() in {"true", "active"}:
        confidence += 0.05
        reasons.append("market appears active")
    if match_date and match_date in combined:
        confidence += 0.05
        reasons.append("date appears in title/event")
    line_text = f"{combined} {' '.join(str(item) for item in outcomes)}"
    spread_line = parse_spread_line(line_text) if category == "spread" else None
    total_line = parse_total_line(line_text) if category == "total" else None
    if category == "spread" and spread_line is not None:
        confidence += 0.05
        reasons.append("spread line parsed")
    if category == "total" and total_line is not None:
        confidence += 0.05
        reasons.append("total line parsed")
    return MarketCandidate(title=title, category=category, market_type=market_type, confidence=min(confidence, 1.0), reasons=reasons, raw=raw, spread_line=spread_line, total_line=total_line)
