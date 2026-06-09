from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from src.config import Config
from src.markets.classifier import MarketCandidate, score_candidate
from src.utils import ensure_parent, normalize_probabilities, safe_float


class PolymarketClient:
    """Read-only public Polymarket helper. No wallet, auth, trading, or orders."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cache_path = Path("data/processed/market_cache.json")

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        url = f"{self.cfg.polymarket_gamma_base_url.rstrip('/')}/{path.lstrip('/')}"
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()

    def search_markets(self, query: str, refresh: bool = False) -> list[dict[str, Any]]:
        if not refresh:
            cached = self._read_cache(query)
            if cached is not None:
                return cached
        try:
            data = self._get("/markets", {"search": query, "limit": 25})
        except Exception:
            return []
        rows = data if isinstance(data, list) else data.get("markets", data.get("data", []))
        if isinstance(rows, list):
            self._write_cache(query, rows)
            return [row for row in rows if isinstance(row, dict)]
        return []

    def candidates_for_match(self, home_team: str, away_team: str, match_date: str, refresh: bool = False) -> list[MarketCandidate]:
        queries = [
            f"{home_team} {away_team} World Cup 2026",
            f"{home_team} vs {away_team}",
            f"{away_team} vs {home_team}",
            f"FIFA World Cup {match_date}",
        ]
        seen: set[str] = set()
        candidates: list[MarketCandidate] = []
        for query in queries:
            for raw in self.search_markets(query, refresh=refresh):
                slug = str(raw.get("slug") or raw.get("id") or raw.get("question") or "")
                if slug in seen:
                    continue
                seen.add(slug)
                candidates.append(score_candidate(raw, home_team, away_team, match_date))
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        return candidates

    def best_moneyline_for_match(
        self,
        home_team: str,
        away_team: str,
        match_date: str,
        refresh: bool = False,
    ) -> dict[str, Any] | None:
        candidates = self.candidates_for_match(home_team, away_team, match_date, refresh=refresh)
        for candidate in candidates:
            if candidate.confidence < self.cfg.polymarket_match_confidence_threshold:
                continue
            if candidate.category != "moneyline" or candidate.market_type != "three_way_moneyline":
                continue
            extracted = extract_three_way_moneyline(candidate.raw, home_team, away_team)
            if extracted:
                extracted["confidence"] = candidate.confidence
                extracted["market_type"] = candidate.market_type
                extracted["title"] = candidate.title
                extracted["slug"] = candidate.raw.get("slug", "")
                extracted["timestamp"] = datetime.now(timezone.utc).isoformat()
                extracted["age_minutes"] = 0.0
                return extracted
        return None

    def _read_cache(self, query: str) -> list[dict[str, Any]] | None:
        if not self.cache_path.exists():
            return None
        try:
            cache = json.loads(self.cache_path.read_text())
        except Exception:
            return None
        item = cache.get(query)
        if not item:
            return None
        timestamp = datetime.fromisoformat(item["timestamp"])
        age_minutes = (datetime.now(timezone.utc) - timestamp).total_seconds() / 60
        if age_minutes > self.cfg.polymarket_cache_minutes:
            return None
        return item.get("data", [])

    def _write_cache(self, query: str, data: list[dict[str, Any]]) -> None:
        ensure_parent(self.cache_path)
        try:
            cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        except Exception:
            cache = {}
        cache[query] = {"timestamp": datetime.now(timezone.utc).isoformat(), "data": data}
        self.cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def _jsonish(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def extract_three_way_moneyline(raw: dict[str, Any], home_team: str, away_team: str) -> dict[str, Any] | None:
    outcomes = _jsonish(raw.get("outcomes", []))
    prices = _jsonish(raw.get("outcomePrices", raw.get("outcome_prices", [])))
    if not isinstance(outcomes, list) or not isinstance(prices, list) or len(outcomes) != len(prices):
        return None
    outcome_prices: dict[str, float] = {}
    for outcome, price in zip(outcomes, prices):
        value = safe_float(price)
        if value is None:
            continue
        text = str(outcome).strip().lower()
        if home_team.lower() in text or text in {"home", "home win"}:
            outcome_prices["home_win"] = value
        elif away_team.lower() in text or text in {"away", "away win"}:
            outcome_prices["away_win"] = value
        elif text in {"draw", "tie"}:
            outcome_prices["draw"] = value
    if {"home_win", "draw", "away_win"} - set(outcome_prices):
        return None
    home, draw, away = normalize_probabilities([
        outcome_prices["home_win"],
        outcome_prices["draw"],
        outcome_prices["away_win"],
    ])
    return {
        "raw": outcome_prices,
        "normalized": {"home_win": home, "draw": draw, "away_win": away},
        "source": "polymarket_gamma_public",
    }
