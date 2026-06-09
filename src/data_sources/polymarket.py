from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any

import requests

from src.config import Config
from src.markets.classifier import MarketCandidate, score_candidate
from src.utils import ensure_parent, normalize_probabilities, safe_float


TEAM_ALIASES = {
    "Bosnia and Herzegovina": ["Bosnia and Herzegovina", "Bosnia-Herzegovina", "BIH"],
    "Cape Verde": ["Cape Verde", "Cabo Verde", "CVI"],
    "Cote d'Ivoire": ["Cote d'Ivoire", "Côte d'Ivoire", "Ivory Coast", "CIV"],
    "Curaçao": ["Curaçao", "Curacao", "CUW", "KOR"],
    "Czechia": ["Czechia", "Czech Republic", "CZE"],
    "DR Congo": ["DR Congo", "Congo DR", "COD"],
    "South Africa": ["South Africa", "RSA"],
    "South Korea": ["South Korea", "Korea Republic", "KR", "KOR"],
    "Turkey": ["Turkey", "Türkiye", "TUR"],
    "United States": ["United States", "USA"],
}

TEAM_CODES = {
    "Algeria": "ALG",
    "Argentina": "ARG",
    "Australia": "AUS",
    "Austria": "AUT",
    "Belgium": "BEL",
    "Brazil": "BRA",
    "Canada": "CAN",
    "Colombia": "COL",
    "Croatia": "CRO",
    "Czechia": "CZE",
    "Denmark": "DEN",
    "Ecuador": "ECU",
    "Egypt": "EGY",
    "England": "ENG",
    "France": "FRA",
    "Germany": "GER",
    "Ghana": "GHA",
    "Haiti": "HAI",
    "Iran": "IRN",
    "Iraq": "IRQ",
    "Japan": "JPN",
    "Jordan": "JOR",
    "Mexico": "MEX",
    "Morocco": "MAR",
    "Netherlands": "NLD",
    "New Zealand": "NZL",
    "Norway": "NOR",
    "Panama": "PAN",
    "Paraguay": "PAR",
    "Portugal": "POR",
    "Qatar": "QAT",
    "Saudi Arabia": "KSA",
    "Scotland": "SCO",
    "Senegal": "SEN",
    "South Africa": "RSA",
    "South Korea": "KR",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "Tunisia": "TUN",
    "Turkey": "TUR",
    "United States": "USA",
    "Uruguay": "URY",
    "Uzbekistan": "UZB",
}


class PolymarketClient:
    """Read-only public Polymarket helper. No wallet, auth, trading, or orders."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cache_path = Path("data/processed/market_cache.json")
        self.page_cache_path = Path("data/processed/polymarket_worldcup_games_page.txt")

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

    def fetch_worldcup_games_text(self, refresh: bool = False) -> str:
        if not refresh and self.page_cache_path.exists():
            try:
                return self.page_cache_path.read_text()
            except Exception:
                pass
        response = requests.get(self.cfg.polymarket_worldcup_games_url, timeout=20)
        response.raise_for_status()
        text = html_to_text(response.text)
        ensure_parent(self.page_cache_path)
        self.page_cache_path.write_text(text)
        return text

    def scrape_worldcup_moneyline(
        self,
        home_team: str,
        away_team: str,
        refresh: bool = False,
    ) -> dict[str, Any] | None:
        try:
            text = self.fetch_worldcup_games_text(refresh=refresh)
        except Exception:
            return None
        scraped = extract_worldcup_games_page_moneyline(text, home_team, away_team)
        if not scraped:
            return None
        scraped["timestamp"] = datetime.now(timezone.utc).isoformat()
        scraped["age_minutes"] = 0.0
        scraped["confidence"] = 0.95
        scraped["market_type"] = "three_way_moneyline"
        scraped["source"] = "polymarket_worldcup_games_page"
        scraped["title"] = f"{home_team} vs {away_team}"
        scraped["slug"] = self.cfg.polymarket_worldcup_games_url
        return scraped

    def best_moneyline_for_match(
        self,
        home_team: str,
        away_team: str,
        match_date: str,
        refresh: bool = False,
    ) -> dict[str, Any] | None:
        scraped = self.scrape_worldcup_moneyline(home_team, away_team, refresh=refresh)
        if scraped:
            return scraped

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


def html_to_text(html: str) -> str:
    text = re.sub(r"<script\b.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def aliases_for_team(team: str) -> list[str]:
    aliases = [team]
    aliases.extend(TEAM_ALIASES.get(team, []))
    code = TEAM_CODES.get(team)
    if code:
        aliases.append(code)
    deduped: list[str] = []
    for alias in aliases:
        if alias and alias not in deduped:
            deduped.append(alias)
    return deduped


def _price_after_label(text: str, label: str) -> float | None:
    pattern = rf"\b{re.escape(label)}\b\s+(\d+(?:\.\d+)?)¢"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return safe_float(match.group(1)) if match else None


def extract_worldcup_games_page_moneyline(text: str, home_team: str, away_team: str) -> dict[str, Any] | None:
    home_aliases = aliases_for_team(home_team)
    away_aliases = aliases_for_team(away_team)
    for home_alias in home_aliases:
        for away_alias in away_aliases:
            if home_alias.lower() not in text.lower() or away_alias.lower() not in text.lower():
                continue
            home_pos = text.lower().find(home_alias.lower())
            away_pos = text.lower().find(away_alias.lower(), max(0, home_pos - 80))
            if home_pos == -1 or away_pos == -1 or abs(home_pos - away_pos) > 250:
                continue
            start = max(0, min(home_pos, away_pos) - 120)
            end = min(len(text), max(home_pos, away_pos) + 260)
            window = text[start:end]
            home_labels = [TEAM_CODES.get(home_team, ""), home_team, home_alias]
            away_labels = [TEAM_CODES.get(away_team, ""), away_team, away_alias]
            home_price = next((value for label in home_labels if label for value in [_price_after_label(window, label)] if value is not None), None)
            draw_price = _price_after_label(window, "Draw")
            away_price = next((value for label in away_labels if label for value in [_price_after_label(window, label)] if value is not None), None)
            if home_price is None or draw_price is None or away_price is None:
                continue
            home, draw, away = normalize_probabilities([home_price, draw_price, away_price])
            return {
                "raw": {"home_win": home_price / 100, "draw": draw_price / 100, "away_win": away_price / 100},
                "normalized": {"home_win": home, "draw": draw, "away_win": away},
                "window": window,
            }
    return None
