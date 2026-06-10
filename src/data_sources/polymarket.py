from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any

import requests

from src.config import Config
from src.markets.classifier import classify_market, parse_spread_line, parse_total_line
from src.utils import normalize_probabilities, safe_float


SPORT_HINTS = {"soccer", "football", "world cup", "fifa", "fifwc"}
MAX_CLOB_SPREAD = 0.15
FIFA_WORLD_CUP_SERIES_ID = "11433"

TEAM_ALIASES = {
    "Bosnia and Herzegovina": ["Bosnia and Herzegovina", "Bosnia-Herzegovina", "BIH"],
    "Cape Verde": ["Cape Verde", "Cabo Verde", "CVI"],
    "Cote d'Ivoire": ["Cote d'Ivoire", "Côte d'Ivoire", "Ivory Coast", "CIV"],
    "Curaçao": ["Curaçao", "Curacao", "CUW"],
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
    "South Korea": "KOR",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "Tunisia": "TUN",
    "Turkey": "TUR",
    "United States": "USA",
    "Uruguay": "URY",
    "Uzbekistan": "UZB",
}

POLYMARKET_TEAM_CODES = {
    "South Korea": "KR",
}


@dataclass
class OutcomeToken:
    outcome: str
    gamma_price: float | None
    token_id: str
    bid: float | None = None
    ask: float | None = None
    midpoint: float | None = None
    spread: float | None = None
    last_trade_price: float | None = None
    implied_probability: float | None = None
    price_source: str = ""
    book_url: str = ""
    book_error: str = ""


@dataclass
class PolymarketDebugCandidate:
    event_title: str
    event_slug: str
    market_question: str
    market_slug: str
    category: str
    market_type: str
    fuzzy_score: float
    reasons: list[str]
    accepted: bool
    rejected_reason: str
    outcomes: list[str]
    gamma_prices: list[float | None]
    clob_token_ids: list[str]
    tokens: list[OutcomeToken] = field(default_factory=list)
    spread_line: float | None = None
    total_line: float | None = None


@dataclass
class PolymarketDebugReport:
    request_urls: list[str] = field(default_factory=list)
    events_fetched: int = 0
    markets_inspected: int = 0
    tags_seen: list[str] = field(default_factory=list)
    sports_seen: list[str] = field(default_factory=list)
    candidates: list[PolymarketDebugCandidate] = field(default_factory=list)


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


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


def norm(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def best_alias_score(team: str, text: str) -> tuple[float, str]:
    best_score = 0.0
    best_alias = ""
    text_norm = norm(text)
    for alias in aliases_for_team(team):
        alias_norm = norm(alias)
        if not alias_norm:
            continue
        if re.search(rf"\b{re.escape(alias_norm)}\b", text_norm):
            return 1.0, alias
        score = ratio(alias_norm, text_norm)
        if score > best_score:
            best_score = score
            best_alias = alias
    return best_score, best_alias


def event_tags(event: dict[str, Any]) -> list[str]:
    rows = event.get("tags") or []
    tags: list[str] = []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                for key in ("label", "slug"):
                    if row.get(key):
                        tags.append(str(row[key]))
            elif row:
                tags.append(str(row))
    return tags


def event_looks_sport_relevant(event: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(event.get("title", "")),
            str(event.get("slug", "")),
            " ".join(event_tags(event)),
        ]
    ).lower()
    return any(hint in text for hint in SPORT_HINTS)


def market_text(event: dict[str, Any], market: dict[str, Any]) -> str:
    return " ".join(
        [
            str(event.get("title", "")),
            str(event.get("slug", "")),
            str(market.get("question", "")),
            str(market.get("slug", "")),
            str(market.get("description", "")),
        ]
    )


def compact_market_text(event: dict[str, Any], market: dict[str, Any]) -> str:
    return " ".join(
        [
            str(event.get("title", "")),
            str(event.get("slug", "")),
            str(market.get("question", "")),
            str(market.get("slug", "")),
        ]
    )


def is_full_match_total(candidate: PolymarketDebugCandidate) -> bool:
    text = norm(f"{candidate.market_question} {candidate.market_slug}")
    return (
        candidate.category == "total"
        and candidate.total_line is not None
        and "first half" not in text
        and "1st half" not in text
        and "team total" not in text
        and "corners" not in text
    )


def is_full_match_spread(candidate: PolymarketDebugCandidate) -> bool:
    text = norm(f"{candidate.market_question} {candidate.market_slug}")
    return (
        candidate.category == "spread"
        and candidate.spread_line is not None
        and "first half" not in text
        and "1st half" not in text
        and "corners" not in text
    )


def question_matches_team_win(question: str, team: str) -> bool:
    text = norm(question)
    if " win " not in f" {text} ":
        return False
    for alias in aliases_for_team(team):
        alias_norm = norm(alias)
        if alias_norm and re.search(rf"\b{re.escape(alias_norm)}\b", text):
            return True
    return False


def classify_match_market(event: dict[str, Any], market: dict[str, Any], home_team: str, away_team: str) -> tuple[str, str]:
    question = str(market.get("question", ""))
    slug = str(market.get("slug", ""))
    text = f"{question} {slug}"
    text_norm = norm(text)
    if "win the 2026 fifa world cup" in text_norm or "win group" in text_norm:
        return "futures", "futures_team"
    if "end in a draw" in text_norm or re.search(r"\bdraw\b", text_norm):
        return "moneyline", "binary_draw"
    if question_matches_team_win(question, home_team):
        return "moneyline", "binary_home_win"
    if question_matches_team_win(question, away_team):
        return "moneyline", "binary_away_win"
    return classify_market(compact_market_text(event, market), parse_jsonish(market.get("outcomes", [])))


def score_market_match(event: dict[str, Any], market: dict[str, Any], home_team: str, away_team: str, match_date: str = "") -> tuple[float, list[str]]:
    text = market_text(event, market)
    home_score, home_alias = best_alias_score(home_team, text)
    away_score, away_alias = best_alias_score(away_team, text)
    score = 0.0
    reasons: list[str] = []
    if home_score >= 1.0:
        score += 0.30
        reasons.append(f"home matched as {home_alias}")
    else:
        score += min(home_score, 0.60) * 0.20
    if away_score >= 1.0:
        score += 0.30
        reasons.append(f"away matched as {away_alias}")
    else:
        score += min(away_score, 0.60) * 0.20
    if home_score >= 1.0 and away_score >= 1.0:
        score += 0.20
        reasons.append("both teams matched")
    if match_date and match_date in text:
        score += 0.05
        reasons.append("match date matched")
    if event_looks_sport_relevant(event):
        score += 0.05
        reasons.append("event tags/title look sport relevant")
    if bool(market.get("active")) and not bool(market.get("closed")):
        score += 0.05
        reasons.append("market active and open")
    if bool(market.get("acceptingOrders", True)):
        score += 0.03
        reasons.append("market accepts orders")
    return min(score, 1.0), reasons


def extract_outcome_tokens(market: dict[str, Any]) -> tuple[list[OutcomeToken], list[str], list[float | None], list[str], str]:
    outcomes_raw = parse_jsonish(market.get("outcomes", []))
    prices_raw = parse_jsonish(market.get("outcomePrices", market.get("outcome_prices", [])))
    token_ids_raw = parse_jsonish(market.get("clobTokenIds", market.get("clob_token_ids", [])))
    if not isinstance(outcomes_raw, list):
        return [], [], [], [], "outcomes not list"
    if not isinstance(prices_raw, list):
        prices_raw = []
    if not isinstance(token_ids_raw, list):
        return [], [str(item) for item in outcomes_raw], [], [], "clobTokenIds not list"
    outcomes = [str(item) for item in outcomes_raw]
    gamma_prices = [safe_float(item) for item in prices_raw]
    token_ids = [str(item) for item in token_ids_raw]
    if len(outcomes) != len(token_ids):
        return [], outcomes, gamma_prices, token_ids, f"len(outcomes)={len(outcomes)} != len(clobTokenIds)={len(token_ids)}"
    while len(gamma_prices) < len(outcomes):
        gamma_prices.append(None)
    tokens = [
        OutcomeToken(outcome=outcome, gamma_price=gamma_prices[idx], token_id=token_ids[idx])
        for idx, outcome in enumerate(outcomes)
    ]
    return tokens, outcomes, gamma_prices[: len(outcomes)], token_ids, ""


def best_order(orders: Any, side: str) -> float | None:
    if not isinstance(orders, list) or not orders:
        return None
    values = []
    for order in orders:
        if isinstance(order, dict):
            value = safe_float(order.get("price"))
        else:
            value = None
        if value is not None:
            values.append(value)
    if not values:
        return None
    return max(values) if side == "bid" else min(values)


def classify_outcome(outcome: str, home_team: str, away_team: str) -> str:
    text = norm(outcome)
    if text in {"draw", "tie"}:
        return "draw"
    for alias in aliases_for_team(home_team):
        if norm(alias) and re.search(rf"\b{re.escape(norm(alias))}\b", text):
            return "home_win"
    for alias in aliases_for_team(away_team):
        if norm(alias) and re.search(rf"\b{re.escape(norm(alias))}\b", text):
            return "away_win"
    if text in {"home", "home win"}:
        return "home_win"
    if text in {"away", "away win"}:
        return "away_win"
    return ""


class PolymarketClient:
    """Read-only public Polymarket helper. No wallet, auth, trading, or orders."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session = requests.Session()

    def _gamma_get(self, path: str, params: dict[str, Any], debug: PolymarketDebugReport | None = None) -> Any:
        url = f"{self.cfg.polymarket_gamma_base_url.rstrip('/')}/{path.lstrip('/')}"
        response = self.session.get(url, params=params, timeout=20)
        if debug is not None:
            debug.request_urls.append(response.url)
        response.raise_for_status()
        return response.json()

    def _clob_get(self, path: str, params: dict[str, Any], debug: PolymarketDebugReport | None = None) -> Any:
        url = f"{self.cfg.polymarket_clob_base_url.rstrip('/')}/{path.lstrip('/')}"
        response = self.session.get(url, params=params, timeout=20)
        if debug is not None:
            debug.request_urls.append(response.url)
        response.raise_for_status()
        return response.json()

    def discover_tags_and_sports(self, debug: PolymarketDebugReport | None = None) -> None:
        if debug is None:
            return
        try:
            tags = self._gamma_get("/tags", {"limit": 500}, debug)
            if isinstance(tags, list):
                debug.tags_seen = [
                    str(item.get("slug") or item.get("label"))
                    for item in tags
                    if isinstance(item, dict)
                    and any(hint in f"{item.get('slug', '')} {item.get('label', '')}".lower() for hint in SPORT_HINTS)
                ][:50]
        except Exception as exc:
            debug.tags_seen = [f"tags request failed: {exc}"]
        try:
            sports = self._gamma_get("/sports", {"limit": 500}, debug)
            if isinstance(sports, list):
                debug.sports_seen = [
                    str(item.get("sport"))
                    for item in sports
                    if isinstance(item, dict)
                    and any(hint in f"{item.get('sport', '')} {item.get('resolution', '')}".lower() for hint in SPORT_HINTS)
                ][:50]
        except Exception as exc:
            debug.sports_seen = [f"sports request failed: {exc}"]

    def fetch_active_events(
        self,
        debug: PolymarketDebugReport | None = None,
        limit: int = 100,
        max_pages: int = 30,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for page in range(max_pages):
            offset = page * limit
            params = {"active": "true", "closed": "false", "limit": limit, "offset": offset}
            rows = self._gamma_get("/events", params, debug)
            if not isinstance(rows, list) or not rows:
                break
            events.extend([row for row in rows if isinstance(row, dict)])
            if len(rows) < limit:
                break
        if debug is not None:
            debug.events_fetched = len(events)
        return events

    def fetch_worldcup_events(
        self,
        debug: PolymarketDebugReport | None = None,
        limit: int = 100,
        max_pages: int = 5,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for page in range(max_pages):
            offset = page * limit
            params = {
                "active": "true",
                "closed": "false",
                "series_id": FIFA_WORLD_CUP_SERIES_ID,
                "limit": limit,
                "offset": offset,
            }
            try:
                rows = self._gamma_get("/events", params, debug)
            except (requests.RequestException, json.JSONDecodeError, ValueError):
                break
            if not isinstance(rows, list) or not rows:
                break
            events.extend([row for row in rows if isinstance(row, dict)])
            if len(rows) < limit:
                break
        if debug is not None:
            debug.events_fetched = len(events)
        return events

    def fetch_event_by_slug(self, slug: str, debug: PolymarketDebugReport | None = None) -> dict[str, Any] | None:
        try:
            event = self._gamma_get(f"/events/slug/{slug}", {}, debug)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise
        return event if isinstance(event, dict) else None

    def event_slug_candidates(self, home_team: str, away_team: str, match_date: str) -> list[str]:
        if not match_date:
            return []
        dates = self._slug_dates(match_date)
        home_parts = self._slug_team_codes(home_team)
        away_parts = self._slug_team_codes(away_team)
        candidates: list[str] = []
        for date_part in dates:
            for home in home_parts:
                for away in away_parts:
                    slug = f"fifwc-{home}-{away}-{date_part}"
                    if slug not in candidates:
                        candidates.append(slug)
        return candidates

    def _slug_dates(self, match_date: str) -> list[str]:
        dates = [match_date]
        try:
            parsed = datetime.strptime(match_date, "%Y-%m-%d").date()
        except ValueError:
            return dates
        for candidate in [parsed - timedelta(days=1), parsed + timedelta(days=1)]:
            value = candidate.isoformat()
            if value not in dates:
                dates.append(value)
        return dates

    def _slug_team_codes(self, team: str) -> list[str]:
        parts: list[str] = []
        polymarket_code = POLYMARKET_TEAM_CODES.get(team)
        if polymarket_code:
            parts.append(polymarket_code.lower())
        code = TEAM_CODES.get(team)
        if code and code.lower() not in parts:
            parts.append(code.lower())
        for alias in aliases_for_team(team):
            slug = norm(alias).replace(" ", "-")
            if slug and slug not in parts:
                parts.append(slug)
        return parts

    def fetch_match_events(
        self,
        home_team: str,
        away_team: str,
        match_date: str,
        debug: PolymarketDebugReport,
        max_pages: int,
        refresh: bool,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        seen: set[str] = set()
        for slug in self.event_slug_candidates(home_team, away_team, match_date):
            event = self.fetch_event_by_slug(slug, debug)
            if event is not None:
                for candidate_slug in [slug, f"{slug}-more-markets"]:
                    candidate_event = event if candidate_slug == slug else self.fetch_event_by_slug(candidate_slug, debug)
                    if candidate_event is None:
                        continue
                    event_slug = str(candidate_event.get("slug", candidate_slug))
                    if event_slug not in seen:
                        seen.add(event_slug)
                        events.append(candidate_event)
                debug.events_fetched = len(events)
                return events
        events = self.fetch_worldcup_events(debug, max_pages=max_pages)
        return events

    def _priced_token(self, token: OutcomeToken, market: dict[str, Any], debug: PolymarketDebugReport | None) -> OutcomeToken:
        token.book_url = f"{self.cfg.polymarket_clob_base_url.rstrip()}/book?token_id={token.token_id}"
        try:
            book = self._clob_get("/book", {"token_id": token.token_id}, debug)
        except Exception as exc:
            token.book_error = str(exc)
            token.last_trade_price = safe_float(market.get("lastTradePrice"))
            if token.last_trade_price is not None:
                token.implied_probability = token.last_trade_price
                token.price_source = "gamma_last_trade_price_after_book_error"
            return token

        token.bid = best_order(book.get("bids"), "bid") if isinstance(book, dict) else None
        token.ask = best_order(book.get("asks"), "ask") if isinstance(book, dict) else None
        book_last_trade = book.get("lastTradePrice") if isinstance(book, dict) else None
        if book_last_trade is None and isinstance(book, dict):
            book_last_trade = book.get("last_trade_price")
        token.last_trade_price = safe_float(book_last_trade)
        if token.last_trade_price is None:
            token.last_trade_price = safe_float(market.get("lastTradePrice"))
        if token.bid is not None and token.ask is not None:
            token.midpoint = (token.bid + token.ask) / 2
            token.spread = token.ask - token.bid
            if token.spread <= MAX_CLOB_SPREAD:
                token.implied_probability = token.midpoint
                token.price_source = "clob_midpoint"
            elif token.last_trade_price is not None:
                token.implied_probability = token.last_trade_price
                token.price_source = "gamma_last_trade_price_wide_clob_spread"
            else:
                token.book_error = f"bid/ask spread too wide ({token.spread:.3f}) and no last trade price"
        elif token.last_trade_price is not None:
            token.implied_probability = token.last_trade_price
            token.price_source = "gamma_last_trade_price_missing_bid_or_ask"
        else:
            token.book_error = "missing bid/ask and last trade price"
        return token

    def discover_match_markets(
        self,
        home_team: str,
        away_team: str,
        match_date: str = "",
        max_pages: int = 30,
        include_debug_discovery: bool = False,
        refresh: bool = False,
    ) -> PolymarketDebugReport:
        debug = PolymarketDebugReport()
        if include_debug_discovery:
            self.discover_tags_and_sports(debug)
        events = self.fetch_match_events(home_team, away_team, match_date, debug, max_pages=max_pages, refresh=refresh)
        inspected = 0
        candidates: list[PolymarketDebugCandidate] = []
        for event in events:
            markets = event.get("markets") if isinstance(event.get("markets"), list) else []
            for market in markets:
                if not isinstance(market, dict):
                    continue
                inspected += 1
                text = market_text(event, market)
                category, market_type = classify_match_market(event, market, home_team, away_team)
                fuzzy_score, reasons = score_market_match(event, market, home_team, away_team, match_date)
                tokens, outcomes, gamma_prices, token_ids, token_error = extract_outcome_tokens(market)
                spread_line = parse_spread_line(text) if category == "spread" else None
                total_line = parse_total_line(text) if category == "total" else None
                accepted = False
                rejected_reason = ""
                if fuzzy_score < self.cfg.polymarket_match_confidence_threshold:
                    rejected_reason = f"fuzzy score {fuzzy_score:.2f} below threshold {self.cfg.polymarket_match_confidence_threshold:.2f}"
                elif category == "unknown":
                    rejected_reason = "market type unknown"
                elif token_error:
                    rejected_reason = token_error
                else:
                    priced_tokens = [self._priced_token(token, market, debug) for token in tokens]
                    tokens = priced_tokens
                    accepted = any(token.implied_probability is not None for token in tokens)
                    if not accepted:
                        rejected_reason = "no usable CLOB midpoint or fallback last trade price"
                candidates.append(
                    PolymarketDebugCandidate(
                        event_title=str(event.get("title", "")),
                        event_slug=str(event.get("slug", "")),
                        market_question=str(market.get("question", "")),
                        market_slug=str(market.get("slug", "")),
                        category=category,
                        market_type=market_type,
                        fuzzy_score=fuzzy_score,
                        reasons=reasons,
                        accepted=accepted,
                        rejected_reason=rejected_reason,
                        outcomes=outcomes,
                        gamma_prices=gamma_prices,
                        clob_token_ids=token_ids,
                        tokens=tokens,
                        spread_line=spread_line,
                        total_line=total_line,
                    )
                )
        debug.markets_inspected = inspected
        debug.candidates = sorted(candidates, key=lambda item: item.fuzzy_score, reverse=True)
        return debug

    def best_moneyline_for_match(
        self,
        home_team: str,
        away_team: str,
        match_date: str,
        refresh: bool = False,
    ) -> dict[str, Any] | None:
        markets = self.best_markets_for_match(home_team, away_team, match_date, refresh=refresh)
        return markets.get("moneyline") if markets else None

    def best_markets_for_match(
        self,
        home_team: str,
        away_team: str,
        match_date: str,
        refresh: bool = False,
    ) -> dict[str, Any]:
        # `refresh` is kept for CLI compatibility; Gamma/CLOB calls are live.
        debug = self.discover_match_markets(home_team, away_team, match_date, max_pages=30, refresh=refresh)
        totals = self._best_totals_from_debug(debug)
        spreads = self._best_spreads_from_debug(debug, home_team, away_team)
        return {
            "moneyline": self._best_moneyline_from_debug(debug, home_team, away_team),
            "total": totals[0] if totals else None,
            "totals": totals,
            "spread": spreads[0] if spreads else None,
            "spreads": spreads,
        }

    def _best_moneyline_from_debug(
        self,
        debug: PolymarketDebugReport,
        home_team: str,
        away_team: str,
    ) -> dict[str, Any] | None:
        pieces: dict[str, float] = {}
        raw: dict[str, float] = {}
        confidence_values: list[float] = []
        source_titles: list[str] = []
        for candidate in debug.candidates:
            if not candidate.accepted or candidate.category != "moneyline":
                continue
            yes_token = next((token for token in candidate.tokens if norm(token.outcome) == "yes" and token.implied_probability is not None), None)
            if candidate.market_type == "binary_home_win" and yes_token:
                pieces["home_win"] = yes_token.implied_probability
                raw["home_win"] = yes_token.implied_probability
            elif candidate.market_type == "binary_draw" and yes_token:
                pieces["draw"] = yes_token.implied_probability
                raw["draw"] = yes_token.implied_probability
            elif candidate.market_type == "binary_away_win" and yes_token:
                pieces["away_win"] = yes_token.implied_probability
                raw["away_win"] = yes_token.implied_probability
            elif candidate.market_type == "three_way_moneyline":
                for token in candidate.tokens:
                    key = classify_outcome(token.outcome, home_team, away_team)
                    if key and token.implied_probability is not None:
                        pieces[key] = token.implied_probability
                        raw[key] = token.implied_probability
            if candidate.market_type in {"binary_home_win", "binary_draw", "binary_away_win", "three_way_moneyline"}:
                confidence_values.append(candidate.fuzzy_score)
                source_titles.append(candidate.market_question)
            if {"home_win", "draw", "away_win"}.issubset(pieces):
                break
        if {"home_win", "draw", "away_win"}.issubset(pieces):
            home, draw, away = normalize_probabilities([pieces["home_win"], pieces["draw"], pieces["away_win"]])
            return {
                "raw": raw,
                "normalized": {"home_win": home, "draw": draw, "away_win": away},
                "raw_by_team": {home_team: raw["home_win"], "Draw": raw["draw"], away_team: raw["away_win"]},
                "normalized_by_team": {home_team: home, "Draw": draw, away_team: away},
                "source": "polymarket_gamma_events_clob",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "age_minutes": 0.0,
                "confidence": min(confidence_values) if confidence_values else 0.0,
                "market_type": "assembled_binary_moneyline",
                "title": " | ".join(source_titles),
                "slug": "",
            }
        return None

    def _best_total_from_debug(self, debug: PolymarketDebugReport) -> dict[str, Any] | None:
        totals = self._best_totals_from_debug(debug)
        return totals[0] if totals else None

    def _best_totals_from_debug(self, debug: PolymarketDebugReport, max_lines: int = 6) -> list[dict[str, Any]]:
        candidates = [
            candidate
            for candidate in debug.candidates
            if candidate.accepted and is_full_match_total(candidate)
        ]
        if not candidates:
            return []
        selected: list[dict[str, Any]] = []
        seen_lines: set[float] = set()
        # Start near 2.5, then add surrounding lines to shape the whole totals curve.
        for candidate in sorted(candidates, key=lambda item: (abs((item.total_line or 0) - 2.5), item.total_line or 0)):
            line = candidate.total_line
            if line is None or line in seen_lines:
                continue
            over = next((token for token in candidate.tokens if norm(token.outcome) == "over" and token.implied_probability is not None), None)
            under = next((token for token in candidate.tokens if norm(token.outcome) == "under" and token.implied_probability is not None), None)
            if not over:
                continue
            over_prob = over.implied_probability
            under_prob = under.implied_probability if under else None
            if under_prob is not None:
                over_prob, under_prob = normalize_probabilities([over_prob, under_prob])
            seen_lines.add(line)
            selected.append(
                {
                    "line": line,
                    "over_price": over.implied_probability,
                    "under_price": under.implied_probability if under else None,
                    "over_probability": over_prob,
                    "under_probability": under_prob,
                    "market_question": candidate.market_question,
                    "market_slug": candidate.market_slug,
                    "confidence": candidate.fuzzy_score,
                }
            )
            if len(selected) >= max_lines:
                break
        return selected

    def _best_spread_from_debug(self, debug: PolymarketDebugReport, home_team: str, away_team: str) -> dict[str, Any] | None:
        spreads = self._best_spreads_from_debug(debug, home_team, away_team)
        return spreads[0] if spreads else None

    def _best_spreads_from_debug(
        self,
        debug: PolymarketDebugReport,
        home_team: str,
        away_team: str,
        max_lines: int = 4,
    ) -> list[dict[str, Any]]:
        candidates = [
            candidate
            for candidate in debug.candidates
            if candidate.accepted and is_full_match_spread(candidate)
        ]
        if not candidates:
            return []
        selected: list[dict[str, Any]] = []
        seen: set[tuple[str, float]] = set()
        # Use closest-to-zero lines first; they carry the most stable margin information.
        for candidate in sorted(candidates, key=lambda item: (abs(item.spread_line or 0), item.spread_line or 0, -item.fuzzy_score)):
            first = next((token for token in candidate.tokens if token.implied_probability is not None), None)
            if first is None or candidate.spread_line is None:
                continue
            team = first.outcome
            team_key = classify_outcome(team, home_team, away_team)
            if team_key not in {"home_win", "away_win"}:
                continue
            key = (team_key, candidate.spread_line)
            if key in seen:
                continue
            second = next((token for token in candidate.tokens if token is not first and token.implied_probability is not None), None)
            cover_prob = first.implied_probability
            other_prob = second.implied_probability if second else None
            if other_prob is not None:
                cover_prob, _ = normalize_probabilities([cover_prob, other_prob])
            seen.add(key)
            selected.append(
                {
                    "team": home_team if team_key == "home_win" else away_team,
                    "team_is_home": team_key == "home_win",
                    "line": candidate.spread_line,
                    "price": first.implied_probability,
                    "cover_probability": cover_prob,
                    "market_question": candidate.market_question,
                    "market_slug": candidate.market_slug,
                    "confidence": candidate.fuzzy_score,
                }
            )
            if len(selected) >= max_lines:
                break
        return selected
