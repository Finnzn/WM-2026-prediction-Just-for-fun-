from src.data_sources.polymarket import classify_match_market, extract_outcome_tokens
from src.data_sources.polymarket import PolymarketClient
from src.config import Config
from src.markets.classifier import score_candidate


def test_low_confidence_polymarket_match_not_used_automatically():
    candidate = score_candidate({"title": "Will it rain tomorrow?"}, "Mexico", "South Africa", "2026-06-11")
    assert candidate.confidence < 0.8


def test_candidate_scores_known_world_cup_match():
    candidate = score_candidate({"title": "Mexico vs South Africa World Cup moneyline Draw", "active": "true"}, "Mexico", "South Africa", "2026-06-11")
    assert candidate.category == "moneyline"
    assert candidate.confidence >= 0.8


def test_polymarket_jsonish_token_extraction_from_gamma_strings():
    market = {
        "outcomes": '["Yes","No"]',
        "outcomePrices": '["0.61","0.39"]',
        "clobTokenIds": '["123","456"]',
    }
    tokens, outcomes, prices, token_ids, error = extract_outcome_tokens(market)
    assert error == ""
    assert outcomes == ["Yes", "No"]
    assert prices == [0.61, 0.39]
    assert token_ids == ["123", "456"]
    assert [token.token_id for token in tokens] == ["123", "456"]


def test_world_cup_futures_do_not_classify_as_match_moneyline():
    event = {"title": "2026 FIFA World Cup Winner", "slug": "fifa-world-cup-2026-winner"}
    market = {"question": "Will Mexico win the 2026 FIFA World Cup?", "slug": "mexico-win-2026-fifa-world-cup"}
    assert classify_match_market(event, market, "Mexico", "South Africa")[0] == "futures"


def test_polymarket_slug_candidates_include_utc_date_and_market_codes():
    client = PolymarketClient(Config())
    assert "fifwc-kr-cze-2026-06-11" in client.event_slug_candidates("South Korea", "Czechia", "2026-06-12")
    assert "fifwc-usa-par-2026-06-12" in client.event_slug_candidates("United States", "Paraguay", "2026-06-13")
