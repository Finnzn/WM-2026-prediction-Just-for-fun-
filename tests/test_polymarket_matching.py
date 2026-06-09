from src.markets.classifier import score_candidate


def test_low_confidence_polymarket_match_not_used_automatically():
    candidate = score_candidate({"title": "Will it rain tomorrow?"}, "Mexico", "South Africa", "2026-06-11")
    assert candidate.confidence < 0.8


def test_candidate_scores_known_world_cup_match():
    candidate = score_candidate({"title": "Mexico vs South Africa World Cup moneyline Draw", "active": "true"}, "Mexico", "South Africa", "2026-06-11")
    assert candidate.category == "moneyline"
    assert candidate.confidence >= 0.8

