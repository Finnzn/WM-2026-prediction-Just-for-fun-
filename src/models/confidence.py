from __future__ import annotations


def confidence_score(
    probs: tuple[float, float, float],
    home_matches: float,
    away_matches: float,
    market_data_used: bool = False,
    market_confidence: float = 0.0,
    unresolved_penalty: float = 0.0,
) -> float:
    favorite_strength = max(probs) - sorted(probs)[1]
    data_depth = min(1.0, (home_matches + away_matches) / 30.0)
    score = 0.25 + favorite_strength * 0.55 + data_depth * 0.20
    if market_data_used:
        score += 0.15 * market_confidence
    score -= unresolved_penalty
    return round(max(0.0, min(1.0, score)), 3)

