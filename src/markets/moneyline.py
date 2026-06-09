from __future__ import annotations

from src.utils import normalize_probabilities


def blend_moneyline(model_probs: tuple[float, float, float], market_probs: tuple[float, float, float], model_weight: float, market_weight: float) -> tuple[float, float, float]:
    total_weight = model_weight + market_weight
    if total_weight <= 0:
        return model_probs
    mw = model_weight / total_weight
    kw = market_weight / total_weight
    blended = [model_probs[i] * mw + market_probs[i] * kw for i in range(3)]
    return tuple(normalize_probabilities(blended))  # type: ignore[return-value]

