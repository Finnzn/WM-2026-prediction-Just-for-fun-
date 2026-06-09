from src.markets.moneyline import blend_moneyline


def test_market_blending_preserves_probability_sum():
    probs = blend_moneyline((0.5, 0.25, 0.25), (0.7, 0.2, 0.1), 0.6, 0.4)
    assert abs(sum(probs) - 1.0) < 1e-9
    assert probs[0] > 0.5

