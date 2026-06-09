from src.markets.classifier import classify_market


def test_market_classifier_categories():
    assert classify_market("Mexico vs South Africa moneyline Draw")[0] == "moneyline"
    assert classify_market("Mexico +2.5 spread")[0] == "spread"
    assert classify_market("Over 2.5 goals")[0] == "total"
    assert classify_market("Will France win the World Cup?")[0] == "futures"
    assert classify_market("Random unrelated question")[0] == "unknown"

