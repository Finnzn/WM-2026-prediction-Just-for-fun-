import numpy as np

from src.markets.classifier import parse_spread_line
from src.markets.spread import calibrate_spread, spread_cover_probability
from src.utils import poisson_score_matrix


def test_spread_parser_interprets_handicap_lines():
    assert parse_spread_line("Mexico +2.5") == 2.5
    assert parse_spread_line("South Africa -1.5") == -1.5
    assert parse_spread_line("fifwc-bra-mar-2026-06-13 Spread: Brazil (-1.5)") == -1.5


def test_spread_calibration_preserves_probability_sum():
    matrix = poisson_score_matrix(1.2, 1.0, 6)
    adjusted = calibrate_spread(matrix, True, 2.5, 0.99, 0.25)
    assert np.isclose(adjusted.sum(), 1.0)
    assert spread_cover_probability(adjusted, True, 2.5) > spread_cover_probability(matrix, True, 2.5)
