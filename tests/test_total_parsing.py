import numpy as np

from src.markets.classifier import parse_total_line
from src.markets.totals import calibrate_total, over_probability
from src.utils import poisson_score_matrix


def test_total_parser_interprets_over_under_lines():
    assert parse_total_line("Over 2.5 goals") == 2.5
    assert parse_total_line("U 4.5") == 4.5


def test_total_calibration_preserves_probability_sum():
    matrix = poisson_score_matrix(1.2, 1.0, 6)
    adjusted = calibrate_total(matrix, 2.5, 0.7, 0.30)
    assert np.isclose(adjusted.sum(), 1.0)
    assert over_probability(adjusted, 2.5) > over_probability(matrix, 2.5)

