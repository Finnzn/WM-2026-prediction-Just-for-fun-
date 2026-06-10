import numpy as np

from src.models.poisson_model import prediction_from_lambdas, prediction_from_matrix
from src.markets.wdl import calibrate_wdl
from src.utils import matrix_to_wdl, poisson_score_matrix


def test_poisson_score_matrix_sums_to_one():
    matrix = poisson_score_matrix(1.4, 0.9, 6)
    assert np.isclose(matrix.sum(), 1.0)


def test_win_draw_loss_probabilities_sum_to_one():
    pred = prediction_from_lambdas(1.4, 0.9, 6)
    total = pred["home_win_prob"] + pred["draw_prob"] + pred["away_win_prob"]
    assert abs(total - 1.0) < 1e-9


def test_wdl_calibration_matches_target_probabilities():
    matrix = poisson_score_matrix(0.8, 1.4, 6)
    adjusted = calibrate_wdl(matrix, (0.5, 0.25, 0.25))
    home, draw, away = matrix_to_wdl(adjusted)
    assert np.isclose(adjusted.sum(), 1.0)
    assert np.isclose(home, 0.5)
    assert np.isclose(draw, 0.25)
    assert np.isclose(away, 0.25)


def test_predicted_score_follows_most_likely_outcome_bucket():
    matrix = np.zeros((4, 4))
    matrix[1, 1] = 0.14
    matrix[1, 0] = 0.13
    matrix[2, 0] = 0.12
    matrix[2, 1] = 0.11
    matrix[0, 1] = 0.08
    matrix[0, 0] = 0.07
    matrix[3, 0] = 0.06
    matrix[1, 2] = 0.05
    matrix[3, 1] = 0.04
    matrix[2, 2] = 0.03
    matrix = matrix / matrix.sum()
    pred = prediction_from_matrix(matrix)
    assert pred["home_win_prob"] > pred["draw_prob"]
    assert pred["predicted_home_goals"] > pred["predicted_away_goals"]
    assert pred["top_5_scorelines"][0]["score"] == "1-1"
