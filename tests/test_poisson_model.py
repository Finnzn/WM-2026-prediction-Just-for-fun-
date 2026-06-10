import numpy as np

from src.models.poisson_model import prediction_from_lambdas
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
