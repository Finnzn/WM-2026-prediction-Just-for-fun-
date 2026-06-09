import numpy as np

from src.models.poisson_model import prediction_from_lambdas
from src.utils import matrix_to_wdl, poisson_score_matrix


def test_poisson_score_matrix_sums_to_one():
    matrix = poisson_score_matrix(1.4, 0.9, 6)
    assert np.isclose(matrix.sum(), 1.0)


def test_win_draw_loss_probabilities_sum_to_one():
    pred = prediction_from_lambdas(1.4, 0.9, 6)
    total = pred["home_win_prob"] + pred["draw_prob"] + pred["away_win_prob"]
    assert abs(total - 1.0) < 1e-9

