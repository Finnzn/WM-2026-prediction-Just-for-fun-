from __future__ import annotations

import numpy as np


def over_probability(matrix: np.ndarray, line: float) -> float:
    prob = 0.0
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            if home_goals + away_goals > line:
                prob += float(matrix[home_goals, away_goals])
    return prob


def calibrate_total(matrix: np.ndarray, line: float, market_over_prob: float, weight: float) -> np.ndarray:
    current = over_probability(matrix, line)
    if current <= 0 or current >= 1:
        return matrix
    adjusted = matrix.copy()
    desired = current * (1 - weight) + market_over_prob * weight
    over_factor = desired / current
    under_factor = (1 - desired) / (1 - current)
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            adjusted[home_goals, away_goals] *= over_factor if home_goals + away_goals > line else under_factor
    return adjusted / adjusted.sum()

