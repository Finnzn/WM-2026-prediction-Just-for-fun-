from __future__ import annotations

import numpy as np


def spread_cover_probability(matrix: np.ndarray, team_is_home: bool, line: float) -> float:
    prob = 0.0
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            margin = home_goals - away_goals if team_is_home else away_goals - home_goals
            if margin + line > 0:
                prob += float(matrix[home_goals, away_goals])
    return prob


def calibrate_spread(matrix: np.ndarray, team_is_home: bool, line: float, market_prob: float, weight: float) -> np.ndarray:
    current = spread_cover_probability(matrix, team_is_home, line)
    if current <= 0 or current >= 1:
        return matrix
    adjusted = matrix.copy()
    desired = current * (1 - weight) + market_prob * weight
    cover_factor = desired / current
    miss_factor = (1 - desired) / (1 - current)
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            margin = home_goals - away_goals if team_is_home else away_goals - home_goals
            adjusted[home_goals, away_goals] *= cover_factor if margin + line > 0 else miss_factor
    return adjusted / adjusted.sum()

