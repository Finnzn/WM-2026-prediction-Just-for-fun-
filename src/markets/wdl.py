from __future__ import annotations

import numpy as np

from src.utils import matrix_to_wdl


def calibrate_wdl(matrix: np.ndarray, target_probs: tuple[float, float, float]) -> np.ndarray:
    current_home, current_draw, current_away = matrix_to_wdl(matrix)
    target_home, target_draw, target_away = target_probs
    factors = {
        "home": target_home / current_home if current_home > 0 else 1.0,
        "draw": target_draw / current_draw if current_draw > 0 else 1.0,
        "away": target_away / current_away if current_away > 0 else 1.0,
    }
    adjusted = matrix.copy()
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            if home_goals > away_goals:
                adjusted[home_goals, away_goals] *= factors["home"]
            elif home_goals == away_goals:
                adjusted[home_goals, away_goals] *= factors["draw"]
            else:
                adjusted[home_goals, away_goals] *= factors["away"]
    total = adjusted.sum()
    return adjusted / total if total > 0 else matrix
