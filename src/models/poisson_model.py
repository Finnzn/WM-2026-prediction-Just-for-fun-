from __future__ import annotations

import math

import numpy as np

from src.config import Config
from src.utils import matrix_to_wdl, poisson_score_matrix, top_scorelines


def estimate_lambdas(
    home_team: str,
    away_team: str,
    elo_ratings: dict[str, float],
    team_stats: dict[str, dict[str, float]],
    global_home_goals: float,
    global_away_goals: float,
    neutral: bool,
    cfg: Config,
) -> tuple[float, float]:
    home_stats = team_stats.get(home_team, {})
    away_stats = team_stats.get(away_team, {})
    home_attack = home_stats.get("goals_for", global_home_goals) / max(0.4, (global_home_goals + global_away_goals) / 2)
    away_attack = away_stats.get("goals_for", global_away_goals) / max(0.4, (global_home_goals + global_away_goals) / 2)
    home_defense_allowed = home_stats.get("goals_against", global_away_goals)
    away_defense_allowed = away_stats.get("goals_against", global_home_goals)
    home_defense_factor = away_defense_allowed / max(0.4, global_home_goals)
    away_defense_factor = home_defense_allowed / max(0.4, global_away_goals)
    elo_diff = elo_ratings.get(home_team, cfg.elo_initial_rating) - elo_ratings.get(away_team, cfg.elo_initial_rating)
    elo_home_mult = math.exp(elo_diff / 1200.0)
    elo_away_mult = math.exp(-elo_diff / 1200.0)
    home_adv = 0.0 if neutral else cfg.home_advantage_goals
    lambda_home = global_home_goals * home_attack * home_defense_factor * elo_home_mult + home_adv
    lambda_away = global_away_goals * away_attack * away_defense_factor * elo_away_mult
    return max(0.15, min(lambda_home, 4.5)), max(0.15, min(lambda_away, 4.5))


def prediction_from_lambdas(lambda_home: float, lambda_away: float, max_goals: int) -> dict:
    matrix = poisson_score_matrix(lambda_home, lambda_away, max_goals)
    return prediction_from_matrix(matrix)


def prediction_from_matrix(matrix: np.ndarray) -> dict:
    home_win, draw, away_win = matrix_to_wdl(matrix)
    top = top_scorelines(matrix, 5)
    best = top[0]
    expected_home = float(sum(i * matrix[i, :].sum() for i in range(matrix.shape[0])))
    expected_away = float(sum(j * matrix[:, j].sum() for j in range(matrix.shape[1])))
    return {
        "matrix": matrix,
        "predicted_home_goals": best["home_score"],
        "predicted_away_goals": best["away_score"],
        "predicted_score": best["score"],
        "expected_home_goals": expected_home,
        "expected_away_goals": expected_away,
        "home_win_prob": home_win,
        "draw_prob": draw,
        "away_win_prob": away_win,
        "top_5_scorelines": top,
        "tail_probability": max(0.0, 1.0 - float(matrix.sum())),
    }
