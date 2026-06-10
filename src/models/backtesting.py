from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.config import Config
from src.data_sources.elo import update_dynamic_elo
from src.data_sources.historical_results import add_time_weights
from src.models.feature_engineering import team_recent_stats
from src.models.poisson_model import estimate_lambdas, prediction_from_lambdas


def simple_backtest(results: pd.DataFrame) -> dict[str, float]:
    """Small time-safe baseline: predict home/draw/away from only prior average outcomes."""
    matches = results.sort_values("date").reset_index(drop=True)
    if len(matches) < 20:
        return {"matches": float(len(matches)), "accuracy": 0.0, "exact_score_hit_rate": 0.0, "average_goal_error": 0.0}
    correct = 0
    exact = 0
    goal_error = 0.0
    evaluated = 0
    prior = []
    for row in matches.itertuples(index=False):
        if len(prior) >= 10:
            home_win_rate = sum(1 for h, a in prior if h > a) / len(prior)
            draw_rate = sum(1 for h, a in prior if h == a) / len(prior)
            away_rate = 1 - home_win_rate - draw_rate
            pred_outcome = max([(home_win_rate, "H"), (draw_rate, "D"), (away_rate, "A")])[1]
            actual = "H" if row.home_score > row.away_score else "A" if row.home_score < row.away_score else "D"
            correct += int(pred_outcome == actual)
            avg_home = round(sum(h for h, _ in prior) / len(prior))
            avg_away = round(sum(a for _, a in prior) / len(prior))
            exact += int(avg_home == row.home_score and avg_away == row.away_score)
            goal_error += abs(avg_home - row.home_score) + abs(avg_away - row.away_score)
            evaluated += 1
        prior.append((int(row.home_score), int(row.away_score)))
    return {
        "matches": float(evaluated),
        "accuracy": correct / evaluated if evaluated else 0.0,
        "exact_score_hit_rate": exact / evaluated if evaluated else 0.0,
        "average_goal_error": goal_error / evaluated if evaluated else 0.0,
    }


def walk_forward_model_backtest(
    results: pd.DataFrame,
    cfg: Config | None = None,
    min_training_matches: int = 250,
    max_evaluated_matches: int | None = 250,
) -> dict[str, Any]:
    """Backtest the statistical score model using only matches before each fixture date.

    This excludes Polymarket because the repo does not contain historical market snapshots.
    Elo is rebuilt from the same prior rows instead of using today's provided Elo file, which
    avoids leaking current team strength into old matches.
    """
    cfg = cfg or Config()
    matches = _prepared_matches(results)
    if len(matches) <= min_training_matches:
        return _empty_metrics(len(matches), min_training_matches)

    candidate_indexes = list(range(min_training_matches, len(matches)))
    if max_evaluated_matches is not None and len(candidate_indexes) > max_evaluated_matches:
        candidate_indexes = candidate_indexes[-max_evaluated_matches:]

    correct = 0
    exact = 0
    goal_error = 0.0
    log_loss = 0.0
    brier = 0.0
    home_actual = draw_actual = away_actual = 0
    home_prob_sum = draw_prob_sum = away_prob_sum = 0.0
    evaluated = 0

    for idx in candidate_indexes:
        row = matches.iloc[idx]
        as_of = pd.Timestamp(row["date"]).date()
        prior = matches[matches["date"].lt(pd.Timestamp(as_of))].copy()
        if len(prior) < min_training_matches:
            continue
        weighted_prior = add_time_weights(prior, cfg, as_of=as_of)
        if weighted_prior.empty:
            continue

        teams = set(weighted_prior["home_team"]) | set(weighted_prior["away_team"]) | {row["home_team"], row["away_team"]}
        elo = {str(team): cfg.elo_initial_rating for team in teams if str(team)}
        elo = update_dynamic_elo(elo, weighted_prior, cfg)
        stats = team_recent_stats(weighted_prior)
        global_home = max(0.4, float(weighted_prior["home_score"].mean()))
        global_away = max(0.4, float(weighted_prior["away_score"].mean()))
        neutral = str(row.get("neutral", "true")).strip().lower() in {"true", "1", "yes", ""}
        lambda_home, lambda_away = estimate_lambdas(
            str(row["home_team"]),
            str(row["away_team"]),
            elo,
            stats,
            global_home,
            global_away,
            neutral,
            cfg,
        )
        pred = prediction_from_lambdas(lambda_home, lambda_away, cfg.max_goals)
        probs = [float(pred["home_win_prob"]), float(pred["draw_prob"]), float(pred["away_win_prob"])]
        actual_idx = _actual_outcome_index(int(row["home_score"]), int(row["away_score"]))
        pred_idx = max(range(3), key=lambda pos: probs[pos])

        correct += int(pred_idx == actual_idx)
        exact += int(int(pred["predicted_home_goals"]) == int(row["home_score"]) and int(pred["predicted_away_goals"]) == int(row["away_score"]))
        goal_error += abs(float(pred["expected_home_goals"]) - int(row["home_score"])) + abs(float(pred["expected_away_goals"]) - int(row["away_score"]))
        log_loss += -math.log(max(1e-12, probs[actual_idx]))
        brier += sum((probs[pos] - (1.0 if pos == actual_idx else 0.0)) ** 2 for pos in range(3))
        home_actual += int(actual_idx == 0)
        draw_actual += int(actual_idx == 1)
        away_actual += int(actual_idx == 2)
        home_prob_sum += probs[0]
        draw_prob_sum += probs[1]
        away_prob_sum += probs[2]
        evaluated += 1

    if not evaluated:
        return _empty_metrics(len(matches), min_training_matches)
    return {
        "model": "walk_forward_statistical_model_no_markets",
        "available_matches": float(len(matches)),
        "matches": float(evaluated),
        "min_training_matches": float(min_training_matches),
        "accuracy": correct / evaluated,
        "exact_score_hit_rate": exact / evaluated,
        "average_expected_goal_error": goal_error / evaluated,
        "wdl_log_loss": log_loss / evaluated,
        "wdl_brier_score": brier / evaluated,
        "predicted_home_win_rate": home_prob_sum / evaluated,
        "actual_home_win_rate": home_actual / evaluated,
        "predicted_draw_rate": draw_prob_sum / evaluated,
        "actual_draw_rate": draw_actual / evaluated,
        "predicted_away_win_rate": away_prob_sum / evaluated,
        "actual_away_win_rate": away_actual / evaluated,
        "market_data_included": False,
    }


def _prepared_matches(results: pd.DataFrame) -> pd.DataFrame:
    matches = results.copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
    matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")
    matches = matches.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"]).copy()
    matches["home_score"] = matches["home_score"].astype(int)
    matches["away_score"] = matches["away_score"].astype(int)
    if "neutral" not in matches.columns:
        matches["neutral"] = "true"
    return matches.sort_values("date").reset_index(drop=True)


def _actual_outcome_index(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def _empty_metrics(available_matches: int, min_training_matches: int) -> dict[str, Any]:
    return {
        "model": "walk_forward_statistical_model_no_markets",
        "available_matches": float(available_matches),
        "matches": 0.0,
        "min_training_matches": float(min_training_matches),
        "accuracy": 0.0,
        "exact_score_hit_rate": 0.0,
        "average_expected_goal_error": 0.0,
        "wdl_log_loss": 0.0,
        "wdl_brier_score": 0.0,
        "market_data_included": False,
    }
