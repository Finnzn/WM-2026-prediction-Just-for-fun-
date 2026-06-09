from __future__ import annotations

import pandas as pd


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

