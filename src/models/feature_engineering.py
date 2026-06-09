from __future__ import annotations

import pandas as pd


def team_recent_stats(results: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    if results.empty:
        return stats
    rows = []
    for row in results.itertuples(index=False):
        weight = float(getattr(row, "weight", 1.0) or 1.0)
        rows.append((row.home_team, int(row.home_score), int(row.away_score), weight))
        rows.append((row.away_team, int(row.away_score), int(row.home_score), weight))
    for team in sorted({item[0] for item in rows}):
        team_rows = [item for item in rows if item[0] == team]
        weight_sum = sum(item[3] for item in team_rows) or 1.0
        goals_for = sum(item[1] * item[3] for item in team_rows) / weight_sum
        goals_against = sum(item[2] * item[3] for item in team_rows) / weight_sum
        clean_sheets = sum((1 if item[2] == 0 else 0) * item[3] for item in team_rows) / weight_sum
        points = sum((3 if item[1] > item[2] else 1 if item[1] == item[2] else 0) * item[3] for item in team_rows) / weight_sum
        stats[team] = {
            "matches": float(len(team_rows)),
            "weighted_matches": weight_sum,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "clean_sheet_rate": clean_sheets,
            "points_per_match": points,
        }
    return stats


def global_goal_rates(results: pd.DataFrame) -> tuple[float, float]:
    if results.empty:
        return 1.35, 1.10
    return max(0.4, float(results["home_score"].mean())), max(0.4, float(results["away_score"].mean()))

