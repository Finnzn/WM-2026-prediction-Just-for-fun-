from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config
from src.data_sources.team_mapping import TeamNameMapper
from src.utils import safe_float, snake_case


def load_elo_ratings(cfg: Config, mapper: TeamNameMapper, teams: set[str]) -> dict[str, float]:
    path = cfg.elo_ratings_path
    ratings: dict[str, float] = {team: cfg.elo_initial_rating for team in teams}
    if not path.exists():
        return ratings
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [snake_case(column) for column in df.columns]
    if "rating" in df.columns and "elo" not in df.columns:
        df = df.rename(columns={"rating": "elo"})
    if "team" not in df.columns or "elo" not in df.columns:
        return ratings
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").drop_duplicates("team", keep="last")
    for team, elo in df[["team", "elo"]].itertuples(index=False):
        canonical = mapper.normalize(team)
        value = safe_float(elo)
        if canonical and value is not None:
            ratings[str(canonical)] = value
    return ratings


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def actual_score(home_score: int, away_score: int) -> tuple[float, float]:
    if home_score > away_score:
        return 1.0, 0.0
    if home_score < away_score:
        return 0.0, 1.0
    return 0.5, 0.5


def goal_diff_multiplier(home_score: int, away_score: int) -> float:
    diff = abs(home_score - away_score)
    if diff <= 1:
        return 1.0
    return 1.0 + (diff - 1) * 0.35


def update_dynamic_elo(base: dict[str, float], played_matches: pd.DataFrame, cfg: Config) -> dict[str, float]:
    ratings = dict(base)
    if played_matches.empty:
        return ratings
    matches = played_matches.copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches = matches.sort_values("date")
    for row in matches.itertuples(index=False):
        home = str(row.home_team)
        away = str(row.away_team)
        home_score = int(row.home_score)
        away_score = int(row.away_score)
        ratings.setdefault(home, cfg.elo_initial_rating)
        ratings.setdefault(away, cfg.elo_initial_rating)
        neutral = str(getattr(row, "neutral", "")).strip().lower() in {"true", "1", "yes"}
        home_rating = ratings[home] + (0.0 if neutral else cfg.host_advantage_elo)
        exp_home = expected_score(home_rating, ratings[away])
        act_home, act_away = actual_score(home_score, away_score)
        mult = goal_diff_multiplier(home_score, away_score)
        change = cfg.elo_k_factor * mult * (act_home - exp_home)
        ratings[home] += change
        ratings[away] -= change
    return ratings
