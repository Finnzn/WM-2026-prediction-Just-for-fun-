from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data_sources.team_mapping import TeamNameMapper
from src.utils import snake_case


def load_historical_results(path: Path, mapper: TeamNameMapper) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [snake_case(column) for column in df.columns]
    for column in ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "city", "country", "neutral"]:
        if column not in df.columns:
            df[column] = ""
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_team"] = df["home_team"].map(mapper.normalize)
    df["away_team"] = df["away_team"].map(mapper.normalize)
    df["country"] = df["country"].map(mapper.normalize)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df


def add_time_weights(df: pd.DataFrame, cfg: Config, as_of: date | None = None) -> pd.DataFrame:
    as_of_ts = pd.Timestamp(as_of or date.today())
    df = df.copy()
    age_days = (as_of_ts - df["date"]).dt.days
    df["weight"] = 0.0
    df.loc[age_days <= 365, "weight"] = cfg.last_12_months_weight
    df.loc[(age_days > 365) & (age_days <= 730), "weight"] = cfg.last_24_months_weight
    df.loc[(age_days > 730) & (age_days <= 1460), "weight"] = cfg.last_48_months_weight
    older = age_days > 1460
    if cfg.use_older_data_as_prior:
        df.loc[older, "weight"] = cfg.older_data_prior_weight
    df = df[df["weight"].gt(0)].copy()
    return df


def effective_results(historical: pd.DataFrame, current_worldcup: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    hist = add_time_weights(historical, cfg)
    wc = current_worldcup.copy()
    if wc.empty:
        return hist.reset_index(drop=True)
    if not wc.empty:
        wc["date"] = pd.to_datetime(wc["date"], errors="coerce")
        wc["home_score"] = pd.to_numeric(wc["home_score"], errors="coerce")
        wc["away_score"] = pd.to_numeric(wc["away_score"], errors="coerce")
        wc = wc.dropna(subset=["date", "home_score", "away_score"]).copy()
        wc["home_score"] = wc["home_score"].astype(int)
        wc["away_score"] = wc["away_score"].astype(int)
        wc["weight"] = cfg.current_worldcup_weight
    return pd.concat([hist, wc], ignore_index=True, sort=False)
