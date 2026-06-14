from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Config:
    historical_results_path: Path = Path(os.getenv("HISTORICAL_RESULTS_PATH", "data/raw/historical_results.csv"))
    clean_historical_results_path: Path = Path("data/processed/clean_historical_results.csv")
    schedule_path: Path = Path(os.getenv("WORLDCUP_2026_SCHEDULE_PATH", "data/manual/worldcup_2026_schedule.csv"))
    elo_ratings_path: Path = Path(os.getenv("ELO_RATINGS_PATH", "data/manual/elo_ratings.csv"))
    team_mapping_path: Path = Path(os.getenv("TEAM_MAPPING_PATH", "data/manual/team_name_mapping.csv"))
    market_snapshots_path: Path = Path(os.getenv("MARKET_SNAPSHOTS_PATH", "data/manual/market_snapshots.csv"))
    prediction_snapshots_path: Path = Path(os.getenv("PREDICTION_SNAPSHOTS_PATH", "data/manual/prediction_snapshots.csv"))

    polymarket_gamma_base_url: str = os.getenv("POLYMARKET_GAMMA_BASE_URL", "https://gamma-api.polymarket.com")
    polymarket_clob_base_url: str = os.getenv("POLYMARKET_CLOB_BASE_URL", "https://clob.polymarket.com")
    polymarket_worldcup_games_url: str = os.getenv(
        "POLYMARKET_WORLDCUP_GAMES_URL",
        "https://polymarket.com/sports/world-cup/games",
    )
    use_polymarket: bool = _bool("USE_POLYMARKET", True)
    polymarket_match_confidence_threshold: float = _float("POLYMARKET_MATCH_CONFIDENCE_THRESHOLD", 0.8)
    polymarket_cache_minutes: int = _int("POLYMARKET_CACHE_MINUTES", 10)
    polymarket_request_timeout_seconds: float = _float("POLYMARKET_REQUEST_TIMEOUT_SECONDS", 8.0)

    historical_lookback_years: int = _int("HISTORICAL_LOOKBACK_YEARS", 4)
    use_older_data_as_prior: bool = _bool("USE_OLDER_DATA_AS_PRIOR", False)
    older_data_prior_weight: float = _float("OLDER_DATA_PRIOR_WEIGHT", 0.2)
    current_worldcup_weight: float = _float("CURRENT_WORLDCUP_WEIGHT", 3.0)
    last_12_months_weight: float = _float("LAST_12_MONTHS_WEIGHT", 2.0)
    last_24_months_weight: float = _float("LAST_24_MONTHS_WEIGHT", 1.5)
    last_48_months_weight: float = _float("LAST_48_MONTHS_WEIGHT", 1.0)

    model_weight: float = _float("MODEL_WEIGHT", 0.4)
    moneyline_market_weight: float = _float("MONEYLINE_MARKET_WEIGHT", 0.6)
    live_model_weight: float = _float("LIVE_MODEL_WEIGHT", 0.3)
    live_moneyline_market_weight: float = _float("LIVE_MONEYLINE_MARKET_WEIGHT", 0.7)
    spread_calibration_weight: float = _float("SPREAD_CALIBRATION_WEIGHT", 0.25)
    total_calibration_weight: float = _float("TOTAL_CALIBRATION_WEIGHT", 0.30)
    team_total_calibration_weight: float = _float("TEAM_TOTAL_CALIBRATION_WEIGHT", 0.25)
    use_futures_as_team_strength: bool = _bool("USE_FUTURES_AS_TEAM_STRENGTH", False)

    max_goals: int = _int("MAX_GOALS", 6)
    recent_form_matches: int = _int("RECENT_FORM_MATCHES", 10)
    elo_initial_rating: float = _float("ELO_INITIAL_RATING", 1500.0)
    elo_k_factor: float = _float("ELO_K_FACTOR", 30.0)
    host_advantage_elo: float = _float("HOST_ADVANTAGE_ELO", 35.0)
    home_advantage_goals: float = _float("HOME_ADVANTAGE_GOALS", 0.12)
