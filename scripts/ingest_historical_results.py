#!/usr/bin/env python3
"""Ingest and clean historical international football results."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "historical_results.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "clean_historical_results.csv"
MAPPING_PATH = PROJECT_ROOT / "data" / "manual" / "team_name_mapping.csv"
WC_2026_SCHEDULE_PATH = PROJECT_ROOT / "data" / "manual" / "worldcup_2026_schedule.csv"
KAGGLE_DATASET = "martj42/international-football-results-from-1872-to-2017"

REQUIRED_COLUMNS = {"date", "home_team", "away_team", "home_score", "away_score"}
PLACEHOLDER_PATTERNS = [
    r"\btbd\b",
    r"\bto be determined\b",
    r"\bwinner\s+group\s+[a-z0-9]+\b",
    r"\brunner[- ]?up\s+group\s+[a-z0-9]+\b",
    r"\bplay-?off\s+winner\b",
    r"\bplaceholder\b",
]
NON_PLAYED_PATTERNS = [
    "abandoned",
    "postponed",
    "cancelled",
    "canceled",
    "not played",
    "void",
    "walkover",
    "suspended",
]
PLAYED_STATUSES = {"played", "finished", "final", "ft", "full time", "full-time", "completed"}


@dataclass
class CleaningReport:
    raw_rows_loaded: int
    missing_scores_removed: int
    future_dates_removed: int
    placeholder_teams_removed: int
    duplicates_removed: int
    final_cleaned_matches: int
    date_min: str
    date_max: str


def snake_case(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [snake_case(str(column)) for column in df.columns]
    aliases = {
        "home": "home_team",
        "away": "away_team",
        "home_team_name": "home_team",
        "away_team_name": "away_team",
        "home_goals": "home_score",
        "away_goals": "away_score",
        "home_team_score": "home_score",
        "away_team_score": "away_score",
        "home_score_ft": "home_score",
        "away_score_ft": "away_score",
    }
    return df.rename(columns={key: value for key, value in aliases.items() if key in df.columns})


def read_csv_standardized(path: Path) -> pd.DataFrame:
    return standardize_columns(pd.read_csv(path))


def has_required_columns(path: Path) -> bool:
    try:
        columns = set(standardize_columns(pd.read_csv(path, nrows=0)).columns)
    except Exception:
        return False
    return REQUIRED_COLUMNS.issubset(columns)


def find_results_csv(download_dir: Path) -> Path:
    candidates = [path for path in download_dir.rglob("*.csv") if has_required_columns(path)]
    if not candidates:
        raise FileNotFoundError(
            f"No CSV with required columns {sorted(REQUIRED_COLUMNS)} found in {download_dir}"
        )

    def score(path: Path) -> tuple[int, int]:
        name = path.name.lower()
        return (10 if "result" in name else 0, -len(path.parts))

    return sorted(candidates, key=score, reverse=True)[0]


def ensure_raw_file(raw_path: Path, download_if_missing: bool) -> Path:
    if raw_path.exists():
        print(f"Using local historical results: {raw_path}")
        return raw_path

    if not download_if_missing:
        raise FileNotFoundError(
            f"{raw_path} does not exist. Add the file locally or rerun with "
            "--download-if-missing / DOWNLOAD_HISTORICAL_DATA=true to use KaggleHub."
        )

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed. Install it with `pip install kagglehub`, then rerun. "
            "If Kaggle requires authentication, set KAGGLE_USERNAME and KAGGLE_KEY or log in "
            "with KaggleHub. Do not commit kaggle.json or credentials."
        ) from exc

    try:
        download_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:
        raise RuntimeError(
            "KaggleHub could not download the public historical results dataset. If Kaggle "
            "requires authentication or consent, log in with KaggleHub or set KAGGLE_USERNAME "
            "and KAGGLE_KEY in your environment. Do not hardcode credentials or commit "
            "kaggle.json."
        ) from exc

    source_csv = find_results_csv(download_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df = read_csv_standardized(source_csv)
    df.to_csv(raw_path, index=False)
    print(f"Downloaded {KAGGLE_DATASET}")
    print(f"Normalized {source_csv} -> {raw_path}")
    return raw_path


def load_team_mapping(mapping_path: Path) -> dict[str, str]:
    if not mapping_path.exists():
        return {}

    mapping_df = standardize_columns(pd.read_csv(mapping_path))
    source_column = first_existing(mapping_df.columns, ["source_name", "raw_name", "old_name", "from", "team"])
    target_column = first_existing(
        mapping_df.columns,
        ["canonical_name", "normalized_name", "new_name", "to", "standard_name"],
    )
    if source_column is None or target_column is None:
        print(f"Skipping {mapping_path}: expected source_name/canonical_name style columns.")
        return {}

    pairs = mapping_df[[source_column, target_column]].dropna()
    return {
        normalize_lookup_key(source): str(target).strip()
        for source, target in pairs.itertuples(index=False)
        if str(source).strip() and str(target).strip()
    }


def first_existing(columns: Iterable[str], names: list[str]) -> str | None:
    column_set = set(columns)
    for name in names:
        if name in column_set:
            return name
    return None


def normalize_lookup_key(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def normalize_team(value: object, mapping: dict[str, str]) -> object:
    if pd.isna(value):
        return value
    text = re.sub(r"\s+", " ", str(value).strip())
    return mapping.get(normalize_lookup_key(text), text)


def status_is_played(series: pd.Series) -> pd.Series:
    status = series.fillna("").astype(str).str.strip().str.lower()
    return status.isin(PLAYED_STATUSES)


def is_non_played_status(series: pd.Series) -> pd.Series:
    status = series.fillna("").astype(str).str.lower()
    pattern = "|".join(re.escape(item) for item in NON_PLAYED_PATTERNS)
    return status.str.contains(pattern, regex=True, na=False)


def placeholder_mask(df: pd.DataFrame) -> pd.Series:
    combined = (
        df["home_team"].fillna("").astype(str).str.strip()
        + " "
        + df["away_team"].fillna("").astype(str).str.strip()
    ).str.lower()
    return combined.str.contains("|".join(PLACEHOLDER_PATTERNS), regex=True, na=False)


def remove_world_cup_2026_unplayed(df: pd.DataFrame) -> pd.DataFrame:
    tournament_mask = pd.Series(False, index=df.index)
    if "tournament" in df.columns:
        tournament_mask = df["tournament"].fillna("").astype(str).str.lower().str.contains(
            "world cup", na=False
        )

    year_mask = df["date"].dt.year.eq(2026)
    schedule_mask = pd.Series(False, index=df.index)
    if WC_2026_SCHEDULE_PATH.exists():
        schedule = read_csv_standardized(WC_2026_SCHEDULE_PATH)
        if {"date", "home_team", "away_team"}.issubset(schedule.columns):
            mapping = load_team_mapping(MAPPING_PATH)
            schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce").dt.date
            schedule["home_team"] = schedule["home_team"].map(lambda value: normalize_team(value, mapping))
            schedule["away_team"] = schedule["away_team"].map(lambda value: normalize_team(value, mapping))
            keys = set(
                schedule.dropna(subset=["date", "home_team", "away_team"])
                .assign(
                    key=lambda frame: frame["date"].astype(str)
                    + "|"
                    + frame["home_team"].astype(str)
                    + "|"
                    + frame["away_team"].astype(str)
                )["key"]
            )
            current_keys = (
                df["date"].dt.date.astype(str)
                + "|"
                + df["home_team"].astype(str)
                + "|"
                + df["away_team"].astype(str)
            )
            schedule_mask = current_keys.isin(keys)

    status_column = first_existing(df.columns, ["status", "match_status", "fixture_status"])
    played_mask = status_is_played(df[status_column]) if status_column else pd.Series(False, index=df.index)
    remove_mask = (schedule_mask | (tournament_mask & year_mask)) & ~played_mask
    return df.loc[~remove_mask].copy()


def clean_historical_results(
    raw_path: Path,
    processed_path: Path,
    mapping_path: Path,
    today: date,
    use_older_data_as_prior: bool,
) -> CleaningReport:
    df = read_csv_standardized(raw_path)
    raw_rows_loaded = len(df)

    missing_required = REQUIRED_COLUMNS - set(df.columns)
    if missing_required:
        raise ValueError(f"{raw_path} is missing required columns: {sorted(missing_required)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    mapping = load_team_mapping(mapping_path)
    df["home_team"] = df["home_team"].map(lambda value: normalize_team(value, mapping))
    df["away_team"] = df["away_team"].map(lambda value: normalize_team(value, mapping))
    if "country" in df.columns:
        df["country"] = df["country"].map(lambda value: normalize_team(value, mapping))

    before = len(df)
    df = df.dropna(subset=["home_team", "away_team"])
    df = df[df["home_team"].astype(str).str.strip().ne("")]
    df = df[df["away_team"].astype(str).str.strip().ne("")]

    home_scores = pd.to_numeric(df["home_score"], errors="coerce")
    away_scores = pd.to_numeric(df["away_score"], errors="coerce")
    valid_scores = home_scores.notna() & away_scores.notna()
    missing_scores_removed = len(df) - int(valid_scores.sum())
    df = df.loc[valid_scores].copy()
    df["home_score"] = home_scores.loc[df.index].astype(int)
    df["away_score"] = away_scores.loc[df.index].astype(int)

    if "date" in df.columns:
        df = df.dropna(subset=["date"])

    status_column = first_existing(df.columns, ["status", "match_status", "fixture_status"])
    if status_column:
        df = df.loc[~is_non_played_status(df[status_column])].copy()

    today_ts = pd.Timestamp(today)
    status_played = (
        status_is_played(df[status_column])
        if status_column
        else pd.Series(False, index=df.index)
    )
    future_mask = df["date"].gt(today_ts) | (df["date"].ge(today_ts) & ~status_played)
    future_dates_removed = int(future_mask.sum())
    df = df.loc[~future_mask].copy()

    before = len(df)
    placeholders = placeholder_mask(df)
    placeholder_teams_removed = int(placeholders.sum())
    df = df.loc[~placeholders].copy()

    df = remove_world_cup_2026_unplayed(df)

    if not use_older_data_as_prior:
        cutoff = today_ts - pd.DateOffset(years=4)
        df = df.loc[df["date"].ge(cutoff)].copy()
    else:
        df["older_data_as_prior"] = df["date"].lt(today_ts - pd.DateOffset(years=4))

    duplicate_subset = ["date", "home_team", "away_team", "home_score", "away_score"]
    before = len(df)
    df = df.drop_duplicates(subset=duplicate_subset, keep="first").copy()
    duplicates_removed = before - len(df)

    df = df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False, date_format="%Y-%m-%d")

    date_min = "" if df.empty else df["date"].min().date().isoformat()
    date_max = "" if df.empty else df["date"].max().date().isoformat()
    return CleaningReport(
        raw_rows_loaded=raw_rows_loaded,
        missing_scores_removed=missing_scores_removed,
        future_dates_removed=future_dates_removed,
        placeholder_teams_removed=placeholder_teams_removed,
        duplicates_removed=duplicates_removed,
        final_cleaned_matches=len(df),
        date_min=date_min,
        date_max=date_max,
    )


def print_report(report: CleaningReport) -> None:
    print("Historical data cleaning report")
    print(f"- raw historical rows loaded: {report.raw_rows_loaded}")
    print(f"- rows removed due to missing scores: {report.missing_scores_removed}")
    print(f"- rows removed due to future dates: {report.future_dates_removed}")
    print(f"- rows removed due to placeholder teams: {report.placeholder_teams_removed}")
    print(f"- rows removed due to duplicates: {report.duplicates_removed}")
    print(f"- final number of cleaned historical matches: {report.final_cleaned_matches}")
    print(f"- cleaned historical date range: {report.date_min or 'n/a'} to {report.date_max or 'n/a'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--processed-path", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--mapping-path", type=Path, default=MAPPING_PATH)
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        default=os.getenv("DOWNLOAD_HISTORICAL_DATA", "").lower() in {"1", "true", "yes"},
        help="Download the public KaggleHub dataset if data/raw/historical_results.csv is absent.",
    )
    parser.add_argument(
        "--use-older-data-as-prior",
        action="store_true",
        default=os.getenv("USE_OLDER_DATA_AS_PRIOR", "").lower() in {"1", "true", "yes"},
        help="Keep matches older than four years and flag them as older_data_as_prior.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        raw_path = ensure_raw_file(args.raw_path, args.download_if_missing)
        report = clean_historical_results(
            raw_path=raw_path,
            processed_path=args.processed_path,
            mapping_path=args.mapping_path,
            today=date.today(),
            use_older_data_as_prior=args.use_older_data_as_prior,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Historical data ingestion failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print_report(report)
    print(f"Saved cleaned data to {args.processed_path}")


if __name__ == "__main__":
    main()
