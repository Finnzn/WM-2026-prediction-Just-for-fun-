#!/usr/bin/env python3
"""Normalize country/team names across local project CSV files."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING_PATH = PROJECT_ROOT / "data" / "manual" / "team_name_mapping.csv"
DEFAULT_FILES = [
    PROJECT_ROOT / "data" / "manual" / "worldcup_2026_schedule.csv",
    PROJECT_ROOT / "data" / "manual" / "world_football_elo_ratings_2026_world_cup_complete.csv",
    PROJECT_ROOT / "data" / "processed" / "clean_historical_results.csv",
]
NAME_COLUMNS = {"home_team", "away_team", "team", "country"}


def normalize_lookup_key(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def load_mapping(path: Path) -> dict[str, str]:
    mapping_df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"source_name", "canonical_name"}
    missing = required - set(mapping_df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    pairs = mapping_df[["source_name", "canonical_name"]].dropna()
    return {
        normalize_lookup_key(source): str(canonical).strip()
        for source, canonical in pairs.itertuples(index=False)
        if str(source).strip() and str(canonical).strip()
    }


def normalize_value(value: object, mapping: dict[str, str]) -> object:
    if pd.isna(value):
        return value
    text = re.sub(r"\s+", " ", str(value).strip())
    return mapping.get(normalize_lookup_key(text), text)


def normalize_file(path: Path, mapping: dict[str, str], dry_run: bool) -> int:
    if not path.exists():
        print(f"Skipping missing file: {path}")
        return 0

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    normalized_columns = [column for column in df.columns if column in NAME_COLUMNS]
    if not normalized_columns:
        print(f"Skipping {path}: no country/team columns found")
        return 0

    changes = 0
    for column in normalized_columns:
        before = df[column].copy()
        df[column] = df[column].map(lambda value: normalize_value(value, mapping))
        changes += int((before.fillna("") != df[column].fillna("")).sum())

    if not dry_run:
        df.to_csv(path, index=False)

    print(f"{path}: normalized {changes} values in {', '.join(normalized_columns)}")
    return changes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-path", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("files", nargs="*", type=Path, default=DEFAULT_FILES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        mapping = load_mapping(args.mapping_path)
        total_changes = sum(normalize_file(path, mapping, args.dry_run) for path in args.files)
    except (OSError, ValueError) as exc:
        print(f"Country/team name normalization failed: {exc}", file=sys.stderr)
        sys.exit(1)

    action = "Would normalize" if args.dry_run else "Normalized"
    print(f"{action} {total_changes} total values.")


if __name__ == "__main__":
    main()
