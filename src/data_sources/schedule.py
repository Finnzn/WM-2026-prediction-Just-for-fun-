from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_sources.team_mapping import TeamNameMapper
from src.utils import is_placeholder_team, snake_case


STATUS_MAP = {
    "to be played": "scheduled",
    "scheduled": "scheduled",
    "played": "played",
    "finished": "played",
    "final": "played",
    "postponed": "postponed",
    "cancelled": "cancelled",
    "canceled": "cancelled",
}
REQUIRED = {"match_id", "date", "home_team", "away_team", "status"}


def load_schedule(path: Path, mapper: TeamNameMapper) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [snake_case(column) for column in df.columns]
    aliases = {
        "time_gmt_2": "kickoff_time",
        "time": "kickoff_time",
        "stadium": "venue",
        "matchday": "group",
    }
    df = df.rename(columns={old: new for old, new in aliases.items() if old in df.columns})
    if "match_id" not in df.columns:
        df.insert(0, "match_id", [f"M{i:03d}" for i in range(1, len(df) + 1)])
    if "stage" not in df.columns:
        df["stage"] = "Group Stage"
        knockout = df["group"].eq("") if "group" in df.columns else pd.Series(False, index=df.index)
        df.loc[knockout, "stage"] = "Knockout"
    for column in ["group", "venue", "city", "country", "neutral", "home_score", "away_score", "notes"]:
        if column not in df.columns:
            df[column] = ""
    if "kickoff_time" not in df.columns:
        df["kickoff_time"] = ""
    if "result" in df.columns and ("home_score" not in df.columns or df["home_score"].eq("").all()):
        scores = df["result"].str.extract(r"(?P<home_score>\d+)\D+(?P<away_score>\d+)")
        df["home_score"] = scores["home_score"].fillna(df["home_score"])
        df["away_score"] = scores["away_score"].fillna(df["away_score"])
    df["status"] = df["status"].str.strip().str.lower().map(STATUS_MAP).fillna(df["status"].str.strip().str.lower())
    df["home_team"] = df["home_team"].map(mapper.normalize)
    df["away_team"] = df["away_team"].map(mapper.normalize)
    if "country" in df.columns:
        df["country"] = df["country"].map(mapper.normalize)
    return df


def validate_schedule(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    missing = REQUIRED - set(df.columns)
    if missing:
        errors.append(f"Schedule missing columns: {sorted(missing)}")
    allowed = {"scheduled", "played", "postponed", "cancelled"}
    bad_status = sorted(set(df.get("status", pd.Series(dtype=str))) - allowed)
    if bad_status:
        errors.append(f"Schedule has invalid statuses: {bad_status}")
    played = df.get("status", pd.Series(dtype=str)).eq("played")
    if played.any():
        missing_scores = df.loc[played, "home_score"].eq("") | df.loc[played, "away_score"].eq("")
        if missing_scores.any():
            errors.append(f"Played matches missing scores: {df.loc[played & missing_scores, 'match_id'].tolist()}")
        home_numeric = pd.to_numeric(df.loc[played, "home_score"], errors="coerce").notna()
        away_numeric = pd.to_numeric(df.loc[played, "away_score"], errors="coerce").notna()
        non_numeric = played.copy()
        non_numeric.loc[played] = ~(home_numeric & away_numeric).to_numpy()
        if non_numeric.any():
            errors.append(f"Played matches with non-numeric scores: {df.loc[non_numeric, 'match_id'].tolist()}")
    scheduled = df.get("status", pd.Series(dtype=str)).eq("scheduled")
    scored_scheduled = scheduled & (df["home_score"].ne("") | df["away_score"].ne(""))
    if scored_scheduled.any():
        warnings.append(f"Scheduled matches with scores present: {df.loc[scored_scheduled, 'match_id'].tolist()}")
    unresolved = df[df["home_team"].map(is_placeholder_team) | df["away_team"].map(is_placeholder_team)]
    if not unresolved.empty:
        warnings.append(f"Unresolved placeholder fixtures: {unresolved['match_id'].tolist()}")
    return errors, warnings


def played_worldcup_results(schedule: pd.DataFrame) -> pd.DataFrame:
    played = schedule[schedule["status"].eq("played")].copy()
    if played.empty:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_score", "away_score", "tournament", "city", "country", "neutral", "source_weight"])
    rows = played[["date", "home_team", "away_team", "home_score", "away_score", "city", "country", "neutral"]].copy()
    rows["tournament"] = "FIFA World Cup 2026"
    rows["source_weight"] = 3.0
    return rows
