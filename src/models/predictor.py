from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from src.config import Config
from src.data_sources.elo import load_elo_ratings, update_dynamic_elo
from src.data_sources.historical_results import effective_results, load_historical_results
from src.data_sources.polymarket import PolymarketClient
from src.data_sources.schedule import load_schedule, played_worldcup_results
from src.data_sources.team_mapping import TeamNameMapper
from src.markets.moneyline import blend_moneyline
from src.markets.spread import calibrate_spread
from src.markets.totals import calibrate_team_total, calibrate_total
from src.markets.wdl import calibrate_wdl
from src.models.confidence import confidence_score
from src.models.feature_engineering import global_goal_rates, team_recent_stats
from src.models.poisson_model import estimate_lambdas, prediction_from_lambdas, prediction_from_matrix
from src.utils import is_placeholder_team, json_dumps, matrix_to_wdl


@dataclass
class ModelState:
    schedule: pd.DataFrame
    historical: pd.DataFrame
    current_worldcup: pd.DataFrame
    effective_results: pd.DataFrame
    elo_ratings: dict[str, float]
    team_stats: dict[str, dict[str, float]]
    training_data_start_date: str
    training_data_end_date: str
    number_of_historical_matches_used: int
    number_of_current_worldcup_matches_used: int
    older_data_priors_used: bool
    as_of_date: str
    as_of_exclusive: bool


def build_state(cfg: Config | None = None, as_of: date | str | None = None, exclusive: bool = False) -> ModelState:
    cfg = cfg or Config()
    mapper = TeamNameMapper(cfg.team_mapping_path)
    schedule = load_schedule(cfg.schedule_path, mapper)
    historical_path = cfg.clean_historical_results_path if cfg.clean_historical_results_path.exists() else cfg.historical_results_path
    historical = load_historical_results(historical_path, mapper)
    as_of_ts = pd.to_datetime(as_of, errors="coerce") if as_of is not None else pd.NaT
    as_of_date = None if pd.isna(as_of_ts) else as_of_ts.date()
    current_wc = played_worldcup_results(schedule)
    effective = effective_results(historical, current_wc, cfg, as_of=as_of_date, exclusive=exclusive)
    teams = set(schedule["home_team"]) | set(schedule["away_team"]) | set(effective["home_team"]) | set(effective["away_team"])
    base_elo = load_elo_ratings(cfg, mapper, {str(team) for team in teams if team})
    elo_current = current_wc.copy()
    if as_of_date is not None and not elo_current.empty:
        dates = pd.to_datetime(elo_current["date"], errors="coerce")
        elo_current = elo_current[dates.lt(pd.Timestamp(as_of_date)) if exclusive else dates.le(pd.Timestamp(as_of_date))].copy()
    elo = update_dynamic_elo(base_elo, elo_current, cfg)
    stats = team_recent_stats(effective)
    if effective.empty:
        start = end = ""
    else:
        start = pd.to_datetime(effective["date"]).min().date().isoformat()
        end = pd.to_datetime(effective["date"]).max().date().isoformat()
    historical_used = len(effective) - len(elo_current)
    return ModelState(
        schedule=schedule,
        historical=historical,
        current_worldcup=elo_current,
        effective_results=effective,
        elo_ratings=elo,
        team_stats=stats,
        training_data_start_date=start,
        training_data_end_date=end,
        number_of_historical_matches_used=historical_used,
        number_of_current_worldcup_matches_used=len(elo_current),
        older_data_priors_used=cfg.use_older_data_as_prior,
        as_of_date=as_of_date.isoformat() if as_of_date is not None else "",
        as_of_exclusive=exclusive,
    )


def skip_reason(row: pd.Series) -> str | None:
    if row.get("status") == "played":
        return "already played"
    if row.get("status") in {"postponed", "cancelled"}:
        return f"status is {row.get('status')}"
    if is_placeholder_team(row.get("home_team")) or is_placeholder_team(row.get("away_team")):
        return "unresolved placeholder team"
    return None


def _head_to_head_summary(results: pd.DataFrame, home: str, away: str) -> dict[str, Any]:
    if results.empty:
        return {"matches": 0, "home_wins": 0, "draws": 0, "away_wins": 0}
    pair = results[
        ((results["home_team"].eq(home)) & (results["away_team"].eq(away)))
        | ((results["home_team"].eq(away)) & (results["away_team"].eq(home)))
    ].copy()
    home_wins = draws = away_wins = 0
    for row in pair.itertuples(index=False):
        home_score = int(row.home_score)
        away_score = int(row.away_score)
        listed_home = str(row.home_team)
        if home_score == away_score:
            draws += 1
        elif (home_score > away_score and listed_home == home) or (home_score < away_score and listed_home == away):
            home_wins += 1
        else:
            away_wins += 1
    return {"matches": int(len(pair)), "home_wins": home_wins, "draws": draws, "away_wins": away_wins}


def _has_team_totals_for_both_sides(team_total_signals: list[dict[str, Any]]) -> bool:
    sides = {bool(signal.get("team_is_home")) for signal in team_total_signals}
    return sides == {False, True}


def predict_match_row(
    row: pd.Series,
    state: ModelState,
    cfg: Config | None = None,
    model_weight: float | None = None,
    market_weight: float | None = None,
    live: bool = False,
    no_markets: bool = False,
    refresh_markets: bool = False,
) -> dict[str, Any]:
    cfg = cfg or Config()
    home = str(row["home_team"])
    away = str(row["away_team"])
    global_home, global_away = global_goal_rates(state.effective_results)
    neutral = str(row.get("neutral", "true")).strip().lower() in {"true", "1", "yes", ""}
    lambda_home, lambda_away = estimate_lambdas(home, away, state.elo_ratings, state.team_stats, global_home, global_away, neutral, cfg)
    pred = prediction_from_lambdas(lambda_home, lambda_away, cfg.max_goals)
    matrix = pred.pop("matrix")
    model_probs = (pred["home_win_prob"], pred["draw_prob"], pred["away_win_prob"])
    market_used = False
    market_source = ""
    market_timestamp = ""
    market_age_minutes = ""
    raw_prices: dict[str, float] | str = ""
    normalized_market: dict[str, float] | str = ""
    market_conf = 0.0
    moneyline_used = False
    spread_used = False
    total_used = False
    team_total_used = False
    spread_info: dict[str, Any] = {}
    total_info: dict[str, Any] = {}
    team_total_info: dict[str, Any] = {}
    notes: list[str] = []
    total_calibration_weight_used = 0.0
    team_total_calibration_weight_used = 0.0
    spread_calibration_weight_used = 0.0
    mw = 1.0 if model_weight is None else model_weight
    kw = 0.0 if market_weight is None else market_weight
    if not no_markets and cfg.use_polymarket:
        polymarket = PolymarketClient(cfg)
        market_signals = polymarket.best_markets_for_match(home, away, str(row.get("date", "")), refresh=refresh_markets)
        pm = market_signals.get("moneyline")
        total_signals = market_signals.get("totals") or []
        team_total_signals = market_signals.get("team_totals") or []
        spread_signals = market_signals.get("spreads") or []
        has_both_team_totals = _has_team_totals_for_both_sides(team_total_signals)
        total_budget = 0.0 if has_both_team_totals else cfg.total_calibration_weight
        total_weight = min(total_budget / max(1, len(total_signals)), 0.08)
        team_total_weight = min(cfg.team_total_calibration_weight / max(1, len(team_total_signals)), 0.06)
        spread_weight = min(cfg.spread_calibration_weight / max(1, len(spread_signals)), 0.06)
        if total_signals and total_weight > 0:
            total_notes: list[str] = []
            for total_signal in total_signals:
                matrix = calibrate_total(
                    matrix,
                    float(total_signal["line"]),
                    float(total_signal["over_probability"]),
                    total_weight,
                )
                total_notes.append(f"{total_signal['line']}={float(total_signal['over_probability']):.1%}")
            total_used = True
            total_calibration_weight_used = total_budget
            total_info = total_signals[0] | {"lines_used": total_signals, "per_line_weight": total_weight}
            market_used = True
            market_source = "polymarket_gamma_events_clob"
            market_timestamp = datetime_now_iso()
            market_age_minutes = 0.0
            market_conf = max(market_conf, max(float(item.get("confidence", 0.0)) for item in total_signals))
            notes.append(
                "Polymarket totals calibrated: "
                + ", ".join(total_notes)
                + f" (weight each {total_weight:.2f})"
            )
        elif total_signals and has_both_team_totals:
            notes.append("Polymarket full-match totals skipped because team totals are available for both teams")
        if team_total_signals:
            team_total_notes: list[str] = []
            for team_total_signal in team_total_signals:
                matrix = calibrate_team_total(
                    matrix,
                    bool(team_total_signal["team_is_home"]),
                    float(team_total_signal["line"]),
                    float(team_total_signal["over_probability"]),
                    team_total_weight,
                )
                team_total_notes.append(
                    f"{team_total_signal['team']} {team_total_signal['line']}={float(team_total_signal['over_probability']):.1%}"
                )
            team_total_used = True
            team_total_calibration_weight_used = cfg.team_total_calibration_weight
            team_total_info = team_total_signals[0] | {"lines_used": team_total_signals, "per_line_weight": team_total_weight}
            market_used = True
            market_source = "polymarket_gamma_events_clob"
            market_timestamp = datetime_now_iso()
            market_age_minutes = 0.0
            market_conf = max(market_conf, max(float(item.get("confidence", 0.0)) for item in team_total_signals))
            notes.append(
                "Polymarket team totals calibrated: "
                + ", ".join(team_total_notes)
                + f" (weight each {team_total_weight:.2f})"
            )
        if spread_signals:
            spread_notes: list[str] = []
            for spread_signal in spread_signals:
                matrix = calibrate_spread(
                    matrix,
                    bool(spread_signal["team_is_home"]),
                    float(spread_signal["line"]),
                    float(spread_signal["cover_probability"]),
                    spread_weight,
                )
                spread_notes.append(
                    f"{spread_signal['team']} {float(spread_signal['line']):+g}={float(spread_signal['cover_probability']):.1%}"
                )
            spread_used = True
            spread_calibration_weight_used = cfg.spread_calibration_weight
            spread_info = spread_signals[0] | {"lines_used": spread_signals, "per_line_weight": spread_weight}
            market_used = True
            market_source = "polymarket_gamma_events_clob"
            market_timestamp = datetime_now_iso()
            market_age_minutes = 0.0
            market_conf = max(market_conf, max(float(item.get("confidence", 0.0)) for item in spread_signals))
            notes.append(
                "Polymarket spreads calibrated: "
                + ", ".join(spread_notes)
                + f" (weight each {spread_weight:.2f})"
            )
        if pm:
            market_probs = (
                pm["normalized"]["home_win"],
                pm["normalized"]["draw"],
                pm["normalized"]["away_win"],
            )
            mw = cfg.live_model_weight if live else cfg.model_weight
            kw = cfg.live_moneyline_market_weight if live else cfg.moneyline_market_weight
            if model_weight is not None:
                mw = model_weight
            if market_weight is not None:
                kw = market_weight
            blended = blend_moneyline(model_probs, market_probs, mw, kw)
            matrix = calibrate_wdl(matrix, blended)
            pred = prediction_from_matrix(matrix)
            matrix = pred.pop("matrix")
            market_used = True
            moneyline_used = True
            market_source = pm["source"]
            market_timestamp = pm["timestamp"]
            market_age_minutes = pm["age_minutes"]
            raw_prices = pm.get("raw_by_team", pm["raw"])
            normalized_market = pm.get("normalized_by_team", pm["normalized"])
            market_conf = max(market_conf, pm["confidence"])
            notes.append(f"automatic Polymarket moneyline used: {pm.get('title') or pm.get('slug')}")
        elif total_used or team_total_used or spread_used:
            pred = prediction_from_matrix(matrix)
            matrix = pred.pop("matrix")
    if not market_used:
        mw = 1.0 if model_weight is None else model_weight
        kw = 0.0 if market_weight is None else market_weight
        if not no_markets and cfg.use_polymarket:
            notes.append("no reliable automatic Polymarket moneyline found")
    pred["confidence"] = confidence_score(
        (pred["home_win_prob"], pred["draw_prob"], pred["away_win_prob"]),
        state.team_stats.get(home, {}).get("matches", 0.0),
        state.team_stats.get(away, {}).get("matches", 0.0),
        market_used,
        market_conf,
    )
    home_stats = state.team_stats.get(home, {})
    away_stats = state.team_stats.get(away, {})
    h2h = _head_to_head_summary(state.effective_results, home, away)
    pred["top_5_scorelines"] = json_dumps(pred["top_5_scorelines"])
    pred.update(
        {
            "match_id": row.get("match_id", ""),
            "date": row.get("date", ""),
            "kickoff_time": row.get("kickoff_time", ""),
            "stage": row.get("stage", ""),
            "group": row.get("group", ""),
            "home_team": home,
            "away_team": away,
            "status": row.get("status", ""),
            "model_weight": mw,
            "moneyline_market_weight": kw if moneyline_used else 0.0,
            "spread_calibration_weight": spread_calibration_weight_used,
            "total_calibration_weight": total_calibration_weight_used,
            "team_total_calibration_weight": team_total_calibration_weight_used,
            "market_data_used": market_used,
            "market_source": market_source,
            "market_timestamp": market_timestamp,
            "market_age_minutes": market_age_minutes,
            "moneyline_used": moneyline_used,
            "moneyline_raw_prices": json_dumps(raw_prices) if raw_prices else "",
            "moneyline_normalized_probabilities": json_dumps(normalized_market) if normalized_market else "",
            "spread_used": spread_used,
            "spread_team": spread_info.get("team", ""),
            "spread_line": spread_info.get("line", ""),
            "spread_price": spread_info.get("price", ""),
            "spread_interpretation": (
                f"{len(spread_info.get('lines_used', [spread_info]))} full-match spread line(s) calibrated"
                if spread_info
                else ""
            ),
            "spread_lines_used": json_dumps(spread_info.get("lines_used", [])) if spread_info else "",
            "spread_per_line_weight": spread_info.get("per_line_weight", ""),
            "total_used": total_used,
            "total_line": total_info.get("line", ""),
            "over_price": total_info.get("over_price", ""),
            "under_price": total_info.get("under_price", ""),
            "total_interpretation": (
                f"{len(total_info.get('lines_used', [total_info]))} full-match total line(s) calibrated"
                if total_info
                else ""
            ),
            "total_lines_used": json_dumps(total_info.get("lines_used", [])) if total_info else "",
            "total_per_line_weight": total_info.get("per_line_weight", ""),
            "team_total_used": team_total_used,
            "team_total_interpretation": (
                f"{len(team_total_info.get('lines_used', [team_total_info]))} team total line(s) calibrated"
                if team_total_info
                else ""
            ),
            "team_total_lines_used": json_dumps(team_total_info.get("lines_used", [])) if team_total_info else "",
            "team_total_per_line_weight": team_total_info.get("per_line_weight", ""),
            "market_match_confidence": market_conf,
            "market_type": "three_way_moneyline" if moneyline_used else "none",
            "training_data_start_date": state.training_data_start_date,
            "training_data_end_date": state.training_data_end_date,
            "number_of_historical_matches_used": state.number_of_historical_matches_used,
            "number_of_current_worldcup_matches_used": state.number_of_current_worldcup_matches_used,
            "home_team_matches_used": home_stats.get("matches", 0.0),
            "away_team_matches_used": away_stats.get("matches", 0.0),
            "home_team_weighted_matches": home_stats.get("weighted_matches", 0.0),
            "away_team_weighted_matches": away_stats.get("weighted_matches", 0.0),
            "home_team_goals_for": home_stats.get("goals_for", 0.0),
            "away_team_goals_for": away_stats.get("goals_for", 0.0),
            "home_team_goals_against": home_stats.get("goals_against", 0.0),
            "away_team_goals_against": away_stats.get("goals_against", 0.0),
            "head_to_head_matches_used": h2h["matches"],
            "head_to_head_home_wins": h2h["home_wins"],
            "head_to_head_draws": h2h["draws"],
            "head_to_head_away_wins": h2h["away_wins"],
            "older_data_priors_used": state.older_data_priors_used,
            "as_of_date": state.as_of_date,
            "as_of_exclusive": state.as_of_exclusive,
            "data_sources_used": "historical_results,schedule,provided_elo,dynamic_elo"
            + (",market_moneyline" if moneyline_used else "")
            + (",market_spread" if spread_used else "")
            + (",market_total" if total_used else "")
            + (",market_team_total" if team_total_used else ""),
            "team_total_data_used": team_total_used,
            "notes": "; ".join(notes),
        }
    )
    pred["predicted_score"] = f"{home} {pred['predicted_home_goals']} - {pred['predicted_away_goals']} {away}"
    return pred


def datetime_now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()


def predict_all(state: ModelState, cfg: Config | None = None, **kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for _, row in state.schedule.iterrows():
        reason = skip_reason(row)
        if reason:
            skipped.append({"match_id": str(row.get("match_id", "")), "home_team": str(row.get("home_team", "")), "away_team": str(row.get("away_team", "")), "reason": reason})
            continue
        rows.append(predict_match_row(row, state, cfg, **kwargs))
    return pd.DataFrame(rows), pd.DataFrame(skipped)

def prediction_report(pred: dict[str, Any]) -> str:
    top = json.loads(pred["top_5_scorelines"])
    lines = [
        "Match:",
        f"{pred['home_team']} vs {pred['away_team']}",
        "",
        "Date:",
        f"{pred['date']} {pred['kickoff_time']}",
        "",
        "Status:",
        str(pred["status"]),
        "",
        "Predicted score:",
        str(pred["predicted_score"]),
        "",
        "Win/draw/loss probabilities:",
        f"{pred['home_team']} win: {pred['home_win_prob']:.1%}",
        f"Draw: {pred['draw_prob']:.1%}",
        f"{pred['away_team']} win: {pred['away_win_prob']:.1%}",
        "",
        "Expected goals:",
        f"{pred['home_team']}: {pred['expected_home_goals']:.2f}",
        f"{pred['away_team']}: {pred['expected_away_goals']:.2f}",
        "",
        "Top 5 scorelines:",
    ]
    for idx, row in enumerate(top, start=1):
        lines.append(f"{idx}. {row['score']}: {row['probability']:.1%}")
    lines += [
        "",
        "Market data:",
        f"Moneyline used: {'yes' if pred['moneyline_used'] else 'no'}",
        f"Market source: {pred['market_source'] or 'none'}",
        f"Notes: {pred['notes'] or 'none'}",
        "",
        "Model weights:",
        f"Statistical model: {float(pred['model_weight']):.0%}",
        f"Moneyline market: {float(pred['moneyline_market_weight']):.0%}",
        "",
        "Confidence:",
        f"{pred['confidence']:.2f}",
        "",
        "Explanation:",
        "This transparent baseline combines recent weighted international results, dynamic Elo, attacking and defensive goal rates, and a Poisson score model. Market probabilities are optional and only used when provided by a reliable manual or public read-only source.",
    ]
    return "\n".join(lines)
