from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data_sources.historical_results import load_historical_results
from src.data_sources.polymarket import PolymarketClient, PolymarketDebugCandidate
from src.data_sources.schedule import load_schedule, validate_schedule
from src.data_sources.team_mapping import TeamNameMapper
from src.models.backtesting import evaluate_market_snapshots, simple_backtest, walk_forward_model_backtest
from src.models.predictor import build_state, predict_all, predict_match_row, prediction_report, skip_reason
from src.models.poisson_model import estimate_lambdas, prediction_from_lambdas, prediction_from_matrix
from src.markets.moneyline import blend_moneyline
from src.markets.spread import calibrate_spread
from src.markets.totals import calibrate_team_total, calibrate_total
from src.markets.wdl import calibrate_wdl
from src.utils import normalize_probabilities


OUTPUT_COLUMNS = [
    "match_id", "date", "kickoff_time", "stage", "group", "home_team", "away_team", "status",
    "predicted_home_goals", "predicted_away_goals", "predicted_score", "expected_home_goals",
    "expected_away_goals", "home_win_prob", "draw_prob", "away_win_prob", "top_5_scorelines",
    "confidence", "model_weight", "moneyline_market_weight", "spread_calibration_weight",
    "total_calibration_weight", "team_total_calibration_weight", "market_data_used", "market_source", "market_timestamp",
    "market_age_minutes", "moneyline_used", "moneyline_raw_prices",
    "moneyline_normalized_probabilities", "spread_used", "spread_team", "spread_line",
    "spread_price", "spread_interpretation", "spread_lines_used", "spread_per_line_weight",
    "total_used", "total_line", "over_price", "under_price", "total_interpretation",
    "total_lines_used", "total_per_line_weight", "team_total_used", "team_total_interpretation",
    "team_total_lines_used", "team_total_per_line_weight", "team_total_data_used",
    "market_match_confidence", "market_type",
    "training_data_start_date", "training_data_end_date", "number_of_historical_matches_used",
    "number_of_current_worldcup_matches_used", "home_team_matches_used", "away_team_matches_used",
    "home_team_weighted_matches", "away_team_weighted_matches", "home_team_goals_for",
    "away_team_goals_for", "home_team_goals_against", "away_team_goals_against",
    "head_to_head_matches_used", "head_to_head_home_wins", "head_to_head_draws",
    "head_to_head_away_wins", "older_data_priors_used", "as_of_date", "as_of_exclusive",
    "data_sources_used", "notes",
]


def _print_list(title: str, rows: list[str]) -> None:
    if rows:
        print(title)
        for row in rows:
            print(f"- {row}")


def validate_data(_: argparse.Namespace) -> int:
    cfg = Config()
    mapper = TeamNameMapper(cfg.team_mapping_path)
    required = [cfg.schedule_path]
    hist_path = cfg.clean_historical_results_path if cfg.clean_historical_results_path.exists() else cfg.historical_results_path
    required.append(hist_path)
    errors: list[str] = []
    warnings: list[str] = []
    for path in required:
        if not path.exists():
            errors.append(f"Missing required file: {path}")
    if not cfg.elo_ratings_path.exists():
        warnings.append(f"Optional file missing: {cfg.elo_ratings_path}")
    if cfg.schedule_path.exists():
        schedule = load_schedule(cfg.schedule_path, mapper)
        schedule_errors, schedule_warnings = validate_schedule(schedule)
        errors.extend(schedule_errors)
        warnings.extend(schedule_warnings)
    if hist_path.exists():
        historical = load_historical_results(hist_path, mapper)
        if historical.empty:
            errors.append(f"No usable historical rows in {hist_path}")
    _print_list("Errors:", errors)
    _print_list("Warnings:", warnings)
    print("Data validation passed." if not errors else "Data validation failed.")
    return 1 if errors else 0


def update_state(_: argparse.Namespace) -> int:
    state = build_state()
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    state.schedule.to_csv("data/processed/tournament_state.csv", index=False)
    state.effective_results.to_csv("data/processed/effective_results.csv", index=False)
    elo_rows = pd.DataFrame([{"team": team, "elo": elo} for team, elo in sorted(state.elo_ratings.items())])
    elo_rows.to_csv("data/processed/dynamic_elo_ratings.csv", index=False)
    print("Tournament state updated.")
    print(f"Played World Cup matches used: {state.number_of_current_worldcup_matches_used}")
    print(f"Historical matches used: {state.number_of_historical_matches_used}")
    print("Wrote data/processed/tournament_state.csv")
    print("Wrote data/processed/effective_results.csv")
    print("Wrote data/processed/dynamic_elo_ratings.csv")
    return 0


def _predict_kwargs(args: argparse.Namespace) -> dict:
    return {
        "model_weight": args.model_weight,
        "market_weight": args.market_weight,
        "live": getattr(args, "live", False),
        "no_markets": getattr(args, "no_markets", False),
        "refresh_markets": getattr(args, "refresh_markets", False),
    }


def predict_all_command(args: argparse.Namespace) -> int:
    state = build_state()
    predictions, skipped = predict_all(state, Config(), **_predict_kwargs(args))
    Path("outputs").mkdir(exist_ok=True)
    predictions = predictions.reindex(columns=OUTPUT_COLUMNS)
    predictions.to_csv("outputs/predictions.csv", index=False)
    predictions.to_json("outputs/predictions.json", orient="records", indent=2, force_ascii=False)
    print(f"Predicted {len(predictions)} matches.")
    if not skipped.empty:
        print(f"Skipped {len(skipped)} matches:")
        for row in skipped.itertuples(index=False):
            print(f"- {row.match_id}: {row.home_team} vs {row.away_team} ({row.reason})")
    print("Wrote outputs/predictions.csv and outputs/predictions.json")
    return 0


def predict_match_command(args: argparse.Namespace) -> int:
    state = build_state()
    rows = state.schedule[state.schedule["match_id"].eq(args.match_id)]
    if rows.empty:
        print(f"No match found with match_id={args.match_id}")
        return 1
    row = rows.iloc[0]
    reason = skip_reason(row)
    if reason and not args.backtest_played:
        print(f"Skipping {args.match_id}: {reason}")
        return 1
    if args.backtest_played:
        state = build_state(as_of=row.get("date", ""), exclusive=True)
        rows = state.schedule[state.schedule["match_id"].eq(args.match_id)]
        row = rows.iloc[0]
    pred = predict_match_row(row, state, Config(), **_predict_kwargs(args))
    print(prediction_report(pred))
    return 0


def market_candidates_command(args: argparse.Namespace) -> int:
    cfg = Config()
    state = build_state(cfg)
    rows = state.schedule[state.schedule["match_id"].eq(args.match_id)]
    if rows.empty:
        print(f"No match found with match_id={args.match_id}")
        return 1
    row = rows.iloc[0]
    client = PolymarketClient(cfg)
    report = client.discover_match_markets(
        row["home_team"],
        row["away_team"],
        row["date"],
        max_pages=args.max_pages,
        refresh=args.refresh_markets,
    )
    candidates = report.candidates[:15]
    if not candidates:
        print("No Polymarket candidates found.")
        return 0
    print(f"Events fetched: {report.events_fetched}")
    print(f"Markets inspected: {report.markets_inspected}")
    for idx, candidate in enumerate(candidates, start=1):
        print_candidate_summary(idx, candidate)
    return 0


def print_candidate_summary(idx: int, candidate: PolymarketDebugCandidate) -> None:
    print(f"{idx}. {candidate.event_title} / {candidate.market_question}")
    print(f"   event_slug={candidate.event_slug}")
    print(f"   market_slug={candidate.market_slug}")
    print(f"   fuzzy_score={candidate.fuzzy_score:.2f}")
    print(f"   category={candidate.category} type={candidate.market_type}")
    print(f"   accepted={'yes' if candidate.accepted else 'no'}")
    if candidate.rejected_reason:
        print(f"   rejected_reason={candidate.rejected_reason}")
    if candidate.spread_line is not None:
        print(f"   spread_line={candidate.spread_line}")
    if candidate.total_line is not None:
        print(f"   total_line={candidate.total_line}")
    print(f"   reasons={'; '.join(candidate.reasons) or 'none'}")
    print(f"   outcomes={candidate.outcomes}")
    print(f"   clobTokenIds={candidate.clob_token_ids}")
    print(f"   Gamma outcomePrices={candidate.gamma_prices}")
    for token in candidate.tokens:
        print(
            "   CLOB "
            f"{token.outcome}: bid={token.bid} ask={token.ask} midpoint={token.midpoint} "
            f"spread={token.spread} last_trade={token.last_trade_price} "
            f"implied={token.implied_probability} source={token.price_source}"
        )
        if token.book_error:
            print(f"      book_error={token.book_error}")


def debug_polymarket_command(args: argparse.Namespace) -> int:
    cfg = Config()
    state = build_state(cfg)
    rows = state.schedule[state.schedule["match_id"].eq(args.match_id)]
    if rows.empty:
        print(f"No match found with match_id={args.match_id}")
        return 1
    row = rows.iloc[0]
    client = PolymarketClient(cfg)
    report = client.discover_match_markets(
        row["home_team"],
        row["away_team"],
        row["date"],
        max_pages=args.max_pages,
        include_debug_discovery=True,
        refresh=True,
    )
    print("Request URLs:")
    for url in report.request_urls:
        print(f"- {url}")
    if report.request_errors:
        print("")
        print("Request errors:")
        for error in report.request_errors:
            print(f"- {error}")
    print("")
    print(f"Events fetched: {report.events_fetched}")
    print(f"Markets inspected: {report.markets_inspected}")
    print(f"Sports hints seen: {report.sports_seen or 'none'}")
    print(f"Relevant tags seen: {report.tags_seen or 'none'}")
    print("")
    print("Top 10 candidate markets:")
    for idx, candidate in enumerate(report.candidates[:10], start=1):
        print_candidate_summary(idx, candidate)
    return 0


def backtest_command(args: argparse.Namespace) -> int:
    cfg = Config()
    mapper = TeamNameMapper(cfg.team_mapping_path)
    path = cfg.clean_historical_results_path if cfg.clean_historical_results_path.exists() else cfg.historical_results_path
    results = load_historical_results(path, mapper)
    if args.baseline:
        metrics = simple_backtest(results)
    else:
        metrics = walk_forward_model_backtest(
            results,
            cfg,
            min_training_matches=args.min_training_matches,
            max_evaluated_matches=args.max_matches,
        )
    print(json.dumps(metrics, indent=2))
    return 0


def market_backtest_command(_: argparse.Namespace) -> int:
    cfg = Config()
    if not cfg.market_snapshots_path.exists():
        print(f"Missing market snapshot file: {cfg.market_snapshots_path}")
        return 1
    mapper = TeamNameMapper(cfg.team_mapping_path)
    schedule = load_schedule(cfg.schedule_path, mapper)
    snapshots = pd.read_csv(cfg.market_snapshots_path, dtype={"match_id": str, "market_type": str, "team": str})
    snapshots["probability"] = pd.to_numeric(snapshots["probability"], errors="coerce")
    snapshots["line"] = pd.to_numeric(snapshots["line"], errors="coerce")
    snapshots = snapshots.dropna(subset=["match_id", "market_type", "team", "probability"])
    metrics = evaluate_market_snapshots(snapshots, schedule)
    print(json.dumps(metrics, indent=2))
    return 0


def calibrate_weights_command(args: argparse.Namespace) -> int:
    cfg = Config()
    mapper = TeamNameMapper(cfg.team_mapping_path)
    if not cfg.market_snapshots_path.exists():
        print(f"Missing market snapshot file: {cfg.market_snapshots_path}")
        return 1

    schedule = load_schedule(cfg.schedule_path, mapper)
    snapshots = pd.read_csv(cfg.market_snapshots_path, dtype=str, keep_default_na=False)
    snapshots["probability"] = pd.to_numeric(snapshots["probability"], errors="coerce")
    rows = _moneyline_calibration_rows(schedule, snapshots, cfg, args.max_match_id, set(args.exclude))
    grid = [_blend_metrics(rows, market_weight / 100) for market_weight in range(0, 101, args.step)]
    fine = [_blend_metrics(rows, market_weight / 100) for market_weight in range(101)]
    best_log_loss = min(fine, key=lambda row: row["log_loss"]) if fine else {}
    best_brier = min(fine, key=lambda row: row["brier"]) if fine else {}
    report = {
        "scope": {
            "max_match_id": args.max_match_id,
            "excluded_match_ids": sorted(args.exclude),
            "matches_used": len(rows),
            "match_ids_used": [row["match_id"] for row in rows],
        },
        "baselines": {
            "model_only": _blend_metrics(rows, 0.0),
            "current_live_90_market": _blend_metrics(rows, 0.9),
            "market_only": _blend_metrics(rows, 1.0),
        },
        "best_by_log_loss": best_log_loss,
        "best_by_brier": best_brier,
        "grid": grid,
        "warning": (
            "Diagnostic only. Fewer than 30 matches is too small for a stable default-weight change."
            if len(rows) < 30
            else "Use as calibration evidence, but validate against future matches before changing defaults."
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def calibrate_market_weights_command(args: argparse.Namespace) -> int:
    cfg = Config()
    mapper = TeamNameMapper(cfg.team_mapping_path)
    if not cfg.market_snapshots_path.exists():
        print(f"Missing market snapshot file: {cfg.market_snapshots_path}")
        return 1

    schedule = load_schedule(cfg.schedule_path, mapper)
    snapshots = pd.read_csv(cfg.market_snapshots_path, dtype=str, keep_default_na=False)
    snapshots["probability"] = pd.to_numeric(snapshots["probability"], errors="coerce")
    snapshots["line"] = pd.to_numeric(snapshots["line"], errors="coerce")
    matches = _market_calibration_matches(schedule, snapshots, cfg, args.max_match_id, set(args.exclude))
    candidates = _float_candidates(args.candidates)
    results = []
    for total_weight in candidates:
        for team_total_weight in candidates:
            for spread_weight in candidates:
                results.append(
                    _score_market_line_weights(
                        matches,
                        total_weight=total_weight,
                        team_total_weight=team_total_weight,
                        spread_weight=spread_weight,
                        moneyline_market_weight=args.moneyline_market_weight,
                    )
                )
    best_score = min(results, key=lambda row: (row["score_log_loss"], row["average_goal_error"])) if results else {}
    best_goal_error = min(results, key=lambda row: (row["average_goal_error"], row["score_log_loss"])) if results else {}
    prior = {
        "total_calibration_weight": cfg.total_calibration_weight,
        "team_total_calibration_weight": cfg.team_total_calibration_weight,
        "spread_calibration_weight": cfg.spread_calibration_weight,
    }
    regularized = [
        row
        | {
            "regularized_objective": _regularized_market_line_objective(
                row,
                prior,
                args.regularization_strength,
            )
        }
        for row in results
    ]
    best_regularized = min(regularized, key=lambda row: (row["regularized_objective"], row["score_log_loss"])) if regularized else {}
    recommended_weights = _shrunk_market_line_weights(best_regularized, prior, len(matches), args.shrinkage_matches)
    recommended = _score_market_line_weights(
        matches,
        total_weight=recommended_weights["total_calibration_weight"],
        team_total_weight=recommended_weights["team_total_calibration_weight"],
        spread_weight=recommended_weights["spread_calibration_weight"],
        moneyline_market_weight=args.moneyline_market_weight,
    )
    current = _score_market_line_weights(
        matches,
        total_weight=cfg.total_calibration_weight,
        team_total_weight=cfg.team_total_calibration_weight,
        spread_weight=cfg.spread_calibration_weight,
        moneyline_market_weight=args.moneyline_market_weight,
    )
    report = {
        "scope": {
            "max_match_id": args.max_match_id,
            "excluded_match_ids": sorted(args.exclude),
            "matches_used": len(matches),
            "match_ids_used": [row["match_id"] for row in matches],
            "moneyline_market_weight_fixed_at": args.moneyline_market_weight,
            "regularization_strength": args.regularization_strength,
            "shrinkage_matches": args.shrinkage_matches,
        },
        "prior_weights": prior,
        "current_weights": current,
        "raw_best_by_score_log_loss": best_score,
        "best_by_goal_error": best_goal_error,
        "best_regularized": best_regularized,
        "recommended_shrunk_weights": recommended_weights,
        "recommended_shrunk_performance": recommended,
        "top_10_by_score_log_loss": sorted(results, key=lambda row: row["score_log_loss"])[:10],
        "top_10_regularized": sorted(regularized, key=lambda row: row["regularized_objective"])[:10],
        "warning": (
            "Diagnostic only. Fewer than 30 matches is too small for a stable market-line weight change."
            if len(matches) < 30
            else "Prefer the recommended shrunk weights over the raw optimum unless future matches confirm a larger move."
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _moneyline_calibration_rows(
    schedule: pd.DataFrame,
    snapshots: pd.DataFrame,
    cfg: Config,
    max_match_id: str,
    excluded: set[str],
) -> list[dict[str, object]]:
    played = schedule[schedule["status"].eq("played")].copy()
    played["match_num"] = pd.to_numeric(played["match_id"].str.extract(r"M(\d+)", expand=False), errors="coerce")
    max_num = int(str(max_match_id).lstrip("M"))
    played = played[played["match_num"].le(max_num) & ~played["match_id"].isin(excluded)].sort_values("match_num")
    rows: list[dict[str, object]] = []
    for match in played.itertuples(index=False):
        match_id = str(match.match_id)
        moneyline = snapshots[snapshots["match_id"].eq(match_id) & snapshots["market_type"].eq("moneyline")]
        if len(moneyline) < 3:
            continue

        state = build_state(cfg, as_of=str(match.date), exclusive=True)
        schedule_row = state.schedule[state.schedule["match_id"].eq(match_id)].iloc[0]
        prediction = predict_match_row(schedule_row, state, cfg, no_markets=True)
        model_probs = [
            float(prediction["home_win_prob"]),
            float(prediction["draw_prob"]),
            float(prediction["away_win_prob"]),
        ]

        home = str(match.home_team)
        away = str(match.away_team)
        raw_market = {str(row.team): float(row.probability) for row in moneyline.itertuples(index=False)}
        market_values = [raw_market.get(home), raw_market.get("Draw"), raw_market.get(away)]
        if any(value is None or pd.isna(value) for value in market_values):
            continue
        market_probs = normalize_probabilities([float(value) for value in market_values])

        home_score = int(match.home_score)
        away_score = int(match.away_score)
        actual_idx = 0 if home_score > away_score else 1 if home_score == away_score else 2
        rows.append(
            {
                "match_id": match_id,
                "actual_idx": actual_idx,
                "model_probs": model_probs,
                "market_probs": market_probs,
            }
        )
    return rows


def _market_calibration_matches(
    schedule: pd.DataFrame,
    snapshots: pd.DataFrame,
    cfg: Config,
    max_match_id: str,
    excluded: set[str],
) -> list[dict[str, object]]:
    played = schedule[schedule["status"].eq("played")].copy()
    played["match_num"] = pd.to_numeric(played["match_id"].str.extract(r"M(\d+)", expand=False), errors="coerce")
    max_num = int(str(max_match_id).lstrip("M"))
    played = played[played["match_num"].le(max_num) & ~played["match_id"].isin(excluded)].sort_values("match_num")
    matches: list[dict[str, object]] = []
    for match in played.itertuples(index=False):
        match_id = str(match.match_id)
        rows = snapshots[snapshots["match_id"].eq(match_id)].copy()
        if rows.empty:
            continue
        state = build_state(cfg, as_of=str(match.date), exclusive=True)
        schedule_row = state.schedule[state.schedule["match_id"].eq(match_id)].iloc[0]
        home = str(match.home_team)
        away = str(match.away_team)
        global_home, global_away = state.effective_results["home_score"].mean(), state.effective_results["away_score"].mean()
        neutral = str(schedule_row.get("neutral", "true")).strip().lower() in {"true", "1", "yes", ""}
        lambda_home, lambda_away = estimate_lambdas(
            home,
            away,
            state.elo_ratings,
            state.team_stats,
            max(0.4, float(global_home)),
            max(0.4, float(global_away)),
            neutral,
            cfg,
        )
        base = prediction_from_lambdas(lambda_home, lambda_away, cfg.max_goals)
        matrix = base.pop("matrix")
        matches.append(
            {
                "match_id": match_id,
                "home_team": home,
                "away_team": away,
                "home_score": int(match.home_score),
                "away_score": int(match.away_score),
                "matrix": matrix,
                "totals": [
                    (float(row.line), float(row.probability))
                    for row in rows[rows["market_type"].eq("total")].dropna(subset=["line", "probability"]).itertuples(index=False)
                ],
                "team_totals": [
                    (str(row.team), float(row.line), float(row.probability))
                    for row in rows[rows["market_type"].eq("team_total")].dropna(subset=["line", "probability"]).itertuples(index=False)
                ],
                "spreads": [
                    (str(row.team), float(row.line), float(row.probability))
                    for row in rows[rows["market_type"].eq("spread")].dropna(subset=["line", "probability"]).itertuples(index=False)
                ],
                "moneyline": {
                    str(row.team): float(row.probability)
                    for row in rows[rows["market_type"].eq("moneyline")].dropna(subset=["probability"]).itertuples(index=False)
                },
            }
        )
    return matches


def _score_market_line_weights(
    matches: list[dict[str, object]],
    total_weight: float,
    team_total_weight: float,
    spread_weight: float,
    moneyline_market_weight: float,
) -> dict[str, float]:
    if not matches:
        return _empty_market_line_score(total_weight, team_total_weight, spread_weight)
    exact = 0
    goal_error = 0.0
    score_log_loss = 0.0
    wdl_log_loss = 0.0
    wdl_brier = 0.0
    for match in matches:
        matrix = match["matrix"].copy()
        home = str(match["home_team"])
        away = str(match["away_team"])
        total_rows = match["totals"]
        team_total_rows = match["team_totals"]
        spread_rows = match["spreads"]

        team_total_sides = {team == home for team, _, _ in team_total_rows if team in {home, away}}
        has_both_team_totals = team_total_sides == {False, True}
        total_budget = 0.0 if has_both_team_totals else total_weight
        total_per_line = min(total_budget / max(1, len(total_rows)), 0.08)
        team_total_per_line = min(team_total_weight / max(1, len(team_total_rows)), 0.06)
        spread_per_line = min(spread_weight / max(1, len(spread_rows)), 0.06)

        if total_per_line > 0:
            for line, probability in total_rows:
                matrix = calibrate_total(matrix, line, probability, total_per_line)
        for team, line, probability in team_total_rows:
            if team not in {home, away}:
                continue
            matrix = calibrate_team_total(matrix, team == home, line, probability, team_total_per_line)
        for team, line, probability in spread_rows:
            if team not in {home, away}:
                continue
            matrix = calibrate_spread(matrix, team == home, line, probability, spread_per_line)

        raw = match["moneyline"]
        if len(raw) >= 3 and moneyline_market_weight > 0:
            values = [raw.get(home), raw.get("Draw"), raw.get(away)]
            if not any(value is None or pd.isna(value) for value in values):
                pred = prediction_from_matrix(matrix)
                model_probs = (float(pred["home_win_prob"]), float(pred["draw_prob"]), float(pred["away_win_prob"]))
                market_probs = normalize_probabilities([float(value) for value in values])
                target = blend_moneyline(model_probs, tuple(market_probs), 1 - moneyline_market_weight, moneyline_market_weight)
                matrix = calibrate_wdl(matrix, target)

        pred = prediction_from_matrix(matrix)
        home_score = int(match["home_score"])
        away_score = int(match["away_score"])
        actual_idx = 0 if home_score > away_score else 1 if home_score == away_score else 2
        wdl_probs = [float(pred["home_win_prob"]), float(pred["draw_prob"]), float(pred["away_win_prob"])]
        score_prob = float(matrix[home_score, away_score]) if home_score < matrix.shape[0] and away_score < matrix.shape[1] else 1e-12
        exact += int(int(pred["predicted_home_goals"]) == home_score and int(pred["predicted_away_goals"]) == away_score)
        goal_error += abs(float(pred["expected_home_goals"]) - home_score) + abs(float(pred["expected_away_goals"]) - away_score)
        score_log_loss += -math.log(max(1e-12, score_prob))
        wdl_log_loss += -math.log(max(1e-12, wdl_probs[actual_idx]))
        wdl_brier += sum((wdl_probs[i] - (1.0 if i == actual_idx else 0.0)) ** 2 for i in range(3))
    count = len(matches)
    return {
        "total_calibration_weight": round(total_weight, 3),
        "team_total_calibration_weight": round(team_total_weight, 3),
        "spread_calibration_weight": round(spread_weight, 3),
        "matches": float(count),
        "exact_score_hit_rate": exact / count,
        "average_goal_error": goal_error / count,
        "score_log_loss": score_log_loss / count,
        "wdl_log_loss": wdl_log_loss / count,
        "wdl_brier": wdl_brier / count,
    }


def _empty_market_line_score(total_weight: float, team_total_weight: float, spread_weight: float) -> dict[str, float]:
    return {
        "total_calibration_weight": round(total_weight, 3),
        "team_total_calibration_weight": round(team_total_weight, 3),
        "spread_calibration_weight": round(spread_weight, 3),
        "matches": 0.0,
        "exact_score_hit_rate": 0.0,
        "average_goal_error": 0.0,
        "score_log_loss": 0.0,
        "wdl_log_loss": 0.0,
        "wdl_brier": 0.0,
    }


def _regularized_market_line_objective(
    row: dict[str, float],
    prior: dict[str, float],
    regularization_strength: float,
) -> float:
    distance = sum(
        (float(row[key]) - float(prior[key])) ** 2
        for key in ["total_calibration_weight", "team_total_calibration_weight", "spread_calibration_weight"]
    )
    return float(row["score_log_loss"]) + regularization_strength * distance


def _shrunk_market_line_weights(
    target: dict[str, float],
    prior: dict[str, float],
    matches: int,
    shrinkage_matches: int,
) -> dict[str, float]:
    if not target:
        return {key: round(float(value), 3) for key, value in prior.items()}
    data_weight = matches / max(matches + shrinkage_matches, 1)
    return {
        key: round(float(prior[key]) * (1 - data_weight) + float(target[key]) * data_weight, 3)
        for key in ["total_calibration_weight", "team_total_calibration_weight", "spread_calibration_weight"]
    } | {"data_weight": round(data_weight, 3)}


def _float_candidates(text: str) -> list[float]:
    return sorted({float(item.strip()) for item in text.split(",") if item.strip()})


def _blend_metrics(rows: list[dict[str, object]], market_weight: float) -> dict[str, float]:
    if not rows:
        return {
            "model_weight": round(1 - market_weight, 2),
            "market_weight": round(market_weight, 2),
            "matches": 0.0,
            "accuracy": 0.0,
            "log_loss": 0.0,
            "brier": 0.0,
        }
    model_weight = 1 - market_weight
    correct = 0
    log_loss = 0.0
    brier = 0.0
    for row in rows:
        model_probs = row["model_probs"]
        market_probs = row["market_probs"]
        actual_idx = int(row["actual_idx"])
        probs = normalize_probabilities(
            [
                float(model_probs[i]) * model_weight + float(market_probs[i]) * market_weight
                for i in range(3)
            ]
        )
        correct += int(max(range(3), key=lambda index: probs[index]) == actual_idx)
        log_loss += -math.log(max(1e-12, probs[actual_idx]))
        brier += sum((probs[i] - (1.0 if i == actual_idx else 0.0)) ** 2 for i in range(3))
    count = len(rows)
    return {
        "model_weight": round(model_weight, 2),
        "market_weight": round(market_weight, 2),
        "matches": float(count),
        "accuracy": correct / count,
        "log_loss": log_loss / count,
        "brier": brier / count,
    }


def export_command(_: argparse.Namespace) -> int:
    path = Path("outputs/predictions.csv")
    if not path.exists():
        state = build_state()
        predictions, _ = predict_all(state, Config())
    else:
        predictions = pd.read_csv(path)
    Path("outputs").mkdir(exist_ok=True)
    predictions.to_csv("outputs/predictions.csv", index=False)
    predictions.to_json("outputs/predictions.json", orient="records", indent=2, force_ascii=False)
    try:
        predictions.to_excel("outputs/predictions.xlsx", index=False)
        print("Wrote outputs/predictions.xlsx")
    except Exception as exc:
        print(f"Skipped Excel export: {exc}")
    print("Wrote outputs/predictions.csv and outputs/predictions.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-data").set_defaults(func=validate_data)
    sub.add_parser("update-state").set_defaults(func=update_state)
    predict_all_parser = sub.add_parser("predict-all")
    predict_all_parser.add_argument("--refresh-markets", action="store_true")
    predict_all_parser.add_argument("--no-markets", action="store_true")
    predict_all_parser.add_argument("--market-weight", type=float, default=None)
    predict_all_parser.add_argument("--model-weight", type=float, default=None)
    predict_all_parser.add_argument("--use-spread", action="store_true")
    predict_all_parser.add_argument("--use-total", action="store_true")
    predict_all_parser.add_argument("--no-spread", action="store_true")
    predict_all_parser.add_argument("--no-total", action="store_true")
    predict_all_parser.set_defaults(func=predict_all_command)
    match_parser = sub.add_parser("predict-match")
    match_parser.add_argument("--match-id", required=True)
    match_parser.add_argument("--refresh-markets", action="store_true")
    match_parser.add_argument("--live", action="store_true")
    match_parser.add_argument("--no-markets", action="store_true")
    match_parser.add_argument("--market-weight", type=float, default=None)
    match_parser.add_argument("--model-weight", type=float, default=None)
    match_parser.add_argument("--backtest-played", action="store_true")
    match_parser.set_defaults(func=predict_match_command)
    candidates = sub.add_parser("market-candidates")
    candidates.add_argument("--match-id", required=True)
    candidates.add_argument("--refresh-markets", action="store_true")
    candidates.add_argument("--max-pages", type=int, default=30)
    candidates.set_defaults(func=market_candidates_command)
    debug_pm = sub.add_parser("debug-polymarket")
    debug_pm.add_argument("--match-id", required=True)
    debug_pm.add_argument("--max-pages", type=int, default=30)
    debug_pm.set_defaults(func=debug_polymarket_command)
    backtest = sub.add_parser("backtest")
    backtest.add_argument("--baseline", action="store_true")
    backtest.add_argument("--min-training-matches", type=int, default=250)
    backtest.add_argument("--max-matches", type=int, default=250)
    backtest.set_defaults(func=backtest_command)
    sub.add_parser("market-backtest").set_defaults(func=market_backtest_command)
    calibration = sub.add_parser("calibrate-weights")
    calibration.add_argument("--max-match-id", default="M020")
    calibration.add_argument("--exclude", nargs="*", default=["M001", "M002", "M008"])
    calibration.add_argument("--step", type=int, default=5)
    calibration.set_defaults(func=calibrate_weights_command)
    market_calibration = sub.add_parser("calibrate-market-weights")
    market_calibration.add_argument("--max-match-id", default="M020")
    market_calibration.add_argument("--exclude", nargs="*", default=["M001", "M002", "M008"])
    market_calibration.add_argument("--candidates", default="0,0.1,0.2,0.25,0.3,0.35,0.4")
    market_calibration.add_argument("--moneyline-market-weight", type=float, default=0.7)
    market_calibration.add_argument("--regularization-strength", type=float, default=0.25)
    market_calibration.add_argument("--shrinkage-matches", type=int, default=30)
    market_calibration.set_defaults(func=calibrate_market_weights_command)
    sub.add_parser("export").set_defaults(func=export_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
