from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data_sources.historical_results import load_historical_results
from src.data_sources.polymarket import PolymarketClient, PolymarketDebugCandidate
from src.data_sources.schedule import load_schedule, validate_schedule
from src.data_sources.team_mapping import TeamNameMapper
from src.models.backtesting import simple_backtest
from src.models.predictor import build_state, predict_all, predict_match_row, prediction_report, skip_reason


OUTPUT_COLUMNS = [
    "match_id", "date", "kickoff_time", "stage", "group", "home_team", "away_team", "status",
    "predicted_home_goals", "predicted_away_goals", "predicted_score", "expected_home_goals",
    "expected_away_goals", "home_win_prob", "draw_prob", "away_win_prob", "top_5_scorelines",
    "confidence", "model_weight", "moneyline_market_weight", "spread_calibration_weight",
    "total_calibration_weight", "market_data_used", "market_source", "market_timestamp",
    "market_age_minutes", "moneyline_used", "moneyline_raw_prices",
    "moneyline_normalized_probabilities", "spread_used", "spread_team", "spread_line",
    "spread_price", "spread_interpretation", "total_used", "total_line", "over_price",
    "under_price", "total_interpretation", "market_match_confidence", "market_type",
    "training_data_start_date", "training_data_end_date", "number_of_historical_matches_used",
    "number_of_current_worldcup_matches_used", "older_data_priors_used", "data_sources_used", "notes",
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
    optional_paths = [cfg.team_mapping_path, cfg.polymarket_mapping_path, cfg.manual_market_probs_path]
    if not cfg.elo_ratings_path.exists():
        optional_paths.append(cfg.elo_ratings_path)
    for optional in optional_paths:
        if not optional.exists():
            warnings.append(f"Optional file missing: {optional}")
    if cfg.schedule_path.exists():
        schedule = load_schedule(cfg.schedule_path, mapper)
        schedule_errors, schedule_warnings = validate_schedule(schedule)
        errors.extend(schedule_errors)
        warnings.extend(schedule_warnings)
    if hist_path.exists():
        historical = load_historical_results(hist_path, mapper)
        if historical.empty:
            errors.append(f"No usable historical rows in {hist_path}")
    if cfg.manual_market_probs_path.exists():
        df = pd.read_csv(cfg.manual_market_probs_path, dtype=str, keep_default_na=False)
        expected = {"match_id", "home_win_prob", "draw_prob", "away_win_prob"}
        missing = expected - set(df.columns)
        if missing:
            errors.append(f"manual_market_probs missing columns: {sorted(missing)}")
        for row in df.itertuples(index=False):
            vals = [float(getattr(row, col) or 0) for col in ["home_win_prob", "draw_prob", "away_win_prob"]]
            if abs(sum(vals) - 1.0) > 0.05 and abs(sum(vals) - 100.0) > 5:
                warnings.append(f"Manual market probabilities for {getattr(row, 'match_id', '?')} do not sum near 1 or 100")
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


def backtest_command(_: argparse.Namespace) -> int:
    cfg = Config()
    mapper = TeamNameMapper(cfg.team_mapping_path)
    path = cfg.clean_historical_results_path if cfg.clean_historical_results_path.exists() else cfg.historical_results_path
    results = load_historical_results(path, mapper)
    metrics = simple_backtest(results)
    print(json.dumps(metrics, indent=2))
    return 0


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
    sub.add_parser("backtest").set_defaults(func=backtest_command)
    sub.add_parser("export").set_defaults(func=export_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
