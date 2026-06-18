# World Cup 2026 Predictor

Local prediction dashboard for FIFA World Cup 2026 matches.

The intended workflow is:

1. Keep the local data files updated.
2. Start the dashboard.
3. Select a match and model/market weights.
4. The app fetches live Polymarket data, combines it with the local statistical model, and shows a prediction.
5. The Overview tab tracks saved predictions against played results.

## Run The Dashboard

```bash
python3 -m src.dashboard
```

Then open:

```text
http://127.0.0.1:8000
```

To stop the dashboard, press `Ctrl+C` in the terminal where it is running.
If it is running in the background on port `8000`, stop it with:

```bash
lsof -ti tcp:8000 | xargs kill
```

The dashboard uses:

- `data/manual/worldcup_2026_schedule.csv`
- `data/manual/elo_ratings.csv`
- `data/manual/team_name_mapping.csv`
- `data/manual/market_snapshots.csv`
- `data/manual/prediction_snapshots.csv`
- `data/processed/clean_historical_results.csv`
- live read-only Polymarket Gamma and CLOB APIs

The dashboard defaults to live weights:

```text
30% statistical model
70% Polymarket moneyline
```

You can change the weights in the dashboard before clicking predict.

## Prediction Logic

For each selected scheduled match, the program:

1. Loads cleaned historical international results.
2. Adds any `status=played` World Cup 2026 matches from the schedule as current tournament data.
3. Loads the provided Elo ratings.
4. Updates Elo dynamically with played World Cup 2026 matches.
5. Estimates attacking and defensive strength from recent weighted results.
6. Builds a Poisson score model.
7. Fetches Polymarket moneyline, full-match totals, team totals, and spread markets when available.
8. Uses totals, team totals, and spreads to calibrate the score matrix.
9. Blends win/draw/loss probabilities with the standard CLI default weights:

```text
40% statistical model
60% Polymarket moneyline
```

For the live dashboard moneyline blend, the default remains:

```text
30% statistical model
70% Polymarket moneyline
```

The current conservative score-matrix market weights are:

```text
Full-match totals: 20%
Team totals:       40%
Spreads:           30%
```

When team-total markets are available for both teams, the model skips
full-match totals for that prediction to avoid double-counting the same goal
volume signal. Full-match totals remain a fallback when team totals are missing
or incomplete.

The schedule is the source of truth for World Cup 2026 state. When a match is played, edit:

```text
status=played
home_score=<home goals>
away_score=<away goals>
```

Future predictions for both teams will then include that played match.

For historical evaluation, the CLI uses an as-of boundary so later matches are
not allowed to influence earlier predictions.

## Polymarket Fetching

The app does not search odds directly by team name.

For a match such as `M001`, Mexico vs South Africa on `2026-06-11`, it:

1. Builds likely World Cup event slugs from team codes and the date, for example:

```text
fifwc-mex-rsa-2026-06-11
```

2. Fetches the Gamma event:

```text
https://gamma-api.polymarket.com/events/slug/fifwc-mex-rsa-2026-06-11
```

3. Reads the event's markets and parses:

```text
outcomes
outcomePrices
clobTokenIds
```

4. Fetches live CLOB order books using each token ID:

```text
https://clob.polymarket.com/book?token_id=<clobTokenId>
```

5. Uses best bid and best ask midpoint when the spread is acceptable.

For `M001`, the useful moneyline markets are the three binary outcome markets:

- Will Mexico win?
- Will the match end in a draw?
- Will South Africa win?

The model normalizes those three implied probabilities into home/draw/away probabilities.
Full-match over/under, team-total, and spread markets are also parsed when
available. Team totals shape each team's goal distribution. Full-match totals
push probability mass toward lower or higher aggregate scorelines only when
team totals are missing or incomplete. Spreads push probability mass toward
scorelines where the selected team covers the line. The final moneyline
calibration is applied last, then the displayed score is selected as the
highest-probability exact scoreline.

The market-line weights are conservative calibrated defaults from the first
played-match diagnostics, not final optimized values. Use the backtest commands
below to evaluate the statistical model and saved market snapshots as more
matches are played.

If exact slug lookup fails, the client falls back to paginated Gamma event discovery:

```text
https://gamma-api.polymarket.com/events?active=true&closed=false&limit=100&offset=...
```

## Useful Commands

Validate local data:

```bash
python3 -m src.cli validate-data
```

Refresh tournament state files:

```bash
python3 -m src.cli update-state
```

Predict one match in the terminal:

```bash
python3 -m src.cli predict-match --match-id M001 --refresh-markets
```

Debug Polymarket matching:

```bash
python3 -m src.cli debug-polymarket --match-id M001
```

Run a walk-forward backtest of the statistical model:

```bash
python3 -m src.cli backtest --max-matches 250 --min-training-matches 250
```

This backtest does not include Polymarket. It evaluates the statistical model
using historical results only.

Evaluate saved pre-match market snapshots:

```bash
python3 -m src.cli market-backtest
```

Run the current moneyline weight calibration diagnostic:

```bash
python3 -m src.cli calibrate-weights --max-match-id M020 --exclude M001 M002 M008
```

This compares model-only W/D/L probabilities with saved Polymarket moneyline
probabilities and searches the model/market blend by log loss and Brier score.
Treat it as a diagnostic until there are at least 30 played matches with saved
market snapshots.

Run the market-line weight calibration diagnostic:

```bash
python3 -m src.cli calibrate-market-weights --max-match-id M020 --exclude M001 M002 M008
```

This searches the full-match total, team-total, and spread calibration weights
against the saved pre-match market snapshots. It evaluates exact-score hit rate,
average goal error, exact-score log loss, and W/D/L log loss. Keep this
diagnostic separate from the moneyline blend because these line markets reshape
the score matrix, while moneyline mainly controls W/D/L probabilities.

The market-line calibration report includes a raw best result, a regularized
best result, and recommended shrunk weights. The shrunk recommendation blends
the data-fitted result back toward the current defaults so that a small match
sample cannot force extreme changes.

The dashboard updates `data/manual/market_snapshots.csv` automatically after
each successful prediction. Predicting the same match again replaces the old
market rows for that match with the latest fetched Polymarket prices. The file
stores moneyline, full-match total, team-total, and spread rows when available.

It also updates `data/manual/prediction_snapshots.csv` with the displayed
prediction for the match. The dashboard Overview tab compares saved predictions
against played results once scores are entered in the schedule, including a
small hit-rate chart for winner, exact score, and goal-difference accuracy.

Regenerate cleaned historical data from the local raw file:

```bash
python3 scripts/ingest_historical_results.py
```

## Data Sources

This project uses the following external data sources:

- Elo ratings: https://www.eloratings.net/2026_World_Cup
- World Cup 2026 schedule and results: https://www.matchesio.com/competition/world-cup/?season=374&season_key=2026
- Historical international results: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017?resource=download

## Data Policy

The `data/` folder is local and ignored by Git, except any files that were already intentionally tracked before. Do not commit Kaggle credentials, `kaggle.json`, raw downloads, generated predictions, or secrets.
