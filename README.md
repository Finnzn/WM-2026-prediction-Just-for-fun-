# World Cup 2026 Predictor

Local prediction dashboard for FIFA World Cup 2026 matches.

The intended workflow is:

1. Keep the local data files updated.
2. Start the dashboard.
3. Select a match and model/market weights.
4. The app fetches live Polymarket data, combines it with the local statistical model, and shows a prediction.

## Run The Dashboard

```bash
python3 -m src.dashboard
```

Then open:

```text
http://127.0.0.1:8000
```

The dashboard uses:

- `data/manual/worldcup_2026_schedule.csv`
- `data/manual/elo_ratings.csv`
- `data/manual/team_name_mapping.csv`
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
7. Fetches Polymarket moneyline markets.
8. Blends probabilities with the standard default weights:

```text
40% statistical model
60% Polymarket moneyline
```

The schedule is the source of truth for World Cup 2026 state. When a match is played, edit:

```text
status=played
home_score=<home goals>
away_score=<away goals>
```

Future predictions for both teams will then include that played match.

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

For `M001`, the useful markets are the three binary moneyline markets:

- Will Mexico win?
- Will the match end in a draw?
- Will South Africa win?

The model normalizes those three implied probabilities into home/draw/away probabilities.

Over/under and spread markets are discovered and classified by the Polymarket
client, but current predictions only use moneyline odds. Totals should calibrate
the score matrix and expected total goals rather than directly replace W/D/L
probabilities, so they are intentionally kept out of the main prediction blend
until their matching and calibration are reviewed.

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

Regenerate cleaned historical data from the local raw file:

```bash
python3 scripts/ingest_historical_results.py
```

## Data Policy

The `data/` folder is local and ignored by Git, except any files that were already intentionally tracked before. Do not commit Kaggle credentials, `kaggle.json`, raw downloads, generated predictions, or secrets.
