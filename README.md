# WM-2026-prediction-Just-for-fun-

## World Cup 2026 Predictor

This is a free, local, transparent prediction project for a private FIFA World
Cup 2026 prediction game. It predicts scheduled matches from local CSV files
using a baseline that combines:

- recent international results
- current World Cup 2026 played matches from the schedule CSV
- dynamic Elo ratings
- recent attacking and defensive goal rates
- a Poisson exact-score model
- optional manual/read-only Polymarket probability inputs

It does not implement gambling, trading, staking, wallet access, private keys,
order placement, bookmaker scraping, Bet365 scraping, or authenticated market
endpoints.

## Data Files

The main local files are:

```text
data/raw/historical_results.csv
data/manual/worldcup_2026_schedule.csv
data/manual/elo_ratings.csv
data/manual/team_name_mapping.csv
data/manual/polymarket_market_mapping.csv
data/manual/manual_market_probs.csv
```

Optional files may be empty or missing. The project still runs without
Polymarket or manual market data.

Example templates live in `examples/`.

## Schedule As Tournament State

The schedule CSV is the source of truth for World Cup 2026 state. Before a
match, set `status=scheduled` and leave scores empty. After a match, enter
`home_score`, `away_score`, and set `status=played`.

Played World Cup matches are appended to the effective training history with a
high weight, so future predictions react to tournament form, current goals
scored/conceded, and dynamic Elo updates. Played matches are skipped by normal
prediction commands. Fixtures with placeholders such as `TBD`, `Winner Group A`,
or `Runner-up Group B` are skipped until actual teams are known.

## Why Four Years

By default, historical results use only the last four years. Football is
non-stationary: squads age, coaches change, tactical systems shift, and national
teams go through player-generation cycles. Older data can become a weak prior,
but it is ignored by default to reduce regime shift and concept drift.

## Commands

Validate local files:

```bash
python -m src.cli validate-data
```

Update tournament state and dynamic Elo:

```bash
python -m src.cli update-state
```

Predict all unplayed known-team matches:

```bash
python -m src.cli predict-all
```

Predict one match:

```bash
python -m src.cli predict-match --match-id M041
```

Right before kickoff, with fresh public Polymarket candidate discovery:

```bash
python -m src.cli predict-match --match-id M041 --refresh-markets --live
```

Show read-only Polymarket candidates for manual review:

```bash
python -m src.cli market-candidates --match-id M041
```

Run a simple time-safe backtest:

```bash
python -m src.cli backtest
```

Export predictions:

```bash
python -m src.cli export
```

Outputs are written to `outputs/predictions.csv`, `outputs/predictions.json`,
and `outputs/predictions.xlsx` when `openpyxl` is installed.

## Markets

Polymarket is optional and read-only. Moneyline, spread, and total-goals markets
are different signals:

- Moneyline can inform win/draw/loss probabilities.
- Spread calibrates goal-difference probabilities.
- Total goals calibrates expected total goals.

The model does not mix these signals directly. Low-confidence automatic market
matches are reported but not used automatically; add them to the manual mapping
file after review. Manual probabilities in `manual_market_probs.csv` take
priority when available.

## Exact Scores And Confidence

Exact-score predictions are inherently uncertain because football is low-scoring
and high-variance. The confidence score summarizes favorite strength, data
depth, and optional market agreement; it is not a guarantee.

## Historical Data Ingestion

Before training a model, create the cleaned historical results file:

```bash
python3 scripts/ingest_historical_results.py
```

The script first uses `data/raw/historical_results.csv` when it already exists.
If the raw file is missing, KaggleHub download mode is opt-in:

```bash
python3 scripts/ingest_historical_results.py --download-if-missing
```

or:

```bash
DOWNLOAD_HISTORICAL_DATA=true python3 scripts/ingest_historical_results.py
```

The KaggleHub dataset is `martj42/international-football-results-from-1872-to-2017`.
Public datasets may download without a Kaggle API key. If Kaggle requires login,
authenticate with KaggleHub or provide `KAGGLE_USERNAME` and `KAGGLE_KEY` in your
environment. Never hardcode Kaggle credentials and never commit `kaggle.json`.

Cleaned output is written to:

```text
data/processed/clean_historical_results.csv
```

By default, only matches from the last four years are kept. To keep older matches
as weak priors, run:

```bash
USE_OLDER_DATA_AS_PRIOR=true python3 scripts/ingest_historical_results.py
```

Team names are normalized with `data/manual/team_name_mapping.csv` when mappings
exist. The World Cup 2026 schedule source of truth is
`data/manual/worldcup_2026_schedule.csv`; scheduled or unplayed 2026 fixtures are
excluded from historical training data.

To normalize team/country names across the local schedule, ELO, and cleaned
historical CSVs after editing the mapping file, run:

```bash
python3 scripts/normalize_country_names.py
```
