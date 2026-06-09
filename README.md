# WM-2026-prediction-Just-for-fun-

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
