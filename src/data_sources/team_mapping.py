from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.utils import snake_case


DEFAULT_ALIASES = {
    "usa": "United States",
    "usmnt": "United States",
    "united states of america": "United States",
    "rsa": "South Africa",
    "mex": "Mexico",
    "korea republic": "South Korea",
    "czech republic": "Czechia",
    "czechia": "Czechia",
    "turkiye": "Turkey",
    "türkiye": "Turkey",
    "cote d'ivoire": "Cote d'Ivoire",
    "côte d'ivoire": "Cote d'Ivoire",
    "ivory coast": "Cote d'Ivoire",
    "curacao": "Curaçao",
    "curaçao": "Curaçao",
    "bosnia & herzegovina": "Bosnia and Herzegovina",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "cape verde islands": "Cape Verde",
    "congo dr": "DR Congo",
    "democratic republic of congo": "DR Congo",
}


def key(value: object) -> str:
    text = "" if value is None else str(value).strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


class TeamNameMapper:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.mapping = dict(DEFAULT_ALIASES)
        if path and path.exists():
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
            df.columns = [snake_case(column) for column in df.columns]
            raw_col = "raw_name" if "raw_name" in df.columns else "source_name"
            if raw_col in df.columns and "canonical_name" in df.columns:
                for raw, canonical in df[[raw_col, "canonical_name"]].itertuples(index=False):
                    if raw.strip() and canonical.strip():
                        self.mapping[key(raw)] = canonical.strip()

    def normalize(self, value: object) -> object:
        if value is None:
            return value
        text = re.sub(r"\s+", " ", str(value).strip())
        if not text:
            return text
        return self.mapping.get(key(text), text)

    def can_map(self, value: object) -> bool:
        return self.normalize(value) == value or key(value) in self.mapping

