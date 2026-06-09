from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


PLACEHOLDER_RE = re.compile(
    r"(?:^|\b)(tbd|to be determined|winner group|runner[- ]?up group|playoff winner|placeholder|"
    r"tbd home|tbd away|winner |runner-up )(?:\b|$)",
    re.IGNORECASE,
)


def snake_case(value: str) -> str:
    value = value.strip().lower().replace("\ufeff", "")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def is_placeholder_team(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return not text or bool(PLACEHOLDER_RE.search(text))


def safe_float(value: object, default: float | None = None) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_probabilities(values: list[float]) -> list[float]:
    total = sum(max(0.0, value) for value in values)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [max(0.0, value) / total for value in values]


def poisson_score_matrix(lambda_home: float, lambda_away: float, max_goals: int) -> np.ndarray:
    home_probs = np.array([math.exp(-lambda_home) * lambda_home**i / math.factorial(i) for i in range(max_goals + 1)])
    away_probs = np.array([math.exp(-lambda_away) * lambda_away**i / math.factorial(i) for i in range(max_goals + 1)])
    matrix = np.outer(home_probs, away_probs)
    total = matrix.sum()
    if total > 0:
        matrix = matrix / total
    return matrix


def matrix_to_wdl(matrix: np.ndarray) -> tuple[float, float, float]:
    home = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, 1).sum())
    home, draw, away = normalize_probabilities([home, draw, away])
    return home, draw, away


def top_scorelines(matrix: np.ndarray, limit: int = 5) -> list[dict[str, Any]]:
    rows: list[tuple[float, int, int]] = []
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            rows.append((float(matrix[home_goals, away_goals]), home_goals, away_goals))
    rows.sort(reverse=True)
    return [
        {"score": f"{home}-{away}", "home_score": home, "away_score": away, "probability": prob}
        for prob, home, away in rows[:limit]
    ]


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

