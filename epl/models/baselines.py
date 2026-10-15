"""Baselines that any real model has to beat.

    FrequencyModel   historical H/D/A rates, the same for every match: zero skill
    MarketModel      bookmaker odds with the margin removed: the bar that matters.
                     Closing prices aggregate everything the market knows, including
                     team news this project never sees, so matching them is a strong
                     result and beating them consistently would be remarkable.

Removing the bookmaker's margin (overround):
    proportional   p_i = (1/o_i) / sum(1/o_j). Simple, but leaves the
                   favourite-longshot bias in place.
    power          p_i = (1/o_i)^k with k solved so the p_i sum to 1. Shrinks longshots
                   more than favourites, which matches how margins are actually priced.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from ..eval.metrics import OUTCOMES


class FrequencyModel:
    name = "frequency"

    def fit(self, train: pd.DataFrame) -> "FrequencyModel":
        counts = train["result"].value_counts()
        rates = np.array([counts.get(o, 0) for o in OUTCOMES], dtype=float) + 1  # Laplace
        self.probs_ = rates / rates.sum()
        return self

    def predict(self, fixtures: pd.DataFrame) -> np.ndarray:
        return np.tile(self.probs_, (len(fixtures), 1))


def demargin(odds: np.ndarray, method: str = "power") -> np.ndarray:
    """Decimal odds (n, 3) -> fair probabilities (n, 3)."""
    implied = 1 / np.asarray(odds, dtype=float)
    if method == "proportional":
        return implied / implied.sum(axis=1, keepdims=True)
    if method != "power":
        raise ValueError(f"unknown method {method!r}")
    out = np.empty_like(implied)
    for i, row in enumerate(implied):
        k = brentq(lambda k, r=row: (r**k).sum() - 1, 0.5, 5.0) if row.sum() > 1 else 1.0
        out[i] = row**k
    return out / out.sum(axis=1, keepdims=True)


class MarketModel:
    name = "market"

    def __init__(self, method: str = "power"):
        self.method = method

    def fit(self, train: pd.DataFrame) -> "MarketModel":
        return self

    def predict(self, fixtures: pd.DataFrame) -> np.ndarray:
        odds = fixtures[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float)
        if np.isnan(odds).any():
            raise ValueError("MarketModel needs odds for every fixture")
        return demargin(odds, self.method)
