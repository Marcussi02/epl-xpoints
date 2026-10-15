"""The interface every match model implements.

    fit(train)          learn from finished matches only (results included)
    predict(fixtures)   (n, 3) H/D/A probabilities for matches not yet played;
                        fixtures carry only the pre-match columns

The backtest is the only caller that decides what is "past", so models never need
to guard against leakage themselves.
"""

from typing import Protocol, Self

import numpy as np
import pandas as pd


class MatchModel(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> Self: ...

    def predict(self, fixtures: pd.DataFrame) -> np.ndarray: ...
