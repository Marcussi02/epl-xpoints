"""Hyper-parameter selection on validation seasons only.

Settings are chosen by mean RPS over walk-forward predictions for the validation
seasons, then frozen. Test seasons are never looked at while tuning, so the reported
test scores are genuinely out of sample (choosing settings on the test seasons would
quietly overfit them).
"""

import itertools
from collections.abc import Callable

import pandas as pd

from .backtest import summarise, walk_forward


def grid_search(
    matches: pd.DataFrame,
    build: Callable[..., object],
    grid: dict[str, list],
    validation_seasons: list[str],
    train_seasons: int | None = None,
) -> pd.DataFrame:
    """Score every combination in `grid`; best first. build(**params) makes a model."""
    rows = []
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values, strict=True))
        preds = walk_forward(matches, lambda p=params: build(**p), validation_seasons,
                             train_seasons)
        rows.append({**params, **summarise(preds, by=None).iloc[0].to_dict()})
    return pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)
