"""Walk-forward backtesting: predict each week using only what was known before it.

For every test week (Monday to Sunday) in the test seasons:

    1. train = every match that finished before the week began
    2. fit a fresh model on train
    3. predict the week's fixtures from their pre-match columns alone
    4. score the predictions against what then happened

Refitting weekly mirrors how the model would really be used, and the leakage guard
makes the "no peeking" rule a hard error rather than a convention: if any training
match is dated on or after the first fixture being predicted, the backtest stops.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from ..data.load import PREMATCH_COLUMNS
from ..eval.metrics import OUTCOMES, score_all
from ..models.base import MatchModel


class LeakageError(RuntimeError):
    """Training data overlaps the matches being predicted."""


def check_no_leakage(train: pd.DataFrame, fixtures: pd.DataFrame) -> None:
    if train.empty or fixtures.empty:
        return
    if train["date"].max() >= fixtures["date"].min():
        raise LeakageError(
            f"training data runs to {train['date'].max().date()} but predictions start "
            f"{fixtures['date'].min().date()}"
        )
    overlap = set(train["match_id"]) & set(fixtures["match_id"])
    if overlap:
        raise LeakageError(f"matches in both train and test: {sorted(overlap)[:3]}")


def walk_forward(
    matches: pd.DataFrame,
    make_model: Callable[[], MatchModel],
    test_seasons: list[str],
    train_seasons: int | None = None,
) -> pd.DataFrame:
    """One row per predicted match: ids, the outcome and p_H, p_D, p_A.

    train_seasons limits history to the N seasons before the test week's season (plus
    the test season so far); None uses everything available.
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    season_order = sorted(matches["season"].unique())
    rows = []
    for season in test_seasons:
        test = matches[matches["season"] == season]
        if test.empty:
            raise ValueError(f"no matches for test season {season}")
        weeks = test["date"].dt.to_period("W-SUN")
        for _, week in test.groupby(weeks, sort=True):
            start = week["date"].min()
            history = matches[matches["date"] < start]
            if train_seasons is not None:
                first = season_order[max(0, season_order.index(season) - train_seasons)]
                history = history[history["season"] >= first]
            fixtures = week[PREMATCH_COLUMNS]
            check_no_leakage(history, fixtures)
            probs = make_model().fit(history).predict(fixtures)
            probs = np.asarray(probs, dtype=float)
            if probs.shape != (len(week), 3):
                raise ValueError(f"model returned shape {probs.shape} for {len(week)} fixtures")
            out = week[["match_id", "season", "date", "home", "away", "result"]].copy()
            out[["p_H", "p_D", "p_A"]] = probs
            rows.append(out)
    return pd.concat(rows, ignore_index=True)


def summarise(predictions: pd.DataFrame, by: str | None = "season") -> pd.DataFrame:
    """Scores per season (or overall with by=None)."""
    cols = [f"p_{o}" for o in OUTCOMES]

    def score(group: pd.DataFrame) -> pd.Series:
        s = score_all(group[cols].to_numpy(), group["result"])
        return pd.Series({**s, "matches": len(group)})

    if by is None:
        return score(predictions).to_frame().T
    return predictions.groupby(by).apply(score, include_groups=False).reset_index()


def compare(results: dict[str, pd.DataFrame], by: str | None = "season") -> pd.DataFrame:
    """Side-by-side scores for several models' walk-forward predictions."""
    frames = []
    for name, preds in results.items():
        table = summarise(preds, by)
        table.insert(0, "model", name)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)
