"""Calibration: when the model says 30%, does it happen about 30% of the time?

reliability() bins every (outcome, forecast) pair, so one table covers home wins,
draws and away wins together; per_outcome=True keeps them apart. ECE is the expected
calibration error: the bin-size-weighted mean gap between forecast and frequency.
"""

import numpy as np
import pandas as pd

from .metrics import OUTCOMES, check_probs, one_hot


def reliability(probs, outcomes, bins: int = 10, per_outcome: bool = False) -> pd.DataFrame:
    p = check_probs(probs)
    o = one_hot(outcomes)
    frames = []
    for k, label in enumerate(OUTCOMES):
        frames.append(pd.DataFrame({"outcome": label, "forecast": p[:, k], "hit": o[:, k]}))
    df = pd.concat(frames, ignore_index=True)
    edges = np.linspace(0, 1, bins + 1)
    df["bin"] = np.clip(np.digitize(df["forecast"], edges[1:-1]), 0, bins - 1)
    keys = ["outcome", "bin"] if per_outcome else ["bin"]
    table = (
        df.groupby(keys)
        .agg(n=("hit", "size"), mean_forecast=("forecast", "mean"), observed=("hit", "mean"))
        .reset_index()
    )
    table["bin_low"] = edges[table["bin"]]
    table["bin_high"] = edges[table["bin"] + 1]
    # 95% Wilson interval on the observed frequency, so sparse bins aren't over-read.
    z, n, f = 1.96, table["n"], table["observed"]
    centre = (f + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(f * (1 - f) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    table["ci_low"], table["ci_high"] = centre - half, centre + half
    return table


def ece(probs, outcomes, bins: int = 10) -> float:
    table = reliability(probs, outcomes, bins)
    gaps = (table["mean_forecast"] - table["observed"]).abs()
    return float((gaps * table["n"]).sum() / table["n"].sum())
