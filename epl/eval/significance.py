"""Is model A really better than model B, or is the gap noise?

Both models predicted the same matches, so compare them match by match: the paired
bootstrap resamples matches (not models) and reports the mean per-match RPS
difference with a 95% interval. If the interval spans zero, the data do not
distinguish the two models. Resampling whole weeks would also respect the fact
that matches on the same weekend share information; by_week=True does that.
"""

import numpy as np
import pandas as pd

from .metrics import one_hot


def per_match_rps(preds: pd.DataFrame) -> pd.Series:
    p = preds[["p_H", "p_D", "p_A"]].to_numpy()
    o = one_hot(preds["result"])
    cum = (np.cumsum(p, axis=1) - np.cumsum(o, axis=1))[:, :-1]
    return pd.Series((cum**2).sum(axis=1) / 2, index=preds["match_id"])


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 4000, seed: int = 0,
                     by_week: bool = True) -> dict[str, float]:
    """Mean RPS(a) - RPS(b) over shared matches, with a 95% bootstrap interval.
    Negative means `a` is better."""
    ra, rb = per_match_rps(a), per_match_rps(b)
    shared = ra.index.intersection(rb.index)
    diff = (ra[shared] - rb[shared]).to_numpy()
    rng = np.random.default_rng(seed)
    if by_week:
        dates = a.set_index("match_id").loc[shared, "date"]
        groups = pd.factorize(dates.dt.to_period("W-SUN"))[0]
        sums = np.bincount(groups, weights=diff)
        counts = np.bincount(groups)
        picks = rng.integers(0, len(sums), size=(n_boot, len(sums)))
        boot = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    else:
        picks = rng.integers(0, len(diff), size=(n_boot, len(diff)))
        boot = diff[picks].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "ci_low": float(low), "ci_high": float(high),
            "matches": int(len(diff))}
