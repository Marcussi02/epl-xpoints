"""Walk-forward evaluation of expected points against what players actually scored.

For each gameweek: fit Dixon-Coles on matches before the gameweek's first kick-off,
build player profiles from earlier gameweeks only, predict, then compare with the
points scored.

FPL's own expected points are carried alongside as a benchmark, with one correction.
The community dataset's `xP` on a gameweek's row is recorded after that gameweek: it
correlates 0.58 (Spearman, players who played) with the same gameweek's points but only
0.22 with the next one's, while consecutive gameweeks' points correlate just 0.21. So
it is FPL's forecast for the *following* gameweek, built from form that already
includes the row's result. benchmark_forecasts() shifts it forward a gameweek so each
gameweek is compared with the forecast FPL actually published before it. FPL's numbers
still include injury and suspension news that the historical files lack, so the
fairest comparison ranks only players who played.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..models.dixon_coles import DixonColesModel
from . import xpoints


def walk_forward_xp(matches: pd.DataFrame, player_gw: pd.DataFrame, season: str,
                    first_gw: int = 3, config: xpoints.XPConfig = xpoints.CONFIG,
                    train_seasons: int = 2) -> pd.DataFrame:
    seasons = sorted(matches["season"].unique())
    first_train = seasons[max(0, seasons.index(season) - train_seasons)]
    current = matches[matches["season"] == season]
    benchmark = benchmark_forecasts(player_gw[player_gw["season"] == season])
    frames = []
    for gw in sorted(current["gw"].unique()):
        if gw < first_gw:
            continue
        fixtures = current[current["gw"] == gw]
        start = fixtures["date"].min()
        train = matches[(matches["date"] < start) & (matches["season"] >= first_train)]
        model = DixonColesModel().fit(train)
        rates = xpoints.team_fixture_rates(model, fixtures)
        profiles = xpoints.player_profiles(player_gw[player_gw["kickoff"] < start],
                                           config=config)
        actual = (player_gw[(player_gw["season"] == season) & (player_gw["gw"] == gw)]
                  .groupby("element")
                  .agg(points=("total_points", "sum"), minutes=("minutes", "sum")))
        actual = actual.join(benchmark.get(gw, pd.Series(dtype=float)).rename("fpl_xp"))
        known = profiles.loc[profiles.index.intersection(actual.index)]
        xp = xpoints.expected_points(known, rates).set_index("element")
        frames.append(xp.join(actual, how="inner").reset_index())
    return pd.concat(frames, ignore_index=True)


def benchmark_forecasts(season_rows: pd.DataFrame) -> dict[int, pd.Series]:
    """{gw: FPL's pre-deadline forecast per element}, taken from the previous gameweek's
    row (see the module docstring for why)."""
    per_gw = season_rows.groupby(["gw", "element"])["fpl_xp"].first()
    return {gw + 1: per_gw.loc[gw] for gw in per_gw.index.get_level_values(0).unique()}


def score_xp(results: pd.DataFrame, column: str = "xp", top: int = 50) -> dict[str, float]:
    d = results.dropna(subset=[column])
    played = d[d["minutes"] > 0]
    top_picks = played.groupby("gw", group_keys=False).apply(lambda g: g.nlargest(top, column))
    return {
        "mae": float((d[column] - d["points"]).abs().mean()),
        "spearman": float(spearmanr(d[column], d["points"]).statistic),
        "spearman_played": float(spearmanr(played[column], played["points"]).statistic),
        f"top{top}_played_points": float(top_picks["points"].mean()),
        "bias": float(d[column].mean() - d["points"].mean()),
        "rows": float(len(d)),
    }


def summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = {"ours": score_xp(results, "xp")}
    if results["fpl_xp"].notna().any():
        rows["fpl"] = score_xp(results, "fpl_xp")
    return pd.DataFrame(rows).T.round(3).replace({np.nan: None})
