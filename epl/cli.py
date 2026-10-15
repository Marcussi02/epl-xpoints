"""Command line entry point.

    epl fetch    [--first 2019-20] [--last 2025-26]       download and cache results
    epl backtest [--test 2022-23 ...] [--odds]           walk-forward scores, written to reports/
    epl tune     [--validation 2021-22 ...]              grid search on validation seasons only
    epl xp       [--season 2024-25]                     expected points vs FPL's own forecast
    epl season   [--season 2024-25] [--horizon 1]       replay an FPL season with each forecaster

Results come from the FPL community dataset (2019-20 onwards). With --odds, matches
come from football-data.co.uk instead, which adds bookmaker odds and the market baseline.
"""

import argparse
from pathlib import Path

import pandas as pd

from .data import fpl, load
from .eval import backtest, calibration
from .eval.significance import paired_bootstrap
from .eval.tune import grid_search
from .fpl import evaluate as fpl_eval
from .fpl import season as fpl_season
from .models.baselines import FrequencyModel, MarketModel
from .models.dixon_coles import DixonColesModel
from .models.elo import EloModel

REPORTS = Path("reports")
DC_TRAIN_SEASONS = 2  # what the tuned settings were chosen with


def load_results(first: str, last: str, odds: bool) -> pd.DataFrame:
    season_list = load.seasons(first, last)
    if odds:
        return load.load_matches(season_list)
    matches = pd.concat([fpl.history_results(s) for s in season_list], ignore_index=True)
    load.validate(matches)
    return matches


def run_backtest(matches: pd.DataFrame, test: list[str], odds: bool) -> dict[str, pd.DataFrame]:
    results = {
        "frequency": backtest.walk_forward(matches, FrequencyModel, test),
        "elo": backtest.walk_forward(matches, EloModel, test),
        "dixon_coles": backtest.walk_forward(matches, DixonColesModel, test, DC_TRAIN_SEASONS),
    }
    if odds:
        priced = matches.dropna(subset=["odds_h", "odds_d", "odds_a"])
        results["market"] = backtest.walk_forward(priced, MarketModel, test)
    return results


def cmd_fetch(args) -> None:
    matches = load_results(args.first, args.last, args.odds)
    path = load.save(matches)
    print(f"{len(matches)} matches, {matches['season'].nunique()} seasons -> {path}")


def cmd_backtest(args) -> None:
    matches = load_results(args.first, args.last, args.odds)
    results = run_backtest(matches, args.test, args.odds)
    REPORTS.mkdir(exist_ok=True)
    overall = backtest.compare(results, by=None)
    overall["ece"] = [
        calibration.ece(p[["p_H", "p_D", "p_A"]].to_numpy(), p["result"]) for p in results.values()
    ]
    by_season = backtest.compare(results)
    overall.to_csv(REPORTS / "backtest_overall.csv", index=False)
    by_season.to_csv(REPORTS / "backtest_by_season.csv", index=False)
    for name, preds in results.items():
        preds.to_csv(REPORTS / f"predictions_{name}.csv", index=False)
        calibration.reliability(preds[["p_H", "p_D", "p_A"]].to_numpy(), preds["result"]).to_csv(
            REPORTS / f"reliability_{name}.csv", index=False)
    pd.set_option("display.width", 160)
    print(overall.round(4).to_string(index=False))
    print()
    print(by_season.pivot(index="season", columns="model", values="rps").round(4).to_string())
    print()
    reference = "market" if "market" in results else "frequency"
    for name in results:
        if name != reference:
            gap = paired_bootstrap(results[name], results[reference])
            print(f"{name} - {reference}: RPS {gap['mean_diff']:+.4f} "
                  f"(95% CI {gap['ci_low']:+.4f} to {gap['ci_high']:+.4f})")


def cmd_tune(args) -> None:
    matches = load_results(args.first, args.last, args.odds)
    pd.set_option("display.width", 160)
    elo = grid_search(matches, EloModel,
                      {"k": [6, 8, 10, 12, 15, 20, 30], "regress": [0.0, 0.05, 0.1, 0.25]},
                      args.validation)
    print("Elo")
    print(elo.head(8)[["k", "regress", "rps", "log_loss"]].round(4).to_string(index=False))
    dc_grid = {"xi": [0.0003, 0.0006, 0.001, 0.0019, 0.003], "l2": [0.0, 1.0, 2.0, 5.0, 10.0]}
    dc = grid_search(matches, DixonColesModel, dc_grid, args.validation, DC_TRAIN_SEASONS)
    print("Dixon-Coles")
    print(dc.head(8)[["xi", "l2", "rps", "log_loss"]].round(4).to_string(index=False))


def _fpl_inputs(season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = int(season[:4])
    matches = load_results(f"{start - 2}-{(start - 1) % 100:02d}", season, odds=False)
    return matches, fpl.history_two_seasons(season)


def cmd_xp(args) -> None:
    matches, player_gw = _fpl_inputs(args.season)
    results = fpl_eval.walk_forward_xp(matches, player_gw, args.season)
    both = results.dropna(subset=["fpl_xp"])  # identical rows for both forecasts
    table = pd.DataFrame({"ours": fpl_eval.score_xp(both, "xp"),
                          "fpl": fpl_eval.score_xp(both, "fpl_xp")}).T
    REPORTS.mkdir(exist_ok=True)
    table.to_csv(REPORTS / f"xp_{args.season}.csv")
    print(table.round(3).to_string())


def cmd_season(args) -> None:
    matches, player_gw = _fpl_inputs(args.season)
    forecasters = {
        "model": (fpl_season.model_forecaster(), args.horizon),
        "form": (fpl_season.form_forecaster(), 1),
        "fpl": (fpl_season.fpl_forecaster(player_gw, args.season), 1),
    }
    REPORTS.mkdir(exist_ok=True)
    for name, (forecaster, horizon) in forecasters.items():
        result = fpl_season.simulate(matches, player_gw, args.season, forecaster, name,
                                     horizon=horizon)
        result.weeks.to_csv(REPORTS / f"season_{args.season}_{name}.csv", index=False)
        w = result.weeks
        print(f"{name:6s} total {result.total:5d}  per gameweek {w.points.mean():.1f}  "
              f"transfers {w.transfers.sum():3d}  hits {w.hits.sum():2d}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="epl")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func, help_text in (("fetch", cmd_fetch, "download and cache results"),
                                  ("backtest", cmd_backtest, "walk-forward backtest"),
                                  ("tune", cmd_tune, "tune on validation seasons")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--first", default="2019-20")
        p.add_argument("--last", default="2025-26")
        p.add_argument("--odds", action="store_true",
                       help="use football-data.co.uk (adds odds and the market baseline)")
        p.set_defaults(func=func)
        if name == "backtest":
            p.add_argument("--test", nargs="+", default=load.seasons("2022-23", "2025-26"))
        if name == "tune":
            p.add_argument("--validation", nargs="+", default=["2021-22"])
    p = sub.add_parser("xp", help="expected points vs FPL's own forecast")
    p.add_argument("--season", default="2024-25")
    p.set_defaults(func=cmd_xp)
    p = sub.add_parser("season", help="replay an FPL season with each forecaster")
    p.add_argument("--season", default="2024-25")
    p.add_argument("--horizon", type=int, default=1)  # best on the 2023-24 replay
    p.set_defaults(func=cmd_season)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
