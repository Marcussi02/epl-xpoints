"""Replay a past FPL season: plan before every deadline, then score what really happened.

At each gameweek's deadline the manager sees only what was known then: results and
player rows from earlier gameweeks, the fixture list, who is registered and at what
price. A forecaster turns that into expected points, the planner picks transfers, the
starting XI and the captain, and the real points are counted with FPL's scoring:

    captain     points doubled; if he does not play, the vice-captain's are doubled
    auto-subs   non-playing starters are replaced by bench players in bench order,
                keeping a valid formation (the reserve keeper only covers the keeper)
    transfers   one free per gameweek, rolling over up to five; extras cost 4 points
    prices      players are bought at the current price and sold at FPL's selling
                price: the purchase price plus half of any rise, rounded down to 0.1

Chips (wildcard, free hit, bench boost, triple captain) are not modelled.

Forecasters to compare, all fed through the same planner:
    model   this project's expected points (Dixon-Coles + player profiles)
    form    average points over the player's last four gameweeks
    fpl     FPL's own published forecast (one gameweek ahead only)
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import optimise, xpoints
from .evaluate import benchmark_forecasts

Forecaster = Callable[["Context", list[int]], pd.DataFrame]


@dataclass
class Context:
    season: str
    gw: int
    start: pd.Timestamp
    matches: pd.DataFrame       # all results; only rows before `start` may be used
    player_gw: pd.DataFrame     # all player rows; only rows before `start` may be used
    universe: pd.DataFrame      # registered players this gameweek: position, team, price
    train_seasons: int = 2

    def past_matches(self) -> pd.DataFrame:
        seasons = sorted(self.matches["season"].unique())
        first = seasons[max(0, seasons.index(self.season) - self.train_seasons)]
        m = self.matches
        return m[(m["date"] < self.start) & (m["season"] >= first)]

    def past_player_rows(self) -> pd.DataFrame:
        return self.player_gw[self.player_gw["kickoff"] < self.start]

    def fixtures(self, gws: list[int]) -> pd.DataFrame:
        m = self.matches
        return m[(m["season"] == self.season) & (m["gw"].isin(gws))][["gw", "home", "away",
                                                                       "season"]]


def model_forecaster(config: xpoints.XPConfig = xpoints.CONFIG) -> Forecaster:
    def forecast(ctx: Context, gws: list[int]) -> pd.DataFrame:
        dc = config.match_model().fit(ctx.past_matches())
        rates = xpoints.team_fixture_rates(dc, ctx.fixtures(gws))
        prof = xpoints.player_profiles(ctx.past_player_rows(), config=config)
        prof = prof.loc[prof.index.intersection(ctx.universe.index)].copy()
        prof["team"] = ctx.universe.loc[prof.index, "team"]  # current club, known pre-deadline
        prof["position"] = ctx.universe.loc[prof.index, "position"]
        xp = xpoints.expected_points(prof, rates)
        return xp.pivot(index="element", columns="gw", values="xp").reindex(columns=gws)
    return forecast


def form_forecaster(window: int = 4) -> Forecaster:
    def forecast(ctx: Context, gws: list[int]) -> pd.DataFrame:
        rows = ctx.past_player_rows().sort_values("kickoff")
        form = rows.groupby("element").tail(window).groupby("element")["total_points"].mean()
        fx = ctx.fixtures(gws)
        counts = pd.concat([fx[["gw", "home"]].rename(columns={"home": "team"}),
                            fx[["gw", "away"]].rename(columns={"away": "team"})])
        per_gw = counts.groupby(["team", "gw"]).size().unstack(fill_value=0).reindex(columns=gws,
                                                                                    fill_value=0)
        teams = ctx.universe["team"].reindex(form.index)
        fixtures = per_gw.reindex(teams.to_numpy()).fillna(0).to_numpy()
        return pd.DataFrame(form.to_numpy()[:, None] * fixtures, index=form.index, columns=gws)
    return forecast


def fpl_forecaster(player_gw: pd.DataFrame, season: str) -> Forecaster:
    published = benchmark_forecasts(player_gw[player_gw["season"] == season])

    def forecast(ctx: Context, gws: list[int]) -> pd.DataFrame:
        cols = {gw: published.get(gw, pd.Series(dtype=float)) for gw in gws[:1]}
        return pd.DataFrame(cols).reindex(columns=gws).fillna(0.0)
    return forecast


def selling_price(bought: float, now: float) -> float:
    if now <= bought:
        return now
    return round(bought + np.floor((now - bought) * 10 / 2) / 10, 1)


def auto_substitute(xi: list[int], bench: list[int], minutes: dict[int, int],
                    positions: dict[int, str]) -> list[int]:
    final = list(xi)

    def played(e: int) -> bool:
        return minutes.get(e, 0) > 0

    keepers = [e for e in bench if positions[e] == "GK"]
    starter_gk = next(e for e in final if positions[e] == "GK")
    if not played(starter_gk) and keepers and played(keepers[0]):
        final[final.index(starter_gk)] = keepers[0]
    for sub in (e for e in bench if positions[e] != "GK"):
        if not played(sub):
            continue
        for out in [e for e in final if positions[e] != "GK" and not played(e)]:
            trial = [sub if e == out else e for e in final]
            counts = pd.Series([positions[e] for e in trial]).value_counts()
            if all(lo <= counts.get(p, 0) <= hi for p, (lo, hi) in optimise.XI_LIMITS.items()):
                final = trial
                break
    return final


@dataclass
class SeasonResult:
    name: str
    weeks: pd.DataFrame
    squads: dict[int, list[int]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return int(self.weeks["points"].sum())


def simulate(matches: pd.DataFrame, player_gw: pd.DataFrame, season: str,
             forecaster: Forecaster, name: str, first_gw: int = 2, last_gw: int = 38,
             horizon: int = 3, budget: float = 100.0, **plan_kwargs) -> SeasonResult:
    season_rows = player_gw[player_gw["season"] == season]
    season_matches = matches[matches["season"] == season]
    squad: list[int] | None = None
    bought: dict[int, float] = {}
    bank, free = 0.0, 1
    weeks, squads = [], {}
    for gw in range(first_gw, last_gw + 1):
        fixtures = season_matches[season_matches["gw"] == gw]
        if fixtures.empty:  # a blank gameweek for everyone
            continue
        start = fixtures["date"].min()
        rows = season_rows[season_rows["gw"] == gw]
        universe = (rows.groupby("element")
                    .agg(position=("position", "first"), team=("team", "first"),
                         price=("price", "first")))
        ctx = Context(season, gw, start, matches, player_gw, universe)
        gws = [g for g in range(gw, min(gw + horizon, last_gw + 1))]
        xp = forecaster(ctx, gws).reindex(universe.index).fillna(0.0)

        players = universe.copy()
        if squad is not None:  # owned players who left the league still have to be sold
            gone = [e for e in squad if e not in players.index]
            for e in gone:
                last = season_rows[season_rows["element"] == e].iloc[-1]
                players.loc[e] = [last["position"], last["team"], last["price"]]
                xp.loc[e] = 0.0
        if squad is None:
            plan = optimise.plan(players, xp, budget=budget, **plan_kwargs)
            bought = {e: float(players.loc[e, "price"]) for e in plan.squad}
            bank = plan.bank
        else:
            sell = {e: selling_price(bought[e], float(players.loc[e, "price"])) for e in squad}
            plan = optimise.plan(players, xp, squad=squad, sell_prices=sell, bank=bank,
                                 free_transfers=free, **plan_kwargs)
            for e in plan.sell:
                bought.pop(e)
            for e in plan.buy:
                bought[e] = float(players.loc[e, "price"])
            bank = plan.bank
            free = min(5, free - (len(plan.buy) - plan.hits) + 1)
        squad = plan.squad
        squads[gw] = squad

        actual = rows.groupby("element").agg(points=("total_points", "sum"),
                                             minutes=("minutes", "sum"))
        pts = actual["points"].to_dict()
        mins = actual["minutes"].to_dict()
        positions = players["position"].to_dict()
        final = auto_substitute(plan.xi, plan.bench, mins, positions)
        captain = plan.captain if mins.get(plan.captain, 0) > 0 else plan.vice
        score = sum(pts.get(e, 0) for e in final) + pts.get(captain, 0) - 4 * plan.hits
        weeks.append({"gw": gw, "points": score, "expected": plan.expected_points,
                      "transfers": len(plan.buy), "hits": plan.hits, "bank": bank,
                      "captain": captain, "captain_points": pts.get(captain, 0)})
    return SeasonResult(name, pd.DataFrame(weeks), squads)
