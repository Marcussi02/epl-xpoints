"""This week's predictions from live data.

    history      results and odds from football-data.co.uk for earlier seasons, plus
                 earlier seasons' FPL player rows from the community dataset
    this season  results and fixtures from the FPL API (fixtures/), and per-gameweek
                 player stats from event/{gw}/live/ (the community dataset lags the
                 live season by weeks)
    news         injury and suspension flags (chance_of_playing_next_round) scale each
                 player's minutes

Nothing here sees a match after the upcoming deadline: models are fitted on finished
matches only, exactly as in the backtests.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import download, fpl, load
from .fpl import optimise, xpoints
from .models.dixon_coles import DixonColesModel

SEASON_START_MONTH = 7


def season_of(date: pd.Timestamp) -> str:
    start = date.year if date.month >= SEASON_START_MONTH else date.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def previous_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}-{(start + 1) % 100:02d}"


def season_results(fixtures: pd.DataFrame, season: str) -> pd.DataFrame:
    """Finished fixtures from the FPL API in the tidy match schema (no odds)."""
    done = fixtures[fixtures["finished"]].copy()
    out = pd.DataFrame({
        "season": season,
        "date": done["kickoff"].dt.normalize(),
        "home": done["home"], "away": done["away"],
        "hg": done["hg"].astype(int), "ag": done["ag"].astype(int),
        "gw": done["gw"].astype(int),
    })
    out["result"] = np.select([out.hg > out.ag, out.hg == out.ag], ["H", "D"], "A")
    for col in ("odds_h", "odds_d", "odds_a"):
        out[col] = np.nan
    out["odds_source"] = None
    out["match_id"] = (out["date"].dt.strftime("%Y%m%d") + "-" + out["home"] + "-"
                       + out["away"]).str.replace(" ", "_")
    return out


def parse_event_live(live: dict, gw: int, season: str, players: pd.DataFrame,
                     fixtures: pd.DataFrame) -> pd.DataFrame:
    """event/{gw}/live/ -> player-gameweek rows in the history schema.

    The live feed totals a double gameweek across both fixtures, so it becomes one row
    dated at the player's first fixture."""
    info = players.set_index("element")
    by_id = fixtures.set_index("fixture_id")
    rows = []
    for e in live["elements"]:
        if e["id"] not in info.index:
            continue
        s = e["stats"]
        p = info.loc[e["id"]]
        fixture_ids = [x["fixture"] for x in e.get("explain", [])]
        kickoff, opponent, home = pd.NaT, None, None
        if fixture_ids and fixture_ids[0] in by_id.index:
            f = by_id.loc[fixture_ids[0]]
            kickoff = f["kickoff"]
            home = f["home"] == p["team"]
            opponent = f["away"] if home else f["home"]
        rows.append({
            "season": season, "gw": gw, "element": e["id"], "name": p["name"],
            "position": p["position"], "team": p["team"], "opponent": opponent,
            "was_home": home, "kickoff": kickoff, "minutes": s["minutes"],
            "total_points": s["total_points"], "goals": s["goals_scored"],
            "assists": s["assists"], "clean_sheets": s["clean_sheets"],
            "goals_conceded": s["goals_conceded"], "saves": s["saves"], "bonus": s["bonus"],
            "xg": float(s.get("expected_goals") or 0), "xa": float(s.get("expected_assists") or 0),
            "price": p["price"], "fpl_xp": np.nan, "code": p["code"],
        })
    out = pd.DataFrame(rows, columns=fpl.PLAYER_GW_COLUMNS)
    return out.dropna(subset=["kickoff"])  # players whose team had no fixture (blank)


@dataclass
class Snapshot:
    season: str
    next_gw: int
    deadline: pd.Timestamp
    players: pd.DataFrame       # element, name, team, position, price, chance_next, status
    fixtures: pd.DataFrame      # this season's fixtures (gw, kickoff, home, away, ...)
    matches: pd.DataFrame       # finished matches: earlier seasons + this season so far
    player_gw: pd.DataFrame     # player rows: last season + this season so far


def snapshot(history_seasons: int = 3, fetch=download.get_bytes) -> Snapshot:
    boot = download.fpl_json("bootstrap-static/", fetch=fetch)
    players, events = fpl.parse_bootstrap(boot)
    upcoming = events[~events["finished"]].sort_values("gw")
    if upcoming.empty:
        raise RuntimeError("no upcoming gameweek: the season is over")
    next_gw = int(upcoming.iloc[0]["gw"])
    deadline = upcoming.iloc[0]["deadline"]
    season = season_of(deadline)

    raw_fixtures = download.fpl_json("fixtures/", fetch=fetch)
    fixtures = fpl.parse_live_fixtures(raw_fixtures, boot)

    first = season_of(deadline - pd.DateOffset(years=history_seasons))
    earlier = load.load_matches(load.seasons(first, previous_season(season)), fetch=fetch)
    current = season_results(fixtures, season)
    matches = pd.concat([earlier, current], ignore_index=True)
    load.validate(matches)

    # Element ids are renumbered each season: re-key last season's rows by player code.
    codes = dict(zip(players["element"], players["code"], strict=True))
    last = fpl.align_elements(fpl.history_player_gw(previous_season(season), fetch=fetch), codes)
    live_rows = []
    for gw in sorted(events.loc[events["finished"], "gw"]):
        live = download.fpl_json(f"event/{gw}/live/", refresh=False, fetch=fetch)
        live_rows.append(parse_event_live(live, int(gw), season, players, fixtures))
    player_gw = pd.concat([last, *live_rows], ignore_index=True)
    return Snapshot(season, next_gw, deadline, players, fixtures, matches, player_gw)


def match_predictions(snap: Snapshot, model: DixonColesModel, gw: int) -> pd.DataFrame:
    fx = snap.fixtures[snap.fixtures["gw"] == gw][["gw", "kickoff", "home", "away"]].copy()
    if fx.empty:
        return fx
    probs = model.predict(fx)
    lam, mu = model.rates(fx)
    fx[["p_home", "p_draw", "p_away"]] = probs
    fx["xg_home"], fx["xg_away"] = lam, mu
    matrix = model.score_matrix(fx)
    likely = matrix.reshape(len(fx), -1).argmax(axis=1)
    fx["likely_score"] = [f"{k // matrix.shape[2]}-{k % matrix.shape[2]}" for k in likely]
    return fx.sort_values("kickoff").reset_index(drop=True)


def expected_points(snap: Snapshot, model: DixonColesModel, gws: list[int]) -> pd.DataFrame:
    """`model` should be fitted with xpoints.CONFIG.match_model(), as in the backtests."""
    fx = snap.fixtures[snap.fixtures["gw"].isin(gws)][["gw", "home", "away"]]
    rates = xpoints.team_fixture_rates(model, fx)
    profiles = xpoints.player_profiles(snap.player_gw[snap.player_gw["kickoff"] < snap.deadline])
    current = snap.players.set_index("element")
    profiles = profiles.loc[profiles.index.intersection(current.index)].copy()
    profiles["team"] = current.loc[profiles.index, "team"]
    profiles["position"] = current.loc[profiles.index, "position"]
    available = current["chance_next"] / 100
    xp = xpoints.expected_points(profiles, rates, availability=available)
    return xp.pivot(index="element", columns="gw", values="xp").reindex(columns=gws).fillna(0.0)


def best_squad(snap: Snapshot, xp: pd.DataFrame, budget: float = 100.0) -> optimise.Plan:
    """The best squad from scratch (as on a wildcard) for the upcoming gameweeks."""
    players = snap.players.set_index("element")[["position", "team", "price"]]
    return optimise.plan(players, xp.reindex(players.index).fillna(0.0), budget=budget)


def fit_model(snap: Snapshot, train_seasons: int = 2,
              model: DixonColesModel | None = None) -> DixonColesModel:
    """Dixon-Coles on the current season so far plus `train_seasons` earlier ones (the
    window the backtests used)."""
    first = snap.season
    for _ in range(train_seasons):
        first = previous_season(first)
    train = snap.matches[(snap.matches["date"] < snap.deadline)
                         & (snap.matches["season"] >= first)]
    return (model or DixonColesModel()).fit(train)
