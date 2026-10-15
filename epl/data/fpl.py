"""Fantasy Premier League data: live API snapshots and past seasons.

    live      official FPL API (bootstrap-static/, fixtures/): players, prices, fixtures
    history   vaastav/Fantasy-Premier-League (MIT): per-season fixtures, teams and
              per-gameweek player rows, used for backtests. Its fixtures carry final
              scores, so they double as a results source (no odds) from 2019-20.

Everything is mapped onto the same canonical team names as the match data.

Player ids ("element") are reassigned every season: element 384 can be a Liverpool
midfielder one year and a Man City goalkeeper the next. FPL's `code` is the stable
identity, so rows carry it, and align_elements() re-keys earlier seasons' rows to the
target season's ids before any history is combined.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd

from . import download
from .teams import canonical

HISTORY_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/{file}"
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
VALID_POSITIONS = frozenset(POSITIONS.values())  # drops 2024-25's "AM" (assistant manager) chip

PLAYER_GW_COLUMNS = [
    "season", "gw", "element", "name", "position", "team", "opponent", "was_home", "kickoff",
    "minutes", "total_points", "goals", "assists", "clean_sheets", "goals_conceded", "saves",
    "bonus", "xg", "xa", "price", "fpl_xp", "code",
]


# ---------- past seasons ----------

def _history_csv(season: str, file: str, cache_dir: Path, fetch) -> pd.DataFrame:
    url = HISTORY_URL.format(season=season, file=file)
    path = cache_dir / "fpl-history" / season / file.replace("/", "_")
    raw = download.cached(url, path, fetch=fetch)
    return pd.read_csv(io.BytesIO(raw), low_memory=False)


def history_results(
    season: str, cache_dir: Path = download.CACHE_DIR, fetch=download.get_bytes
) -> pd.DataFrame:
    """Finished matches of one season in the tidy match schema (odds left empty)."""
    teams = _history_csv(season, "teams.csv", cache_dir, fetch)
    fixtures = _history_csv(season, "fixtures.csv", cache_dir, fetch)
    return parse_results(fixtures, teams, season)


def parse_results(fixtures: pd.DataFrame, teams: pd.DataFrame, season: str) -> pd.DataFrame:
    names = {int(i): canonical(n) for i, n in zip(teams["id"], teams["name"], strict=True)}
    done = fixtures[fixtures["finished"].astype(str).str.lower() == "true"].copy()
    out = pd.DataFrame({
        "season": season,
        "date": pd.to_datetime(done["kickoff_time"], utc=True).dt.tz_localize(None).dt.normalize(),
        "home": done["team_h"].astype(int).map(names),
        "away": done["team_a"].astype(int).map(names),
        "hg": done["team_h_score"].astype(int),
        "ag": done["team_a_score"].astype(int),
        "gw": done["event"].astype(int),
    })
    out["result"] = np.select([out.hg > out.ag, out.hg == out.ag], ["H", "D"], "A")
    for col in ("odds_h", "odds_d", "odds_a"):
        out[col] = np.nan
    out["odds_source"] = None
    out["match_id"] = (
        out["date"].dt.strftime("%Y%m%d") + "-" + out["home"] + "-" + out["away"]
    ).str.replace(" ", "_")
    cols = ["match_id", "season", "date", "home", "away", "hg", "ag", "result",
            "odds_h", "odds_d", "odds_a", "odds_source", "gw"]
    return out[cols].sort_values(["date", "home"]).reset_index(drop=True)


def history_codes(
    season: str, cache_dir: Path = download.CACHE_DIR, fetch=download.get_bytes
) -> dict[int, int]:
    """{element id that season: stable player code}."""
    raw = _history_csv(season, "players_raw.csv", cache_dir, fetch)
    return dict(zip(raw["id"].astype(int), raw["code"].astype(int), strict=True))


def history_player_gw(
    season: str, cache_dir: Path = download.CACHE_DIR, fetch=download.get_bytes
) -> pd.DataFrame:
    teams = _history_csv(season, "teams.csv", cache_dir, fetch)
    merged = _history_csv(season, "gws/merged_gw.csv", cache_dir, fetch)
    return parse_player_gw(merged, teams, season, history_codes(season, cache_dir, fetch))


def align_elements(rows: pd.DataFrame, codes: dict[int, int]) -> pd.DataFrame:
    """Re-key rows from any season to the target season's element ids.

    codes maps the target season's element id to the stable player code. Rows whose
    player is not in the target season (left the league) are dropped."""
    to_element = {code: element for element, code in codes.items()}
    out = rows.copy()
    out["element"] = out["code"].map(to_element)
    out = out.dropna(subset=["element"])
    out["element"] = out["element"].astype(int)
    return out.reset_index(drop=True)


def history_two_seasons(
    season: str, cache_dir: Path = download.CACHE_DIR, fetch=download.get_bytes
) -> pd.DataFrame:
    """Last season's and this season's player rows, keyed by this season's ids."""
    start = int(season[:4])
    previous = f"{start - 1}-{start % 100:02d}"
    rows = pd.concat([history_player_gw(previous, cache_dir, fetch),
                      history_player_gw(season, cache_dir, fetch)], ignore_index=True)
    return align_elements(rows, history_codes(season, cache_dir, fetch))


def parse_player_gw(merged: pd.DataFrame, teams: pd.DataFrame, season: str,
                    codes: dict[int, int] | None = None) -> pd.DataFrame:
    names = {int(i): canonical(n) for i, n in zip(teams["id"], teams["name"], strict=True)}
    df = merged[merged["position"].isin(VALID_POSITIONS)].copy()

    def num(col: str) -> pd.Series:  # expected_* columns only exist from 2022-23
        if col not in df:
            return pd.Series(np.nan, index=df.index)
        return pd.to_numeric(df[col], errors="coerce")

    out = pd.DataFrame({
        "season": season,
        "gw": df["GW"].astype(int),
        "element": df["element"].astype(int),
        "name": df["name"],
        "position": df["position"],
        "team": df["team"].map(canonical),
        "opponent": df["opponent_team"].astype(int).map(names),
        "was_home": df["was_home"].astype(str).str.lower() == "true",
        "kickoff": pd.to_datetime(df["kickoff_time"], utc=True).dt.tz_localize(None),
        "minutes": df["minutes"].astype(int),
        "total_points": df["total_points"].astype(int),
        "goals": df["goals_scored"].astype(int),
        "assists": df["assists"].astype(int),
        "clean_sheets": df["clean_sheets"].astype(int),
        "goals_conceded": df["goals_conceded"].astype(int),
        "saves": df["saves"].astype(int),
        "bonus": df["bonus"].astype(int),
        "xg": num("expected_goals"),
        "xa": num("expected_assists"),
        "price": df["value"].astype(float) / 10,
        "fpl_xp": num("xP"),  # FPL's own expected points, a benchmark for ours
    })
    out["code"] = out["element"].map(codes or {}).astype("Int64")
    return out[PLAYER_GW_COLUMNS].sort_values(["gw", "element", "kickoff"]).reset_index(drop=True)


# ---------- live API ----------

def parse_bootstrap(bootstrap: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """bootstrap-static/ -> (players, gameweeks)."""
    teams = {t["id"]: canonical(t["name"]) for t in bootstrap["teams"]}
    rows = []
    for e in bootstrap["elements"]:
        position = POSITIONS.get(e["element_type"])
        if position is None:
            continue
        chance = e.get("chance_of_playing_next_round")
        rows.append({
            "element": e["id"],
            "code": e.get("code"),
            "name": e["web_name"],
            "team": teams[e["team"]],
            "position": position,
            "price": e["now_cost"] / 10,
            "status": e.get("status", "a"),
            "chance_next": 100.0 if chance is None else float(chance),
            "minutes": int(e.get("minutes", 0)),
            "xg": float(e.get("expected_goals") or 0),
            "xa": float(e.get("expected_assists") or 0),
        })
    events = pd.DataFrame([
        {"gw": ev["id"], "deadline": pd.Timestamp(ev["deadline_time"]).tz_localize(None),
         "finished": bool(ev["finished"]), "is_next": bool(ev["is_next"])}
        for ev in bootstrap["events"]
    ])
    return pd.DataFrame(rows), events


def parse_live_fixtures(fixtures: list[dict], bootstrap: dict) -> pd.DataFrame:
    """fixtures/ -> one row per scheduled match (finished or not)."""
    teams = {t["id"]: canonical(t["name"]) for t in bootstrap["teams"]}
    return pd.DataFrame([
        {"fixture_id": f["id"], "gw": f["event"],
         "kickoff": pd.Timestamp(f["kickoff_time"]).tz_localize(None),
         "home": teams[f["team_h"]], "away": teams[f["team_a"]], "finished": bool(f["finished"]),
         "hg": f.get("team_h_score"), "ag": f.get("team_a_score")}
        for f in fixtures
        if f.get("event") is not None and f.get("kickoff_time")  # unscheduled fixtures
    ])
