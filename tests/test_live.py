"""Live-season helpers and the static page."""

import json

import numpy as np
import pandas as pd

from conftest import FIXTURES
from epl import live
from epl.data import fpl
from epl.fpl.optimise import Plan
from epl.report import build


def test_season_boundaries():
    assert live.season_of(pd.Timestamp("2026-08-15")) == "2026-27"
    assert live.season_of(pd.Timestamp("2027-05-23")) == "2026-27"
    assert live.season_of(pd.Timestamp("2027-07-01")) == "2027-28"
    assert live.previous_season("2000-01") == "1999-00"


def _boot():
    return json.loads((FIXTURES / "fpl_bootstrap.json").read_text())


def _fixtures():
    raw = [
        {"id": 10, "event": 1, "kickoff_time": "2024-08-17T14:00:00Z", "team_h": 1,
         "team_a": 14, "finished": True, "team_h_score": 2, "team_a_score": 2},
        {"id": 11, "event": 1, "kickoff_time": "2024-08-20T19:45:00Z", "team_h": 14,
         "team_a": 1, "finished": True, "team_h_score": 0, "team_a_score": 1},
        {"id": 12, "event": 2, "kickoff_time": "2024-08-24T14:00:00Z", "team_h": 1,
         "team_a": 14, "finished": False, "team_h_score": None, "team_a_score": None},
    ]
    return fpl.parse_live_fixtures(raw, _boot())


def test_season_results_keep_finished_matches_only():
    m = live.season_results(_fixtures(), "2024-25")
    assert m[["home", "away", "hg", "ag", "result", "gw"]].values.tolist() == [
        ["Arsenal", "Man United", 2, 2, "D", 1],
        ["Man United", "Arsenal", 0, 1, "A", 1],
    ]
    assert m["date"].iloc[0] == pd.Timestamp("2024-08-17")
    assert m["odds_h"].isna().all()


def _stats(**kw):
    base = {"minutes": 0, "total_points": 0, "goals_scored": 0, "assists": 0,
            "clean_sheets": 0, "goals_conceded": 0, "saves": 0, "bonus": 0,
            "expected_goals": "0.00", "expected_assists": "0.00"}
    return {**base, **kw}


def test_event_live_rows_follow_the_history_schema():
    players, _ = fpl.parse_bootstrap(_boot())
    live_json = {"elements": [
        # a double gameweek: one row, dated at the first fixture
        {"id": 2, "stats": _stats(minutes=180, total_points=9, goals_scored=1,
                                  expected_goals="0.85"),
         "explain": [{"fixture": 10}, {"fixture": 11}]},
        {"id": 1, "stats": _stats(), "explain": []},       # no fixture: dropped
        {"id": 99, "stats": _stats(minutes=90), "explain": [{"fixture": 10}]},  # unknown
    ]}
    rows = live.parse_event_live(live_json, 1, "2024-25", players, _fixtures())
    assert list(rows.columns) == fpl.PLAYER_GW_COLUMNS
    assert len(rows) == 1
    saka = rows.iloc[0]
    assert (saka["name"], saka["team"], saka["opponent"], saka["was_home"]) == (
        "Saka", "Arsenal", "Man United", True)
    assert saka["kickoff"] == pd.Timestamp("2024-08-17 14:00")
    assert (saka["minutes"], saka["goals"], saka["xg"]) == (180, 1, 0.85)
    assert np.isnan(saka["fpl_xp"])


def _page(record=None, reliability=None):
    players = pd.DataFrame({
        "name": ["Keeper <b>", "Striker"], "team": ["Arsenal", "Man United"],
        "position": ["GK", "FWD"], "price": [5.0, 8.5], "chance_next": [100.0, 50.0],
    }, index=[1, 2])
    matches = pd.DataFrame({
        "gw": [7], "kickoff": [pd.Timestamp("2026-10-17 11:30")], "home": ["Arsenal"],
        "away": ["Man United"], "p_home": [0.6], "p_draw": [0.25], "p_away": [0.15],
        "xg_home": [1.9], "xg_away": [0.9], "likely_score": ["1-0"],
    })
    plan = Plan(squad=[1, 2], xi=[1, 2], captain=2, vice=1, bench=[], expected_points=12.3)
    xp = pd.Series({1: 3.2, 2: 6.1})
    return build.render(7, "2026-27", pd.Timestamp("2026-10-17 10:00"), matches, players, xp,
                        plan, record, reliability, "2022-23 to 2025-26")


def test_page_renders_predictions_picks_and_flags():
    page = _page()
    assert "Gameweek 7 predictions" in page
    assert "Arsenal v Man United" in page
    assert "Home 60%, draw 25%, away 15%" in page
    assert "50% to play" in page               # injury flag on the striker
    assert "Keeper &lt;b&gt;" in page          # names are escaped
    assert "Track record" not in page          # no record requested


def test_page_track_record_and_reliability_chart():
    record = pd.DataFrame({"model": ["market", "dixon_coles"], "rps": [0.195, 0.205],
                           "log_loss": [0.96, 0.99], "accuracy": [0.55, 0.52],
                           "matches": [1520, 1520]})
    reliability = pd.DataFrame({"mean_forecast": [0.2, 0.5], "observed": [0.22, 0.48],
                                "ci_low": [0.18, 0.44], "ci_high": [0.26, 0.52], "n": [400, 300]})
    page = _page(record, reliability)
    assert "1520 matches" in page
    assert "0.1950" in page and "Bookmaker closing odds" in page
    assert page.count("<circle") == 2


def test_percentages_always_sum_to_100():
    for probs in ([0.333, 0.333, 0.334], [0.855, 0.105, 0.04], [0.5, 0.25, 0.25]):
        assert sum(build._pct(np.array(probs))) == 100
