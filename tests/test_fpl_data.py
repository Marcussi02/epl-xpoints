"""FPL data parsing: history CSVs (vaastav layout) and live API JSON."""

import json

import pandas as pd

from conftest import FIXTURES
from epl.data import fpl


def _csv(name):
    return pd.read_csv(FIXTURES / name)


def test_results_from_fpl_fixtures_keep_only_finished_and_map_names():
    m = fpl.parse_results(_csv("fpl_fixtures.csv"), _csv("fpl_teams.csv"), "2024-25")
    assert m[["home", "away", "hg", "ag", "result", "gw"]].values.tolist() == [
        ["Man United", "Fulham", 1, 0, "H", 1],
        ["Tottenham", "Arsenal", 2, 2, "D", 2],
    ]
    assert m.date.iloc[0] == pd.Timestamp("2024-08-16")
    assert m.odds_h.isna().all()


def test_player_gw_drops_assistant_managers_and_maps_teams():
    g = fpl.parse_player_gw(_csv("fpl_merged_gw.csv"), _csv("fpl_teams.csv"), "2024-25")
    assert list(g.columns) == fpl.PLAYER_GW_COLUMNS
    assert g.position.tolist() == ["FWD", "GK"]
    striker = g.iloc[0]
    assert (striker.team, striker.opponent, striker.was_home) == ("Man United", "Fulham", True)
    assert (striker.goals, striker.price, striker.xg) == (1, 7.0, 0.6)


def test_bootstrap_parsing():
    boot = json.loads((FIXTURES / "fpl_bootstrap.json").read_text())
    players, events = fpl.parse_bootstrap(boot)
    assert players.name.tolist() == ["Raya", "Saka"]  # managers (type 5) are skipped
    saka = players.iloc[1]
    assert (saka.position, saka.price, saka.chance_next, saka.xg) == ("MID", 10.1, 75.0, 3.2)
    assert players.iloc[0].chance_next == 100.0  # null means no injury flag
    assert events.loc[events.is_next, "gw"].item() == 2


def test_live_fixtures_skip_unscheduled():
    boot = json.loads((FIXTURES / "fpl_bootstrap.json").read_text())
    fixtures = [
        {"id": 11, "event": 2, "kickoff_time": "2024-08-24T14:00:00Z", "team_h": 1,
         "team_a": 14, "finished": False, "team_h_score": None, "team_a_score": None},
        {"id": 12, "event": None, "kickoff_time": None, "team_h": 14, "team_a": 1,
         "finished": False},
    ]
    f = fpl.parse_live_fixtures(fixtures, boot)
    assert f[["fixture_id", "gw", "home", "away"]].values.tolist() == [
        [11, 2, "Arsenal", "Man United"]]


def test_rows_carry_stable_player_codes():
    g = fpl.parse_player_gw(_csv("fpl_merged_gw.csv"), _csv("fpl_teams.csv"), "2024-25",
                            codes={10: 5001, 20: 5002})
    assert g.code.tolist() == [5001, 5002]


def test_align_elements_rekeys_by_code_not_by_reused_ids():
    # Element 384 is a Liverpool outfielder last season and a Man City keeper this season.
    rows = pd.DataFrame({
        "season": ["2025-26", "2025-26", "2026-27"],
        "element": [384, 120, 384],
        "code": [111, 222, 222],          # the keeper is code 222 in both seasons
        "name": ["Outfielder", "Keeper", "Keeper"],
    })
    this_season = {384: 222, 501: 333}    # code 111 has left the league
    out = fpl.align_elements(rows, this_season)
    assert out[["season", "element", "name"]].values.tolist() == [
        ["2025-26", 384, "Keeper"], ["2026-27", 384, "Keeper"]]
