"""Data layer: canonical names, CSV parsing across eras, validation and caching."""

import pandas as pd
import pytest

from epl.data import download, load, teams


def test_aliases_all_map_to_canonical_names():
    assert set(teams.ALIASES.values()) <= teams.CANONICAL


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Man Utd", "Man United"), ("Spurs", "Tottenham"), ("Nott'm Forest", "Nott'm Forest"),
     ("Wolverhampton Wanderers", "Wolves"), ("  Man  City ", "Man City")],
)
def test_canonical_names(raw, expected):
    assert teams.canonical(raw) == expected


def test_unknown_team_fails_loudly():
    with pytest.raises(teams.UnknownTeamError):
        teams.canonical("Real Madrid")


def test_season_code():
    assert download.season_code("2024-25") == "2425"
    assert download.season_code("1999-00") == "9900"
    with pytest.raises(ValueError):
        download.season_code("2024-26")


def test_seasons_range():
    assert load.seasons("2019-20", "2021-22") == ["2019-20", "2020-21", "2021-22"]


def test_parse_modern_season_prefers_closing_odds(sample_2425):
    m = load.parse_football_data(sample_2425, "2024-25")
    assert len(m) == 6  # the blank trailing row is dropped
    first = m.iloc[0]
    assert (first.home, first.away, first.hg, first.ag, first.result) == (
        "Man United", "Fulham", 1, 0, "H")
    assert first.date == pd.Timestamp("2024-08-16")
    assert first.odds_source == "avg_closing" and first.odds_h == 1.60
    # Everton v Brighton has no average closing odds, so Pinnacle closing fills in.
    row = m.set_index("match_id").loc["20240817-Everton-Brighton"]
    assert row.odds_source == "pinnacle_closing" and row.odds_a == 2.85
    assert list(m.columns[:5]) == ["match_id", "season", "date", "home", "away"]


def test_parse_old_season_two_digit_years_and_fallback_odds(sample_0506):
    m = load.parse_football_data(sample_0506, "2005-06")
    assert m.date.min() == pd.Timestamp("2005-08-13")
    assert set(m.odds_source) == {"betbrain_avg"}
    assert m.result.tolist() == ["D", "A", "A"]


def test_validate_rejects_duplicates_and_bad_odds(sample_2425):
    m = load.parse_football_data(sample_2425, "2024-25")
    load.validate(m)
    with pytest.raises(ValueError, match="duplicate"):
        load.validate(pd.concat([m, m.iloc[[0]]]))
    broken = m.copy()
    broken.loc[0, ["odds_h", "odds_d", "odds_a"]] = [5.0, 5.0, 5.0]  # overround 0.6
    with pytest.raises(ValueError, match="overround"):
        load.validate(broken)


def test_cache_downloads_once(tmp_path, sample_2425):
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return sample_2425

    for _ in range(2):
        raw = download.football_data_csv("2024-25", tmp_path, fetch=fake_fetch)
    assert raw == sample_2425
    assert calls == ["https://www.football-data.co.uk/mmz4281/2425/E0.csv"]
    assert not list(tmp_path.rglob("*.part"))


def test_load_matches_and_duckdb_query(tmp_path, sample_2425, sample_0506):
    raw = {"0506": sample_0506, "2425": sample_2425}
    m = load.load_matches(["2005-06", "2024-25"], tmp_path,
                          fetch=lambda url: raw[url.split("/")[-2]])
    assert len(m) == 9
    out = load.query("SELECT season, count(*) AS n FROM m GROUP BY 1 ORDER BY 1", m=m)
    assert out.to_dict("records") == [{"season": "2005-06", "n": 3}, {"season": "2024-25", "n": 6}]
    assert load.complete_seasons(m) == []  # samples are partial seasons


def test_validate_accepts_rows_without_odds(sample_2425):
    m = load.parse_football_data(sample_2425, "2024-25")
    m[["odds_h", "odds_d", "odds_a"]] = float("nan")
    load.validate(m)
